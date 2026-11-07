//! Native strtol/strtoul/atoi/atol implementations.
//!
//! Concrete fast path: byte-by-byte parsing with whitespace skip, sign,
//! optional base prefix, and digit accumulation.
//!
//! Symbolic-byte support: when the digit region contains symbolic bytes
//! (e.g. an attacker-controlled buffer constrained to ASCII digits), we
//! build an ITE accumulator chain over the up-to-MAX_DIGITS positions:
//!     accum_{i+1} = ITE(stay_i, accum_i, accum_i*base + digit_value(b_i))
//!     stay_i = terminated_i OR !is_digit(b_i)
//! The final result is `accum_n` (and is negated if a concrete '-' prefix
//! was consumed). The sign and base-prefix bytes must be decided: a symbolic
//! byte the path constraints let be a sign (or, for bases 0 and 16, the `0` /
//! `x` of a base prefix) falls back to Python, whose `strtol_inner` forks
//! those cases (angr-mx00p.132.75). A symbolic byte that cannot be one starts
//! the digit region.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** the string
//! bytes are guest data, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and anything it cannot
//! parse concretely takes the symbolic path or falls back to Python. There are
//! no `unwrap`/`expect` sites left: the all-concrete digit probe now collects
//! into `Option<Vec<u8>>` in one short-circuiting pass instead of an
//! `all(..)` check followed by a re-`unwrap`ping second pass.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::ctype::{is_c_space, may_continue_numeric_subject};
use super::{ProcedureError, arch_word};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Initial raw-scan window, and the hard cap on the *symbolic* accumulator's
/// digit positions (the concrete path grows past it — see `MAX_SCAN_BYTES`).
///
/// Native parses up to 64 digits symbolically. Python caps at `state.libc.max_strtol_len`
/// (= 11, angr/state_plugins/libc.py). This is a **deliberate divergence**: a
/// 12+-digit concrete string makes Python's `final_constraint` Or(...) evaluate
/// False, so the state goes unsat and is silently discarded, while native
/// parses it C-style. Native's behavior is closer to C; Python's unsat-at-12
/// is arguably its own bug (angr-vu1q4). We do NOT replicate
/// Python's cap because truncating to 11 digits would match neither engine
/// (Python rejects the input outright rather than truncating), and forcing the
/// state unsat is an invasive behavior change with real benchmark risk.
const MAX_DIGITS: usize = 64;

/// Hard ceiling on the raw scan window `read_subject_bytes` will grow to.
///
/// `MAX_DIGITS` is only the *initial* window: a concrete digit run longer than
/// it used to be truncated mid-run, so `parse_concrete_digits` returned a
/// magnitude short of the real one (visible for low bases, where the `u128`
/// accumulator does not saturate early) and `endptr` landed mid-digit-run
/// instead of past the subject sequence (bd angr-fs8kb.37). The window now
/// doubles while the byte it ended on could still continue the subject
/// sequence, so any subject sequence up to this many bytes parses exactly.
///
/// Past this ceiling the scan truncates the literal, deliberately: the
/// scan is O(window) guest memory loads with no null in sight, and a 4 KiB
/// numeric literal is not a shape real code produces. Pinned by
/// `test_strtol_scan_window_ceiling_truncates`.
const MAX_SCAN_BYTES: usize = 4096;

/// Python's `_string_to_int` clamps the accumulated magnitude at the SIGNED max
/// `2^(bits-1) - 1` — `strtol.run` always calls `strtol_inner(..., signed=True)`,
/// even for `strtoul`, so the whole family shares this cap. Only defined for
/// `bits` in `[2, 64]`; `magnitude_cap` gates on `bits < 64`.
fn signed_max(bits: u32) -> u128 {
    (1u128 << (bits - 1)) - 1
}

/// The magnitude every native path clamps at before negating: `signed_max`
/// below 64 bits (Python's clamp), the full unsigned register max at 64 bits
/// (C's `ULONG_MAX`, so strtoull keeps values above `i64::MAX`; see
/// `MAX_DIGITS` for why 64-bit does not follow Python). Clamping instead of
/// wrapping keeps an unsigned digit run from turning negative.
fn magnitude_cap(bits: u32) -> u128 {
    if bits < 64 {
        signed_max(bits)
    } else {
        u128::MAX >> (128 - bits)
    }
}

/// Mask a concrete register value to `w` low bits when `ret_int_bits` requests
/// an int-width return (atoi: Python returns `val[sizeof(int)*8 - 1 : 0]`).
fn apply_int_width(val: u128, ret_int_bits: Option<u32>) -> u128 {
    match ret_int_bits {
        Some(w) if w < 128 => val & ((1u128 << w) - 1),
        _ => val,
    }
}

/// Read up to `max_len` bytes from `addr`, stopping at the first concrete
/// null terminator (symbolic bytes do not stop the scan).
fn read_bytes_until_null(
    state: &RustSimState,
    addr: u64,
    max_len: usize,
) -> Result<Vec<RustBV>, ProcedureError> {
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let byte = state.memory_load(addr.wrapping_add(i as u64), 1)?;
        if let Some(b) = byte.as_u64()
            && (b as u8) == 0
        {
            break;
        }
        result.push(byte);
    }
    Ok(result)
}

/// Could the subject sequence continue *through* `b`?
///
/// Base-agnostic on purpose — `read_subject_bytes` runs before
/// `parse_concrete_prefix` has picked a base — so the byte test is
/// `ctype::may_continue_numeric_subject`, shared with `strtod`'s window.
///
/// A symbolic byte answers `false`: the symbolic path caps its accumulator at
/// `MAX_DIGITS` digit positions (see `run_strtol`), so growing the window past
/// a symbolic byte buys nothing.
fn may_continue_subject(b: &RustBV) -> bool {
    match b.as_u64() {
        Some(v) => may_continue_numeric_subject(v as u8),
        None => false,
    }
}

/// Read the raw subject-sequence window at `addr`, doubling from `MAX_DIGITS`
/// up to `MAX_SCAN_BYTES` while the window keeps ending on a byte the subject
/// sequence could continue through.
///
/// Stopping early is safe in both directions: a window that ended on a concrete
/// null (short read) or on a byte no prefix/digit parser can consume already
/// contains the whole subject sequence, and reading *more* than the sequence
/// never changes a parse result.
fn read_subject_bytes(state: &RustSimState, addr: u64) -> Result<Vec<RustBV>, ProcedureError> {
    let mut window = MAX_DIGITS;
    loop {
        let bytes = read_bytes_until_null(state, addr, window)?;
        if bytes.len() < window
            || window >= MAX_SCAN_BYTES
            || !bytes.last().is_some_and(may_continue_subject)
        {
            return Ok(bytes);
        }
        window = (window * 2).min(MAX_SCAN_BYTES);
    }
}

/// Is the byte after a `0x`/`0X` base-16 prefix one the digit parser can
/// actually consume?
///
/// C's longest-valid-subject-sequence rule: when `0x` is *not* followed by a
/// hex digit, the subject sequence is just the leading `0`, which is itself a
/// complete conversion. Consuming the prefix anyway discards that `0`, so the
/// caller sees "no conversion" and `*endptr` lands past the `x`
/// (`strtol("0xg", &e, 16)` gave offset 2 where glibc and
/// `angr/procedures/libc/strtol.py::strtol_inner`'s empty-prefix branch both
/// give 1). `idx` points at the `0`, so the byte under test is `idx + 2`.
///
/// A symbolic byte counts as usable: the symbolic accumulator handles it, and
/// per this module's design note callers typically constrain such bytes to
/// digits. Same backtrack corollary as `strtod.rs::floating_prefix_len`
/// (bd angr-6cp06.2).
fn hex_prefix_has_digit(bytes: &[RustBV], idx: usize) -> bool {
    // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
    match bytes.get(idx + 2) {
        // Past end: "0x" alone converts as the single digit `0`.
        None => false,
        Some(b) => match b.as_u64() {
            Some(v) => (v as u8).is_ascii_hexdigit(),
            // Symbolic — assume a digit and let the accumulator decide.
            None => true,
        },
    }
}

/// Parse the whitespace / sign / base-prefix prefix of `bytes`. Returns the
/// index immediately after the prefix and, if a digit region can follow, the
/// chosen base + sign.
///
/// Every *concrete* rejection this step can reach — input exhausted, base
/// outside 2..=36 — is expressed as a `None` base/sign, not an error, so the
/// caller always has a `prefix_end` to write through `endptr` (angr-6cp06.4).
/// The one `Err` is the Python fallback: a symbolic byte that `ctx` lets be a
/// sign, or the `0` / `x` of a base prefix the base may auto-detect. The digit
/// accumulator models neither, so treating such a byte as a digit drops the
/// signed and prefixed paths and keeps ones C cannot reach (angr-mx00p.132.75).
///
/// A symbolic byte that cannot be one of those breaks out of the
/// prefix-handling steps and starts the digit region. Symbolic whitespace is
/// not skipped: Python's `strtol` skips none, and the accumulator reads such a
/// byte as a non-digit.
fn parse_concrete_prefix(
    ctx: &SymContext,
    bytes: &[RustBV],
    base_arg: i64,
) -> Result<(usize, Option<(u32, bool)>), ProcedureError> {
    let undecided = |what: &str| {
        Err(ProcedureError::SymbolicArgument(format!(
            "strtol: symbolic byte may be a {what}"
        )))
    };
    let mut idx = 0;

    // Whitespace: only consume *concretely* whitespace bytes. A symbolic
    // byte is treated as start-of-digit-region. The classifier is C-locale
    // `isspace` (`\v` included) via `ctype::is_c_space`, matching libc —
    // angr's Python `strtol` SimProcedure skips no whitespace at all, so
    // there is no Python behaviour to mirror here.
    while idx < bytes.len() {
        match bytes[idx].as_u64() {
            // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
            Some(b) if is_c_space(b as u8) => idx += 1,
            _ => break,
        }
    }
    if idx >= bytes.len() {
        return Ok((idx, None));
    }

    // Sign: only consume a *concrete* '+' or '-'.
    if bytes[idx].as_u64().is_none() && may_be_any(ctx, &bytes[idx], b"+-") {
        return undecided("sign");
    }
    let mut negative = false;
    if let Some(b) = bytes[idx].as_u64() {
        if (b as u8) == b'-' {
            negative = true;
            // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
            idx += 1;
        } else if (b as u8) == b'+' {
            // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
            idx += 1;
        }
    }
    if idx >= bytes.len() {
        return Ok((idx, None));
    }

    // A base prefix is the `0` (base 0: octal) or `0x` (bases 0 and 16) the
    // detection below consumes; a symbolic byte that may complete one forks
    // octal / hex / decimal.
    if base_arg == 0 || base_arg == 16 {
        let lead = &bytes[idx];
        // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
        let next = bytes.get(idx + 1);
        let lead_symbolic = lead.as_u64().is_none();
        let next_symbolic = next.is_some_and(|n| n.as_u64().is_none());
        let lead_may_be_zero = may_be_any(ctx, lead, b"0");
        let next_may_be_x = next.is_some_and(|n| may_be_any(ctx, n, b"xX"));
        let hex_undecided = (lead_symbolic || next_symbolic) && lead_may_be_zero && next_may_be_x;
        let octal_undecided = base_arg == 0 && lead_symbolic && lead_may_be_zero;
        if hex_undecided || octal_undecided {
            return undecided("base prefix");
        }
    }

    // Base detection.
    let base: u32 = if base_arg == 0 {
        // Auto-detect. A symbolic byte at the leading position cannot
        // disambiguate 0/0x/decimal — default to base 10.
        match bytes[idx].as_u64() {
            Some(b) if (b as u8) == b'0' => {
                match bytes
                    // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
                    .get(idx + 1)
                    .and_then(super::super::symbolic::RustBV::as_u64)
                {
                    Some(n)
                        if ((n as u8) == b'x' || (n as u8) == b'X')
                            && hex_prefix_has_digit(bytes, idx) =>
                    {
                        // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
                        idx += 2;
                        16
                    }
                    Some(_) => {
                        // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
                        idx += 1;
                        8
                    }
                    None => {
                        // Either past-end or symbolic. Past-end: leading '0'
                        // is the entire input → treat as base 10 (parses 0).
                        // Symbolic next: don't speculatively consume the '0';
                        // let it parse as digit 0 in base 10.
                        10
                    }
                }
            }
            Some(_) => 10,
            None => 10,
        }
    } else if base_arg == 16 {
        if let (Some(b0), Some(b1)) = (
            bytes[idx].as_u64(),
            bytes
                // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
                .get(idx + 1)
                .and_then(super::super::symbolic::RustBV::as_u64),
        ) && (b0 as u8) == b'0'
            && ((b1 as u8) == b'x' || (b1 as u8) == b'X')
            && hex_prefix_has_digit(bytes, idx)
        {
            // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
            idx += 2;
        }
        16
    } else {
        base_arg as u32
    };

    if !(2..=36).contains(&base) {
        return Ok((idx, None));
    }
    Ok((idx, Some((base, negative))))
}

/// Can `byte` equal any of `set` under the path constraints in `ctx`? A
/// concrete byte answers without the solver; an undecided solver answers yes, so the caller falls back to Python.
fn may_be_any(ctx: &SymContext, byte: &RustBV, set: &[u8]) -> bool {
    if let Some(v) = byte.as_u64() {
        return set.contains(&(v as u8));
    }
    let byte_8 = if byte.width() == 8 {
        byte.clone()
    } else {
        byte.extract(7, 0)
    };
    let mut cond = RustBV::concrete(0, 1);
    for &c in set {
        cond = cond.or(&byte_8.eq(&RustBV::concrete(u128::from(c), 8)));
    }
    ctx.can_be_true(&cond)
}

/// Concrete digit-only parser. Returns (magnitude, num_consumed).
///
/// The magnitude accumulates *saturating* in `u128` (never panics on a
/// pathological 64-digit input) and is UNsigned — the sign and the signed
/// overflow clamp are applied by the caller so the clamp can compare against
/// `signed_max(bits)` before negation, matching Python's clamp-then-negate
/// order.
fn parse_concrete_digits(bytes: &[u8], base: u32) -> (u128, usize) {
    let mut value: u128 = 0;
    let mut found_digit = false;
    let mut idx = 0;
    while idx < bytes.len() {
        let digit = match bytes[idx] {
            b'0'..=b'9' => (bytes[idx] - b'0') as u32,
            b'a'..=b'z' => (bytes[idx] - b'a' + 10) as u32,
            b'A'..=b'Z' => (bytes[idx] - b'A' + 10) as u32,
            _ => break,
        };
        if digit >= base {
            break;
        }
        found_digit = true;
        value = value
            .saturating_mul(base as u128)
            .saturating_add(digit as u128);
        // overflow-ok: `idx` indexes `bytes`, so it is below `bytes.len()` <= isize::MAX.
        idx += 1;
    }
    if !found_digit { (0, 0) } else { (value, idx) }
}

/// Build the ITE accumulator over `bytes[start..]` for the given concrete
/// `base` (in [2, 36]) and the chosen result width.
///
/// The accumulator runs at twice `result_bits` and clamps at `cap` after each
/// step, so no digit run wraps: `cap * 36 + 35` fits the double width, and
/// accumulation is monotone, so the per-step clamp equals a final clamp
/// (angr-mx00p.132.78).
///
/// At each position the accumulator either advances (if the byte is a
/// valid digit and we have not yet been terminated by a non-digit) or
/// stays put. The "terminated" flag is sticky: once a non-digit appears,
/// later digits do not contribute.
///
/// Returns `(accum, run_len)`: `run_len` (also `result_bits` wide) counts the
/// positions that advanced, built from the same `stay` predicates, so it is
/// the symbolic length of the converted digit run (angr-vmi2l.101).
fn build_symbolic_accumulator(
    bytes: &[RustBV],
    start: usize,
    base: u32,
    result_bits: u32,
    cap: u128,
) -> (RustBV, RustBV) {
    // overflow-ok: result_bits is an arch width, at most 64.
    let wide = result_bits * 2;
    let zero_acc = RustBV::concrete(0, wide);
    let base_bv = RustBV::concrete(base as u128, wide);
    let cap_bv = RustBV::concrete(cap, wide);
    let zero_byte = RustBV::concrete(b'0' as u128, 8);
    let max_dec = (base.min(10) - 1) as u8;
    let dec_high = RustBV::concrete((b'0' + max_dec) as u128, 8);

    let one = RustBV::concrete(1, result_bits);
    let mut run_len = RustBV::concrete(0, result_bits);
    let mut accum = zero_acc;
    let mut terminated = RustBV::concrete(0, 1);

    for byte_val in &bytes[start..] {
        let byte_8 = if byte_val.width() == 8 {
            byte_val.clone()
        } else {
            byte_val.extract(7, 0)
        };

        // Decimal-digit predicate: byte in ['0', '0' + min(base,10) - 1].
        let is_dec = byte_8.uge(&zero_byte).and(&byte_8.ule(&dec_high));
        let dec_value = byte_8.sub(&zero_byte).zero_extend(wide);

        let (is_digit, digit_value) = if base > 10 {
            // overflow-ok: base > 10 is guaranteed by the enclosing `if`.
            let max_alpha = (base - 10 - 1) as u8;
            let lower_lo = RustBV::concrete(b'a' as u128, 8);
            let lower_hi = RustBV::concrete((b'a' + max_alpha) as u128, 8);
            let upper_lo = RustBV::concrete(b'A' as u128, 8);
            let upper_hi = RustBV::concrete((b'A' + max_alpha) as u128, 8);
            let in_lower = byte_8.uge(&lower_lo).and(&byte_8.ule(&lower_hi));
            let in_upper = byte_8.uge(&upper_lo).and(&byte_8.ule(&upper_hi));
            let is_alpha = in_lower.or(&in_upper);
            let ten_bv = RustBV::concrete(10, wide);
            let lower_v = byte_8.sub(&lower_lo).zero_extend(wide).add(&ten_bv);
            let upper_v = byte_8.sub(&upper_lo).zero_extend(wide).add(&ten_bv);
            let alpha_v = in_lower.ite(&lower_v, &upper_v);
            let any = is_dec.or(&is_alpha);
            let val = is_dec.ite(&dec_value, &alpha_v);
            (any, val)
        } else {
            (is_dec, dec_value)
        };

        let accum_advanced = accum.mul(&base_bv).add(&digit_value);
        let accum_advanced = accum_advanced.ule(&cap_bv).ite(&accum_advanced, &cap_bv);
        let stay = terminated.or(&is_digit.not());
        accum = stay.ite(&accum, &accum_advanced);
        run_len = stay.ite(&run_len, &run_len.add(&one));
        terminated = stay;
    }
    // overflow-ok: result_bits >= 1.
    (accum.extract(result_bits - 1, 0), run_len)
}

/// Common engine for atoi/atol/strtol/strtoul.
///
/// `endptr` is `Some(0)` for an explicit NULL pointer and `None` for callers
/// that take no `endptr` argument (atoi/atol).
/// `ret_int_bits` requests an int-width (rather than arch-width) return, as
/// `atoi` does: Python's `atoi.run` returns `val[sizeof(int)*8 - 1 : 0]`.
/// `None` returns the full arch width (atol/strtol/strtoul).
fn run_strtol(
    state: &mut RustSimState,
    addr: u64,
    endptr: Option<u64>,
    base_arg: i64,
    ret_int_bits: Option<u32>,
) -> Result<Option<RustBV>, ProcedureError> {
    let bytes = read_subject_bytes(state, addr)?;
    let bits = state.arch().bits();

    let (prefix_end, prefix) = {
        let ctx = state.solver().borrow();
        parse_concrete_prefix(&ctx, &bytes, base_arg)?
    };

    // No digits possible (empty after prefix or invalid base).
    let (base, negative) = match prefix {
        Some(p) => p,
        None => {
            // No conversion: C and Python store `nptr` itself, not past the
            // skipped whitespace/sign (angr-vmi2l.101).
            if let Some(end) = endptr
                && end != 0
            {
                state.memory_store(end, arch_word(state, addr))?;
            }
            return Ok(Some(arch_word(state, 0u64)));
        }
    };

    // Base-0 auto-detect reaches octal only by consuming a leading `0`, which
    // is itself a converted digit: "0xg" and "09" convert to 0 with the end
    // past that `0`, not at `nptr`.
    let prefix_converted = base_arg == 0 && base == 8;

    // All-concrete digit region: take the original fast path. Collecting into
    // `Option<Vec<_>>` short-circuits on the first symbolic byte, so this is the
    // same single pass the old `all(..)` probe did — without a second pass that
    // re-`unwrap`s what the probe just proved.
    let concrete_digits: Option<Vec<u8>> = bytes[prefix_end..]
        .iter()
        .map(|b| b.as_u64().map(|v| v as u8))
        .collect();
    if let Some(cb) = concrete_digits {
        let (mag, consumed) = parse_concrete_digits(&cb, base);
        // Python clamps the magnitude at the signed max before negating. On
        // 32-bit archs this clamp is reachable for <=max_strtol_len-digit
        // inputs where native's old wrapping accumulator diverged (e.g. x86
        // atoi("3000000000") -> Python 0x7FFFFFFF vs native 0xB2D05E00). On
        // 64-bit archs Python's 11-digit cap rejects any i63-overflowing
        // concrete input as unsat *before* the clamp fires, so there is no
        // Python value to match — native keeps its C-style full-width parse
        // (this is what lets strtoull return values above i64::MAX), clamped
        // at the register max rather than wrapped. See the MAX_DIGITS note
        // (angr-vu1q4).
        let clamped = mag.min(magnitude_cap(bits));
        let reg_val = if negative {
            0u128.wrapping_sub(clamped)
        } else {
            clamped
        };
        let reg_val = apply_int_width(reg_val, ret_int_bits);
        if let Some(end) = endptr
            && end != 0
        {
            let end_addr = if consumed == 0 && !prefix_converted {
                // No conversion: C and Python store `nptr` (angr-vmi2l.101).
                addr
            } else {
                // overflow-ok: prefix_end/consumed are usize offsets into the
                // parsed input string, nowhere near usize::MAX.
                addr.wrapping_add((prefix_end + consumed) as u64)
            };
            state.memory_store(end, arch_word(state, end_addr))?;
        }
        return Ok(Some(RustBV::concrete(reg_val, bits)));
    }

    // Symbolic digit region: build accumulator over at most `MAX_DIGITS` digit
    // positions. `read_subject_bytes` may have grown the window past that for
    // the concrete path's sake; an ITE chain one term per byte is the one
    // consumer that cannot absorb a longer window, so it keeps the old cap.
    let sym_end = bytes.len().min(prefix_end.saturating_add(MAX_DIGITS));
    let ctx_handle = state.solver().clone();
    let ctx = ctx_handle.borrow();
    let (accum, run_len) = build_symbolic_accumulator(
        &bytes[..sym_end],
        prefix_end,
        base,
        bits,
        magnitude_cap(bits),
    );
    let result = if negative { accum.neg() } else { accum };
    // atoi returns int-width bits (Python: `val[sizeof(int)*8 - 1 : 0]`).
    // Extract the low `w` bits then zero-extend back into the return register.
    let result = match ret_int_bits {
        Some(w) if w < bits => result.extract(w - 1, 0).zero_extend(bits),
        _ => result,
    };
    drop(ctx);

    // Endptr follows the path's digit run: `nptr` when no digit converts (C;
    // Python's `num_bytes` is 0 there), else past the prefix and the run. A
    // single concrete end made `end == nptr` / `*end != '\0'` checks one-sided
    // (angr-vmi2l.101).
    if let Some(end) = endptr
        && end != 0
    {
        let nptr = arch_word(state, addr);
        let past_run = arch_word(state, addr.wrapping_add(prefix_end as u64)).add(&run_len);
        let end_addr = if prefix_converted {
            past_run
        } else {
            run_len.eq(&arch_word(state, 0)).ite(&nptr, &past_run)
        };
        state.memory_store(end, end_addr)?;
    }

    Ok(Some(result))
}

crate::declare_proc! {
    /// Native strtol: `long strtol(const char *nptr, char **endptr, int base)`.
    name = "strtol",
    struct = NativeStrtol,
    args = [nptr: concrete, endptr: concrete, base: int],
    call |state| {
        run_strtol(state, nptr, Some(endptr), i64::from(base), None)
    }
}

crate::declare_proc! {
    /// Native strtoul: same impl as strtol; unsigned semantics fall out of bit
    /// interpretation.
    name = "strtoul",
    struct = NativeStrtoul,
    args = [nptr: concrete, endptr: concrete, base: int],
    call |state| {
        run_strtol(state, nptr, Some(endptr), i64::from(base), None)
    }
}

crate::declare_proc! {
    /// Native atoi: `int atoi(const char *nptr)` (base 10, no endptr).
    name = "atoi",
    struct = NativeAtoi,
    args = [nptr: concrete],
    call |state| {
        // atoi returns int-width (Python: `val[sizeof(int)*8 - 1 : 0]`).
        run_strtol(state, nptr, None, 10, Some(32))
    }
}

crate::declare_proc! {
    /// Native atol: `long atol(const char *nptr)` (base 10, no endptr).
    name = "atol",
    struct = NativeAtol,
    args = [nptr: concrete],
    call |state| {
        run_strtol(state, nptr, None, 10, None)
    }
}

crate::declare_proc! {
    /// Native strtoll: `long long strtoll(const char *nptr, char **endptr, int base)`.
    /// On the LP64 targets we support (amd64 / aarch64), `long long` is 64 bits
    /// — the same width as `long` — so the shared `run_strtol` engine produces
    /// the correct result. On ILP32 (x86 / arm32), the dispatch return-register
    /// width is 32 bits while `long long` is 64 bits; we leave that combination
    /// to the Python fallback rather than try to plumb a wide return through
    /// the integer return register (caller usually uses split eax:edx).
    ///
    /// This is a deliberate, tested fallback — not a TODO. The single-register
    /// return path in the dispatcher (`run_loop_single.rs` / `stepping.rs`, which call
    /// `set_register_by_offset(return_register(), rv)`) has no way to populate
    /// the high half of a split-register result, and silently returning only
    /// the low 32 bits would corrupt any value above 2^32. The Python
    /// SimProcedure knows the split calling convention, so deferring is correct.
    /// Pinned by `test_strtoll_ilp32_falls_back_to_python` in strtol_tests.rs.
    name = "strtoll",
    struct = NativeStrtoll,
    args = [nptr: concrete, endptr: concrete, base: int],
    call |state| {
        if state.arch().bits() < 64 {
            return Err(ProcedureError::NotImplemented);
        }
        run_strtol(state, nptr, Some(endptr), i64::from(base), None)
    }
}

crate::declare_proc! {
    /// Native strtoull: unsigned sibling of strtoll. Same ILP32 split-return
    /// caveat — deliberate Python fallback on 32-bit, pinned by
    /// `test_strtoull_ilp32_falls_back_to_python`. See strtoll above.
    name = "strtoull",
    struct = NativeStrtoull,
    args = [nptr: concrete, endptr: concrete, base: int],
    call |state| {
        if state.arch().bits() < 64 {
            return Err(ProcedureError::NotImplemented);
        }
        run_strtol(state, nptr, Some(endptr), i64::from(base), None)
    }
}

test_submod!("strtol_tests.rs" => strtol_tests);
