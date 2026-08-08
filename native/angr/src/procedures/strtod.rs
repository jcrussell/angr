//! Native `strtod` implementation.
//!
//! ```c
//! double strtod(const char *nptr, char **endptr);
//! ```
//!
//! Concrete fast path: scan up to a fixed number of bytes until the first
//! null terminator, leading whitespace + sign + optional digits-dot-digits
//! + optional exponent, and feed the slice through Rust's `f64::from_str`.
//!
//! C99 hex-float literals (`0x1.8p3`) are the exception: `f64::from_str`
//! rejects that syntax outright, so `parse_hex_float` handles them.
//!
//! Symbolic bytes anywhere in the parsed region — or a target whose calling
//! convention has no modelled FP-return slot — fall back to Python. The slot
//! comes from `CallingConvention::fp_return_register()`: amd64 (`xmm0`) and
//! AArch64 (`v0`) provide one; x86 (x87 `st0`), ARM EABI soft-float (`r0:r1`)
//! and MIPS (`$f0` for hard-float builds, `$v0`/`$v1` for `-msoft-float` ones,
//! indistinguishable from the register file) do not, so those still defer.
//!
//! Return value is the 64-bit IEEE-754 bit-pattern of the parsed double,
//! written to the low 64 bits of that register. The dispatcher's default
//! integer-return store is suppressed by returning `Ok(None)`.

use super::arch_word;
use super::ctype::is_c_space;
use super::strings::scan_concrete_bounded;
use super::{ProcedureError, extract_concrete_arg};
use crate::arch::cc_for_arch;
use crate::symbolic::RustBV;

/// Maximum byte scan length when reading the numeric literal. Real strings
/// rarely need more than ~64 bytes; cap the work even for hostile inputs.
const MAX_LEN: usize = 256;

/// Walk `bytes` from the front and return the byte index immediately past
/// the longest prefix that looks like a C99 floating-point literal. Returns
/// 0 if no parseable prefix was found.
///
/// Grammar (case-insensitive `e`/`x`/`p`):
/// ```text
///   [ws]* [+-]? ( digits? '.' digits ([eE][+-]?digits)?
///               | digits ('.' digits?)? ([eE][+-]?digits)?
///               | '0x' hex_digits? '.' hex_digits? ([pP][+-]?digits)?
///               | "inf" | "infinity"
///               | "nan" )
/// ```
fn floating_prefix_len(bytes: &[u8]) -> usize {
    let mut i = 0;
    // whitespace (C-locale `isspace`, `\v` included — see `ctype::is_c_space`)
    while i < bytes.len() && is_c_space(bytes[i]) {
        i += 1;
    }
    let prefix_start = i;
    // sign
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    // inf / infinity
    if i + 3 <= bytes.len() {
        let tag = &bytes[i..i + 3];
        if tag.eq_ignore_ascii_case(b"inf") {
            let mut j = i + 3;
            if j + 5 <= bytes.len() && bytes[j..j + 5].eq_ignore_ascii_case(b"inity") {
                j += 5;
            }
            return j;
        }
    }
    // nan
    if i + 3 <= bytes.len() && bytes[i..i + 3].eq_ignore_ascii_case(b"nan") {
        return i + 3;
    }
    // hex form: 0x...
    if i + 2 <= bytes.len() && bytes[i] == b'0' && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
        let start = i;
        i += 2;
        let mut saw_digit = false;
        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
            saw_digit = true;
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                saw_digit = true;
                i += 1;
            }
        }
        if !saw_digit {
            return prefix_start;
        }
        if i < bytes.len() && (bytes[i] == b'p' || bytes[i] == b'P') {
            let exp_start = i;
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            let exp_digits_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == exp_digits_start {
                return exp_start; // 'p' without digits — drop it
            }
        }
        return i.max(start);
    }
    // decimal form
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        saw_digit = true;
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            saw_digit = true;
            i += 1;
        }
    }
    if !saw_digit {
        return prefix_start;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let exp_start = i;
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_digits_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_digits_start {
            return exp_start;
        }
    }
    i
}

/// `x * 2^n`, clamped in musl's three steps so an out-of-range `n` cannot
/// over/underflow an intermediate. A plain `x * 2f64.powi(n)` collapses to
/// `inf`/`0.0` in the exponent tails and loses results that *are*
/// representable (e.g. a 2^124-scale significand with a 2^-1100 exponent),
/// and the stepped-down subnormal path avoids double-rounding.
fn scale_by_pow2(x: f64, n: i32) -> f64 {
    let p1023 = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
    let p53 = f64::from_bits(0x4340_0000_0000_0000); // 2^53
    let pm1022 = f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
    let mut y = x;
    let mut n = n;
    if n > 1023 {
        y *= p1023;
        n -= 1023;
        if n > 1023 {
            y *= p1023;
            n = (n - 1023).min(1023);
        }
    } else if n < -1022 {
        // Step by 1022-53 so the final scaling still has 53 bits of headroom
        // above the subnormal boundary.
        y *= pm1022 * p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= pm1022 * p53;
            n = (n + (1022 - 53)).max(-1022);
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// Parse a complete C99 hex-float literal — `[+-]?0[xX]hex[.hex][pP[+-]?dec]`
/// — as already delimited by `floating_prefix_len`'s hex branch.
///
/// Rust's `f64::from_str` does not accept hex-float syntax at all, so without
/// this the recognized literal parsed as 0.0 while `*endptr` was still
/// advanced past it: a silently wrong `strtod("0x1.8p3")` (angr-sqfj8.81).
///
/// Returns `None` when `s` is not a complete hex-float literal, so the caller
/// falls through to `f64::from_str` for the decimal / `inf` / `nan` forms.
fn parse_hex_float(s: &[u8]) -> Option<f64> {
    let mut i = 0;
    let negative = match s.first() {
        Some(b'+') => {
            i = 1;
            false
        }
        Some(b'-') => {
            i = 1;
            true
        }
        _ => false,
    };
    if s.len() < i + 2 || s[i] != b'0' || !(s[i + 1] == b'x' || s[i + 1] == b'X') {
        return None;
    }
    i += 2;

    // Significand: `mant` accumulates the leading hex digits as an integer and
    // `exp2` the binary exponent that scales it back. Digits beyond 124 bits
    // cannot change a correctly-rounded f64 (53 bits of significand), so they
    // only raise a sticky flag, folded into `mant`'s LSB below — the standard
    // guard against a spurious round-to-even on an exact-looking tie.
    let mut mant: u128 = 0;
    let mut exp2: i64 = 0;
    let mut sticky = false;
    let mut saw_digit = false;
    while i < s.len() {
        let Some(d) = (s[i] as char).to_digit(16) else {
            break;
        };
        saw_digit = true;
        if mant <= (u128::MAX >> 4) {
            mant = (mant << 4) | u128::from(d);
        } else {
            sticky |= d != 0;
            exp2 += 4;
        }
        i += 1;
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() {
            let Some(d) = (s[i] as char).to_digit(16) else {
                break;
            };
            saw_digit = true;
            if mant <= (u128::MAX >> 4) {
                mant = (mant << 4) | u128::from(d);
                exp2 -= 4;
            } else {
                sticky |= d != 0;
            }
            i += 1;
        }
    }
    if !saw_digit {
        return None;
    }

    // Binary exponent (decimal digits, base 2). Saturating: an absurd exponent
    // only has to land on the correct side of `scale_by_pow2`'s clamp.
    if i < s.len() && (s[i] == b'p' || s[i] == b'P') {
        i += 1;
        let exp_negative = match s.get(i) {
            Some(b'+') => {
                i += 1;
                false
            }
            Some(b'-') => {
                i += 1;
                true
            }
            _ => false,
        };
        let digits_start = i;
        let mut e: i64 = 0;
        while i < s.len() && s[i].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add(i64::from(s[i] - b'0'));
            e = e.min(1 << 20);
            i += 1;
        }
        if i == digits_start {
            return None;
        }
        exp2 += if exp_negative { -e } else { e };
    }
    if i != s.len() {
        return None; // trailing bytes — not a complete literal
    }

    if mant == 0 {
        return Some(if negative { -0.0 } else { 0.0 });
    }
    if sticky {
        mant |= 1;
    }
    let value = scale_by_pow2(mant as f64, exp2.clamp(-100_000, 100_000) as i32);
    Some(if negative { -value } else { value })
}

crate::declare_proc! {
    /// Native `strtod` SimProcedure.
    ///
    /// Declared with `bv` arg modes (not `concrete`) so the FP-return calling-
    /// convention guard runs *before* the concrete-arg extraction; the macro's
    /// eager `concrete` extraction would otherwise reorder the symbolic-arg
    /// Python fallback ahead of the unsupported-ABI `NotImplemented` path.
    name = "strtod",
    struct = NativeStrtod,
    args = [nptr: bv, endptr: bv],
    call |state| {
        // Without a modelled FP-return register (x86 st0, ARM soft-float
        // r0:r1, MIPS hard-float $f0 vs soft-float $v0) we have nowhere to put
        // the double — defer to Python.
        let fp_ret = cc_for_arch(state.arch().name())
            .and_then(|cc| cc.fp_return_register())
            .ok_or(ProcedureError::NotImplemented)?;

        let nptr = extract_concrete_arg(&nptr, "nptr")?;
        let endptr = extract_concrete_arg(&endptr, "endptr")?;

        // Scan the concrete numeric prefix up to the first null (cap = MAX_LEN;
        // hitting the cap is fine — parsing stops at the first non-numeric byte
        // anyway). A symbolic byte propagates as `Err(SymbolicArgument)` and
        // falls back to Python: strtod has no useful symbolic-FP story without
        // a real floating-point solver.
        let (bytes, _null_found) = scan_concrete_bounded(state, nptr, MAX_LEN, "nptr")?;

        let prefix_end = floating_prefix_len(&bytes);
        let (value, end_offset) = if prefix_end == 0 {
            (0.0f64, 0u64)
        } else {
            // Skip leading whitespace to find the start of the parseable run
            // for f64::from_str (it does not accept leading whitespace).
            let mut start = 0;
            while start < bytes.len() && is_c_space(bytes[start]) {
                start += 1;
            }
            let slice = &bytes[start..prefix_end];
            // SILENT(cat-a): both `.ok()`s collapse into the `None` arm below,
            // which is an explicit loud fallback to Python — nothing is
            // discarded in favour of a degraded result.
            let decimal = || std::str::from_utf8(slice).ok()?.parse::<f64>().ok();
            let parsed = match parse_hex_float(slice).or_else(decimal) {
                Some(v) => v,
                // `floating_prefix_len` accepted a prefix neither parser can
                // turn into a value. Substituting 0.0 while still advancing
                // `*endptr` past the whole literal reports a full, correct
                // parse of the wrong number (angr-sqfj8.81) — hand the call to
                // Python instead.
                None => return Err(ProcedureError::NotImplemented),
            };
            (parsed, prefix_end as u64)
        };

        // Write *endptr if requested. C contract: if endptr is non-null, store
        // a pointer to the first byte past the parsed literal (or `nptr` if
        // nothing was consumed).
        if endptr != 0 {
            let end_addr = nptr.wrapping_add(end_offset);
            state.memory_store(endptr, arch_word(state, end_addr))?;
        }

        // Write the f64 bit pattern to the low 64 bits of the FP-return
        // register (xmm0 on amd64, v0 on AArch64). Leave the upper 64 bits
        // untouched — both ABIs carry the scalar double in the low slot only.
        let bits = value.to_bits();
        let ret_bv = RustBV::concrete(bits as u128, 64);
        state.set_register_by_offset(fp_ret, ret_bv);

        // Suppress the dispatcher's default integer-return-register store —
        // we've placed the value in the FP register already, and writing the
        // integer return register would pollute an unrelated slot.
        Ok(None)
    }
}

#[cfg(test)]
#[path = "strtod_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
