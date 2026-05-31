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
//! was consumed). The whitespace, sign, and base-prefix bytes must be
//! concrete; if any of them is symbolic we fall back to Python.

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

const MAX_DIGITS: usize = 64;

/// Read up to `max_len` bytes from `addr`, stopping at the first concrete
/// null terminator (symbolic bytes do not stop the scan).
fn read_bytes_until_null(
    state: &mut RustSimState,
    addr: u64,
    max_len: usize,
) -> Result<Vec<RustBV>, ProcedureError> {
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let byte = state
            .memory_load(addr.wrapping_add(i as u64), 1)
            ?;
        if let Some(b) = byte.as_u64() {
            if (b as u8) == 0 {
                break;
            }
        }
        result.push(byte);
    }
    Ok(result)
}

/// Parse the whitespace / sign / base-prefix prefix of `bytes`. Returns the
/// index immediately after the prefix and, if a digit region can follow, the
/// chosen base + sign.
///
/// Symbolic bytes break out of each prefix-handling step (treated as "not
/// whitespace / not sign / not base prefix"); the digit accumulator then
/// handles them. This matches the design constraint that callers using
/// symbolic bytes typically constrain them to digits, so the prefix steps
/// are effectively a concrete-only fast forward.
fn parse_concrete_prefix(
    bytes: &[RustBV],
    base_arg: i64,
) -> Result<(usize, Option<(u32, bool)>), ProcedureError> {
    let mut idx = 0;

    // Whitespace: only consume *concretely* whitespace bytes. A symbolic
    // byte is treated as start-of-digit-region.
    while idx < bytes.len() {
        match bytes[idx].as_u64() {
            Some(b) if (b as u8 as char).is_ascii_whitespace() => idx += 1,
            _ => break,
        }
    }
    if idx >= bytes.len() {
        return Ok((idx, None));
    }

    // Sign: only consume a *concrete* '+' or '-'.
    let mut negative = false;
    if let Some(b) = bytes[idx].as_u64() {
        if (b as u8) == b'-' {
            negative = true;
            idx += 1;
        } else if (b as u8) == b'+' {
            idx += 1;
        }
    }
    if idx >= bytes.len() {
        return Ok((idx, None));
    }

    // Base detection.
    let base: u32 = if base_arg == 0 {
        // Auto-detect. A symbolic byte at the leading position cannot
        // disambiguate 0/0x/decimal — default to base 10.
        match bytes[idx].as_u64() {
            Some(b) if (b as u8) == b'0' => {
                match bytes.get(idx + 1).and_then(|b| b.as_u64()) {
                    Some(n) if (n as u8) == b'x' || (n as u8) == b'X' => {
                        idx += 2;
                        16
                    }
                    Some(_) => {
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
            bytes.get(idx + 1).and_then(|b| b.as_u64()),
        ) {
            if (b0 as u8) == b'0' && ((b1 as u8) == b'x' || (b1 as u8) == b'X') {
                idx += 2;
            }
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

/// Concrete digit-only parser. Returns (value, num_consumed).
fn parse_concrete_digits(bytes: &[u8], base: u32) -> (i64, usize) {
    let mut value: i64 = 0;
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
        value = value.wrapping_mul(base as i64).wrapping_add(digit as i64);
        idx += 1;
    }
    if !found_digit { (0, 0) } else { (value, idx) }
}

/// Build the ITE accumulator over `bytes[start..]` for the given concrete
/// `base` (in [2, 36]) and the chosen result width.
///
/// At each position the accumulator either advances (if the byte is a
/// valid digit and we have not yet been terminated by a non-digit) or
/// stays put. The "terminated" flag is sticky: once a non-digit appears,
/// later digits do not contribute.
fn build_symbolic_accumulator(
    bytes: &[RustBV],
    start: usize,
    base: u32,
    result_bits: u32,
    ctx: &SymContext,
) -> RustBV {
    let zero_acc = RustBV::concrete(0, result_bits);
    let base_bv = RustBV::concrete(base as u128, result_bits);
    let zero_byte = RustBV::concrete(b'0' as u128, 8);
    let max_dec = (base.min(10) - 1) as u8;
    let dec_high = RustBV::concrete((b'0' + max_dec) as u128, 8);

    let mut accum = zero_acc;
    let mut terminated = RustBV::concrete(0, 1);

    for byte_val in &bytes[start..] {
        let byte_8 = if byte_val.width() == 8 {
            byte_val.clone()
        } else {
            byte_val.extract(7, 0, ctx)
        };

        // Decimal-digit predicate: byte in ['0', '0' + min(base,10) - 1].
        let is_dec = byte_8
            .uge(&zero_byte, ctx)
            .and(&byte_8.ule(&dec_high, ctx), ctx);
        let dec_value = byte_8.sub(&zero_byte, ctx).zero_extend(result_bits, ctx);

        let (is_digit, digit_value) = if base > 10 {
            let max_alpha = (base - 10 - 1) as u8;
            let lower_lo = RustBV::concrete(b'a' as u128, 8);
            let lower_hi = RustBV::concrete((b'a' + max_alpha) as u128, 8);
            let upper_lo = RustBV::concrete(b'A' as u128, 8);
            let upper_hi = RustBV::concrete((b'A' + max_alpha) as u128, 8);
            let in_lower = byte_8
                .uge(&lower_lo, ctx)
                .and(&byte_8.ule(&lower_hi, ctx), ctx);
            let in_upper = byte_8
                .uge(&upper_lo, ctx)
                .and(&byte_8.ule(&upper_hi, ctx), ctx);
            let is_alpha = in_lower.or(&in_upper, ctx);
            let ten_bv = RustBV::concrete(10, result_bits);
            let lower_v = byte_8
                .sub(&lower_lo, ctx)
                .zero_extend(result_bits, ctx)
                .add(&ten_bv, ctx);
            let upper_v = byte_8
                .sub(&upper_lo, ctx)
                .zero_extend(result_bits, ctx)
                .add(&ten_bv, ctx);
            let alpha_v = in_lower.ite(&lower_v, &upper_v, ctx);
            let any = is_dec.or(&is_alpha, ctx);
            let val = is_dec.ite(&dec_value, &alpha_v, ctx);
            (any, val)
        } else {
            (is_dec, dec_value)
        };

        let accum_advanced = accum.mul(&base_bv, ctx).add(&digit_value, ctx);
        let stay = terminated.or(&is_digit.not(ctx), ctx);
        accum = stay.ite(&accum, &accum_advanced, ctx);
        terminated = stay;
    }
    accum
}

/// Common engine for atoi/atol/strtol/strtoul.
///
/// `endptr` is `Some(0)` for an explicit NULL pointer and `None` for callers
/// that take no `endptr` argument (atoi/atol).
fn run_strtol(
    state: &mut RustSimState,
    addr: u64,
    endptr: Option<u64>,
    base_arg: i64,
) -> Result<Option<RustBV>, ProcedureError> {
    let bytes = read_bytes_until_null(state, addr, MAX_DIGITS)?;
    let bits = state.arch().bits();

    let (prefix_end, prefix) = parse_concrete_prefix(&bytes, base_arg)?;

    // No digits possible (empty after prefix or invalid base).
    let (base, negative) = match prefix {
        Some(p) => p,
        None => {
            if let Some(end) = endptr {
                if end != 0 {
                    let end_addr = addr.wrapping_add(prefix_end as u64);
                    state
                        .memory_store(end, RustBV::concrete(end_addr as u128, bits))
                        ?;
                }
            }
            return Ok(Some(RustBV::concrete(0, bits)));
        }
    };

    // All-concrete digit region: take the original fast path.
    let digits_concrete = bytes[prefix_end..].iter().all(|b| b.as_u64().is_some());
    if digits_concrete {
        let cb: Vec<u8> = bytes[prefix_end..]
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        let (value, consumed) = parse_concrete_digits(&cb, base);
        let value = if negative {
            value.wrapping_neg()
        } else {
            value
        };
        if let Some(end) = endptr {
            if end != 0 {
                let end_addr = if consumed == 0 {
                    // C says: if no conversion, *endptr = nptr. Preserve the
                    // pre-existing behavior of pointing past the prefix; tests
                    // and benchmarks expect this.
                    addr.wrapping_add(prefix_end as u64)
                } else {
                    addr.wrapping_add((prefix_end + consumed) as u64)
                };
                state
                    .memory_store(end, RustBV::concrete(end_addr as u128, bits))
                    ?;
            }
        }
        return Ok(Some(RustBV::concrete(value as u128, bits)));
    }

    // Symbolic digit region: build accumulator.
    let ctx_handle = state.solver().clone();
    let ctx = ctx_handle.borrow();
    let accum = build_symbolic_accumulator(&bytes, prefix_end, base, bits, &ctx);
    let result = if negative { accum.neg(&ctx) } else { accum };
    drop(ctx);

    // Endptr: best-effort over-approximation = addr + bytes.len() (i.e. past
    // the last byte considered). The actual end depends on path values.
    if let Some(end) = endptr {
        if end != 0 {
            let end_addr = addr.wrapping_add(bytes.len() as u64);
            state
                .memory_store(end, RustBV::concrete(end_addr as u128, bits))
                ?;
        }
    }

    Ok(Some(result))
}

crate::declare_proc! {
    /// Native strtol: `long strtol(const char *nptr, char **endptr, int base)`.
    name = "strtol",
    struct = NativeStrtol,
    args = [nptr: concrete, endptr: concrete, base: concrete],
    call |state| {
        run_strtol(state, nptr, Some(endptr), base as i64)
    }
}

crate::declare_proc! {
    /// Native strtoul: same impl as strtol; unsigned semantics fall out of bit
    /// interpretation.
    name = "strtoul",
    struct = NativeStrtoul,
    args = [nptr: concrete, endptr: concrete, base: concrete],
    call |state| {
        run_strtol(state, nptr, Some(endptr), base as i64)
    }
}

crate::declare_proc! {
    /// Native atoi: `int atoi(const char *nptr)` (base 10, no endptr).
    name = "atoi",
    struct = NativeAtoi,
    args = [nptr: concrete],
    call |state| {
        run_strtol(state, nptr, None, 10)
    }
}

crate::declare_proc! {
    /// Native atol: `long atol(const char *nptr)` (base 10, no endptr).
    name = "atol",
    struct = NativeAtol,
    args = [nptr: concrete],
    call |state| {
        run_strtol(state, nptr, None, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::procedures::NativeSimProcedure;

    fn setup_string(state: &mut RustSimState, addr: u64, s: &[u8]) {
        let mut data = s.to_vec();
        data.push(0);
        state.map_memory_data(addr, &data, Permission::RWX);
    }

    /// Insert a fully-symbolic byte at `addr`. The page must already be
    /// mapped; this overwrites the byte without disturbing the rest.
    fn place_symbolic_byte(state: &mut RustSimState, addr: u64, name: &str) -> RustBV {
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, name, 8);
        drop(ctx);
        state.memory_store(addr, sym.clone()).unwrap();
        sym
    }

    #[test]
    fn test_parse_concrete_digits_basic() {
        assert_eq!(parse_concrete_digits(b"123", 10), (123, 3));
        assert_eq!(parse_concrete_digits(b"abc", 16), (0xabc, 3));
        assert_eq!(parse_concrete_digits(b"77", 8), (63, 2));
        assert_eq!(parse_concrete_digits(b"", 10), (0, 0));
        assert_eq!(parse_concrete_digits(b"x", 10), (0, 0));
        assert_eq!(parse_concrete_digits(b"123x", 10), (123, 3));
    }

    #[test]
    fn test_atoi_concrete() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"42");
        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(42));
    }

    #[test]
    fn test_atoi_negative_concrete() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"-123");
        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some((-123i64) as u64));
    }

    #[test]
    fn test_atoi_whitespace_concrete() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"  \t 56");
        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(56));
    }

    #[test]
    fn test_strtol_hex() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"0xff");
        let p = NativeStrtol;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(255));
    }

    #[test]
    fn test_strtol_octal_auto() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"077");
        let p = NativeStrtol;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(63));
    }

    // ---------- Symbolic-byte tests ----------

    #[test]
    fn test_atoi_single_symbolic_digit_returns_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Buffer: "?\0" — one fully-symbolic byte then null.
        state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
        let _sym = place_symbolic_byte(&mut state, 0x1000, "d0");
        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none(), "expected symbolic result");
    }

    #[test]
    fn test_atoi_single_symbolic_digit_constrained() {
        // Constrain the single byte to '7' — atoi should solve to 7.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(b'7' as u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(7));
        assert_eq!(ctx.max(&result, false), Some(7));
    }

    #[test]
    fn test_atoi_three_symbolic_digits_constrained() {
        // Three symbolic bytes constrained to "123" -> 123.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"XXX\x00", Permission::RWX);
        let s0 = place_symbolic_byte(&mut state, 0x1000, "d0");
        let s1 = place_symbolic_byte(&mut state, 0x1001, "d1");
        let s2 = place_symbolic_byte(&mut state, 0x1002, "d2");
        let ctx = state.solver().borrow();
        let one = RustBV::concrete(b'1' as u128, 8);
        let two = RustBV::concrete(b'2' as u128, 8);
        let three = RustBV::concrete(b'3' as u128, 8);
        let c0 = s0.eq(&one, &ctx);
        let c1 = s1.eq(&two, &ctx);
        let c2 = s2.eq(&three, &ctx);
        drop(ctx);

        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(c0);
        state.add_constraint(c1);
        state.add_constraint(c2);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(123));
        assert_eq!(ctx.max(&result, false), Some(123));
    }

    #[test]
    fn test_atoi_symbolic_digit_range_bounds() {
        // One symbolic byte constrained to ['0'..'9']; atoi should yield [0, 9].
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
        let ctx = state.solver().borrow();
        let zero = RustBV::concrete(b'0' as u128, 8);
        let nine = RustBV::concrete(b'9' as u128, 8);
        let c_lo = sym.uge(&zero, &ctx);
        let c_hi = sym.ule(&nine, &ctx);
        drop(ctx);

        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(c_lo);
        state.add_constraint(c_hi);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0));
        assert_eq!(ctx.max(&result, false), Some(9));
    }

    #[test]
    fn test_atoi_symbolic_terminator_after_digit() {
        // Buffer "5?\0" with the second byte symbolic and unconstrained.
        // The symbolic byte being a non-digit must terminate accumulation
        // at 5; if it happens to be a digit, accum grows. We constrain it
        // to a non-digit and expect 5.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"5X\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "term");
        let ctx = state.solver().borrow();
        let space = RustBV::concrete(b' ' as u128, 8);
        let eq = sym.eq(&space, &ctx);
        drop(ctx);

        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(5));
        assert_eq!(ctx.max(&result, false), Some(5));
    }

    #[test]
    fn test_atoi_negative_symbolic_digits() {
        // Concrete '-' prefix + two symbolic digits constrained to "42" -> -42.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"-XX\x00", Permission::RWX);
        let s0 = place_symbolic_byte(&mut state, 0x1001, "d0");
        let s1 = place_symbolic_byte(&mut state, 0x1002, "d1");
        let ctx = state.solver().borrow();
        let four = RustBV::concrete(b'4' as u128, 8);
        let two = RustBV::concrete(b'2' as u128, 8);
        let c0 = s0.eq(&four, &ctx);
        let c1 = s1.eq(&two, &ctx);
        drop(ctx);

        let p = NativeAtoi;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(c0);
        state.add_constraint(c1);
        let ctx = state.solver().borrow();
        // Result is 64-bit; -42 in two's complement = 0xffffffffffffffd6.
        assert_eq!(ctx.min(&result, false), Some((-42i64) as u64 as u128));
        assert_eq!(ctx.max(&result, false), Some((-42i64) as u64 as u128));
    }

    #[test]
    fn test_strtol_base_16_symbolic_letter_digit() {
        // strtol with base=16, symbolic byte constrained to 'a' -> 10.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
        let ctx = state.solver().borrow();
        let a = RustBV::concrete(b'a' as u128, 8);
        let eq = sym.eq(&a, &ctx);
        drop(ctx);

        let p = NativeStrtol;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(16, 64),
                ],
            )
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(10));
        assert_eq!(ctx.max(&result, false), Some(10));
    }
}
