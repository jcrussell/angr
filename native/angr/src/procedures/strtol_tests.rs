//! Tests for the strtol/strtoll SimProcedure (extracted from strtol.rs).

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
    // atoi returns int-width bits: Python `val[31:0]` = 0xFFFFFF85, then
    // zero-extended into rax (NOT sign-extended to 0xFFFF..FF85). See angr-vu1q4.
    assert_eq!(result.as_u64(), Some(0xFFFF_FF85));
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

/// The leading-whitespace skip must use the C-locale `isspace` set, which
/// includes vertical tab (0x0b) and form feed (0x0c) — Rust's
/// `is_ascii_whitespace` drops `\v`, so `"\v42"` used to parse as 0
/// (angr-2j9sk). See `procedures::ctype::is_c_space`.
#[test]
fn test_atoi_skips_full_c_isspace_set() {
    for ws in [b' ', 0x09, 0x0a, 0x0b, 0x0c, 0x0d] {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, &[ws, b'4', b'2']);
        let result = NativeAtoi
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(42), "atoi with leading {ws:#04x}");
    }
}

/// Bytes just outside the whitespace run must *not* be skipped.
#[test]
fn test_atoi_does_not_skip_non_space_control_bytes() {
    for non_ws in [0x08u8, 0x0e] {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, &[non_ws, b'4', b'2']);
        let result = NativeAtoi
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0), "atoi with leading {non_ws:#04x}");
    }
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

#[cfg(feature = "vex-engine-z3")]
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

#[cfg(feature = "vex-engine-z3")]
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

#[cfg(feature = "vex-engine-z3")]
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

#[cfg(feature = "vex-engine-z3")]
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

#[cfg(feature = "vex-engine-z3")]
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
    // atoi extracts int-width bits: -42 -> low 32 bits (0xFFFFFFD6) zero-extended
    // into the 64-bit register, matching Python's `val[31:0]`. See angr-vu1q4.
    assert_eq!(ctx.min(&result, false), Some(0xFFFF_FFD6));
    assert_eq!(ctx.max(&result, false), Some(0xFFFF_FFD6));
}

#[test]
fn test_strtoll_concrete_64bit_value() {
    // strtoll must accept values that exceed 32-bit range — pick a value
    // that would overflow i32 but fits in i64.
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"9223372036854775806"); // i64::MAX - 1
    let p = NativeStrtoll;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(10, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(9223372036854775806u64));
}

#[test]
fn test_strtoull_concrete_above_i64_max() {
    // strtoull must accept the full u64 range (above i64::MAX).
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"18446744073709551614"); // u64::MAX - 1
    let p = NativeStrtoull;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(10, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(u64::MAX - 1));
}

#[test]
fn test_strtoll_negative_hex() {
    // strtoll with base 16 and a negative sign.
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"-0xff");
    let p = NativeStrtoll;
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
    assert_eq!(result.as_u64(), Some((-255i64) as u64));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_strtoll_symbolic_digits_constrained() {
    // Symbolic digits in a buffer constrained to "456" → 456 via strtoll.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"XXX\x00", Permission::RWX);
    let s0 = place_symbolic_byte(&mut state, 0x1000, "lld0");
    let s1 = place_symbolic_byte(&mut state, 0x1001, "lld1");
    let s2 = place_symbolic_byte(&mut state, 0x1002, "lld2");
    let ctx = state.solver().borrow();
    let four = RustBV::concrete(b'4' as u128, 8);
    let five = RustBV::concrete(b'5' as u128, 8);
    let six = RustBV::concrete(b'6' as u128, 8);
    let c0 = s0.eq(&four, &ctx);
    let c1 = s1.eq(&five, &ctx);
    let c2 = s2.eq(&six, &ctx);
    drop(ctx);

    let p = NativeStrtoll;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(10, 64),
            ],
        )
        .unwrap()
        .unwrap();
    state.add_constraint(c0);
    state.add_constraint(c1);
    state.add_constraint(c2);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(456));
    assert_eq!(ctx.max(&result, false), Some(456));
}

#[cfg(feature = "vex-engine-z3")]
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

#[test]
fn test_strtoll_ilp32_falls_back_to_python() {
    // On ILP32 (x86 / arm32) `long long` is 64 bits while the integer return
    // register is 32 bits (the ABI splits the result across edx:eax). The
    // native engine cannot plumb a wide return through a single return
    // register, so it declines with NotImplemented and lets the Python
    // SimProcedure (which knows the split-register calling convention) take
    // over. This test pins that intentional fallback so the guard in
    // strtol.rs is not silently removed.
    let mut state = RustSimState::new("x86").unwrap();
    setup_string(&mut state, 0x1000, b"123");
    let p = NativeStrtoll;
    let err = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(10, 32),
            ],
        )
        .unwrap_err();
    assert!(matches!(err, ProcedureError::NotImplemented));
}

#[test]
fn test_strtoull_ilp32_falls_back_to_python() {
    // Unsigned sibling of the test above — same ILP32 split-return rationale.
    let mut state = RustSimState::new("x86").unwrap();
    setup_string(&mut state, 0x1000, b"123");
    let p = NativeStrtoull;
    let err = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(10, 32),
            ],
        )
        .unwrap_err();
    assert!(matches!(err, ProcedureError::NotImplemented));
}

#[test]
fn test_atoi_overflow_clamps_x86() {
    // Axis 1 (angr-vu1q4): on 32-bit archs Python's `_string_to_int` clamps
    // the magnitude at the signed max `2^31 - 1`. atoi("3000000000") on x86
    // → Python eax 0x7FFFFFFF, whereas the old wrapping accumulator produced
    // 0xB2D05E00. Pin the clamp.
    let mut state = RustSimState::new("x86").unwrap();
    setup_string(&mut state, 0x1000, b"3000000000");
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 32)])
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x7FFF_FFFF));
}

#[test]
fn test_atoi_int_width_truncates_amd64() {
    // Axis 3 (angr-vu1q4): atoi returns int-width bits (Python
    // `val[sizeof(int)*8 - 1 : 0]`). atoi("10000000000") on amd64:
    // magnitude 0x2_540B_E400, low 32 bits 0x540B_E400 zero-extended into
    // rax — NOT the full arch-width 0x2_540B_E400. On amd64 the 11-digit cap
    // does not bind (10 digits) so the value survives to the int-width slice.
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"10000000000");
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x540B_E400));
}
