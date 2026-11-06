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
    let sym = RustBV::symbol(name, 8);
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
fn test_atoi_symbolic_digit_returns_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Buffer: "?\0" — one symbolic byte, constrained to a digit, then null.
    state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
    state.add_constraint(sym.uge(&RustBV::concrete(b'0' as u128, 8)));
    state.add_constraint(sym.ule(&RustBV::concrete(b'9' as u128, 8)));
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u64().is_none(), "expected symbolic result");
}

/// Run native strtol(0x10000, NULL, `base`) over `text`, with the bytes at
/// `sym_at` replaced by free symbolic bytes, and return the call's result.
fn strtol_with_free_bytes(
    text: &[u8],
    sym_at: &[u64],
    base: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x10000, text);
    for &i in sym_at {
        place_symbolic_byte(&mut state, 0x10000 + i, &format!("b{i}"));
    }
    NativeStrtol.call(
        &mut state,
        &[
            RustBV::concrete(0x10000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(u128::from(base), 64),
        ],
    )
}

/// A symbolic byte that may be a sign, or may complete a `0` / `0x` base
/// prefix the base auto-detects, defers to Python: the digit accumulator
/// models neither, so test_strtol's `+8` / `-0x` paths were lost and a
/// wrapped unsigned run reached its `-0xcdcd` branch (angr-mx00p.132.75).
#[test]
fn test_strtol_undecided_sign_or_prefix_falls_back_to_python() {
    // (text, symbolic offsets, base)
    let cases: &[(&[u8], &[u64], u64)] = &[
        (b"XX", &[0, 1], 10),  // sign
        (b"  X5", &[2], 10),   // sign after concrete whitespace
        (b"X5", &[0], 0),      // leading `0`: octal
        (b"-X5", &[1], 0),     // leading `0` after a concrete sign
        (b"0X5", &[1], 0),     // `0` then `x` or an octal digit
        (b"0X5", &[1], 16),    // `0` then `x`
        (b"Xx5", &[0], 16),    // `0` before a concrete `x`
    ];
    for &(text, sym_at, base) in cases {
        let shown = String::from_utf8_lossy(text);
        assert!(
            matches!(
                strtol_with_free_bytes(text, sym_at, base),
                Err(ProcedureError::SymbolicArgument(_))
            ),
            "{shown:?} base {base} with {sym_at:?} symbolic must defer to Python"
        );
    }
}

/// The same shapes stay native once the sign or prefix is decided: a concrete
/// sign, or a base whose detection the symbolic byte cannot change.
#[test]
fn test_strtol_decided_prefix_with_symbolic_digits_stays_native() {
    let cases: &[(&[u8], &[u64], u64)] = &[
        (b"-XX", &[1, 2], 10),
        (b"+X", &[1], 8),
        (b"0X5", &[1], 8),
        (b"0x5X", &[3], 16),
        (b"7X", &[1], 0),
    ];
    for &(text, sym_at, base) in cases {
        let shown = String::from_utf8_lossy(text);
        let out = strtol_with_free_bytes(text, sym_at, base);
        assert!(
            matches!(out, Ok(Some(_))),
            "{shown:?} base {base} with {sym_at:?} symbolic must stay native: {out:?}"
        );
    }
}

#[test]
fn test_atoi_single_symbolic_digit_constrained() {
    // Constrain the single byte to '7' — atoi should solve to 7.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
    let ctx = state.solver().borrow();
    let target = RustBV::concrete(b'7' as u128, 8);
    let eq = sym.eq(&target);
    drop(ctx);

    state.add_constraint(eq);
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
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
    let c0 = s0.eq(&one);
    let c1 = s1.eq(&two);
    let c2 = s2.eq(&three);
    drop(ctx);

    state.add_constraint(c0);
    state.add_constraint(c1);
    state.add_constraint(c2);
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
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
    let c_lo = sym.uge(&zero);
    let c_hi = sym.ule(&nine);
    drop(ctx);

    state.add_constraint(c_lo);
    state.add_constraint(c_hi);
    let p = NativeAtoi;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
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
    let eq = sym.eq(&space);
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
    let c0 = s0.eq(&four);
    let c1 = s1.eq(&two);
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
    let c0 = s0.eq(&four);
    let c1 = s1.eq(&five);
    let c2 = s2.eq(&six);
    drop(ctx);

    state.add_constraint(c0);
    state.add_constraint(c1);
    state.add_constraint(c2);
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
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(456));
    assert_eq!(ctx.max(&result, false), Some(456));
}

#[test]
fn test_strtol_base_16_symbolic_letter_digit() {
    // strtol with base=16, symbolic byte constrained to 'a' -> 10.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"X\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1000, "d0");
    let ctx = state.solver().borrow();
    let a = RustBV::concrete(b'a' as u128, 8);
    let eq = sym.eq(&a);
    drop(ctx);

    state.add_constraint(eq);
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

// ---------- '0x' prefix with no hex digit after it (bd angr-bmfj0) ----------

/// Drive `strtol(nptr, &e, base)` with the string at 0x1000 and `*endptr`
/// at 0x2000; return `(value, endptr_offset_from_nptr)`.
fn strtol_with_endptr(s: &[u8], base: u64) -> (u64, u64) {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, s);
    state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);
    let result = NativeStrtol
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(base as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    let end = state.memory_load(0x2000, 8).unwrap();
    (
        result.as_u64().unwrap(),
        end.as_u64().unwrap() - 0x1000,
    )
}

/// C's longest-valid-subject-sequence rule: `0x` not followed by a hex digit
/// backtracks to the leading `0`, which converts on its own. Pre-fix the
/// scanner consumed the prefix unconditionally, so `*endptr` landed past the
/// `x` and the `0` was discarded.
#[test]
fn test_strtol_hex_prefix_without_digits_backtracks_to_zero() {
    // (input, base, expected endptr offset) — value is 0 in every row.
    let cases: &[(&[u8], u64, u64)] = &[
        (b"0xg", 16, 1),
        (b"0Xg", 16, 1),
        (b"0x", 16, 1),
        (b"0x.", 16, 1),
        (b"-0xz", 16, 2),
        (b"+0xz", 16, 2),
        (b"  -0xz", 16, 4),
        // base 0 auto-detect takes the same backtrack (falls back to octal).
        (b"0xg", 0, 1),
        (b"-0xz", 0, 2),
    ];
    for &(s, base, want_end) in cases {
        let (val, end) = strtol_with_endptr(s, base);
        let shown = String::from_utf8_lossy(s);
        assert_eq!(val, 0, "value for {shown:?} base {base}");
        assert_eq!(end, want_end, "endptr for {shown:?} base {base}");
    }
}

/// The success path is unchanged: a hex digit after `0x` still consumes the
/// prefix rather than parsing the leading `0` as a lone digit.
#[test]
fn test_strtol_hex_prefix_with_digits_still_consumes_prefix() {
    let cases: &[(&[u8], u64, u64, u64)] = &[
        (b"0x1g", 16, 1, 3),
        (b"0xff", 16, 255, 4),
        (b"-0x10", 16, (-16i64) as u64, 5),
        (b"0x1g", 0, 1, 3),
    ];
    for &(s, base, want_val, want_end) in cases {
        let (val, end) = strtol_with_endptr(s, base);
        let shown = String::from_utf8_lossy(s);
        assert_eq!(val, want_val, "value for {shown:?} base {base}");
        assert_eq!(end, want_end, "endptr for {shown:?} base {base}");
    }
}

// ---------- scan window longer than MAX_DIGITS (bd angr-fs8kb.37) ----------

/// A concrete digit run longer than the initial `MAX_DIGITS` window used to be
/// truncated at 64 raw bytes, so `*endptr` landed mid-run. The window now grows
/// while it keeps ending on a byte the subject sequence could continue through.
#[test]
fn test_strtol_endptr_past_long_digit_run() {
    // 100 digits, then a terminator the digit parser cannot consume.
    let mut s = vec![b'7'; 100];
    s.push(b'!');
    let (_, end) = strtol_with_endptr(&s, 10);
    assert_eq!(end, 100, "endptr must point past the whole digit run");

    // The prefix counts against the same window: 40 spaces + a sign + 40
    // digits is 81 raw bytes, past MAX_DIGITS but well short of the run above.
    let mut s = vec![b' '; 40];
    s.push(b'-');
    s.extend_from_slice(&[b'3'; 40]);
    s.push(b'x');
    let (_, end) = strtol_with_endptr(&s, 10);
    assert_eq!(end, 81);
}

/// Base 2 is the case where truncation corrupted the *value* too: the `u128`
/// accumulator needs 128 binary digits before it saturates, so a run past the
/// old 64-byte window was neither fully parsed nor clamped — it just came back
/// short by a factor of two per dropped digit.
#[test]
fn test_strtol_base2_value_past_initial_window() {
    // 10 leading zeros + '1' + 60 zeros = 71 digits, value 2^60. Truncating the
    // run at 64 raw bytes drops 7 trailing zeros and yields 2^53 instead.
    let mut s = vec![b'0'; 10];
    s.push(b'1');
    s.extend_from_slice(&[b'0'; 60]);
    s.push(b'2'); // not a base-2 digit: terminates the run
    let (val, end) = strtol_with_endptr(&s, 2);
    assert_eq!(val, 1u64 << 60);
    assert_eq!(end, 71);
}

/// The growth loop stops at `MAX_SCAN_BYTES`; past it the old truncation
/// semantics stand, deliberately.
#[test]
fn test_strtol_scan_window_ceiling_truncates() {
    let mut state = RustSimState::new("amd64").unwrap();
    // 5000 digits: longer than MAX_SCAN_BYTES (4096).
    let digits = vec![b'9'; 5000];
    setup_string(&mut state, 0x10000, &digits);
    state.map_memory_data(0x100000, &[0u8; 8], Permission::RWX);
    NativeStrtol
        .call(
            &mut state,
            &[
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(0x100000, 64),
                RustBV::concrete(10, 64),
            ],
        )
        .unwrap()
        .unwrap();
    let end = state.memory_load(0x100000, 8).unwrap();
    assert_eq!(end.as_u64().unwrap() - 0x10000, MAX_SCAN_BYTES as u64);
}

/// Call native strtol(0x10000, 0x100000, 10) and return the stored `*endptr`.
fn strtol_symbolic_endptr(state: &mut RustSimState) -> RustBV {
    state.map_memory_data(0x100000, &[0u8; 8], Permission::RWX);
    NativeStrtol
        .call(
            state,
            &[
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(0x100000, 64),
                RustBV::concrete(10, 64),
            ],
        )
        .unwrap()
        .unwrap();
    state.memory_load(0x100000, 8).unwrap()
}

/// A symbolic byte stops the growth: the symbolic accumulator caps at
/// `MAX_DIGITS` digit positions, so the longest `*endptr` stays at that bound
/// rather than tracking a window the concrete path grew.
#[test]
fn test_strtol_symbolic_digits_endptr_capped_at_max_digits() {
    let mut state = RustSimState::new("amd64").unwrap();
    let digits = vec![b'1'; 200];
    setup_string(&mut state, 0x10000, &digits);
    // Make the first byte symbolic so the all-concrete fast path is skipped.
    let d0 = place_symbolic_byte(&mut state, 0x10000, "d0");
    state.add_constraint(d0.eq(&RustBV::concrete(b'1' as u128, 8)));
    let end = strtol_symbolic_endptr(&mut state);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&end, false), Some(0x10000 + MAX_DIGITS as u128));
    assert_eq!(ctx.max(&end, false), Some(0x10000 + MAX_DIGITS as u128));
}

/// Two digits then two free bytes: `*endptr` covers every run length the path
/// allows (s+2, s+3, s+4), not just the end of the window (angr-vmi2l.101).
#[test]
fn test_strtol_symbolic_short_run_endptr_tracks_run() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x10000, b"XXXX");
    let mut syms = Vec::new();
    for i in 0..4u64 {
        syms.push(place_symbolic_byte(&mut state, 0x10000 + i, &format!("d{i}")));
    }
    for d in &syms[..2] {
        state.add_constraint(d.uge(&RustBV::concrete(b'0' as u128, 8)));
        state.add_constraint(d.ule(&RustBV::concrete(b'9' as u128, 8)));
    }
    let end = strtol_symbolic_endptr(&mut state);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&end, false), Some(0x10002));
    assert_eq!(ctx.max(&end, false), Some(0x10004));
    drop(ctx);
    // A non-digit third byte pins the end at s+2.
    state.add_constraint(syms[2].eq(&RustBV::concrete(b'z' as u128, 8)));
    let ctx = state.solver().borrow();
    assert_eq!(ctx.max(&end, false), Some(0x10002));
}

/// No digit converts (`s[0] == 'x'`): C stores `nptr` itself.
#[test]
fn test_strtol_symbolic_no_conversion_endptr_is_nptr() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x10000, b"XXXX");
    let mut syms = Vec::new();
    for i in 0..4u64 {
        syms.push(place_symbolic_byte(&mut state, 0x10000 + i, &format!("d{i}")));
    }
    state.add_constraint(syms[0].eq(&RustBV::concrete(b'x' as u128, 8)));
    let end = strtol_symbolic_endptr(&mut state);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&end, false), Some(0x10000));
    assert_eq!(ctx.max(&end, false), Some(0x10000));
}

/// No conversion stores `nptr` itself, not past the skipped whitespace and
/// sign; a base-0 leading `0` is a conversion, so its end stays past it
/// (angr-vmi2l.101).
#[test]
fn test_strtol_no_conversion_endptr_is_nptr() {
    // (input, base, expected endptr offset) — value is 0 in every row.
    let cases: &[(&[u8], u64, u64)] = &[
        (b"  -x", 10, 0),
        (b"x", 10, 0),
        (b"+", 10, 0),
        (b"   ", 10, 0),
        (b"09", 0, 1),
        (b"0z", 0, 1),
    ];
    for &(s, base, want_end) in cases {
        let (val, end) = strtol_with_endptr(s, base);
        let shown = String::from_utf8_lossy(s);
        assert_eq!(val, 0, "value for {shown:?} base {base}");
        assert_eq!(end, want_end, "endptr for {shown:?} base {base}");
    }
}

/// Symbolic arm, base 0: the octal-detecting `0` converts even when no digit
/// follows it, so `*endptr` is past it, not `nptr`.
#[test]
fn test_strtol_symbolic_base0_leading_zero_endptr_past_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x10000, b"0zX");
    place_symbolic_byte(&mut state, 0x10002, "d2");
    state.map_memory_data(0x100000, &[0u8; 8], Permission::RWX);
    NativeStrtol
        .call(
            &mut state,
            &[
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(0x100000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap()
        .unwrap();
    let end = state.memory_load(0x100000, 8).unwrap();
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&end, false), Some(0x10001));
    assert_eq!(ctx.max(&end, false), Some(0x10001));
}
