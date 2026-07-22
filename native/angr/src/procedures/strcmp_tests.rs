//! Tests for the strcmp/strncmp/strcasecmp SimProcedures (extracted from strcmp.rs).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

#[test]
fn test_strcmp_equal() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_strcmp_less_than() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"abd\x00", Permission::RWX);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    // 'c' - 'd' = -1
    let val = result.unwrap().as_u128().unwrap() as i32;
    assert!(val < 0);
}

#[test]
fn test_strcmp_greater_than() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"abd\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"abc\x00", Permission::RWX);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    // 'd' - 'c' = 1
    let val = result.unwrap().as_u128().unwrap() as i32;
    assert!(val > 0);
}

/// Python's `strncmp` (and `strcmp`, which inline-calls it) returns exactly
/// -1 or 1 on a concrete mismatch, never the raw byte difference. Pin the
/// magnitude on byte pairs more than 1 apart, where a glibc-style raw diff
/// would show up as -2 / 2 (angr-e71o4).
#[test]
fn test_strcmp_concrete_mismatch_is_plus_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"a\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"c\x00", Permission::RWX);

    let proc = NativeStrcmp;
    let less = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    assert_eq!(less.unwrap().as_u128().unwrap() as i32, -1);

    // Swapped operands: 'c' vs 'a' is +2 as a raw diff, +1 under Python parity.
    let greater = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x1000, 64)],
        )
        .unwrap();
    assert_eq!(greater.unwrap().as_u128().unwrap() as i32, 1);
}

#[test]
fn test_strncmp_and_memcmp_concrete_mismatch_is_plus_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"xa\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"xd\x00", Permission::RWX);

    let args = [
        RustBV::concrete(0x1000, 64),
        RustBV::concrete(0x2000, 64),
        RustBV::concrete(2, 64),
    ];
    let n = NativeStrncmp.call(&mut state, &args).unwrap();
    assert_eq!(n.unwrap().as_u128().unwrap() as i32, -1);

    // memcmp shares the same concrete closure, and Python's memcmp.py also
    // returns BVV(-1)/BVV(1).
    let m = crate::procedures::memcmp::NativeMemcmp
        .call(&mut state, &args)
        .unwrap();
    assert_eq!(m.unwrap().as_u128().unwrap() as i32, -1);
}

/// Case folding happens before the sign is taken, so 'A' vs 'b' is still -1.
#[test]
fn test_strcasecmp_concrete_mismatch_is_plus_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"A\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"d\x00", Permission::RWX);

    let result = NativeStrcasecmp
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u128().unwrap() as i32, -1);
}

#[test]
fn test_strcmp_prefix() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello world\x00", Permission::RWX);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    // '\0' - ' ' < 0
    let val = result.unwrap().as_u128().unwrap() as i32;
    assert!(val < 0);
}

#[test]
fn test_strncmp_limit() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"hello1\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello2\x00", Permission::RWX);

    let proc = NativeStrncmp;

    // Compare only first 5 chars (should be equal)
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0));

    // Compare 6 chars (should differ)
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(6, 64),
            ],
        )
        .unwrap();

    let val = result.unwrap().as_u128().unwrap() as i32;
    assert!(val != 0);
}

#[test]
fn test_strcasecmp() {
    let mut state = RustSimState::new("amd64").unwrap();

    state.map_memory_data(0x1000, b"Hello\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

    let proc = NativeStrcasecmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_strncasecmp_limit() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Differ only in case (offsets 0-4) AND in the trailing digit (offset 5).
    state.map_memory_data(0x1000, b"HELLO1\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello2\x00", Permission::RWX);

    let proc = NativeStrncasecmp;

    // First 5 chars are case-insensitively equal.
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // Including the 6th char ('1' vs '2') makes them differ.
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(6, 64),
            ],
        )
        .unwrap();
    let val = result.unwrap().as_u128().unwrap() as i32;
    assert!(val != 0);
}

#[test]
fn test_strncasecmp_registered() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    assert!(registry.has_native("strncasecmp"));
}

// ---------- Symbolic-byte tests ----------

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
fn test_strcmp_symbolic_byte_returns_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Concrete s2 = "ab\0", s1 = [a, ?, \0]. Symbolic byte at offset 1.
    state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
    state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
    let _sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.width(), 32);
    assert!(result.as_u64().is_none(), "expected symbolic result");
}

#[test]
fn test_strcmp_symbolic_byte_solver_evaluation_equal() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
    state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");
    // Constrain symbolic byte to 'b' so strings are equal -> 0.
    let ctx = state.solver().borrow();
    let target = RustBV::concrete(b'b' as u128, 8);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0));
    assert_eq!(ctx.max(&result, false), Some(0));
}

#[test]
fn test_strcmp_symbolic_byte_solver_evaluation_mismatch() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
    state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");
    // Constrain symbolic byte to 'c' so 'c' - 'b' = 1.
    let ctx = state.solver().borrow();
    let target = RustBV::concrete(b'c' as u128, 8);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);

    let proc = NativeStrcmp;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    // The 32-bit result equals 1 (signed).
    assert_eq!(ctx.min(&result, false), Some(1));
    assert_eq!(ctx.max(&result, false), Some(1));
}

#[test]
fn test_strncmp_symbolic_byte_within_limit_equal() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, b"abx", Permission::RWX);
    state.map_memory_data(0x1000, b"a\x00x", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

    // Force sym == 'b'; with n=3, expecting 0 (since byte 2 is 'x' both).
    let ctx = state.solver().borrow();
    let target = RustBV::concrete(b'b' as u128, 8);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);

    let proc = NativeStrncmp;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0));
    assert_eq!(ctx.max(&result, false), Some(0));
}
