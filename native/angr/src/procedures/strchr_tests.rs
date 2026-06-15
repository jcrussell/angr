//! Tests for the strchr/strrchr/memchr SimProcedures (extracted from strchr.rs).
use super::*;
use crate::memory::Permission;

#[test]
fn test_strchr_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let p = NativeStrchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'l' as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1002));
}

#[test]
fn test_strchr_not_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let p = NativeStrchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'z' as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0)); // NULL
}

#[test]
fn test_strchr_finds_null() {
    // C standard: strchr(s, '\0') returns pointer to the null terminator.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);
    let p = NativeStrchr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0u128, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1002));
}

#[test]
fn test_memchr_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

    let p = NativeMemchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1002));
}

#[test]
fn test_memchr_not_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

    let p = NativeMemchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0xFF, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0)); // NULL
}

#[test]
fn test_strchr_symbolic_target_returns_symbolic() {
    // Symbolic target should produce a symbolic result, not a fallback error.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    drop(ctx);
    let p = NativeStrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u64().is_none(), "expected symbolic, got concrete");
}

#[test]
fn test_strchr_symbolic_target_solver_evaluation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    // Constrain c == 'b' so the only match is at 0x1001.
    let target = RustBV::concrete(b'b' as u128, 64);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);
    let p = NativeStrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0x1001));
    assert_eq!(ctx.max(&result, false), Some(0x1001));
}

#[test]
fn test_strchr_symbolic_target_null_match() {
    // Constraining target to 0 should yield the address of the null byte.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    let zero = RustBV::concrete(0u128, 64);
    let eq = sym.eq(&zero, &ctx);
    drop(ctx);
    let p = NativeStrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0x1003));
    assert_eq!(ctx.max(&result, false), Some(0x1003));
}

#[test]
fn test_strchr_symbolic_target_no_match_returns_null() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    // Constrain c == 'z' (not in the string and not null) -> NULL result.
    let target = RustBV::concrete(b'z' as u128, 64);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);
    let p = NativeStrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0));
    assert_eq!(ctx.max(&result, false), Some(0));
}

// --- strrchr ---

#[test]
fn test_strrchr_found_last() {
    let mut state = RustSimState::new("amd64").unwrap();
    // "hello" — 'l' at indices 2 and 3; last is 0x1003.
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    let p = NativeStrrchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'l' as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1003));
}

#[test]
fn test_strrchr_not_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    let p = NativeStrrchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'z' as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0));
}

#[test]
fn test_strrchr_finds_null() {
    // C standard: strrchr(s, '\0') → pointer to null terminator.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);
    let p = NativeStrrchr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0u128, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1002));
}

#[test]
fn test_strrchr_single_match() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let p = NativeStrrchr;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'b' as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1001));
}

#[test]
fn test_strrchr_symbolic_target_returns_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    drop(ctx);
    let p = NativeStrrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u64().is_none(), "expected symbolic, got concrete");
}

#[test]
fn test_strrchr_symbolic_target_picks_last() {
    // Constrain symbolic c to 'l' → last 'l' is at 0x1003 in "hello".
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    let l_byte = RustBV::concrete(b'l' as u128, 64);
    let eq = sym.eq(&l_byte, &ctx);
    drop(ctx);
    let p = NativeStrrchr;
    let result = p
        .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0x1003));
    assert_eq!(ctx.max(&result, false), Some(0x1003));
}

#[test]
fn test_memchr_symbolic_target_solver_evaluation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 64);
    let target = RustBV::concrete(3u128, 64);
    let eq = sym.eq(&target, &ctx);
    drop(ctx);
    let p = NativeMemchr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), sym, RustBV::concrete(4, 64)],
        )
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(0x1002));
    assert_eq!(ctx.max(&result, false), Some(0x1002));
}
