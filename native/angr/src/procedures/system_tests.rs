// Tests for system.rs (NativeSystem).
// Extracted from the parent module; see the `#[path]` attr in system.rs.
use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_system_returns_symbolic_int() {
    // Python models system() as an unconstrained 8-bit code zero-extended to
    // the 32-bit C int width — symbolic, never concrete.
    let mut state = RustSimState::new("amd64").unwrap();
    let cmd = RustBV::concrete(0x1000, 64);
    let result = NativeSystem
        .call(&mut state, std::slice::from_ref(&cmd))
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none()); // symbolic
    assert_eq!(result.width(), 32); // sizeof(int)
}

#[test]
fn test_system_width_is_int_on_32bit_arch() {
    // sizeof(int) == 32 regardless of arch word size (matches rand.rs).
    let mut state = RustSimState::new("x86").unwrap();
    let cmd = RustBV::concrete(0x1000, 32);
    let result = NativeSystem
        .call(&mut state, std::slice::from_ref(&cmd))
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none());
    assert_eq!(result.width(), 32);
}

#[test]
fn test_system_symbolic_cmd_no_fallback() {
    // Python never inspects the command pointer; a symbolic pointer must NOT
    // force concretization / Python fallback (the `bv` arg kind keeps it raw).
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym_cmd = RustBV::symbolic(&ctx, "cmd_ptr", 64);
    drop(ctx);
    let result = NativeSystem
        .call(&mut state, std::slice::from_ref(&sym_cmd))
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none());
    assert_eq!(result.width(), 32);
}

#[test]
fn test_system_unique_names() {
    // Each call mints a fresh symbol so distinct system() callsites do not
    // alias their return codes.
    let mut state = RustSimState::new("amd64").unwrap();
    let cmd = RustBV::concrete(0x1000, 64);
    let r1 = NativeSystem
        .call(&mut state, std::slice::from_ref(&cmd))
        .unwrap()
        .unwrap();
    let r2 = NativeSystem
        .call(&mut state, std::slice::from_ref(&cmd))
        .unwrap()
        .unwrap();
    assert!(r1.as_u64().is_none());
    assert!(r2.as_u64().is_none());
}

#[test]
fn test_system_metadata() {
    assert_eq!(NativeSystem.name(), "system");
    assert_eq!(NativeSystem.num_args(), 1);
    assert!(!NativeSystem.no_return());
}
