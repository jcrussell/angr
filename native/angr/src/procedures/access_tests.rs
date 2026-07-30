// Tests for access.rs (NativeAccess).
// Extracted from the parent module; see the `#[path]` attr in access.rs.
use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_access_returns_symbolic_int() {
    // Python models access() as a fresh symbolic int constrained to {0, -1} —
    // symbolic, never concrete, 32-bit C int width.
    let mut state = RustSimState::new("amd64").unwrap();
    let path = RustBV::concrete(0x1000, 64);
    let mode = RustBV::concrete(0, 64);
    let result = NativeAccess
        .call(&mut state, &[path, mode])
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none()); // symbolic
    assert_eq!(result.width(), 32); // sizeof(int)
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_access_constrained_to_zero_or_minus_one() {
    // The returned int must be feasibly 0 and feasibly -1, but NOT any other
    // value (e.g. 5) — that is the Or(ret==0, ret==-1) constraint.
    let mut state = RustSimState::new("amd64").unwrap();
    let path = RustBV::concrete(0x1000, 64);
    let mode = RustBV::concrete(0, 64);
    let result = NativeAccess
        .call(&mut state, &[path, mode])
        .unwrap()
        .unwrap();
    let ctx = state.solver().borrow();
    let is_zero = result.eq(&RustBV::concrete(0, 32), &ctx);
    let is_minus_one = result.eq(&RustBV::concrete(0xFFFF_FFFF, 32), &ctx);
    let is_five = result.eq(&RustBV::concrete(5, 32), &ctx);
    assert!(ctx.can_be_true(&is_zero), "ret==0 must be feasible");
    assert!(ctx.can_be_true(&is_minus_one), "ret==-1 must be feasible");
    assert!(!ctx.can_be_true(&is_five), "ret==5 must be infeasible");
}

#[test]
fn test_access_width_is_int_on_32bit_arch() {
    // sizeof(int) == 32 regardless of arch word size (matches rand.rs).
    let mut state = RustSimState::new("x86").unwrap();
    let path = RustBV::concrete(0x1000, 32);
    let mode = RustBV::concrete(0, 32);
    let result = NativeAccess
        .call(&mut state, &[path, mode])
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none());
    assert_eq!(result.width(), 32);
}

#[test]
fn test_access_symbolic_path_no_fallback() {
    // Python never inspects the path/mode args; a symbolic pointer must NOT
    // force concretization / Python fallback (the `bv` arg kind keeps it raw).
    let mut state = RustSimState::new("amd64").unwrap();
    let (sym_path, mode) = {
        let ctx = state.solver().borrow();
        (
            RustBV::symbolic(&ctx, "path_ptr", 64),
            RustBV::concrete(0, 64),
        )
    };
    let result = NativeAccess
        .call(&mut state, &[sym_path, mode])
        .unwrap()
        .unwrap();
    assert!(result.as_u64().is_none());
    assert_eq!(result.width(), 32);
}

#[test]
fn test_access_metadata() {
    assert_eq!(NativeAccess.name(), "access");
    assert_eq!(NativeAccess.num_args(), 2);
    assert!(!NativeAccess.no_return());
}
