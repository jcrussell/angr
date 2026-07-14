// Tests for stub.rs (NativeReturnUnconstrained).
// Extracted from the parent module; see the `#[path]` attr in stub.rs.
use super::*;
use crate::procedures::{NativeProcedureRegistry, NativeSimProcedure};
use crate::state::RustSimState;
use std::sync::Arc;

#[test]
fn test_stub_returns_symbolic_of_prototype_width() {
    let mut state = RustSimState::new("amd64").unwrap();
    let proc = NativeReturnUnconstrained::new("get_flag", 32);
    let ret = proc.call(&mut state, &[]).unwrap().unwrap();
    assert!(ret.as_u64().is_none(), "return must be unconstrained");
    assert_eq!(
        ret.width(),
        32,
        "width comes from the prototype return type"
    );
}

#[test]
fn test_stub_is_unconstrained_not_zero() {
    // "Unconstrained" means every value stays feasible — the Python stub adds
    // no constraint at all.
    let mut state = RustSimState::new("amd64").unwrap();
    let proc = NativeReturnUnconstrained::new("mystery", 64);
    let ret = proc.call(&mut state, &[]).unwrap().unwrap();
    let ctx = state.solver().borrow();
    for v in [0u128, 1, 0xdead_beef] {
        let eq = ret.eq(&RustBV::concrete(v, 64), &ctx);
        assert!(ctx.can_be_true(&eq), "ret=={v} must stay feasible");
    }
}

#[test]
fn test_stub_mints_a_fresh_symbol_per_call() {
    let mut state = RustSimState::new("amd64").unwrap();
    let proc = NativeReturnUnconstrained::new("fresh", 64);
    let a = proc.call(&mut state, &[]).unwrap().unwrap();
    let b = proc.call(&mut state, &[]).unwrap().unwrap();
    let ctx = state.solver().borrow();
    let differ = a.eq(&b, &ctx).not(&ctx);
    assert!(
        ctx.can_be_true(&differ),
        "two calls must mint independent symbols"
    );
}

#[test]
fn test_stub_registers_under_its_display_name() {
    let mut registry = NativeProcedureRegistry::empty();
    registry.register(Arc::new(NativeReturnUnconstrained::new("get_flag", 32)));
    assert!(registry.has_native("get_flag"));
    assert_eq!(registry.get("get_flag").unwrap().num_args(), 0);
}
