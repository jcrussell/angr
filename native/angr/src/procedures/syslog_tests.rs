// Tests for syslog.rs (NativeOpenlog / NativeCloselog).
// Extracted from the parent module; see the `#[path]` attr in syslog.rs.
use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_openlog_void_noop() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ident = RustBV::concrete(0x1000, 64);
    let option = RustBV::concrete(0, 64);
    let facility = RustBV::concrete(8, 64);
    let result = NativeOpenlog
        .call(&mut state, &[ident, option, facility])
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn test_closelog_void_noop() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeCloselog.call(&mut state, &[]).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_openlog_symbolic_args_no_fallback() {
    // Python ignores every argument; symbolic ident/option/facility must NOT
    // force concretization or a Python fallback (the `bv` arg kind keeps the
    // raw BV).
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let ident = RustBV::symbolic(&ctx, "ident", 64);
    let option = RustBV::symbolic(&ctx, "option", 64);
    let facility = RustBV::symbolic(&ctx, "facility", 64);
    drop(ctx);
    let result = NativeOpenlog
        .call(&mut state, &[ident, option, facility])
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn test_arg_counts() {
    assert_eq!(NativeOpenlog.num_args(), 3);
    assert_eq!(NativeCloselog.num_args(), 0);
}

#[test]
fn test_names() {
    assert_eq!(NativeOpenlog.name(), "openlog");
    assert_eq!(NativeCloselog.name(), "closelog");
}
