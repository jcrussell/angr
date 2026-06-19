use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_exit_no_return() {
    let proc = NativeExit;
    assert!(proc.no_return());
    assert_eq!(proc.num_args(), 1);
    assert_eq!(proc.name(), "exit");
}

#[test]
fn test_exit_returns_none() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeExit
        .call(&mut state, &[RustBV::concrete(0, 32)])
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn test_underscore_exit() {
    let proc = NativeUnderscoreExit;
    assert!(proc.no_return());
    assert_eq!(proc.name(), "_exit");
    let mut state = RustSimState::new("amd64").unwrap();
    let result = proc.call(&mut state, &[RustBV::concrete(1, 32)]).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_abort() {
    let proc = NativeAbort;
    assert!(proc.no_return());
    assert_eq!(proc.num_args(), 0);
    assert_eq!(proc.name(), "abort");
    let mut state = RustSimState::new("amd64").unwrap();
    let result = proc.call(&mut state, &[]).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_stack_chk_fail() {
    let proc = NativeStackChkFail;
    assert!(proc.no_return());
    assert_eq!(proc.num_args(), 0);
    assert_eq!(proc.name(), "__stack_chk_fail");
    let mut state = RustSimState::new("amd64").unwrap();
    let result = proc.call(&mut state, &[]).unwrap();
    assert!(result.is_none());
}
