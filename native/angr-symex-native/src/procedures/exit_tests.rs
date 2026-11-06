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
fn test_exit_returns_its_status() {
    let mut state = RustSimState::new("amd64").unwrap();
    let status = NativeExit
        .call(&mut state, &[RustBV::concrete(3, 64)])
        .unwrap()
        .expect("exit status");
    assert_eq!(status.as_u64(), Some(3));
}

#[test]
fn test_underscore_exit() {
    let proc = NativeUnderscoreExit;
    assert!(proc.no_return());
    assert_eq!(proc.name(), "_exit");
    let mut state = RustSimState::new("amd64").unwrap();
    let status = proc.call(&mut state, &[RustBV::concrete(1, 64)]).unwrap();
    assert_eq!(status.and_then(|s| s.as_u64()), Some(1));
}

#[test]
fn test_abort() {
    let proc = NativeAbort;
    assert!(proc.no_return());
    assert_eq!(proc.num_args(), 0);
    assert_eq!(proc.name(), "abort");
    // Python's abort.py does `self.exit(1)` at the arch's width.
    let mut state = RustSimState::new("x86").unwrap();
    let status = proc.call(&mut state, &[]).unwrap().expect("exit status");
    assert_eq!((status.as_u64(), status.width()), (Some(1), 32));
}

#[test]
fn test_stack_chk_fail() {
    let proc = NativeStackChkFail;
    assert!(proc.no_return());
    assert_eq!(proc.num_args(), 0);
    assert_eq!(proc.name(), "__stack_chk_fail");
    // Python's abort.py does `self.exit(1)` at the arch's width.
    let mut state = RustSimState::new("x86").unwrap();
    let status = proc.call(&mut state, &[]).unwrap().expect("exit status");
    assert_eq!((status.as_u64(), status.width()), (Some(1), 32));
}
