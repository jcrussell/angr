use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_mutex_lock_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativePthreadMutexLock
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0));
    assert_eq!(result.width(), 64);
}

#[test]
fn test_mutex_unlock_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativePthreadMutexUnlock
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0));
}

#[test]
fn test_symbolic_mutex_ptr_no_fallback() {
    // A symbolic mutex pointer must NOT trigger a Python fallback — the
    // pointer is ignored, so the proc still succeeds with 0.
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "mutex_ptr", 64);
    drop(ctx);
    let result = NativePthreadMutexLock.call(&mut state, &[sym]).unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_metadata() {
    assert_eq!(NativePthreadMutexLock.name(), "pthread_mutex_lock");
    assert_eq!(NativePthreadMutexLock.num_args(), 1);
    assert!(!NativePthreadMutexLock.no_return());
    assert_eq!(NativePthreadMutexUnlock.name(), "pthread_mutex_unlock");
    assert_eq!(NativePthreadMutexUnlock.num_args(), 1);
}
