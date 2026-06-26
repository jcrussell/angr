use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcOutcome, ProcedureError};
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

// --- pthread_once (native sub-call, bead xxukz) ------------------------------

const ONCE_CONTROL: u64 = 0x2000;
const ONCE_FUNC: u64 = 0x0040_1000;

/// amd64 state with a writable data page (holding the once-guard at
/// `ONCE_CONTROL`) and a concrete stack, so `call_ex` passes the SP guard.
fn once_state(guard_byte: u8) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
        .memory_mut()
        .store_concrete(ONCE_CONTROL, RustBV::concrete(guard_byte as u128, 8))
        .unwrap();
    let sp = 0x7fff_0000u64;
    state.map_memory(sp - 0x1000, 0x2000, Permission::RWX);
    state.set_sp(RustBV::concrete(sp as u128, 64));
    state
}

fn once_args() -> Vec<RustBV> {
    vec![
        RustBV::concrete(ONCE_CONTROL as u128, 64),
        RustBV::concrete(ONCE_FUNC as u128, 64),
    ]
}

fn guard_byte(state: &RustSimState) -> u64 {
    state
        .memory_load(ONCE_CONTROL, 1)
        .unwrap()
        .as_u64()
        .unwrap()
}

#[test]
fn test_once_metadata() {
    assert_eq!(NativePthreadOnce.name(), "pthread_once");
    assert_eq!(NativePthreadOnce.num_args(), 2);
    assert!(!NativePthreadOnce.no_return());
}

#[test]
fn test_once_call_is_not_implemented() {
    // Return-only entry must defer to Python — pthread_once needs call_ex.
    let mut state = once_state(0x00);
    assert!(matches!(
        NativePthreadOnce.call(&mut state, &once_args()),
        Err(ProcedureError::NotImplemented)
    ));
}

#[test]
fn test_once_already_done_returns_zero_without_subcall() {
    // Guard bit 2 already set: return 0, no sub-call, guard untouched.
    let mut state = once_state(0x02);
    match NativePthreadOnce.call_ex(&mut state, &once_args()).unwrap() {
        ProcOutcome::Return(Some(bv)) => {
            assert_eq!(bv.as_u64(), Some(0));
            assert_eq!(bv.width(), 64);
        }
        _ => panic!("expected Return(Some(0)) for already-initialised once"),
    }
    assert_eq!(guard_byte(&state), 0x02);
    assert!(state.native_resume_stack().is_empty());
}

#[test]
fn test_once_first_call_sets_bit_and_subcalls() {
    let mut state = once_state(0x00);
    match NativePthreadOnce.call_ex(&mut state, &once_args()).unwrap() {
        ProcOutcome::CallAndResume {
            target,
            args,
            resume_tag,
        } => {
            assert_eq!(target, ONCE_FUNC);
            assert!(args.is_empty());
            assert_eq!(resume_tag, 0);
        }
        _ => panic!("expected CallAndResume on first pthread_once"),
    }
    // Done-bit set; sub-call setup (sentinel/frame) is the dispatcher's job, not
    // call_ex's, so the resume stack is still empty here.
    assert_eq!(guard_byte(&state), 0x02);
}

#[test]
fn test_once_first_call_preserves_other_bits() {
    // 0x05 -> set bit 2 -> 0x07 (other bits untouched).
    let mut state = once_state(0x05);
    assert!(matches!(
        NativePthreadOnce.call_ex(&mut state, &once_args()).unwrap(),
        ProcOutcome::CallAndResume { .. }
    ));
    assert_eq!(guard_byte(&state), 0x07);
}

#[test]
fn test_once_symbolic_sp_falls_back_without_mutation() {
    // Symbolic SP must abort BEFORE the guard write, so Python (the fallback)
    // re-runs pthread_once and still invokes func — no silently-skipped init.
    let mut state = once_state(0x00);
    let ctx = state.solver().borrow();
    let sym_sp = RustBV::symbolic(&ctx, "sp", 64);
    drop(ctx);
    state.set_sp(sym_sp);
    assert!(matches!(
        NativePthreadOnce.call_ex(&mut state, &once_args()),
        Err(ProcedureError::SymbolicArgument(_))
    ));
    assert_eq!(
        guard_byte(&state),
        0x00,
        "guard must be untouched on fallback"
    );
}

#[test]
fn test_once_symbolic_control_ptr_falls_back() {
    let mut state = once_state(0x00);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "control", 64);
    drop(ctx);
    let args = vec![sym, RustBV::concrete(ONCE_FUNC as u128, 64)];
    assert!(matches!(
        NativePthreadOnce.call_ex(&mut state, &args),
        Err(ProcedureError::SymbolicArgument(_))
    ));
}

#[test]
fn test_once_symbolic_guard_byte_falls_back() {
    // A symbolic once-guard in memory must defer to Python (matches the Python
    // proc raising on a symbolic control word).
    let mut state = once_state(0x00);
    let ctx = state.solver().borrow();
    let sym_byte = RustBV::symbolic(&ctx, "guard", 8);
    drop(ctx);
    state
        .memory_mut()
        .store_concrete(ONCE_CONTROL, sym_byte)
        .unwrap();
    assert!(matches!(
        NativePthreadOnce.call_ex(&mut state, &once_args()),
        Err(ProcedureError::SymbolicArgument(_))
    ));
}

#[test]
fn test_once_resume_returns_zero() {
    let mut state = once_state(0x02);
    match NativePthreadOnce.resume(&mut state, 0, &[]).unwrap() {
        ProcOutcome::Return(Some(bv)) => {
            assert_eq!(bv.as_u64(), Some(0));
            assert_eq!(bv.width(), 64);
        }
        _ => panic!("expected Return(Some(0)) from pthread_once resume"),
    }
}
