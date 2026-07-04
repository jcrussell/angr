//! Tests for the native sub-call (CallAndResume) dispatcher core (S2, bead
//! angr-5gf0s). Exercises the two new dispatcher methods directly —
//! `setup_native_subcall` and `handle_native_resume` — plus the `ProcOutcome`
//! sibling-trait defaults (Option B).
//!
//! The guest routine that runs *between* setup and resume is generic VEX
//! interpreter machinery already covered by other tests; here we faithfully
//! stand in for its single observable effect (an amd64 `ret` that pops the
//! sentinel return slot: `sp += 8`, `pc = sentinel`) so the test stays focused
//! on the new dispatcher logic and free of a full lift/execute harness.

use super::core_outcome::NativeSubcall;
use super::*;
use crate::memory::Permission;
use crate::procedures::{
    NATIVE_RESUME_SENTINEL_NAME, NativeSimProcedure, ProcOutcome, ProcedureError,
    native_resume_sentinel,
};
use crate::state::RustSimState;
use std::sync::Arc;

/// A native proc with no sub-call: relies on the default `call_ex`/`resume`.
struct PlainProc;
impl NativeSimProcedure for PlainProc {
    fn name(&self) -> &'static str {
        "plain_test"
    }
    fn num_args(&self) -> usize {
        0
    }
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(Some(RustBV::concrete(5, 64)))
    }
}

/// A native proc that sub-calls a guest routine and resumes, mirroring the
/// shape of `pthread_once` (call -> CallAndResume -> retsite returns).
struct SubcallTestProc;
impl SubcallTestProc {
    const GUEST_TARGET: u64 = 0x0040_1000;
    const RESUME_TAG: u32 = 7;
}
impl NativeSimProcedure for SubcallTestProc {
    fn name(&self) -> &'static str {
        "subcall_test"
    }
    fn num_args(&self) -> usize {
        1
    }
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Should never be hit: call_ex is overridden to sub-call.
        Ok(Some(RustBV::concrete(0xdead, 64)))
    }
    fn call_ex(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<ProcOutcome, ProcedureError> {
        Ok(ProcOutcome::CallAndResume {
            target: Self::GUEST_TARGET,
            args: vec![],
            resume_tag: Self::RESUME_TAG,
        })
    }
    fn resume(
        &self,
        _state: &mut RustSimState,
        resume_tag: u32,
        saved_args: &[RustBV],
    ) -> Result<ProcOutcome, ProcedureError> {
        // The continuation re-receives the original proc args, like Python's
        // retsite(self, ...). Return saved_arg + 1 to prove the round-trip.
        assert_eq!(resume_tag, Self::RESUME_TAG);
        assert_eq!(saved_args.len(), 1);
        let v = saved_args[0].as_u64().expect("saved arg concrete");
        Ok(ProcOutcome::Return(Some(RustBV::concrete(
            (v + 1) as u128,
            64,
        ))))
    }
}

/// amd64 state with a 2-page stack mapped around `sp` and `[sp]` seeded with
/// the original caller return address.
fn state_with_stack(sp: u64, caller_ret: u64) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(sp - 0x1000, 0x2000, Permission::RWX);
    state
        .memory_mut()
        .store_concrete(sp, RustBV::concrete(caller_ret as u128, 64))
        .unwrap();
    state.set_sp(RustBV::concrete(sp as u128, 64));
    state
}

#[test]
fn default_call_ex_wraps_call_and_resume_is_notimplemented() {
    // Option B: a return-only proc inherits call_ex (wraps call -> Return) and
    // resume (NotImplemented) with zero changes.
    let p = PlainProc;
    let mut s = RustSimState::new("amd64").unwrap();
    match p.call_ex(&mut s, &[]).unwrap() {
        ProcOutcome::Return(Some(bv)) => assert_eq!(bv.as_u64(), Some(5)),
        other => panic!(
            "expected Return(Some(5)), got {:?}",
            matches!(other, ProcOutcome::Return(_))
        ),
    }
    assert!(matches!(
        p.resume(&mut s, 0, &[]),
        Err(ProcedureError::NotImplemented)
    ));
}

#[test]
fn native_subcall_setup_and_resume_roundtrip() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        Arc::make_mut(&mut mgr.native_procedures).register(Arc::new(SubcallTestProc));

        let sp = 0x7fff_0000u64;
        let caller_ret = 0x0040_0123u64;
        let mut state = state_with_stack(sp, caller_ret);

        let saved_args = vec![RustBV::concrete(41, 64)];
        mgr.setup_native_subcall(
            &mut state,
            NativeSubcall {
                proc_name: "subcall_test".into(),
                saved_args,
                caller_return_addr: caller_ret,
                target: SubcallTestProc::GUEST_TARGET,
                sub_args: vec![],
                resume_tag: SubcallTestProc::RESUME_TAG,
            },
        )
        .unwrap();

        // Frame pushed with every field, including the new caller_return_addr.
        assert_eq!(state.native_resume_stack().len(), 1);
        let f = &state.native_resume_stack()[0];
        assert_eq!(f.proc_name, "subcall_test");
        assert_eq!(f.resume_tag, SubcallTestProc::RESUME_TAG);
        assert_eq!(f.caller_return_addr, caller_ret);
        assert_eq!(f.saved_args[0].as_u64(), Some(41));

        // PC jumped into the guest routine.
        assert_eq!(state.pc(), SubcallTestProc::GUEST_TARGET);

        // [sp] overwritten with the sentinel so the guest `ret` lands on it.
        let sentinel = native_resume_sentinel(8);
        assert_eq!(state.memory_load(sp, 8).unwrap().as_u64(), Some(sentinel));

        // --- stand in for the guest routine running and `ret`-ing to the
        // sentinel: amd64 ret pops [sp]=sentinel (sp += 8) and jumps there. ---
        state.set_sp(RustBV::concrete((sp + 8) as u128, 64));
        state.set_pc(sentinel);

        let succ = match mgr.handle_native_resume(
            state,
            vec![],
            FxHashMap::default(),
            FxHashMap::default(),
        ) {
            Ok(s) => s,
            Err(_) => panic!("handle_native_resume returned a StepError"),
        };
        assert_eq!(succ.len(), 1);
        let out = &succ[0];

        // Resumed at the original caller; resume stack drained.
        assert_eq!(out.pc(), caller_ret);
        assert!(out.native_resume_stack().is_empty());
        // SP balanced: the sub-call net-popped one slot, exactly like a plain
        // return-only proc would (entry sp -> sp + 8).
        assert_eq!(out.get_sp().as_u64(), Some(sp + 8));
        // Return register holds saved_arg + 1 = 42 (proves resume + saved_args).
        // NB: get_register_by_offset's `size` is in *bytes* (8 = 64-bit reg).
        let ret_reg = mgr.environment.calling_convention.return_register();
        assert_eq!(out.get_register_by_offset(ret_reg, 8).as_u64(), Some(42));
    });
}

#[test]
fn setup_native_subcall_rejects_too_many_args() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        let sp = 0x7fff_0000u64;
        let mut state = state_with_stack(sp, 0x400123);
        // amd64 SysV exposes 6 integer arg registers; 7 guest args has no
        // register slot, so setup must decline (-> Python fallback) without
        // mutating the state.
        let too_many: Vec<RustBV> = (0..7).map(|i| RustBV::concrete(i, 64)).collect();
        let err = mgr
            .setup_native_subcall(
                &mut state,
                NativeSubcall {
                    proc_name: "subcall_test".into(),
                    saved_args: vec![],
                    caller_return_addr: 0x400123,
                    target: 0x401000,
                    sub_args: too_many,
                    resume_tag: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            SubcallSetupError::TooManyArgs {
                requested: 7,
                available: 6
            }
        ));
        // State untouched: no frame pushed, sp unchanged, [sp] still the caller
        // return address (sentinel never written).
        assert!(state.native_resume_stack().is_empty());
        assert_eq!(state.get_sp().as_u64(), Some(sp));
        assert_eq!(state.memory_load(sp, 8).unwrap().as_u64(), Some(0x400123));
    });
}

#[test]
fn path_a_captures_caller_return_addr_via_get_return_addr() {
    // The run-loop inline fast path (Path A, run_loop.rs) sets up a CallAndResume
    // by capturing the caller return address with `get_return_addr` (reads [sp])
    // and feeding it to `setup_native_subcall`. This locks that composition: the
    // captured address must equal [sp] at fresh entry and must be where the proc
    // resumes after the guest returns. (The full run-loop dispatch is exercised
    // by the Python e2e; the run loop needs a Python FFI context to drive.)
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        Arc::make_mut(&mut mgr.native_procedures).register(Arc::new(SubcallTestProc));

        let sp = 0x7fff_0000u64;
        let caller_ret = 0x0040_0123u64;
        let mut state = state_with_stack(sp, caller_ret);

        // Path A reads the caller return address from [sp] before setup
        // overwrites that slot with the sentinel.
        let captured = mgr
            .get_return_addr(&state)
            .expect("get_return_addr reads [sp]");
        assert_eq!(captured, caller_ret);

        // Drive the same sequence Path A's CallAndResume arm performs.
        mgr.setup_native_subcall(
            &mut state,
            NativeSubcall {
                proc_name: "subcall_test".into(),
                saved_args: vec![RustBV::concrete(41, 64)],
                caller_return_addr: captured,
                target: SubcallTestProc::GUEST_TARGET,
                sub_args: vec![],
                resume_tag: SubcallTestProc::RESUME_TAG,
            },
        )
        .unwrap();
        assert_eq!(state.pc(), SubcallTestProc::GUEST_TARGET);

        // Guest runs and `ret`s to the sentinel (amd64: sp += 8, pc = sentinel).
        state.set_sp(RustBV::concrete((sp + 8) as u128, 64));
        state.set_pc(native_resume_sentinel(8));

        let succ = match mgr.handle_native_resume(
            state,
            vec![],
            FxHashMap::default(),
            FxHashMap::default(),
        ) {
            Ok(s) => s,
            Err(_) => panic!("handle_native_resume returned a StepError"),
        };
        assert_eq!(succ.len(), 1);
        // Resumes exactly at the captured caller address — proving Path A's
        // get_return_addr capture targets the right slot.
        assert_eq!(succ[0].pc(), captured);
    });
}

#[test]
fn resume_sentinel_name_and_address_are_stable() {
    // The sentinel name is the reserved constant, and the address is the
    // top-of-space aligned slot per pointer width.
    assert_eq!(NATIVE_RESUME_SENTINEL_NAME, "__native_resume__");
    assert_eq!(native_resume_sentinel(8), 0xFFFF_FFFF_FFFF_FFF0);
    assert_eq!(native_resume_sentinel(4), 0xFFFF_FFF0);
}
