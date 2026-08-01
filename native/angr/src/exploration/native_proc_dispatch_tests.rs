// Tests for exploration/native_proc_dispatch.rs (split out of
// helpers_tests.rs alongside the source split, angr-9ke6b.76).

use super::*;
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

// ---------------------------------------------------------------------------
// dispatch_native_proc — the shared proc-dispatch decision consumed by BOTH
// `step_one` (run_loop.rs) and `handle_simprocedure_core`
// (core_outcome_handlers.rs). Only the parallel copy was covered before
// angr-ph300.73, so the serial copy was free to drift.
// ---------------------------------------------------------------------------

/// Stub proc whose `call_ex` replays a canned outcome, so the dispatcher's
/// classification + counter bookkeeping can be tested without a real
/// procedure. The outcome is described by plain data (not a prebuilt
/// `ProcOutcome`) because `NativeSimProcedure` is `Send + Sync` and `RustBV`
/// is neither.
#[derive(Clone, Copy)]
enum StubOutcome {
    Ret(Option<u64>),
    SubCall { target: u64, resume_tag: u32 },
    Fail(StubError),
}

#[derive(Clone, Copy)]
enum StubError {
    Symbolic,
    NotImplemented,
    MaxIterations,
    Unmapped,
}

struct StubProc(StubOutcome);

impl crate::procedures::NativeSimProcedure for StubProc {
    fn name(&self) -> &'static str {
        "stub"
    }
    fn num_args(&self) -> usize {
        0
    }
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        unreachable!("call_ex is overridden")
    }
    fn call_ex(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<ProcOutcome, ProcedureError> {
        match self.0 {
            StubOutcome::Ret(v) => Ok(ProcOutcome::Return(
                v.map(|v| RustBV::concrete(v as u128, 64)),
            )),
            StubOutcome::SubCall { target, resume_tag } => Ok(ProcOutcome::CallAndResume {
                target,
                args: vec![RustBV::concrete(3, 64)],
                resume_tag,
            }),
            StubOutcome::Fail(e) => Err(match e {
                StubError::Symbolic => ProcedureError::SymbolicArgument("n".into()),
                StubError::NotImplemented => ProcedureError::NotImplemented,
                StubError::MaxIterations => ProcedureError::MaxIterations(0),
                StubError::Unmapped => {
                    ProcedureError::Memory(crate::memory::MemoryError::Unmapped {
                        addr: 0x1234,
                        size: 4096,
                    })
                }
            }),
        }
    }
}

#[derive(Default)]
struct Counters {
    native_calls: u64,
    python_fallbacks: u64,
    call_counts: HashMap<String, u64>,
    symbolic: HashMap<String, u64>,
    not_implemented: HashMap<String, u64>,
    other: HashMap<String, u64>,
}

impl Counters {
    fn view(&mut self) -> NativeProcCounters<'_> {
        NativeProcCounters {
            native_calls: &mut self.native_calls,
            python_fallbacks: &mut self.python_fallbacks,
            call_counts: &mut self.call_counts,
            symbolic_fallbacks_by_name: &mut self.symbolic,
            not_implemented_fallbacks_by_name: &mut self.not_implemented,
            other_fallbacks_by_name: &mut self.other,
        }
    }
}

fn dispatch(
    outcome: StubOutcome,
    args: Result<Vec<RustBV>, crate::arch::ExtractionError>,
    mirror_segfault: bool,
    state: &mut RustSimState,
    counters: &mut Counters,
) -> NativeProcDisposition {
    dispatch_native_proc(
        &StubProc(outcome),
        state,
        "stub",
        false,
        args,
        mirror_segfault,
        &mut counters.view(),
    )
}

/// A failed argument extraction never runs the proc: it books an `other`
/// fallback (NOT a native call) and defers to Python.
#[test]
fn dispatch_native_proc_arg_extraction_failure_is_an_other_fallback() {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut counters = Counters::default();
    let d = dispatch(
        StubOutcome::Ret(None),
        Err(crate::arch::ExtractionError::SpSymbolic),
        false,
        &mut state,
        &mut counters,
    );
    assert!(matches!(d, NativeProcDisposition::Fallback));
    assert_eq!(counters.native_calls, 0);
    assert_eq!(counters.python_fallbacks, 1);
    assert_eq!(counters.other.get("stub"), Some(&1));
    assert!(counters.call_counts.is_empty());
}

/// `ProcedureError` variants land in the three distinct per-name buckets the
/// stats API reports (`symbolic + not_implemented + other` is the per-proc
/// total), and none of them count as a native call.
#[test]
fn dispatch_native_proc_buckets_errors_by_variant() {
    for (err, pick) in [
        (StubError::Symbolic, "symbolic" as &str),
        (StubError::NotImplemented, "not_implemented"),
        (StubError::MaxIterations, "other"),
    ] {
        let mut state = RustSimState::new("amd64").unwrap();
        let mut counters = Counters::default();
        let d = dispatch(
            StubOutcome::Fail(err),
            Ok(vec![]),
            false,
            &mut state,
            &mut counters,
        );
        assert!(matches!(d, NativeProcDisposition::Fallback));
        assert_eq!(counters.native_calls, 0);
        assert_eq!(counters.python_fallbacks, 1);
        let map = match pick {
            "symbolic" => &counters.symbolic,
            "not_implemented" => &counters.not_implemented,
            _ => &counters.other,
        };
        assert_eq!(map.get("stub"), Some(&1), "bucket {pick}");
    }
}

/// `mirror_segfault` is what separates the two callers: the parallel path
/// terminal-errors an unmapped-page fault natively (counting it as a native
/// call), while `step_one` passes `false` and still bounces to Python.
#[test]
fn dispatch_native_proc_segfault_mirroring_is_caller_gated() {
    let unmapped = StubOutcome::Fail(StubError::Unmapped);

    let mut state = RustSimState::new("amd64").unwrap();
    state.set_enforce_permissions(true);

    let mut mirrored = Counters::default();
    let d = dispatch(unmapped, Ok(vec![]), true, &mut state, &mut mirrored);
    match d {
        NativeProcDisposition::Segfault(msg) => assert_eq!(msg, "0x1000 (unmapped)"),
        _ => panic!("expected Segfault"),
    }
    assert_eq!(mirrored.native_calls, 1);
    assert_eq!(mirrored.python_fallbacks, 0);

    let mut bounced = Counters::default();
    let d = dispatch(unmapped, Ok(vec![]), false, &mut state, &mut bounced);
    assert!(matches!(d, NativeProcDisposition::Fallback));
    assert_eq!(bounced.native_calls, 0);
    assert_eq!(bounced.python_fallbacks, 1);
    assert_eq!(bounced.other.get("stub"), Some(&1));
}

/// A `Return` counts as a native call here; a `CallAndResume` deliberately does
/// NOT — the two callers count sub-calls at different points (core at
/// `call_ex`, `step_one` only once `setup_native_subcall` succeeds), so the
/// bump stays caller-side.
#[test]
fn dispatch_native_proc_counts_return_but_leaves_subcall_to_the_caller() {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut counters = Counters::default();
    let d = dispatch(
        StubOutcome::Ret(Some(7)),
        Ok(vec![]),
        false,
        &mut state,
        &mut counters,
    );
    match d {
        NativeProcDisposition::Returned { no_return, ret_val } => {
            assert!(!no_return);
            assert_eq!(ret_val.and_then(|v| v.as_u64()), Some(7));
        }
        _ => panic!("expected Returned"),
    }
    assert_eq!(counters.native_calls, 1);
    assert_eq!(counters.call_counts.get("stub"), Some(&1));

    let mut counters = Counters::default();
    let saved = vec![RustBV::concrete(1, 64), RustBV::concrete(2, 64)];
    let d = dispatch(
        StubOutcome::SubCall {
            target: 0x400100,
            resume_tag: 9,
        },
        Ok(saved.clone()),
        false,
        &mut state,
        &mut counters,
    );
    match d {
        NativeProcDisposition::SubCall {
            proc_name,
            saved_args,
            target,
            sub_args,
            resume_tag,
        } => {
            assert_eq!(proc_name, "stub");
            assert_eq!(saved_args.len(), saved.len());
            assert_eq!(target, 0x400100);
            assert_eq!(sub_args.len(), 1);
            assert_eq!(resume_tag, 9);
        }
        _ => panic!("expected SubCall"),
    }
    assert_eq!(counters.native_calls, 0, "caller owns the sub-call bump");
    assert!(counters.call_counts.is_empty());
}

// ---------------------------------------------------------------------------
// effective_pc — the address a state reports to Python when `self.pc` is
// stale (angr-4rq7).
// ---------------------------------------------------------------------------

// amd64 RIP guest-state offset — writing here bypasses `set_ip`, which would
// otherwise sync `self.pc` and hide the stale-pc case we need to reproduce.
const RIP: u32 = 184;

/// Build the angr-4rq7 shape: IP register holds the real branch target while
/// `self.pc` is still 0 (register-file-replace path that never went through
/// `set_pc`). Returns the state id.
fn push_active_stale_pc(mgr: &mut RustExplorationManager, target: u64) -> u64 {
    let mut s = RustSimState::new("amd64").expect("state");
    s.set_register_by_offset(RIP, RustBV::concrete(target as u128, 64));
    assert_eq!(s.pc(), 0, "fixture must leave self.pc stale at 0");
    let sid = s.state_id();
    mgr.sm.push(STASH_ACTIVE, s);
    sid
}

/// All three "what address is this state at" accessors must agree for a
/// stale-pc forked state: get_state_pc (by stash index), get_state_pc_by_id,
/// and the addr field of get_state_predicate_info (angr-ph300.23).
#[test]
fn stale_pc_state_reports_ip_from_every_accessor() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let sid = push_active_stale_pc(&mut mgr, 0x40_1234);

    assert_eq!(mgr.get_state_pc_by_id(sid), Some(0x40_1234));
    assert_eq!(
        mgr.get_state_pc(STASH_ACTIVE, 0),
        Some(0x40_1234),
        "index accessor must not report the stale 0"
    );
    let info = mgr.get_state_predicate_info(STASH_ACTIVE);
    assert_eq!(
        info,
        vec![(sid, 0x40_1234, 0)],
        "predicate cache would key on (sid, 0) without the IP fallback"
    );
}

/// A nonzero `self.pc` stays authoritative — the fallback must not override it
/// even when the IP register disagrees (gate-off path keeps them in sync, and a
/// genuinely-zero IP still reports 0).
#[test]
fn nonzero_pc_is_authoritative_and_zero_ip_stays_zero() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    // A plain active state whose `self.pc` is genuinely set (no callstack
    // needed — `effective_pc` only reads pc and the IP register).
    let live = {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_pc(0x2000);
        let sid = s.state_id();
        mgr.sm.push(STASH_ACTIVE, s);
        sid
    };
    // Desync the IP register behind the back of set_pc.
    mgr.sm
        .get_mut(STASH_ACTIVE)
        .unwrap()
        .get_mut(0)
        .unwrap()
        .set_register_by_offset(RIP, RustBV::concrete(0x9999, 64));
    assert_eq!(mgr.get_state_pc_by_id(live), Some(0x2000));
    assert_eq!(mgr.get_state_pc(STASH_ACTIVE, 0), Some(0x2000));

    // pc == 0 and IP == 0 -> 0, not None.
    let zero = push_active_stale_pc(&mut mgr, 0);
    assert_eq!(mgr.get_state_pc_by_id(zero), Some(0));
    assert_eq!(mgr.get_state_pc(STASH_ACTIVE, 1), Some(0));
}
