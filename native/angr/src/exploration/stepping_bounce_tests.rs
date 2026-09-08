//! Regression coverage for [`RustExplorationManager::dispatch_bounce`]'s
//! pre-callback snapshot contract (angr-6cp06.19) and its `UnmodeledCall`
//! arm's dropped-fork accounting (angr-6cp06.20).

use super::*;
use crate::exploration::core_outcome::{BounceKind, PendingBounce};
use crate::exploration::test_support::mgr_and_state;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

fn deferred_fork() -> crate::callbacks::DeferredFork {
    crate::callbacks::DeferredFork {
        branch_addr: 0x40_0500,
        path_taken: true,
        unexplored_target: 0x40_2000,
        condition_id: 11,
        condition_ast: None,
    }
}

fn bounce_with(kind: BounceKind, state: RustSimState, forks: Vec<DeferredFork>) -> PendingBounce {
    PendingBounce {
        kind,
        state,
        deferred_forks: forks,
        stored_conditions: FxHashMap::default(),
        fork_snapshots: FxHashMap::default(),
    }
}

fn snapshot_of(result: Result<Vec<RustSimState>, StepError>) -> Option<RustSimState> {
    match result {
        Err(StepError::NeedCallback(pending)) => pending.pre_callback_snapshot,
        _ => panic!("dispatch_bounce must bounce to Python"),
    }
}

/// `bounce()` forwards the step's deferred forks into the `PendingBounce`
/// regardless of `BounceKind`, so a block that deferred a fork and *then* hit
/// an unsupported VEX op reaches the `PythonVEXFallback` arm with a non-empty
/// `deferred_forks`. It resumes through `_resume_after_simprocedure`, which
/// falls back to `state.fork()` *after* the fallback block's writes and
/// constraints have been applied — so without a snapshot here the unexplored
/// side is based on the post-block state (same shape as bd memory
/// `avoid-deferred-fork-base-mismatch`).
#[test]
fn vex_fallback_bounce_snapshots_state_when_forks_are_deferred() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_1000);
        let callbacks = PythonCallbacks::new();
        let bounce = bounce_with(
            BounceKind::PythonVEXFallback {
                addr: 0x40_1000,
                reason: "unsupported op".to_string(),
            },
            state,
            vec![deferred_fork()],
        );

        let snapshot = snapshot_of(mgr.dispatch_bounce(&callbacks, bounce));

        let snapshot = snapshot.expect("deferred forks need a pre-callback fork base");
        assert_eq!(
            snapshot.pc(),
            0x40_1000,
            "the snapshot is the state as it stood before the Python round-trip"
        );
    });
}

/// The other half of the contract: `state.fork()` clones the Z3 solver, so an
/// ordinary fallback with nothing deferred must still pay nothing.
#[test]
fn vex_fallback_bounce_skips_the_snapshot_without_deferred_forks() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_1000);
        let callbacks = PythonCallbacks::new();
        let bounce = bounce_with(
            BounceKind::PythonVEXFallback {
                addr: 0x40_1000,
                reason: "unsupported op".to_string(),
            },
            state,
            Vec::new(),
        );

        assert!(
            snapshot_of(mgr.dispatch_bounce(&callbacks, bounce)).is_none(),
            "no deferred forks means no solver clone"
        );
    });
}

/// The Hook arm shares the snapshot decision through
/// `helpers::pre_callback_snapshot_for`; pin it so the DRY'd call site keeps
/// the behaviour its inline copy had.
#[test]
fn hook_bounce_snapshots_state_when_forks_are_deferred() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_1000);
        let callbacks = PythonCallbacks::new();
        let bounce = bounce_with(
            BounceKind::Hook { addr: 0x40_5000 },
            state,
            vec![deferred_fork()],
        );

        let snapshot = snapshot_of(mgr.dispatch_bounce(&callbacks, bounce));

        assert_eq!(
            snapshot.expect("deferred forks need a fork base").pc(),
            0x40_5000,
            "the Hook arm sets the hook pc before snapshotting"
        );
    });
}

/// A `resolve_function` callback that raises turns the whole block into an
/// error, and `StepError::Error` has no room for the forks the block deferred
/// — so they are lost. That loss must at least be *counted*: the sibling
/// `Ok(None)` / no-callback arms materialize the same forks through
/// `unmodeled_call_generic_skip`, and losing them without a trace makes a
/// truncated exploration indistinguishable from an exhausted one
/// (angr-6cp06.20).
#[test]
fn unmodeled_call_resolve_error_counts_the_forks_it_drops() {
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"def boom(addr, symbol_name):
    raise ValueError('no resolver')
",
            Some(&globals),
            None,
        )
        .expect("define raising resolve_function");
        let mut callbacks = PythonCallbacks::new();
        callbacks.set_resolve_function(
            globals
                .get_item("boom")
                .unwrap()
                .expect("boom defined")
                .unbind(),
        );

        let (mut mgr, state, _id) = mgr_and_state(0x40_1000);
        assert_eq!(mgr.deferred_forks_dropped(), 0);
        let bounce = bounce_with(
            BounceKind::UnmodeledCall {
                addr: 0x40_7000,
                return_addr: 0x40_1010,
                symbol_name: None,
            },
            state,
            vec![deferred_fork(), deferred_fork()],
        );

        match mgr.dispatch_bounce(&callbacks, bounce) {
            Err(StepError::Error(_, message)) => {
                assert!(
                    message.contains("resolve_function error"),
                    "unexpected error message: {message}"
                );
            }
            _ => panic!("a raising resolve_function must error the state"),
        }

        assert_eq!(
            mgr.deferred_forks_dropped(),
            2,
            "both deferred forks died with the errored state and must be counted"
        );
    });
}
