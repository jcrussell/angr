//! angr-6cp06.24: `handle_simprocedure_core`'s *success* path with a non-empty
//! deferred-fork list — the fork base it hands
//! `process_deferred_forks_into_core`.
//!
//! Every other driver of this arm (`hooks`, `native_return`) passes
//! `deferred_forks: Vec::new()`, so the whole materialization branch was
//! unpinned: neither a fix nor a regression to it would show up in any test.
//! These are characterization tests — they pin what the code does today, which
//! is what makes the open audit item in bd memory
//! `avoid-deferred-fork-base-mismatch` (does the unexplored sibling observe the
//! mutations a *successful* native call applied to the main successor?)
//! decidable rather than invisible.
//!
//! The pinned answer, for the shape the interpreter actually emits — a stored
//! condition *plus* a pre-branch `BranchSnapshot` — is **no**: the sibling's
//! registers and solver come from the snapshot, which predates both the branch
//! guard and the call, so the return-register write and the SP bump stay on the
//! main successor only. Changing that deliberately means updating these
//! assertions in the same commit.

use super::*;

const SP_SEED: u64 = 0x7fff_0000;
const UNEXPLORED: u64 = 0x40_7000;

/// Dispatch the hook on amd64 with one guarded deferred fork on the
/// `path_taken` side, and hand back the pieces the assertions need.
fn dispatch_with_fork(path_taken: bool, with_snapshot: bool) -> SeededDispatch {
    dispatch_hook_seeded(
        "amd64",
        SpSeed::Concrete(SP_SEED),
        Some(ForkSeed {
            path_taken,
            unexplored_target: UNEXPLORED,
            with_snapshot,
        }),
        |_| {},
    )
}

/// amd64 return register, for reading back what the proc wrote.
fn amd64_return_register() -> u32 {
    RustExplorationManager::new("amd64", None)
        .unwrap()
        .environment
        .calling_convention
        .return_register()
}

/// The successful native return still materializes the block's deferred fork:
/// main successor first, sibling second, tagged as a fork rooted at the main
/// state and parked at the unexplored target.
#[test]
fn simproc_success_materializes_deferred_fork_alongside_main() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let d = dispatch_with_fork(true, true);
        assert_eq!(d.outcome.counters.native_calls, 1, "native proc ran");
        assert!(d.outcome.pruned.is_empty(), "the snapshot fork is SAT");
        assert!(d.outcome.terminal_pushes.is_empty(), "the proc returns");
        assert_eq!(d.outcome.fork_ids.len(), 1, "one fork dispatched");

        let CoreReturn::Continue(succ) = d.outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ.len(), 2, "main + materialized fork");
        assert_eq!(succ[0].0.state_id(), d.sid, "main comes first");
        assert!(!succ[0].1.is_fork);
        assert_eq!(succ[0].0.pc(), HOOK_RET, "main landed at the return address");

        assert!(succ[1].1.is_fork);
        assert_eq!(succ[1].1.root_hint, Some(d.sid));
        assert_ne!(succ[1].0.state_id(), d.sid);
        assert_eq!(d.outcome.fork_ids[0], succ[1].0.state_id());
        assert_eq!(
            succ[1].0.pc(),
            UNEXPLORED,
            "sibling parked at the branch's other side"
        );
    });
}

/// The pinned fork-base semantics: the sibling is rebuilt from the pre-branch
/// snapshot, so the native call's mutations — the return-register write and the
/// popped-return-address SP bump — stay on the main successor.
///
/// This is the assertion bd memory `avoid-deferred-fork-base-mismatch` needs:
/// for the snapshot shape the interpreter emits, the SimProc-native success
/// path does *not* leak the call into its siblings, even though it passes the
/// post-call state as `base`.
#[test]
fn simproc_success_fork_does_not_inherit_the_calls_register_mutations() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let d = dispatch_with_fork(true, true);
        let ret_reg = amd64_return_register();
        let CoreReturn::Continue(succ) = d.outcome.ret else {
            panic!("expected Continue");
        };
        let (main, fork) = (&succ[0].0, &succ[1].0);

        assert_eq!(
            main.get_register_by_offset(ret_reg, 8).as_u64(),
            Some(PROC_RET_VAL as u64),
            "main successor holds the proc's return value"
        );
        assert_ne!(
            fork.get_register_by_offset(ret_reg, 8).as_u64(),
            Some(PROC_RET_VAL as u64),
            "the sibling never ran the call, so it must not hold its return value"
        );
        assert_eq!(
            main.get_sp().as_u64(),
            Some(SP_SEED + 8),
            "amd64 pops the return address"
        );
        assert_eq!(
            fork.get_sp().as_u64(),
            Some(SP_SEED),
            "the sibling keeps the pre-call SP"
        );
    });
}

/// Both states get the branch guard, on opposite polarities: the main successor
/// is assumed onto the taken side in place, the sibling is replayed onto the
/// other one from the guard-free snapshot solver.
///
/// Only observable where a model exists — the no-z3 mock solver's `eval`
/// returns `None` for every symbolic BV, so that build asserts exactly that and
/// keeps the structural coverage above (angr-c7xno.100).
#[test]
fn simproc_success_fork_guard_polarities_are_opposite() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for path_taken in [true, false] {
            let d = dispatch_with_fork(path_taken, true);
            let guard = d.condition.clone().expect("a seeded fork mints a guard");
            let CoreReturn::Continue(succ) = d.outcome.ret else {
                panic!("expected Continue");
            };
            let want = |v: bool| cfg!(feature = "vex-engine-z3").then_some(u128::from(v));
            assert_eq!(
                succ[0].0.eval(&guard),
                want(path_taken),
                "main successor carries the taken-path guard (path_taken={path_taken})"
            );
            assert_eq!(
                succ[1].0.eval(&guard),
                want(!path_taken),
                "sibling carries the opposite guard (path_taken={path_taken})"
            );
        }
    });
}

/// The snapshot-less shape is where the post-call `base` actually bites: with
/// no `BranchSnapshot` to rebuild from, `build_unexplored_fork` forks the base
/// — which `add_fork_guard_constraint` has *already* assumed onto the taken
/// side — and then assumes the opposite guard on top. The sibling is UNSAT by
/// construction and lands in `pruned` instead of being explored.
///
/// Not reachable from the interpreter (its `GuardClass::Symbolic` arm inserts a
/// snapshot next to every `stored_conditions` entry), so this pins a latent
/// shape rather than a live path — but it is the concrete failure mode any
/// future unification of the two fork bases has to avoid.
#[test]
fn simproc_success_fork_without_snapshot_is_pruned_unsat() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let d = dispatch_with_fork(true, false);
        assert_eq!(d.outcome.fork_ids.len(), 1, "the fork is still minted");
        let CoreReturn::Continue(succ) = d.outcome.ret else {
            panic!("expected Continue");
        };
        if cfg!(feature = "vex-engine-z3") {
            assert_eq!(succ.len(), 1, "only the main successor survives");
            assert_eq!(d.outcome.pruned.len(), 1, "the sibling is pruned UNSAT");
            assert_eq!(d.outcome.pruned[0].state_id(), d.outcome.fork_ids[0]);
        } else {
            // The mock solver asserts nothing, so nothing can be proven UNSAT.
            assert_eq!(succ.len(), 2);
            assert!(d.outcome.pruned.is_empty());
        }
    });
}
