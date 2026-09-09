//! Deferred-fork materialization: turning an interpreter-deferred branch into
//! a real successor state.
//!
//! Split out of `exploration::helpers` (angr-9ke6b.76). The interpreter records
//! an unexplored branch instead of forking eagerly; [`materialize_deferred_forks`]
//! replays the prior guards ([`PriorGuards`]) onto a clone and installs the
//! inverted guard ([`build_unexplored_fork`]) when the exploration loop actually
//! needs the other side.
//!
//! **Panic policy (angr-qwyti.11):** carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// Install a deferred fork's branch guard on the continuing state, firing the
/// `state.inspect.constraints` BP around the add (angr-op0dn.14.4.1).
///
/// `is_true` selects the polarity: `assume_true(guard)` for the taken path,
/// `assume_false(guard)` for the fallthrough. Mirrors Python's
/// `state_plugins/solver.py::add`, which fires `constraints` BP_BEFORE with
/// `added_constraints` and BP_AFTER once the solver has them — here the
/// successor receiving the guard is the base state itself (Rust mutates the
/// stepped state in place and mints a *separate* state for the unexplored
/// side, whose inverted guard is installed inside `build_unexplored_fork`).
///
/// Bit 19 = `_INSPECT_EVENT_SPECS["constraints"]`; the no-BP case costs a
/// single relaxed atomic load. Mutation of `added_constraints` by the BP is
/// NOT honored on this path (the guard is already lowered to a `RustBV`);
/// the `RustSolverProxyPlugin.add` path does honor it.
#[inline]
fn add_fork_guard_constraint(
    callbacks: Option<&PythonCallbacks>,
    state: &RustSimState,
    guard: &RustBV,
    is_true: bool,
) {
    let cb =
        callbacks.filter(|c| c.inspect_event_enabled(crate::callbacks::InspectBit::Constraints));
    let state_id = state.state_id() as i64;
    if let Some(c) = cb
        && let Err(e) = c.call_inspect_constraints(state_id, "before", guard, is_true)
    {
        log::debug!("constraints inspect dispatch raised (state {state_id}, before): {e}");
    }
    if is_true {
        state.solver().borrow().assume_true(guard);
    } else {
        state.solver().borrow().assume_false(guard);
    }
    if let Some(c) = cb
        && let Err(e) = c.call_inspect_constraints(state_id, "after", guard, is_true)
    {
        log::debug!("constraints inspect dispatch raised (state {state_id}, after): {e}");
    }
}

/// Reconstruct a deferred fork's branch condition from its stored claripy AST
/// — the `condition_ast` fallback — when the condition is absent from
/// `stored_conditions`.
///
/// Returns `None` when `stored_condition` is already `Some` (nothing to
/// reconstruct), when the fork carries no `condition_ast`, or when the
/// claripy→RustBV conversion fails. The returned owned `RustBV` is evaluated
/// against `fork_base`'s solver context so it shares the base's z3
/// declarations.
///
/// This is the single source of truth for the four verbatim copies that used
/// to live inline in `_resume_after_simprocedure`, `_deadend_pending_callback`,
/// `_resume_after_symbolic_branch` (resume.rs) and the run-loop callback path
/// (run_loop.rs). angr-ph300.7 fixed a drop-bug by porting this arm into the
/// deadend copy verbatim; consolidating removes that copy-paste hazard (memory
/// invariant-deferred-fork-condition-fallback-arms).
pub(crate) fn reconstruct_deferred_fork_condition(
    stored_condition: Option<&RustBV>,
    fork: &DeferredFork,
    fork_base: &RustSimState,
) -> Option<RustBV> {
    if stored_condition.is_some() {
        return None;
    }
    let py_ast = fork.condition_ast.as_ref()?;
    Python::attach(|py| {
        let ast = py_ast.bind(py);
        let solver_ref = fork_base.solver();
        let ctx: &SymContext = &solver_ref.borrow();
        crate::claripy_bridge::claripy_to_rustbv(py, ast, ctx).ok()
    })
}

/// Build the unexplored-path fork for one deferred branch.
///
/// Every deferred-fork materialization site (`process_deferred_forks_into` in
/// `stepping_forks.rs`,
/// [`materialize_deferred_forks`], and the parallel mirror in
/// `core_outcome_handlers.rs`) constructs the opposite-path state with the same
/// byte-identical 3-way branch: prefer a pre-branch solver `snapshot` (and
/// re-assume the *opposite* constraint onto it) when one was captured,
/// otherwise `fork_false` / `fork_true` off `base` depending on which side the
/// main path took. The forked state's PC is then set to
/// `fork.unexplored_target`.
fn build_unexplored_fork(
    base: &RustSimState,
    fork: &DeferredFork,
    condition: &RustBV,
    snapshots: &mut FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    priors: &PriorGuards,
) -> RustSimState {
    let mut forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
        // Snapshot predates the branch constraint, so re-assume the opposite
        // side to keep the unexplored path's constraints consistent.
        let f = base.fork_from_snapshot(snapshot);
        // The snapshot is a clone of the *interpreter's* block solver, which
        // deliberately never assumes taken-path guards permanently (see the
        // NOTE in `interpreter/statements.rs`) — so it carries NONE of the
        // guards from earlier deferred branches in the same step, no matter
        // how well-guarded `base` is. Replay them here (angr-62ar5).
        priors.replay_onto(&f);
        if fork.path_taken {
            f.solver().borrow().assume_false(condition);
        } else {
            f.solver().borrow().assume_true(condition);
        }
        f
    } else {
        let f = if fork.path_taken {
            base.fork_false(condition)
        } else {
            base.fork_true(condition)
        };
        if !priors.base_carries {
            priors.replay_onto(&f);
        }
        f
    };
    forked.set_pc(fork.unexplored_target);
    forked
}

/// Mint the unexplored-side fork for one deferred branch and *then* install the
/// taken-path guard on `base` — the only correct order when the fork base and
/// the guard sink are the same state, which they are for all three
/// `base`-is-also-the-continuing-state materializers
/// (`stepping_forks.rs::process_deferred_forks_into` and
/// `core_outcome_handlers.rs`'s `materialize_deferred_forks_core` /
/// `process_deferred_forks_into_core`).
///
/// Guard-first is wrong and silently so (angr-z3obp): when the fork carries no
/// `BranchSnapshot`, [`build_unexplored_fork`] falls back to
/// `base.fork_false`/`fork_true`, and a `base` that has already assumed the
/// taken side yields a sibling that is UNSAT by construction — it is minted,
/// fails the SAT check and lands in `pruned` rather than being explored. The
/// snapshot arm is immune (it rebuilds from a pre-branch clone), which is why
/// the bug stayed latent: the interpreter's `GuardClass::Symbolic` arm records
/// a snapshot next to every `stored_conditions` entry.
///
/// The two halves are module-private so this ordering cannot be re-derived at a
/// fourth call site. Callers that guard a state *distinct* from the fork base
/// (`materialize_deferred_forks`'s `guard_sink`) are structurally immune and do
/// not need this helper.
pub(crate) fn fork_unexplored_and_guard_base(
    callbacks: Option<&PythonCallbacks>,
    base: &RustSimState,
    fork: &DeferredFork,
    condition: &RustBV,
    snapshots: &mut FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    priors: &PriorGuards,
) -> RustSimState {
    let forked = build_unexplored_fork(base, fork, condition, snapshots, priors);
    add_fork_guard_constraint(callbacks, base, condition, fork.path_taken);
    forked
}

/// The taken-path guards of every deferred fork already materialized in this
/// step, in dispatch order, plus whether the caller's fork base already carries
/// them.
///
/// A fork built for branch *i* must inherit the decisions of branches
/// `0..i` — they are on its path by construction. Callers that apply each
/// guard to the continuing state as they iterate (`stepping_forks.rs`,
/// `core_outcome_handlers.rs`) fork off a base that already has them and set
/// `base_carries: true`; [`materialize_deferred_forks`] forks off a fixed
/// guard-free base and sets `false`. The snapshot path replays them
/// unconditionally — see [`build_unexplored_fork`].
pub(crate) struct PriorGuards {
    guards: Vec<(RustBV, bool)>,
    base_carries: bool,
}

impl PriorGuards {
    /// Fresh accumulator. `base_carries` records whether the caller's fork base
    /// already holds the taken-path guards (see the struct doc); it gates the
    /// non-snapshot replay in [`build_unexplored_fork`].
    pub(crate) fn new(base_carries: bool) -> Self {
        Self {
            guards: Vec::new(),
            base_carries,
        }
    }

    /// Append the taken-path guard of a just-materialized fork so later forks in
    /// the same step inherit it. Callers must `record` in dispatch order.
    pub(crate) fn record(&mut self, cond: RustBV, path_taken: bool) {
        self.guards.push((cond, path_taken));
    }

    fn replay_onto(&self, state: &RustSimState) {
        for (cond, taken) in &self.guards {
            let solver = state.solver();
            let solver = solver.borrow();
            if *taken {
                solver.assume_true(cond);
            } else {
                solver.assume_false(cond);
            }
        }
    }
}

/// SAT / UNSAT split produced by [`materialize_deferred_forks`], in fork
/// dispatch order. Neither vector has had `set_root` applied — lineage needs
/// `&mut StateManager`, which the callers own; they register both vectors
/// (UNSAT forks were registered by every legacy copy too, before the SAT check).
pub(crate) struct MaterializedForks {
    pub(crate) sat: Vec<RustSimState>,
    pub(crate) unsat: Vec<RustSimState>,
}

/// Everything [`materialize_deferred_forks`] needs beyond the fork list itself.
pub(crate) struct MaterializeForkCtx<'a> {
    /// The guard-free base every unexplored side forks from. Callers pick this
    /// (pre-callback snapshot, bounce state, …); it must NOT carry the taken
    /// path's branch guard or the unexplored side inherits it.
    pub(crate) fork_base: &'a RustSimState,
    pub(crate) stored_conditions: &'a FxHashMap<u64, RustBV>,
    /// Pre-branch solver snapshots, consumed (`remove`d) per fork by
    /// [`build_unexplored_fork`].
    pub(crate) snapshots: &'a mut FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    /// When true, skip the per-fork SAT check and treat every fork as SAT.
    pub(crate) lazy_solves: bool,
    /// Continuing state that should receive the *taken*-path guard as each fork
    /// is materialized (`assume_true` when `path_taken`, `assume_false`
    /// otherwise). `None` for callers whose continuing state already carries the
    /// guard (`_resume_after_simprocedure`) or that mint both directions
    /// themselves (`_resume_after_symbolic_branch`).
    pub(crate) guard_sink: Option<&'a RustSimState>,
    /// `Some` iff profiling is enabled; receives the deferred-fork / fork-op /
    /// SAT timers and counts.
    pub(crate) stats: Option<&'a mut ExecutionStats>,
}

/// Materialize a step's deferred forks into SAT / UNSAT state lists.
///
/// Single source of truth for the condition-lookup →
/// [`reconstruct_deferred_fork_condition`] → conservative-fork arm →
/// [`build_unexplored_fork`] → SAT
/// check pipeline that used to be open-coded in three near-identical copies:
/// `step_one`'s find/avoid callback arm (`run_loop.rs`), and both
/// `_resume_after_simprocedure` / `_deadend_pending_callback` /
/// `_resume_after_symbolic_branch` loops (`resume.rs`). Only the parallel
/// mirror (`core_outcome_handlers.rs::materialize_deferred_forks_core`) was
/// unit-tested, so the serial copies were free to drift — angr-ph300.7 was
/// exactly that drift (a missing `condition_ast` arm silently dropping
/// AST-only forks).
///
/// The parallel mirror is deliberately NOT folded in here: it has no
/// `condition_ast` fallback, fires the `constraints` inspect BP via
/// [`add_fork_guard_constraint`], and accumulates into atomics rather than
/// `ExecutionStats`. That split is a knowingly-accepted duplication, not an
/// outstanding TODO: the bd memory `invariant-stepping-decomposition`
/// enumerates all four deferred-fork materializers (this one, the parallel
/// mirror, and the two `process_deferred_forks_into*` helpers) and records
/// that unifying them would either lose profiling or force it everywhere, so
/// it needs an explicit decision rather than a drive-by refactor.
pub(crate) fn materialize_deferred_forks(
    forks: Vec<DeferredFork>,
    ctx: MaterializeForkCtx<'_>,
) -> MaterializedForks {
    let MaterializeForkCtx {
        fork_base,
        stored_conditions,
        snapshots,
        lazy_solves,
        guard_sink,
        mut stats,
    } = ctx;
    let mut out = MaterializedForks {
        sat: Vec::new(),
        unsat: Vec::new(),
    };
    // NOTE: no early return on an empty fork list. The batch timer below is
    // charged unconditionally when profiling is on, matching the legacy
    // `step_one` site — `deferred_fork_time_ns` is the liveness signal the
    // profiling-gate regression test reads, and on binaries whose find/avoid
    // callbacks carry no deferred forks the empty batches are its only source.
    let total = forks.len() as u64;
    let batch_start = stats.is_some().then(std::time::Instant::now);
    // Taken-path guards of the forks already materialized, replayed onto each
    // later fork (`fork_base` is fixed and guard-free here — angr-62ar5).
    let mut prior_guards = PriorGuards::new(false);

    for fork in forks {
        let condition = stored_conditions.get(&fork.condition_id);
        // Reconstruct from the stored claripy AST when the condition is
        // absent from `stored_conditions`.
        let reconstructed = reconstruct_deferred_fork_condition(condition, &fork, fork_base);

        if let Some(cond) = condition.or(reconstructed.as_ref()) {
            if let Some(sink) = guard_sink {
                if fork.path_taken {
                    sink.solver().borrow().assume_true(cond);
                } else {
                    sink.solver().borrow().assume_false(cond);
                }
            }
            let fork_start = stats.is_some().then(std::time::Instant::now);
            let forked = build_unexplored_fork(fork_base, &fork, cond, snapshots, &prior_guards);
            prior_guards.record(cond.clone(), fork.path_taken);
            if let (Some(s), Some(start)) = (stats.as_deref_mut(), fork_start) {
                s.solver_fork_time_ns += crate::elapsed_ns(start);
                s.solver_fork_count += 1;
            }
            if reconstructed.is_some() {
                log::debug!(
                    "Reconstructed condition from condition_ast for fork at 0x{:x}",
                    fork.branch_addr
                );
            }
            let sat_start = stats.is_some().then(std::time::Instant::now);
            let sat = forked.survives_sat_prune(lazy_solves);
            if let (Some(s), Some(start)) = (stats.as_deref_mut(), sat_start) {
                s.solver_sat_time_ns += crate::elapsed_ns(start);
                s.solver_sat_count += 1;
            }
            if sat {
                out.sat.push(forked);
            } else {
                log::debug!(
                    "Forked state at 0x{:x} is UNSAT, adding to pruned",
                    fork.unexplored_target
                );
                out.unsat.push(forked);
            }
        } else {
            // No condition from either source — build a conservative
            // unconstrained fork so the unexplored branch is still routed
            // rather than dropped.
            log::warn!(
                "Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                 Creating conservative fork to explore the path.",
                fork.branch_addr,
                fork.condition_id
            );
            let mut forked = fork_base.fork();
            prior_guards.replay_onto(&forked);
            forked.set_pc(fork.unexplored_target);
            if forked.survives_sat_prune(lazy_solves) {
                out.sat.push(forked);
            } else {
                log::debug!(
                    "Unconstrained fork at 0x{:x} is UNSAT, adding to pruned",
                    fork.unexplored_target
                );
                out.unsat.push(forked);
            }
        }
    }

    if let (Some(s), Some(start)) = (stats, batch_start) {
        s.deferred_fork_time_ns += crate::elapsed_ns(start);
        s.deferred_fork_count += total;
    }
    out
}

test_submod!("fork_materialize_tests.rs" => tests);
