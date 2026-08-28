// Tests for exploration/constraints.rs (angr-c7xno.43).
//
// The three functions here had only transitive coverage before: the
// `apply_*_deterministic` seams via `manager_methods_tests.rs`'s
// `set_deterministic_reaches_parked_pending_callback_states`, and
// `import_python_constraints` via whatever `step_core_tests.rs` /
// `mod_tests.rs` happened to route through `_add_constraints_to_state`. Its
// own doc comment names it the single seam where a constraint-import
// correctness fix lands (angr-9ke6b.71), so both of its paths get pinned
// directly here.
//
// Tests that need claripy skip silently when it is not importable, matching
// the convention in `constraint_sync_tests.rs` / `claripy_bridge_tests.rs` —
// the cargo-test binary has no guaranteed venv.

use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::state::RustSimState;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyListMethods};

// ---------------------------------------------------------------------------
// apply_state_deterministic / apply_fork_snapshots_deterministic
// ---------------------------------------------------------------------------

/// The single seam for the `deterministic` flag round-trips in both
/// directions. Asserted as a flip sequence rather than a single `true` so a
/// one-way latch (or a fresh `SymContext` that happens to default the same
/// way) cannot pass.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn apply_state_deterministic_round_trips_both_directions() {
    let state = RustSimState::new("amd64").expect("state");
    assert!(
        !state_is_deterministic(&state),
        "a fresh state starts on the default witness-selection mode"
    );
    for v in [true, false, true] {
        apply_state_deterministic(&state, v);
        assert_eq!(
            state_is_deterministic(&state),
            v,
            "apply_state_deterministic({v}) must be observable via state_is_deterministic"
        );
    }
}

/// Build a `PendingCallback` carrying all three solver contexts
/// `set_deterministic` has to reach: the parked state, the pre-callback
/// snapshot (the deferred forks' `fork_base`), and one pre-branch fork
/// snapshot. The first two are `RustSimState`s that `all_live_states` yields;
/// only the third needs `apply_fork_snapshots_deterministic`.
#[cfg(feature = "vex-engine-z3")]
fn pending_with_all_contexts() -> PendingCallback {
    use rustc_hash::FxHashMap;

    let state = RustSimState::new("amd64").expect("state");
    let mut fork_snapshots = FxHashMap::default();
    fork_snapshots.insert(
        7u64,
        crate::interpreter::BranchSnapshot {
            solver: state.solver().borrow().fork(),
            registers: state.registers().fork(),
            memory: None,
        },
    );
    let pre_callback_snapshot = Some(state.fork());
    PendingCallback {
        state,
        pre_callback_snapshot,
        reason: CallbackReason::Syscall { num: Some(60) },
        jumpkind: None,
        solver_ctx: None,
        deferred_forks: Vec::new(),
        stored_conditions: FxHashMap::default(),
        fork_snapshots,
    }
}

/// Direct coverage of the angr-0jh0j.82 split: this seam owns exactly the one
/// context of the three that is NOT a `RustSimState` — the pre-branch fork
/// snapshot — and leaves the other two to `set_deterministic`'s
/// `all_live_states` walk. Asserting the two *untouched* contexts is the point:
/// re-widening this function would silently duplicate that walk's work, and
/// narrowing it further would drop the angr-sqfj8.32 lineage entirely.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn apply_fork_snapshots_deterministic_reaches_only_the_branch_snapshots() {
    let pending = pending_with_all_contexts();

    for v in [true, false, true] {
        apply_fork_snapshots_deterministic(&pending, v);
        assert_eq!(
            pending.fork_snapshots[&7].solver.is_deterministic(),
            v,
            "pre-branch fork snapshot follows apply_fork_snapshots_deterministic({v})"
        );
        assert!(
            !state_is_deterministic(&pending.state),
            "the parked state is a RustSimState `all_live_states` yields — this \
             seam must not touch it"
        );
        assert!(
            !state_is_deterministic(
                pending
                    .pre_callback_snapshot
                    .as_ref()
                    .expect("pre-callback snapshot"),
            ),
            "so is the pre-callback snapshot (the deferred forks' fork_base)"
        );
    }
}

/// A lightweight callback owns only its state: no `pre_callback_snapshot` and
/// no `fork_snapshots`. The seam must be a no-op on it rather than panicking
/// on the absent map.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn apply_fork_snapshots_deterministic_handles_absent_snapshots() {
    let state = RustSimState::new("amd64").expect("state");
    let pending = PendingCallback::lightweight(
        state,
        CallbackReason::FindPredicate { addr: 0x400000 },
    );

    apply_fork_snapshots_deterministic(&pending, true);
    assert!(
        !state_is_deterministic(&pending.state),
        "a lightweight callback has no fork snapshots, and its own state is \
         covered by `all_live_states` rather than by this seam"
    );
}

// ---------------------------------------------------------------------------
// import_python_constraints
// ---------------------------------------------------------------------------

/// Collect bound objects into the `PyList` shape the importer takes.
fn constraint_list<'py>(py: Python<'py>, items: &[Bound<'py, PyAny>]) -> Bound<'py, PyList> {
    let list = PyList::empty(py);
    for item in items {
        list.append(item).expect("append constraint");
    }
    list
}

/// A duck-typed claripy AST: any object exposing `.op` / `.args` is what
/// `claripy_to_rustbv` reads, and `types.SimpleNamespace` supplies both
/// without pulling in claripy. `BVV` is deliberate — it is the one op the
/// importer does not cache, so the object never needs to be hashable.
///
/// The point of the duck type is the *fast path declining*: claripy's z3
/// backend raises on a non-AST, so these constraints are the only way to reach
/// `import_python_constraints`'s slow path deterministically on a Z3 build
/// (with a real claripy AST the raw-pointer fast path always wins).
fn duck_bvv(py: Python<'_>, value: u64, width: u32) -> Bound<'_, PyAny> {
    let types = py.import("types").expect("import types");
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("op", "BVV").expect("set op");
    kwargs.set_item("args", (value, width)).expect("set args");
    types
        .call_method("SimpleNamespace", (), Some(&kwargs))
        .expect("SimpleNamespace")
}

/// Slow path, `width == 1` arm: the constraint is assumed true as-is. Pinned
/// by polarity — a false one must make the context UNSAT, which no-op import
/// behavior could not produce.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_slow_path_boolean_is_assumed_as_is() {
    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let list = constraint_list(py, &[duck_bvv(py, 1, 1)]);
        assert_eq!(
            import_python_constraints(py, &ctx, &list, "test"),
            1,
            "a convertible constraint counts as added"
        );
        assert!(ctx.is_sat(), "assuming true keeps the context satisfiable");

        let false_ctx = crate::symbolic::SymContext::new();
        let false_list = constraint_list(py, &[duck_bvv(py, 0, 1)]);
        assert_eq!(
            import_python_constraints(py, &false_ctx, &false_list, "test"),
            1
        );
        assert!(
            !false_ctx.is_sat(),
            "a width-1 constraint is assumed true verbatim, so assuming 0 must be UNSAT"
        );
    });
}

/// Slow path, `width != 1` arm: a wider value is widened to `bv != 0`, not
/// truncated to its low bit. Value 2 discriminates the two: its low bit is 0,
/// so a truncating implementation would report UNSAT here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_slow_path_widens_non_boolean_to_nonzero() {
    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let before = ctx.assumed_local_len();
        let list = constraint_list(py, &[duck_bvv(py, 2, 64)]);
        assert_eq!(import_python_constraints(py, &ctx, &list, "test"), 1);
        assert!(
            ctx.is_sat(),
            "2 != 0 holds -- the widening is `!= 0`, not a low-bit truncation"
        );
        let assumed = ctx.get_assumed_constraints();
        assert_eq!(
            assumed.len() - before,
            1,
            "the widened constraint is recorded once in the export IR"
        );
        assert_eq!(
            assumed[before].0.width(),
            1,
            "what lands in the export IR is the width-1 comparison, not the 64-bit operand"
        );

        let zero_ctx = crate::symbolic::SymContext::new();
        let zero_list = constraint_list(py, &[duck_bvv(py, 0, 64)]);
        assert_eq!(
            import_python_constraints(py, &zero_ctx, &zero_list, "test"),
            1
        );
        assert!(
            !zero_ctx.is_sat(),
            "0 != 0 is false, so the widened constraint must make the context UNSAT"
        );
    });
}

/// An item neither path can convert is logged and skipped, never raised: it
/// must not abort the import of its well-formed siblings, and it must not be
/// counted as added.
#[test]
fn import_skips_unconvertible_items_without_aborting_siblings() {
    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let junk = py
            .import("builtins")
            .expect("builtins")
            .call_method0("object")
            .expect("object()");
        let list = constraint_list(py, &[junk, duck_bvv(py, 1, 1)]);
        assert_eq!(
            import_python_constraints(py, &ctx, &list, "test"),
            1,
            "only the convertible sibling counts; the junk item is skipped, not raised"
        );
    });
}

/// Fast path (Z3 builds), convertible arm: a real claripy constraint is
/// asserted via its raw Z3 pointer *and* recorded in the assumed IR, because
/// `claripy_to_rustbv` also succeeds for it (angr-op0dn.14.2 — otherwise the
/// constraint would be exported as an opaque residual).
///
/// `#[ignore]` for the same reason as `constraint_sync_tests.rs`'s
/// `sync_constraints_fp_rescued_via_z3_pointer`: reaching the raw-pointer path
/// means claripy's `Z3_context` has to be the Rust thread-local one, and
/// `install_python_z3_context` is process-global while libtest runs each test
/// on its own thread. Production installs it once before any `SymContext`
/// exists (`rust_manager._setup_shared_z3_context`). Run explicitly:
///
/// ```text
/// cargo test --release import_fast_path -- --ignored --test-threads=1
/// ```
#[cfg(feature = "vex-engine-z3")]
#[test]
#[ignore = "installs claripy's Z3 context process-wide; needs --test-threads=1"]
fn import_fast_path_asserts_and_records_convertible_constraint() {
    Python::initialize();
    Python::attach(|py| {
        if !crate::engine::install_python_z3_context(py) {
            return;
        }
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let ctx = crate::symbolic::SymContext::new();
        let x = claripy
            .call_method1("BVS", ("import_fast_x", 64u32))
            .expect("BVS");
        let c = x.call_method1("__eq__", (0x41u64,)).expect("eq");

        let before = ctx.assumed_local_len();
        let list = constraint_list(py, &[c]);
        assert_eq!(import_python_constraints(py, &ctx, &list, "initial"), 1);
        assert_eq!(
            ctx.assumed_local_len() - before,
            1,
            "a fast-path constraint that also converts to a RustBV is recorded in the \
             assumed IR, not left as an opaque residual"
        );

        let rust_x = crate::claripy_bridge::claripy_to_rustbv(py, &x, &ctx).expect("import symbol");
        assert_eq!(
            ctx.eval(&rust_x),
            Some(0x41),
            "the raw Z3 pointer must really be asserted on this context's solver"
        );
    });
}

/// Fast path, *unconvertible* arm: an FP comparison has no `claripy_to_rustbv`
/// arm, so the importer asserts the raw pointer with `add_constraint_raw` and
/// adds nothing to the assumed IR. Proven by a second import: if the first
/// constraint had merely been counted, `f < 0.0` alone would still be SAT.
///
/// `#[ignore]` for the same process-global-context reason as its sibling above.
#[cfg(feature = "vex-engine-z3")]
#[test]
#[ignore = "installs claripy's Z3 context process-wide; needs --test-threads=1"]
fn import_fast_path_raw_only_for_unconvertible_constraint() {
    Python::initialize();
    Python::attach(|py| {
        if !crate::engine::install_python_z3_context(py) {
            return;
        }
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let ctx = crate::symbolic::SymContext::new();
        let sort = claripy.getattr("FSORT_DOUBLE").expect("FSORT_DOUBLE");
        let f = claripy
            .call_method1("FPS", ("import_fast_f", &sort))
            .expect("FPS");
        let one = claripy.call_method1("FPV", (1.0f64, &sort)).expect("FPV 1");
        let zero = claripy.call_method1("FPV", (0.0f64, &sort)).expect("FPV 0");
        let gt = f.call_method1("__gt__", (&one,)).expect("f > 1.0");

        // Precondition: the typed converter must genuinely decline this op,
        // otherwise the test covers the sibling arm instead.
        assert!(
            crate::claripy_bridge::claripy_to_rustbv(py, &gt, &ctx).is_err(),
            "fpGT must be unsupported by the typed converter for this test to mean anything"
        );

        let before = ctx.assumed_local_len();
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[gt]), "pending"),
            1
        );
        assert_eq!(
            ctx.assumed_local_len(),
            before,
            "an unconvertible constraint is asserted raw only -- no assumed-IR entry"
        );
        assert!(ctx.is_sat(), "f > 1.0 alone is satisfiable");

        let lt = f.call_method1("__lt__", (&zero,)).expect("f < 0.0");
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[lt]), "pending"),
            1
        );
        assert!(
            !ctx.is_sat(),
            "the raw-asserted constraint must really be on the solver: \
             f > 1.0 && f < 0.0 is UNSAT"
        );
    });
}
