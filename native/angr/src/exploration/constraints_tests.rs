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
            ConstraintImportSummary {
                added: 1,
                failed: 0
            },
            "a convertible constraint counts as added"
        );
        assert!(ctx.is_sat(), "assuming true keeps the context satisfiable");

        let false_ctx = crate::symbolic::SymContext::new();
        let false_list = constraint_list(py, &[duck_bvv(py, 0, 1)]);
        assert_eq!(
            import_python_constraints(py, &false_ctx, &false_list, "test").added,
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
        assert_eq!(import_python_constraints(py, &ctx, &list, "test").added, 1);
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
            import_python_constraints(py, &zero_ctx, &zero_list, "test").added,
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
            ConstraintImportSummary {
                added: 1,
                failed: 1
            },
            "only the convertible sibling counts as added; the junk item is skipped \
             rather than raised, but it is tallied so the caller knows the import \
             was partial"
        );
    });
}

// ---------------------------------------------------------------------------
// import_constraints_and_recheck
// ---------------------------------------------------------------------------

/// The dropped-constraint mitigation (angr-fs8kb.12): a partial import must
/// still re-derive satisfiability from whatever *did* land. The junk item is
/// dropped, the width-1 `0` that survives is contradictory, so the re-check
/// has to report `Some(false)` — `_add_constraints_to_pending`, which did no
/// check at all before this seam existed, could only ever have reported
/// "nothing".
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_constraints_and_recheck_reports_unsat_after_a_partial_import() {
    Python::initialize();
    Python::attach(|py| {
        let state = RustSimState::new("amd64").expect("state");
        let junk = py
            .import("builtins")
            .expect("builtins")
            .call_method0("object")
            .expect("object()");
        let list = constraint_list(py, &[junk, duck_bvv(py, 0, 1)]);
        assert_eq!(
            import_constraints_and_recheck(py, &state, &list, "pending", 42),
            Some(false),
            "assuming a width-1 `0` is UNSAT, and a dropped sibling must not keep \
             the re-check from seeing it"
        );
    });
}

/// Control for the test above: a fully convertible, consistent import reports
/// SAT. Without it, a body that returned `Some(false)` unconditionally would
/// pass the partial-import test.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_constraints_and_recheck_reports_sat_for_a_clean_import() {
    Python::initialize();
    Python::attach(|py| {
        let state = RustSimState::new("amd64").expect("state");
        let list = constraint_list(py, &[duck_bvv(py, 1, 1)]);
        assert_eq!(
            import_constraints_and_recheck(py, &state, &list, "initial", 7),
            Some(true),
            "a consistent import leaves the state satisfiable"
        );
    });
}

/// Fast path (Z3 builds), convertible arm: a real claripy constraint is
/// asserted via its raw Z3 pointer *and* recorded in the assumed IR, because
/// `claripy_to_rustbv` also succeeds for it (angr-op0dn.14.2 — otherwise the
/// constraint would be exported as an opaque residual).
///
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_fast_path_asserts_and_records_convertible_constraint() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let x_ast = GLOBAL_CONTEXT.bvs("import_fast_x", 64).expect("BVS");
        let value = GLOBAL_CONTEXT.bvv(BitVec::from((0x41u64, 64))).expect("BVV");
        let x = claripy_object(py, x_ast.clone());
        let c = claripy_object(py, GLOBAL_CONTEXT.eq_(x_ast, value).expect("eq"));

        let before = ctx.assumed_local_len();
        let list = constraint_list(py, &[c]);
        assert_eq!(import_python_constraints(py, &ctx, &list, "initial").added, 1);
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
            "the lowered Z3 term must really be asserted on this context's solver"
        );
    });
}

/// A literal wider than 128 bits has no `RustBV::Concrete` form, but a
/// comparison is imported whole as a clarirs term (angr-mx00p.12), which holds
/// the literal losslessly, so the typed tier takes it and records it in the
/// assumed IR. Proven by a second import: if the first constraint had merely
/// been counted, `x == 2^201` alone would still be SAT.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_types_a_wide_literal_constraint() {
    use clarirs_core::prelude::*;
    use num_bigint::BigUint;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let x = GLOBAL_CONTEXT.bvs("import_fast_wide_x", 256).expect("BVS");
        let wide = |bit: usize| {
            GLOBAL_CONTEXT
                .bvv(BitVec::from((BigUint::from(1u8) << bit, 256)))
                .expect("wide BVV")
        };
        let first = claripy_object(py, GLOBAL_CONTEXT.eq_(&x, wide(200)).expect("x == 2^200"));

        assert!(
            crate::claripy_bridge::claripy_to_rustbv(py, &first, &ctx).is_ok(),
            "a comparison against a >128-bit literal imports as a clarirs term"
        );

        let before = ctx.assumed_local_len();
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[first]), "pending").added,
            1
        );
        assert_eq!(
            ctx.assumed_local_len() - before,
            1,
            "the typed constraint is recorded in the assumed IR"
        );
        assert!(ctx.is_sat(), "x == 2^200 alone is satisfiable");

        let second = claripy_object(py, GLOBAL_CONTEXT.eq_(&x, wide(201)).expect("x == 2^201"));
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[second]), "pending").added,
            1
        );
        assert!(
            !ctx.is_sat(),
            "the imported constraint must really be on the solver: \
             x == 2^200 && x == 2^201 is UNSAT"
        );
    });
}

/// An FP comparison converts to a `RustBV::Ast` (angr-mx00p.11), so it is
/// recorded in the assumed IR like any other converted constraint rather than
/// asserted raw only, and it still binds: `f > 1.0 && f < 0.0` is UNSAT.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_fp_comparison_converts_to_the_assumed_ir() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let f = GLOBAL_CONTEXT.fps("import_fp_ir_f", FSort::f64()).expect("FPS");
        let one = GLOBAL_CONTEXT.fpv(1.0f64).expect("FPV 1");
        let zero = GLOBAL_CONTEXT.fpv(0.0f64).expect("FPV 0");
        let gt = claripy_object(py, GLOBAL_CONTEXT.fp_gt(f.clone(), one).expect("f > 1.0"));

        let before = ctx.assumed_local_len();
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[gt]), "pending").added,
            1
        );
        assert_eq!(ctx.assumed_local_len() - before, 1, "recorded in the assumed IR");
        assert!(ctx.is_sat(), "f > 1.0 alone is satisfiable");

        let lt = claripy_object(py, GLOBAL_CONTEXT.fp_lt(f, zero).expect("f < 0.0"));
        assert_eq!(
            import_python_constraints(py, &ctx, &constraint_list(py, &[lt]), "pending").added,
            1
        );
        assert!(!ctx.is_sat(), "f > 1.0 && f < 0.0 is UNSAT");
    });
}

/// angr-mx00p.41: the raw-pointer tier hands only a Bool to the solver. Both
/// `import_one_constraint` sites that consume a lowering — the `Z3PtrFirst`
/// fast path and the `RustBvFirst` rescue of an op `claripy_to_rustbv`
/// declines — go through `claripy_z3_ptr`, so a BV-sorted lowering must come
/// back `None` there (the rescue then reports `Failed`) rather than reach
/// `add_constraint_raw` as a mis-sorted Bool.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn claripy_z3_ptr_declines_a_bv_sorted_lowering() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let x = GLOBAL_CONTEXT.bvs("import_bv_sorted_x", 8).expect("BVS");
        let zero = GLOBAL_CONTEXT.bvv(BitVec::from((0u64, 8))).expect("BVV");
        let bv = claripy_object(py, x.clone());
        assert!(
            crate::claripy_bridge::lower_bv(&x).is_ok(),
            "the BV term itself lowers; only its sort disqualifies it"
        );
        assert!(
            claripy_z3_ptr(&bv).is_none(),
            "a BV-sorted lowering must not be offered as a constraint"
        );
        let cond = claripy_object(py, GLOBAL_CONTEXT.eq_(x, zero).expect("eq"));
        assert!(claripy_z3_ptr(&cond).is_some(), "a Bool lowering is");
    });
}

/// angr-mx00p.41, end to end on the `Z3PtrFirst` tier: a BV-sorted constraint
/// falls through to the RustBV tier and is asserted as `x != 0`. Before the
/// Bool gate it was asserted raw as a mis-sorted Bool, which Z3's sort guard
/// drops silently (angr-d01qu), so `x == 0` stayed satisfiable.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_fast_path_widens_a_bv_sorted_constraint_to_nonzero() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = crate::symbolic::SymContext::new();
        let x = GLOBAL_CONTEXT.bvs("import_bv_truthy_x", 8).expect("BVS");
        let zero = GLOBAL_CONTEXT.bvv(BitVec::from((0u64, 8))).expect("BVV");
        let truthy = claripy_object(py, x.clone());
        assert!(matches!(
            import_one_constraint(py, &ctx, &truthy, ConstraintTier::Z3PtrFirst),
            ConstraintImport::Assumed
        ));
        assert!(ctx.is_sat(), "x != 0 alone is satisfiable");

        let is_zero = claripy_object(py, GLOBAL_CONTEXT.eq_(x, zero).expect("eq"));
        assert!(matches!(
            import_one_constraint(py, &ctx, &is_zero, ConstraintTier::Z3PtrFirst),
            ConstraintImport::Assumed
        ));
        assert!(
            !ctx.is_sat(),
            "the BV constraint must really be on the solver as x != 0"
        );
    });
}
