// Tests for exploration/constraint_sync.rs (split out of helpers_tests.rs
// alongside the source split, angr-9ke6b.76).

use super::*;
use crate::state::RustSimState;

// ---------------------------------------------------------------------------
// sync_constraints_from_python (angr-9ke6b.78)
//
// The P12 prune-on-UNSAT contract: `Ok(false)` means "this state became
// unsatisfiable while importing the constraints a Python callback added, so
// `resume.rs` must prune it". These tests pin all three conversion tiers
// (typed convert / Z3-pointer rescue / failure) and the SAT gate.
//
// Every test skips silently when claripy is not importable, matching the
// convention in `claripy_bridge_tests.rs` — the cargo-test binary has no
// guaranteed venv.
// ---------------------------------------------------------------------------

/// Build a `PyList` out of already-bound Python objects (the shape
/// `sync_constraints_from_python` takes).
#[cfg(feature = "vex-engine-z3")]
fn constraint_list<'py>(
    py: Python<'py>,
    items: &[Bound<'py, PyAny>],
) -> Bound<'py, pyo3::types::PyList> {
    use pyo3::types::PyListMethods;
    let list = pyo3::types::PyList::empty(py);
    for item in items {
        list.append(item).expect("append constraint");
    }
    list
}

/// Tier 1 (typed convert): a claripy constraint that `claripy_to_rustbv`
/// understands must land on the state's own solver, not just be counted.
/// Proven by solving for the symbol afterwards rather than by the return
/// value alone — a no-op import would also return `Ok(true)`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn sync_constraints_typed_convert_binds_constraint() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("claripy") else {
            return; // claripy not importable in this env — skip
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        let x = claripy
            .call_method1("BVS", ("sync_typed_x", 64u32))
            .expect("BVS");
        let c = x.call_method1("__eq__", (0x41u64,)).expect("eq");

        let list = constraint_list(py, &[c]);
        assert!(
            mgr.sync_constraints_from_python(py, &state, &list)
                .expect("sync ok"),
            "a satisfiable constraint must report SAT"
        );

        let solver = state.solver();
        let ctx = solver.borrow();
        let rust_x = claripy_to_rustbv(py, &x, &ctx).expect("import symbol");
        assert_eq!(
            ctx.eval(&rust_x),
            Some(0x41),
            "the imported constraint must actually bind the symbol on the state's solver"
        );
    });
}

/// P12: contradictory constraints make the state UNSAT, and the method reports
/// that as `Ok(false)` so `resume.rs` prunes instead of exploring a dead state.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn sync_constraints_unsat_returns_false_for_pruning() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        let x = claripy
            .call_method1("BVS", ("sync_unsat_x", 64u32))
            .expect("BVS");
        let c1 = x.call_method1("__eq__", (1u64,)).expect("eq 1");
        let c2 = x.call_method1("__eq__", (2u64,)).expect("eq 2");

        let list = constraint_list(py, &[c1, c2]);
        assert!(
            !mgr.sync_constraints_from_python(py, &state, &list)
                .expect("sync ok"),
            "P12: an UNSAT state must be reported as false so the caller prunes it"
        );
    });
}

/// Tier 2 (Z3-pointer rescue): an FP comparison has no arm in
/// `claripy_to_rustbv` (`import.rs` models no `fp*` op), so it takes the
/// `claripy.backends.z3` pointer path. The rescue must be lossless, which the
/// second sync proves: if the first constraint had merely been counted and
/// dropped, `f < 0.0` would still be SAT.
///
/// `#[ignore]` — unlike its siblings this test must install claripy's
/// `Z3_context` as the Rust thread-local (`install_python_z3_context`), which
/// is the one piece of *process*-global state the shared-Z3 design has. libtest
/// runs each test on its own thread, so under the default multi-threaded runner
/// this test's context install races the Rust-owned contexts its sibling tests
/// stand up, and the rescued assertion intermittently lands somewhere the
/// state's own solver never sees (observed: ~1 run in 3 with
/// `sync_constraints_typed_convert_binds_constraint` scheduled alongside it).
/// That is a property of the test harness, not of the code under test —
/// production installs the shared context once, before any `SymContext` exists
/// (`rust_manager._setup_shared_z3_context`). Run it explicitly — the
/// `Z3_LIBRARY_PATH_OVERRIDE` is not optional (angr-exwth; see the recipe on
/// `constraints_tests.rs`'s
/// `import_fast_path_asserts_and_records_convertible_constraint` for why):
///
/// ```text
/// Z3_LIBRARY_PATH_OVERRIDE=$VIRTUAL_ENV/lib/python3.12/site-packages/z3/lib \
///   cargo test --release sync_constraints_fp -- --ignored --test-threads=1
/// ```
#[cfg(feature = "vex-engine-z3")]
#[test]
#[ignore = "installs claripy's Z3 context process-wide; needs --test-threads=1"]
fn sync_constraints_fp_rescued_via_z3_pointer() {
    Python::initialize();
    Python::attach(|py| {
        if !crate::engine::install_python_z3_context(py) {
            return;
        }
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        let sort = claripy.getattr("FSORT_DOUBLE").expect("FSORT_DOUBLE");
        let f = claripy
            .call_method1("FPS", ("sync_fp_f", &sort))
            .expect("FPS");
        let one = claripy.call_method1("FPV", (1.0f64, &sort)).expect("FPV 1");
        let zero = claripy.call_method1("FPV", (0.0f64, &sort)).expect("FPV 0");

        let gt = f.call_method1("__gt__", (&one,)).expect("f > 1.0");
        // Precondition for the test to mean anything: the typed tier must
        // genuinely decline this op, otherwise tier 2 is never reached.
        {
            let solver = state.solver();
            let ctx = solver.borrow();
            assert!(
                claripy_to_rustbv(py, &gt, &ctx).is_err(),
                "fpGT must be unsupported by the typed converter for this test to cover tier 2"
            );
        }

        let list = constraint_list(py, &[gt]);
        assert!(
            mgr.sync_constraints_from_python(py, &state, &list)
                .expect("sync ok"),
            "f > 1.0 alone is satisfiable"
        );

        let lt = f.call_method1("__lt__", (&zero,)).expect("f < 0.0");
        let list2 = constraint_list(py, &[lt]);
        assert!(
            !mgr.sync_constraints_from_python(py, &state, &list2)
                .expect("sync ok"),
            "the rescued FP constraint must really be on the solver: f > 1.0 && f < 0.0 is UNSAT"
        );
    });
}

/// Tier 3 (failure): an item neither tier can convert is counted and logged,
/// never raised — one unconvertible entry must not abort the import of its
/// well-formed siblings, and a still-satisfiable state stays alive.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn sync_constraints_unconvertible_item_does_not_abort_siblings() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        // A bare `object()` is not a claripy AST, so the typed tier errors and
        // `claripy.backends.z3.convert` raises — both tiers decline.
        let junk = py
            .import("builtins")
            .expect("builtins")
            .call_method0("object")
            .expect("object()");
        let x = claripy
            .call_method1("BVS", ("sync_partial_x", 64u32))
            .expect("BVS");
        let good = x.call_method1("__eq__", (0x55u64,)).expect("eq");

        let list = constraint_list(py, &[junk, good]);
        assert!(
            mgr.sync_constraints_from_python(py, &state, &list)
                .expect("a failed conversion must not raise"),
            "the state is still satisfiable, so sync reports SAT"
        );

        let solver = state.solver();
        let ctx = solver.borrow();
        let rust_x = claripy_to_rustbv(py, &x, &ctx).expect("import symbol");
        assert_eq!(
            ctx.eval(&rust_x),
            Some(0x55),
            "the convertible sibling must still bind despite the failed item"
        );
    });
}

/// The partial-sync UNSAT case the P14 gate was written for: some constraints
/// failed to convert AND the ones that landed are contradictory. It must be
/// pruned.
///
/// Note for future readers: under `vex-engine-z3` this outcome is decided by
/// the P12 gate, which SAT-checks unconditionally and therefore dominates P14
/// (P14 re-checks only when `failed_count > 0 && success_count > 0`, a strict
/// subset of what P12 already rejected). P14 is retained as defence-in-depth
/// for a build where P12's check is compiled out; this test pins the observable
/// contract — partial sync + UNSAT => `Ok(false)` — not which gate fires.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn sync_constraints_partial_failure_unsat_is_pruned() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("claripy") else {
            return;
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        let junk = py
            .import("builtins")
            .expect("builtins")
            .call_method0("object")
            .expect("object()");
        let x = claripy
            .call_method1("BVS", ("sync_p14_x", 64u32))
            .expect("BVS");
        let c1 = x.call_method1("__eq__", (7u64,)).expect("eq 7");
        let c2 = x.call_method1("__eq__", (9u64,)).expect("eq 9");

        let list = constraint_list(py, &[c1, junk, c2]);
        assert!(
            !mgr.sync_constraints_from_python(py, &state, &list)
                .expect("sync ok"),
            "partial conversion + contradictory constraints must still prune"
        );
    });
}
