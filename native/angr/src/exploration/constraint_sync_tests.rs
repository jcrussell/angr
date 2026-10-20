// Tests for exploration/constraint_sync.rs (split out of helpers_tests.rs
// alongside the source split, angr-9ke6b.76).

use super::*;
use crate::state::RustSimState;

// ---------------------------------------------------------------------------
// sync_constraints_from_python (angr-9ke6b.78)
//
// The prune-on-UNSAT contract: `Ok(false)` means "this state became
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
#[test]
fn sync_constraints_typed_convert_binds_constraint() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("angr.rustylib.claripy") else {
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

/// Contradictory constraints make the state UNSAT, and the method reports
/// that as `Ok(false)` so `resume.rs` prunes instead of exploring a dead state.
#[test]
fn sync_constraints_unsat_returns_false_for_pruning() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("angr.rustylib.claripy") else {
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
            "an UNSAT state must be reported as false so the caller prunes it"
        );
    });
}

/// A literal wider than 128 bits has no `RustBV::Concrete` form, but the
/// comparison imports whole as a clarirs term (angr-mx00p.12), so the typed
/// tier takes it. The import must be lossless, which the second sync proves:
/// if the first constraint had
/// merely been counted and dropped, `x == 2^201` would still be SAT.
#[test]
fn sync_constraints_wide_literal_imported_by_the_typed_tier() {
    use clarirs_core::prelude::*;
    use num_bigint::BigUint;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        let x = GLOBAL_CONTEXT.bvs("sync_wide_x", 256).expect("BVS");
        let wide = |bit: usize| {
            GLOBAL_CONTEXT
                .bvv(BitVec::from((BigUint::from(1u8) << bit, 256)))
                .expect("wide BVV")
        };
        let first = claripy_object(py, GLOBAL_CONTEXT.eq_(&x, wide(200)).expect("x == 2^200"));
        {
            let solver = state.solver();
            let ctx = solver.borrow();
            assert!(
                claripy_to_rustbv(py, &first, &ctx).is_ok(),
                "a comparison against a >128-bit literal imports as a clarirs term"
            );
        }

        let list = constraint_list(py, &[first]);
        assert!(
            mgr.sync_constraints_from_python(py, &state, &list)
                .expect("sync ok"),
            "x == 2^200 alone is satisfiable"
        );

        let second = claripy_object(py, GLOBAL_CONTEXT.eq_(&x, wide(201)).expect("x == 2^201"));
        let list2 = constraint_list(py, &[second]);
        assert!(
            !mgr.sync_constraints_from_python(py, &state, &list2)
                .expect("sync ok"),
            "the imported constraint must really be on the solver: x == 2^200 && x == 2^201 is UNSAT"
        );
    });
}

/// Tier 3 (failure): an item neither tier can convert is counted and logged,
/// never raised — one unconvertible entry must not abort the import of its
/// well-formed siblings, and a still-satisfiable state stays alive.
#[test]
fn sync_constraints_unconvertible_item_does_not_abort_siblings() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("angr.rustylib.claripy") else {
            return;
        };
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("state");

        // A bare `object()` is not a claripy AST, so the typed tier errors and
        // the Z3 lowering refuses it — both tiers decline.
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

/// The partial-sync UNSAT case the partial-sync re-check was written for: some
/// constraints failed to convert AND the ones that landed are contradictory. It
/// must be pruned.
///
/// Note for future readers: this outcome is decided by the unconditional
/// post-sync SAT check, which dominates the partial-sync re-check (that one
/// fires only when `failed_count > 0 && success_count > 0`, a strict subset of
/// what the unconditional check already rejected). The re-check is retained
/// as defence-in-depth; this test pins the observable contract — partial
/// sync + UNSAT => `Ok(false)` — not which gate fires.
#[test]
fn sync_constraints_partial_failure_unsat_is_pruned() {
    Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("angr.rustylib.claripy") else {
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
