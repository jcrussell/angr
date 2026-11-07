use super::*;

#[test]
fn test_solver_creation() {
    let _ctx = RustSolverContext::new();
}

#[test]
fn test_solver_fork() {
    let ctx = RustSolverContext::new();
    let _forked = ctx.fork();
}

/// angr-87e56: `close()` on the owning thread drops the payload (idempotent)
/// so a later off-owner tp_dealloc finds `None` and leaks nothing.
#[test]
fn test_solver_close_on_owner_empties_payload() {
    let mut ctx = RustSolverContext::new();
    assert!(!ctx.is_closed());
    ctx.close();
    assert!(ctx.is_closed());
    // Idempotent: a second close() is a no-op, not a double-free/panic.
    ctx.close();
    assert!(ctx.is_closed());
}

/// angr-n0irt.24: exercise the OFF-owner branch of `close()` directly.
///
/// The sibling test above only proves the on-owner drop; the negative
/// direction (a `close()` whose current thread != recorded owner must NOT
/// drop the non-Send payload) was previously "covered by construction" with
/// no assertion. It cannot be tested with a real cross-thread call — the
/// pyclass is `unsendable` and the crate builds `panic = "abort"`, so pyo3's
/// thread checker hard-aborts on any cross-thread borrow (the Python
/// `test_solver_ctx_close.py` header comment records this). Instead spoof the
/// `owner` field to a foreign `ThreadId` while the context stays on this
/// thread, so `close()` takes its non-owner branch. If someone drops the
/// `current().id() == owner` guard (reintroducing the unsound cross-thread
/// drop), the first assertion below flips to true and this test fails.
#[test]
fn test_solver_close_off_owner_is_noop() {
    let mut ctx = RustSolverContext::new();
    // The constructor records the constructing (this) thread as owner — the
    // sole gate `close()` checks before dropping the non-Send payload.
    assert_eq!(ctx.owner, std::thread::current().id());
    // A ThreadId that is provably not this thread's.
    let foreign = std::thread::spawn(|| std::thread::current().id())
        .join()
        .expect("foreign thread id");
    assert_ne!(foreign, std::thread::current().id());

    ctx.owner = foreign;
    ctx.close();
    // Off-owner: the payload is preserved, not dropped cross-thread.
    assert!(!ctx.is_closed());

    // Restore ownership so the real drop empties it on the owning (this)
    // thread — the SymContext/Z3 was constructed here, so this is sound.
    ctx.owner = std::thread::current().id();
    ctx.close();
    assert!(ctx.is_closed());
}

/// Documentation-grade size probe for the `large_enum_variant` allow on
/// `SolverCtxStorage`. Backs the decision not to box it: `Owned` exists
/// precisely to skip the alloc that `Shared` (Rc) requires; boxing
/// `Owned` defeats that. Run with
/// `cargo test --release -p angr -- --nocapture print_solver_storage_sizes`.
#[test]
fn print_solver_storage_sizes() {
    use std::mem::size_of;
    let storage = size_of::<SolverCtxStorage>();
    let sym_ctx = size_of::<SymContext>();
    let shared_payload = size_of::<Rc<RefCell<SymContext>>>();
    println!("SolverCtxStorage          = {storage} bytes");
    println!("  SymContext (Owned)      = {sym_ctx} bytes");
    println!("  Rc<RefCell<SymContext>> = {shared_payload} bytes");
    println!(
        "boxing-Owned-would-save   = {} bytes per fork (and add a heap alloc)",
        sym_ctx.saturating_sub(shared_payload)
    );
}

// =============================================================================
// Solver scope stack (angr-sqfj8.125)
// =============================================================================

/// angr-ph300.48: an unbalanced `pop()` must come back as a `ValueError`,
/// not reach z3-rs's under-pop panic (a process abort under `panic=abort`).
#[test]
fn test_pop_without_push_is_a_value_error() {
    let ctx = RustSolverContext::new();
    let err = ctx.pop().expect_err("under-pop must be refused");
    assert!(err.to_string().contains("no matching push()"), "{err}");

    ctx.push();
    ctx.pop().expect("balanced pop");
    // The stack is empty again, so the next pop is unbalanced too.
    assert!(ctx.pop().is_err());
}

// =============================================================================
// Constraint bookkeeping (angr-sqfj8.125)
// =============================================================================

/// `num_constraints` is what the Python side polls to report constraint growth
/// (`rust_callback_dispatch.py`).
#[test]
fn test_num_constraints_tracks_adds() {
    let ctx = RustSolverContext::new();
    let base = ctx.num_constraints();

    {
        let sc = ctx.i().ctx();
        let x = RustBV::symbol("solver_delta_x", 8);
        sc.assume_true(&x.ne(&RustBV::concrete(5, 8)));
    }

    assert_eq!(ctx.num_constraints(), base + 1);
}

/// `get_all_constraints_str` is the debugging view of the assertion list; it
/// must name the asserted symbol.
#[test]
fn test_constraint_strings_name_the_symbol() {
    let ctx = RustSolverContext::new();
    {
        let sc = ctx.i().ctx();
        let x = RustBV::symbol("solver_export_x", 8);
        sc.assume_true(&x.eq(&RustBV::concrete(5, 8)));
    }

    let strs = ctx.get_all_constraints_str();
    assert!(
        strs.iter().any(|s| s.contains("solver_export_x")),
        "the constraint strings must name the symbol: {strs:?}"
    );
}

/// A fresh context owns its `SymContext` outright; `is_shared` only reports
/// true for one borrowed from a pending state.
#[test]
fn test_new_context_is_not_shared_and_mints_unique_names() {
    let ctx = RustSolverContext::new();
    assert!(!ctx.is_shared());

    let a = ctx.unique_name("solver_uniq");
    let b = ctx.unique_name("solver_uniq");
    assert_ne!(a, b, "unique_name must not repeat");
    assert!(a.starts_with("solver_uniq"), "{a}");
    assert!(b.starts_with("solver_uniq"), "{b}");
}

// claripy-AST constraint API. The ASTs are built as this binary's own claripy
// objects (`claripy_object`), so `add_constraint_ast` takes the lowered-Z3
// fast path exactly as it does under the extension module.

/// An FP constraint has no `RustBV` form, so `add_constraint_ast` can only
/// honor it through the Z3 lowering; the second constraint proves it landed.
#[test]
fn test_add_constraint_ast_lowers_fp_constraint() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = RustSolverContext::new();
        let f = GLOBAL_CONTEXT.fps("solver_fp_f", FSort::f64()).expect("FPS");
        let one = GLOBAL_CONTEXT.fpv(1.0f64).expect("FPV 1");
        let zero = GLOBAL_CONTEXT.fpv(0.0f64).expect("FPV 0");
        let gt = claripy_object(py, GLOBAL_CONTEXT.fp_gt(f.clone(), one).expect("f > 1.0"));
        let lt = claripy_object(py, GLOBAL_CONTEXT.fp_lt(f, zero).expect("f < 0.0"));

        ctx.add_constraint_ast(py, &gt).expect("add f > 1.0");
        assert!(ctx.satisfiable(), "f > 1.0 alone is satisfiable");
        ctx.add_constraint_ast(py, &lt).expect("add f < 0.0");
        assert!(!ctx.satisfiable(), "f > 1.0 && f < 0.0 is UNSAT");
    });
}
