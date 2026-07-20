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

/// angr-87e56: a `close()` routed from a non-owning thread must NOT drop the
/// non-Send payload cross-thread — it stays intact for pyo3's leak-safe
/// dealloc refusal. `RustSolverContext` is `unsendable`, so we cannot move it
/// across a thread boundary in a Rust test; instead assert the guard's core
/// invariant directly: the recorded owner is the constructing thread, and a
/// close() from that same thread does empty it (the negative direction is
/// covered by construction — the `owner` comparison is the only gate).
#[test]
fn test_solver_close_is_owner_guarded() {
    let mut ctx = RustSolverContext::new();
    assert_eq!(ctx.owner, std::thread::current().id());
    ctx.close();
    assert!(ctx.is_closed());
}

#[test]
fn test_z3_available() {
    let available = RustSolverContext::z3_available();
    #[cfg(feature = "vex-engine-z3")]
    assert!(available);
    #[cfg(not(feature = "vex-engine-z3"))]
    assert!(!available);
}

/// Documentation-grade size probe for the `large_enum_variant` allow on
/// `SolverCtxStorage`. Validates iter-61's decision: `Owned` exists
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
