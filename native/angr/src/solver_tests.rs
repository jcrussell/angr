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
