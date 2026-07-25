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
