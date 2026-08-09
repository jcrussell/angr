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

/// `pop_to_level` pops exactly `current - target` times, refuses to pop past
/// the bottom of the stack, and is a no-op when the target is not below the
/// current level (the `saturating_sub` case).
#[test]
fn test_pop_to_level_pops_the_difference_and_refuses_to_overshoot() {
    let ctx = RustSolverContext::new();
    for _ in 0..3 {
        ctx.push();
    }
    // 3 -> 1 pops twice, leaving one scope on the stack.
    ctx.pop_to_level(1, 3).expect("two pops available");
    ctx.pop().expect("one scope still open");
    assert!(ctx.pop().is_err(), "stack must be empty now");

    // target >= current pops nothing, so it succeeds on an empty stack.
    ctx.pop_to_level(4, 4).expect("no-op");
    ctx.pop_to_level(5, 2).expect("saturating: no-op");

    // Asking for more pops than there are scopes reports the overshoot
    // instead of panicking partway through.
    ctx.push();
    let err = ctx.pop_to_level(0, 3).expect_err("only one scope open");
    assert!(
        err.to_string().contains("exceeds the solver scope depth"),
        "{err}"
    );
}

// =============================================================================
// Constraint bookkeeping (angr-sqfj8.125)
// =============================================================================

/// `num_constraints`/`constraint_delta` are what the Python side polls to
/// decide whether a callback added constraints; `constraint_delta` saturates
/// rather than underflowing when the baseline is already ahead.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_num_constraints_and_delta_track_adds() {
    let ctx = RustSolverContext::new();
    let base = ctx.num_constraints();
    assert_eq!(ctx.constraint_delta(base), 0);

    let x = ctx.create_symbolic("solver_delta_x", 8).unwrap().id();
    let five = ctx.create_concrete(5, 8).unwrap().id();
    ctx.add_constraint_handle(ctx.op_ne(x, five).unwrap().id())
        .unwrap();

    assert_eq!(ctx.num_constraints(), base + 1);
    assert_eq!(ctx.constraint_delta(base), 1);
    // Baseline ahead of the live count: saturate to 0, never underflow.
    assert_eq!(ctx.constraint_delta(base + 99), 0);
}

/// `get_all_constraints_str` / `export_constraint_info` / `z3_assertion_count`
/// are the debugging + Python-sync views of the same assertion list; they must
/// agree with each other.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_constraint_export_views_agree() {
    let ctx = RustSolverContext::new();
    let x = ctx.create_symbolic("solver_export_x", 8).unwrap().id();
    let five = ctx.create_concrete(5, 8).unwrap().id();
    ctx.add_constraint_handle(ctx.op_eq(x, five).unwrap().id())
        .unwrap();

    let strs = ctx.get_all_constraints_str();
    assert!(!strs.is_empty());
    assert_eq!(ctx.z3_assertion_count(), strs.len());

    let info = ctx.export_constraint_info();
    assert_eq!(info.len(), strs.len());
    assert!(info.iter().all(|(_, trackable)| *trackable));
    assert!(
        info.iter().any(|(s, _)| s.contains("solver_export_x")),
        "the exported strings must name the symbol: {info:?}"
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

// =============================================================================
// claripy-AST constraint API (angr-sqfj8.125) — deliberately NOT covered here
//
// `add_constraint_ast` / `add_constraints` / `add_constraint_tracked_ast` and
// the `z3_ptr.rs` helpers that feed them stay covered by the Python suite
// (`tests/engines/rust/test_solver_ops.py`, which already carries the
// angr-sqfj8.121 double-listing regression) rather than by cargo tests. Two
// environment facts block an in-crate version, both established while writing
// this file:
//
//  1. Reaching those paths means adopting Python's `Z3_context` as this
//     thread's Rust thread-local (`engine::install_python_z3_context`).
//     A `Z3_context` is not thread-safe and libtest runs tests on parallel
//     threads, so a second adopter touching it concurrently SIGSEGVs the whole
//     test binary — not something a gate can carry.
//  2. Even pinned to `--test-threads=1`, the resulting behaviour is
//     order-dependent: a core read back via `unsat_core_assumed` came out
//     non-empty or empty for the *same* fixed code depending on which tests
//     had already run in the process. Under the real extension module (where
//     `_setup_shared_z3_context` runs once at manager construction) that
//     ambiguity does not exist, which is exactly why the Python suite is the
//     right home for this coverage.
//
// Everything above this banner is deliberately Python-free, so it runs in the
// `cargo test` gate.
// =============================================================================
