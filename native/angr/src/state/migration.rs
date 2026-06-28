//! Cross-worker state migration transport (angr-1ilq.1).
//!
//! A work-stealing scheduler (angr-1ilq.3) moves a [`RustSimState`] from the
//! worker that owns it to a stealing worker on another thread. `RustSimState`
//! is irreducibly `!Send`: its register/memory/solver graph is built from
//! [`RustBV`](crate::symbolic::RustBV), which carries thread-local-context-bound
//! Z3 ASTs (`z3::ast::BV`), and the solver holds a `z3::Solver`/`z3::Model`.
//! None of those may cross a thread.
//!
//! Rather than asserting `unsafe impl Send for RustSimState` — which is
//! **unsound** (it would let the stealing thread read the owner's *live* Z3
//! context, and race the non-atomic `Rc` solver refcount; see the
//! `angr-1ilq.1` bead notes for hazards A/B/C) — this module crosses the
//! thread boundary with data that is `Send` *by construction*:
//!
//! * `state_bytes` — the versioned serde envelope
//!   ([`RustSimState::to_serialized`]). Every Z3 AST is serialized to a
//!   context-free `RustBVData` shadow, so the bytes carry no Z3 handles. The
//!   stealing worker rebuilds the ASTs in *its own* Z3 context via
//!   [`RustSimState::from_serialized`] — the source context is never read off
//!   the owning thread.
//! * the three `Py<PyAny>` overlay maps — Python claripy ASTs are GIL-managed,
//!   not Z3-context-bound, so they cross threads as live handles
//!   (`Py<T>: Send`). `to_serialized` drops these (Bucket D), so migration
//!   carries them explicitly; otherwise a migrated state would silently lose
//!   its Python-side symbolic overlays on callback-heavy workloads.
//!
//! `last_time` is intentionally NOT carried: it resets to `None`, identical to
//! [`RustSimState::fork`] and [`RustSimState::from_snapshot`]. (Only the
//! rejected `translate_state` path preserved it.)

use super::*;

/// `Send` transport for moving one state between worker threads (angr-1ilq.1).
///
/// Built on the owning worker with [`RustSimState::detach_for_migration`];
/// rebuilt on the stealing worker with [`StateMigrationPayload::reattach`]
/// after that worker's Z3 context is the active thread-local.
pub struct StateMigrationPayload {
    state_bytes: Vec<u8>,
    symbolic_pages: HashMap<u64, Py<PyAny>>,
    hook_symbolic_memory: HashMap<u64, (Py<PyAny>, u32)>,
    addr_to_ast: HashMap<u64, (Py<PyAny>, u32)>,
}

// angr-1ilq.1: the migration transport is `Send` by construction, proven at
// compile time. This REPLACES the unsound `unsafe impl Send for RustSimState`
// the bead originally proposed — this design contains no `unsafe`. If a future
// field reintroduces a non-`Send` member, this assertion fails the build.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<StateMigrationPayload>();
};

impl RustSimState {
    /// Detach this state into a [`StateMigrationPayload`] for cross-worker
    /// migration (angr-1ilq.1).
    ///
    /// Runs on the **owning** worker thread: the only read of this state's
    /// (source) Z3 context happens here, via [`Self::to_serialized`], never
    /// cross-thread. Consumes `self` so the `Py` overlay handles move out
    /// without a GIL `clone_ref`.
    pub fn detach_for_migration(self) -> StateMigrationPayload {
        let state_bytes = self.to_serialized();
        StateMigrationPayload {
            state_bytes,
            symbolic_pages: self.symbolic_pages,
            hook_symbolic_memory: self.hook_symbolic_memory,
            addr_to_ast: self.addr_to_ast,
        }
    }
}

impl StateMigrationPayload {
    /// Rebuild the migrated state on the **stealing** worker thread
    /// (angr-1ilq.1).
    ///
    /// # Precondition
    ///
    /// `target_ctx` must already be the active thread-local Z3 context
    /// (`z3::Context::set_thread_local(target_ctx)`). Every Z3 AST is
    /// reconstructed in the *active thread-local* context through
    /// [`RustSimState::from_serialized`] — NOT in `target_ctx` directly — so if
    /// the caller has not swapped the thread-local first, rebuilding in the
    /// stale context would later mix ASTs across contexts (Z3 UB). This is
    /// checked unconditionally (not a `debug_assert`, which a release build
    /// would compile out, re-opening exactly the silent cross-context footgun
    /// this `unsafe`-free design exists to remove) and returns
    /// [`SnapshotError::ContextMismatch`] on violation.
    pub fn reattach(self, target_ctx: &z3::Context) -> Result<RustSimState, SnapshotError> {
        if !std::ptr::eq(
            target_ctx.get_z3_context().as_ptr(),
            z3::Context::thread_local().get_z3_context().as_ptr(),
        ) {
            return Err(SnapshotError::ContextMismatch);
        }
        let mut state = RustSimState::from_serialized(&self.state_bytes)?;
        state.symbolic_pages = self.symbolic_pages;
        state.hook_symbolic_memory = self.hook_symbolic_memory;
        state.addr_to_ast = self.addr_to_ast;
        Ok(state)
    }
}
