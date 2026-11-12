//! The per-thread cache of built Z3 solvers (angr-az6ru.2).
//!
//! A [`SymContext`](super::SymContext) owns no `z3::Solver`. Its constraints
//! are the assertion log plus the open savepoints, and a solver is a cache
//! of that list: [`SymContext::with_z3_solver`](super::SymContext) takes the
//! context's solver out of this cache (or builds one), brings it up to date
//! with the list, runs the query, and puts it back.
//!
//! The cache holds at most [`capacity`] solvers per thread, least recently
//! used first out, so the number of live solvers no longer follows the number
//! of live states. A context whose solver was evicted pays one rebuild from
//! its list at its next query. A solver belongs to the one context whose
//! `SolverSlot` names it and is never handed to another: a fork gets a new
//! slot and builds its own on first use.
//!
//! **Panic policy:** `#![deny(clippy::unwrap_used, clippy::expect_used)]`,
//! as for the rest of the `SymContext` files.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::solver_build::build_solver;
use super::stats::SOLVER_CACHE_EVICT_COUNT;

/// Solvers kept per thread when `ANGR_RUST_SOLVER_CACHE_SIZE` is unset.
pub const DEFAULT_CAPACITY: usize = 128;

/// How many built solvers one thread keeps. Read once from
/// `ANGR_RUST_SOLVER_CACHE_SIZE`; at least one, so the context in the middle
/// of a run of queries keeps its solver between them.
pub fn capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("ANGR_RUST_SOLVER_CACHE_SIZE")
            .ok() // SILENT(cat-a): an unset variable is the default
            .and_then(|s| s.trim().parse::<usize>().ok()) // SILENT(cat-a): an unparsable value is the default
            .unwrap_or(DEFAULT_CAPACITY)
            .max(1)
    })
}

/// One open `push()` scope as the solver holds it.
pub(super) struct Frame {
    /// The savepoint this frame was pushed for.
    pub(super) savepoint: u64,
    /// How many log entries were asserted below the frame.
    pub(super) start: usize,
}

/// A built solver and how much of its context's list it holds.
pub(super) struct BuiltSolver {
    pub(super) solver: z3::Solver,
    /// The log entries `[0, asserted)` are on the solver.
    pub(super) asserted: usize,
    /// The solver's open scopes, outermost first.
    pub(super) frames: Vec<Frame>,
    /// The timeout the solver's params were last set to.
    pub(super) timeout_ms: u32,
}

impl BuiltSolver {
    /// An empty solver with `timeout_ms` applied.
    pub(super) fn new(timeout_ms: u32) -> Self {
        Self {
            solver: build_solver(timeout_ms),
            asserted: 0,
            frames: Vec::new(),
            timeout_ms,
        }
    }
}

/// A context's identity in the cache. Dropping it drops the context's solver
/// on this thread, so a dead state's solver does not wait for eviction.
pub(super) struct SolverSlot(u64);

impl SolverSlot {
    /// A slot no other context has.
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Take this context's solver out of the cache, if it has one here.
    pub(super) fn take(&self) -> Option<BuiltSolver> {
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let at = cache.iter().position(|(owner, _)| *owner == self.0)?;
            Some(cache.remove(at).1)
        })
    }

    /// Put this context's solver back as the most recently used, evicting
    /// the least recently used ones past [`capacity`].
    pub(super) fn put(&self, built: BuiltSolver) {
        let evicted: Vec<(u64, BuiltSolver)> = CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push((self.0, built));
            let over = cache.len().saturating_sub(capacity());
            cache.drain(..over).collect()
        });
        SOLVER_CACHE_EVICT_COUNT.fetch_add(evicted.len() as u64, Ordering::Relaxed);
        // Dropped here, outside the borrow.
        drop(evicted);
    }

    /// Whether this context has a built solver on this thread.
    pub(super) fn is_built(&self) -> bool {
        CACHE.with(|cache| cache.borrow().iter().any(|(owner, _)| *owner == self.0))
    }
}

impl Drop for SolverSlot {
    fn drop(&mut self) {
        // `try_with`: a context dropped while the thread is tearing down has
        // no cache left to clear. The solver is dropped outside the borrow.
        let mine = CACHE.try_with(|cache| {
            let mut cache = cache.try_borrow_mut().ok()?; // SILENT(cat-a): a held borrow means the solver is out on loan and not in the cache
            let at = cache.iter().position(|(owner, _)| *owner == self.0)?;
            Some(cache.remove(at))
        });
        drop(mine);
    }
}

thread_local! {
    /// Built solvers by owning slot, least recently used first.
    static CACHE: RefCell<Vec<(u64, BuiltSolver)>> = const { RefCell::new(Vec::new()) };
}

/// How many built solvers this thread holds.
pub fn live_solvers() -> usize {
    CACHE.with(|cache| cache.borrow().len())
}
