//! Pluggable state-selection / scheduling policy seam (angr-a32jl.1).
//!
//! The exploration loop makes two ordering decisions about the `active`
//! stash on every step:
//!   1. *select* — which active state to step next (the pop side), and
//!   2. *on_fork* — where a freshly-forked successor lands in the active
//!      deque (the push side).
//!
//! Historically these were a single `use_lifo: bool` open-coded as
//! `pop_front`/`pop_back` in `run_loop.rs`, with an unconditional `push_back`
//! at every fork site. This trait factors that decision behind two hooks so
//! richer policies (random-path, coverage-guided, CFG-distance directed —
//! angr-a32jl.2+) can slot in without touching the run loop.
//!
//! The two built-ins reproduce the pre-refactor behavior exactly. `Fifo`
//! (BFS, the default) selects the front; `Lifo` (DFS) selects the back.
//! Both append new forks at the tail — matching the historical `push_back`
//! at every fork site — so step traces are byte-identical under either.

use std::collections::VecDeque;

use crate::state::RustSimState;

/// A pluggable active-state selection / fork-insertion policy.
///
/// `Send + Sync` supertraits keep the built-ins (zero-sized unit structs)
/// trivially thread-safe so a future parallel scheduler can share a policy.
pub trait SelectionPolicy: Send + Sync {
    /// Choose and remove the next active state to step, or `None` when the
    /// active deque is empty.
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState>;

    /// Insert a freshly-forked successor into the active deque.
    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState);

    /// Stable, human-readable name for logging / stats.
    fn name(&self) -> &'static str;
}

/// Breadth-first (FIFO queue): step the oldest state first, append new forks
/// at the tail. The default policy — historical `use_lifo == false`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Fifo;

impl SelectionPolicy for Fifo {
    #[inline]
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        active.pop_front()
    }

    #[inline]
    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "fifo"
    }
}

/// Depth-first (LIFO stack): step the most-recent state first, append new
/// forks at the tail. Historical `use_lifo == true`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lifo;

impl SelectionPolicy for Lifo {
    #[inline]
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        active.pop_back()
    }

    #[inline]
    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "lifo"
    }
}
