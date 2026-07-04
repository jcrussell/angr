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
use std::sync::Mutex;

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

/// Random-state selection (angr-a32jl.2 prototype): step a uniformly-random
/// active state, append new forks at the tail. Deterministic under a fixed
/// `seed` so a run is byte-reproducible — the seed knob the bead calls for.
///
/// This is the first-cut random searcher. True KLEE random-*path* weights
/// selection by fork-subtree size; that refinement needs subtree bookkeeping
/// on the `on_fork` side and is deferred to a follow-up. Uniform random over
/// the active set is a legitimate baseline searcher and stays fully
/// self-contained within the two-hook seam (no run-loop plumbing).
///
/// Opt-in only via `set_state_selection_random`; never a default.
pub struct RandomState {
    /// SplitMix64 state behind a `Mutex` for interior mutability under the
    /// `&self` `select` hook. `Mutex` (not `Cell`) keeps the `Send + Sync`
    /// supertrait bound so a parallel scheduler can share the policy `Arc`.
    rng: Mutex<u64>,
}

impl RandomState {
    /// Construct with an explicit seed. Any `u64` (including 0) is a valid,
    /// deterministic seed — SplitMix64 does not degenerate at zero.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(seed),
        }
    }

    /// One SplitMix64 draw. Fast, seekable, and dependency-free — no `rand`
    /// crate pulled in for a prototype policy.
    fn next_u64(&self) -> u64 {
        let mut state = self.rng.lock().expect("RandomState rng poisoned");
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl SelectionPolicy for RandomState {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        let len = active.len();
        if len == 0 {
            return None;
        }
        let idx = (self.next_u64() % len as u64) as usize;
        // `remove` (not `swap_remove_*`) keeps the surviving order stable so
        // the selected index maps predictably onto the insertion sequence,
        // which the determinism test relies on. O(n) shift is negligible next
        // to a step's Z3 work.
        active.remove(idx)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "random"
    }
}

#[cfg(test)]
mod tests {
    use super::{RandomState, SelectionPolicy};
    use crate::state::RustSimState;
    use std::collections::{HashMap, VecDeque};
    use z3::Context;

    /// Run a full drain under `RandomState(seed)` over `n` fresh states and
    /// return the dispatch order expressed as *insertion indices* (0..n). Two
    /// runs with the same seed must return the same index sequence even though
    /// absolute `state_id`s differ between runs.
    fn drain_order(seed: u64, n: usize) -> Vec<usize> {
        let _ctx = Context::thread_local();
        let policy = RandomState::new(seed);
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
        for i in 0..n {
            let st = RustSimState::new("amd64").unwrap();
            id_to_idx.insert(st.state_id(), i);
            active.push_back(st);
        }
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(id_to_idx[&st.state_id()]);
        }
        order
    }

    #[test]
    fn test_random_select_is_permutation() {
        let order = drain_order(0xC0FFEE, 12);
        assert_eq!(order.len(), 12, "every inserted state must be dispatched");
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..12).collect::<Vec<_>>(),
            "dispatch order must be a permutation of the insertion set (no drops/dupes)",
        );
    }

    #[test]
    fn test_random_select_deterministic_under_seed() {
        assert_eq!(
            drain_order(42, 16),
            drain_order(42, 16),
            "same seed + same insertion order must reproduce the dispatch sequence",
        );
    }

    #[test]
    fn test_random_select_seed_sensitive() {
        // 16! outcomes: a collision between two distinct seeds is astronomically
        // unlikely, so this is a stable (non-flaky) inequality.
        assert_ne!(
            drain_order(1, 16),
            drain_order(2, 16),
            "different seeds should drive different dispatch orders",
        );
    }

    #[test]
    fn test_random_select_empty_is_none() {
        let policy = RandomState::new(7);
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_random_policy_name() {
        assert_eq!(RandomState::new(0).name(), "random");
    }
}
