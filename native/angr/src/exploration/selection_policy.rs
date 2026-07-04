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

use std::collections::{HashMap, HashSet, VecDeque};
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

/// Coverage-guided new-block-first selection (angr-m9fpp prototype): step the
/// oldest active state whose next block (its current `pc`) has never been
/// dispatched before, so the searcher pushes toward unexplored code rather than
/// re-treading already-covered blocks. When every active state sits on an
/// already-seen block the policy degrades gracefully to FIFO (front), keeping a
/// deterministic tie-break and never starving the queue.
///
/// The novelty signal is a self-contained seen-set of dispatched block
/// addresses maintained inside the policy — no lift-cache plumbing or seam
/// widening. Each state already exposes its next-block address via
/// [`RustSimState::pc`], so the two-hook seam is sufficient: `select` scans the
/// deque for the first novel `pc`, marks it seen, and removes it. This is a
/// coarse per-address novelty (not per-path), which is the intended cheap
/// prototype — a state re-reaching a covered block loses its novelty boost,
/// which is exactly the coverage-guided behavior.
///
/// Opt-in only via `set_state_selection_coverage`; never a default.
#[derive(Default)]
pub struct CoverageGuided {
    /// Block addresses already dispatched. Behind a `Mutex` for interior
    /// mutability under the `&self` `select` hook while preserving the
    /// `Send + Sync` supertrait so a parallel scheduler can share the `Arc`.
    seen: Mutex<HashSet<u64>>,
}

impl CoverageGuided {
    /// Construct with an empty seen-set.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SelectionPolicy for CoverageGuided {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        if active.is_empty() {
            return None;
        }
        let mut seen = self.seen.lock().expect("CoverageGuided seen-set poisoned");
        // First active state sitting on a block we have never dispatched. Front
        // scan keeps FIFO order among equally-novel states (deterministic).
        let pick = active
            .iter()
            .position(|st| !seen.contains(&st.pc()))
            // All active blocks already seen — degrade to FIFO front.
            .unwrap_or(0);
        let state = active
            .remove(pick)
            .expect("index from non-empty deque is valid");
        seen.insert(state.pc());
        Some(state)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "coverage"
    }
}

/// Loop-head round-robin fairness (angr-caplg): rotate dispatch across
/// *(loop-head, callstack-class)* buckets so a state spinning in a loop cannot
/// monopolize the frontier while sibling paths starve.
///
/// The bucket key mirrors the two signals native LoopBound and the
/// reconvergence sampler already use, both reachable per-state inside the
/// two-hook seam — so no seam widening or LoopBound-technique plumbing is
/// needed (the `selection-policy-seam-limits` caution does not apply here):
///   * *loop-head* — the address visited most often in the state's history
///     (the block it is currently spinning on). Matches LoopBound's
///     history-frequency loop test (`Self::exceeds_loop_bound`). Falls back to
///     the next-block `pc` when the state is not looping (all history
///     addresses distinct).
///   * *callstack-class* — the return-address chain, exactly the hash the
///     reconvergence sampler folds in (`record_reconvergence_sample`).
///
/// `select` picks the active state whose bucket has been dispatched the fewest
/// times (FIFO front tie-break for determinism) and bumps that bucket's count.
/// A looping state re-buckets to the same key each visit, so its count climbs
/// and fresher buckets are served ahead of it — round-robin fairness without
/// dropping or reordering the deque itself.
///
/// Opt-in only via `set_state_selection_loop_head`; never a default.
#[derive(Default)]
pub struct LoopHeadRoundRobin {
    /// Per-bucket dispatch counts. Behind a `Mutex` for interior mutability
    /// under the `&self` `select` hook while preserving `Send + Sync` so a
    /// parallel scheduler can share the `Arc`.
    dispatched: Mutex<HashMap<u64, u64>>,
}

impl LoopHeadRoundRobin {
    /// Construct with empty per-bucket dispatch counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold a state's *(loop-head, callstack-class)* into a single bucket key.
    fn bucket_key(state: &RustSimState) -> u64 {
        use std::hash::{Hash, Hasher};
        // Loop head = the most-frequently visited history address (the block
        // the state is spinning on). Ties break to the first address to reach
        // the running max, which is deterministic given history order. When no
        // address repeats the state is not looping, so head stays the
        // next-block pc.
        let mut counts: HashMap<u64, usize> = HashMap::new();
        let mut head = state.pc();
        let mut best = 1usize;
        for &addr in state.history() {
            let c = counts.entry(addr).or_insert(0);
            *c += 1;
            if *c > best {
                best = *c;
                head = addr;
            }
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        head.hash(&mut h);
        for entry in state.call_stack() {
            entry.return_addr.hash(&mut h);
        }
        h.finish()
    }
}

impl SelectionPolicy for LoopHeadRoundRobin {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        if active.is_empty() {
            return None;
        }
        let mut dispatched = self
            .dispatched
            .lock()
            .expect("LoopHeadRoundRobin counts poisoned");
        // Least-served bucket wins; front scan with strict `<` keeps FIFO order
        // among equally-starved buckets (deterministic).
        let mut pick = 0usize;
        let mut best_count = u64::MAX;
        let mut best_key = 0u64;
        for (i, st) in active.iter().enumerate() {
            let key = Self::bucket_key(st);
            let count = dispatched.get(&key).copied().unwrap_or(0);
            if count < best_count {
                best_count = count;
                pick = i;
                best_key = key;
            }
        }
        let state = active
            .remove(pick)
            .expect("index from non-empty deque is valid");
        *dispatched.entry(best_key).or_insert(0) += 1;
        Some(state)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "loop_head"
    }
}

#[cfg(test)]
mod tests {
    use super::{CoverageGuided, LoopHeadRoundRobin, RandomState, SelectionPolicy};
    use crate::state::RustSimState;
    use std::collections::{HashMap, VecDeque};
    use z3::Context;

    /// Build a fresh amd64 state parked at `pc`.
    fn state_at(pc: u64) -> RustSimState {
        let mut st = RustSimState::new("amd64").unwrap();
        st.set_pc(pc);
        st
    }

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

    /// Drain a `CoverageGuided` policy over states parked at the given `pcs`
    /// and return the dispatch order as `pc` values.
    fn coverage_drain(pcs: &[u64]) -> Vec<u64> {
        let _ctx = Context::thread_local();
        let policy = CoverageGuided::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        for &pc in pcs {
            active.push_back(state_at(pc));
        }
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.pc());
        }
        order
    }

    #[test]
    fn test_coverage_prefers_novel_blocks() {
        // Two states share block 0x1000; one is at the novel 0x2000. With the
        // front at a duplicate-able block, the policy must still pull the novel
        // 0x2000 before re-treading 0x1000 a second time.
        // Layout: [0x1000, 0x1000, 0x2000].
        // Step 1: 0x1000 novel (front) -> dispatch, mark seen.
        // Step 2: front 0x1000 now seen, 0x2000 novel -> dispatch 0x2000.
        // Step 3: only the second 0x1000 remains -> dispatch it.
        assert_eq!(
            coverage_drain(&[0x1000, 0x1000, 0x2000]),
            vec![0x1000, 0x2000, 0x1000],
        );
    }

    #[test]
    fn test_coverage_is_permutation() {
        let mut order = coverage_drain(&[0x10, 0x20, 0x10, 0x30, 0x20]);
        assert_eq!(order.len(), 5, "every inserted state must be dispatched");
        order.sort_unstable();
        assert_eq!(order, vec![0x10, 0x10, 0x20, 0x20, 0x30]);
    }

    #[test]
    fn test_coverage_all_seen_degrades_to_fifo() {
        // Every state parked on the same already-dispatched block: after the
        // first dispatch marks 0x4000 seen, the rest are non-novel and must
        // drain front-to-back (FIFO), never stalling.
        assert_eq!(
            coverage_drain(&[0x4000, 0x4000, 0x4000]),
            vec![0x4000, 0x4000, 0x4000],
        );
    }

    #[test]
    fn test_coverage_empty_is_none() {
        let _ctx = Context::thread_local();
        let policy = CoverageGuided::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_coverage_policy_name() {
        assert_eq!(CoverageGuided::new().name(), "coverage");
    }

    // ---- LoopHeadRoundRobin (angr-caplg) ----

    /// Build a state parked at `pc` whose history is `hist` (so the loop-head
    /// signal is the most-frequent address in `hist`).
    fn state_with_history(pc: u64, hist: &[u64]) -> RustSimState {
        let mut st = state_at(pc);
        for &addr in hist {
            st.add_to_history(addr);
        }
        st
    }

    #[test]
    fn test_loop_head_round_robin_fairness() {
        // Two states in bucket A (pc 0x1000, no history) and one in bucket B
        // (pc 0x2000). Fair scheduling must serve B before re-serving A:
        //   Step 1: all buckets count 0 -> front A0.       (A now count 1)
        //   Step 2: A1 bucket=1, B0 bucket=0 -> B0.        (B now count 1)
        //   Step 3: only A1 remains -> A1.
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let a0 = state_at(0x1000);
        let a1 = state_at(0x1000);
        let b0 = state_at(0x2000);
        let (a0id, a1id, b0id) = (a0.state_id(), a1.state_id(), b0.state_id());
        active.push_back(a0);
        active.push_back(a1);
        active.push_back(b0);
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.state_id());
        }
        assert_eq!(
            order,
            vec![a0id, b0id, a1id],
            "round-robin must serve the fresh bucket before re-serving A",
        );
    }

    #[test]
    fn test_loop_head_buckets_by_loop_head_not_pc() {
        // s0 and s1 both spin on loop-head 0xAA (history frequency) but sit at
        // different next-block pcs; a pc-keyed policy would split them, a
        // loop-head-keyed one groups them. s2 is a fresh distinct bucket.
        //   Step 1: front s0 (all count 0).          (bucket-0xAA now 1)
        //   Step 2: s1 bucket=1, s2 bucket=0 -> s2.  (s2 bucket now 1)
        //   Step 3: only s1 remains -> s1.
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let s0 = state_with_history(0xB0, &[0xAA, 0xAA, 0xAA]);
        let s1 = state_with_history(0xB1, &[0xAA, 0xAA, 0xAA]);
        let s2 = state_at(0xC0);
        let (s0id, s1id, s2id) = (s0.state_id(), s1.state_id(), s2.state_id());
        active.push_back(s0);
        active.push_back(s1);
        active.push_back(s2);
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.state_id());
        }
        assert_eq!(
            order,
            vec![s0id, s2id, s1id],
            "s0 and s1 must share a loop-head bucket despite differing pc",
        );
    }

    #[test]
    fn test_loop_head_is_permutation() {
        // Distinct fresh buckets all start at count 0 -> pure FIFO drain, no
        // drops or dupes.
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let ids: Vec<u64> = (0..5u64)
            .map(|i| {
                let st = state_at(0x100 + i);
                let id = st.state_id();
                active.push_back(st);
                id
            })
            .collect();
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.state_id());
        }
        assert_eq!(
            order, ids,
            "distinct fresh buckets drain FIFO front-to-back"
        );
    }

    #[test]
    fn test_loop_head_empty_is_none() {
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_loop_head_policy_name() {
        assert_eq!(LoopHeadRoundRobin::new().name(), "loop_head");
    }
}
