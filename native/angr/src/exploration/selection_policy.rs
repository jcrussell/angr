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

/// CFG-distance directed beam selection (angr-a32jl.4): step the active states
/// closest to a find target first, where "distance" is a one-time
/// `addr -> distance-to-target` snapshot computed Python-side from the angr CFG
/// and shipped in as plain metadata. Zero runtime bounces — the map is
/// immutable after construction and consulted purely from the next-block `pc`
/// each state already exposes, so the policy stays entirely in Rust.
///
/// **Not pure-greedy** (bd memory `ds-directed-search-greedy-trap`): a
/// `beam_width == 1` best-first search TRAPS on data-dependent reachability —
/// correct- and wrong-branch successors of a char check share an *identical*
/// CFG distance, so best-first cannot pick between them and stalls. The default
/// `beam_width >= 2` steps the whole near-optimal frontier, recovering the
/// data-dependent case (matching BFS steps-to-goal with a smaller per-round
/// active frontier). Within the beam, dispatch round-robins by `pc` so no
/// single closest state monopolizes the frontier; a path-depth (history length)
/// tiebreak supplies the "data/constraint" signal, and the front index is the
/// final deterministic tie-break.
///
/// **Bounded discard is deliberately omitted.** Per the greedy-trap memory the
/// beam defers rather than discards overflow, and there is no *total*-memory win
/// over BFS on the corpus; hard-dropping a path would also risk soundness. Wide
/// (`distance == u64::MAX`) states — blocks the CFG snapshot never mapped, i.e.
/// with no known route to the target — sort to the back and only enter the beam
/// when fewer than `beam_width` reachable states remain. That is the "re-add
/// safety valve": an unreachable state is never permanently dropped and `select`
/// never returns `None` while the deque is non-empty.
///
/// Opt-in only via `set_state_selection_directed`; never a default.
pub struct DirectedCfgDistance {
    /// One-time `addr -> distance-to-target` snapshot from the angr CFG.
    /// Immutable after construction — no runtime Python bounces.
    distances: HashMap<u64, u64>,
    /// Number of closest states forming the beam. Clamped to `>= 1` at
    /// construction; the Python setter defaults it to 2 so best-first never
    /// degrades to the greedy trap by accident.
    beam_width: usize,
    /// Per-`pc` dispatch counts for round-robin fairness within the beam.
    /// Behind a `Mutex` for interior mutability under the `&self` `select` hook
    /// while preserving `Send + Sync`.
    dispatched: Mutex<HashMap<u64, u64>>,
}

impl DirectedCfgDistance {
    /// Construct from a distance snapshot and beam width. `beam_width` is
    /// clamped to at least 1 (0 would make the beam empty); callers should pass
    /// `>= 2` to avoid the greedy trap.
    pub fn new(distances: HashMap<u64, u64>, beam_width: usize) -> Self {
        Self {
            distances,
            beam_width: beam_width.max(1),
            dispatched: Mutex::new(HashMap::new()),
        }
    }

    /// Distance-to-target for a state's next block. Unmapped blocks (no known
    /// route to the target) get `u64::MAX` so they sort behind every reachable
    /// state.
    fn distance(&self, state: &RustSimState) -> u64 {
        self.distances.get(&state.pc()).copied().unwrap_or(u64::MAX)
    }
}

impl SelectionPolicy for DirectedCfgDistance {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        let len = active.len();
        if len == 0 {
            return None;
        }
        // Distance for every active state (single pass; indices stay stable).
        let dists: Vec<u64> = active.iter().map(|st| self.distance(st)).collect();
        // The beam = the `beam_width` closest states by distance (front index
        // breaks distance ties so beam membership is deterministic).
        let mut ranked: Vec<usize> = (0..len).collect();
        ranked.sort_by_key(|&i| (dists[i], i));
        let beam = &ranked[..self.beam_width.min(len)];

        let mut dispatched = self
            .dispatched
            .lock()
            .expect("DirectedCfgDistance counts poisoned");
        // Within the beam: least-dispatched first (round-robin), then closest,
        // then deeper path (data/constraint tiebreak via history length), then
        // front index (deterministic).
        let pick = *beam
            .iter()
            .min_by_key(|&&i| {
                let count = dispatched.get(&active[i].pc()).copied().unwrap_or(0);
                (
                    count,
                    dists[i],
                    std::cmp::Reverse(active[i].history().len()),
                    i,
                )
            })
            .expect("beam is non-empty for a non-empty deque");
        let key = active[pick].pc();
        let state = active
            .remove(pick)
            .expect("index from non-empty deque is valid");
        *dispatched.entry(key).or_insert(0) += 1;
        Some(state)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "directed"
    }
}

/// Find-directed novelty/CFG-distance selection (angr-lnzcu): under
/// `num_find == 1`, dispatch the active state most likely to reach a find
/// target first. It fuses the two signals the a32jl policy family already
/// exposes inside the two-hook seam — a one-time `addr -> distance-to-target`
/// CFG snapshot (the [`DirectedCfgDistance`] steering signal) and a
/// self-contained seen-block novelty set (the [`CoverageGuided`] forward-progress
/// signal) — into a single find-first ordering.
///
/// **Why novelty is primary, distance secondary.** A num_find=1 run wants to
/// reach *a* find state as fast as possible, so it should never re-tread a block
/// while an un-dispatched block remains reachable: novel blocks rank ahead of
/// seen ones. Distance-to-find then steers *among the novel frontier* toward the
/// target. This ordering also sidesteps the greedy trap
/// (`ds-directed-search-greedy-trap`) without a beam: at a char-check fork the
/// correct- and wrong-branch successors share an identical CFG distance, but
/// both are novel, so both are dispatched (closest-first) before either block is
/// re-tread — best-first can never stall on the tie. A path that wanders away
/// from the target drifts into higher-distance / unmapped (`u64::MAX`) blocks and
/// sinks behind the still-advancing frontier, but is never permanently dropped
/// (`select` returns `None` only for an empty deque).
///
/// Distinct from [`DirectedCfgDistance`], which uses a fixed-width beam with
/// per-`pc` round-robin *fairness* (deliberately anti-greedy so no closest state
/// monopolizes a multi-find sweep). `FindDirected` is intentionally the opposite:
/// greedy toward the single find, with novelty — not fairness — as the anti-trap
/// mechanism. Opt-in only via `set_state_selection_find_directed`; never a
/// default.
pub struct FindDirected {
    /// One-time `addr -> distance-to-find` snapshot from the angr CFG.
    /// Immutable after construction — no runtime Python bounces.
    distances: HashMap<u64, u64>,
    /// Block addresses already dispatched. Behind a `Mutex` for interior
    /// mutability under the `&self` `select` hook while preserving `Send + Sync`
    /// so a parallel scheduler can share the `Arc`.
    seen: Mutex<HashSet<u64>>,
}

impl FindDirected {
    /// Construct from a distance-to-find snapshot with an empty seen-set.
    pub fn new(distances: HashMap<u64, u64>) -> Self {
        Self {
            distances,
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Distance-to-find for a state's next block. Unmapped blocks (no known
    /// route to the find target) get `u64::MAX` so they sink behind every
    /// reachable state.
    fn distance(&self, state: &RustSimState) -> u64 {
        self.distances.get(&state.pc()).copied().unwrap_or(u64::MAX)
    }
}

impl SelectionPolicy for FindDirected {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        if active.is_empty() {
            return None;
        }
        let mut seen = self.seen.lock().expect("FindDirected seen-set poisoned");
        // Novelty primary (0 = never-dispatched block, 1 = already seen),
        // distance-to-find secondary (closest steers the novel frontier), front
        // index the deterministic final tie-break.
        let pick = (0..active.len())
            .min_by_key(|&i| {
                let pc = active[i].pc();
                let novelty = u8::from(seen.contains(&pc));
                (novelty, self.distance(&active[i]), i)
            })
            .expect("index range from non-empty deque is non-empty");
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
        "find_directed"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoverageGuided, DirectedCfgDistance, FindDirected, LoopHeadRoundRobin, RandomState,
        SelectionPolicy,
    };
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

    // ---- DirectedCfgDistance (angr-a32jl.4) ----

    /// Drain a `DirectedCfgDistance` over states parked at `pcs` (distinct)
    /// against the `dist` snapshot and return the dispatch order as `pc`s.
    fn directed_drain(dist: &[(u64, u64)], pcs: &[u64], beam: usize) -> Vec<u64> {
        let _ctx = Context::thread_local();
        let policy = DirectedCfgDistance::new(dist.iter().copied().collect(), beam);
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
    fn test_directed_prefers_closest() {
        // With a beam wide enough to cover every state, round-robin counts all
        // start at 0, so the tie-break collapses to distance-ascending: the
        // state nearest the target is dispatched first.
        let dist = [(0x1000, 5), (0x2000, 1), (0x3000, 3)];
        assert_eq!(
            directed_drain(&dist, &[0x1000, 0x2000, 0x3000], 3),
            vec![0x2000, 0x3000, 0x1000],
        );
    }

    #[test]
    fn test_directed_beam_round_robin_fairness() {
        // Two states share pc 0x1000 (distance 1); a third sits far at 0x2000
        // (distance 5). beam_width=2 keeps both nearest states in the beam, and
        // round-robin fairness serves the fresh far bucket before re-serving the
        // already-dispatched 0x1000 bucket:
        //   Step 1: A0/A1 (count 0, d1) beat B (d5) -> front A0.  (0x1000 -> 1)
        //   Step 2: A1 bucket=1, B bucket=0 -> B despite d5 > d1. (0x2000 -> 1)
        //   Step 3: only A1 remains -> A1.
        let _ctx = Context::thread_local();
        let policy =
            DirectedCfgDistance::new([(0x1000u64, 1u64), (0x2000, 5)].into_iter().collect(), 2);
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let a0 = state_at(0x1000);
        let a1 = state_at(0x1000);
        let b = state_at(0x2000);
        let (a0id, a1id, bid) = (a0.state_id(), a1.state_id(), b.state_id());
        active.push_back(a0);
        active.push_back(a1);
        active.push_back(b);
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.state_id());
        }
        assert_eq!(
            order,
            vec![a0id, bid, a1id],
            "round-robin serves the fresh bucket before re-serving 0x1000",
        );
    }

    #[test]
    fn test_directed_unreachable_sorts_last_but_survives() {
        // 0x99 is absent from the snapshot (no known route to target => u64::MAX
        // distance). It sorts behind the reachable state yet is still dispatched
        // — the re-add safety valve never permanently drops a path.
        let dist = [(0x10, 3)];
        assert_eq!(directed_drain(&dist, &[0x10, 0x99], 2), vec![0x10, 0x99],);
    }

    #[test]
    fn test_directed_is_permutation() {
        let _ctx = Context::thread_local();
        let policy = DirectedCfgDistance::new(
            [(0x10u64, 1u64), (0x20, 2), (0x30, 3)]
                .into_iter()
                .collect(),
            2,
        );
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let ids: Vec<u64> = [0x10u64, 0x20, 0x30, 0x20, 0x10]
            .iter()
            .map(|&pc| {
                let st = state_at(pc);
                let id = st.state_id();
                active.push_back(st);
                id
            })
            .collect();
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(st.state_id());
        }
        order.sort_unstable();
        let mut want = ids.clone();
        want.sort_unstable();
        assert_eq!(order, want, "drain is a permutation: no drops or dupes");
    }

    #[test]
    fn test_directed_empty_is_none() {
        let _ctx = Context::thread_local();
        let policy = DirectedCfgDistance::new(HashMap::new(), 2);
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_directed_beam_width_clamped_to_one() {
        // beam_width 0 would make the beam empty; construction clamps to 1 so a
        // non-empty deque always yields a state.
        let dist = [(0x10, 1)];
        assert_eq!(directed_drain(&dist, &[0x10], 0), vec![0x10]);
    }

    #[test]
    fn test_directed_policy_name() {
        assert_eq!(
            DirectedCfgDistance::new(HashMap::new(), 2).name(),
            "directed"
        );
    }

    // ---- FindDirected (angr-lnzcu) ----

    /// Drain a `FindDirected` over states parked at `pcs` against the `dist`
    /// snapshot and return the dispatch order as `pc`s.
    fn find_directed_drain(dist: &[(u64, u64)], pcs: &[u64]) -> Vec<u64> {
        let _ctx = Context::thread_local();
        let policy = FindDirected::new(dist.iter().copied().collect());
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
    fn test_find_directed_novel_frontier_steers_by_distance() {
        // All three blocks are novel on the first pass, so novelty ties and
        // distance-to-find decides: the closest (0x2000, d1) goes first, then
        // 0x3000 (d3), then 0x1000 (d5).
        let dist = [(0x1000, 5), (0x2000, 1), (0x3000, 3)];
        assert_eq!(
            find_directed_drain(&dist, &[0x1000, 0x2000, 0x3000]),
            vec![0x2000, 0x3000, 0x1000],
        );
    }

    #[test]
    fn test_find_directed_novelty_beats_distance() {
        // Layout [0x1000(d5), 0x1000(d5), 0x2000(d1)]: the near 0x2000 is novel
        // and wins step 1. Step 2 both 0x1000s are still novel (0x2000 now seen)
        // so a 0x1000 goes despite its larger distance — a novel block outranks a
        // closer already-seen one. Step 3 the remaining seen 0x1000 drains.
        //   Step 1: 0x2000 novel d1 -> dispatch (seen {0x2000}).
        //   Step 2: both 0x1000 novel (rank 0) beat nothing else -> 0x1000.
        //   Step 3: last 0x1000 seen -> dispatch.
        let dist = [(0x1000, 5), (0x2000, 1)];
        assert_eq!(
            find_directed_drain(&dist, &[0x1000, 0x1000, 0x2000]),
            vec![0x2000, 0x1000, 0x1000],
        );
    }

    #[test]
    fn test_find_directed_char_check_fork_dispatches_both_branches() {
        // Greedy-trap shape: a char check forks two successors at an identical
        // CFG distance (both d2, both novel distinct blocks) plus a far sibling.
        // Novelty dispatches BOTH equal-distance branches before re-treading, so
        // best-first never stalls on the tie.
        //   Step 1: 0x10 & 0x11 novel d2 beat 0x20 d9; front 0x10 wins.
        //   Step 2: 0x11 still novel d2 -> dispatch (over far novel 0x20 d9).
        //   Step 3: only 0x20 remains -> dispatch.
        let dist = [(0x10, 2), (0x11, 2), (0x20, 9)];
        assert_eq!(
            find_directed_drain(&dist, &[0x10, 0x11, 0x20]),
            vec![0x10, 0x11, 0x20],
        );
    }

    #[test]
    fn test_find_directed_unmapped_sinks_but_survives() {
        // 0x99 is absent from the snapshot (u64::MAX distance). On the novel
        // frontier it sorts behind the mapped 0x10 yet is still dispatched — no
        // reachable-or-not path is permanently dropped.
        let dist = [(0x10, 3)];
        assert_eq!(find_directed_drain(&dist, &[0x99, 0x10]), vec![0x10, 0x99]);
    }

    #[test]
    fn test_find_directed_is_permutation() {
        let mut order = find_directed_drain(
            &[(0x10, 1), (0x20, 2), (0x30, 3)],
            &[0x10, 0x20, 0x10, 0x30, 0x20],
        );
        assert_eq!(order.len(), 5, "every inserted state must be dispatched");
        order.sort_unstable();
        assert_eq!(order, vec![0x10, 0x10, 0x20, 0x20, 0x30]);
    }

    #[test]
    fn test_find_directed_empty_is_none() {
        let _ctx = Context::thread_local();
        let policy = FindDirected::new(HashMap::new());
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_find_directed_policy_name() {
        assert_eq!(FindDirected::new(HashMap::new()).name(), "find_directed");
    }
}
