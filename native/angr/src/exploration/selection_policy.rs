//! Pluggable state-selection / scheduling policy seam (angr-a32jl.1).
//!
//! The exploration loop makes two ordering decisions about the `active`
//! stash on every step:
//!   1. *select* — which active state to step next (the pop side), and
//!   2. *on_fork* — where a freshly-forked successor lands in the active
//!      deque (the push side).
//!
//! Historically these were a single `use_lifo: bool` open-coded as
//! `pop_front`/`pop_back` in `run_loop_single.rs`, with an unconditional `push_back`
//! at every fork site. This trait factors that decision behind two hooks so
//! richer policies (random-path, coverage-guided, CFG-distance directed —
//! angr-a32jl.2+) can slot in without touching the run loop.
//!
//! The two built-ins reproduce the pre-refactor behavior exactly. `Fifo`
//! (BFS, the default) selects the front; `Lifo` (DFS) selects the back.
//! Both append new forks at the tail — matching the historical `push_back`
//! at every fork site — so step traces are byte-identical under either.
//!
//! # Order-determinism contract (angr-op0dn.10.5)
//!
//! Single-worker exploration is *order-deterministic*: the same binary, the
//! same seed states and the same policy must dispatch states in the same
//! sequence on every run. Two chokepoints decide that sequence, and both are
//! in this module's seam:
//!
//!   * **The select chokepoint** — `policy.select` is the only place the run
//!     loop (`run_loop_single.rs`) removes a state from `active`. Every built-in below
//!     picks an index into the `VecDeque` and calls `remove(idx)`, which
//!     preserves the relative order of the survivors. Where a policy ranks
//!     states it must terminate its key with the *front index* `i` so ties are
//!     broken by insertion order (`Fifo`/`Lifo` are trivially positional;
//!     `CoverageGuided`/`FindDirected`/`LoopHeadRoundRobin` front-scan;
//!     `DirectedCfgDistance` ranks by `(distance, i)`; `RandomSelection` draws from
//!     a seeded SplitMix64). No policy may leave a tie unresolved.
//!   * **The fork-insertion chokepoint** — `policy.on_fork` is the only place
//!     *any* state enters `active`, whether it is a run-loop successor
//!     (`helpers.rs::push_to_active_or_drop`) or a Python-API insertion/move
//!     (`helpers.rs::push_to_stash`, used by every `state_lifecycle.rs` entry
//!     point that takes a caller-supplied destination stash); both funnel into
//!     `StashManager::push_active` in `stash.rs`. Successors arrive in
//!     `forks_out` `Vec` order from `core_outcome_handlers.rs`, and every
//!     built-in appends with `push_back`, so the deque order is a pure function
//!     of the emission order. The mirror obligation is
//!     `policy.on_state_removed` on every departure from `active` that does not
//!     go through `policy.select`.
//!
//! **No hash-order may enter selection.** The per-policy `HashMap`/`HashSet`
//! fields (`CoverageGuided::seen`, `FindDirected::seen`,
//! `LoopHeadRoundRobin::{dispatched, key_cache}`, `DirectedCfgDistance::{distances,
//! dispatched}`) are *only ever indexed* (`get`/`contains`/`entry`) — never
//! iterated. Rust's `HashMap` seeds a fresh
//! `std::collections::hash_map::RandomState` hasher per instance (spelled in
//! full here because it is std's type, unrelated to this module's
//! `RandomSelection` policy),
//! so a single `for (k, v) in map` in a `select` path would make the dispatch
//! order vary between runs *within the same process*. The
//! `test_<policy>_selection_trace_deterministic` guards below drive a scripted
//! fork program through each policy three times over freshly-built maps and
//! assert an identical dispatch trace, which is what makes such a refactor fail
//! loudly. (`LoopHeadRoundRobin::bucket_key` does build a local `HashMap` of
//! history-address counts, but it iterates the history *slice*, not the map.)
//!
//! This contract is single-worker only. Parallel steal order is
//! design-nondeterministic; there the contract is set-equality of results, not
//! sequence equality (see `docs/advanced-topics/rust_parallel_design.rst`).
//!
//! # Panic policy (angr-9ke6b.212)
//!
//! Nothing in this module is reachable with untrusted input — a policy sees
//! only the run loop's own `VecDeque<RustSimState>` — and every surviving
//! `.expect` is one of exactly two invariant shapes:
//!
//!   * **Mutex poison guards** on the per-policy `Mutex` fields (`rng`, `seen`,
//!     `key_cache`, `dispatched`). The crate ships with `[profile.release]
//!     panic = "abort"` (workspace `Cargo.toml`, angr-1cue), so no thread can
//!     unwind out of a live `MutexGuard` to flag the lock; poison is
//!     unreachable in this build. Same argument as the
//!     [`scheduler`](super::scheduler) Panic policy, which spells it out in
//!     full.
//!   * **Index-after-`is_empty()` guards** — `active.remove(pick)` and the
//!     `min_by_key(..)` picks. Every `select` returns early on an empty deque,
//!     and each `pick` is an index *into* that deque produced by a `position` /
//!     `min_by_key` over `0..active.len()`, so both the range and the removal
//!     are in bounds.
//!
//! The index guards are deliberately **not** softened to `?`. `select` returns
//! `Option<RustSimState>`, and `None` from a non-empty deque is the run loop's
//! signal that the active stash is exhausted — it would end exploration early
//! and silently drop states. A loud panic on a broken index invariant is the
//! correct trade here (CLAUDE.md: never trade a loud panic for a silent wrong
//! answer).
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; each function holding
//! one of the guards above carries a narrow
//! `#[allow(clippy::expect_used, reason = ...)]` pointing back here.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::state::RustSimState;

/// The policy a freshly-constructed `RustExplorationManager` runs (BFS).
///
/// Exists so the scheduler tests can drive the *production* default instead of
/// naming a policy of their own and silently drifting from it — the whole
/// `scheduler_worker_tests.rs` suite pinned `Lifo` while production had
/// defaulted to `Fifo` for as long as the seam existed, which is what let the
/// hardcoded-`pop_front` offload bug live untested (angr-03vl4.19).
pub(crate) fn default_policy() -> Arc<dyn SelectionPolicy> {
    Arc::new(Fifo)
}

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

    /// Choose and remove the state this policy is *least* likely to dispatch
    /// next — the pick `scheduler_worker.rs::offload_one` ships to another
    /// worker across a full Z3 detach/reattach round-trip. Offloading the
    /// state `select` was about to hand back for free is pure serde cost (the
    /// same waste angr-faorh / angr-8shhe removed from the halving shed), so
    /// this is the offload-side mirror of `select`, not a second `select`.
    ///
    /// The default pops the **back**, which is the coldest end for every
    /// built-in but [`Lifo`]: `Fifo` selects the front outright, and the
    /// ranking policies (`CoverageGuided`, `FindDirected`,
    /// `LoopHeadRoundRobin`, `DirectedCfgDistance`, and the front-biased
    /// `min_by_key` in `RandomSelection`'s tie order) all break ties toward the
    /// front per this module's order-determinism contract. `Lifo` overrides.
    /// A future policy whose hot end is the tail must override too.
    fn select_for_offload(&self, local: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        local.pop_back()
    }

    /// Notify the policy that `state_id` has left the local/active deque
    /// through a path other than `select` — namely cross-worker migration
    /// (`scheduler_worker.rs::offload_surplus` detaches straight from the
    /// worker's local `VecDeque` via `select_for_offload`, bypassing `select`
    /// entirely). Default no-op; only policies that memoize a per-`state_id`
    /// side table need to override this to evict the entry. Without this hook
    /// a migrated state's memo entry is never revisited by `select` (a
    /// stolen-back state re-enters via `dispatch_next`'s steal branch
    /// directly, not through `on_fork`), so it would otherwise leak for the
    /// life of the policy (angr-ua7fd). Not a correctness issue on its own —
    /// `state_id` is never reused — just an unbounded-over-session-lifetime
    /// memory leak in the memo table.
    fn on_state_removed(&self, _state_id: u64) {}
}

/// Breadth-first (FIFO queue): step the oldest state first, append new forks
/// at the tail. The default policy — historical `use_lifo == false`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Fifo;

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
pub(crate) struct Lifo;

impl SelectionPolicy for Lifo {
    #[inline]
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        active.pop_back()
    }

    #[inline]
    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    /// The only built-in that dispatches from the tail, so its coldest end is
    /// the front — the mirror image of the trait's default.
    #[inline]
    fn select_for_offload(&self, local: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        local.pop_front()
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
pub(crate) struct RandomSelection {
    /// SplitMix64 state behind a `Mutex` for interior mutability under the
    /// `&self` `select` hook. `Mutex` (not `Cell`) keeps the `Send + Sync`
    /// supertrait bound so a parallel scheduler can share the policy `Arc`.
    rng: Mutex<u64>,
}

impl RandomSelection {
    /// Construct with an explicit seed. Any `u64` (including 0) is a valid,
    /// deterministic seed — SplitMix64 does not degenerate at zero.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(seed),
        }
    }

    /// One SplitMix64 draw. Fast, seekable, and dependency-free — no `rand`
    /// crate pulled in for a prototype policy.
    #[allow(
        clippy::expect_used,
        reason = "`RandomSelection::rng` poison guard: poison requires a thread to unwind out of a live `MutexGuard`, which `panic = \"abort\"` forecloses — see the module Panic policy header"
    )]
    fn next_u64(&self) -> u64 {
        let mut state = self.rng.lock().expect("RandomSelection rng poisoned");
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        // Release the RNG mutex before the pure-arithmetic mix — nothing below
        // touches the shared state, so holding the lock across it only widens
        // the contention window between workers.
        drop(state);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl SelectionPolicy for RandomSelection {
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
pub(crate) struct CoverageGuided {
    /// Block addresses already dispatched. Behind a `Mutex` for interior
    /// mutability under the `&self` `select` hook while preserving the
    /// `Send + Sync` supertrait so a parallel scheduler can share the `Arc`.
    seen: Mutex<HashSet<u64>>,
}

impl CoverageGuided {
    /// Construct with an empty seen-set.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl SelectionPolicy for CoverageGuided {
    // `seen` is held across the whole front-scan (contains) and the post-remove
    // insert — the guard cannot be tightened without dropping correctness.
    #[allow(
        clippy::expect_used,
        reason = "`CoverageGuided::seen` poison guard plus the index-after-`is_empty()` guards: the poison state is unreachable under `panic = \"abort\"`, and every `pick` is an index into the same non-empty deque the early return above already checked — see the module Panic policy header"
    )]
    #[allow(clippy::significant_drop_tightening)]
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
pub(crate) struct LoopHeadRoundRobin {
    /// Per-bucket dispatch counts. Behind a `Mutex` for interior mutability
    /// under the `&self` `select` hook while preserving `Send + Sync` so a
    /// parallel scheduler can share the `Arc`.
    dispatched: Mutex<HashMap<u64, u64>>,
    /// Memoized `state_id -> bucket_key` so `select` recomputes a state's key
    /// (a full history scan) at most once while it sits in `active`, not once
    /// per active state per step. A state's key changes only when the state is
    /// stepped, and a state is stepped only after `select` *removes* it — so a
    /// state still in the deque has a fixed key, and the dispatched state's
    /// entry is evicted on removal (children re-enter with fresh `state_id`s).
    /// This bounds the cache to roughly `active.len()` and turns select from
    /// O(active x history) into O(active) after each state's first sighting.
    /// Only ever indexed (`get`/`insert`/`remove`) — never iterated — so it
    /// respects the no-hash-order-in-selection contract above.
    key_cache: Mutex<HashMap<u64, u64>>,
    /// Test-only tally of `bucket_key` computations (cache misses), used to
    /// assert the memoization actually collapses a full drain from
    /// O(active^2) key computations down to O(active).
    #[cfg(test)]
    key_computations: std::sync::atomic::AtomicU64,
}

impl LoopHeadRoundRobin {
    /// Construct with empty per-bucket dispatch counts.
    pub(crate) fn new() -> Self {
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
    // Both guards span the full deque scan (cache read/insert per state,
    // dispatched read for the least-served pick) plus the post-remove eviction
    // and count bump — no tighter scope is correct here.
    #[allow(
        clippy::expect_used,
        reason = "`LoopHeadRoundRobin::{key_cache, dispatched}` poison guard plus the index-after-`is_empty()` guards: the poison state is unreachable under `panic = \"abort\"`, and every `pick` is an index into the same non-empty deque the early return above already checked — see the module Panic policy header"
    )]
    #[allow(clippy::significant_drop_tightening)]
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        if active.is_empty() {
            return None;
        }
        let mut cache = self
            .key_cache
            .lock()
            .expect("LoopHeadRoundRobin key cache poisoned");
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
            // A state's bucket key is fixed while it sits in the deque, so
            // compute it at most once and memoize on `state_id`.
            let key = match cache.get(&st.state_id()) {
                Some(&k) => k,
                None => {
                    let k = Self::bucket_key(st);
                    #[cfg(test)]
                    self.key_computations
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cache.insert(st.state_id(), k);
                    k
                }
            };
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
        // The dispatched state is stepped next (mutating its history) and its
        // successors re-enter with fresh `state_id`s, so its cached key is now
        // dead — evict to keep the cache bounded to ~active.len().
        cache.remove(&state.state_id());
        *dispatched.entry(best_key).or_insert(0) += 1;
        Some(state)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        active.push_back(state);
    }

    fn name(&self) -> &'static str {
        "loop_head"
    }

    // Evict the migrated state's memoized key so `key_cache` cannot outlive
    // the state when it is offloaded to another worker instead of dispatched
    // through `select` (angr-ua7fd). A stolen-back state gets a cache miss on
    // its next `select` scan and recomputes its key — correct, just one extra
    // `bucket_key` call, exactly like a state that was never cached yet.
    #[allow(
        clippy::expect_used,
        reason = "`LoopHeadRoundRobin::key_cache` poison guard: poison requires a thread to unwind out of a live `MutexGuard`, which `panic = \"abort\"` forecloses — see the module Panic policy header"
    )]
    fn on_state_removed(&self, state_id: u64) {
        self.key_cache
            .lock()
            .expect("LoopHeadRoundRobin key cache poisoned")
            .remove(&state_id);
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
/// Distance-to-target for a state's next block against a fixed CFG snapshot.
/// Unmapped blocks (no known route to the target) get `u64::MAX` so they sort
/// behind every reachable state. Shared by both CFG-distance policies
/// ([`DirectedCfgDistance`] and [`FindDirected`]) since the lookup is identical.
fn distance_in(distances: &HashMap<u64, u64>, state: &RustSimState) -> u64 {
    distances.get(&state.pc()).copied().unwrap_or(u64::MAX)
}

pub(crate) struct DirectedCfgDistance {
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
    pub(crate) fn new(distances: HashMap<u64, u64>, beam_width: usize) -> Self {
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
        distance_in(&self.distances, state)
    }
}

impl SelectionPolicy for DirectedCfgDistance {
    // `dispatched` is held across the beam's min-by-key scan (reads the count
    // per beam member) and the post-remove count bump — no tighter scope works.
    #[allow(clippy::significant_drop_tightening)]
    #[allow(
        clippy::expect_used,
        reason = "`DirectedCfgDistance::dispatched` poison guard plus the index-after-`is_empty()` guards: the poison state is unreachable under `panic = \"abort\"`, `beam` is a non-empty prefix of `ranked` (`beam_width.min(len)` with `len > 0`), and `pick` indexes the same deque the `len == 0` early return above already checked — see the module Panic policy header"
    )]
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        let len = active.len();
        if len == 0 {
            return None;
        }
        // Distance for every active state (single pass; indices stay stable).
        let dists: Vec<u64> = active.iter().map(|st| self.distance(st)).collect();
        // The beam = the `beam_width` closest states by distance (front index
        // breaks distance ties so beam membership is deterministic).
        //
        // Partial selection, not a full sort (angr-9ke6b.60): only *which*
        // `beam_len` indices land in the beam matters — the pick below is a
        // `min_by_key` over the whole beam, and its key ends in the front index
        // `i`, so it is a total order and the winner is independent of the
        // beam's internal order. `select_nth_unstable_by_key` partitions in
        // O(n) instead of sorting all `n` in O(n log n), which matters on the
        // wide active frontier this beam design targets.
        let beam_len = self.beam_width.min(len);
        let mut ranked: Vec<usize> = (0..len).collect();
        if beam_len < len {
            ranked.select_nth_unstable_by_key(beam_len, |&i| (dists[i], i));
        }
        let beam = &ranked[..beam_len];

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
pub(crate) struct FindDirected {
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
    pub(crate) fn new(distances: HashMap<u64, u64>) -> Self {
        Self {
            distances,
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Distance-to-find for a state's next block. Unmapped blocks (no known
    /// route to the find target) get `u64::MAX` so they sink behind every
    /// reachable state.
    fn distance(&self, state: &RustSimState) -> u64 {
        distance_in(&self.distances, state)
    }
}

impl SelectionPolicy for FindDirected {
    // `seen` is held across the novelty scan (contains) and the post-remove
    // insert — both halves need the lock, so it cannot be tightened.
    #[allow(
        clippy::expect_used,
        reason = "`FindDirected::seen` poison guard plus the index-after-`is_empty()` guards: the poison state is unreachable under `panic = \"abort\"`, and every `pick` is an index into the same non-empty deque the early return above already checked — see the module Panic policy header"
    )]
    #[allow(clippy::significant_drop_tightening)]
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

#[cfg(all(test, feature = "vex-engine-z3"))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` above overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests {
    use super::{
        CoverageGuided, DirectedCfgDistance, Fifo, FindDirected, Lifo, LoopHeadRoundRobin,
        RandomSelection, SelectionPolicy,
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

    /// Run a full drain under `RandomSelection(seed)` over `n` fresh states and
    /// return the dispatch order expressed as *insertion indices* (0..n). Two
    /// runs with the same seed must return the same index sequence even though
    /// absolute `state_id`s differ between runs.
    fn drain_order(seed: u64, n: usize) -> Vec<usize> {
        let _ctx = Context::thread_local();
        let policy = RandomSelection::new(seed);
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
        let policy = RandomSelection::new(7);
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(policy.select(&mut active).is_none());
    }

    #[test]
    fn test_random_policy_name() {
        assert_eq!(RandomSelection::new(0).name(), "random");
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
    fn test_loop_head_key_cache_computes_each_key_once_per_drain() {
        use std::sync::atomic::Ordering;
        // A full drain of N states must compute bucket_key exactly N times —
        // once per state on its first sighting — not the O(N^2) recompute the
        // un-memoized select did (N + (N-1) + ... + 1). Each state carries a
        // multi-address history so an un-cached scan would be visibly costly.
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        const N: u64 = 8;
        for i in 0..N {
            active.push_back(state_with_history(0x100 + i, &[0xAA, 0xBB, 0xAA, i]));
        }
        let mut drained = 0u64;
        while policy.select(&mut active).is_some() {
            drained += 1;
        }
        assert_eq!(drained, N, "every state must be dispatched");
        assert_eq!(
            policy.key_computations.load(Ordering::Relaxed),
            N,
            "memoized select must compute each state's key exactly once, not once per select",
        );
        // The dispatched-state eviction leaves the cache empty after a full drain.
        assert!(
            policy.key_cache.lock().expect("cache poisoned").is_empty(),
            "every dispatched state's cache entry must be evicted",
        );
    }

    #[test]
    fn test_loop_head_on_state_removed_evicts_key_cache_entry() {
        // Reproduces the offload+steal path (angr-ua7fd): select() scans past a
        // state without picking it, memoizing its bucket key in key_cache. In
        // the parallel scheduler that state can then leave `active`/`local` via
        // `scheduler_worker.rs::offload_surplus`'s pop_front — a path that never
        // goes through select() again (a stolen-back state is reattached
        // directly by dispatch_next's steal branch, bypassing on_fork). Without
        // the eviction hook the memo entry would never be revisited and would
        // leak for the life of the policy. `on_state_removed` is the hook
        // offload_surplus calls to close that gap.
        let _ctx = Context::thread_local();
        let policy = LoopHeadRoundRobin::new();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        // Two states: only one will be picked by select(), leaving the other's
        // key_cache entry behind exactly as a scan-but-not-pick would.
        let picked = state_at(0x1000);
        let left_behind = state_at(0x2000);
        let left_behind_id = left_behind.state_id();
        active.push_back(picked);
        active.push_back(left_behind);

        let dispatched = policy.select(&mut active).expect("non-empty active");
        assert_eq!(dispatched.pc(), 0x1000, "front state wins the tie-break");
        assert_eq!(active.len(), 1, "the scanned-but-not-picked state remains");
        assert!(
            policy
                .key_cache
                .lock()
                .expect("cache poisoned")
                .contains_key(&left_behind_id),
            "select() must have memoized the scanned state's key",
        );

        // Simulate offload_surplus detaching the remaining state directly from
        // the local deque (pop_front), bypassing select() entirely — the exact
        // path the bug report describes.
        let offloaded = active.pop_front().expect("the left-behind state");
        assert_eq!(offloaded.state_id(), left_behind_id);
        policy.on_state_removed(offloaded.state_id());

        assert!(
            !policy
                .key_cache
                .lock()
                .expect("cache poisoned")
                .contains_key(&left_behind_id),
            "on_state_removed must evict the offloaded state's memo entry",
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

    /// Reference beam pick built on a **full sort** — the pre-angr-9ke6b.60
    /// implementation of `DirectedCfgDistance::select`, kept here as the oracle
    /// the partial `select_nth_unstable_by_key` selection must agree with.
    /// Returns the dispatch order as *insertion indices* into `pcs`.
    ///
    /// The real `select`'s `Reverse(history.len())` tiebreak is omitted: every
    /// `state_at` state has an empty history, so that term is constant across
    /// the beam and cannot change the pick.
    fn directed_drain_full_sort_oracle(
        dist: &HashMap<u64, u64>,
        pcs: &[u64],
        beam_width: usize,
    ) -> Vec<usize> {
        let mut active: Vec<(usize, u64)> = pcs.iter().copied().enumerate().collect();
        let mut dispatched: HashMap<u64, u64> = HashMap::new();
        let mut order = Vec::new();
        while !active.is_empty() {
            let len = active.len();
            let dists: Vec<u64> = active
                .iter()
                .map(|&(_, pc)| dist.get(&pc).copied().unwrap_or(u64::MAX))
                .collect();
            let mut ranked: Vec<usize> = (0..len).collect();
            ranked.sort_by_key(|&i| (dists[i], i));
            let beam = &ranked[..beam_width.max(1).min(len)];
            let pick = *beam
                .iter()
                .min_by_key(|&&i| {
                    let count = dispatched.get(&active[i].1).copied().unwrap_or(0);
                    (count, dists[i], i)
                })
                .unwrap();
            let (id, pc) = active.remove(pick);
            *dispatched.entry(pc).or_insert(0) += 1;
            order.push(id);
        }
        order
    }

    #[test]
    fn test_directed_wide_frontier_matches_full_sort() {
        // angr-9ke6b.60: `select` partitions with `select_nth_unstable_by_key`
        // instead of sorting the whole ranked vec. Partial selection fixes only
        // *which* indices are in the beam, not their internal order — so this
        // drives a wide frontier (200 states, heavy distance ties, a slice of
        // unmapped/`u64::MAX` states) through the real policy and asserts the
        // dispatch order is identical to the full-sort oracle above.
        let _ctx = Context::thread_local();
        // 40 distinct pcs, 5 states each => every distance value is a 5+-way
        // tie, which is exactly where an unstable partition could disagree with
        // a stable sort if beam membership were not uniquely determined.
        let pcs: Vec<u64> = (0..200u64).map(|i| 0x1000 + (i * 7 % 40) * 0x10).collect();
        // Map only 30 of the 40 pcs; the other 10 fall through to u64::MAX.
        let dist: HashMap<u64, u64> = (0..30u64).map(|j| (0x1000 + j * 0x10, j % 6)).collect();

        for beam_width in [1usize, 2, 5, 200, 500] {
            let policy = DirectedCfgDistance::new(dist.clone(), beam_width);
            let mut active: VecDeque<RustSimState> = VecDeque::new();
            let ids: Vec<u64> = pcs
                .iter()
                .map(|&pc| {
                    let st = state_at(pc);
                    let id = st.state_id();
                    active.push_back(st);
                    id
                })
                .collect();
            let mut got = Vec::new();
            while let Some(st) = policy.select(&mut active) {
                let id = st.state_id();
                got.push(ids.iter().position(|&x| x == id).expect("dispatched id"));
            }
            assert_eq!(
                got,
                directed_drain_full_sort_oracle(&dist, &pcs, beam_width),
                "partial beam selection diverged from the full-sort oracle at beam_width={beam_width}",
            );
        }
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

    // ---- Order-determinism regression guards (angr-op0dn.10.5) ----

    /// The built-in policies, by `name()`. A new policy added to this module
    /// belongs here too, so it inherits the determinism guard below.
    const BUILT_IN_POLICIES: [&str; 7] = [
        "fifo",
        "lifo",
        "random",
        "coverage",
        "loop_head",
        "directed",
        "find_directed",
    ];

    fn build_policy(name: &str, distances: &HashMap<u64, u64>) -> Box<dyn SelectionPolicy> {
        match name {
            "fifo" => Box::new(Fifo),
            "lifo" => Box::new(Lifo),
            "random" => Box::new(RandomSelection::new(0x0D15_EA5E)),
            "coverage" => Box::new(CoverageGuided::new()),
            "loop_head" => Box::new(LoopHeadRoundRobin::new()),
            "directed" => Box::new(DirectedCfgDistance::new(distances.clone(), 2)),
            "find_directed" => Box::new(FindDirected::new(distances.clone())),
            other => panic!("unknown policy {other}"),
        }
    }

    /// A CFG-distance snapshot covering the blocks the scripted fork program
    /// below reaches. Blocks it omits fall through to `u64::MAX` (unmapped),
    /// which is itself part of the ordering contract.
    fn scripted_distances() -> HashMap<u64, u64> {
        [
            (0x1000, 8),
            (0x1010, 4),
            (0x1020, 2),
            (0x2000, 6),
            (0x2010, 3),
            (0x3000, 6),
            (0x3010, 0),
        ]
        .into_iter()
        .collect()
    }

    /// Drive a scripted exploration through both seam hooks and return the
    /// dispatch trace as *insertion labels* (`0, 1, 2, …` in the order states
    /// entered the deque). Absolute `state_id`s differ between runs, so labels —
    /// not ids — are what two runs are compared on: an identical label trace
    /// means both runs selected and forked in exactly the same sequence.
    ///
    /// The fork program is a deterministic function of the dispatched state: a
    /// state shallower than `DEPTH` emits two successors, one at a fresh block
    /// (`pc + 0x10`) and one re-treading `0x1000`, so the novelty seen-sets and
    /// the loop-head buckets both observe repeats. Each successor inherits the
    /// parent's history plus the parent's `pc`.
    fn scripted_trace(policy: &dyn SelectionPolicy) -> Vec<usize> {
        const DEPTH: usize = 2;
        const MAX_DISPATCH: usize = 64;
        let _ctx = Context::thread_local();

        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let mut label_of: HashMap<u64, usize> = HashMap::new();
        let mut next_label = 0usize;

        for &pc in &[0x1000u64, 0x1000, 0x2000, 0x3000] {
            let st = state_at(pc);
            label_of.insert(st.state_id(), next_label);
            next_label += 1;
            active.push_back(st);
        }

        let mut trace = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            trace.push(label_of[&st.state_id()]);
            assert!(
                trace.len() <= MAX_DISPATCH,
                "scripted fork program diverged"
            );
            let mut child_history: Vec<u64> = st.history().iter().copied().collect();
            if child_history.len() >= DEPTH {
                continue;
            }
            child_history.push(st.pc());
            for &child_pc in &[st.pc() + 0x10, 0x1000] {
                let child = state_with_history(child_pc, &child_history);
                label_of.insert(child.state_id(), next_label);
                next_label += 1;
                policy.on_fork(&mut active, child);
            }
        }
        trace
    }

    /// Every built-in policy must dispatch the scripted fork program in the same
    /// order on every run. Each repeat rebuilds the policy, so its `seen` /
    /// `dispatched` maps get a fresh per-instance hasher seed: if any `select`
    /// path ever started *iterating* one of those maps instead of indexing it,
    /// hash order would leak into selection and the repeats would diverge here.
    #[test]
    fn test_selection_trace_deterministic_across_runs() {
        let distances = scripted_distances();
        for name in BUILT_IN_POLICIES {
            let baseline = scripted_trace(build_policy(name, &distances).as_ref());
            // 4 seeds + 8 children + 16 grandchildren, all dispatched, none dropped.
            assert_eq!(baseline.len(), 28, "policy {name} must drain every state");
            let mut labels = baseline.clone();
            labels.sort_unstable();
            assert_eq!(
                labels,
                (0..28).collect::<Vec<_>>(),
                "policy {name} dispatch trace must be a permutation of the insertion set",
            );
            for repeat in 1..3 {
                assert_eq!(
                    scripted_trace(build_policy(name, &distances).as_ref()),
                    baseline,
                    "policy {name} changed its dispatch order on repeat {repeat}",
                );
            }
        }
    }

    /// The guard above is only meaningful if the policies actually disagree —
    /// otherwise an all-FIFO regression would pass it unnoticed.
    #[test]
    fn test_selection_traces_differ_across_policies() {
        let distances = scripted_distances();
        let fifo = scripted_trace(build_policy("fifo", &distances).as_ref());
        for name in BUILT_IN_POLICIES.iter().filter(|n| **n != "fifo") {
            assert_ne!(
                scripted_trace(build_policy(name, &distances).as_ref()),
                fifo,
                "policy {name} degenerated to the FIFO dispatch order",
            );
        }
    }

    /// The policy list the guards iterate must stay in sync with what each
    /// policy reports as its own name.
    #[test]
    fn test_built_in_policy_names_match() {
        let distances = scripted_distances();
        for name in BUILT_IN_POLICIES {
            assert_eq!(build_policy(name, &distances).name(), name);
        }
    }

    // -- Fifo / Lifo -------------------------------------------------------
    //
    // The two order-only policies are the reference the tests above compare
    // against ("degenerated to the FIFO dispatch order"), and `Lifo` is what the
    // scheduler's worker-local frontier runs by default — so their contracts are
    // pinned directly rather than only implied by the guards.

    /// Drain `n` fresh states through `policy`, returning the dispatch order as
    /// insertion indices. Forks are inserted through `on_fork`, exactly as
    /// `absorb_continues` does.
    fn order_only_drain(policy: &dyn SelectionPolicy, n: usize) -> Vec<usize> {
        let _ctx = Context::thread_local();
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
        for i in 0..n {
            let st = RustSimState::new("amd64").unwrap();
            id_to_idx.insert(st.state_id(), i);
            policy.on_fork(&mut active, st);
        }
        let mut order = Vec::new();
        while let Some(st) = policy.select(&mut active) {
            order.push(id_to_idx[&st.state_id()]);
        }
        order
    }

    #[test]
    fn test_fifo_is_breadth_first() {
        assert_eq!(
            order_only_drain(&Fifo, 5),
            vec![0, 1, 2, 3, 4],
            "Fifo dispatches the oldest state first",
        );
    }

    #[test]
    fn test_lifo_is_depth_first() {
        assert_eq!(
            order_only_drain(&Lifo, 5),
            vec![4, 3, 2, 1, 0],
            "Lifo dispatches the most recent fork first",
        );
    }

    // Both append at the tail, so a fork lands behind the existing frontier for
    // `Fifo` and in front of it for `Lifo`. This is the property `dispatch_next`
    // relies on to keep the freshest child hot in the Z3 context.
    #[test]
    fn test_on_fork_appends_at_the_tail_for_both() {
        let _ctx = Context::thread_local();
        for policy in [&Fifo as &dyn SelectionPolicy, &Lifo] {
            let mut active: VecDeque<RustSimState> = VecDeque::new();
            let first = RustSimState::new("amd64").unwrap();
            let first_id = first.state_id();
            active.push_back(first);

            let child = RustSimState::new("amd64").unwrap();
            let child_id = child.state_id();
            policy.on_fork(&mut active, child);

            assert_eq!(active.len(), 2, "{}: on_fork enqueues", policy.name());
            assert_eq!(
                active.back().unwrap().state_id(),
                child_id,
                "{}: the fork lands at the tail",
                policy.name(),
            );
            assert_eq!(
                active.front().unwrap().state_id(),
                first_id,
                "{}: the existing frontier is not reordered",
                policy.name(),
            );
        }
    }

    #[test]
    fn test_order_only_policies_empty_is_none() {
        let mut active: VecDeque<RustSimState> = VecDeque::new();
        assert!(Fifo.select(&mut active).is_none());
        assert!(Lifo.select(&mut active).is_none());
    }

    #[test]
    fn test_order_only_policy_names() {
        assert_eq!(Fifo.name(), "fifo");
        assert_eq!(Lifo.name(), "lifo");
    }
}
