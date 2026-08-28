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
    /// ranking policies all break ties toward the front per this module's
    /// order-determinism contract — `CoverageGuided` and `LoopHeadRoundRobin`
    /// by front-scanning (`position` / a strict-`<` loop), `FindDirected` and
    /// `DirectedCfgDistance` via a `min_by_key` whose key *ends* in the front
    /// index. `RandomSelection` is the one built-in with no hot end at all —
    /// `select` draws a uniform index, so the back is as good an offload pick
    /// as any and it keeps the default for lack of a better one, not because
    /// the tail is cold. `Lifo` overrides. A future policy whose hot end is the
    /// tail must override too.
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

test_submod!(z3 "selection_policy_tests.rs" => tests);
