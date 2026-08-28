// Tests for exploration/selection_policy.rs (split out of the inline
// `#[cfg(test)] mod tests` alongside its scheduler-fileset siblings,
// angr-5mnx3.21).

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
