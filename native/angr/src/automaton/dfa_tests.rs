// Unit tests for dfa.rs (DFA construction, transitions, acceptance).
// Extracted from the inline mod tests block to keep the parent module focused.

use super::*;

#[test]
fn test_dfa_basic() {
    let mut dfa = DFA::new();
    let s0 = dfa.add_state();
    let s1 = dfa.add_state();
    let s2 = dfa.add_state();

    dfa.set_start_state(s0);
    dfa.add_final_state(s2);
    dfa.add_transition(s0, 0, s1);
    dfa.add_transition(s1, 1, s2);

    assert_eq!(dfa.num_states(), 3);
    assert_eq!(dfa.start_state(), Some(0));
    assert!(!dfa.is_empty());
}

#[test]
fn test_dfa_minimization() {
    // Create a DFA with equivalent states:
    // 0 -a-> 1 -b-> 3(final)
    // 0 -a-> 2 -b-> 4(final)
    // States 1 and 2 should be merged, as should 3 and 4

    let mut dfa = DFA::new();
    for _ in 0..5 {
        dfa.add_state();
    }

    dfa.set_start_state(0);
    dfa.add_final_state(3);
    dfa.add_final_state(4);

    dfa.add_transition(0, 0, 1); // 'a' = 0
    dfa.add_transition(0, 1, 2); // 'b' = 1 (different path to equivalent states)
    dfa.add_transition(1, 1, 3);
    dfa.add_transition(2, 1, 4);

    let minimized = dfa.minimize();

    // States 1 and 2 collapse into one partition, as do finals 3 and 4, leaving
    // exactly {0}, {1,2}, {3,4} -> 3 states. `<= num_states()` was vacuously
    // true even for a no-op or under-merging minimize(); pin the exact count so
    // a regression that stopped merging (giving 4-5 states) is caught.
    assert_eq!(
        minimized.num_states(),
        3,
        "minimize() should merge 1&2 and 3&4, yielding 3 states"
    );
    // The two equivalent finals (3 and 4) must collapse to a single final state.
    assert_eq!(
        minimized.final_states().len(),
        1,
        "equivalent final states 3 and 4 should merge into one"
    );
    assert!(!minimized.is_empty());
}

#[test]
fn test_overwriting_transition_retires_stale_reverse_edge() {
    // 0 -a-> 1, then re-add 0 -a-> 2. State 0 must no longer be reported as a
    // predecessor of 1 on 'a', or Hopcroft refinement in minimize() would split
    // partitions on an edge that does not exist.
    let mut dfa = DFA::new();
    for _ in 0..3 {
        dfa.add_state();
    }
    dfa.set_start_state(0);

    dfa.add_transition(0, 0, 1);
    assert_eq!(dfa.transition(0, 0), Some(1));
    assert!(
        dfa.find_predecessors(&StateSet::singleton(1, 3), 0)
            .contains(0)
    );

    dfa.add_transition(0, 0, 2);
    assert_eq!(dfa.transition(0, 0), Some(2));
    assert!(
        !dfa.find_predecessors(&StateSet::singleton(1, 3), 0)
            .contains(0),
        "stale reverse edge to the overwritten destination survived"
    );
    assert!(
        dfa.find_predecessors(&StateSet::singleton(2, 3), 0)
            .contains(0)
    );

    // Re-adding the same destination is idempotent, and other sources sharing
    // the old destination are untouched.
    dfa.add_transition(1, 0, 1);
    dfa.add_transition(0, 0, 2);
    assert!(
        dfa.find_predecessors(&StateSet::singleton(1, 3), 0)
            .contains(1)
    );
    assert_eq!(
        dfa.find_predecessors(&StateSet::singleton(2, 3), 0).len(),
        1
    );
}

/// Run `word` through `dfa` from its start state; a missing transition (or a
/// DFA with no start state) rejects.
fn accepts(dfa: &DFA, word: &[SymbolId]) -> bool {
    let Some(mut current) = dfa.start_state() else {
        return false;
    };
    for &symbol in word {
        match dfa.transition(current, symbol) {
            Some(next) => current = next,
            None => return false,
        }
    }
    dfa.final_states().contains(current)
}

/// Every word over `alphabet` of length `0..=max_len`, shortest first.
fn words_up_to(alphabet: &[SymbolId], max_len: usize) -> Vec<Vec<SymbolId>> {
    let mut all = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for word in &frontier {
            for &symbol in alphabet {
                let mut extended = word.clone();
                extended.push(symbol);
                next.push(extended);
            }
        }
        all.extend(next.iter().cloned());
        frontier = next;
    }
    all
}

/// Assert two DFAs accept exactly the same words up to `max_len`.
///
/// Cheap stand-in for a product-construction equivalence check: minimize() must
/// preserve the language, not merely shrink the state count.
fn assert_same_language(a: &DFA, b: &DFA, alphabet: &[SymbolId], max_len: usize) {
    for word in words_up_to(alphabet, max_len) {
        assert_eq!(
            accepts(a, &word),
            accepts(b, &word),
            "minimization changed the language on word {word:?}"
        );
    }
}

#[test]
fn test_minimize_multiple_refinement_rounds() {
    // Two isomorphic 3-step chains joined at the start state; L = { w : |w| >= 3 }.
    //
    //   0 -a-> 1 -{a,b}-> 2 -{a,b}-> 3(final) -{a,b}-> 3
    //   0 -b-> 4 -{a,b}-> 5 -{a,b}-> 6(final) -{a,b}-> 6
    //
    // Equivalence classes are {0}, {1,4}, {2,5}, {3,6} -- 4 of them, so the
    // single initial final/non-final split cannot be enough. The refinement
    // needs at least two rounds: round 1 peels {2,5} (the a-predecessors of the
    // finals) off the non-final block, and only then does that block's own
    // worklist entry peel off {0}.
    let mut dfa = DFA::new();
    for _ in 0..7 {
        dfa.add_state();
    }
    dfa.set_start_state(0);
    dfa.add_final_state(3);
    dfa.add_final_state(6);

    dfa.add_transition(0, 0, 1);
    dfa.add_transition(0, 1, 4);
    for (from, to) in [(1, 2), (2, 3), (3, 3), (4, 5), (5, 6), (6, 6)] {
        dfa.add_transition(from, 0, to);
        dfa.add_transition(from, 1, to);
    }

    let minimized = dfa.minimize();

    assert_eq!(
        minimized.num_states(),
        4,
        "the two chains must collapse pairwise into {{0}}, {{1,4}}, {{2,5}}, {{3,6}}"
    );
    assert_eq!(
        minimized.final_states().len(),
        1,
        "equivalent finals 3 and 6 should merge"
    );
    assert_same_language(&dfa, &minimized, &[0, 1], 5);
}

#[test]
fn test_minimize_splits_pending_splitter() {
    // Exercises the non-textbook (partition_index, symbol) worklist: a partition
    // is split while entries naming it are still pending, and the larger half
    // silently inherits those entries.
    //
    //   0 -a-> 1   0 -b-> 2
    //   1 -a-> 3   1 -b-> 4
    //   2 -a-> 4   2 -b-> 3
    //   3 -{a,b}-> 5(final)      4 -{a,b}-> 6
    //   5(final) -{a,b}-> 6      6 -{a,b}-> 6   (6 = sink)
    //
    // The initial non-final block N = {0,1,2,3,4,6} is enqueued for both symbols.
    // Processing (F={5}, a) splits N into {3} and the larger {0,1,2,4,6}, which
    // keeps N's index -- so the two still-pending (N, a) / (N, b) entries now
    // name a *shrunken* set. States 1 and 2 are told apart only because the
    // a-predecessors of the halves are disjoint: 1 -a-> 3 lands in the smaller
    // half (re-enqueued under a fresh index) while 2 -a-> 4 stays in the larger
    // one. If splitting a pending splitter dropped either side's obligation,
    // 1 and 2 would wrongly merge and the count below would come out low.
    //
    // Languages: L(0)={aaa,bba}, L(1)={aa}, L(2)={ba}, L(3)={a,b}, L(5)={eps},
    // and L(4)=L(6)=empty -- so 4 and 6 (both dead) are the only equivalent pair.
    let mut dfa = DFA::new();
    for _ in 0..7 {
        dfa.add_state();
    }
    dfa.set_start_state(0);
    dfa.add_final_state(5);

    dfa.add_transition(0, 0, 1);
    dfa.add_transition(0, 1, 2);
    dfa.add_transition(1, 0, 3);
    dfa.add_transition(1, 1, 4);
    dfa.add_transition(2, 0, 4);
    dfa.add_transition(2, 1, 3);
    for (from, to) in [(3, 5), (4, 6), (5, 6), (6, 6)] {
        dfa.add_transition(from, 0, to);
        dfa.add_transition(from, 1, to);
    }

    let minimized = dfa.minimize();

    // Only the two dead states merge: 6 classes. In particular 1 and 2 must stay
    // apart -- that is the distinction the inherited worklist entry is
    // responsible for.
    assert_eq!(
        minimized.num_states(),
        6,
        "only the dead states 4 and 6 are equivalent; splitting a pending splitter must not lose a distinction"
    );
    assert_eq!(
        moore_class_count(&dfa, &[0, 1]),
        6,
        "reference disagrees; the hand-computed class list in the comment is stale"
    );
    assert_same_language(&dfa, &minimized, &[0, 1], 5);
}

#[test]
fn test_minimize_drops_unreachable_states() {
    // Reachable part: 0 -a-> 1(final) -a-> 1, plus an equivalent duplicate
    // 0 -b-> 2(final) -{a,b}-> 2 so a real merge also happens.
    // States 3..5 form an unreachable island (including an unreachable final).
    let mut dfa = DFA::new();
    for _ in 0..6 {
        dfa.add_state();
    }
    dfa.set_start_state(0);
    dfa.add_final_state(1);
    dfa.add_final_state(2);
    dfa.add_final_state(4);

    dfa.add_transition(0, 0, 1);
    dfa.add_transition(0, 1, 2);
    for (from, to) in [(1, 1), (2, 2)] {
        dfa.add_transition(from, 0, to);
        dfa.add_transition(from, 1, to);
    }
    // Unreachable island.
    dfa.add_transition(3, 0, 4);
    dfa.add_transition(3, 1, 5);
    dfa.add_transition(4, 0, 5);
    dfa.add_transition(4, 1, 3);
    dfa.add_transition(5, 0, 5);
    dfa.add_transition(5, 1, 5);

    let minimized = dfa.minimize();

    assert_eq!(
        minimized.num_states(),
        2,
        "unreachable states 3-5 must be dropped and finals 1&2 merged"
    );
    assert_eq!(minimized.final_states().len(), 1);
    assert_same_language(&dfa, &minimized, &[0, 1], 4);

    // Minimizing a DFA whose start state reaches nothing accepting yields the
    // empty DFA even though final states exist elsewhere in the graph.
    let mut stranded = DFA::new();
    for _ in 0..3 {
        stranded.add_state();
    }
    stranded.set_start_state(0);
    stranded.add_final_state(2);
    stranded.add_transition(0, 0, 1);
    stranded.add_transition(1, 0, 0);
    stranded.add_transition(2, 0, 2);
    let minimized = stranded.minimize();
    assert!(minimized.is_empty());
    assert_same_language(&stranded, &minimized, &[0], 4);
}

/// Reference implementation: Moore's table-filling algorithm.
///
/// Deliberately unrelated to `minimize`'s worklist scheme -- it marks
/// distinguishable pairs to a fixpoint -- so agreement between the two is
/// evidence about the algorithm, not a restatement of it. Returns the number of
/// equivalence classes among the reachable states.
fn moore_class_count(dfa: &DFA, alphabet: &[SymbolId]) -> usize {
    let reachable: Vec<StateId> = {
        let mut seen = vec![false; dfa.num_states() as usize];
        let mut stack = dfa.start_state().into_iter().collect::<Vec<_>>();
        let mut order = Vec::new();
        while let Some(state) = stack.pop() {
            if seen[state as usize] {
                continue;
            }
            seen[state as usize] = true;
            order.push(state);
            for &symbol in alphabet {
                if let Some(next) = dfa.transition(state, symbol) {
                    stack.push(next);
                }
            }
        }
        order.sort_unstable();
        order
    };

    let mut distinguishable: HashSet<(StateId, StateId)> = HashSet::new();
    let pair = |p: StateId, q: StateId| if p < q { (p, q) } else { (q, p) };

    for (i, &p) in reachable.iter().enumerate() {
        for &q in &reachable[i + 1..] {
            if dfa.final_states().contains(p) != dfa.final_states().contains(q) {
                distinguishable.insert(pair(p, q));
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (i, &p) in reachable.iter().enumerate() {
            for &q in &reachable[i + 1..] {
                if distinguishable.contains(&pair(p, q)) {
                    continue;
                }
                let split = alphabet.iter().any(|&symbol| {
                    match (dfa.transition(p, symbol), dfa.transition(q, symbol)) {
                        (Some(dp), Some(dq)) => dp != dq && distinguishable.contains(&pair(dp, dq)),
                        (None, None) => false,
                        // One state has the transition and the other does not.
                        _ => true,
                    }
                });
                if split {
                    distinguishable.insert(pair(p, q));
                    changed = true;
                }
            }
        }
    }

    // Greedily group each state with the first earlier state it is equivalent to.
    let mut representatives: Vec<StateId> = Vec::new();
    for &state in &reachable {
        if !representatives
            .iter()
            .any(|&rep| !distinguishable.contains(&pair(rep, state)))
        {
            representatives.push(state);
        }
    }
    representatives.len()
}

#[test]
fn test_minimize_matches_moore_reference() {
    // Deterministic stress test over complete 2-symbol DFAs: the transition
    // table and the final-state mask are both derived from a fixed LCG, so the
    // corpus is reproducible and needs no rand dependency. Complete DFAs keep
    // the Moore reference simple and match how minimize() is used downstream.
    let alphabet = [0, 1];
    let mut seed: u64 = 0x243f_6a88_85a3_08d3;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };

    let mut saw_merge = false;
    for case in 0..400u32 {
        let num_states = 3 + (case % 6);
        let mut dfa = DFA::new();
        for _ in 0..num_states {
            dfa.add_state();
        }
        dfa.set_start_state(0);

        let mut has_final = false;
        for state in 0..num_states {
            for &symbol in &alphabet {
                dfa.add_transition(state, symbol, next() % num_states);
            }
            if next() % 3 == 0 {
                dfa.add_final_state(state);
                has_final = true;
            }
        }
        if !has_final {
            // minimize() returns the empty DFA for a language-empty input; the
            // interesting cases all have at least one accepting state.
            dfa.add_final_state(next() % num_states);
        }

        let minimized = dfa.minimize();
        // The start state is always reachable here, so minimize() never takes its
        // empty-DFA early return -- even a language-empty case collapses to the
        // single non-final class Moore also reports.
        let expected = moore_class_count(&dfa, &alphabet);

        assert_eq!(
            minimized.num_states(),
            expected as StateId,
            "case {case}: minimize() disagrees with the Moore reference"
        );
        assert_same_language(&dfa, &minimized, &alphabet, 6);
        saw_merge |= minimized.num_states() < num_states;
    }

    assert!(
        saw_merge,
        "corpus never exercised an actual merge; the generator regressed"
    );
}

#[test]
fn test_empty_dfa() {
    let dfa = DFA::new();
    assert!(dfa.is_empty());

    let mut dfa2 = DFA::new();
    dfa2.add_state();
    dfa2.set_start_state(0);
    // No final states - should be empty
    assert!(dfa2.is_empty());
}

#[test]
fn test_add_transition_auto_grows_state_count() {
    // add_transition on unregistered endpoints must grow num_states rather than
    // record an edge referencing a state that 0..num_states never visits (which
    // left PyDFA::to_networkx emitting edges to nodes it had not added).
    let mut dfa = DFA::new();
    dfa.add_transition(2, 0, 5);

    assert_eq!(dfa.num_states(), 6, "destination id 5 must be covered");
    assert_eq!(dfa.transition(2, 0), Some(5));

    for (src, _, dst) in dfa.transitions() {
        assert!(src < dfa.num_states() && dst < dfa.num_states());
    }

    // A subsequent add_state() hands out the next id past the auto-grown range,
    // so ids stay unique.
    assert_eq!(dfa.add_state(), 6);
}

#[test]
fn test_start_and_final_states_auto_grow_state_count() {
    let mut dfa = DFA::new();
    dfa.set_start_state(1);
    assert_eq!(dfa.num_states(), 2);

    dfa.add_final_state(4);
    assert_eq!(dfa.num_states(), 5);
    assert!(dfa.final_states().contains(4));

    // Growing never shrinks: a lower id leaves the count alone.
    dfa.add_final_state(0);
    assert_eq!(dfa.num_states(), 5);
}
