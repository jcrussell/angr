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
