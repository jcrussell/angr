use super::*;

#[test]
fn test_subset_construction_basic() {
    // NFA: 0 -a-> 1, 0 -a-> 2, 1 -b-> 3(final), 2 -b-> 3(final)
    let mut nfa = EpsilonNFA::new();
    nfa.add_transition(0, 0, 1); // 'a' = 0
    nfa.add_transition(0, 0, 2);
    nfa.add_transition(1, 1, 3); // 'b' = 1
    nfa.add_transition(2, 1, 3);
    nfa.add_start_state(0);
    nfa.add_final_state(3);

    let dfa = subset_construction(&nfa);

    assert!(dfa.start_state().is_some());
    assert!(!dfa.final_states().is_empty());
}

#[test]
fn test_subset_construction_with_epsilon() {
    // NFA: 0 -ε-> 1 -a-> 2(final)
    let mut nfa = EpsilonNFA::new();
    nfa.add_epsilon_transition(0, 1);
    nfa.add_transition(1, 0, 2); // 'a' = 0
    nfa.add_start_state(0);
    nfa.add_final_state(2);

    let dfa = subset_construction(&nfa);

    // DFA should recognize "a"
    let start = dfa
        .start_state()
        .expect("subset construction has a start state");
    assert!(!dfa.final_states().is_empty());

    // Initial DFA state is {0, 1} (epsilon closure of {0}); on 'a' (symbol 0)
    // it must transition to a final state (the {2} subset). Previously this was
    // only described in a trailing comment and never asserted, so the test
    // passed even if subset_construction dropped the transition entirely.
    let next = dfa
        .transition(start, 0)
        .expect("start state should have a transition on 'a'");
    assert!(
        dfa.final_states().contains(next),
        "transition on 'a' should land in a final state"
    );
}

#[test]
fn test_empty_nfa() {
    let nfa = EpsilonNFA::new();
    let dfa = subset_construction(&nfa);
    assert!(dfa.start_state().is_none());
}
