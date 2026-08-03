//! Unit tests for the parent epsilon-NFA module.
//! Extracted from the parent module's `#[cfg(test)] mod tests` block.

use super::*;

#[test]
fn test_epsilon_nfa_basic() {
    let mut nfa = EpsilonNFA::new();

    // Create a simple NFA: 0 -a-> 1 -ε-> 2 (final)
    nfa.add_transition(0, 0, 1); // symbol 0 = 'a'
    nfa.add_epsilon_transition(1, 2);
    nfa.add_start_state(0);
    nfa.add_final_state(2);

    assert_eq!(nfa.num_states(), 3);
    assert!(!nfa.is_empty());
}

#[test]
fn test_epsilon_closure() {
    let mut nfa = EpsilonNFA::new();

    // 0 -ε-> 1 -ε-> 2
    nfa.add_epsilon_transition(0, 1);
    nfa.add_epsilon_transition(1, 2);
    nfa.add_start_state(0);

    let start = StateSet::singleton(0, 3);
    let closure = nfa.epsilon_closure(&start);

    assert!(closure.contains(0));
    assert!(closure.contains(1));
    assert!(closure.contains(2));
    assert_eq!(closure.len(), 3);
}

#[test]
fn test_move_on_symbol() {
    let mut nfa = EpsilonNFA::new();

    // 0 -a-> 1, 0 -a-> 2, 1 -ε-> 3
    nfa.add_transition(0, 0, 1); // 'a' = 0
    nfa.add_transition(0, 0, 2);
    nfa.add_epsilon_transition(1, 3);

    let start = StateSet::singleton(0, 4);
    let reached = nfa
        .move_on_symbol(&start, 0)
        .expect("symbol 0 is not the epsilon marker");

    assert!(reached.contains(1));
    assert!(reached.contains(2));
    assert!(reached.contains(3)); // via epsilon from 1
    assert_eq!(reached.len(), 3);
}

#[test]
fn test_empty_nfa() {
    let mut nfa = EpsilonNFA::new();
    nfa.add_start_state(0);
    nfa.add_final_state(1);
    // No transitions - NFA is empty (no path from 0 to 1)
    assert!(nfa.is_empty());

    // Add transition
    nfa.add_transition(0, 0, 1);
    assert!(!nfa.is_empty());
}

/// angr-9ke6b.191: `move_on_symbol` reports the epsilon marker as an error
/// instead of panicking, so the PyO3 layer can raise a clean `ValueError`.
#[test]
fn test_move_on_symbol_rejects_epsilon() {
    let mut nfa = EpsilonNFA::new();
    nfa.add_epsilon_transition(0, 1);

    let start = StateSet::singleton(0, 2);
    assert_eq!(nfa.move_on_symbol(&start, EPSILON), Err(EpsilonSymbolError));
}
