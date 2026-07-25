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
fn test_empty_dfa() {
    let dfa = DFA::new();
    assert!(dfa.is_empty());

    let mut dfa2 = DFA::new();
    dfa2.add_state();
    dfa2.set_start_state(0);
    // No final states - should be empty
    assert!(dfa2.is_empty());
}
