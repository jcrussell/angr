//! Subset construction algorithm for converting ε-NFA to DFA.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. The one
//! surviving `expect` in [`subset_construction`] is a worklist invariant, not
//! an input check — see its `#[allow]` reason.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::automaton::dfa::DFA;
use crate::automaton::epsilon_nfa::EpsilonNFA;
use crate::automaton::state::{StateId, StateSet};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Convert an epsilon-NFA to a DFA using the powerset construction algorithm.
#[allow(
    clippy::expect_used,
    reason = "worklist invariant: `worklist.push(set)` happens only where the same `set` was just inserted into `state_mapping` (the initial-set seed and the new-DFA-state branch), so a popped entry always has a mapping. Nothing here reads guest data"
)]
pub(super) fn subset_construction(nfa: &EpsilonNFA) -> DFA {
    // Each DFA state corresponds to a set of NFA states
    // We map sets of NFA states to DFA state IDs
    let mut state_mapping: IndexMap<Vec<StateId>, StateId> = IndexMap::new();
    let mut dfa = DFA::new();

    // Queue of DFA states to process (as NFA state sets)
    let mut worklist: Vec<StateSet> = Vec::new();

    // Initial DFA state is the epsilon closure of NFA start states
    let initial_set = nfa.epsilon_closure(nfa.start_states());

    if initial_set.is_empty() {
        // No reachable states - return empty DFA
        return dfa;
    }

    let initial_vec = initial_set.to_vec();
    let initial_dfa_state = 0;
    state_mapping.insert(initial_vec, initial_dfa_state);
    dfa.add_state();
    dfa.set_start_state(initial_dfa_state);

    // Check if initial state is final
    if initial_set.intersects(nfa.final_states()) {
        dfa.add_final_state(initial_dfa_state);
    }

    worklist.push(initial_set);

    while let Some(current_nfa_set) = worklist.pop() {
        let current_vec = current_nfa_set.to_vec();
        // Invariant: every NFA set pushed onto `worklist` was first inserted
        // into `state_mapping` (the initial-set seed above and the
        // new-DFA-state branch below), so this lookup always succeeds.
        let current_dfa_state = *state_mapping
            .get(&current_vec)
            .expect("worklist entry missing from state_mapping");

        // For each symbol in the alphabet
        for &symbol in nfa.alphabet() {
            // Compute the set of NFA states reachable on this symbol
            let next_nfa_set = nfa.move_on_symbol(&current_nfa_set, symbol);

            if next_nfa_set.is_empty() {
                // No transition on this symbol - skip (DFA will have no transition)
                continue;
            }

            let next_vec = next_nfa_set.to_vec();

            // Check if we've seen this DFA state before
            let next_dfa_state = if let Some(&existing) = state_mapping.get(&next_vec) {
                existing
            } else {
                // Create new DFA state
                let new_state = dfa.add_state();
                state_mapping.insert(next_vec, new_state);

                // Check if this new state is final
                if next_nfa_set.intersects(nfa.final_states()) {
                    dfa.add_final_state(new_state);
                }

                worklist.push(next_nfa_set);
                new_state
            };

            // Add transition
            dfa.add_transition(current_dfa_state, symbol, next_dfa_state);
        }
    }

    // Store the NFA-to-DFA state mapping in the DFA for later use
    let inverse_mapping: HashMap<StateId, Vec<StateId>> = state_mapping
        .into_iter()
        .map(|(nfa_states, dfa_state)| (dfa_state, nfa_states))
        .collect();
    dfa.set_state_mapping(inverse_mapping);

    dfa
}

#[cfg(test)]
#[path = "subset_construction_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
