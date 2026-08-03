//! Formal language automata implementation for type inference.
//!
//! This module provides epsilon-NFA and DFA implementations with:
//! - Epsilon closure computation
//! - Subset construction (NFA to DFA conversion)
//! - Hopcroft-style DFA minimization by partition refinement (see `DFA::minimize`)
//! - PyO3 bindings for Python interoperability

mod dfa;
mod epsilon_nfa;
mod python_bindings;
mod reachability;
mod state;
mod subset_construction;
mod symbol;

pub use dfa::DFA;
pub use epsilon_nfa::EpsilonNFA;
pub use python_bindings::automaton;
pub use state::{StateId, StateSet};
pub use symbol::{EPSILON, SymbolId};
