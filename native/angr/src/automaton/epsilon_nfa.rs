//! Epsilon Non-deterministic Finite Automaton (ε-NFA) implementation.

use crate::automaton::reachability::reachable_from;
use crate::automaton::state::{StateId, StateSet};
use crate::automaton::symbol::{EPSILON, SymbolId, is_epsilon};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// [`EpsilonNFA::move_on_symbol`] was handed the [`EPSILON`] marker.
///
/// Epsilon moves are the job of [`EpsilonNFA::epsilon_closure`]; treating the
/// marker as an ordinary symbol would return the raw epsilon successors without
/// taking their closure, i.e. a wrong answer. This is an internal-invariant
/// violation rather than user input (see `move_on_symbol`'s docs), but it is
/// reported rather than asserted so the PyO3 layer can surface a clean
/// `ValueError` instead of an opaque `PanicException` (angr-9ke6b.191).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpsilonSymbolError;

impl fmt::Display for EpsilonSymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("epsilon is not a move symbol; use epsilon_closure for epsilon moves")
    }
}

impl std::error::Error for EpsilonSymbolError {}

/// An Epsilon Non-deterministic Finite Automaton.
#[derive(Debug, Clone)]
pub struct EpsilonNFA {
    /// Number of states (states are numbered 0..num_states)
    num_states: StateId,
    /// Start states
    start_states: StateSet,
    /// Final (accepting) states
    final_states: StateSet,
    /// Transitions: (source, symbol) -> set of destination states
    /// For epsilon transitions, symbol == EPSILON
    transitions: HashMap<(StateId, SymbolId), StateSet>,
    /// All symbols used (excluding epsilon)
    alphabet: HashSet<SymbolId>,
    /// Cached epsilon closures for each state
    epsilon_closures: Option<Vec<StateSet>>,
}

impl EpsilonNFA {
    /// Create a new empty epsilon-NFA.
    pub fn new() -> Self {
        Self {
            num_states: 0,
            start_states: StateSet::with_capacity(16),
            final_states: StateSet::with_capacity(16),
            transitions: HashMap::new(),
            alphabet: HashSet::new(),
            epsilon_closures: None,
        }
    }

    /// Ensure a state exists, expanding num_states if needed.
    fn ensure_state(&mut self, state: StateId) {
        if state >= self.num_states {
            self.num_states = state + 1;
            // Invalidate cached epsilon closures
            self.epsilon_closures = None;
        }
    }

    /// Add a transition from source to destination on the given symbol.
    pub fn add_transition(&mut self, source: StateId, symbol: SymbolId, destination: StateId) {
        self.ensure_state(source);
        self.ensure_state(destination);

        if !is_epsilon(symbol) {
            self.alphabet.insert(symbol);
        }

        self.transitions
            .entry((source, symbol))
            .or_insert_with(|| StateSet::with_capacity(self.num_states as usize))
            .insert(destination);

        // Invalidate cached epsilon closures
        self.epsilon_closures = None;
    }

    /// Add an epsilon transition from source to destination.
    pub fn add_epsilon_transition(&mut self, source: StateId, destination: StateId) {
        self.add_transition(source, EPSILON, destination);
    }

    /// Add a start state.
    pub fn add_start_state(&mut self, state: StateId) {
        self.ensure_state(state);
        self.start_states.insert(state);
    }

    /// Add a final (accepting) state.
    pub fn add_final_state(&mut self, state: StateId) {
        self.ensure_state(state);
        self.final_states.insert(state);
    }

    /// Get the number of states.
    pub fn num_states(&self) -> StateId {
        self.num_states
    }

    /// Get the start states.
    pub fn start_states(&self) -> &StateSet {
        &self.start_states
    }

    /// Get the final states.
    pub fn final_states(&self) -> &StateSet {
        &self.final_states
    }

    /// Get the alphabet (all symbols except epsilon).
    pub fn alphabet(&self) -> &HashSet<SymbolId> {
        &self.alphabet
    }

    /// Compute the epsilon closure of a single state using DFS.
    fn epsilon_closure_single(&self, state: StateId) -> StateSet {
        let mut closure = StateSet::with_capacity(self.num_states as usize);
        let mut stack = vec![state];

        while let Some(s) = stack.pop() {
            if closure.contains(s) {
                continue;
            }
            closure.insert(s);

            // Follow epsilon transitions
            if let Some(destinations) = self.transitions.get(&(s, EPSILON)) {
                for dest in destinations.iter() {
                    if !closure.contains(dest) {
                        stack.push(dest);
                    }
                }
            }
        }

        closure
    }

    /// Compute epsilon closures for all states (cached).
    pub fn compute_epsilon_closures(&mut self) {
        if self.epsilon_closures.is_some() {
            return;
        }

        let mut closures = Vec::with_capacity(self.num_states as usize);
        for state in 0..self.num_states {
            closures.push(self.epsilon_closure_single(state));
        }
        self.epsilon_closures = Some(closures);
    }

    /// Get the epsilon closure of a set of states.
    ///
    /// # Panics
    ///
    /// Every id in `states` must be a state of this NFA (`< num_states`), which
    /// holds by construction for the in-tree callers — they draw their state
    /// sets from `start_states` or from `transitions` destinations. A violation
    /// therefore means either a caller-supplied bogus id or, once
    /// `compute_epsilon_closures` has run, a mutation that grew the NFA without
    /// clearing the cache (`ensure_state` and `add_transition` both clear it).
    /// Both are bugs and both panic rather than skip the offending state:
    /// skipping drops states from the closure, which surfaces later as a
    /// silently wrong automaton instead of a debuggable failure (angr-9ke6b.193).
    pub fn epsilon_closure(&self, states: &StateSet) -> StateSet {
        let mut closure = StateSet::with_capacity(self.num_states as usize);

        if let Some(cached) = &self.epsilon_closures {
            // Use cached closures
            for state in states.iter() {
                let Some(state_closure) = cached.get(state as usize) else {
                    panic!(
                        "epsilon_closure: state {state} is outside the {} cached \
                         closures (num_states={}); the epsilon-closure cache is \
                         stale or the state id is not a state of this NFA",
                        cached.len(),
                        self.num_states
                    );
                };
                closure.union_with(state_closure);
            }
        } else {
            // Compute on-the-fly using DFS
            let mut stack: Vec<StateId> = states.iter().collect();

            while let Some(s) = stack.pop() {
                if closure.contains(s) {
                    continue;
                }
                closure.insert(s);

                if let Some(destinations) = self.transitions.get(&(s, EPSILON)) {
                    for dest in destinations.iter() {
                        if !closure.contains(dest) {
                            stack.push(dest);
                        }
                    }
                }
            }
        }

        closure
    }

    /// Get the states reachable from a set of states on a given symbol.
    /// Returns the epsilon closure of the reached states.
    ///
    /// # Errors
    ///
    /// Returns `EpsilonSymbolError` if `symbol` is the [`EPSILON`] marker.
    /// Callers are expected to draw symbols from [`Self::alphabet`], which
    /// excludes epsilon by construction (see `add_transition`), so this is an
    /// invariant check, not an input check — but it is a `Result` rather than
    /// an `assert!` because the only caller chain reaches here from PyO3
    /// (`PyEpsilonNFA::minimize` -> `subset_construction`), where a panic would
    /// surface to Python as an opaque `PanicException`.
    pub fn move_on_symbol(
        &self,
        states: &StateSet,
        symbol: SymbolId,
    ) -> Result<StateSet, EpsilonSymbolError> {
        if is_epsilon(symbol) {
            return Err(EpsilonSymbolError);
        }

        let mut reached = StateSet::with_capacity(self.num_states as usize);

        for state in states.iter() {
            if let Some(destinations) = self.transitions.get(&(state, symbol)) {
                reached.union_with(destinations);
            }
        }

        Ok(self.epsilon_closure(&reached))
    }

    /// Check if the NFA accepts any string (i.e., if the language is non-empty).
    /// Uses BFS from the epsilon closure of the start states, following every
    /// non-epsilon transition into the closure of its destinations.
    pub fn is_empty(&self) -> bool {
        if self.start_states.is_empty() || self.final_states.is_empty() {
            return true;
        }

        let start_closure = self.epsilon_closure(&self.start_states);
        let reachable = reachable_from(
            start_closure.iter(),
            self.num_states as usize,
            |state, out| {
                for &symbol in &self.alphabet {
                    if let Some(destinations) = self.transitions.get(&(state, symbol)) {
                        out.extend(self.epsilon_closure(destinations).iter());
                    }
                }
            },
        );

        !self.final_states.intersects(&reachable)
    }
}

impl Default for EpsilonNFA {
    fn default() -> Self {
        Self::new()
    }
}

test_submod!("epsilon_nfa_tests.rs" => tests);
