//! Deterministic Finite Automaton (DFA) implementation with Hopcroft minimization.

use crate::automaton::reachability::reachable_from;
use crate::automaton::state::{StateId, StateSet};
use crate::automaton::symbol::SymbolId;
use std::collections::{HashMap, HashSet, VecDeque};

/// A Deterministic Finite Automaton.
#[derive(Debug, Clone)]
pub struct DFA {
    /// Number of states
    num_states: StateId,
    /// Start state (None if empty)
    start_state: Option<StateId>,
    /// Final (accepting) states
    final_states: StateSet,
    /// Transitions: (source, symbol) -> destination
    transitions: HashMap<(StateId, SymbolId), StateId>,
    /// Reverse transitions: (destination, symbol) -> set of sources
    reverse_transitions: HashMap<(StateId, SymbolId), StateSet>,
    /// All symbols used
    alphabet: HashSet<SymbolId>,
    /// Mapping from DFA states to original NFA states (if created via subset construction)
    state_mapping: Option<HashMap<StateId, Vec<StateId>>>,
}

impl DFA {
    /// Create a new empty DFA.
    pub fn new() -> Self {
        Self {
            num_states: 0,
            start_state: None,
            final_states: StateSet::with_capacity(16),
            transitions: HashMap::new(),
            reverse_transitions: HashMap::new(),
            alphabet: HashSet::new(),
            state_mapping: None,
        }
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> StateId {
        let id = self.num_states;
        self.num_states += 1;
        id
    }

    /// Ensure a state exists, expanding `num_states` if needed.
    ///
    /// Mirrors `EpsilonNFA::ensure_state`: every entry point that names a state
    /// id auto-grows the state count rather than silently recording an id that
    /// `num_states()` (and everything iterating `0..num_states`, e.g. the node
    /// list `PyDFA::to_networkx` emits) would not report.
    fn ensure_state(&mut self, state: StateId) {
        if state >= self.num_states {
            self.num_states = state + 1;
        }
    }

    /// Set the start state, registering it if it was not added via
    /// [`add_state`](Self::add_state).
    pub fn set_start_state(&mut self, state: StateId) {
        self.ensure_state(state);
        self.start_state = Some(state);
    }

    /// Add a final (accepting) state, registering it if it was not added via
    /// [`add_state`](Self::add_state).
    pub fn add_final_state(&mut self, state: StateId) {
        self.ensure_state(state);
        self.final_states.insert(state);
    }

    /// Add a transition.
    ///
    /// Both endpoints are registered via [`add_state`](Self::add_state)-equivalent
    /// auto-growth if they are at or past the current state count, so an edge can
    /// never reference a state that `num_states()` does not cover.
    ///
    /// Re-adding a transition for an existing `(source, symbol)` pair overwrites
    /// the previous destination; the stale `reverse_transitions` entry for the
    /// old destination is retired so `find_predecessors` (and therefore
    /// `minimize`'s Hopcroft refinement) never sees an edge that no longer exists.
    pub fn add_transition(&mut self, source: StateId, symbol: SymbolId, destination: StateId) {
        self.ensure_state(source);
        self.ensure_state(destination);
        self.alphabet.insert(symbol);
        let previous = self.transitions.insert((source, symbol), destination);

        // Drop the reverse edge left behind by an overwritten destination.
        if let Some(old_destination) = previous
            && old_destination != destination
            && let Some(sources) = self.reverse_transitions.get_mut(&(old_destination, symbol))
        {
            sources.remove(source);
            if sources.is_empty() {
                self.reverse_transitions.remove(&(old_destination, symbol));
            }
        }

        // Also update reverse transitions
        self.reverse_transitions
            .entry((destination, symbol))
            .or_insert_with(|| StateSet::with_capacity(self.num_states as usize))
            .insert(source);
    }

    /// Get the transition from a state on a symbol.
    pub fn transition(&self, source: StateId, symbol: SymbolId) -> Option<StateId> {
        self.transitions.get(&(source, symbol)).copied()
    }

    /// Get the number of states.
    pub fn num_states(&self) -> StateId {
        self.num_states
    }

    /// Get the start state.
    pub fn start_state(&self) -> Option<StateId> {
        self.start_state
    }

    /// Get the final states.
    pub fn final_states(&self) -> &StateSet {
        &self.final_states
    }

    /// Set the state mapping from original NFA states.
    pub fn set_state_mapping(&mut self, mapping: HashMap<StateId, Vec<StateId>>) {
        self.state_mapping = Some(mapping);
    }

    /// Check if the DFA is empty (accepts no strings).
    ///
    /// A DFA with no start state reaches nothing, so `find_reachable_states`
    /// already returns the empty set for it.
    pub fn is_empty(&self) -> bool {
        if self.final_states.is_empty() {
            return true;
        }

        !self.final_states.intersects(&self.find_reachable_states())
    }

    /// Get all transitions as an iterator.
    pub fn transitions(&self) -> impl Iterator<Item = (StateId, SymbolId, StateId)> + '_ {
        self.transitions
            .iter()
            .map(|(&(src, sym), &dst)| (src, sym, dst))
    }

    /// Minimize the DFA using Hopcroft's algorithm.
    /// Returns a new minimized DFA.
    pub fn minimize(&self) -> DFA {
        if self.start_state.is_none() || self.num_states == 0 {
            return DFA::new();
        }

        // First, remove unreachable states
        let reachable = self.find_reachable_states();

        // If no reachable states, return empty DFA
        if reachable.is_empty() {
            return DFA::new();
        }

        // Hopcroft's partition refinement algorithm
        // Initial partition: final states and non-final states
        let final_reachable = self.final_states.intersection(&reachable);
        let non_final_reachable = reachable.difference(&self.final_states);

        let mut partitions: Vec<StateSet> = Vec::new();

        if !final_reachable.is_empty() {
            partitions.push(final_reachable);
        }
        if !non_final_reachable.is_empty() {
            partitions.push(non_final_reachable);
        }

        if partitions.is_empty() {
            return DFA::new();
        }

        // Worklist of (partition_index, symbol) pairs to process.
        //
        // This diverges from the textbook Hopcroft presentation, which enqueues the
        // splitter *set* and, when a pending splitter Y is itself split into Y1/Y2,
        // replaces the pending entry with both halves. Here an entry names a partition
        // *index*: when a partition splits, the larger half keeps the old index (and so
        // inherits every still-pending entry that referenced it), and only the smaller
        // half is enqueued, under a fresh index.
        //
        // That bookkeeping is still sound. Say `(i, s)` is pending when partition
        // `i` = Y splits into `keep` (staying at `i`) and `add` (new index `j`).
        // Because the automaton is deterministic, a state has at most one `s`-successor,
        // which lands in exactly one of the two halves -- so
        // `pred(Y, s) = pred(keep, s) ∪ pred(add, s)` and the two are *disjoint*.
        // Refining by `pred(keep, s)` (what the inherited entry now does) and by
        // `pred(add, s)` (what the freshly enqueued `(j, s)` does) therefore separates
        // every pair the original `pred(Y, s)` would have. No distinction is lost; the
        // refinement is merely spread over two pops.
        //
        // Regression coverage for exactly this case lives in `dfa_tests.rs`:
        // `test_minimize_splits_pending_splitter` and
        // `test_minimize_matches_moore_reference`.
        let mut worklist: VecDeque<(usize, SymbolId)> = VecDeque::new();

        // Initialize worklist with all (partition, symbol) pairs
        for (idx, _) in partitions.iter().enumerate() {
            for &symbol in &self.alphabet {
                worklist.push_back((idx, symbol));
            }
        }

        // Main refinement loop
        while let Some((splitter_idx, symbol)) = worklist.pop_front() {
            // Get states that can reach the splitter partition on this symbol
            let splitter = if splitter_idx < partitions.len() {
                partitions[splitter_idx].clone()
            } else {
                continue;
            };

            let predecessors = self.find_predecessors(&splitter, symbol);

            if predecessors.is_empty() {
                continue;
            }

            // Try to split each partition
            let mut new_partitions = Vec::new();

            for (part_idx, partition) in partitions.iter().enumerate() {
                let intersection = partition.intersection(&predecessors);
                let difference = partition.difference(&predecessors);

                if !intersection.is_empty() && !difference.is_empty() {
                    // Partition is split
                    // Keep the larger part in place, add smaller to new_partitions
                    let (keep, add) = if intersection.len() <= difference.len() {
                        (difference, intersection)
                    } else {
                        (intersection, difference)
                    };

                    new_partitions.push((part_idx, keep, add));
                }
            }

            // Apply splits
            for (part_idx, keep, add) in new_partitions {
                let new_idx = partitions.len();
                partitions[part_idx] = keep;
                partitions.push(add);

                // Add new partition to worklist for all symbols
                for &sym in &self.alphabet {
                    worklist.push_back((new_idx, sym));
                }
            }
        }

        // Build minimized DFA from partitions
        self.build_minimized_dfa(&partitions)
    }

    /// Find all states reachable from the start state.
    fn find_reachable_states(&self) -> StateSet {
        reachable_from(self.start_state, self.num_states as usize, |state, out| {
            out.extend(
                self.alphabet
                    .iter()
                    .filter_map(|&symbol| self.transition(state, symbol)),
            );
        })
    }

    /// Find all states that can reach the target set on a given symbol.
    fn find_predecessors(&self, targets: &StateSet, symbol: SymbolId) -> StateSet {
        let mut predecessors = StateSet::with_capacity(self.num_states as usize);

        for target in targets.iter() {
            if let Some(sources) = self.reverse_transitions.get(&(target, symbol)) {
                predecessors.union_with(sources);
            }
        }

        predecessors
    }

    /// Build a minimized DFA from partitions.
    fn build_minimized_dfa(&self, partitions: &[StateSet]) -> DFA {
        let mut minimized = DFA::new();

        // Map old states to their partition (new state)
        let mut state_to_partition: HashMap<StateId, StateId> = HashMap::new();
        for (part_idx, partition) in partitions.iter().enumerate() {
            for state in partition.iter() {
                state_to_partition.insert(state, part_idx as StateId);
            }
        }

        // Create states in minimized DFA
        for _ in 0..partitions.len() {
            minimized.add_state();
        }

        // Set start state
        if let Some(start) = self.start_state
            && let Some(&new_start) = state_to_partition.get(&start)
        {
            minimized.set_start_state(new_start);
        }

        // Set final states
        for final_state in self.final_states.iter() {
            if let Some(&new_state) = state_to_partition.get(&final_state) {
                minimized.add_final_state(new_state);
            }
        }

        // Add transitions (use representative state from each partition)
        for (part_idx, partition) in partitions.iter().enumerate() {
            // Get any representative state from the partition
            if let Some(representative) = partition.iter().next() {
                for &symbol in &self.alphabet {
                    if let Some(dest) = self.transition(representative, symbol)
                        && let Some(&new_dest) = state_to_partition.get(&dest)
                    {
                        minimized.add_transition(part_idx as StateId, symbol, new_dest);
                    }
                }
            }
        }

        // Build state mapping from minimized states to original NFA states
        if let Some(orig_mapping) = &self.state_mapping {
            let mut new_mapping: HashMap<StateId, Vec<StateId>> = HashMap::new();
            for (part_idx, partition) in partitions.iter().enumerate() {
                let mut nfa_states = Vec::new();
                for old_dfa_state in partition.iter() {
                    if let Some(states) = orig_mapping.get(&old_dfa_state) {
                        nfa_states.extend(states.iter().copied());
                    }
                }
                nfa_states.sort_unstable();
                nfa_states.dedup();
                new_mapping.insert(part_idx as StateId, nfa_states);
            }
            minimized.state_mapping = Some(new_mapping);
        }

        minimized
    }
}

impl Default for DFA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "dfa_tests.rs"]
mod tests;
