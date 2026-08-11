//! Shared breadth-first reachability traversal for the automaton types.

use crate::automaton::state::{StateId, StateSet};
use std::collections::VecDeque;

/// Compute the set of states reachable from `seeds` under an arbitrary
/// successor relation.
///
/// `DFA::find_reachable_states` (and `DFA::is_empty` through it) and
/// `EpsilonNFA::is_empty` walk different transition representations but share
/// this queue + visited-set loop; keeping it in one place means a change to
/// reachability semantics only has to be made once.
///
/// `successors` appends the out-neighbours of `state` to `out`, which is
/// emptied before every call. Duplicates and already-visited states are
/// harmless — they are filtered by the visited set.
pub(super) fn reachable_from<I, F>(seeds: I, capacity: usize, mut successors: F) -> StateSet
where
    I: IntoIterator<Item = StateId>,
    F: FnMut(StateId, &mut Vec<StateId>),
{
    let mut reachable = StateSet::with_capacity(capacity);
    let mut queue: VecDeque<StateId> = seeds.into_iter().collect();
    let mut buffer: Vec<StateId> = Vec::new();

    while let Some(state) = queue.pop_front() {
        if reachable.contains(state) {
            continue;
        }
        reachable.insert(state);

        buffer.clear();
        successors(state, &mut buffer);
        for next in buffer.drain(..) {
            if !reachable.contains(next) {
                queue.push_back(next);
            }
        }
    }

    reachable
}

test_submod!("reachability_tests.rs" => tests);
