// Unit tests for reachability.rs (the shared BFS traversal helper).

use super::*;
use std::collections::HashMap;

/// Build a successor callback over an explicit adjacency map.
fn successors_of(graph: HashMap<StateId, Vec<StateId>>) -> impl FnMut(StateId, &mut Vec<StateId>) {
    move |state, out| {
        if let Some(next) = graph.get(&state) {
            out.extend(next.iter().copied());
        }
    }
}

#[test]
fn test_reachable_from_no_seeds_is_empty() {
    let reached = reachable_from(std::iter::empty(), 8, |_, out: &mut Vec<StateId>| {
        out.push(0);
    });
    assert!(reached.is_empty());
}

#[test]
fn test_reachable_from_seed_with_no_successors() {
    let reached = reachable_from([3], 8, |_, _| {});
    assert_eq!(reached.to_vec(), vec![3]);
}

#[test]
fn test_reachable_from_follows_chain() {
    let graph = HashMap::from([(0, vec![1]), (1, vec![2]), (2, vec![3])]);
    let reached = reachable_from([0], 8, successors_of(graph));
    assert_eq!(reached.to_vec(), vec![0, 1, 2, 3]);
}

#[test]
fn test_reachable_from_skips_unreachable_states() {
    // 0 -> 1, with an isolated 5 -> 6 component that must not be reached.
    let graph = HashMap::from([(0, vec![1]), (5, vec![6])]);
    let reached = reachable_from([0], 8, successors_of(graph));
    assert_eq!(reached.to_vec(), vec![0, 1]);
}

#[test]
fn test_reachable_from_terminates_on_cycle() {
    let graph = HashMap::from([(0, vec![1]), (1, vec![2]), (2, vec![0])]);
    let reached = reachable_from([0], 8, successors_of(graph));
    assert_eq!(reached.to_vec(), vec![0, 1, 2]);
}

#[test]
fn test_reachable_from_multiple_seeds() {
    let graph = HashMap::from([(0, vec![1]), (4, vec![5])]);
    let reached = reachable_from([0, 4], 8, successors_of(graph));
    assert_eq!(reached.to_vec(), vec![0, 1, 4, 5]);
}

#[test]
fn test_reachable_from_tolerates_duplicate_successors() {
    // The callback may emit the same neighbour repeatedly; the visited set
    // filters them, and each state is expanded exactly once.
    let mut expansions = 0usize;
    let reached = reachable_from([0], 8, |state, out| {
        expansions += 1;
        if state == 0 {
            out.extend([1, 1, 1, 0]);
        }
    });
    assert_eq!(reached.to_vec(), vec![0, 1]);
    assert_eq!(expansions, 2);
}

#[test]
fn test_reachable_from_grows_past_capacity_hint() {
    // The capacity argument is only a hint; StateSet::insert grows on demand.
    let graph = HashMap::from([(0, vec![64])]);
    let reached = reachable_from([0], 2, successors_of(graph));
    assert_eq!(reached.to_vec(), vec![0, 64]);
}
