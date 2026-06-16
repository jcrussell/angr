//! State types for automata.

use fixedbitset::FixedBitSet;
use std::fmt;

/// A state identifier represented as a u32.
pub type StateId = u32;

/// A set of states implemented using a fixed-size bit set for efficiency.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StateSet {
    bits: FixedBitSet,
}

impl StateSet {
    /// Create a new empty state set with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: FixedBitSet::with_capacity(capacity),
        }
    }

    /// Create a state set containing a single state.
    pub fn singleton(state: StateId, capacity: usize) -> Self {
        let mut set = Self::with_capacity(capacity);
        set.insert(state);
        set
    }

    /// Insert a state into the set.
    pub fn insert(&mut self, state: StateId) {
        let idx = state as usize;
        if idx >= self.bits.len() {
            self.bits.grow(idx + 1);
        }
        self.bits.insert(idx);
    }

    /// Check if the set contains a state.
    pub fn contains(&self, state: StateId) -> bool {
        let idx = state as usize;
        if idx >= self.bits.len() {
            false
        } else {
            self.bits.contains(idx)
        }
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.is_clear()
    }

    /// Get the number of states in the set.
    pub fn len(&self) -> usize {
        self.bits.count_ones(..)
    }

    /// Iterate over all states in the set.
    pub fn iter(&self) -> impl Iterator<Item = StateId> + '_ {
        self.bits.ones().map(|i| i as StateId)
    }

    /// Union this set with another, modifying self in place.
    pub fn union_with(&mut self, other: &StateSet) {
        if other.bits.len() > self.bits.len() {
            self.bits.grow(other.bits.len());
        }
        self.bits.union_with(&other.bits);
    }

    /// Check if this set intersects with another.
    pub fn intersects(&self, other: &StateSet) -> bool {
        self.bits.intersection(&other.bits).next().is_some()
    }

    /// Create a new set that is the intersection of this set and another.
    pub fn intersection(&self, other: &StateSet) -> StateSet {
        let mut result = self.clone();
        let max_len = std::cmp::max(result.bits.len(), other.bits.len());
        result.bits.grow(max_len);
        result.bits.intersect_with(&other.bits);
        result
    }

    /// Create a new set with states not in other.
    pub fn difference(&self, other: &StateSet) -> StateSet {
        let mut result = self.clone();
        result.bits.difference_with(&other.bits);
        result
    }

    /// Remove a state from the set.
    pub fn remove(&mut self, state: StateId) {
        let idx = state as usize;
        if idx < self.bits.len() {
            self.bits.set(idx, false);
        }
    }

    /// Clear all states from the set.
    pub fn clear(&mut self) {
        self.bits.clear();
    }

    /// Get a canonical representation for hashing (as a sorted vec).
    pub fn to_vec(&self) -> Vec<StateId> {
        self.iter().collect()
    }
}

impl fmt::Debug for StateSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<StateId> for StateSet {
    fn from_iter<I: IntoIterator<Item = StateId>>(iter: I) -> Self {
        let items: Vec<StateId> = iter.into_iter().collect();
        let capacity = items.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut set = Self::with_capacity(capacity);
        for state in items {
            set.insert(state);
        }
        set
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
