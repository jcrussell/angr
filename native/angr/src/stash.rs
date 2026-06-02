//! Stash management for symbolic execution state tracking.
//!
//! Provides a StashManager that owns all stash-related data:
//! - Named stashes (active, found, avoid, deadended, errored, unconstrained)
//! - State index for O(1) lookups by state_id
//! - State root tracking for fork lineage
//! - Terminal state counters
//!
//! **Invariant I6 (cross-mixin, mirror of rust_manager.py:74):** the
//! Python-side `_cleanup_state_cache` orchestrates eviction with pinning
//! over `_state_roots ∪ {_current_callback_state_id,
//! _current_stepping_state_id}`. The Rust counterpart here owns the
//! `state_index` and `state_roots` maps that back lineage tracking. Both
//! maps must move together — every push/pop on a stash that participates
//! in lineage must update both, and `remove_state` / `clear` /
//! `push_or_drop_terminal` are the chokepoints that enforce this.
//! Without consistent maps, `_state_roots` drift produces orphaned roots
//! and a freshly-mutated state can race-evict between callbacks on the
//! same state. Regression tests:
//! `TestStateCacheSizeBound.test_cleanup_state_cache_evicts_oldest_first`,
//! `..._drops_dead_states`, `..._skips_pinned` (tests/engines/
//! test_rust_exploration.py:2017,2064,2096).

use std::collections::{HashMap, VecDeque};

use crate::state::RustSimState;

/// Well-known stash names.
pub const STASH_ACTIVE: &str = "active";
pub const STASH_FOUND: &str = "found";
pub const STASH_AVOID: &str = "avoid";
pub const STASH_DEADENDED: &str = "deadended";
pub const STASH_ERRORED: &str = "errored";
pub const STASH_PRUNED: &str = "pruned";
pub const STASH_UNCONSTRAINED: &str = "unconstrained";

/// Standard stash names pre-registered at construction. Anything else triggers
/// a one-time `log::warn!` on first creation via `ensure_stash` — typo guard
/// per angr-630x (follow-up from the angr-9l9j PyO3 trust-model audit).
pub const STANDARD_STASHES: &[&str] = &[
    STASH_ACTIVE,
    STASH_FOUND,
    STASH_AVOID,
    STASH_DEADENDED,
    STASH_ERRORED,
    STASH_PRUNED,
    STASH_UNCONSTRAINED,
];

/// Manages state stashes, indices, and lineage tracking.
pub struct StashManager {
    /// Named stashes holding exploration states.
    stashes: HashMap<String, VecDeque<RustSimState>>,
    /// Index mapping state_id -> stash name for O(1) lookups.
    state_index: HashMap<u64, String>,
    /// Maps state_id -> root_state_id for lineage tracking.
    state_roots: HashMap<u64, u64>,
    /// Whether to drop terminal states instead of storing them.
    drop_terminal_states: bool,
    /// Counters for terminal states (tracked even when dropping).
    pub avoided_count: u64,
    pub pruned_count: u64,
    pub deadended_count: u64,
    pub errored_count: u64,
    pub unconstrained_count: u64,
}

impl Default for StashManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StashManager {
    /// Create a new StashManager with default stashes.
    pub fn new() -> Self {
        let mut stashes = HashMap::new();
        stashes.insert(STASH_ACTIVE.to_string(), VecDeque::new());
        stashes.insert(STASH_FOUND.to_string(), VecDeque::new());
        stashes.insert(STASH_AVOID.to_string(), VecDeque::new());
        stashes.insert(STASH_DEADENDED.to_string(), VecDeque::new());
        stashes.insert(STASH_ERRORED.to_string(), VecDeque::new());
        stashes.insert(STASH_PRUNED.to_string(), VecDeque::new());
        stashes.insert(STASH_UNCONSTRAINED.to_string(), VecDeque::new());

        StashManager {
            stashes,
            state_index: HashMap::new(),
            state_roots: HashMap::new(),
            drop_terminal_states: false,
            avoided_count: 0,
            pruned_count: 0,
            deadended_count: 0,
            errored_count: 0,
            unconstrained_count: 0,
        }
    }

    // =========================================================================
    // Stash access
    // =========================================================================

    /// Get immutable reference to a stash by name.
    #[inline]
    pub fn get(&self, stash: &str) -> Option<&VecDeque<RustSimState>> {
        self.stashes.get(stash)
    }

    /// Get mutable reference to a stash by name.
    #[inline]
    pub fn get_mut(&mut self, stash: &str) -> Option<&mut VecDeque<RustSimState>> {
        self.stashes.get_mut(stash)
    }

    /// Get or create a stash, returning mutable reference.
    #[inline]
    pub fn entry(&mut self, stash: &str) -> &mut VecDeque<RustSimState> {
        self.stashes.entry(stash.to_string()).or_default()
    }

    /// Iterate over all stashes.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &VecDeque<RustSimState>)> {
        self.stashes.iter()
    }

    /// Iterate mutably over all stashes.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut VecDeque<RustSimState>)> {
        self.stashes.iter_mut()
    }

    /// Direct access to the underlying stashes HashMap for entry() API.
    #[inline]
    pub fn stashes_mut(&mut self) -> &mut HashMap<String, VecDeque<RustSimState>> {
        &mut self.stashes
    }

    /// Direct immutable access to the underlying stashes HashMap.
    #[inline]
    pub fn stashes(&self) -> &HashMap<String, VecDeque<RustSimState>> {
        &self.stashes
    }

    /// Remove a stash and return its contents.
    pub fn remove(&mut self, stash: &str) -> Option<VecDeque<RustSimState>> {
        self.stashes.remove(stash)
    }

    /// Insert a stash with given contents.
    pub fn insert(&mut self, stash: &str, states: VecDeque<RustSimState>) {
        self.stashes.insert(stash.to_string(), states);
    }

    /// Get or create a stash by name, returning a mutable reference. Emits a
    /// `log::warn!` the first time a non-standard stash name is created so
    /// that typos like `actve` for `active` are surfaced rather than silently
    /// vanishing into an invisible stash. Subsequent calls with the same name
    /// are silent — the stash exists in the map after the first creation.
    pub fn ensure_stash(&mut self, name: &str) -> &mut VecDeque<RustSimState> {
        if !self.stashes.contains_key(name) {
            log::warn!(
                "Rust exploration: creating new stash '{}' (not one of the \
                 standard stashes {:?}); if this is a typo, the state will be \
                 invisible to mgr.active / mgr.found / mgr.deadended etc.",
                name,
                STANDARD_STASHES,
            );
        }
        self.stashes.entry(name.to_string()).or_default()
    }

    // =========================================================================
    // Counts and queries
    // =========================================================================

    /// Number of states in the active stash.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.stashes.get(STASH_ACTIVE).map_or(0, |s| s.len())
    }

    /// Number of states in the found stash.
    #[inline]
    pub fn found_count(&self) -> usize {
        self.stashes.get(STASH_FOUND).map_or(0, |s| s.len())
    }

    /// Number of states in a named stash.
    #[inline]
    pub fn count(&self, stash: &str) -> usize {
        self.stashes.get(stash).map_or(0, |s| s.len())
    }

    /// Whether the active stash is non-empty.
    #[inline]
    pub fn has_active(&self) -> bool {
        self.stashes
            .get(STASH_ACTIVE)
            .is_some_and(|s| !s.is_empty())
    }

    /// Get state IDs in a stash.
    pub fn state_ids(&self, stash: &str) -> Vec<u64> {
        self.stashes
            .get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Get stash counts as a HashMap.
    pub fn counts(&self) -> HashMap<String, usize> {
        let mut result = HashMap::new();
        for (name, stash) in &self.stashes {
            result.insert(name.clone(), stash.len());
        }
        result
    }

    // =========================================================================
    // Push / Pop / Move
    // =========================================================================

    /// Push a state to a stash and index it.
    pub fn push(&mut self, stash: &str, state: RustSimState) {
        let state_id = state.state_id();
        self.index(state_id, stash);
        self.entry(stash).push_back(state);
    }

    /// Pop the front state from active (BFS) or back (DFS).
    pub fn pop_active(&mut self, use_lifo: bool) -> Option<RustSimState> {
        let stash = self.stashes.get_mut(STASH_ACTIVE)?;
        let state = if use_lifo {
            stash.pop_back()
        } else {
            stash.pop_front()
        };
        if let Some(ref s) = state {
            self.unindex(s.state_id());
        }
        state
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    ///
    /// **Invariant I6:** on the drop path, the corresponding `state_roots`
    /// entry must be removed to keep lineage tracking consistent with
    /// `state_index`. Both maps move together here. See module-level I6.
    pub fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        match stash_name {
            "avoid" => self.avoided_count += 1,
            "pruned" => self.pruned_count += 1,
            "deadended" => self.deadended_count += 1,
            "errored" => self.errored_count += 1,
            "unconstrained" => self.unconstrained_count += 1,
            _ => {}
        }
        if !self.drop_terminal_states {
            self.push(stash_name, state);
        } else {
            let sid = state.state_id();
            self.state_roots.remove(&sid);
            // I6 consistency: the dropped state should not remain in
            // state_index. The caller pops from a stash before invoking
            // us, which un-indexes via `pop_active`; this assert documents
            // that contract and catches a regression where a drop path
            // skips the un-index step.
            #[cfg(debug_assertions)]
            debug_assert!(
                !self.state_index.contains_key(&sid),
                "I6: state {} dropped while still indexed under stash {:?}",
                sid,
                self.state_index.get(&sid)
            );
            // State is dropped here, freeing its Z3 solver clone.
        }
    }

    /// Clear all states from a stash, removing index and root entries.
    pub fn clear(&mut self, stash: &str) {
        if let Some(s) = self.stashes.get_mut(stash) {
            for state in s.drain(..) {
                let sid = state.state_id();
                self.state_index.remove(&sid);
                self.state_roots.remove(&sid);
            }
        }
    }

    /// Set whether terminal states should be dropped instead of stored.
    pub fn set_drop_terminal_states(&mut self, drop: bool) {
        self.drop_terminal_states = drop;
    }

    pub fn drop_terminal_states(&self) -> bool {
        self.drop_terminal_states
    }

    // =========================================================================
    // Index operations
    // =========================================================================

    /// Track a state in the state_index.
    #[inline]
    pub fn index(&mut self, state_id: u64, stash: &str) {
        self.state_index.insert(state_id, stash.to_string());
    }

    /// Remove a state from the state_index.
    #[inline]
    pub fn unindex(&mut self, state_id: u64) {
        self.state_index.remove(&state_id);
    }

    /// Rebuild the state_index from scratch by scanning all stashes.
    pub fn rebuild_index(&mut self) {
        self.state_index.clear();
        for (stash_name, stash) in &self.stashes {
            for state in stash {
                self.state_index
                    .insert(state.state_id(), stash_name.clone());
            }
        }
    }

    /// Get the stash name for a state by ID.
    pub fn stash_of(&self, state_id: u64) -> Option<&str> {
        self.state_index.get(&state_id).map(|s| s.as_str())
    }

    // =========================================================================
    // State lookup
    // =========================================================================

    /// Find an immutable reference to a state by ID.
    pub fn find_state(&self, state_id: u64) -> Option<&RustSimState> {
        // Fast path: use index
        if let Some(stash_name) = self.state_index.get(&state_id)
            && let Some(stash) = self.stashes.get(stash_name)
        {
            for state in stash {
                if state.state_id() == state_id {
                    return Some(state);
                }
            }
        }
        // Slow fallback: linear scan
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Some(state);
                }
            }
        }
        None
    }

    /// Find a mutable reference to a state by ID.
    pub fn find_state_mut(&mut self, state_id: u64) -> Option<&mut RustSimState> {
        let stash_name = if let Some(name) = self.state_index.get(&state_id) {
            Some(name.clone())
        } else {
            let mut found = None;
            for (name, stash) in &self.stashes {
                for state in stash {
                    if state.state_id() == state_id {
                        found = Some(name.clone());
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found
        };

        if let Some(name) = stash_name
            && let Some(stash) = self.stashes.get_mut(&name)
        {
            for state in stash.iter_mut() {
                if state.state_id() == state_id {
                    return Some(state);
                }
            }
        }
        None
    }

    // =========================================================================
    // State roots (lineage tracking)
    // =========================================================================

    /// Set the root state for a state ID.
    #[inline]
    pub fn set_root(&mut self, state_id: u64, root_id: u64) {
        self.state_roots.insert(state_id, root_id);
    }

    /// Get the root state ID for a state.
    #[inline]
    pub fn get_root(&self, state_id: u64) -> Option<u64> {
        self.state_roots.get(&state_id).copied()
    }

    /// Remove a root tracking entry.
    #[inline]
    pub fn remove_root(&mut self, state_id: u64) {
        self.state_roots.remove(&state_id);
    }

    /// Get all state roots.
    pub fn roots(&self) -> &HashMap<u64, u64> {
        &self.state_roots
    }

    // =========================================================================
    // Snapshot / Serialization (angr-x04s.1.3)
    // =========================================================================

    /// Serialize this stash manager to a versioned envelope:
    /// `[STASH_SNAPSHOT_VERSION: u8] ++ serde_json(StashManagerSnapshot)`.
    ///
    /// Each state inside is round-tripped via [`RustSimState::to_snapshot`]
    /// (bucket A/B/C, see `RustSimStateSnapshot`); the manager-level fields
    /// captured here are the stash map, the lineage `state_roots`, the
    /// terminal counters, and the `drop_terminal_states` flag. The
    /// `state_index` is rebuilt from the dumped stashes on load.
    pub fn dump_snapshot(&self) -> Vec<u8> {
        let snap = self.to_snapshot();
        let body = serde_json::to_vec(&snap).expect("stash snapshot encode");
        let mut out = Vec::with_capacity(1 + body.len());
        out.push(STASH_SNAPSHOT_VERSION);
        out.extend_from_slice(&body);
        out
    }

    /// Inverse of [`Self::dump_snapshot`]. Rejects an empty envelope or a
    /// version-byte mismatch with [`crate::state::SnapshotError`].
    pub fn load_snapshot(bytes: &[u8]) -> Result<Self, crate::state::SnapshotError> {
        if bytes.is_empty() {
            return Err(crate::state::SnapshotError::EmptyEnvelope);
        }
        let version = bytes[0];
        if version != STASH_SNAPSHOT_VERSION {
            return Err(crate::state::SnapshotError::VersionMismatch {
                found: version,
                expected: STASH_SNAPSHOT_VERSION,
            });
        }
        // Mirror `RustSimState::from_serialized`: deep RustBV op-trees in
        // accumulated per-state constraints can blow past serde_json's
        // default recursion limit (128). The on-disk envelope is trusted
        // (written by our own `dump_snapshot`) so DoS hardening is not
        // load-bearing.
        let mut de = serde_json::Deserializer::from_slice(&bytes[1..]);
        de.disable_recursion_limit();
        let snap: StashManagerSnapshot = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| crate::state::SnapshotError::Decode(e.to_string()))?;
        Self::from_snapshot(snap).map_err(crate::state::SnapshotError::Decode)
    }

    /// Build a [`StashManagerSnapshot`] (in-Rust round-trip shape).
    pub fn to_snapshot(&self) -> StashManagerSnapshot {
        let stashes: std::collections::BTreeMap<String, Vec<crate::state::RustSimStateSnapshot>> =
            self.stashes
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().map(|s| s.to_snapshot()).collect()))
                .collect();
        let state_roots: std::collections::BTreeMap<u64, u64> =
            self.state_roots.iter().map(|(k, v)| (*k, *v)).collect();
        StashManagerSnapshot {
            stashes,
            state_roots,
            drop_terminal_states: self.drop_terminal_states,
            avoided_count: self.avoided_count,
            pruned_count: self.pruned_count,
            deadended_count: self.deadended_count,
            errored_count: self.errored_count,
            unconstrained_count: self.unconstrained_count,
        }
    }

    /// Restore a [`StashManagerSnapshot`] into a fresh manager. Rebuilds
    /// the `state_index` from the dumped stash contents so cross-stash
    /// state_id lookups stay consistent.
    pub fn from_snapshot(snap: StashManagerSnapshot) -> Result<Self, String> {
        let mut stashes: HashMap<String, VecDeque<RustSimState>> = HashMap::new();
        let mut state_index: HashMap<u64, String> = HashMap::new();
        for (stash_name, state_snaps) in snap.stashes {
            let mut deque: VecDeque<RustSimState> = VecDeque::with_capacity(state_snaps.len());
            for state_snap in state_snaps {
                let st = RustSimState::from_snapshot(state_snap)?;
                state_index.insert(st.state_id(), stash_name.clone());
                deque.push_back(st);
            }
            stashes.insert(stash_name, deque);
        }
        // Make sure every standard stash exists so callers can index without panic.
        for name in STANDARD_STASHES {
            stashes.entry((*name).to_string()).or_default();
        }
        let state_roots: HashMap<u64, u64> = snap.state_roots.into_iter().collect();
        Ok(StashManager {
            stashes,
            state_index,
            state_roots,
            drop_terminal_states: snap.drop_terminal_states,
            avoided_count: snap.avoided_count,
            pruned_count: snap.pruned_count,
            deadended_count: snap.deadended_count,
            errored_count: snap.errored_count,
            unconstrained_count: snap.unconstrained_count,
        })
    }
}

/// Format-version byte at the head of every
/// [`StashManager::dump_snapshot`] envelope. Independent of
/// [`crate::state::SNAPSHOT_VERSION`] (the per-state codec) so the two
/// can evolve without churn at the wrong layer.
pub const STASH_SNAPSHOT_VERSION: u8 = 1;

/// Manager-level snapshot of all stash contents + lineage maps + terminal
/// counters. Each entry inside `stashes` is a per-state
/// [`crate::state::RustSimStateSnapshot`].
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct StashManagerSnapshot {
    pub stashes: std::collections::BTreeMap<String, Vec<crate::state::RustSimStateSnapshot>>,
    pub state_roots: std::collections::BTreeMap<u64, u64>,
    pub drop_terminal_states: bool,
    pub avoided_count: u64,
    pub pruned_count: u64,
    pub deadended_count: u64,
    pub errored_count: u64,
    pub unconstrained_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_default_stashes() {
        let mgr = StashManager::new();
        assert!(mgr.get(STASH_ACTIVE).is_some());
        assert!(mgr.get(STASH_FOUND).is_some());
        assert!(mgr.get(STASH_AVOID).is_some());
        assert!(mgr.get(STASH_DEADENDED).is_some());
        assert!(mgr.get(STASH_ERRORED).is_some());
        assert!(mgr.get(STASH_PRUNED).is_some());
        assert!(mgr.get(STASH_UNCONSTRAINED).is_some());
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.found_count(), 0);
    }

    #[test]
    fn test_counts() {
        let mgr = StashManager::new();
        let counts = mgr.counts();
        assert_eq!(counts.len(), 7);
        for (_, &count) in &counts {
            assert_eq!(count, 0);
        }
    }

    /// Round-trip a non-empty StashManager via the per-state codec.
    ///
    /// Pushes two states into `active` and one into `found`, then dumps via
    /// `dump_snapshot` → `load_snapshot` and asserts (a) stash counts
    /// match, (b) state_ids survive in their respective stashes, (c) the
    /// `state_index` is rebuilt so `stash_of` answers consistently.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_stash_manager_dump_load_round_trip() {
        let mut mgr = StashManager::new();
        let s1 = RustSimState::new("amd64").unwrap();
        let s2 = RustSimState::new("amd64").unwrap();
        let s3 = RustSimState::new("amd64").unwrap();
        let id1 = s1.state_id();
        let id2 = s2.state_id();
        let id3 = s3.state_id();

        mgr.push(STASH_ACTIVE, s1);
        mgr.push(STASH_ACTIVE, s2);
        mgr.push(STASH_FOUND, s3);
        mgr.set_root(id2, id1);
        mgr.avoided_count = 4;
        mgr.deadended_count = 7;

        let bytes = mgr.dump_snapshot();
        assert_eq!(
            bytes[0], STASH_SNAPSHOT_VERSION,
            "stash envelope must carry version byte"
        );
        let restored = StashManager::load_snapshot(&bytes).expect("load_snapshot");

        assert_eq!(restored.count(STASH_ACTIVE), 2);
        assert_eq!(restored.count(STASH_FOUND), 1);
        assert_eq!(restored.avoided_count, 4);
        assert_eq!(restored.deadended_count, 7);

        // state_index rebuilt: stash_of() answers for every id.
        assert_eq!(restored.stash_of(id1), Some(STASH_ACTIVE));
        assert_eq!(restored.stash_of(id2), Some(STASH_ACTIVE));
        assert_eq!(restored.stash_of(id3), Some(STASH_FOUND));

        // state_roots survived.
        assert_eq!(restored.get_root(id2), Some(id1));
    }

    /// Bad envelope handling: empty bytes and mismatched version byte both
    /// route through the typed `SnapshotError` variants.
    #[test]
    fn test_stash_manager_load_snapshot_errors() {
        match StashManager::load_snapshot(&[]) {
            Err(crate::state::SnapshotError::EmptyEnvelope) => {}
            Err(other) => panic!("expected EmptyEnvelope, got {other:?}"),
            Ok(_) => panic!("empty envelope must fail"),
        }

        let mgr = StashManager::new();
        let mut bytes = mgr.dump_snapshot();
        bytes[0] = STASH_SNAPSHOT_VERSION.wrapping_add(7);
        match StashManager::load_snapshot(&bytes) {
            Err(crate::state::SnapshotError::VersionMismatch { found, expected }) => {
                assert_eq!(expected, STASH_SNAPSHOT_VERSION);
                assert_eq!(found, STASH_SNAPSHOT_VERSION.wrapping_add(7));
            }
            Err(other) => panic!("expected VersionMismatch, got {other:?}"),
            Ok(_) => panic!("bad version must fail"),
        }
    }
}
