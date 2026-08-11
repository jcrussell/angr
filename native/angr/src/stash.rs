//! Stash management for symbolic execution state tracking.
//!
//! Provides a StashManager that owns all stash-related data:
//! - Named stashes (active, found, avoid, deadended, errored, unconstrained)
//! - State index for O(1) lookups by state_id
//! - State root tracking for fork lineage
//! - Terminal state counters
//!
//! **Invariant I6 (cross-mixin, mirror of the `I6. State-cache pinning +
//! manager-vs-mixin override` entry in the
//! `angr/exploration/rust_manager.py` module docstring):** the
//! Python-side `_cleanup_state_cache` orchestrates eviction with pinning
//! over `_state_roots ∪ {_current_callback_state_id,
//! _current_stepping_state_id}`. The Rust counterpart here owns the
//! `state_index` and `state_roots` maps that back lineage tracking. Both
//! maps must move together — every push/pop on a stash that participates
//! in lineage must update both, and `remove_state` / `clear` / `insert` /
//! `push_or_drop_terminal` are the chokepoints that enforce this.
//! Without consistent maps, `_state_roots` drift produces orphaned roots
//! and a freshly-mutated state can race-evict between callbacks on the
//! same state. Regression tests:
//! `TestStateCacheSizeBound.test_cleanup_state_cache_evicts_oldest_first`,
//! `..._drops_dead_states`, `..._skips_pinned`
//! (tests/engines/rust/test_plugins.py).
//!
//! **Panic policy (angr-9ke6b.212):** [`StashManager::load_snapshot`] takes
//! untrusted bytes and is fully `Result`-typed ([`crate::state::SnapshotError`])
//! — including the 8-byte origin-token header, which routes a short slice
//! through `SnapshotError::Decode` rather than a slice-conversion panic. The
//! one surviving `expect` is on the *encode* side; see its `#[allow]`.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use rustc_hash::FxHashMap;
use std::collections::{HashMap, VecDeque};

use crate::exploration::selection_policy::SelectionPolicy;
use crate::state::RustSimState;

/// Well-known stash names.
pub const STASH_ACTIVE: &str = "active";
pub const STASH_FOUND: &str = "found";
pub const STASH_AVOID: &str = "avoid";
pub const STASH_DEADENDED: &str = "deadended";
pub const STASH_ERRORED: &str = "errored";
pub const STASH_PRUNED: &str = "pruned";
pub const STASH_UNCONSTRAINED: &str = "unconstrained";
/// Quarantine stash holding the successors of a `step_state()` call (E1.a).
/// `step_state` deliberately does NOT auto-stash: the Python caller owns
/// placement and moves each returned state out with `move_state()`.
pub const STASH_STEP_OUT: &str = "_step_out";

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
    /// `u64`-keyed → FxHashMap (faster than SipHash on the lineage hot path).
    state_index: FxHashMap<u64, String>,
    /// Maps state_id -> root_state_id for lineage tracking.
    /// `u64`-keyed → FxHashMap (see `state_index`).
    state_roots: FxHashMap<u64, u64>,
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
            state_index: FxHashMap::default(),
            state_roots: FxHashMap::default(),
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

    /// Insert a stash with given contents, replacing any existing stash of
    /// that name.
    ///
    /// **Invariant I6 chokepoint (angr-03vl4.80):** unlike a bare
    /// `stashes.insert`, this re-points `state_index` at `stash` for every
    /// state in `states`, and drops the `state_index` / `state_roots` entries
    /// of the states it evicts — the same bookkeeping [`Self::clear`] does —
    /// so the maps cannot desync from the stashes when a caller replaces a
    /// non-empty stash. Its one in-tree caller (`_move_states` in
    /// `exploration/state_lifecycle.rs`) passes an empty deque into a name it
    /// has just [`Self::remove`]d and re-indexed by hand, so both loops are
    /// no-ops there; they exist so a future caller that does neither stays
    /// correct.
    pub fn insert(&mut self, stash: &str, states: VecDeque<RustSimState>) {
        if let Some(evicted) = self.stashes.get(stash) {
            for state in evicted {
                let sid = state.state_id();
                self.state_index.remove(&sid);
                self.state_roots.remove(&sid);
            }
        }
        for state in &states {
            self.state_index.insert(state.state_id(), stash.to_string());
        }
        self.stashes.insert(stash.to_string(), states);
    }

    /// Get or create a stash by name, returning a mutable reference. Emits a
    /// `log::warn!` the first time a non-standard stash name is created so
    /// that typos like `actve` for `active` are surfaced rather than silently
    /// vanishing into an invisible stash. Subsequent calls with the same name
    /// are silent — the stash exists in the map after the first creation.
    ///
    /// [`STASH_STEP_OUT`] is created on demand (not pre-registered — it must
    /// not show up in `stash_counts()` for managers that never call
    /// `step_state()`), so it is exempt from the typo warning.
    pub fn ensure_stash(&mut self, name: &str) -> &mut VecDeque<RustSimState> {
        if !self.stashes.contains_key(name) && name != STASH_STEP_OUT {
            log::warn!(
                "Rust exploration: creating new stash '{name}' (not one of the \
                 standard stashes {STANDARD_STASHES:?}); if this is a typo, the state will be \
                 invisible to mgr.active / mgr.found / mgr.deadended etc.",
            );
        }
        self.declare_stash(name)
    }

    /// Get or create a stash by name **without** [`Self::ensure_stash`]'s
    /// typo warning.
    ///
    /// For engine-internal stash names the caller already knows are
    /// deliberate — `"cut"`, `"timeout"`, `"not_unique"`,
    /// `merge_waiting_<addr>` — where the warning would be pure noise. The
    /// native-technique registrars in `manager_methods_techniques.rs` call
    /// this at registration time, which is also what keeps the later
    /// `ensure_stash` push in `native_technique.rs` quiet: the stash already
    /// exists by then (angr-sqfj8.35).
    ///
    /// Prefer `ensure_stash` whenever the name originated in Python and a
    /// typo is possible.
    pub fn declare_stash(&mut self, name: &str) -> &mut VecDeque<RustSimState> {
        self.stashes.entry(name.to_string()).or_default()
    }

    // =========================================================================
    // Counts and queries
    // =========================================================================

    /// Number of states in the active stash.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.stashes
            .get(STASH_ACTIVE)
            .map_or(0, std::collections::VecDeque::len)
    }

    /// Number of states in the found stash.
    #[inline]
    pub fn found_count(&self) -> usize {
        self.stashes
            .get(STASH_FOUND)
            .map_or(0, std::collections::VecDeque::len)
    }

    /// Number of states in a named stash.
    #[inline]
    pub fn count(&self, stash: &str) -> usize {
        self.stashes
            .get(stash)
            .map_or(0, std::collections::VecDeque::len)
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
            .map(|s| s.iter().map(super::state::RustSimState::state_id).collect())
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

    /// Pop the next active state chosen by `policy` (BFS front / DFS back /
    /// future coverage-guided / …), keeping the state index in sync.
    pub fn pop_active(&mut self, policy: &dyn SelectionPolicy) -> Option<RustSimState> {
        let stash = self.stashes.get_mut(STASH_ACTIVE)?;
        let state = policy.select(stash);
        if let Some(ref s) = state {
            self.unindex(s.state_id());
        }
        state
    }

    /// Push a freshly-forked successor onto the active stash at the position
    /// chosen by `policy` (tail for the FIFO/LIFO built-ins), keeping the
    /// state index in sync. The fork-insertion chokepoint (angr-a32jl.1).
    pub fn push_active(&mut self, policy: &dyn SelectionPolicy, state: RustSimState) {
        let state_id = state.state_id();
        self.index(state_id, STASH_ACTIVE);
        policy.on_fork(self.entry(STASH_ACTIVE), state);
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    ///
    /// **Invariant I6:** on the drop path, the corresponding `state_roots`
    /// entry must be removed to keep lineage tracking consistent with
    /// `state_index`. Both maps move together here. See module-level I6.
    ///
    /// # Panics
    ///
    /// On `STASH_ERRORED` (angr-sqfj8.136). Errored states are the one terminal
    /// disposition that must survive `drop_terminal_states`, so routing one
    /// here would silently discard a diagnostic the caller always needs. The
    /// arm used to just bump `errored_count`, which made the wrong call look
    /// correct at the call site; `push_errored` is the only sanctioned path.
    pub fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        match stash_name {
            STASH_AVOID => self.avoided_count += 1,
            STASH_PRUNED => self.pruned_count += 1,
            STASH_DEADENDED => self.deadended_count += 1,
            STASH_ERRORED => unreachable!(
                "errored states must be pushed via StashManager::push_errored: \
                 push_or_drop_terminal would drop them under drop_terminal_states, \
                 violating module invariant I6 (errored is never dropped)"
            ),
            STASH_UNCONSTRAINED => self.unconstrained_count += 1,
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
            // skips the un-index step. Always-on (angr-9ke6b.220): a stale
            // index entry pointing at a dropped state makes `find_state`
            // silently resolve to nothing (or, after ID churn, to the wrong
            // state). One hash lookup per dropped terminal.
            assert!(
                !self.state_index.contains_key(&sid),
                "I6: state {} dropped while still indexed under stash {:?}",
                sid,
                self.state_index.get(&sid)
            );
            // State is dropped here, freeing its Z3 solver clone.
        }
    }

    /// Push a state onto `STASH_ERRORED`, incrementing `errored_count`.
    ///
    /// Errored states are the one terminal disposition that is **never**
    /// dropped (`drop_terminal_states` does not apply — an error is a
    /// diagnostic the caller always needs), so they cannot go through
    /// `push_or_drop_terminal`. Before angr-9ke6b.231 the serial sites open-coded
    /// `stashes_mut().entry(STASH_ERRORED).push_back(...)`, which skipped both
    /// the counter and the index: a serial run's `StashSnapshot` reported
    /// `errored_count == 0` while the parallel loop folded
    /// `stats.summarized_errored` into the same field. This method is the
    /// errored-only counterpart of `push_or_drop_terminal`; route every errored
    /// push through it — since angr-sqfj8.136 that is enforced, not merely
    /// documented: `push_or_drop_terminal` panics on `STASH_ERRORED`.
    pub fn push_errored(&mut self, state: RustSimState) {
        self.errored_count += 1;
        self.push(STASH_ERRORED, state);
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
        self.state_index
            .get(&state_id)
            .map(std::string::String::as_str)
    }

    // =========================================================================
    // State lookup
    // =========================================================================

    /// Resolve `(stash name, position)` for a state, treating `state_index` as
    /// a **hint** rather than an authority.
    ///
    /// The indexed stash is consulted first, but only accepted once the state
    /// is actually found in it: a *present-but-stale* entry (index says stash
    /// A, the state now lives in stash B) falls through to the full scan
    /// exactly as a missing entry does. Before angr-c7xno.97 only `find_state`
    /// self-healed this way, while `find_state_mut` / `take_state` trusted a
    /// present entry and returned `None` for a live state — an observable
    /// read-vs-write divergence whenever a raw stash move skipped
    /// [`index`](Self::index) (the `NativeTechnique` `Timeout` /
    /// `evict_active_states` bypass). Single-sourcing the lookup here means a
    /// future bypass degrades to an O(n) scan for all three, never to a
    /// spurious "state not found".
    fn locate_state(&self, state_id: u64) -> Option<(&str, usize)> {
        if let Some(stash_name) = self.state_index.get(&state_id)
            && let Some(stash) = self.stashes.get(stash_name)
            && let Some(idx) = stash.iter().position(|s| s.state_id() == state_id)
        {
            return Some((stash_name.as_str(), idx));
        }
        // Slow fallback: linear scan (index missing or stale).
        self.stashes.iter().find_map(|(name, stash)| {
            stash
                .iter()
                .position(|s| s.state_id() == state_id)
                .map(|idx| (name.as_str(), idx))
        })
    }

    /// Like [`locate_state`](Self::locate_state) but yields an owned stash
    /// name, releasing the borrow so the caller can take `&mut self`.
    fn locate_state_owned(&self, state_id: u64) -> Option<(String, usize)> {
        self.locate_state(state_id)
            .map(|(name, idx)| (name.to_string(), idx))
    }

    /// Find an immutable reference to a state by ID.
    pub fn find_state(&self, state_id: u64) -> Option<&RustSimState> {
        let (stash_name, idx) = self.locate_state(state_id)?;
        self.stashes.get(stash_name)?.get(idx)
    }

    /// Remove a state from whichever stash holds it and return it by value.
    ///
    /// Used by `step_state()` (E1.a), which steps a state out-of-band: the
    /// state must leave its stash for the duration of the step, exactly as the
    /// run loop's `pop_active` takes it off the active stash.
    pub fn take_state(&mut self, state_id: u64) -> Option<RustSimState> {
        let (stash_name, idx) = self.locate_state_owned(state_id)?;
        let state = self.stashes.get_mut(&stash_name)?.remove(idx);
        self.unindex(state_id);
        state
    }

    /// Remove `state_id` from a *specific* `stash`, returning it by value and
    /// clearing its `state_index` entry.
    ///
    /// Unlike [`take_state`](Self::take_state), this does not fall back to
    /// scanning other stashes — the caller asserts which stash holds the state,
    /// so a state living elsewhere yields `None` rather than being silently
    /// relocated. Root bookkeeping is left to the caller: a drop clears the
    /// root, a move preserves it. Single-sources the find-index-then-remove
    /// dance open-coded by `drop_state_from_stash` and `_move_state`
    /// (angr-ph300.27).
    pub fn take_state_from(&mut self, state_id: u64, stash: &str) -> Option<RustSimState> {
        let s = self.stashes.get_mut(stash)?;
        let idx = s.iter().position(|st| st.state_id() == state_id)?;
        let state = s.remove(idx);
        self.unindex(state_id);
        state
    }

    /// Find a mutable reference to a state by ID.
    pub fn find_state_mut(&mut self, state_id: u64) -> Option<&mut RustSimState> {
        let (stash_name, idx) = self.locate_state_owned(state_id)?;
        self.stashes.get_mut(&stash_name)?.get_mut(idx)
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

    /// Resolve a state's lineage root, falling back to the state itself
    /// when no root mapping is recorded.
    #[inline]
    pub fn root_or_self(&self, state_id: u64) -> u64 {
        self.get_root(state_id).unwrap_or(state_id)
    }

    /// Remove a root tracking entry.
    #[inline]
    pub fn remove_root(&mut self, state_id: u64) {
        self.state_roots.remove(&state_id);
    }

    /// Get all state roots.
    pub fn roots(&self) -> &FxHashMap<u64, u64> {
        &self.state_roots
    }

    // =========================================================================
    // Snapshot / Serialization (angr-x04s.1.3)
    // =========================================================================

    /// Serialize this stash manager to a versioned envelope:
    /// `[STASH_SNAPSHOT_VERSION: u8] ++ [process_token: u64 LE] ++
    /// serde_json(StashManagerSnapshot)`.
    ///
    /// The token rides in the header — not the JSON body — because
    /// [`Self::load_snapshot`] must know whether the id space is foreign
    /// *before* it deserializes the first [`crate::symbolic::RustBV`]
    /// (angr-euw28).
    ///
    /// Each state inside is round-tripped via [`RustSimState::to_snapshot`]
    /// (bucket A/B/C, see `RustSimStateSnapshot`); the manager-level fields
    /// captured here are the stash map, the lineage `state_roots`, the
    /// terminal counters, and the `drop_terminal_states` flag. The
    /// `state_index` is rebuilt from the dumped stashes on load.
    #[allow(
        clippy::expect_used,
        reason = "`serde_json::to_vec` over `StashManagerSnapshot`, whose derived `Serialize` has no fallible arm (no non-string map keys, no custom impl) — the only `Err` shape is an io error, which cannot arise writing to a `Vec`. Left as a panic rather than propagated because `dump_snapshot` is the pyclass-facing API returning `Vec<u8>`; giving it a `Result` would ripple through the PyO3 surface and its callers, which is out of scope for angr-9ke6b.212"
    )]
    pub fn dump_snapshot(&mut self) -> Vec<u8> {
        let snap = self.to_snapshot();
        let body = serde_json::to_vec(&snap).expect("stash snapshot encode");
        let mut out = Vec::with_capacity(1 + 8 + body.len());
        out.push(STASH_SNAPSHOT_VERSION);
        out.extend_from_slice(&process_token().to_le_bytes());
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
        if bytes.len() < 9 {
            return Err(crate::state::SnapshotError::Decode(
                "truncated stash snapshot header".to_string(),
            ));
        }
        // `bytes.len() >= 9` was just checked, so the slice is exactly 8 bytes;
        // route the impossible case through the same Decode error anyway rather
        // than panicking on caller-supplied bytes.
        let Ok(origin_bytes) = <[u8; 8]>::try_from(&bytes[1..9]) else {
            return Err(crate::state::SnapshotError::Decode(
                "truncated stash snapshot header".to_string(),
            ));
        };
        let origin = u64::from_le_bytes(origin_bytes);

        // angr-euw28: a foreign process minted these ids from an allocator that
        // also started at 0, so they collide with ids we already handed out (the
        // seed state's own symbols). Shift the whole envelope's id space above
        // our watermark for the duration of the load — deserialization AND the
        // `restore_from_snapshot` replay, which reserves past the rebased top.
        let _rebase = (origin != process_token()).then(|| {
            crate::symbolic::SymbolIdRebase::activate(crate::symbolic::symbol_id_watermark())
        });

        // Mirror `RustSimState::from_serialized`: deep RustBV op-trees in
        // accumulated per-state constraints can blow past serde_json's
        // default recursion limit (128). The on-disk envelope is trusted
        // (written by our own `dump_snapshot`) so DoS hardening is not
        // load-bearing.
        let mut de = serde_json::Deserializer::from_slice(&bytes[9..]);
        de.disable_recursion_limit();
        let snap: StashManagerSnapshot = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| crate::state::SnapshotError::Decode(e.to_string()))?;
        Self::from_snapshot(snap)
    }

    /// Build a [`StashManagerSnapshot`] (in-Rust round-trip shape).
    pub fn to_snapshot(&mut self) -> StashManagerSnapshot {
        let stashes: std::collections::BTreeMap<String, Vec<crate::state::RustSimStateSnapshot>> =
            self.stashes
                .iter_mut()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.iter_mut()
                            .map(super::state::RustSimState::to_snapshot)
                            .collect(),
                    )
                })
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
    pub fn from_snapshot(snap: StashManagerSnapshot) -> Result<Self, crate::state::SnapshotError> {
        let mut stashes: HashMap<String, VecDeque<RustSimState>> = HashMap::new();
        let mut state_index: FxHashMap<u64, String> = FxHashMap::default();
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
        let state_roots: FxHashMap<u64, u64> = snap.state_roots.into_iter().collect();
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
///
/// v2 (angr-euw28) inserted the 8-byte little-endian [`process_token`] between
/// the version byte and the JSON body.
pub const STASH_SNAPSHOT_VERSION: u8 = 2;

/// Token identifying the process that wrote a snapshot envelope (angr-euw28).
///
/// Symbol ids are unique per *process* ([`crate::symbolic::reserve_symbol_id`]),
/// so ids in an envelope written by this process are still ours on load and
/// must be kept verbatim — while ids in a *foreign* envelope come from an
/// allocator that also started at 0 and therefore alias our own. `load_snapshot`
/// compares this token against the envelope's to decide whether to rebase the
/// restored id space (see [`crate::symbolic::SymbolIdRebase`]).
///
/// Derived from the pid plus the process start-up wall clock, which is enough
/// to distinguish two live processes (and a recycled pid on a later run).
pub fn process_token() -> u64 {
    static TOKEN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *TOKEN.get_or_init(|| {
        // SILENT(cat-b): a system clock reading before the Unix epoch makes
        // `duration_since` fail. Taking the error's magnitude keeps the
        // wall-clock entropy the token relies on (a plain `0` fallback would
        // collapse the token to just the pid, so a recycled pid on a later
        // pre-epoch run could alias ours and suppress the `SymbolIdRebase` in
        // `StashManager::load_snapshot`). Lossy only in sign: two clocks
        // equidistant either side of the epoch hash alike (angr-sqfj8.137).
        let nanos = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as u64,
            Err(e) => {
                log::warn!(
                    "system clock reads before the Unix epoch; process token falls back to the \
                     pre-epoch offset ({} ns) — snapshot id-space rebasing may misjudge a foreign \
                     envelope written by a same-pid process with a mirrored clock skew",
                    e.duration().as_nanos()
                );
                e.duration().as_nanos() as u64
            }
        };
        nanos.rotate_left(17) ^ u64::from(std::process::id())
    })
}

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

test_submod!("stash_tests.rs" => tests);
