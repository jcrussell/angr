//! Unified Symbolic Identity Registry.
//!
//! This module provides a registry for preserving symbolic identity across
//! the Python<->Rust boundary. When a claripy AST is converted to RustBV
//! and back, it should return the original Python object, not a new symbol.
//!
//! # Problem
//!
//! Without identity preservation:
//! 1. Python creates `x = BVS("x", 32)`
//! 2. Rust converts to RustBV with id=42
//! 3. On export, Rust creates NEW `BVS("rust_sym_42", 32)`
//! 4. Constraints on original `x` don't apply to the new symbol!
//!
//! # Solution
//!
//! The `SymbolicIdentityRegistry` maintains bidirectional mappings:
//! - `py_hash_to_rust_id`: Python AST hash -> Rust symbol ID
//! - `rust_id_to_py`: Rust symbol ID -> Original Python AST
//! - `name_to_info`: Symbol name+width+sort -> (id, width, sort) for name-based lookup
//!
//! On import: Check registry before creating new symbol
//! On export: Return original `Py<PyAny>` if in registry

use parking_lot::RwLock;
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Live-symbol counts at which [`SymbolicIdentityRegistry::register`] emits a
/// one-shot unbounded-growth warning.
///
/// The registry has no production GC caller (see [`SymbolicIdentityRegistry::retain`]
/// for what one would still cost), so an exploration that
/// mints fresh symbols without bound grows all four maps without bound, each
/// entry pinning a claripy AST alive from the Rust side. That is invisible
/// today; warning once per order of magnitude makes it loud instead of
/// silently degrading, and gives a future GC a metric to gate on.
///
/// **Measured cost per entry ≈ 1.7 KB** (angr-9ke6b.224, 2026-08-01): ~1.44 KB
/// is the pinned claripy leaf (RSS delta over 100k `BVS(.., 32,
/// explicit_name=True)` leaves), the remaining ~0.3 KB the four Rust maps. So
/// these thresholds are ~170 MB and ~1.7 GB of resident memory. A third
/// order-of-magnitude step used to sit at 10 million entries — ~17 GB, past the
/// point any host survives to log it — so it is gone; 1 million is the last
/// count a process can plausibly reach and still warn.
const GROWTH_WARN_THRESHOLDS: [usize; 2] = [100_000, 1_000_000];

/// Claripy-side sort of a registered symbol.
///
/// `name_to_info` is keyed by symbol name, and a claripy `BoolS` leaf imports
/// with a hardcoded width of 1 (the `"BoolS"` arm of
/// `claripy_bridge::import::claripy_to_rustbv_depth`). Without a sort tag in
/// the key, `BVS("flag", 1)` and `BoolS("flag")` — same explicit name — would
/// collide, and the second import would resolve to the first's `rust_id`.
/// Export then answers `get_original_ast` for that id with whichever claripy
/// AST registered first, handing a `BV` back where the caller had a `Bool`
/// (angr-9ke6b.38).
///
/// Separating identity is not enough on its own: `RustBV` has no Bool sort, so
/// a Bool leaf is modelled as a width-1 `Symbolic` whose Z3 term
/// `RustBV::from_parts` builds from the leaf's *name*. Two leaves with distinct
/// `rust_id`s but the same name would therefore still be *one* variable to Z3.
/// [`SymbolKind::rust_symbol_name`] closes that by tagging the Rust-side name
/// of a Bool leaf, which `from_parts` decodes back into a real Bool-sorted Z3
/// constant (angr-9ke6b.223).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// A claripy `BVS` bitvector leaf.
    BitVector,
    /// A claripy `BoolS` boolean leaf.
    Bool,
}

/// Prefix that namespaces the Rust-side name of a claripy `Bool` leaf.
///
/// Chosen to contain `!`, which claripy's own auto-renamer never emits (it
/// appends `_<counter>_<width>`), so a mangled name can only collide with a
/// caller that passes `explicit_name=True` and a literal `!bool!` prefix.
const BOOL_NAME_PREFIX: &str = "!bool!";

impl SymbolKind {
    /// Short tag used to namespace the `name_to_info` key.
    fn tag(self) -> &'static str {
        match self {
            SymbolKind::BitVector => "bv",
            SymbolKind::Bool => "bool",
        }
    }

    /// Map a claripy symbol name to the name Rust uses for it.
    ///
    /// `RustBV` has no Bool sort: a Bool leaf is a width-1 `Symbolic`, and its
    /// Z3 term is built from the *name*. The name therefore has to carry the
    /// sort, or `BVS("x", 1, explicit_name=True)` and `BoolS("x",
    /// explicit_name=True)` collapse to one Z3 constant even though the
    /// registry hands them separate `rust_id`s, and constraining one silently
    /// constrains the other (angr-9ke6b.223).
    ///
    /// `BitVector` is the identity, so every existing name — and every Z3 term
    /// already built from one — is unchanged. Callers must apply this *before*
    /// touching the registry so the `name_to_info` key, `rust_id_to_name`, the
    /// `RustBV` name and the Z3 term all agree; `lookup_symbol_name_by_id` then
    /// hands importers back the same tagged name, which is what keeps
    /// re-imports bound to the same Z3 variable (angr-izov2).
    pub fn rust_symbol_name(self, name: &str) -> Cow<'_, str> {
        match self {
            SymbolKind::BitVector => Cow::Borrowed(name),
            SymbolKind::Bool => Cow::Owned(format!("{BOOL_NAME_PREFIX}{name}")),
        }
    }
}

/// Recover the claripy name from a Bool leaf's Rust name, or `None` for a
/// bitvector leaf.
///
/// Inverse of [`SymbolKind::rust_symbol_name`], and the reason the tag lives in
/// the name rather than in a `RustBV` field: `RustBV::from_parts` is the single
/// place that rebuilds a leaf's Z3 term — from the constructors, from
/// `symbolic_with_id`, and from the `RustBVData` deserialize arm — and the name
/// is the only part of a leaf all three already carry, so the sort survives a
/// snapshot round-trip for free.
///
/// The name it returns is the *claripy* one on purpose. `from_parts` builds
/// `Bool::new_const(claripy_name)`, which is bit-for-bit the declaration
/// claripy's own z3 backend emits for `BoolS(claripy_name)`. That matters
/// because `RustSolverContext::add_constraint_ast` has a raw-passthrough fast
/// path that hands claripy's Z3 term straight to the solver: without the match,
/// a constraint added through the fast path and a leaf imported through
/// `claripy_to_rustbv` would name two unrelated Z3 declarations and the
/// constraint would silently fail to bind.
pub(crate) fn strip_bool_symbol_name(name: &str) -> Option<&str> {
    name.strip_prefix(BOOL_NAME_PREFIX)
}

/// Build the `name_to_info` key for a symbol.
///
/// Single source of truth for the key format so `register` and
/// `lookup_by_name_and_width` cannot drift apart.
fn qualified_key(name: &str, width: u32, kind: SymbolKind) -> String {
    format!("{}:{name}_w{width}", kind.tag())
}

/// Information about a registered symbol.
#[derive(Clone, Debug)]
pub struct SymbolInfo {
    /// Rust-side symbol ID.
    pub rust_id: u64,
    /// Bit width of the symbol.
    pub width: u32,
    /// Claripy-side sort the symbol was registered under.
    pub kind: SymbolKind,
    /// Original Python hash (for reverse lookup).
    pub py_hash: i64,
}

/// Unified registry for preserving symbolic identity across FFI boundary.
///
/// This registry is designed to be shared across the exploration manager
/// and all state instances, ensuring consistent symbol identity.
pub struct SymbolicIdentityRegistry {
    /// Map from Python AST hash to Rust symbol ID.
    /// Using hash instead of object ID because Python may reuse addresses.
    py_hash_to_rust_id: RwLock<HashMap<i64, u64>>,

    /// Map from Rust symbol ID to original Python AST.
    /// The `Py<PyAny>` is stored as a reference to preserve the original.
    rust_id_to_py: RwLock<HashMap<u64, Py<PyAny>>>,

    /// Map from symbol name to info (for name-based lookup).
    /// This is used when we receive a symbol by name and need to find
    /// its Rust ID and original Python AST.
    name_to_info: RwLock<HashMap<String, SymbolInfo>>,

    /// Map from Rust symbol ID to the Rust-side symbol NAME.
    ///
    /// A `RustBV::Symbolic`'s Z3 constant is built from its *name*
    /// (`RustBV::from_parts` → `z3::ast::BV::new_const(name, width)`), not from
    /// its id. Re-binding an imported claripy AST to an existing id while
    /// carrying claripy's own (renamed) string therefore yields a RustBV that
    /// *looks* like the original symbol at the RustBV/export layer but is a
    /// completely different variable to Z3 — every constraint on the original
    /// silently stops binding (angr-izov2). Importers use this map to recover
    /// the canonical Rust name whenever they resolve a symbol by id.
    rust_id_to_name: RwLock<HashMap<u64, String>>,

    /// Next available ID for new symbols.
    next_id: AtomicU64,

    /// Index of the next unfired entry in [`GROWTH_WARN_THRESHOLDS`].
    ///
    /// Advanced monotonically by `maybe_warn_growth` so each threshold warns
    /// at most once per registry lifetime; reset by [`SymbolicIdentityRegistry::clear`].
    growth_warn_idx: AtomicUsize,

    /// Statistics for debugging.
    stats: RwLock<RegistryStats>,
}

/// Statistics for registry operations.
#[derive(Default, Clone, Debug)]
pub struct RegistryStats {
    /// Number of successful identity preservations.
    pub identity_hits: u64,
    /// Number of new symbols registered.
    pub new_registrations: u64,
    /// Number of failed lookups.
    pub lookup_misses: u64,
}

impl Default for SymbolicIdentityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolicIdentityRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        SymbolicIdentityRegistry {
            py_hash_to_rust_id: RwLock::new(HashMap::new()),
            rust_id_to_py: RwLock::new(HashMap::new()),
            name_to_info: RwLock::new(HashMap::new()),
            rust_id_to_name: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            growth_warn_idx: AtomicUsize::new(0),
            stats: RwLock::new(RegistryStats::default()),
        }
    }

    /// Register a Python AST with its corresponding Rust symbol ID.
    ///
    /// # Arguments
    /// * `py_hash` - The Python AST's stable hash (__hash__)
    /// * `rust_id` - The Rust symbol ID
    /// * `name` - The symbol name
    /// * `width` - The bit width
    /// * `kind` - The claripy-side sort (see [`SymbolKind`])
    /// * `py_ast` - The original Python AST object
    pub fn register(
        &self,
        py_hash: i64,
        rust_id: u64,
        name: &str,
        width: u32,
        kind: SymbolKind,
        py_ast: Py<PyAny>,
    ) {
        // Store mappings. Each guard is scoped to its own statement so no two
        // of the four maps are ever write-locked at once here — `remove` and
        // `retain` acquire them in a different order (angr-zi35f.12).
        self.py_hash_to_rust_id.write().insert(py_hash, rust_id);
        let live = {
            let mut id_to_py = self.rust_id_to_py.write();
            id_to_py.insert(rust_id, py_ast);
            id_to_py.len()
        };
        // D2 Fix: Include width in the name key to prevent collisions
        // when symbols have the same name but different widths.
        // E.g., "x" with width 32 vs "x" with width 64 should not collide.
        // The sort tag additionally separates BVS(name, 1) from BoolS(name),
        // which share a width — see [`SymbolKind`].
        self.name_to_info.write().insert(
            qualified_key(name, width, kind),
            SymbolInfo {
                rust_id,
                width,
                kind,
                py_hash,
            },
        );
        self.rust_id_to_name
            .write()
            .insert(rust_id, name.to_string());

        // Update stats
        self.stats.write().new_registrations += 1;

        self.maybe_warn_growth(live);
    }

    /// Warn once per crossed entry of [`GROWTH_WARN_THRESHOLDS`].
    ///
    /// Split out of `register` so the growth policy is testable without
    /// actually minting a hundred thousand symbols.
    fn maybe_warn_growth(&self, live: usize) {
        // Fast path for the overwhelmingly common case: one relaxed load, no
        // branch into the CAS.
        let idx = self.growth_warn_idx.load(Ordering::Relaxed);
        let Some(&threshold) = GROWTH_WARN_THRESHOLDS.get(idx) else {
            return;
        };
        if live < threshold {
            return;
        }
        // Racing registrations must not both warn: only the thread that wins
        // the CAS logs. A loser simply skips — the next registration re-reads
        // the advanced index and warns for the following threshold if the
        // count already blew past it.
        if self
            .growth_warn_idx
            .compare_exchange(idx, idx + 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            log::warn!(
                "symbolic identity registry holds {live} live symbols (crossed {threshold}); \
                 it has no garbage collector and nothing clears it in production, so every \
                 symbol minted in this process stays pinned — along with its original \
                 claripy AST — for the process lifetime"
            );
        }
    }

    /// Look up the canonical Rust-side name of a registered symbol id.
    ///
    /// The name is what backs the symbol's Z3 constant, so any importer that
    /// resolves a symbol by id must rebuild it with THIS name rather than with
    /// whatever string the incoming claripy AST carries (angr-izov2).
    pub fn lookup_name_by_id(&self, rust_id: u64) -> Option<String> {
        self.rust_id_to_name.read().get(&rust_id).cloned()
    }

    /// Look up a Rust symbol ID by Python AST hash.
    ///
    /// Returns the Rust ID if the symbol was previously registered.
    pub fn lookup_by_hash(&self, py_hash: i64) -> Option<u64> {
        let result = self.py_hash_to_rust_id.read().get(&py_hash).copied();

        if result.is_some() {
            self.stats.write().identity_hits += 1;
        } else {
            self.stats.write().lookup_misses += 1;
        }

        result
    }

    /// Look up symbol info by name, width and sort.
    ///
    /// Returns the `SymbolInfo` if a symbol with this exact name, width and
    /// [`SymbolKind`] was registered. All three are part of the key: width
    /// separates `BVS("x", 32)` from `BVS("x", 64)` (the D2 fix), and the sort
    /// separates `BVS("x", 1)` from `BoolS("x")` (angr-9ke6b.38).
    pub fn lookup_by_name_and_width(
        &self,
        name: &str,
        width: u32,
        kind: SymbolKind,
    ) -> Option<SymbolInfo> {
        self.name_to_info
            .read()
            .get(&qualified_key(name, width, kind))
            .cloned()
    }

    /// Get the original Python AST for a Rust symbol ID.
    ///
    /// This is the critical method for identity preservation on export.
    /// If the symbol was imported from Python, return the original AST.
    pub fn get_original_ast(&self, rust_id: u64) -> Option<Py<PyAny>> {
        let result = self.rust_id_to_py.read().get(&rust_id).cloned();

        if result.is_some() {
            self.stats.write().identity_hits += 1;
        }

        result
    }

    /// Register a Python AST with just its Rust symbol ID.
    ///
    /// This is a simplified registration that doesn't require hash or name info.
    /// Used when storing claripy ASTs during conversion.
    pub fn register_by_id(&self, rust_id: u64, py_ast: Py<PyAny>) {
        self.rust_id_to_py.write().insert(rust_id, py_ast);
    }

    /// Check if a symbol ID has a registered Python AST.
    pub fn has_original(&self, rust_id: u64) -> bool {
        self.rust_id_to_py.read().contains_key(&rust_id)
    }

    /// Phase 4 Fix: Update hash mapping for an existing symbol.
    ///
    /// This is used when we find an existing symbol by name+width lookup
    /// but want to also register it under a new hash (e.g., when Python's
    /// hash changes but the symbol is still the same).
    ///
    /// # Arguments
    /// * `py_hash` - The new Python hash to map
    /// * `rust_id` - The existing Rust symbol ID
    pub fn update_hash_mapping(&self, py_hash: i64, rust_id: u64) {
        self.py_hash_to_rust_id.write().insert(py_hash, rust_id);
    }

    /// Allocate a new unique symbol ID.
    ///
    /// This is used when creating a symbol that doesn't have a Python origin.
    pub fn allocate_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Set the next ID to at least the given value.
    ///
    /// Used when importing symbols with existing IDs.
    pub fn ensure_id_at_least(&self, id: u64) {
        let mut current = self.next_id.load(Ordering::SeqCst);
        while current <= id {
            match self
                .next_id
                .compare_exchange(current, id + 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Get registry statistics.
    pub fn stats(&self) -> RegistryStats {
        self.stats.read().clone()
    }

    /// Clear the registry.
    ///
    /// This should be called when starting a new exploration to prevent
    /// stale mappings from previous runs.
    pub fn clear(&self) {
        self.py_hash_to_rust_id.write().clear();
        self.rust_id_to_py.write().clear();
        self.name_to_info.write().clear();
        self.rust_id_to_name.write().clear();
        self.growth_warn_idx.store(0, Ordering::SeqCst);
        *self.stats.write() = RegistryStats::default();
    }

    /// Get the number of registered symbols.
    pub fn len(&self) -> usize {
        self.rust_id_to_py.read().len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.rust_id_to_py.read().is_empty()
    }

    /// Remove a symbol by Rust ID.
    ///
    /// This is used for cleanup when a symbol is no longer referenced.
    pub fn remove(&self, rust_id: u64) {
        // Find and remove from all maps. Acquire only the primary map up
        // front; the absent-id path then takes no other write locks (and
        // avoids the 3-lock cross-lock window). The other two write guards
        // are bound only once the id is confirmed present.
        let mut id_to_py = self.rust_id_to_py.write();

        if id_to_py.remove(&rust_id).is_some() {
            // Drop the id→Rust-name entry too, so all four maps stay symmetric
            // under removal (angr-4xaga.3). Ids are process-global and never
            // reused, so a leaked entry is not a correctness bug today — but a
            // symmetric mutator keeps the invariant honest for any future GC
            // caller.
            self.rust_id_to_name.write().remove(&rust_id);

            // Each secondary map is scoped so its write guard drops before the
            // next map's is taken — the id→py primary guard already serializes
            // the whole removal, so we never hold two secondary write locks at
            // once across a filter/collect/loop (angr-zi35f.12; the 167yo.1 fix
            // left these two co-held from the same acquire point).
            {
                let mut hash_to_id = self.py_hash_to_rust_id.write();
                let hash_to_remove: Vec<i64> = hash_to_id
                    .iter()
                    .filter(|(_, id)| **id == rust_id)
                    .map(|(h, _)| *h)
                    .collect();
                for h in hash_to_remove {
                    hash_to_id.remove(&h);
                }
            }

            {
                let mut name_to_info = self.name_to_info.write();
                let names_to_remove: Vec<String> = name_to_info
                    .iter()
                    .filter(|(_, info)| info.rust_id == rust_id)
                    .map(|(n, _)| n.clone())
                    .collect();
                for n in names_to_remove {
                    name_to_info.remove(&n);
                }
            }
        }
    }

    /// Prune symbols not in the given set of active IDs.
    ///
    /// # What dropping a still-reachable id costs
    ///
    /// Object identity and any annotations attached to the collected leaf — not
    /// soundness. On the next export, `rustbv_to_claripy` misses the registry
    /// and re-mints the leaf; because it mints with `explicit_name=True`
    /// (angr-9ke6b.222) the claripy name is the Rust name verbatim, and
    /// `RustBV::from_parts` derives the Z3 constant from that name, so the
    /// re-minted leaf denotes the *same* Z3 variable and every constraint on the
    /// original still binds.
    ///
    /// That was not always true. Minting without `explicit_name` let claripy
    /// rename the leaf to `name_<counter>_<width>`, making a re-export a
    /// brand-new unconstrained variable — silently wrong, and the reason
    /// angr-9ke6b.40 ruled an approximate active set out. Keep the export path's
    /// `explicit_name` flag if you keep this method: it is the whole reason an
    /// approximate `active_ids` is now merely lossy.
    ///
    /// # Why there is no production caller yet
    ///
    /// Nothing computes an active set at a quiescent point in the exploration
    /// loop, and nothing clears the registry in production either
    /// (`reset_for_new_exploration` is `#[cfg(test)]` — angr-9ke6b.218 item 4),
    /// so the registry only ever grows for the life of the process. That growth
    /// is surfaced by `maybe_warn_growth` / the `symbol_registry_size` stat
    /// rather than collected. Wiring this up is a memory/fidelity trade now, no
    /// longer a correctness one.
    ///
    /// That trade was measured and declined (angr-9ke6b.224, 2026-08-01). Ten
    /// `mma_howtouse` iterations in one process (450 Callable invocations,
    /// `tests/benchmarks/run_leak_check.py`, which now reports the
    /// `symbol_registry_size` series) grow the registry perfectly linearly at
    /// 810 entries/iteration to 8100, ≈ 13.8 MB at the ~1.7 KB/entry measured
    /// for [`GROWTH_WARN_THRESHOLDS`] — 3.5% of the 394 MB peak RSS, about a
    /// third of that soak's total RSS growth, and the gate still passes at
    /// 1.09x against a 1.5x threshold. Two things a future GC should know:
    ///
    /// - 84% of an entry is the pinned claripy AST, not the Rust maps. Dropping
    ///   only `name_to_info` / `py_hash_to_rust_id` (the cheap, purely-Rust
    ///   half) would recover almost nothing; `rust_id_to_py` is the one that
    ///   matters.
    /// - This implementation is O(removed × live): each removed id rescans
    ///   `py_hash_to_rust_id` and `name_to_info` in full. Fine at the 8k scale
    ///   above, quadratic at the 100k threshold — invert to a per-id index
    ///   before wiring a caller that collects at that size.
    pub fn retain(&self, active_ids: &std::collections::HashSet<u64>) {
        let mut id_to_py = self.rust_id_to_py.write();
        let mut hash_to_id = self.py_hash_to_rust_id.write();
        let mut name_to_info = self.name_to_info.write();
        let mut id_to_name = self.rust_id_to_name.write();

        // Collect IDs to remove
        let to_remove: Vec<u64> = id_to_py
            .keys()
            .filter(|id| !active_ids.contains(id))
            .copied()
            .collect();

        for id in to_remove {
            id_to_py.remove(&id);

            // Remove corresponding hash entries
            hash_to_id.retain(|_, &mut v| v != id);

            // Remove corresponding name entries
            name_to_info.retain(|_, info| info.rust_id != id);

            // Drop the id→Rust-name entry too (angr-4xaga.3): keep all four
            // maps symmetric so a wired GC caller cannot leak this map.
            id_to_name.remove(&id);
        }
    }
}

/// Global registry instance.
///
/// This is used when a per-manager registry is not available.
/// Thread-safe through internal synchronization.
static GLOBAL_REGISTRY: std::sync::OnceLock<SymbolicIdentityRegistry> = std::sync::OnceLock::new();

/// Get the global registry instance.
pub fn global_registry() -> &'static SymbolicIdentityRegistry {
    GLOBAL_REGISTRY.get_or_init(SymbolicIdentityRegistry::new)
}

/// Clear the global registry.
///
/// **Test-only** (angr-9ke6b.218 item 4). Nothing in production clears the
/// registry: symbol identity is process-global precisely so several
/// `RustExplorationManager`s in one process agree on it, and wiping it while
/// any earlier manager's states or `RustBV`s are still live would strand their
/// rust ids (`lookup_name_by_id` -> `None` -> a renamed BVS that no existing
/// constraint binds — the angr-izov2 failure mode). Bounding registry growth is
/// a GC problem, tracked on angr-9ke6b.222, not a clear-it-at-startup one.
/// Call `claripy_bridge::reset_for_new_exploration` rather than this directly;
/// see cross-cache invariant C3.
#[cfg(test)]
pub fn clear_global_registry() {
    if let Some(registry) = GLOBAL_REGISTRY.get() {
        registry.clear();
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
