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
//! - `name_to_info`: Symbol name -> (id, width) for name-based lookup
//!
//! On import: Check registry before creating new symbol
//! On export: Return original `Py<PyAny>` if in registry

use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Live-symbol counts at which [`SymbolicIdentityRegistry::register`] emits a
/// one-shot unbounded-growth warning.
///
/// The registry has no production GC caller (see [`SymbolicIdentityRegistry::retain`]
/// for why a sound one is not trivially available), so an exploration that
/// mints fresh symbols without bound grows all four maps without bound, each
/// entry pinning a claripy AST alive from the Rust side. That is invisible
/// today; warning once per order of magnitude makes it loud instead of
/// silently degrading, and gives a future GC a metric to gate on.
const GROWTH_WARN_THRESHOLDS: [usize; 3] = [100_000, 1_000_000, 10_000_000];

/// Information about a registered symbol.
#[derive(Clone, Debug)]
pub struct SymbolInfo {
    /// Rust-side symbol ID.
    pub rust_id: u64,
    /// Bit width of the symbol.
    pub width: u32,
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
    /// * `py_ast` - The original Python AST object
    pub fn register(&self, py_hash: i64, rust_id: u64, name: &str, width: u32, py_ast: Py<PyAny>) {
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
        let qualified_name = format!("{name}_w{width}");
        self.name_to_info.write().insert(
            qualified_name,
            SymbolInfo {
                rust_id,
                width,
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
                 it has no garbage collector, so every symbol minted by this exploration is \
                 pinned — along with its original claripy AST — until the next \
                 reset_for_new_exploration"
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

    /// Look up symbol info by name (deprecated - use lookup_by_name_and_width).
    ///
    /// Returns the SymbolInfo if a symbol with this name was registered.
    /// Note: Since D2 fix, names are stored as "{name}_w{width}", so this
    /// method is only useful for backwards compatibility.
    pub fn lookup_by_name(&self, name: &str) -> Option<SymbolInfo> {
        self.name_to_info.read().get(name).cloned()
    }

    /// Look up symbol info by name and width.
    ///
    /// Returns the SymbolInfo if a symbol with this name and width was registered.
    /// This is the preferred method after D2 fix which uses width-qualified names.
    pub fn lookup_by_name_and_width(&self, name: &str, width: u32) -> Option<SymbolInfo> {
        let qualified_name = format!("{name}_w{width}");
        self.name_to_info.read().get(&qualified_name).cloned()
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
    /// # Safety precondition — read before adding a caller
    ///
    /// `active_ids` MUST be a **superset** of every symbol id reachable from
    /// anything still live in the process: the registers, memory and
    /// constraints of every state in every stash, states in flight on worker
    /// threads, values held by pending Python callbacks, and the `RustBV`s
    /// pinned inside the thread-local bridge caches (`claripy_bridge::cache`).
    ///
    /// Dropping an id that is still reachable is **silently wrong, not loud**:
    /// on the next export, `rustbv_to_claripy` misses the registry and mints a
    /// fresh `claripy.BVS(name, width)`. Without `explicit_name`, claripy
    /// renames that to `name_<counter>_<width>`, so the re-exported leaf is a
    /// brand-new unconstrained variable and every constraint carried by the
    /// original stops binding (the angr-izov2 failure mode).
    ///
    /// # Why there is no production caller yet
    ///
    /// Computing that superset needs a full live-symbol traversal at a
    /// quiescent point in the exploration loop; no such traversal exists today
    /// (angr-9ke6b.40 → follow-up bead). Until one does, the only production
    /// mutation of the registry is the wholesale `reset_for_new_exploration`
    /// clear at exploration start, and unbounded growth within a single run is
    /// surfaced by `maybe_warn_growth` / the `symbol_registry_size` stat rather
    /// than collected. Do not wire an approximate active set into this method.
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
/// Should be called at the start of each new exploration.
pub fn clear_global_registry() {
    if let Some(registry) = GLOBAL_REGISTRY.get() {
        registry.clear();
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
