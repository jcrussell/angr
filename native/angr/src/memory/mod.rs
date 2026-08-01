//! Symbolic memory system for the VEX execution engine.
//!
//! This module provides a paged memory model with:
//! - O(1) forking via copy-on-write (using Grudge's RustPage)
//! - Mixed concrete/symbolic value storage
//! - Efficient symbolic address handling
//!
//! **Panic policy (angr-9ke6b.212):** every address that reaches this module is
//! guest-derived, so nothing here may panic on address shape — unresolvable
//! addresses surface as [`MemoryError`] variants the caller routes to the
//! Python memory model. The one surviving `expect` pair is in
//! [`SymbolicMemory::merge`] and is guarded by the page bitmap read one line
//! earlier, not by anything the guest controls.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`, which also reaches the
//! `address`/`concretize_glue`/`ite_builder`/`multi`/`page`/`store`/`load`/
//! `symbolic_objects` child modules, so a new panic on an untrusted address
//! anywhere under `memory/` needs a reviewed, reasoned `#[allow]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;

use im::OrdMap;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::Endness;

mod address;
mod concretize_glue;
mod ite_builder;
mod load;
mod multi;
mod page;
mod store;
mod symbolic_objects;
#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` in the parent overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
pub use address::Address;
pub use multi::{MultiAlternative, MultiPayload};
pub use page::{BITMAP_WORDS, MemoryPage, PAGE_MASK, PAGE_SIZE, Permission};

/// A deferred symbolic store. Instead of eagerly concretizing symbolic addresses
/// and building ITE chains at store time (275 Z3 calls for sym-write), we record
/// the store and only materialize it when a load touches the same region.
#[derive(Debug, Clone)]
pub struct PendingWrite {
    /// The symbolic address expression.
    pub addr: RustBV,
    /// The value to store.
    pub value: RustBV,
    /// Size in bytes.
    pub size: u32,
    /// Optional condition (for conditional stores).
    pub condition: Option<RustBV>,
    /// Hint: page range that this write could touch (min_page, max_page).
    /// If None, the write could be to any address.
    pub page_hint: Option<(u64, u64)>,
}

impl PendingWrite {
    /// Deep-translate the symbolic `addr`/`value`/`condition` BVs into
    /// `target_ctx` (angr-ahypj). Size and page-hint metadata are
    /// context-independent and copied verbatim.
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> PendingWrite {
        PendingWrite {
            addr: self.addr.translate_into(target_ctx),
            value: self.value.translate_into(target_ctx),
            size: self.size,
            condition: self
                .condition
                .as_ref()
                .map(|c| c.translate_into(target_ctx)),
            page_hint: self.page_hint,
        }
    }
}

/// Errors from memory operations.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants; intra-crate matches must include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum MemoryError {
    /// Unmapped memory access.
    #[error("unmapped memory at 0x{addr:x} (size {size})")]
    Unmapped { addr: u64, size: u64 },
    /// Unmapped page in a mapped region (can be fetched on-demand).
    #[error("unmapped page at 0x{page_addr:x} in mapped region")]
    UnmappedPageInRegion { page_addr: u64 },
    /// Permission violation.
    #[error("permission violation at 0x{addr:x}: required {required:?}, have {actual:?}")]
    Permission {
        addr: u64,
        required: Permission,
        actual: Permission,
    },
    /// Unresolvable symbolic address.
    #[error("symbolic address: {description}")]
    SymbolicAddress { description: String },
    /// Out of bounds access.
    #[error("out of bounds access at 0x{addr:x} (size {size})")]
    OutOfBounds { addr: u64, size: u64 },
    /// A zero-size access. Rejected up-front by [`end_page_inclusive`] rather
    /// than allowed to compute `addr + 0 - 1` (see that function for why).
    /// A zero-byte load could not produce a valid BV anyway (Z3 has no
    /// 0-width bitvector), and a zero-byte store has nothing to write, so
    /// both are caller bugs — surfacing them lets the Python side fall back
    /// or raise instead of hanging.
    #[error("zero-size memory access at 0x{addr:x}")]
    ZeroSize { addr: u64 },
    /// A concrete value was expected but the BV was symbolic. Defense-in-depth:
    /// the concrete store path is guarded by an `is_symbolic()` early-return, so
    /// this should be unreachable in practice — it converts a would-be panic
    /// into a recoverable error if that guard is ever bypassed.
    #[error("expected concrete value at 0x{addr:x}, found symbolic")]
    UnexpectedSymbolic { addr: u64 },
}

/// Inclusive number of the last page touched by `[addr, addr + size)`.
///
/// angr-9ke6b.99: every page-range check used to inline
/// `(addr + size - 1) >> 12`. That underflows when `size == 0` — and the
/// workspace `[profile.release]` does not set `overflow-checks`, so a release
/// build wraps to `u64::MAX` instead of panicking. `check_perms_range` then
/// iterates `start_page..=u64::MAX >> 12`, i.e. ~4.5e15 iterations: a hang,
/// not an error. Route every such computation through here so `size == 0` is
/// rejected as [`MemoryError::ZeroSize`] before any arithmetic happens.
///
/// `size == 0` is reachable from untrusted input: `_pending_memory_load`
/// (`exploration/pending_api.rs`) forwards a Python-supplied size straight to
/// `load_concrete`, and on the store side `size = value.width() / 8` is zero
/// for any sub-byte-width BV.
pub(super) fn end_page_inclusive(addr: u64, size: u64) -> Result<u64, MemoryError> {
    if size == 0 {
        return Err(MemoryError::ZeroSize { addr });
    }
    Ok((addr + size - 1) >> 12)
}

impl ConcretizationResult {
    /// Map a *failed* concretization (`TooLarge` / `Failed`) onto the
    /// `MemoryError::SymbolicAddress` it should surface, so callers can fall
    /// back to Python's memory model (angr-24pv4.3). Returns `None` for the
    /// success variants (`Single` / `Multiple` / `Strided`), which every
    /// caller handles before delegating here. Centralizes the "address range
    /// too large" message string that was previously copy-pasted at six load
    /// and store concretization-dispatch sites.
    pub(crate) fn to_symbolic_address_error(&self) -> Option<MemoryError> {
        match self {
            ConcretizationResult::TooLarge { min, max, .. } => Some(MemoryError::SymbolicAddress {
                description: format!(
                    "address range too large for concretization: 0x{min:x} - 0x{max:x}"
                ),
            }),
            ConcretizationResult::Failed(reason) => Some(MemoryError::SymbolicAddress {
                description: reason.clone(),
            }),
            _ => None,
        }
    }

    /// Total form of [`Self::to_symbolic_address_error`] for the concretization
    /// dispatch sites, whose catch-all arm has already peeled off every success
    /// variant.
    ///
    /// Those arms used to end in `.to_symbolic_address_error().expect(..)`,
    /// which turned a hypothetical new success variant into a panic. This maps
    /// it onto the same recoverable `SymbolicAddress` error the caller already
    /// routes to the Python memory model — the same defense-in-depth trade
    /// [`MemoryError::UnexpectedSymbolic`] documents — while naming the
    /// unexpected variant so the message is still diagnosable.
    pub(crate) fn as_symbolic_address_error(&self) -> MemoryError {
        self.to_symbolic_address_error()
            .unwrap_or_else(|| MemoryError::SymbolicAddress {
                description: format!("unexpected concretization result: {self:?}"),
            })
    }
}

/// Symbolic memory model.
///
/// This provides a paged memory model with:
/// - O(1) forking via CoW
/// - Mixed concrete/symbolic storage
/// - Endianness-aware loads and stores
/// - Dirty page tracking for efficient sync
/// - Lazy page regions for on-demand fetching
pub struct SymbolicMemory {
    /// Pages indexed by page number (addr >> 12).
    pages: OrdMap<u64, MemoryPage>,
    /// Symbolic objects (for values that span multiple bytes).
    symbolic_objects: FxHashMap<Address, RustBV>,
    /// Next symbolic object ID.
    next_sym_id: u64,
    /// Default permissions for new pages.
    default_permissions: Permission,
    /// Endianness for this memory.
    endness: Endness,
    /// Pages that have been modified since last clear.
    /// Stores page numbers (addr >> 12) for efficient tracking.
    dirty_pages: FxHashSet<u64>,
    /// Lazy regions: page ranges that CAN have pages fetched on-demand.
    /// Stores (start_page_num, end_page_num) pairs.
    /// When a load hits an unmapped page in a lazy region, the interpreter
    /// should fetch it from Python rather than failing.
    lazy_regions: Vec<(u64, u64)>,
    /// Reverse index for symbolic objects: maps each byte offset within a
    /// symbolic object to (base_addr, width_bits). Enables O(1) lookup when
    /// loading a byte that falls inside a wider symbolic object.
    symbolic_spans: FxHashMap<Address, (Address, u32)>,
    /// Multi-cell side table indexed by byte address. A byte is "Multi"
    /// (has an entry here) when a symbolic-address store emitted lazy
    /// alternatives at that cell instead of folding into an ITE chain.
    ///
    /// Parallel to `symbolic_objects` but distinct: a byte may be marked
    /// Multi *or* Symbolic but not both at the same time. Load-time
    /// collapse (Phase 1.2, angr-n082) prefers Multi when both happen to
    /// be set, and store-side helpers (Phase 1.3, angr-aija) clear any
    /// stale Symbolic entry before installing a Multi.
    ///
    /// See `memory/multi.rs` for the `MultiPayload` invariants.
    multi_objects: FxHashMap<Address, MultiPayload>,
    /// Deferred symbolic stores. Instead of eagerly concretizing symbolic
    /// addresses at store time, we append here and materialize on load.
    pending_writes: Vec<PendingWrite>,
    /// If true, fill unconstrained memory with zeros instead of symbolic values.
    /// Corresponds to angr's ZERO_FILL_UNCONSTRAINED_MEMORY option.
    zero_fill_unconstrained: bool,
    /// Addresses of symbolic values imported from Python.
    /// Used to filter get_state_symbolic_z3_asts: even if the binary modifies
    /// an imported value (turning Symbolic→Expression), the address should be
    /// excluded from export since Python already has the correct original value.
    imported_addrs: FxHashSet<Address>,
    /// If true, enforce per-page R/W permissions on load and store. Mirrors
    /// angr's STRICT_PAGE_ACCESS option. Default is false to keep existing
    /// callers (which often map all memory as RWX or rely on Python perms)
    /// working unchanged.
    enforce_permissions: bool,
    /// If true (and `enforce_permissions` is also true), reject instruction
    /// fetches from mapped pages without the X bit. Mirrors angr's ENABLE_NX
    /// option: Python's heavy VEX engine only fires the non-executable check
    /// when BOTH STRICT_PAGE_ACCESS and ENABLE_NX are in state.options
    /// (angr/engines/vex/heavy/heavy.py:115-124). Default false.
    enforce_nx: bool,
    /// Per-byte monotonic version counter for Multi cells. Bumped on every
    /// `set_multi_alternatives` and `clear_multi_at` so the Phase 4.1
    /// wider-load cache (`wider_load_cache`) can detect any installation
    /// change at a byte address without comparing payload contents.
    /// Versions persist across flush/reinstall so the fingerprint of a
    /// post-flush Multi byte differs from the cached pre-flush snapshot
    /// even when both happen to have the same alternative count.
    pub(super) multi_versions: FxHashMap<Address, u64>,
    /// Phase 4.1 (angr-mmdh.1): cached results from
    /// `assemble_load_with_multi`, keyed by `(addr, size)`. Each entry
    /// stores a per-byte fingerprint (Multi version+default_byte, or the
    /// concrete byte for non-Multi cells) and the assembled `RustBV`.
    /// Lookups rebuild the fingerprint and reuse the cached BV when it
    /// matches — skipping the per-byte concat + ITE-rebuild that
    /// dominated Phase 2 gate-on cost. RefCell because the cache lives
    /// on the load path (`&self`).
    wider_load_cache: RefCell<FxHashMap<(Address, u32), CachedWiderLoad>>,
}

/// Phase 4.1: one byte's role in a cached wider-load result. Loads that
/// touch any plain Symbolic byte (`page.is_symbolic()` true but not Multi)
/// are not cached — `symbolic_objects` and `symbolic_spans` have a
/// different mutation profile this cache does not track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ByteFingerprint {
    Multi { version: u64, default_byte: u8 },
    Concrete { byte: u8 },
}

/// Phase 4.1: cached wider-load result. `total_ite_depth` replays the
/// `record_mem_ite_depth` calls the existing per-byte path would have
/// made — required by memory `invariant-mem-ite-depth-counter`.
#[derive(Debug, Clone)]
pub(super) struct CachedWiderLoad {
    pub(super) byte_fingerprints: Vec<ByteFingerprint>,
    pub(super) total_ite_depth: u32,
    pub(super) bv: RustBV,
}

/// Soft cap on the wider-load cache to bound memory across long
/// explorations. Loads with a distinct `(addr, size)` key fill this — at
/// 1024 entries with ~10 bytes of fingerprint + one BV handle each, total
/// memory stays in the tens of KB per state.
pub(super) const WIDER_LOAD_CACHE_CAP: usize = 1024;

impl SymbolicMemory {
    /// Create a new empty memory.
    pub fn new(endness: Endness) -> Self {
        SymbolicMemory {
            pages: OrdMap::new(),
            symbolic_objects: FxHashMap::default(),
            next_sym_id: 0,
            default_permissions: Permission::RWX,
            endness,
            dirty_pages: FxHashSet::default(),
            lazy_regions: Vec::new(),
            symbolic_spans: FxHashMap::default(),
            multi_objects: FxHashMap::default(),
            pending_writes: Vec::new(),
            zero_fill_unconstrained: false,
            imported_addrs: FxHashSet::default(),
            enforce_permissions: false,
            enforce_nx: false,
            multi_versions: FxHashMap::default(),
            wider_load_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Bump the Multi-cell version counter at `addr`. Called by both
    /// `set_multi_alternatives` and `clear_multi_at` (and by
    /// `flush_multi_cells` for each byte it converts to Symbolic) so the
    /// Phase 4.1 wider-load cache fingerprint changes on any Multi
    /// installation/removal. Versions monotonically increase per addr; the
    /// entry persists even after the Multi cell is cleared so a later
    /// reinstall still produces a fresh version distinct from any cached
    /// snapshot.
    pub(super) fn bump_multi_version(&mut self, addr: Address) {
        let v = self.multi_versions.entry(addr).or_insert(0);
        *v = v.wrapping_add(1);
    }

    /// angr-jvjf: returns true iff every byte in `[addr, addr+size)` is
    /// marked symbolic in the owning page's bitmap. The bitmap is the
    /// authoritative source of truth — a concrete write inside the range
    /// of a wider symbolic object correctly clears the byte's bit but
    /// leaves `symbolic_objects` / `symbolic_spans` claiming the byte is
    /// still symbolic. Loads must consult this helper before trusting
    /// the wider-sym fast paths or they will return a stale extract.
    pub(super) fn bytes_all_marked_symbolic(&self, addr: Address, size: u32) -> bool {
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr.page_num();
            let offset = byte_addr.page_offset();
            match self.pages.get(&page_num) {
                Some(page) => {
                    if !page.is_symbolic(offset) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Phase 4.1: compute the per-byte fingerprint for a wider load.
    /// Returns `None` when any byte is plain Symbolic (the wider-load
    /// cache only covers Multi + Concrete bytes) or when a byte's page
    /// is unmapped (the caller's existing error path handles that).
    pub(super) fn compute_wider_load_fingerprint(
        &self,
        addr: Address,
        size: u32,
    ) -> Option<Vec<ByteFingerprint>> {
        let mut fp = Vec::with_capacity(size as usize);
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr.page_num();
            let offset = byte_addr.page_offset();
            let page = self.pages.get(&page_num)?;
            if self.multi_objects.contains_key(&byte_addr) {
                let version = self.multi_versions.get(&byte_addr).copied().unwrap_or(0);
                let default_byte = page.load_concrete(offset, 1).first().copied().unwrap_or(0);
                fp.push(ByteFingerprint::Multi {
                    version,
                    default_byte,
                });
            } else if page.is_symbolic(offset) {
                return None;
            } else {
                let byte = page.load_concrete(offset, 1).first().copied().unwrap_or(0);
                fp.push(ByteFingerprint::Concrete { byte });
            }
        }
        Some(fp)
    }

    /// Phase 4.1: write a wider-load cache entry, evicting an arbitrary
    /// existing entry if the cache is at capacity. Eviction picks the
    /// `HashMap::keys().next()` (arbitrary, no LRU bookkeeping); the cap
    /// is high enough that this only matters for explorations that touch
    /// thousands of distinct load shapes.
    pub(super) fn insert_wider_load_cache(&self, key: (Address, u32), entry: CachedWiderLoad) {
        let mut cache = self.wider_load_cache.borrow_mut();
        if cache.len() >= WIDER_LOAD_CACHE_CAP
            && !cache.contains_key(&key)
            && let Some(victim) = cache.keys().next().copied()
        {
            cache.remove(&victim);
        }
        cache.insert(key, entry);
    }

    /// Phase 4.1: test-only helper — current cache size.
    #[cfg(test)]
    pub(crate) fn wider_load_cache_len(&self) -> usize {
        self.wider_load_cache.borrow().len()
    }

    /// Enable or disable strict per-page permission enforcement on load/store.
    ///
    /// When enabled, `load*` requires R on every touched page and `store*`
    /// requires W. A violation returns `MemoryError::Permission`. Default off.
    pub fn set_enforce_permissions(&mut self, enabled: bool) {
        self.enforce_permissions = enabled;
    }

    /// Whether strict permission enforcement is enabled.
    pub fn enforce_permissions(&self) -> bool {
        self.enforce_permissions
    }

    /// Enable or disable non-executable page enforcement on instruction
    /// fetch. Mirrors angr's ENABLE_NX option. The X check only fires when
    /// BOTH `enforce_permissions` (STRICT_PAGE_ACCESS) and `enforce_nx`
    /// (ENABLE_NX) are on, matching Python's heavy VEX engine semantics.
    pub fn set_enforce_nx(&mut self, enabled: bool) {
        self.enforce_nx = enabled;
    }

    /// Whether non-executable page enforcement is enabled.
    pub fn enforce_nx(&self) -> bool {
        self.enforce_nx
    }

    /// Check that the page containing `addr` carries execute permission.
    /// No-op unless BOTH `enforce_permissions` and `enforce_nx` are true
    /// (matches Python: STRICT_PAGE_ACCESS gates the permissions lookup,
    /// ENABLE_NX gates raising on non-executable). If the page is unmapped
    /// we return Ok so the caller can fall back to its existing lift paths
    /// (native libpyvex region / Python lift_block callback) — only mapped
    /// pages without the X bit produce a `Permission` error here.
    pub fn check_executable(&self, addr: impl Into<Address>) -> Result<(), MemoryError> {
        if !self.enforce_permissions || !self.enforce_nx {
            return Ok(());
        }
        let addr = addr.into();
        let page_num = addr.page_num();
        if let Some(page) = self.pages.get(&page_num) {
            let actual = page.permissions();
            if !actual.allows(Permission::X) {
                return Err(MemoryError::Permission {
                    addr: addr.raw(),
                    required: Permission::X,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Check that every mapped page in `start_page..=end_page` allows the
    /// required access. No-op if `enforce_permissions` is false. Pages that
    /// are unmapped are skipped here and surfaced as `Unmapped` /
    /// `UnmappedPageInRegion` by callers' existing checks.
    fn check_perms_range(
        &self,
        start_page: u64,
        end_page: u64,
        required: Permission,
    ) -> Result<(), MemoryError> {
        if !self.enforce_permissions {
            return Ok(());
        }
        for page_num in start_page..=end_page {
            if let Some(page) = self.pages.get(&page_num) {
                let actual = page.permissions();
                if !actual.allows(required) {
                    return Err(MemoryError::Permission {
                        addr: page_num << 12,
                        required,
                        actual,
                    });
                }
            }
        }
        Ok(())
    }

    /// Set whether to fill unconstrained memory with zeros.
    pub fn set_zero_fill_unconstrained(&mut self, enabled: bool) {
        self.zero_fill_unconstrained = enabled;
    }

    /// Get whether zero fill is enabled.
    pub fn zero_fill_unconstrained(&self) -> bool {
        self.zero_fill_unconstrained
    }

    /// Get the endianness.
    pub fn endness(&self) -> Endness {
        self.endness
    }

    /// Map a memory region.
    pub fn map(&mut self, addr: impl Into<Address>, size: u64, permissions: Permission) {
        // A zero-length map is a true no-op regardless of alignment, matching
        // Python's paged_memory_mixin (`while size_done < length` never runs).
        // Without this guard a non-page-aligned addr spuriously maps one page.
        if size == 0 {
            return;
        }
        let addr = addr.into();
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            let base = page_num << 12;
            if !self.pages.contains_key(&page_num) {
                self.pages
                    .insert(page_num, MemoryPage::new(base, permissions));
            }
        }
    }

    /// Map a region and initialize with data.
    pub fn map_data(&mut self, addr: impl Into<Address>, data: &[u8], permissions: Permission) {
        let addr = addr.into();
        let mut remaining = data;
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr.page_num();
            let page_offset = current_addr.page_offset() as usize;
            let bytes_in_page = (PAGE_SIZE as usize - page_offset).min(remaining.len());

            // Get or create page, modify in place (COW handled by Arc::make_mut in store_concrete)
            let page = self
                .pages
                .entry(page_num)
                .or_insert_with(|| MemoryPage::new(page_num << 12, permissions));
            page.store_concrete(page_offset as u16, &remaining[..bytes_in_page]);

            remaining = &remaining[bytes_in_page..];
            current_addr = current_addr + bytes_in_page as u64;
        }
    }

    /// Unmap a memory region.
    pub fn unmap(&mut self, addr: impl Into<Address>, size: u64) {
        // Mirror map(): a zero-length unmap is a true no-op regardless of
        // alignment (a non-aligned addr would otherwise drop one page).
        if size == 0 {
            return;
        }
        let addr = addr.into();
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            self.pages.remove(&page_num);
        }
    }

    /// Check if an address is mapped.
    pub fn is_mapped(&self, addr: impl Into<Address>) -> bool {
        let page_num = addr.into().page_num();
        self.pages.contains_key(&page_num)
    }

    /// Read up to `max_size` concrete bytes starting at `addr` for native
    /// lifting. Returns the concrete prefix found before the first symbolic
    /// or unmapped byte. `None` is returned only if the very first byte is
    /// unmapped or symbolic. The lifter accepts a partial buffer and stops
    /// at the byte boundary, so a short read is still useful.
    pub fn read_concrete_bytes_for_lift(
        &self,
        addr: impl Into<Address>,
        max_size: usize,
    ) -> Option<Vec<u8>> {
        if max_size == 0 {
            return Some(Vec::new());
        }
        let mut result = Vec::with_capacity(max_size);
        let mut current = addr.into();
        while result.len() < max_size {
            let page_num = current.page_num();
            let page = match self.pages.get(&page_num) {
                Some(p) => p,
                None => break,
            };
            let offset_in_page = current.page_offset();
            let remaining = max_size - result.len();
            let to_read = remaining.min((PAGE_SIZE - (current.raw() & PAGE_MASK)) as usize);
            // Stop at the first symbolic byte; native lift can't use it.
            let mut concrete_run = 0usize;
            for i in 0..to_read {
                if page.is_symbolic(offset_in_page + i as u16) {
                    break;
                }
                concrete_run += 1;
            }
            if concrete_run == 0 {
                break;
            }
            let bytes = page.load_concrete(offset_in_page, concrete_run as u16);
            result.extend(bytes);
            current = Address(current.raw().saturating_add(concrete_run as u64));
            if concrete_run < to_read {
                break; // hit a symbolic byte
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // ==================== UNIFIED SYMBOLIC MEMORY OPERATIONS ====================
    // These methods handle all symbolic memory operations entirely in Rust,
    // eliminating the need for Python callbacks that were previously broken.
    // See memory/concretize_glue.rs for prepare_addresses_for_ite and
    // prepare_strided_region. See memory/load.rs for load_*/load_symbolic_unified
    // and memory/store.rs for store_*/store_symbolic_unified.

    /// Fork the memory (O(1) via CoW).
    pub fn fork(&self) -> Self {
        SymbolicMemory {
            pages: self.pages.clone(), // OrdMap clones in O(1)
            symbolic_objects: self.symbolic_objects.clone(),
            next_sym_id: self.next_sym_id,
            default_permissions: self.default_permissions,
            endness: self.endness,
            dirty_pages: FxHashSet::default(), // Fresh dirty tracking for fork
            lazy_regions: self.lazy_regions.clone(), // Share lazy regions
            symbolic_spans: self.symbolic_spans.clone(),
            multi_objects: self.multi_objects.clone(),
            pending_writes: self.pending_writes.clone(),
            zero_fill_unconstrained: self.zero_fill_unconstrained,
            imported_addrs: self.imported_addrs.clone(),
            enforce_permissions: self.enforce_permissions,
            enforce_nx: self.enforce_nx,
            multi_versions: self.multi_versions.clone(),
            // Start the fork with a cold wider-load cache rather than deep-cloning
            // up to WIDER_LOAD_CACHE_CAP entries. The cache is a rebuildable,
            // fingerprint-validated read-side memo with no correctness role: the
            // snapshot restore path (from_snapshot) already starts it empty and
            // re-validates loads against page fingerprints, so a cold child is
            // faithful by construction. Avoids the per-fork clone cost; the child
            // repopulates lazily on its first wider load.
            wider_load_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Cross-context twin of [`Self::fork`] (angr-ahypj): produce a copy of
    /// this memory whose every symbolic value lives in `target_ctx`.
    ///
    /// Pages, bitmaps, spans, lazy regions and concrete data are all
    /// context-independent and cloned verbatim (the `OrdMap`/`Arc` shares are
    /// O(1)). Only the three BV-bearing maps need `Z3_translate`:
    /// `symbolic_objects` (per-byte values), `multi_objects` (lazy-store ITE
    /// alternatives), and `pending_writes` (deferred symbolic stores). The
    /// wider-load cache starts cold for the same reason `fork` clears it — it
    /// is a rebuildable, fingerprint-validated memo, and its cached BVs belong
    /// to the source context.
    ///
    /// Never mutates `self`: `translate_into` reads the immutable source BVs
    /// and emits fresh ones, so `Arc`-shared sibling pages are safe.
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> Self {
        SymbolicMemory {
            pages: self.pages.clone(),
            symbolic_objects: self
                .symbolic_objects
                .iter()
                .map(|(&addr, bv)| (addr, bv.translate_into(target_ctx)))
                .collect(),
            next_sym_id: self.next_sym_id,
            default_permissions: self.default_permissions,
            endness: self.endness,
            dirty_pages: FxHashSet::default(),
            lazy_regions: self.lazy_regions.clone(),
            symbolic_spans: self.symbolic_spans.clone(),
            multi_objects: self
                .multi_objects
                .iter()
                .map(|(&addr, payload)| (addr, payload.translate_into(target_ctx)))
                .collect(),
            pending_writes: self
                .pending_writes
                .iter()
                .map(|w| w.translate_into(target_ctx))
                .collect(),
            zero_fill_unconstrained: self.zero_fill_unconstrained,
            imported_addrs: self.imported_addrs.clone(),
            enforce_permissions: self.enforce_permissions,
            enforce_nx: self.enforce_nx,
            multi_versions: self.multi_versions.clone(),
            wider_load_cache: RefCell::new(FxHashMap::default()),
        }
    }

    /// Get pending writes count.
    pub fn pending_writes_count(&self) -> usize {
        self.pending_writes.len()
    }

    /// Get a reference to the pending writes.
    pub fn pending_writes(&self) -> &[PendingWrite] {
        &self.pending_writes
    }

    /// Add a deferred symbolic store.
    pub fn add_pending_write(&mut self, write: PendingWrite) {
        self.pending_writes.push(write);
    }

    /// Flush all pending writes by materializing ITE chains into memory.
    /// This must be called before exporting state to Python to ensure
    /// memory pages contain all written values.
    ///
    /// Phase 2 (angr-qh5u): also flushes Multi cells installed by the
    /// lazy STORE path (`install_multi_for_candidates_safe`). Each Multi
    /// byte collapses to a 1-byte symbolic_object so the existing
    /// `_sync_rust_symbolic_objects_to_state` exporter picks it up.
    /// Without this step Multi bytes would be invisible to Python on
    /// state export — their pages would be exported as if the bytes
    /// were concrete.
    pub fn flush_pending_writes(
        &mut self,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<(), MemoryError> {
        // Flush Multi cells regardless of pending_writes status — Phase 2
        // makes Multi cells the default for symbolic-address stores, so
        // export correctness depends on flushing them even when the
        // pending_writes queue (a separate, scaffolded path) is empty.
        self.flush_multi_cells(ctx);

        if self.pending_writes.is_empty() {
            return Ok(());
        }

        let writes = std::mem::take(&mut self.pending_writes);
        for pw in writes {
            // Try to concretize the address
            if let Some(concrete_addr) = pw.addr.as_u64() {
                self.store_concrete_lazy(concrete_addr, pw.value)?;
                continue;
            }

            match concretizer.concretize_write(&pw.addr, ctx) {
                ConcretizationResult::Single(addr) => {
                    self.store_concrete_lazy(addr, pw.value)?;
                }
                ConcretizationResult::Multiple(addrs) => {
                    // Build ITE chains for each candidate address
                    for &candidate in &addrs {
                        self.materialize_pending_ite(candidate, &pw, ctx)?;
                    }
                }
                ConcretizationResult::Strided {
                    base,
                    stride,
                    count,
                } => {
                    for i in 0..count {
                        self.materialize_pending_ite(base + i * stride, &pw, ctx)?;
                    }
                }
                ConcretizationResult::TooLarge { .. } | ConcretizationResult::Failed(_) => {
                    // Cannot materialize — skip (data was already applied via load-time ITE)
                }
            }
        }
        Ok(())
    }

    /// Materialize one candidate of a pending symbolic-address write
    /// (angr-24pv4.3) shared by the `Multiple` and `Strided` arms of
    /// `flush_pending_writes`. Builds `mem[candidate] = If(pw.addr ==
    /// candidate [&& pw.condition], pw.value, current)` where `current` is
    /// the existing value (or zero if the cell is unmapped).
    fn materialize_pending_ite(
        &mut self,
        candidate: u64,
        pw: &PendingWrite,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let addr_const = RustBV::concrete(candidate as u128, pw.addr.width());
        let cond = pw.addr.eq(&addr_const, ctx);
        let effective_cond = if let Some(ref c) = pw.condition {
            cond.and(c, ctx)
        } else {
            cond
        };
        let current = match self.load_concrete_lazy_inner(Address(candidate), pw.size, ctx) {
            Ok(v) => v,
            Err(_) => RustBV::concrete(0, pw.size * 8),
        };
        let ite_val = effective_cond.ite(&pw.value, &current, ctx);
        self.store_concrete_lazy(candidate, ite_val)
    }

    /// Get page count.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get list of dirty page numbers (pages modified since last clear).
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.dirty_pages.iter().copied().collect()
    }

    /// Get list of dirty page addresses (page-aligned addresses).
    pub fn get_dirty_page_addrs(&self) -> Vec<u64> {
        self.dirty_pages.iter().map(|&pn| pn << 12).collect()
    }

    /// Load an entire page as concrete bytes (4096 bytes).
    /// Returns Err if the page is not mapped.
    pub fn load_page_concrete(
        &self,
        page_addr: impl Into<Address>,
    ) -> Result<Vec<u8>, MemoryError> {
        let page_addr = page_addr.into();
        let page_num = page_addr.page_num();
        if let Some(page) = self.pages.get(&page_num) {
            Ok(page.load_concrete(0, PAGE_SIZE as u16))
        } else {
            Err(MemoryError::Unmapped {
                addr: page_addr.raw(),
                size: PAGE_SIZE,
            })
        }
    }

    /// Clear dirty page tracking (called after sync to Python).
    pub fn clear_dirty_pages(&mut self) {
        self.dirty_pages.clear();
    }

    /// Get the pages OrdMap for iteration.
    pub fn pages(&self) -> &OrdMap<u64, MemoryPage> {
        &self.pages
    }

    /// Get the permissions for the page containing `page_num` (`addr >> 12`).
    /// Returns `None` if the page is unmapped.
    pub fn page_permissions(&self, page_num: u64) -> Option<Permission> {
        self.pages.get(&page_num).map(page::MemoryPage::permissions)
    }

    /// Set the permissions on the page at `page_num` (`addr >> 12`).
    /// Returns `false` if the page is unmapped (no change made).
    pub fn set_page_permissions(&mut self, page_num: u64, perm: Permission) -> bool {
        match self.pages.get_mut(&page_num) {
            Some(page) => {
                page.set_permissions(perm);
                true
            }
            None => false,
        }
    }

    // =========================================================================
    // Symbolic Memory Preservation — see memory/symbolic_objects.rs
    // =========================================================================

    /// Add a lazy region where pages can be fetched on-demand.
    ///
    /// When a load hits an unmapped page within this region, the memory
    /// will return `UnmappedPageInRegion` instead of `Unmapped`, signaling
    /// to the interpreter that it should fetch the page from Python.
    ///
    /// # Arguments
    /// * `start_addr` - Start address of the region (will be page-aligned down)
    /// * `size` - Size of the region in bytes
    pub fn add_lazy_region(&mut self, start_addr: impl Into<Address>, size: u64) {
        let start_addr = start_addr.into();
        let start_page = start_addr.page_num();
        let end_page = (start_addr.raw() + size + PAGE_SIZE - 1) >> 12;
        self.lazy_regions.push((start_page, end_page));
    }

    /// Check if a page number is within a lazy region.
    pub fn is_in_lazy_region(&self, page_num: u64) -> bool {
        for &(start, end) in &self.lazy_regions {
            if page_num >= start && page_num < end {
                return true;
            }
        }
        false
    }

    /// Check if an address is within a lazy region.
    pub fn is_addr_in_lazy_region(&self, addr: impl Into<Address>) -> bool {
        self.is_in_lazy_region(addr.into().page_num())
    }

    /// Get the number of lazy regions.
    pub fn lazy_region_count(&self) -> usize {
        self.lazy_regions.len()
    }

    /// Get all unmapped page addresses in the lazy region containing the trigger page.
    ///
    /// This is used for eager region prefetch - when a page is missing, we can
    /// batch-fetch all unmapped pages in that region at once.
    ///
    /// # Arguments
    /// * `trigger_page_addr` - Page address that triggered the fetch
    /// * `max_pages` - Maximum number of pages to return (for batching)
    ///
    /// # Returns
    /// List of page addresses to fetch, or None if the page is not in a lazy region.
    pub fn get_region_prefetch_list(
        &self,
        trigger_page_addr: impl Into<Address>,
        max_pages: usize,
    ) -> Option<Vec<u64>> {
        let trigger_page_num = trigger_page_addr.into().page_num();

        // Find the lazy region containing this page
        let region = self
            .lazy_regions
            .iter()
            .find(|&&(start, end)| trigger_page_num >= start && trigger_page_num < end)?;

        let (region_start, region_end) = *region;

        // Collect all unmapped pages in this region
        let mut pages_to_fetch = Vec::new();

        for page_num in region_start..region_end {
            if !self.pages.contains_key(&page_num) {
                pages_to_fetch.push(page_num << 12); // Convert to page address
                if pages_to_fetch.len() >= max_pages {
                    break;
                }
            }
        }

        if pages_to_fetch.is_empty() {
            None
        } else {
            Some(pages_to_fetch)
        }
    }

    /// Get unmapped pages around a trigger page (for locality-based prefetch).
    ///
    /// # Arguments
    /// * `trigger_page_addr` - Page address that triggered the fetch
    /// * `count_before` - Number of pages to check before the trigger
    /// * `count_after` - Number of pages to check after the trigger
    ///
    /// # Returns
    /// List of unmapped page addresses in the region around the trigger.
    pub fn get_nearby_prefetch_list(
        &self,
        trigger_page_addr: impl Into<Address>,
        count_before: u64,
        count_after: u64,
    ) -> Vec<u64> {
        let trigger_page_addr = trigger_page_addr.into();
        let trigger_page_num = trigger_page_addr.page_num();
        let mut pages_to_fetch = Vec::new();

        // Check pages before the trigger
        for i in 1..=count_before {
            if let Some(page_num) = trigger_page_num.checked_sub(i)
                && self.is_in_lazy_region(page_num)
                && !self.pages.contains_key(&page_num)
            {
                pages_to_fetch.push(page_num << 12);
            }
        }

        // Add the trigger page itself if not mapped
        if !self.pages.contains_key(&trigger_page_num) {
            pages_to_fetch.push(trigger_page_addr.raw());
        }

        // Check pages after the trigger
        for i in 1..=count_after {
            let page_num = trigger_page_num + i;
            if self.is_in_lazy_region(page_num) && !self.pages.contains_key(&page_num) {
                pages_to_fetch.push(page_num << 12);
            }
        }

        pages_to_fetch
    }

    /// Map a page with data directly (used for on-demand page fetching).
    ///
    /// This is a convenience method for the interpreter to add fetched pages.
    pub fn map_page(
        &mut self,
        page_addr: impl Into<Address>,
        data: Vec<u8>,
        permissions: Permission,
    ) {
        let page_addr = page_addr.into();
        let page_num = page_addr.page_num();
        let page = MemoryPage::from_data(page_addr.raw(), &data, permissions);
        self.pages.insert(page_num, page);
    }

    /// Auto-map a zero page for an unmapped address in a lazy region.
    ///
    /// # DEPRECATION WARNING
    ///
    /// This function is deprecated for use in interpreter callbacks. Creating
    /// speculative zero pages causes state divergence when Python has actual
    /// data (from backers like file contents or initialized sections). Use
    /// this function only for internal Rust memory operations where Python
    /// state is not involved.
    ///
    /// For interpreter callbacks that need memory, prefer falling back to
    /// the Python callback which can provide correct backer data.
    ///
    /// This creates a speculative zero page that can be validated later
    /// against Python state. Returns true if a page was created.
    pub fn auto_map_zero_page(&mut self, addr: impl Into<Address>) -> bool {
        let page_num = addr.into().page_num();

        // Only auto-map if not already mapped and in a lazy region
        if self.pages.contains_key(&page_num) {
            return false;
        }

        if !self.is_in_lazy_region(page_num) {
            return false;
        }

        // Create a zero page with RWX permissions
        let page_addr = page_num << 12;
        let page = MemoryPage::new(page_addr, Permission::RWX);
        self.pages.insert(page_num, page);

        // Mark as dirty so it gets synced if modified
        self.dirty_pages.insert(page_num);

        true
    }

    /// Merge another memory into this one using a merge condition.
    ///
    /// For each byte that differs between `self` and `other`, the merged
    /// value is `ITE(merge_cond_other, other_byte, self_byte)`.
    ///
    /// Returns true if any memory was actually merged (values differed).
    #[allow(
        clippy::expect_used,
        reason = "`s_multi`/`o_multi` are `page.has_multi() && page.is_multi(i)` bitmap reads taken from the same byte index a few lines above, and that bit is set only alongside the `multi_objects` entry for the byte's address — so both lookups are proved by the guard that gates this branch, and neither depends on guest input"
    )]
    pub fn merge(
        &mut self,
        other: &SymbolicMemory,
        merge_cond_other: &crate::symbolic::RustBV,
        ctx: &crate::symbolic::SymContext,
    ) -> bool {
        use crate::symbolic::RustBV;

        let mut merged = false;

        // Collect all page numbers from both memories
        let self_pages: std::collections::HashSet<u64> = self.pages.keys().copied().collect();
        let other_pages: std::collections::HashSet<u64> = other.pages.keys().copied().collect();
        let all_pages: std::collections::HashSet<u64> =
            self_pages.union(&other_pages).copied().collect();

        // Collect merge operations first to avoid borrow conflicts
        let mut merge_ops: Vec<(u64, Address, RustBV)> = Vec::new(); // (page_num, addr, ite_val)
        let mut pages_to_add: Vec<(u64, MemoryPage)> = Vec::new();
        // Multi-cell unions (M3-2b, angr-op0dn.11.2.2): a byte that is Multi
        // on *both* arms merges to a single lazy Multi cell whose alternatives
        // union the arms under merge-condition guards, staying lazy rather
        // than collapsing to a `symbolic_objects` ITE. Collected here and
        // installed after the borrow of `self.pages` is released.
        let mut multi_ops: Vec<(Address, crate::memory::multi::MultiPayload)> = Vec::new();

        for &page_num in &all_pages {
            let self_page = self.pages.get(&page_num);
            let other_page = other.pages.get(&page_num);

            match (self_page, other_page) {
                (Some(sp), Some(op)) => {
                    // CoW fast path (angr-op0dn.11.2.1, productionizes S5a): a
                    // structurally-shared page with no symbolic/Multi overlay is
                    // provably byte-identical, so it contributes nothing to the
                    // merge. Skip it without materializing PAGE_SIZE bytes,
                    // making merge cost proportional to divergent pages rather
                    // than all shared pages. This is a strict subset of the
                    // value-equality early-out below, so it is semantics-neutral.
                    if sp.is_shared_identical(op) {
                        continue;
                    }
                    #[cfg(test)]
                    tests::merge_instrument::note_page_walked();
                    // Both have this page — compare concrete data. The Multi
                    // guards mirror the plain-symbolic ones: a page carrying
                    // Multi cells can never take the concrete-equality
                    // early-out because Multi divergence lives in
                    // `multi_objects`, invisible to a `data[]` compare
                    // (angr-op0dn.11.2.2).
                    let s_data = sp.load_concrete(0, PAGE_SIZE as u16);
                    let o_data = op.load_concrete(0, PAGE_SIZE as u16);

                    if s_data == o_data
                        && !sp.has_symbolic()
                        && !op.has_symbolic()
                        && !sp.has_multi()
                        && !op.has_multi()
                    {
                        continue;
                    }

                    let base_addr = Address(page_num << 12);
                    for i in 0..PAGE_SIZE as usize {
                        let s_byte = s_data[i];
                        let o_byte = o_data[i];

                        let s_sym = sp.has_symbolic() && sp.is_symbolic(i as u16);
                        let o_sym = op.has_symbolic() && op.is_symbolic(i as u16);
                        let s_multi = sp.has_multi() && sp.is_multi(i as u16);
                        let o_multi = op.has_multi() && op.is_multi(i as u16);

                        // A byte contributes nothing only when it is plain
                        // concrete on both arms AND the bytes are equal.
                        if !s_sym && !o_sym && !s_multi && !o_multi && s_byte == o_byte {
                            continue;
                        }

                        let addr = base_addr + i as u64;

                        // Both-Multi lazy union (design-doc merge rule): keep
                        // the result a Multi cell, guarding each arm's
                        // alternatives so exactly one arm's chain is live per
                        // model. The page's concrete byte is a don't-care
                        // default (each arm's `exactly-one-cond-true`
                        // invariant means the else-leaf is never selected).
                        if s_multi && o_multi {
                            let not_cond = merge_cond_other.not(ctx);
                            let sp_pl = self
                                .multi_objects
                                .get(&addr)
                                .expect("s_multi implies a payload");
                            let op_pl = other
                                .multi_objects
                                .get(&addr)
                                .expect("o_multi implies a payload");
                            let mut alts = Vec::with_capacity(sp_pl.len() + op_pl.len());
                            for alt in sp_pl.alternatives() {
                                alts.push(crate::memory::multi::MultiAlternative::new(
                                    not_cond.and(&alt.cond, ctx),
                                    alt.value.clone(),
                                ));
                            }
                            for alt in op_pl.alternatives() {
                                alts.push(crate::memory::multi::MultiAlternative::new(
                                    merge_cond_other.and(&alt.cond, ctx),
                                    alt.value.clone(),
                                ));
                            }
                            multi_ops.push((
                                addr,
                                crate::memory::multi::MultiPayload::from_alternatives(alts),
                            ));
                            continue;
                        }

                        // Otherwise collapse any Multi side to a BV and ITE it
                        // against the other side (concrete / plain-symbolic).
                        let self_val = Self::merge_byte_value(
                            &self.symbolic_objects,
                            &self.multi_objects,
                            addr,
                            s_byte,
                            s_sym,
                            s_multi,
                            ctx,
                        );
                        let other_val = Self::merge_byte_value(
                            &other.symbolic_objects,
                            &other.multi_objects,
                            addr,
                            o_byte,
                            o_sym,
                            o_multi,
                            ctx,
                        );

                        let ite_val = merge_cond_other.ite(&other_val, &self_val, ctx);
                        merge_ops.push((page_num, addr, ite_val));
                    }
                }
                (None, Some(op)) => {
                    pages_to_add.push((page_num, op.clone()));
                }
                (Some(_), None) | (None, None) => {}
            }
        }

        // Apply collected merge operations
        for (page_num, addr, ite_val) in merge_ops {
            self.next_sym_id += 1;
            self.symbolic_objects.insert(addr, ite_val);
            self.symbolic_spans.insert(addr, (addr, 8));
            let offset_in_page = addr.page_offset();
            if let Some(page) = self.pages.get_mut(&page_num) {
                page.mark_symbolic(offset_in_page, 1);
            }
            merged = true;
        }

        for (page_num, page) in pages_to_add {
            self.pages.insert(page_num, page);
            merged = true;
        }

        // Install both-Multi unions. `set_multi_alternatives` clears any
        // conflicting plain-Symbolic state at the byte and marks the page
        // Multi, keeping the bitmaps and sidecars in sync. Applied before the
        // symbolic-object merge below so a stray `other` entry can't shadow a
        // freshly-installed Multi cell.
        for (addr, payload) in multi_ops {
            self.set_multi_alternatives(addr, payload);
            merged = true;
        }

        // Merge symbolic objects from other that aren't page-based
        for (&addr, other_obj) in &other.symbolic_objects {
            if let std::collections::hash_map::Entry::Vacant(e) = self.symbolic_objects.entry(addr)
            {
                e.insert(other_obj.clone());
                self.symbolic_spans.insert(addr, (addr, other_obj.width()));
                merged = true;
            }
        }

        // Merge pending writes with merge-condition guards (M3-2b,
        // angr-op0dn.11.2.2). A deferred symbolic store is arm-specific, so a
        // blind `extend` would let `other`'s writes fire on `self`'s paths (and
        // leaves `self`'s own writes firing on `other`'s paths). Guard each
        // arm's writes so a write only materializes on the path that issued it:
        // `self`'s writes under `!merge_cond_other`, `other`'s under
        // `merge_cond_other`. Guarding composes with any pre-existing
        // conditional-store condition via `And`.
        if !self.pending_writes.is_empty() {
            let not_cond = merge_cond_other.not(ctx);
            let guarded: Vec<PendingWrite> = self
                .pending_writes
                .iter()
                .map(|pw| Self::guard_pending_write(pw, &not_cond, ctx))
                .collect();
            self.pending_writes = guarded;
            merged = true;
        }
        if !other.pending_writes.is_empty() {
            self.pending_writes.extend(
                other
                    .pending_writes
                    .iter()
                    .map(|pw| Self::guard_pending_write(pw, merge_cond_other, ctx)),
            );
            merged = true;
        }

        merged
    }

    /// Extract a single byte's merge value: collapse a Multi cell to its ITE
    /// BV, read a plain-Symbolic byte from `symbolic_objects`, or fall back to
    /// the concrete page byte. Shared by the two arms of the byte-merge loop so
    /// Multi and plain-symbolic bytes both feed the same `ITE(cond, other,
    /// self)` (angr-op0dn.11.2.2). Takes the two side tables by reference so it
    /// can serve either `self` or `other` without a borrow conflict.
    fn merge_byte_value(
        symbolic_objects: &FxHashMap<Address, RustBV>,
        multi_objects: &FxHashMap<Address, crate::memory::multi::MultiPayload>,
        addr: Address,
        concrete_byte: u8,
        is_sym: bool,
        is_multi: bool,
        ctx: &crate::symbolic::SymContext,
    ) -> RustBV {
        if is_multi {
            multi_objects
                .get(&addr)
                .map(|p| p.collapse(concrete_byte, ctx))
                .unwrap_or_else(|| RustBV::concrete(concrete_byte as u128, 8))
        } else if is_sym {
            symbolic_objects
                .get(&addr)
                .cloned()
                .unwrap_or_else(|| RustBV::concrete(concrete_byte as u128, 8))
        } else {
            RustBV::concrete(concrete_byte as u128, 8)
        }
    }

    /// Guard a deferred symbolic store with a merge condition, composing with
    /// any pre-existing conditional-store condition via `And`
    /// (angr-op0dn.11.2.2). The returned write only materializes when `guard`
    /// holds, so an arm-specific pending write stays scoped to its own path
    /// after a merge.
    fn guard_pending_write(
        pw: &PendingWrite,
        guard: &RustBV,
        ctx: &crate::symbolic::SymContext,
    ) -> PendingWrite {
        let condition = Some(match &pw.condition {
            Some(c) => guard.and(c, ctx),
            None => guard.clone(),
        });
        PendingWrite {
            addr: pw.addr.clone(),
            value: pw.value.clone(),
            size: pw.size,
            condition,
            page_hint: pw.page_hint,
        }
    }
}

impl Clone for SymbolicMemory {
    fn clone(&self) -> Self {
        self.fork()
    }
}

/// Snapshot of a [`SymbolicMemory`]'s persistable state (angr-x04s.1.3).
///
/// Captures the concrete page map, symbolic-overlay objects, lazy-region
/// hints, and per-state policy flags. Per-state runtime caches
/// (`dirty_pages`, `wider_load_cache`, `multi_versions`) are NOT included;
/// they rebuild lazily after restore. `symbolic_spans` is reconstructed
/// from `symbolic_objects` at load time so the reverse index stays
/// consistent.
///
/// **Deferred to a follow-up snapshot phase:** `multi_objects` and
/// `pending_writes` carry lazy-store state used by the
/// symbolic-address optimization path. The fauxware prototype does not
/// exercise either; for now both restore to empty. Callers that snapshot
/// a state mid-Multi/Pending must `flush_multi_cells` / drain pending
/// writes first or accept that the lazy queue is dropped.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolicMemorySnapshot {
    pub pages: std::collections::BTreeMap<u64, MemoryPage>,
    pub symbolic_objects: std::collections::BTreeMap<u64, RustBV>,
    pub next_sym_id: u64,
    pub default_permissions: Permission,
    pub endness: Endness,
    pub lazy_regions: Vec<(u64, u64)>,
    pub imported_addrs: Vec<u64>,
    pub zero_fill_unconstrained: bool,
    pub enforce_permissions: bool,
    pub enforce_nx: bool,
    /// Pages written since the last `clear_dirty_pages()` (angr-ype54).
    ///
    /// This one runtime cache MUST survive the round-trip: the Python
    /// callback state is refreshed by replaying Rust's dirty pages
    /// (`rust_state_sync::_replay_rust_dirty_pages`). Dropping the set on
    /// migration strands every write made between the last callback and the
    /// migration, so the cached Python `SimState` keeps stale (zero) bytes
    /// and any SimProcedure reading them computes a concrete guard.
    ///
    /// `#[serde(default)]` keeps older snapshots loadable (empty set == the
    /// pre-fix behaviour).
    #[serde(default)]
    pub dirty_pages: Vec<u64>,
}

impl SymbolicMemory {
    /// Build a serializable snapshot (angr-x04s.1.3).
    pub fn to_snapshot(&self) -> SymbolicMemorySnapshot {
        let pages: std::collections::BTreeMap<u64, MemoryPage> =
            self.pages.iter().map(|(k, v)| (*k, v.clone())).collect();
        let symbolic_objects: std::collections::BTreeMap<u64, RustBV> = self
            .symbolic_objects
            .iter()
            .map(|(addr, bv)| (addr.raw(), bv.clone()))
            .collect();
        let mut imported_addrs: Vec<u64> = self.imported_addrs.iter().map(|a| a.raw()).collect();
        imported_addrs.sort_unstable();
        SymbolicMemorySnapshot {
            pages,
            symbolic_objects,
            next_sym_id: self.next_sym_id,
            default_permissions: self.default_permissions,
            endness: self.endness,
            lazy_regions: self.lazy_regions.clone(),
            imported_addrs,
            zero_fill_unconstrained: self.zero_fill_unconstrained,
            enforce_permissions: self.enforce_permissions,
            enforce_nx: self.enforce_nx,
            dirty_pages: {
                let mut d: Vec<u64> = self.dirty_pages.iter().copied().collect();
                d.sort_unstable();
                d
            },
        }
    }

    /// Restore a snapshot into a fresh [`SymbolicMemory`]. Rebuilds the
    /// `symbolic_spans` reverse index from `symbolic_objects` (per-byte
    /// entries keyed by base + offset, width recorded as the BV width in
    /// bits). Lazy-store side tables (`multi_objects`, `pending_writes`)
    /// start empty — see [`SymbolicMemorySnapshot`].
    pub fn from_snapshot(snap: SymbolicMemorySnapshot) -> Self {
        let mut pages: OrdMap<u64, MemoryPage> = OrdMap::new();
        for (k, v) in snap.pages {
            pages.insert(k, v);
        }
        let symbolic_objects: FxHashMap<Address, RustBV> = snap
            .symbolic_objects
            .into_iter()
            .map(|(k, v)| (Address::new(k), v))
            .collect();
        let mut symbolic_spans: FxHashMap<Address, (Address, u32)> = FxHashMap::default();
        for (base, bv) in symbolic_objects.iter() {
            let width = bv.width();
            let bytes = width.div_ceil(8);
            for i in 0..bytes {
                symbolic_spans.insert(Address::new(base.raw() + i as u64), (*base, width));
            }
        }
        let imported_addrs: FxHashSet<Address> =
            snap.imported_addrs.into_iter().map(Address::new).collect();
        SymbolicMemory {
            pages,
            symbolic_objects,
            next_sym_id: snap.next_sym_id,
            default_permissions: snap.default_permissions,
            endness: snap.endness,
            dirty_pages: snap.dirty_pages.into_iter().collect(),
            lazy_regions: snap.lazy_regions,
            symbolic_spans,
            multi_objects: FxHashMap::default(),
            pending_writes: Vec::new(),
            zero_fill_unconstrained: snap.zero_fill_unconstrained,
            imported_addrs,
            enforce_permissions: snap.enforce_permissions,
            enforce_nx: snap.enforce_nx,
            multi_versions: FxHashMap::default(),
            wider_load_cache: RefCell::new(FxHashMap::default()),
        }
    }
}
