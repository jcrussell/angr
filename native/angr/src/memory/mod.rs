//! Symbolic memory system for the VEX execution engine.
//!
//! This module provides a paged memory model with:
//! - O(1) forking via copy-on-write (using Grudge's RustPage)
//! - Mixed concrete/symbolic value storage
//! - Efficient symbolic address handling
//!
//! **Which `load_*` / `store_*` variant do I want?** The entry points differ on
//! four axes — permission checking, unmapped-page handling, auto-mapping, and
//! `Multi`-cell awareness — and the `_lazy` / `_automap` / `_unified` suffixes
//! do not signal them consistently. A row-per-variant matrix lives in the
//! `memory::load` and `memory::store` module docs (angr-9ke6b.102); read it
//! before adding a variant or a new cross-cutting feature, and add a row when
//! you do.
//!
//! **Panic policy (angr-9ke6b.212):** every address that reaches this module is
//! guest-derived, so nothing here may panic on address shape — unresolvable
//! addresses surface as [`MemoryError`] variants the caller routes to the
//! Python memory model. A handful of `expect`s do survive (in
//! [`SymbolicMemory::merge`] and under `memory::multi`); rather than
//! enumerate them here — an inventory that goes stale the moment one moves
//! (angr-sqfj8.72) — the invariant is: every surviving `expect` sits under a
//! reviewed `#[allow(clippy::expect_used, reason = "...")]` whose reason
//! names the local guard proving it, and that guard is never guest input. If
//! you cannot write such a reason, the code needs a [`MemoryError`], not an
//! `expect`.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`, which also reaches the
//! `address`/`concretize_glue`/`ite_builder`/`merge`/`multi`/`page`/`store`/
//! `load`/`symbolic_objects` child modules, so a new panic on an untrusted address
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
mod merge;
mod multi;
mod page;
mod store;
mod symbolic_objects;
test_submod!(tests);
pub use address::Address;
pub use multi::{MultiAlternative, MultiFlushed, MultiPayload};
pub use page::{BITMAP_WORDS, MemoryPage, PAGE_MASK, PAGE_SIZE, PageIndex, Permission};

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
}

impl PendingWrite {
    /// Deep-translate the symbolic `addr`/`value`/`condition` BVs into
    /// `target_ctx` (angr-ahypj). `size` is context-independent and copied
    /// verbatim.
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
    /// Out of bounds access. Also the wraparound rejection surfaced by
    /// `end_page_inclusive` / `end_page_exclusive` when `addr + size` runs
    /// past the top of the 64-bit address space (angr-03vl4.34/.35).
    #[error("out of bounds access at 0x{addr:x} (size {size})")]
    OutOfBounds { addr: u64, size: u64 },
    /// A zero-size access. Rejected up-front by `end_page_inclusive` rather
    /// than allowed to compute `addr + 0 - 1` (see that function for why).
    /// A zero-byte load could not produce a valid BV anyway (Z3 has no
    /// 0-width bitvector), and a zero-byte store has nothing to write, so
    /// both are caller bugs — surfacing them lets the Python side fall back
    /// or raise instead of hanging.
    #[error("zero-size memory access at 0x{addr:x}")]
    ZeroSize { addr: u64 },
    /// A symbolic value whose width is not a whole number of bytes. Memory is
    /// byte-addressed, so `width_bits / 8` truncates: a sub-byte width yields
    /// size 0 (no byte marked symbolic at all) and a width like 12 silently
    /// drops the high nibble. `import_symbolic_value` (angr-sqfj8.73) and
    /// `store_concrete`'s symbolic branch (angr-03vl4.40) both reject up-front
    /// rather than storing an object no load can reconstruct — the sibling of
    /// [`MemoryError::ZeroSize`], which only catches the widths that truncate
    /// all the way to size 0.
    #[error("symbolic value at 0x{addr:x} has non-byte-multiple width {width_bits}")]
    UnalignedWidth { addr: u64, width_bits: u32 },
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
///
/// angr-03vl4.34/.35: the addition is `checked_add`, not bare `+`, for the
/// mirror-image reason. `addr + size - 1` *overflows* when the access runs off
/// the top of the address space (`write_fallback_max` concretizes an
/// under-constrained store pointer to `u64::MAX` by default, so this is
/// reachable, not theoretical). With release overflow-checks off that wrapped
/// to a tiny `end_page < start_page`, making both `start_page..=end_page` and
/// `check_perms_range` empty — every mapped-page and permission check silently
/// skipped, after which the byte loop's wrapping `Address` addition walked down
/// through 0 and read/wrote a real low page. Reject as
/// [`MemoryError::OutOfBounds`] instead.
pub(super) fn end_page_inclusive(addr: u64, size: u64) -> Result<u64, MemoryError> {
    if size == 0 {
        return Err(MemoryError::ZeroSize { addr });
    }
    let last = addr
        .checked_add(size - 1)
        .ok_or(MemoryError::OutOfBounds { addr, size })?;
    Ok(last >> 12)
}

/// Exclusive page number one past the last page touched by
/// `[addr, addr + size)` — i.e. the upper bound of a `start_page..end_page`
/// half-open range.
///
/// The exclusive sibling of [`end_page_inclusive`], centralized per
/// angr-sqfj8.76 (the formula was copy-pasted at six sites). Unlike the
/// inclusive form this only ever *adds*, so `size == 0` cannot underflow —
/// it simply yields `ceil(addr / PAGE_SIZE)`, an empty range for a
/// page-aligned `addr`. Callers that must reject a zero-size access do so
/// themselves (`map` / `unmap` early-return; the store wrappers inherit
/// `store_concrete`'s `end_page_inclusive` check).
///
/// angr-03vl4.34: it *can* overflow at the top of the address space, with the
/// same silently-empty-range consequence described on [`end_page_inclusive`] —
/// `check_pages_mapped_lazy` iterates `start_page..end_page` and skips every
/// page when the sum wraps. Hence the error channel. The ceil-div is applied
/// to the checked exclusive end address rather than folded into the addition
/// (`+ PAGE_SIZE - 1` has its own overflow), and the one region that ends
/// *exactly* at the top of the address space — `addr + size == 2^64`, e.g.
/// mapping the last page — is a legal range, not an overflow.
pub(super) fn end_page_exclusive(addr: u64, size: u64) -> Result<u64, MemoryError> {
    match addr.checked_add(size) {
        Some(end_addr) => Ok(end_addr.div_ceil(PAGE_SIZE)),
        None if addr.wrapping_add(size) == 0 => Ok((u64::MAX >> 12) + 1),
        None => Err(MemoryError::OutOfBounds { addr, size }),
    }
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
///
/// # Merge coverage (angr-91vj9.3)
///
/// `RustSimState` labels `memory` `#[merge_policy = "delegate"]`, which stops
/// the top-level derive at this struct's boundary — and the same
/// "field-added-without-a-merge-line" bug family promptly recurred one level
/// deeper (angr-c7xno.50: [`Self::merge`] adopted an other-only page's
/// `multi_bitmap` but never copied the matching `multi_objects` payloads,
/// because that sidecar is a flat map here rather than nested in
/// [`MemoryPage`]). Deriving [`angr_macros::MergePolicy`] here extends the
/// guarantee: a new field — especially a new address-keyed sidecar map — does
/// not compile until it declares how [`Self::merge`] treats it. The behavioural
/// half lives in `memory/tests/merge_sidecars.rs`, which exercises every
/// address-keyed sidecar across the other-only-page adoption path.
///
/// `in_place_self` is the common label here because [`Self::merge`] mutates
/// `self` rather than building a new struct, so "self wins" means "never
/// assigned" and has no line to generate.
#[derive(angr_macros::MergePolicy)]
pub struct SymbolicMemory {
    /// Pages indexed by page number (addr >> 12).
    #[merge_policy = "joint"]
    pages: OrdMap<u64, MemoryPage>,
    /// Symbolic objects (for values that span multiple bytes).
    #[merge_policy = "joint"]
    symbolic_objects: FxHashMap<Address, RustBV>,
    /// Default permissions for new pages.
    ///
    /// Merge-invariant: merge arms are fork siblings, which inherit this from
    /// the common ancestor and have no API that reassigns it mid-branch.
    #[merge_policy = "in_place_self"]
    default_permissions: Permission,
    /// Endianness for this memory.
    ///
    /// Merge-invariant: fixed at construction from the arch, never reassigned.
    #[merge_policy = "in_place_self"]
    endness: Endness,
    /// Pages that have been modified since last clear.
    /// Stores page numbers (addr >> 12) for efficient tracking.
    ///
    /// Merged as a set union — see the accumulating-sidecar block at the end
    /// of [`Self::merge`].
    #[merge_policy = "joint"]
    dirty_pages: FxHashSet<u64>,
    /// Lazy regions: page ranges that CAN have pages fetched on-demand.
    /// Stores (start_page_num, end_page_num) pairs.
    /// When a load hits an unmapped page in a lazy region, the interpreter
    /// should fetch it from Python rather than failing.
    ///
    /// Merged as a deduplicated union — see the accumulating-sidecar block at
    /// the end of [`Self::merge`].
    #[merge_policy = "joint"]
    lazy_regions: Vec<(u64, u64)>,
    /// Reverse index for symbolic objects: maps each byte offset within a
    /// symbolic object to (base_addr, width_bits). Enables O(1) lookup when
    /// loading a byte that falls inside a wider symbolic object.
    #[merge_policy = "joint"]
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
    #[merge_policy = "joint"]
    multi_objects: FxHashMap<Address, MultiPayload>,
    /// Deferred symbolic stores. Instead of eagerly concretizing symbolic
    /// addresses at store time, we append here and materialize on load.
    #[merge_policy = "joint"]
    pending_writes: Vec<PendingWrite>,
    /// If true, fill unconstrained memory with zeros instead of symbolic values.
    /// Corresponds to angr's ZERO_FILL_UNCONSTRAINED_MEMORY option.
    ///
    /// Merge-invariant: a SimOption mirror, pushed from Python at state setup
    /// and identical across fork siblings.
    #[merge_policy = "in_place_self"]
    zero_fill_unconstrained: bool,
    /// Addresses of symbolic values imported from Python.
    /// Used to filter get_state_symbolic_z3_asts: even if the binary modifies
    /// an imported value (turning Symbolic→Expression), the address should be
    /// excluded from export since Python already has the correct original value.
    ///
    /// Merged as a set union — see the accumulating-sidecar block at the end
    /// of [`Self::merge`].
    #[merge_policy = "joint"]
    imported_addrs: FxHashSet<Address>,
    /// If true, enforce per-page R/W permissions on load and store. Mirrors
    /// angr's STRICT_PAGE_ACCESS option. Default is false to keep existing
    /// callers (which often map all memory as RWX or rely on Python perms)
    /// working unchanged.
    ///
    /// Merge-invariant: a SimOption mirror, see `zero_fill_unconstrained`.
    #[merge_policy = "in_place_self"]
    enforce_permissions: bool,
    /// If true (and `enforce_permissions` is also true), reject instruction
    /// fetches from mapped pages without the X bit. Mirrors angr's ENABLE_NX
    /// option: Python's heavy VEX engine only fires the non-executable check
    /// when BOTH STRICT_PAGE_ACCESS and ENABLE_NX are in state.options
    /// (`HeavyVEXMixin::process_successors` in
    /// angr/engines/vex/heavy/heavy.py). Default false.
    ///
    /// Merge-invariant: a SimOption mirror, see `zero_fill_unconstrained`.
    #[merge_policy = "in_place_self"]
    enforce_nx: bool,
    /// Per-byte monotonic version counter for Multi cells. Bumped on every
    /// `set_multi_alternatives` and `clear_multi_at` so the Phase 4.1
    /// wider-load cache (`wider_load_cache`) can detect any installation
    /// change at a byte address without comparing payload contents.
    /// Versions persist across flush/reinstall so the fingerprint of a
    /// post-flush Multi byte differs from the cached pre-flush snapshot
    /// even when both happen to have the same alternative count.
    ///
    /// Merged implicitly but genuinely: every merge-installed Multi cell goes
    /// through `set_multi_alternatives`, which bumps the version at that byte.
    #[merge_policy = "joint"]
    pub(super) multi_versions: FxHashMap<Address, u64>,
    /// Phase 4.1 (angr-mmdh.1): cached results from
    /// `assemble_load_with_multi`, keyed by `(addr, size)`. Each entry
    /// stores a per-byte fingerprint (Multi version+default_byte, or the
    /// concrete byte for non-Multi cells) and the assembled `RustBV`.
    /// Lookups rebuild the fingerprint and reuse the cached BV when it
    /// matches — skipping the per-byte concat + ITE-rebuild that
    /// dominated Phase 2 gate-on cost. RefCell because the cache lives
    /// on the load path (`&self`).
    ///
    /// Not merged, and does not need to be: entries are validated against a
    /// freshly recomputed fingerprint on every hit, and
    /// `compute_wider_load_fingerprint` returns `None` for any range holding a
    /// plain-Symbolic byte — which is exactly what a merged byte becomes. A
    /// merge therefore cannot leave a stale entry reachable.
    #[merge_policy = "in_place_self"]
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
        // angr-03vl4.34: a region running off the top of the address space is
        // a caller bug. Skipping it keeps the pre-fix effective behaviour (the
        // wrapped range was empty too) without the silence, and without the
        // alternative — clamping to the last page — turning `map(0, u64::MAX)`
        // from a no-op into 2^52 page insertions.
        let end_page = silent_default!(
            cat_c,
            end_page_exclusive(addr.raw(), size),
            return,
            |err| "map: {err}"
        );

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
        // angr-03vl4.34: mirror `map`'s overflow handling — see the note there.
        let end_page = silent_default!(
            cat_c,
            end_page_exclusive(addr.raw(), size),
            return,
            |err| "unmap: {err}"
        );

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
    ) -> Result<MultiFlushed, MemoryError> {
        // Flush Multi cells regardless of pending_writes status — Phase 2
        // makes Multi cells the default for symbolic-address stores, so
        // export correctness depends on flushing them even when the
        // pending_writes queue (a separate, scaffolded path) is empty.
        let proof = self.flush_multi_cells(ctx);

        if self.pending_writes.is_empty() {
            return Ok(proof);
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
        Ok(proof)
    }

    /// Materialize one candidate of a pending symbolic-address write
    /// (angr-24pv4.3) shared by the `Multiple` and `Strided` arms of
    /// `flush_pending_writes`. Builds `mem[candidate] = If(pw.addr ==
    /// candidate [&& pw.condition], pw.value, current)` where `current` is
    /// the existing value.
    ///
    /// Loads that fail are classified rather than collapsed to a single
    /// zero default (angr-9ke6b.100): `UnmappedPageInRegion` — the page is
    /// unmapped *in Rust* but declared lazy, so Python still holds its
    /// backer data — propagates to the caller, because zero-filling
    /// `current` would bake `If(addr == candidate, value, 0)` over bytes
    /// the binary actually initialized. Any other load failure means there
    /// is no prior value to preserve, so `current` is zero.
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
            // Surface the fetch-me signal instead of inventing zeros — same
            // contract as the `store_concrete_lazy` below, which fails the
            // identical page range anyway. When the pending-writes queue is
            // activated the caller must fetch `page_addr` from Python and
            // re-run the flush.
            Err(e @ MemoryError::UnmappedPageInRegion { .. }) => return Err(e),
            // SILENT(cat-a): a never-mapped page or a permission denial means
            // there is no prior value to preserve, so zero is the correct
            // `current` for a fresh cell. `store_concrete_lazy` re-checks the
            // mapping below, so nothing lands on absent memory.
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
        // Mirror `map`/`unmap`: a zero-length region covers no pages regardless
        // of alignment. Without this guard a non-page-aligned `start_addr`
        // rounds `end_page` up past `start_page` and registers a spurious
        // one-page region, which turns that page's hard `Unmapped` load error
        // into an `UnmappedPageInRegion` fetch request Python cannot serve.
        if size == 0 {
            return;
        }
        let start_addr = start_addr.into();
        let start_page = start_addr.page_num();
        // angr-03vl4.34: mirror `map`'s overflow handling — see the note there.
        // A dropped lazy region only costs on-demand page fetching, not
        // correctness, but the wrapped range was silently empty all the same.
        let end_page = silent_default!(
            cat_c,
            end_page_exclusive(start_addr.raw(), size),
            return,
            |err| "add_lazy_region: {err}"
        );
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
/// `multi_objects` and `pending_writes` carry lazy-store state used by the
/// symbolic-address optimization path; both restore to empty (their content
/// is already folded into `pages`/`symbolic_objects` by the time
/// `to_snapshot` runs). [`SymbolicMemory::to_snapshot`] requires a
/// [`MultiFlushed`] proof precisely so that folding has always happened by
/// construction (angr-sqfj8.71) — there is no discipline left for a caller to
/// get wrong here.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolicMemorySnapshot {
    pub pages: std::collections::BTreeMap<u64, MemoryPage>,
    pub symbolic_objects: std::collections::BTreeMap<u64, RustBV>,
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
    ///
    /// Requires proof of a prior [`Self::flush_multi_cells`] call (angr-sqfj8.71):
    /// without it, Multi-covered bytes installed by the lazy symbolic-address
    /// store path are silently absent from both `pages` and `symbolic_objects`,
    /// so a persisted/migrated state would silently lose data. `_proof` is
    /// unused by value — its existence is the contract.
    pub fn to_snapshot(&self, _proof: &MultiFlushed) -> SymbolicMemorySnapshot {
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
