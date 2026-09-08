//! Memory page primitives: `Permission`, `MemoryPage`, and page-size constants.
//!
//! Extracted from the previous monolithic `memory.rs`. Re-exported from
//! `crate::memory` so existing call sites that use `crate::memory::MemoryPage`
//! / `crate::memory::Permission` / `crate::memory::PAGE_SIZE` keep working.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Page size in bytes (4KB).
pub const PAGE_SIZE: u64 = 4096;

/// Page mask for address calculation.
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// A page *number* — `addr >> 12`, not a byte address.
///
/// Both are bare `u64` otherwise, and every audit round has turned up a site
/// that open-codes the shift (angr-c7xno.49 found `PAGE_SIZE` expressed three
/// incompatible ways in `interpreter/` alone; angr-sqfj8.140 found
/// `syscalls/page.rs` redefining the constants). The wrapper makes the two
/// layers distinct types wherever a *collection* is keyed by page number, so
/// handing it an address — or a page number where an address belongs — stops
/// compiling instead of silently addressing page 0x1000-ish.
///
/// The `pages` / `dirty_pages` / `lazy_regions` structures inside `memory/`
/// deliberately stay raw `u64`: see the `invariant-address-vs-page-number`
/// note on `Address`. `PageIndex` is for callers *above* that layer, which
/// convert once at the boundary via [`PageIndex::get`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct PageIndex(u64);

impl PageIndex {
    /// Bits of in-page byte offset — the shift between an address and a page
    /// number. Derived from `PAGE_SIZE` so the two can never disagree.
    pub const SHIFT: u32 = PAGE_SIZE.trailing_zeros();

    /// The page containing byte address `addr`.
    #[inline]
    pub fn of(addr: u64) -> Self {
        Self(addr >> Self::SHIFT)
    }

    /// Wrap an already-shifted page number (e.g. one that arrived from the raw
    /// `u64` page API inside `memory/`).
    #[inline]
    pub fn from_raw(page_num: u64) -> Self {
        Self(page_num)
    }

    /// The raw page number, for the raw-`u64` page API inside `memory/`.
    #[inline]
    pub fn get(self) -> u64 {
        self.0
    }

    /// First byte address of this page.
    #[inline]
    pub fn base_addr(self) -> u64 {
        self.0 << Self::SHIFT
    }

    /// Every page touched by `[addr, addr + len)`, inclusive of both ends.
    ///
    /// Empty when `len == 0` — a zero-length access touches no page, whereas
    /// the open-coded `first..=last` form the call sites used returns one.
    /// `addr + len` is saturating, so a range running off the top of the
    /// address space clamps rather than wrapping to page 0.
    pub fn range_covering(addr: u64, len: u64) -> impl Iterator<Item = PageIndex> + Clone {
        let first = Self::of(addr).get();
        // `len - 1` is safe: `len == 0` is filtered out below, and `.max(1)`
        // only keeps the expression well-defined until it is.
        let last = Self::of(addr.saturating_add(len.max(1) - 1)).get();
        (len != 0)
            .then_some(first..=last)
            .into_iter()
            .flatten()
            .map(PageIndex)
    }
}

/// Number of u64 words in the per-page symbolic bitmap.
/// One bit per byte: PAGE_SIZE bytes / 64 bits per u64 = PAGE_SIZE / 64 words.
pub const BITMAP_WORDS: usize = (PAGE_SIZE / 64) as usize;
const _: () = assert!(
    BITMAP_WORDS * 64 == PAGE_SIZE as usize,
    "bitmap must cover one bit per page byte"
);

/// Bits per bitmap word (u64).
const BITMAP_BITS_PER_WORD: u16 = 64;

/// One page overlay bitmap: one bit per page byte, lazily allocated.
///
/// `MemoryPage` holds two of these (`symbolic_bitmap`, `multi_bitmap`). Before
/// angr-12jjk.12 each accessor pair (`has_*`, `is_*`, `mark_*`, `clear_*`)
/// open-coded the same `Option<Box<[u64; BITMAP_WORDS]>>` word/bit math once
/// per overlay, so a bit-math fix could land in only one of the two copies.
/// All of it lives here now; the `MemoryPage` methods are thin named wrappers
/// that keep the public API (and its debug asserts) unchanged.
///
/// `None` means "no bits set" — the allocation is dropped by
/// [`clear_range`](Self::clear_range) once the last bit clears, which is what
/// makes [`any_set`](Self::any_set) an O(1) `is_some()` check.
#[derive(Clone, Default, PartialEq, Eq)]
struct PageBitmap(Option<Box<[u64; BITMAP_WORDS]>>);

impl PageBitmap {
    /// Decompose a page byte offset into (word index, bit index).
    #[inline]
    fn word_bit(offset: u16) -> (usize, u16) {
        (
            (offset / BITMAP_BITS_PER_WORD) as usize,
            offset % BITMAP_BITS_PER_WORD,
        )
    }

    /// Whether any bit is set (O(1) — see the drop-when-empty invariant).
    #[inline]
    fn any_set(&self) -> bool {
        self.0.is_some()
    }

    /// Whether the bit for byte `offset` is set.
    #[inline]
    fn is_set(&self, offset: u16) -> bool {
        match &self.0 {
            None => false,
            Some(bitmap) => {
                let (word_idx, bit_idx) = Self::word_bit(offset);
                bitmap[word_idx] & (1u64 << bit_idx) != 0
            }
        }
    }

    /// Set bits `[offset, end)`, allocating the bitmap if needed. Setting can
    /// never empty the bitmap, so there is no drop step (unlike
    /// [`clear_range`](Self::clear_range)).
    fn set_range(&mut self, offset: u16, end: u16) {
        let bitmap = self.0.get_or_insert_with(|| Box::new([0u64; BITMAP_WORDS]));
        for i in offset..end {
            let (word_idx, bit_idx) = Self::word_bit(i);
            bitmap[word_idx] |= 1u64 << bit_idx;
        }
    }

    /// Clear bits `[offset, end)`, dropping the allocation when it becomes
    /// all-zero (angr-24pv4.3).
    fn clear_range(&mut self, offset: u16, end: u16) {
        if let Some(bitmap) = &mut self.0 {
            for i in offset..end {
                let (word_idx, bit_idx) = Self::word_bit(i);
                bitmap[word_idx] &= !(1u64 << bit_idx);
            }
            if bitmap.iter().all(|&w| w == 0) {
                self.0 = None;
            }
        }
    }

    /// Page byte offsets whose bit is set, ascending.
    fn set_offsets(&self) -> Vec<u16> {
        let bitmap = match &self.0 {
            Some(b) => b,
            None => return Vec::new(),
        };
        let mut offsets = Vec::new();
        for (word_idx, &word) in bitmap.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (word_idx as u16) * BITMAP_BITS_PER_WORD;
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as u16;
                // overflow-ok: `word_idx < BITMAP_WORDS` and `bit < 64`, so the
                // sum is a page offset < PAGE_SIZE — far below `u16::MAX`.
                offsets.push(base + bit);
                bits &= bits - 1; // Clear lowest set bit
            }
        }
        offsets
    }

    /// Serde shadow form: `None` when no bits are set.
    fn to_words(&self) -> Option<Vec<u64>> {
        self.0.as_ref().map(|b| b.to_vec())
    }

    /// Rebuild from the serde shadow form. A wrong-length word vector falls
    /// back to "no bits set" so a corrupted snapshot still loads.
    fn from_words(words: Option<Vec<u64>>) -> Self {
        PageBitmap(words.and_then(|v| {
            if v.len() != BITMAP_WORDS {
                return None;
            }
            let mut arr = [0u64; BITMAP_WORDS];
            arr.copy_from_slice(&v);
            Some(Box::new(arr))
        }))
    }
}

/// Memory permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permission {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Permission {
    pub const NONE: Permission = Permission {
        read: false,
        write: false,
        execute: false,
    };
    pub const R: Permission = Permission {
        read: true,
        write: false,
        execute: false,
    };
    pub const RW: Permission = Permission {
        read: true,
        write: true,
        execute: false,
    };
    pub const RX: Permission = Permission {
        read: true,
        write: false,
        execute: true,
    };
    pub const RWX: Permission = Permission {
        read: true,
        write: true,
        execute: true,
    };
    pub const W: Permission = Permission {
        read: false,
        write: true,
        execute: false,
    };
    pub const X: Permission = Permission {
        read: false,
        write: false,
        execute: true,
    };

    pub fn from_bits(bits: u8) -> Self {
        Permission {
            read: bits & 0x4 != 0,
            write: bits & 0x2 != 0,
            execute: bits & 0x1 != 0,
        }
    }

    pub fn to_bits(&self) -> u8 {
        let mut bits = 0u8;
        if self.read {
            bits |= 0x4;
        }
        if self.write {
            bits |= 0x2;
        }
        if self.execute {
            bits |= 0x1;
        }
        bits
    }

    pub fn allows(&self, other: Permission) -> bool {
        (!other.read || self.read)
            && (!other.write || self.write)
            && (!other.execute || self.execute)
    }
}

/// A memory page with copy-on-write semantics.
///
/// ## Serialization
///
/// Implements `Serialize`/`Deserialize` via the `MemoryPageData` shadow
/// type — the `Arc<Vec<u8>>` collapses to an owned `Vec` and the
/// `Box<[u64; BITMAP_WORDS]>` bitmaps collapse to `Vec<u64>` on the wire,
/// rebuilding the Arc / Box on load. Symbolic AST values are NOT carried
/// on the page itself (they live in `SymbolicMemory::symbolic_objects` etc.);
/// only the per-byte symbolic/multi bitmaps are part of the snapshot.
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "MemoryPageData", into = "MemoryPageData")]
pub struct MemoryPage {
    /// Concrete data (4KB) - uses Arc for CoW.
    data: Arc<Vec<u8>>,
    /// Permissions for this page.
    permissions: Permission,
    /// Base address of the page.
    base_addr: u64,
    /// Bitmap tracking symbolic bytes: bit i set means byte i is symbolic.
    /// BITMAP_WORDS u64s = PAGE_SIZE bits = one bit per byte in a page.
    /// Boxed to keep MemoryPage small when not needed (empty = fully concrete).
    symbolic_bitmap: PageBitmap,
    /// Bitmap tracking Multi-cell bytes: bit i set means byte i carries
    /// lazy alternatives in `SymbolicMemory::multi_objects` (Phase 1 of
    /// angr-czph). Parallel to `symbolic_bitmap`; a byte may be marked in
    /// at most one of the two bitmaps at a time. Empty means no Multi
    /// cells on this page — common case, so we pay one Option discriminant
    /// rather than a 512-byte bitmap allocation.
    multi_bitmap: PageBitmap,
}

impl MemoryPage {
    /// End offset of the page-local range `[offset, offset + len)`, clamped to
    /// the page end.
    ///
    /// The five accessors below (`load_concrete`, `store_concrete`,
    /// `mark_symbolic`, `clear_symbolic`, `mark_multi`/`clear_multi`) each
    /// open-coded `(offset as usize + len).min(PAGE_SIZE as usize)`, and the
    /// single-byte pair spelled it `offset + 1` in `u16` — which *wraps* to 0
    /// for `offset == u16::MAX`, turning a `set_range`/`clear_range` into a
    /// silent no-op rather than the out-of-range panic the bitmap index would
    /// otherwise raise (angr-xloth.4). Widening to `usize` first makes the sum
    /// exact for any `u16` offset and any real slice length, and the clamp
    /// makes the cast back to `u16` lossless.
    ///
    /// `offset` past the page end yields an end *below* `offset`, i.e. an empty
    /// range for the bitmap helpers and a panicking slice index for `data` —
    /// both matching the pre-existing release behaviour of a caller that
    /// violates the `debug_assert!`s.
    #[inline]
    fn clamped_end(offset: u16, len: usize) -> u16 {
        (offset as usize)
            .saturating_add(len)
            .min(PAGE_SIZE as usize) as u16
    }

    /// Create a new zeroed page.
    pub fn new(base_addr: u64, permissions: Permission) -> Self {
        MemoryPage {
            data: Arc::new(vec![0u8; PAGE_SIZE as usize]),
            permissions,
            base_addr,
            symbolic_bitmap: PageBitmap::default(),
            multi_bitmap: PageBitmap::default(),
        }
    }

    /// Create a page from existing data.
    pub fn from_data(base_addr: u64, data: &[u8], permissions: Permission) -> Self {
        let mut page_data = vec![0u8; PAGE_SIZE as usize];
        let copy_len = data.len().min(PAGE_SIZE as usize);
        page_data[..copy_len].copy_from_slice(&data[..copy_len]);

        MemoryPage {
            data: Arc::new(page_data),
            permissions,
            base_addr,
            symbolic_bitmap: PageBitmap::default(),
            multi_bitmap: PageBitmap::default(),
        }
    }

    /// Get the base address.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Get the permissions.
    pub fn permissions(&self) -> Permission {
        self.permissions
    }

    /// Set the permissions.
    pub fn set_permissions(&mut self, perm: Permission) {
        self.permissions = perm;
    }

    /// Check if this page has any symbolic bytes.
    #[inline]
    pub fn has_symbolic(&self) -> bool {
        self.symbolic_bitmap.any_set()
    }

    /// CoW structural-sharing skip predicate (angr-op0dn.11.1.1, S5a).
    ///
    /// Returns `true` when this page is provably byte-for-byte identical to
    /// `other` *without walking any bytes*: the concrete `data` buffers are the
    /// same `Arc` allocation (ptr-equal — untouched since a common fork, because
    /// [`store_concrete`] breaks sharing via `Arc::make_mut`) and both symbolic
    /// overlay bitmaps match. A merge can skip such a page entirely, making merge
    /// cost proportional to *divergent* pages instead of *all shared* pages.
    ///
    /// Bitmap comparison is a fixed 512-byte `[u64; BITMAP_WORDS]` check (O(1) per
    /// page), independent of PAGE_SIZE; it is only reached when `data` already
    /// ptr-matches, so it is off the hot path for divergent pages.
    #[cfg(test)]
    #[inline]
    pub(crate) fn shares_data_with(&self, other: &MemoryPage) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && self.symbolic_bitmap == other.symbolic_bitmap
            && self.multi_bitmap == other.multi_bitmap
    }

    /// Production merge skip predicate (angr-op0dn.11.2.1, productionizes S5a).
    ///
    /// Returns `true` when the two pages are provably byte-for-byte identical
    /// with no symbolic/Multi overlay on *either* side, established WITHOUT
    /// materializing any bytes: `data` is the same `Arc` allocation (untouched
    /// since a common fork — [`store_concrete`](Self::store_concrete) is the
    /// only thing that breaks
    /// sharing, via `Arc::make_mut`) and neither page carries a symbolic or Multi
    /// bitmap. `SymbolicMemory::merge` skips such pages, making merge cost
    /// proportional to *divergent* pages instead of *all shared* pages.
    ///
    /// This is intentionally narrower than the `#[cfg(test)]`
    /// `shares_data_with`: it does NOT
    /// admit ptr-shared pages whose symbolic bitmaps merely *match*. A symbolic
    /// store touches only the bitmap + `SymbolicMemory::symbolic_objects` and
    /// leaves `data` ptr-shared (see `store` in `store.rs`), so two arms can
    /// share `data` and have equal symbolic bitmaps yet hold *different*
    /// `symbolic_objects` values at an already-symbolic byte — a divergence merge
    /// must still ITE. Requiring no symbolic overlay makes the skip a strict
    /// subset of the existing value-equality early-out (`s_data == o_data &&
    /// !has_symbolic`), so it is provably semantics-preserving.
    #[inline]
    pub(crate) fn is_shared_identical(&self, other: &MemoryPage) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
            && !self.symbolic_bitmap.any_set()
            && !other.symbolic_bitmap.any_set()
            && !self.multi_bitmap.any_set()
            && !other.multi_bitmap.any_set()
    }

    /// Get the symbolic byte offsets.
    pub fn symbolic_offsets(&self) -> Vec<u16> {
        self.symbolic_bitmap.set_offsets()
    }

    /// Get the Multi byte offsets.
    ///
    /// Parallel to [`symbolic_offsets`](Self::symbolic_offsets). `merge` uses
    /// it to walk the Multi cells of a page it is adopting wholesale from the
    /// other arm, so the flat `SymbolicMemory::multi_objects` payloads can be
    /// copied alongside the bitmap the page clone brings with it.
    pub fn multi_offsets(&self) -> Vec<u16> {
        self.multi_bitmap.set_offsets()
    }

    /// Load bytes from this page (concrete only).
    pub fn load_concrete(&self, offset: u16, size: u16) -> Vec<u8> {
        let start = offset as usize;
        let end = Self::clamped_end(offset, size as usize) as usize;
        self.data[start..end].to_vec()
    }

    /// Store bytes to this page (concrete).
    pub fn store_concrete(&mut self, offset: u16, bytes: &[u8]) {
        debug_assert!(
            (offset as usize).saturating_add(bytes.len()) <= PAGE_SIZE as usize,
            "store_concrete: offset {} + len {} exceeds PAGE_SIZE {}",
            offset,
            bytes.len(),
            PAGE_SIZE
        );
        let loop_end = Self::clamped_end(offset, bytes.len());
        // Copy-on-write: if shared, make a unique copy
        let data = Arc::make_mut(&mut self.data);
        let start = offset as usize;
        let end = loop_end as usize;
        data[start..end].copy_from_slice(&bytes[..end.saturating_sub(start)]);

        // Clear symbolic bitmap bits for overwritten bytes
        self.symbolic_bitmap.clear_range(offset, loop_end);
        // A concrete overwrite also clears any Multi-cell marker — the cell
        // is no longer carrying lazy alternatives. The owning
        // `SymbolicMemory::multi_objects` entries must be dropped by the
        // caller (the page does not own that map).
        self.multi_bitmap.clear_range(offset, loop_end);
    }

    /// Mark bytes as symbolic.
    pub fn mark_symbolic(&mut self, offset: u16, size: u16) {
        debug_assert!(
            (offset as usize).saturating_add(size as usize) <= PAGE_SIZE as usize,
            "mark_symbolic: offset {offset} + size {size} exceeds PAGE_SIZE {PAGE_SIZE}"
        );
        let loop_end = Self::clamped_end(offset, size as usize);
        self.symbolic_bitmap.set_range(offset, loop_end);
    }

    /// Clear symbolic bitmap bits in a range, without touching `data`.
    ///
    /// Used by `SymbolicMemory::store_concrete_lazy` (angr-7qon) when a
    /// concrete write truncates a wider sym based at the same address —
    /// the surviving trailing bytes lose their sym tracking and must be
    /// reclassified as concrete so subsequent loads don't blow up with
    /// "symbolic bytes not fully tracked". Drops the bitmap when empty.
    pub fn clear_symbolic(&mut self, offset: u16, size: u16) {
        debug_assert!(
            (offset as usize).saturating_add(size as usize) <= PAGE_SIZE as usize,
            "clear_symbolic: offset {offset} + size {size} exceeds PAGE_SIZE {PAGE_SIZE}"
        );
        let loop_end = Self::clamped_end(offset, size as usize);
        self.symbolic_bitmap.clear_range(offset, loop_end);
    }

    /// Check if a byte is symbolic.
    #[inline]
    pub fn is_symbolic(&self, offset: u16) -> bool {
        self.symbolic_bitmap.is_set(offset)
    }

    /// Mark a single byte as Multi (carrying lazy alternatives in
    /// `SymbolicMemory::multi_objects`). The caller is responsible for
    /// retiring the `symbolic_objects` entry covering this address —
    /// `SymbolicMemory::set_multi_alternatives`, the only caller, does that
    /// via `retire_symbolic_object_at`.
    ///
    /// It deliberately does *not* clear the symbolic bit: the two bitmaps are
    /// not mutually exclusive, and a byte promoted from Symbolic to Multi
    /// keeps both bits set. Multi wins by dispatch order instead — every
    /// reader probes `multi_objects` / `has_multi` before `is_symbolic`. See
    /// `memory::tests::multi::payload::test_multi_supersedes_existing_symbolic`
    /// (angr-6cp06.64), which pins that state and enumerates the readers.
    pub fn mark_multi(&mut self, offset: u16) {
        debug_assert!(
            (offset as usize) < PAGE_SIZE as usize,
            "mark_multi: offset {offset} out of range"
        );
        self.multi_bitmap
            .set_range(offset, Self::clamped_end(offset, 1));
    }

    /// Check if this page has any Multi bytes.
    ///
    /// Parallel to [`has_symbolic`](Self::has_symbolic): `merge` uses it to decide whether the
    /// page can take the concrete-equality early-out. A page carrying Multi
    /// cells must always walk its bytes because Multi divergence lives in
    /// `SymbolicMemory::multi_objects`, invisible to a `data[]` compare.
    #[inline]
    pub fn has_multi(&self) -> bool {
        self.multi_bitmap.any_set()
    }

    /// Check if a byte carries lazy Multi alternatives.
    #[inline]
    pub fn is_multi(&self, offset: u16) -> bool {
        self.multi_bitmap.is_set(offset)
    }

    /// Clear the Multi marker on a single byte. The caller is responsible
    /// for removing the corresponding entry from
    /// `SymbolicMemory::multi_objects`. Drops the page-level bitmap when
    /// it becomes empty.
    pub fn clear_multi(&mut self, offset: u16) {
        self.multi_bitmap
            .clear_range(offset, Self::clamped_end(offset, 1));
    }

    /// Fork this page (O(1) for concrete pages, bitmap clone for symbolic).
    pub fn fork(&self) -> Self {
        MemoryPage {
            data: Arc::clone(&self.data),
            permissions: self.permissions,
            base_addr: self.base_addr,
            symbolic_bitmap: self.symbolic_bitmap.clone(),
            multi_bitmap: self.multi_bitmap.clone(),
        }
    }
}

/// Serde shadow form for [`MemoryPage`].
///
/// Collapses `Arc<Vec<u8>>` to `Vec<u8>` and `Box<[u64; BITMAP_WORDS]>`
/// bitmaps to `Vec<u64>` on the wire. On deserialize, malformed bitmap
/// lengths fall back to `None` (treated as "no symbolic bytes on this
/// page"); the data buffer is right-padded / truncated to `PAGE_SIZE` so
/// a corrupted snapshot still produces a structurally-valid page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPageData {
    pub data: Vec<u8>,
    pub permissions: Permission,
    pub base_addr: u64,
    pub symbolic_bitmap: Option<Vec<u64>>,
    pub multi_bitmap: Option<Vec<u64>>,
}

impl From<MemoryPage> for MemoryPageData {
    fn from(page: MemoryPage) -> Self {
        MemoryPageData {
            data: (*page.data).clone(),
            permissions: page.permissions,
            base_addr: page.base_addr,
            symbolic_bitmap: page.symbolic_bitmap.to_words(),
            multi_bitmap: page.multi_bitmap.to_words(),
        }
    }
}

impl From<MemoryPageData> for MemoryPage {
    fn from(data: MemoryPageData) -> Self {
        let mut page_data = vec![0u8; PAGE_SIZE as usize];
        let copy_len = data.data.len().min(PAGE_SIZE as usize);
        page_data[..copy_len].copy_from_slice(&data.data[..copy_len]);

        MemoryPage {
            data: Arc::new(page_data),
            permissions: data.permissions,
            base_addr: data.base_addr,
            symbolic_bitmap: PageBitmap::from_words(data.symbolic_bitmap),
            multi_bitmap: PageBitmap::from_words(data.multi_bitmap),
        }
    }
}
test_submod!("serde_tests.rs" => serde_tests);
test_submod!("page_index_tests.rs" => page_index_tests);
