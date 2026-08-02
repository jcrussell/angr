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

/// Number of u64 words in the per-page symbolic bitmap.
/// One bit per byte: PAGE_SIZE bytes / 64 bits per u64 = PAGE_SIZE / 64 words.
pub const BITMAP_WORDS: usize = (PAGE_SIZE / 64) as usize;
const _: () = assert!(
    BITMAP_WORDS * 64 == PAGE_SIZE as usize,
    "bitmap must cover one bit per page byte"
);

/// Bits per bitmap word (u64).
const BITMAP_BITS_PER_WORD: u16 = 64;

/// Clear bits `[offset, end)` in an optional page bitmap, dropping the
/// allocation when it becomes all-zero (angr-24pv4.3). Shared by
/// `store_concrete` (symbolic + multi clears), `clear_symbolic`, and
/// `clear_multi`. Centralizes the word/bit decomposition and the
/// drop-when-empty invariant that was previously copy-pasted at each site.
fn clear_bitmap_range(slot: &mut Option<Box<[u64; BITMAP_WORDS]>>, offset: u16, end: u16) {
    if let Some(bitmap) = slot {
        for i in offset..end {
            bitmap[(i / BITMAP_BITS_PER_WORD) as usize] &= !(1u64 << (i % BITMAP_BITS_PER_WORD));
        }
        if bitmap.iter().all(|&w| w == 0) {
            *slot = None;
        }
    }
}

/// Set bits `[offset, end)` in a page bitmap (angr-24pv4.3). Setting a bit
/// can never empty the bitmap, so there is no drop step (unlike
/// `clear_bitmap_range`).
fn set_bitmap_range(bitmap: &mut [u64; BITMAP_WORDS], offset: u16, end: u16) {
    for i in offset..end {
        bitmap[(i / BITMAP_BITS_PER_WORD) as usize] |= 1u64 << (i % BITMAP_BITS_PER_WORD);
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
    /// Boxed to keep MemoryPage small when not needed (None = fully concrete).
    symbolic_bitmap: Option<Box<[u64; BITMAP_WORDS]>>,
    /// Bitmap tracking Multi-cell bytes: bit i set means byte i carries
    /// lazy alternatives in `SymbolicMemory::multi_objects` (Phase 1 of
    /// angr-czph). Parallel to `symbolic_bitmap`; a byte may be marked in
    /// at most one of the two bitmaps at a time. `None` means no Multi
    /// cells on this page — common case, so we pay one Option discriminant
    /// rather than a 512-byte bitmap allocation.
    multi_bitmap: Option<Box<[u64; BITMAP_WORDS]>>,
}

impl MemoryPage {
    /// Create a new zeroed page.
    pub fn new(base_addr: u64, permissions: Permission) -> Self {
        MemoryPage {
            data: Arc::new(vec![0u8; PAGE_SIZE as usize]),
            permissions,
            base_addr,
            symbolic_bitmap: None,
            multi_bitmap: None,
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
            symbolic_bitmap: None,
            multi_bitmap: None,
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
        self.symbolic_bitmap.is_some()
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
    /// since a common fork — [`store_concrete`] is the only thing that breaks
    /// sharing, via `Arc::make_mut`) and neither page carries a symbolic or Multi
    /// bitmap. `SymbolicMemory::merge` skips such pages, making merge cost
    /// proportional to *divergent* pages instead of *all shared* pages.
    ///
    /// This is intentionally narrower than [`shares_data_with`]: it does NOT
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
            && self.symbolic_bitmap.is_none()
            && other.symbolic_bitmap.is_none()
            && self.multi_bitmap.is_none()
            && other.multi_bitmap.is_none()
    }

    /// Get the symbolic byte offsets.
    pub fn symbolic_offsets(&self) -> Vec<u16> {
        let bitmap = match &self.symbolic_bitmap {
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
                offsets.push(base + bit);
                bits &= bits - 1; // Clear lowest set bit
            }
        }
        offsets
    }

    /// Load bytes from this page (concrete only).
    pub fn load_concrete(&self, offset: u16, size: u16) -> Vec<u8> {
        let start = offset as usize;
        let end = (start + size as usize).min(PAGE_SIZE as usize);
        self.data[start..end].to_vec()
    }

    /// Store bytes to this page (concrete).
    pub fn store_concrete(&mut self, offset: u16, bytes: &[u8]) {
        debug_assert!(
            (offset as usize + bytes.len()) <= PAGE_SIZE as usize,
            "store_concrete: offset {} + len {} exceeds PAGE_SIZE {}",
            offset,
            bytes.len(),
            PAGE_SIZE
        );
        // Copy-on-write: if shared, make a unique copy
        let data = Arc::make_mut(&mut self.data);
        let start = offset as usize;
        let end = (start + bytes.len()).min(PAGE_SIZE as usize);
        data[start..end].copy_from_slice(&bytes[..(end - start)]);

        let loop_end = (offset as usize + bytes.len()).min(PAGE_SIZE as usize) as u16;
        // Clear symbolic bitmap bits for overwritten bytes
        clear_bitmap_range(&mut self.symbolic_bitmap, offset, loop_end);
        // A concrete overwrite also clears any Multi-cell marker — the cell
        // is no longer carrying lazy alternatives. The owning
        // `SymbolicMemory::multi_objects` entries must be dropped by the
        // caller (the page does not own that map).
        clear_bitmap_range(&mut self.multi_bitmap, offset, loop_end);
    }

    /// Mark bytes as symbolic.
    pub fn mark_symbolic(&mut self, offset: u16, size: u16) {
        debug_assert!(
            (offset as usize + size as usize) <= PAGE_SIZE as usize,
            "mark_symbolic: offset {offset} + size {size} exceeds PAGE_SIZE {PAGE_SIZE}"
        );
        let bitmap = self
            .symbolic_bitmap
            .get_or_insert_with(|| Box::new([0u64; BITMAP_WORDS]));
        let loop_end = ((offset as usize) + (size as usize)).min(PAGE_SIZE as usize) as u16;
        set_bitmap_range(bitmap, offset, loop_end);
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
            (offset as usize + size as usize) <= PAGE_SIZE as usize,
            "clear_symbolic: offset {offset} + size {size} exceeds PAGE_SIZE {PAGE_SIZE}"
        );
        let loop_end = ((offset as usize) + (size as usize)).min(PAGE_SIZE as usize) as u16;
        clear_bitmap_range(&mut self.symbolic_bitmap, offset, loop_end);
    }

    /// Check if a byte is symbolic.
    #[inline]
    pub fn is_symbolic(&self, offset: u16) -> bool {
        match &self.symbolic_bitmap {
            None => false,
            Some(bitmap) => {
                let word_idx = (offset / BITMAP_BITS_PER_WORD) as usize;
                let bit_idx = offset % BITMAP_BITS_PER_WORD;
                bitmap[word_idx] & (1u64 << bit_idx) != 0
            }
        }
    }

    /// Mark a single byte as Multi (carrying lazy alternatives in
    /// `SymbolicMemory::multi_objects`). The byte stops being treated as
    /// plain Symbolic — the caller is responsible for clearing the
    /// symbolic bit and the `symbolic_objects` entry at this address.
    pub fn mark_multi(&mut self, offset: u16) {
        debug_assert!(
            (offset as usize) < PAGE_SIZE as usize,
            "mark_multi: offset {offset} out of range"
        );
        let bitmap = self
            .multi_bitmap
            .get_or_insert_with(|| Box::new([0u64; BITMAP_WORDS]));
        let word_idx = (offset / BITMAP_BITS_PER_WORD) as usize;
        let bit_idx = offset % BITMAP_BITS_PER_WORD;
        bitmap[word_idx] |= 1u64 << bit_idx;
    }

    /// Check if this page has any Multi bytes.
    ///
    /// Parallel to [`has_symbolic`](Self::has_symbolic): `merge` uses it to decide whether the
    /// page can take the concrete-equality early-out. A page carrying Multi
    /// cells must always walk its bytes because Multi divergence lives in
    /// `SymbolicMemory::multi_objects`, invisible to a `data[]` compare.
    #[inline]
    pub fn has_multi(&self) -> bool {
        self.multi_bitmap.is_some()
    }

    /// Check if a byte carries lazy Multi alternatives.
    #[inline]
    pub fn is_multi(&self, offset: u16) -> bool {
        match &self.multi_bitmap {
            None => false,
            Some(bitmap) => {
                let word_idx = (offset / BITMAP_BITS_PER_WORD) as usize;
                let bit_idx = offset % BITMAP_BITS_PER_WORD;
                bitmap[word_idx] & (1u64 << bit_idx) != 0
            }
        }
    }

    /// Clear the Multi marker on a single byte. The caller is responsible
    /// for removing the corresponding entry from
    /// `SymbolicMemory::multi_objects`. Drops the page-level bitmap when
    /// it becomes empty.
    pub fn clear_multi(&mut self, offset: u16) {
        clear_bitmap_range(&mut self.multi_bitmap, offset, offset + 1);
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
            symbolic_bitmap: page.symbolic_bitmap.map(|b| b.to_vec()),
            multi_bitmap: page.multi_bitmap.map(|b| b.to_vec()),
        }
    }
}

impl From<MemoryPageData> for MemoryPage {
    fn from(data: MemoryPageData) -> Self {
        let mut page_data = vec![0u8; PAGE_SIZE as usize];
        let copy_len = data.data.len().min(PAGE_SIZE as usize);
        page_data[..copy_len].copy_from_slice(&data.data[..copy_len]);

        let to_bitmap = |v: Vec<u64>| -> Option<Box<[u64; BITMAP_WORDS]>> {
            if v.len() != BITMAP_WORDS {
                return None;
            }
            let mut arr = [0u64; BITMAP_WORDS];
            arr.copy_from_slice(&v);
            Some(Box::new(arr))
        };

        MemoryPage {
            data: Arc::new(page_data),
            permissions: data.permissions,
            base_addr: data.base_addr,
            symbolic_bitmap: data.symbolic_bitmap.and_then(to_bitmap),
            multi_bitmap: data.multi_bitmap.and_then(to_bitmap),
        }
    }
}
#[cfg(test)]
#[path = "serde_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` in the parent overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod serde_tests;
