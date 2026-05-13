//! Memory page primitives: `Permission`, `MemoryPage`, and page-size constants.
//!
//! Extracted from the previous monolithic `memory.rs`. Re-exported from
//! `crate::memory` so existing call sites that use `crate::memory::MemoryPage`
//! / `crate::memory::Permission` / `crate::memory::PAGE_SIZE` keep working.

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

/// Memory permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Clone)]
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
}

impl MemoryPage {
    /// Create a new zeroed page.
    pub fn new(base_addr: u64, permissions: Permission) -> Self {
        MemoryPage {
            data: Arc::new(vec![0u8; PAGE_SIZE as usize]),
            permissions,
            base_addr,
            symbolic_bitmap: None,
        }
    }

    /// Create a page from existing data.
    pub fn from_data(base_addr: u64, data: Vec<u8>, permissions: Permission) -> Self {
        let mut page_data = vec![0u8; PAGE_SIZE as usize];
        let copy_len = data.len().min(PAGE_SIZE as usize);
        page_data[..copy_len].copy_from_slice(&data[..copy_len]);

        MemoryPage {
            data: Arc::new(page_data),
            permissions,
            base_addr,
            symbolic_bitmap: None,
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

        // Clear symbolic bitmap bits for overwritten bytes
        if let Some(ref mut bitmap) = self.symbolic_bitmap {
            let loop_end = (offset as usize + bytes.len()).min(PAGE_SIZE as usize) as u16;
            for i in offset..loop_end {
                let word_idx = (i / BITMAP_BITS_PER_WORD) as usize;
                let bit_idx = i % BITMAP_BITS_PER_WORD;
                bitmap[word_idx] &= !(1u64 << bit_idx);
            }
            // If bitmap is now empty, drop it
            if bitmap.iter().all(|&w| w == 0) {
                self.symbolic_bitmap = None;
            }
        }
    }

    /// Mark bytes as symbolic.
    pub fn mark_symbolic(&mut self, offset: u16, size: u16) {
        debug_assert!(
            (offset as usize + size as usize) <= PAGE_SIZE as usize,
            "mark_symbolic: offset {} + size {} exceeds PAGE_SIZE {}",
            offset,
            size,
            PAGE_SIZE
        );
        let bitmap = self
            .symbolic_bitmap
            .get_or_insert_with(|| Box::new([0u64; BITMAP_WORDS]));
        let loop_end = ((offset as usize) + (size as usize)).min(PAGE_SIZE as usize) as u16;
        for i in offset..loop_end {
            let word_idx = (i / BITMAP_BITS_PER_WORD) as usize;
            let bit_idx = i % BITMAP_BITS_PER_WORD;
            bitmap[word_idx] |= 1u64 << bit_idx;
        }
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

    /// Fork this page (O(1) for concrete pages, bitmap clone for symbolic).
    pub fn fork(&self) -> Self {
        MemoryPage {
            data: Arc::clone(&self.data),
            permissions: self.permissions,
            base_addr: self.base_addr,
            symbolic_bitmap: self.symbolic_bitmap.clone(),
        }
    }
}
