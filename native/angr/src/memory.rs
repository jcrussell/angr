//! Symbolic memory system for the VEX execution engine.
//!
//! This module provides a paged memory model with:
//! - O(1) forking via copy-on-write (using Grudge's RustPage)
//! - Mixed concrete/symbolic value storage
//! - Efficient symbolic address handling

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use im::OrdMap;

use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::Endness;

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

/// Page size in bytes (4KB).
pub const PAGE_SIZE: u64 = 4096;

/// Page mask for address calculation.
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Errors from memory operations.
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
}

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
    /// 64 u64s = 4096 bits = one bit per byte in a 4KB page.
    /// Boxed to keep MemoryPage small when not needed (None = fully concrete).
    symbolic_bitmap: Option<Box<[u64; 64]>>,
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
            if word == 0 { continue; }
            let base = (word_idx as u16) * 64;
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
            offset, bytes.len(), PAGE_SIZE
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
                let word_idx = (i / 64) as usize;
                let bit_idx = i % 64;
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
            offset, size, PAGE_SIZE
        );
        let bitmap = self.symbolic_bitmap.get_or_insert_with(|| Box::new([0u64; 64]));
        let loop_end = ((offset as usize) + (size as usize)).min(PAGE_SIZE as usize) as u16;
        for i in offset..loop_end {
            let word_idx = (i / 64) as usize;
            let bit_idx = i % 64;
            bitmap[word_idx] |= 1u64 << bit_idx;
        }
    }

    /// Check if a byte is symbolic.
    #[inline]
    pub fn is_symbolic(&self, offset: u16) -> bool {
        match &self.symbolic_bitmap {
            None => false,
            Some(bitmap) => {
                let word_idx = (offset / 64) as usize;
                let bit_idx = offset % 64;
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
    symbolic_objects: HashMap<u64, RustBV>,
    /// Next symbolic object ID.
    next_sym_id: u64,
    /// Default permissions for new pages.
    default_permissions: Permission,
    /// Endianness for this memory.
    endness: Endness,
    /// Pages that have been modified since last clear.
    /// Stores page numbers (addr >> 12) for efficient tracking.
    dirty_pages: HashSet<u64>,
    /// Lazy regions: page ranges that CAN have pages fetched on-demand.
    /// Stores (start_page_num, end_page_num) pairs.
    /// When a load hits an unmapped page in a lazy region, the interpreter
    /// should fetch it from Python rather than failing.
    lazy_regions: Vec<(u64, u64)>,
    /// Reverse index for symbolic objects: maps each byte offset within a
    /// symbolic object to (base_addr, width_bits). Enables O(1) lookup when
    /// loading a byte that falls inside a wider symbolic object.
    symbolic_spans: HashMap<u64, (u64, u32)>,
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
    imported_addrs: HashSet<u64>,
    /// If true, enforce per-page R/W permissions on load and store. Mirrors
    /// angr's STRICT_PAGE_ACCESS option. Default is false to keep existing
    /// callers (which often map all memory as RWX or rely on Python perms)
    /// working unchanged.
    enforce_permissions: bool,
}

impl PendingWrite {
    /// Check if a concrete address could possibly overlap with this pending write.
    pub fn could_overlap_page(&self, addr: u64) -> bool {
        match self.page_hint {
            Some((min_page, max_page)) => {
                let page = addr >> 12;
                page >= min_page && page <= max_page
            }
            None => true, // Unknown range, must assume overlap
        }
    }
}

impl SymbolicMemory {
    /// Create a new empty memory.
    pub fn new(endness: Endness) -> Self {
        SymbolicMemory {
            pages: OrdMap::new(),
            symbolic_objects: HashMap::new(),
            next_sym_id: 0,
            default_permissions: Permission::RWX,
            endness,
            dirty_pages: HashSet::new(),
            lazy_regions: Vec::new(),
            symbolic_spans: HashMap::new(),
            pending_writes: Vec::new(),
            zero_fill_unconstrained: false,
            imported_addrs: HashSet::new(),
            enforce_permissions: false,
        }
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

    /// Check that the page containing `addr` carries execute permission.
    /// No-op if `enforce_permissions` is false. If the page is unmapped we
    /// return Ok so the caller can fall back to its existing lift paths
    /// (native libpyvex region / Python lift_block callback) — only mapped
    /// pages without the X bit produce a `Permission` error here.
    pub fn check_executable(&self, addr: u64) -> Result<(), MemoryError> {
        if !self.enforce_permissions {
            return Ok(());
        }
        let page_num = addr >> 12;
        if let Some(page) = self.pages.get(&page_num) {
            let actual = page.permissions();
            if !actual.allows(Permission::X) {
                return Err(MemoryError::Permission {
                    addr,
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
    pub fn map(&mut self, addr: u64, size: u64, permissions: Permission) {
        let start_page = addr >> 12;
        let end_page = (addr + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            let base = page_num << 12;
            if !self.pages.contains_key(&page_num) {
                self.pages.insert(page_num, MemoryPage::new(base, permissions));
            }
        }
    }

    /// Map a region and initialize with data.
    pub fn map_data(&mut self, addr: u64, data: &[u8], permissions: Permission) {
        let _start_page = addr >> 12;
        let _offset_in_page = addr & PAGE_MASK;

        let mut remaining = data;
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr >> 12;
            let page_offset = (current_addr & PAGE_MASK) as usize;
            let bytes_in_page = (PAGE_SIZE as usize - page_offset).min(remaining.len());

            // Get or create page, modify in place (COW handled by Arc::make_mut in store_concrete)
            let page = self.pages.entry(page_num).or_insert_with(|| {
                MemoryPage::new(page_num << 12, permissions)
            });
            page.store_concrete(page_offset as u16, &remaining[..bytes_in_page]);

            remaining = &remaining[bytes_in_page..];
            current_addr += bytes_in_page as u64;
        }
    }

    /// Unmap a memory region.
    pub fn unmap(&mut self, addr: u64, size: u64) {
        let start_page = addr >> 12;
        let end_page = (addr + size + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            self.pages.remove(&page_num);
        }
    }

    /// Check if an address is mapped.
    pub fn is_mapped(&self, addr: u64) -> bool {
        let page_num = addr >> 12;
        self.pages.contains_key(&page_num)
    }

    /// Load bytes from memory as a RustBV.
    pub fn load(
        &self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // For symbolic addresses, we need to concretize or fork
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => {
                // Try to evaluate the address
                match ctx.eval(&addr) {
                    Some(a) => a as u64,
                    None => {
                        return Err(MemoryError::SymbolicAddress {
                            description: "could not resolve address".to_string(),
                        });
                    }
                }
            }
        };

        self.load_concrete(concrete_addr, size, ctx)
    }

    /// Load from a concrete address.
    pub fn load_concrete(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Check for stored symbolic object at exact address first
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            if sym.width() == size * 8 {
                return Ok(sym.clone());
            }
            // Partial read from a wider symbolic object
            // e.g., reading 1 byte from a 128-byte BVS
            if sym.width() > size * 8 {
                let total_bits = sym.width();
                // Big-endian extraction: byte 0 is the MSB (highest bits)
                // This matches angr's Python convention for memory.store with Iend_BE
                let hi = total_bits - 1;
                let lo = total_bits - size * 8;
                return Ok(sym.extract(hi, lo, ctx));
            }
        }
        // Check if this address falls WITHIN a wider symbolic object
        // stored at a lower address (e.g., reading byte 5 of a 128-byte BVS)
        // Uses the symbolic_spans reverse index for O(1) lookup.
        if let Some(&(base_addr, _width_bits)) = self.symbolic_spans.get(&addr) {
            if let Some(sym) = self.symbolic_objects.get(&base_addr) {
                let base_offset = addr - base_addr;
                let sym_bytes = sym.width() / 8;
                if base_offset < sym_bytes as u64
                    && base_offset + size as u64 <= sym_bytes as u64
                {
                    let total_bits = sym.width();
                    let hi = total_bits - (base_offset as u32 * 8) - 1;
                    let lo = hi + 1 - size * 8;
                    return Ok(sym.extract(hi, lo, ctx));
                }
            }
        }

        let start_page = addr >> 12;
        let end_page = (addr + size as u64 - 1) >> 12;

        self.check_perms_range(start_page, end_page, Permission::R)?;

        let mut bytes;
        let mut has_symbolic = false;

        if start_page == end_page {
            // Fast path: entire load within a single page (common case)
            let page = self.pages.get(&start_page).ok_or(MemoryError::Unmapped {
                addr: start_page << 12,
                size: PAGE_SIZE,
            })?;
            let offset = (addr & PAGE_MASK) as u16;
            bytes = page.load_concrete(offset, size as u16);
            // Check symbolic markers
            for i in 0..size as u16 {
                if page.is_symbolic(offset + i) {
                    has_symbolic = true;
                    break;
                }
            }
        } else {
            // Slow path: load spans multiple pages
            bytes = Vec::with_capacity(size as usize);
            for i in 0..size {
                let byte_addr = addr + i as u64;
                let page_num = byte_addr >> 12;
                let offset = (byte_addr & PAGE_MASK) as u16;
                if let Some(page) = self.pages.get(&page_num) {
                    if page.is_symbolic(offset) {
                        has_symbolic = true;
                    }
                    let byte = page.load_concrete(offset, 1);
                    bytes.push(byte.get(0).copied().unwrap_or(0));
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: byte_addr,
                        size: 1,
                    });
                }
            }
        }

        if has_symbolic {
            // Return stored symbolic object if available at exact address+width
            if let Some(sym) = self.symbolic_objects.get(&addr) {
                if sym.width() == size * 8 {
                    return Ok(sym.clone());
                }
            }
            // Try to reconstruct from per-byte symbolic objects
            // by concatenating individual byte-level entries
            let mut parts: Vec<RustBV> = Vec::new();
            let mut all_found = true;
            for i in 0..size {
                let byte_addr = addr + i as u64;
                if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                    if sym.width() == 8 {
                        parts.push(sym.clone());
                    } else if sym.width() > 8 {
                        // Extract the relevant byte
                        parts.push(sym.extract(7, 0, ctx));
                    } else {
                        all_found = false;
                        break;
                    }
                } else {
                    all_found = false;
                    break;
                }
            }
            if all_found && !parts.is_empty() {
                // Concatenate bytes: first byte is at lowest address
                // For little-endian: byte 0 is LSB, byte N is MSB
                // Concat: MSB .. LSB  →  parts[N-1] .. parts[0]
                let mut result = parts[parts.len() - 1].clone();
                for i in (0..parts.len() - 1).rev() {
                    result = result.concat(&parts[i], ctx);
                }
                return Ok(result);
            }
            // Check for wider symbolic objects that contain our range
            for (&sym_addr, sym_val) in &self.symbolic_objects {
                let sym_size = sym_val.width() / 8;
                if sym_addr <= addr && addr + size as u64 <= sym_addr + sym_size as u64 {
                    let offset = (addr - sym_addr) as u32;
                    let high = (offset + size) * 8 - 1;
                    let low = offset * 8;
                    return Ok(sym_val.extract(high, low, ctx));
                }
            }
            // Cannot reconstruct - return error for Python fallback
            return Err(MemoryError::SymbolicAddress {
                description: "symbolic bytes not fully tracked".to_string(),
            });
        }

        // Convert bytes to value based on endianness
        let value = match self.endness {
            Endness::Little => {
                let mut v: u128 = 0;
                for (i, &byte) in bytes.iter().enumerate() {
                    v |= (byte as u128) << (i * 8);
                }
                v
            }
            Endness::Big => {
                let mut v: u128 = 0;
                for &byte in &bytes {
                    v = (v << 8) | (byte as u128);
                }
                v
            }
        };

        Ok(RustBV::concrete(value, size * 8))
    }

    /// Store a value to memory.
    pub fn store(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        // For symbolic addresses, we need to concretize
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => {
                match ctx.eval(&addr) {
                    Some(a) => a as u64,
                    None => {
                        return Err(MemoryError::SymbolicAddress {
                            description: "could not resolve address for store".to_string(),
                        });
                    }
                }
            }
        };

        self.store_concrete(concrete_addr, value)
    }

    /// Store to a concrete address.
    pub fn store_concrete(
        &mut self,
        addr: u64,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        // Check if pages are mapped (fast path for same-page stores)
        let start_page = addr >> 12;
        let end_page = (addr + size as u64 - 1) >> 12;

        if start_page == end_page {
            if !self.pages.contains_key(&start_page) {
                return Err(MemoryError::Unmapped {
                    addr: start_page << 12,
                    size: PAGE_SIZE,
                });
            }
        } else {
            for page_num in start_page..=end_page {
                if !self.pages.contains_key(&page_num) {
                    return Err(MemoryError::Unmapped {
                        addr: page_num << 12,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        self.check_perms_range(start_page, end_page, Permission::W)?;

        // If symbolic, store in symbolic_objects
        if value.is_symbolic() {
            self.symbolic_objects.insert(addr, value.clone());
            // Update reverse span index: map each byte offset to (base_addr, width)
            let width_bits = value.width();
            let sym_bytes = width_bits / 8;
            for i in 1..sym_bytes {
                self.symbolic_spans.insert(addr + i as u64, (addr, width_bits));
            }
            // Mark pages as having symbolic bytes — batch per-page
            let mut current_page_num = u64::MAX;
            let mut current_page: Option<MemoryPage> = None;
            for i in 0..size {
                let byte_addr = addr + i as u64;
                let page_num = byte_addr >> 12;
                let offset = (byte_addr & PAGE_MASK) as u16;
                if page_num != current_page_num {
                    // Flush previous page
                    if let Some(p) = current_page.take() {
                        self.pages.insert(current_page_num, p);
                        self.dirty_pages.insert(current_page_num);
                    }
                    current_page_num = page_num;
                    current_page = self.pages.get(&page_num).cloned();
                }
                if let Some(ref mut p) = current_page {
                    p.mark_symbolic(offset, 1);
                }
            }
            if let Some(p) = current_page {
                self.pages.insert(current_page_num, p);
                self.dirty_pages.insert(current_page_num);
            }
            return Ok(());
        }

        // Concrete store
        let concrete_val = value.to_u128();

        // Convert to bytes based on endianness
        let bytes: Vec<u8> = match self.endness {
            Endness::Little => (0..size).map(|i| (concrete_val >> (i * 8)) as u8).collect(),
            Endness::Big => (0..size)
                .rev()
                .map(|i| (concrete_val >> (i * 8)) as u8)
                .collect(),
        };

        // Write to pages
        let mut remaining = &bytes[..];
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr >> 12;
            let page_offset = (current_addr & PAGE_MASK) as u16;
            let bytes_in_page = ((PAGE_SIZE - page_offset as u64) as usize).min(remaining.len());

            if let Some(page) = self.pages.get_mut(&page_num) {
                page.store_concrete(page_offset, &remaining[..bytes_in_page]);
                // Mark page as dirty
                self.dirty_pages.insert(page_num);
            }

            remaining = &remaining[bytes_in_page..];
            current_addr += bytes_in_page as u64;
        }

        // Clear any symbolic object at this address and its span entries
        if let Some(old_sym) = self.symbolic_objects.remove(&addr) {
            let old_bytes = old_sym.width() / 8;
            for i in 1..old_bytes {
                self.symbolic_spans.remove(&(addr + i as u64));
            }
        }

        Ok(())
    }

    /// Load from a symbolic address with concretization support.
    ///
    /// This method handles symbolic addresses by:
    /// 1. Trying to concretize the address to a single value (fast path)
    /// 2. Building a balanced ITE tree for strided access patterns (efficient)
    /// 3. Building an ITE chain for multiple possible addresses
    /// 4. Returning an error if the address range is too large
    ///
    /// For unmapped pages in lazy regions, returns `UnmappedPageInRegion` so
    /// the interpreter can fetch the page on-demand.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer configuration
    ///
    /// # Returns
    /// The loaded value as a RustBV, or a MemoryError if loading fails.
    pub fn load_symbolic(
        &self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<RustBV, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.load_concrete_lazy(concrete_addr, size, ctx);
        }

        // Try to concretize the address (read mode: falls back to Any single solution)
        let base_value = match concretizer.concretize_read(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_lazy(concrete_addr, size, ctx)?
            }
            ConcretizationResult::Strided { base, stride, count } => {
                self.load_strided_balanced(&addr, base, stride, count, size, ctx)?
            }
            ConcretizationResult::Multiple(addrs) => {
                self.build_balanced_ite_load(&addr, &addrs, size, ctx)?
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                return Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                });
            }
            ConcretizationResult::Failed(reason) => {
                return Err(MemoryError::SymbolicAddress {
                    description: reason,
                });
            }
        };

        // Apply any pending writes that might overlap this symbolic load
        Ok(self.apply_pending_writes_symbolic(&addr, size, base_value, ctx))
    }

    /// Load from strided addresses using a balanced ITE tree.
    ///
    /// For a strided pattern like base, base+stride, base+2*stride, ...,
    /// this builds a balanced binary tree of ITE expressions with O(log N) depth
    /// instead of the linear O(N) depth of a chain.
    ///
    /// The tree structure:
    /// ```text
    ///                      ITE(addr <= mid_addr)
    ///                     /                    \
    ///        ITE(addr <= lo_mid)        ITE(addr <= hi_mid)
    ///           /      \                    /       \
    ///         ...     ...                ...        ...
    /// ```
    fn load_strided_balanced(
        &self,
        addr_expr: &RustBV,
        base: u64,
        stride: u64,
        count: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if count == 0 {
            return Err(MemoryError::SymbolicAddress {
                description: "strided access with zero count".to_string(),
            });
        }
        if count == 1 {
            return self.load_concrete_lazy(base, size, ctx);
        }

        // Build the balanced tree recursively
        self.build_strided_ite_tree(addr_expr, base, stride, 0, count - 1, size, ctx)
    }

    /// Recursive helper to build a balanced ITE tree for strided access.
    ///
    /// # Arguments
    /// * `addr_expr` - The symbolic address expression
    /// * `base` - Base address of the strided pattern
    /// * `stride` - Stride between consecutive addresses
    /// * `lo` - Lowest index in the current subtree
    /// * `hi` - Highest index in the current subtree
    /// * `size` - Number of bytes to load
    /// * `ctx` - Solver context
    fn build_strided_ite_tree(
        &self,
        addr_expr: &RustBV,
        base: u64,
        stride: u64,
        lo: u64,
        hi: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Base case: single element
        if lo == hi {
            let addr = base + lo * stride;
            return self.load_concrete_lazy(addr, size, ctx);
        }

        // Split at midpoint for balanced tree
        let mid = (lo + hi) / 2;
        let mid_addr = base + mid * stride;

        // Build condition: addr <= mid_addr
        let mid_const = RustBV::concrete(mid_addr as u128, addr_expr.width());
        let cond = addr_expr.ule(&mid_const, ctx);

        // Recursively build left subtree (lo..mid) and right subtree (mid+1..hi)
        let left = self.build_strided_ite_tree(addr_expr, base, stride, lo, mid, size, ctx)?;
        let right = self.build_strided_ite_tree(addr_expr, base, stride, mid + 1, hi, size, ctx)?;

        // Build ITE: if (addr <= mid_addr) then left else right
        Ok(cond.ite(&left, &right, ctx))
    }

    /// Build a balanced ITE tree for arbitrary addresses.
    ///
    /// This is similar to the strided version but works with any sorted
    /// list of addresses. Uses binary search style partitioning.
    fn build_balanced_ite_load(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if addrs.is_empty() {
            return Err(MemoryError::SymbolicAddress {
                description: "empty address list".to_string(),
            });
        }
        if addrs.len() == 1 {
            return self.load_concrete_lazy(addrs[0], size, ctx);
        }

        self.build_balanced_ite_load_inner(addr_expr, addrs, size, ctx)
    }

    /// Recursive helper for balanced ITE tree with arbitrary addresses.
    fn build_balanced_ite_load_inner(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Base case: single address
        if addrs.len() == 1 {
            return self.load_concrete_lazy(addrs[0], size, ctx);
        }

        // Base case: two addresses - simple ITE
        if addrs.len() == 2 {
            let left_val = self.load_concrete_lazy(addrs[0], size, ctx)?;
            let right_val = self.load_concrete_lazy(addrs[1], size, ctx)?;

            let left_const = RustBV::concrete(addrs[0] as u128, addr_expr.width());
            let cond = addr_expr.eq(&left_const, ctx);

            return Ok(cond.ite(&left_val, &right_val, ctx));
        }

        // Split at midpoint
        let mid = addrs.len() / 2;
        let mid_addr = addrs[mid];

        // Build condition: addr < mid_addr (for binary partition)
        let mid_const = RustBV::concrete(mid_addr as u128, addr_expr.width());
        let cond = addr_expr.ult(&mid_const, ctx);

        // Recursively build left (addrs < mid) and right (addrs >= mid) subtrees
        let left = self.build_balanced_ite_load_inner(addr_expr, &addrs[..mid], size, ctx)?;
        let right = self.build_balanced_ite_load_inner(addr_expr, &addrs[mid..], size, ctx)?;

        // Build ITE: if (addr < mid_addr) then left else right
        Ok(cond.ite(&left, &right, ctx))
    }

    /// Store to a symbolic address with concretization support.
    ///
    /// This method handles symbolic addresses by:
    /// 1. Trying to concretize the address to a single value (fast path)
    /// 2. Performing conditional stores for strided access patterns
    /// 3. Performing conditional stores for multiple possible addresses
    /// 4. Returning an error if the address range is too large
    ///
    /// For multiple addresses, each candidate gets a conditional store:
    /// `mem[candidate] = If(addr == candidate, new_value, mem[candidate])`
    ///
    /// For unmapped pages in lazy regions, returns `UnmappedPageInRegion` so
    /// the interpreter can fetch the page on-demand.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to store to
    /// * `value` - The value to store
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer configuration
    ///
    /// # Returns
    /// Ok(()) on success, or a MemoryError if storing fails.
    pub fn store_symbolic(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<(), MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.store_concrete_lazy(concrete_addr, value);
        }

        // Try to concretize the address (write mode: falls back to Max solution)
        match concretizer.concretize_write(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(concrete_addr, value)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                self.store_strided(&addr, &value, base, stride, count, ctx)
            }
            ConcretizationResult::Multiple(addrs) => {
                let size = value.width() / 8;
                for &candidate in &addrs {
                    let addr_const = RustBV::concrete(candidate as u128, addr.width());
                    let cond = addr.eq(&addr_const, ctx);
                    let current = self.load_concrete_lazy(candidate, size, ctx)?;
                    let conditional_value = cond.ite(&value, &current, ctx);
                    self.store_concrete_lazy(candidate, conditional_value)?;
                }
                Ok(())
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                })
            }
            ConcretizationResult::Failed(reason) => {
                Err(MemoryError::SymbolicAddress {
                    description: reason,
                })
            }
        }
    }

    /// Store to strided addresses with conditional stores.
    ///
    /// For each address in the strided pattern, performs:
    /// `mem[addr] = If(symbolic_addr == addr, new_value, mem[addr])`
    fn store_strided(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        base: u64,
        stride: u64,
        count: u64,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        for i in 0..count {
            let candidate = base + i * stride;

            // Build condition: addr == candidate
            let addr_const = RustBV::concrete(candidate as u128, addr_expr.width());
            let cond = addr_expr.eq(&addr_const, ctx);

            // Load current value at candidate address
            let current = self.load_concrete_lazy(candidate, size, ctx)?;

            // Build conditional value
            let conditional_value = cond.ite(value, &current, ctx);

            // Store the conditional value
            self.store_concrete_lazy(candidate, conditional_value)?;
        }

        Ok(())
    }

    // ==================== UNIFIED SYMBOLIC MEMORY OPERATIONS ====================
    // These methods handle all symbolic memory operations entirely in Rust,
    // eliminating the need for Python callbacks that were previously broken.

    /// Prepare addresses for ITE construction by auto-mapping unmapped pages.
    ///
    /// For each candidate address, if its page is unmapped but in a lazy region,
    /// auto-map it as a zero page. Returns the list of addresses that are ready
    /// for ITE construction (i.e., their pages are mapped).
    ///
    /// # Arguments
    /// * `addrs` - List of candidate addresses
    /// * `size` - Size of the access in bytes
    ///
    /// # Returns
    /// List of addresses whose pages are mapped. Unmapped addresses are skipped
    /// to allow fallback to Python callback which has access to actual backer data.
    ///
    /// # Note
    /// This function no longer auto-maps zero pages. When addresses are in lazy
    /// regions but unmapped, they are skipped. Callers should check if the result
    /// is incomplete and fall back to Python if needed.
    pub fn prepare_addresses_for_ite(&mut self, addrs: &[u64], _size: u32) -> Vec<u64> {
        let mut ready_addrs = Vec::with_capacity(addrs.len());

        for &addr in addrs {
            let page_num = addr >> 12;

            // Check if page is already mapped
            if self.pages.contains_key(&page_num) {
                ready_addrs.push(addr);
                continue;
            }

            // Page not mapped - skip this address
            // The caller should fall back to Python callback which can provide
            // actual backer data instead of speculative zeros
            //
            // NOTE: We intentionally do NOT auto-map zero pages here. Python may
            // have actual data for this page from backers (file contents, initialized
            // data). Speculatively creating zero pages causes state divergence.
        }

        ready_addrs
    }

    /// Load from a concrete address, returning an unconstrained symbolic value if unmapped.
    ///
    /// This is used as a fallback in ITE construction when an address cannot be mapped.
    /// Instead of failing, we return a fresh symbolic value representing unknown memory.
    ///
    /// # Arguments
    /// * `addr` - The address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    /// * `counter` - A counter for generating unique symbolic names
    ///
    /// # Returns
    /// The loaded value, or a fresh unconstrained symbolic value if unmapped.
    pub fn load_concrete_or_unconstrained(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
        counter: &mut u64,
    ) -> RustBV {
        match self.load_concrete_lazy(addr, size, ctx) {
            Ok(value) => value,
            Err(_) => {
                if self.zero_fill_unconstrained {
                    RustBV::concrete(0, size * 8)
                } else {
                    // Generate a unique name for the unconstrained memory read
                    *counter += 1;
                    RustBV::symbolic(ctx, format!("unc_mem_{:x}_{}", addr, counter), size * 8)
                }
            }
        }
    }

    /// Unified symbolic load that handles all concretization results in Rust.
    ///
    /// This method replaces the Python fallback for symbolic memory loads.
    /// It handles all cases:
    /// - Single address: direct load
    /// - Multiple addresses: build balanced ITE tree with auto-mapping
    /// - Strided access: build balanced ITE tree
    /// - Too large range: return unconstrained symbolic value
    /// - Failed concretization: return error
    ///
    /// The key improvement is that unmapped pages in lazy regions are auto-mapped
    /// before ITE construction, and truly unmapped addresses use unconstrained
    /// symbolic values instead of failing.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer
    ///
    /// # Returns
    /// The loaded value as a RustBV, or an error if loading fails.
    pub fn load_symbolic_unified(
        &mut self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<RustBV, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.load_concrete_automap(concrete_addr, size, ctx);
        }

        // Try to concretize the address (read mode: falls back to Any single solution)
        let base_value = match concretizer.concretize_read(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_automap(concrete_addr, size, ctx)?
            }
            ConcretizationResult::Multiple(addrs) => {
                let ready_addrs = self.prepare_addresses_for_ite(&addrs, size);

                if ready_addrs.is_empty() {
                    return Ok(RustBV::symbolic(
                        ctx,
                        format!("mem_all_unmapped_{}", size),
                        size * 8,
                    ));
                }

                let addr_clone = addr.clone();
                self.build_balanced_ite_load_after_prep(&addr_clone, &ready_addrs, size, ctx)?
            }
            ConcretizationResult::Strided { base, stride, count } => {
                self.prepare_strided_region(base, stride, count, size);
                self.load_strided_balanced(&addr, base, stride, count, size, ctx)?
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Return error so caller can fall back to Python's memory model,
                // which handles large symbolic address ranges natively.
                return Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                });
            }
            ConcretizationResult::Failed(reason) => {
                return Err(MemoryError::SymbolicAddress { description: reason });
            }
        };

        // Apply any pending writes that might overlap this symbolic load
        Ok(self.apply_pending_writes_symbolic(&addr, size, base_value, ctx))
    }

    /// Build a balanced ITE tree after addresses have been prepared.
    ///
    /// This is the immutable part of the unified load, called after prepare_addresses_for_ite.
    fn build_balanced_ite_load_after_prep(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if addrs.is_empty() {
            return Ok(RustBV::symbolic(
                ctx,
                format!("mem_empty_ite_{}", size),
                size * 8,
            ));
        }
        if addrs.len() == 1 {
            let mut counter = 0u64;
            return Ok(self.load_concrete_or_unconstrained(addrs[0], size, ctx, &mut counter));
        }

        let mut counter = 0u64;
        self.build_ite_tree_inner(addr_expr, addrs, size, ctx, &mut counter)
    }

    /// Recursive helper for building ITE tree (immutable borrow).
    fn build_ite_tree_inner(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
        counter: &mut u64,
    ) -> Result<RustBV, MemoryError> {
        if addrs.len() == 1 {
            return Ok(self.load_concrete_or_unconstrained(addrs[0], size, ctx, counter));
        }

        if addrs.len() == 2 {
            let left_val = self.load_concrete_or_unconstrained(addrs[0], size, ctx, counter);
            let right_val = self.load_concrete_or_unconstrained(addrs[1], size, ctx, counter);
            let left_const = RustBV::concrete(addrs[0] as u128, addr_expr.width());
            let cond = addr_expr.eq(&left_const, ctx);
            return Ok(cond.ite(&left_val, &right_val, ctx));
        }

        let mid = addrs.len() / 2;
        let mid_addr = addrs[mid];
        let mid_const = RustBV::concrete(mid_addr as u128, addr_expr.width());
        let cond = addr_expr.ult(&mid_const, ctx);

        let left = self.build_ite_tree_inner(addr_expr, &addrs[..mid], size, ctx, counter)?;
        let right = self.build_ite_tree_inner(addr_expr, &addrs[mid..], size, ctx, counter)?;

        Ok(cond.ite(&left, &right, ctx))
    }

    /// Prepare a strided memory region (no-op - kept for API compatibility).
    ///
    /// # Note
    /// This function previously auto-mapped zero pages for unmapped addresses,
    /// but that caused state divergence with Python's actual backer data.
    /// Now it does nothing - unmapped pages will trigger Python fallback.
    ///
    /// # Arguments
    /// * `base` - Base address of the strided pattern
    /// * `stride` - Stride between consecutive addresses
    /// * `count` - Number of addresses in the pattern
    /// * `size` - Size of each access in bytes
    fn prepare_strided_region(&mut self, _base: u64, _stride: u64, _count: u64, _size: u32) {
        // No longer auto-maps zero pages.
        // The interpreter will fall back to Python callback which can provide
        // actual backer data instead of speculative zeros.
        //
        // NOTE: Strided loads/stores may fail and trigger Python fallback.
        // This is intentional - Python has the correct memory state.
    }

    /// Unified symbolic store that handles all concretization results in Rust.
    ///
    /// This method replaces the Python fallback for symbolic memory stores.
    /// It handles all cases by performing conditional stores for each candidate address:
    /// `mem[candidate] = If(addr == candidate, new_value, mem[candidate])`
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to store to
    /// * `value` - The value to store
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer
    ///
    /// # Returns
    /// Ok(()) on success, or an error if storing fails.
    pub fn store_symbolic_unified(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<Option<ConcretizationResult>, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            self.store_concrete_automap(concrete_addr, value)?;
            return Ok(Some(ConcretizationResult::Single(concrete_addr)));
        }

        // Try to concretize the address (write mode: falls back to Max solution)
        let result = concretizer.concretize_write(&addr, ctx);
        match &result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(*concrete_addr, value)?;
                Ok(Some(result))
            }
            ConcretizationResult::Multiple(addrs) => {
                // Prepare addresses by auto-mapping
                let ready_addrs = self.prepare_addresses_for_ite(addrs, value.width() / 8);

                // Perform conditional stores for each ready address
                self.store_conditional_multiple(&addr, &value, &ready_addrs, ctx)?;
                Ok(Some(result))
            }
            ConcretizationResult::Strided { base, stride, count } => {
                let (base, stride, count) = (*base, *stride, *count);
                // Prepare strided region
                self.prepare_strided_region(base, stride, count, value.width() / 8);
                // Use existing strided store
                self.store_strided(&addr, &value, base, stride, count, ctx)?;
                Ok(Some(result))
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Return error so caller can fall back to Python's memory model,
                // which handles large symbolic address ranges natively.
                Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                })
            }
            ConcretizationResult::Failed(reason) => {
                Err(MemoryError::SymbolicAddress { description: reason.clone() })
            }
        }
    }

    /// Store using a pre-computed concretization result.
    /// Single addresses store concretely. All other symbolic results are deferred
    /// to pending_writes for lazy materialization on load.
    pub fn store_with_concretization(
        &mut self,
        addr: &RustBV,
        value: RustBV,
        conc_result: &ConcretizationResult,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        match conc_result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(*concrete_addr, value)
            }
            ConcretizationResult::Multiple(addrs) => {
                let ready_addrs = self.prepare_addresses_for_ite(addrs, value.width() / 8);
                self.store_conditional_multiple(addr, &value, &ready_addrs, ctx)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                self.prepare_strided_region(*base, *stride, *count, value.width() / 8);
                self.store_strided(addr, &value, *base, *stride, *count, ctx)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Return error so caller can fall back to Python's memory model,
                // which handles large symbolic address ranges natively.
                Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                })
            }
            ConcretizationResult::Failed(reason) => {
                Err(MemoryError::SymbolicAddress { description: reason.clone() })
            }
        }
    }

    /// Perform conditional stores to multiple addresses.
    ///
    /// For each candidate address, performs:
    /// `mem[candidate] = If(addr == candidate, new_value, mem[candidate])`
    fn store_conditional_multiple(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        addrs: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        let mut counter = 0u64;

        for &candidate in addrs {
            // Build condition: addr == candidate
            let addr_const = RustBV::concrete(candidate as u128, addr_expr.width());
            let cond = addr_expr.eq(&addr_const, ctx);

            // Load current value (with unconstrained fallback)
            let current = self.load_concrete_or_unconstrained(candidate, size, ctx, &mut counter);

            // Build conditional value: If(addr == candidate, new_value, current)
            let conditional_value = cond.ite(value, &current, ctx);

            // Store the conditional value with auto-mapping
            self.store_concrete_automap(candidate, conditional_value)?;
        }

        Ok(())
    }

    /// Fork the memory (O(1) via CoW).
    pub fn fork(&self) -> Self {
        SymbolicMemory {
            pages: self.pages.clone(), // OrdMap clones in O(1)
            symbolic_objects: self.symbolic_objects.clone(),
            next_sym_id: self.next_sym_id,
            default_permissions: self.default_permissions,
            endness: self.endness,
            dirty_pages: HashSet::new(), // Fresh dirty tracking for fork
            lazy_regions: self.lazy_regions.clone(), // Share lazy regions
            symbolic_spans: self.symbolic_spans.clone(),
            pending_writes: self.pending_writes.clone(),
            zero_fill_unconstrained: self.zero_fill_unconstrained,
            imported_addrs: self.imported_addrs.clone(),
            enforce_permissions: self.enforce_permissions,
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

    /// Drain all pending writes (for materialization).
    pub fn drain_pending_writes(&mut self) -> Vec<PendingWrite> {
        std::mem::take(&mut self.pending_writes)
    }

    /// Flush all pending writes by materializing ITE chains into memory.
    /// This must be called before exporting state to Python to ensure
    /// memory pages contain all written values.
    pub fn flush_pending_writes(
        &mut self,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<(), MemoryError> {
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
                        let addr_const = RustBV::concrete(candidate as u128, pw.addr.width());
                        let cond = pw.addr.eq(&addr_const, ctx);
                        let effective_cond = if let Some(ref c) = pw.condition {
                            cond.and(c, ctx)
                        } else {
                            cond
                        };
                        let current = match self.load_concrete_lazy_inner(candidate, pw.size, ctx) {
                            Ok(v) => v,
                            Err(_) => RustBV::concrete(0, pw.size * 8),
                        };
                        let ite_val = effective_cond.ite(&pw.value, &current, ctx);
                        self.store_concrete_lazy(candidate, ite_val)?;
                    }
                }
                ConcretizationResult::Strided { base, stride, count } => {
                    for i in 0..count {
                        let candidate = base + i * stride;
                        let addr_const = RustBV::concrete(candidate as u128, pw.addr.width());
                        let cond = pw.addr.eq(&addr_const, ctx);
                        let effective_cond = if let Some(ref c) = pw.condition {
                            cond.and(c, ctx)
                        } else {
                            cond
                        };
                        let current = match self.load_concrete_lazy_inner(candidate, pw.size, ctx) {
                            Ok(v) => v,
                            Err(_) => RustBV::concrete(0, pw.size * 8),
                        };
                        let ite_val = effective_cond.ite(&pw.value, &current, ctx);
                        self.store_concrete_lazy(candidate, ite_val)?;
                    }
                }
                ConcretizationResult::TooLarge { .. } | ConcretizationResult::Failed(_) => {
                    // Cannot materialize — skip (data was already applied via load-time ITE)
                }
            }
        }
        Ok(())
    }

    /// Get page count.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get total mapped size in bytes.
    pub fn mapped_size(&self) -> u64 {
        self.pages.len() as u64 * PAGE_SIZE
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
    pub fn load_page_concrete(&self, page_addr: u64) -> Result<Vec<u8>, MemoryError> {
        let page_num = page_addr >> 12;
        if let Some(page) = self.pages.get(&page_num) {
            Ok(page.load_concrete(0, PAGE_SIZE as u16))
        } else {
            Err(MemoryError::Unmapped { addr: page_addr, size: PAGE_SIZE })
        }
    }

    /// Clear dirty page tracking (called after sync to Python).
    pub fn clear_dirty_pages(&mut self) {
        self.dirty_pages.clear();
    }

    /// Check if a page is dirty.
    pub fn is_page_dirty(&self, page_num: u64) -> bool {
        self.dirty_pages.contains(&page_num)
    }

    /// Get page data for syncing to Python.
    /// Returns (data, permissions) for the page, or None if not mapped.
    pub fn get_page_data(&self, page_num: u64) -> Option<(Vec<u8>, u8)> {
        self.pages.get(&page_num).map(|p| {
            (p.load_concrete(0, PAGE_SIZE as u16), p.permissions().to_bits())
        })
    }

    /// Get the pages OrdMap for iteration.
    pub fn pages(&self) -> &OrdMap<u64, MemoryPage> {
        &self.pages
    }

    // =========================================================================
    // Symbolic Memory Preservation
    // =========================================================================

    /// Get all symbolic regions for export to Python.
    ///
    /// Returns a list of (address, width, symbol_id) tuples where symbol_id
    /// is the Rust symbol ID that can be used to look up the original Python AST.
    ///
    /// This is critical for preserving symbolic identity when syncing state
    /// back to Python - without it, symbolic values would be recreated as
    /// fresh symbols, losing their relationship to constraints.
    pub fn get_symbolic_regions(&self) -> Vec<(u64, u32, Option<u64>)> {
        let mut regions = Vec::new();

        for (&addr, bv) in &self.symbolic_objects {
            let width = bv.width();
            // Try to get the symbol ID for Symbolic variants
            let sym_id = match bv {
                RustBV::Symbolic { id, .. } => Some(*id),
                RustBV::Expression { .. } => {
                    // For expressions, try to get the hash as an identifier
                    None
                }
                _ => None,
            };
            regions.push((addr, width, sym_id));
        }

        regions
    }

    /// Import a symbolic value with identity preservation.
    ///
    /// This stores a symbolic value at the given address and ensures the
    /// symbol ID is tracked for later export back to Python.
    ///
    /// # Arguments
    /// * `addr` - The address to store the value at
    /// * `value` - The symbolic value to store
    /// * `symbol_id` - Optional symbol ID for identity tracking
    pub fn import_symbolic_value(&mut self, addr: u64, value: RustBV, _symbol_id: Option<u64>) {
        // Track as Python-imported so get_state_symbolic_z3_asts can filter it out
        self.imported_addrs.insert(addr);
        // Store in symbolic_objects for lookup
        self.symbolic_objects.insert(addr, value.clone());
        // Update reverse span index
        let width_bits = value.width();
        let sym_bytes = width_bits / 8;
        for i in 1..sym_bytes {
            self.symbolic_spans.insert(addr + i as u64, (addr, width_bits));
        }

        // Mark pages as having symbolic bytes
        // Create pages if they don't exist (critical for stack addresses)
        let size = value.width() / 8;
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr >> 12;
            let offset = (byte_addr & PAGE_MASK) as u16;
            let page_addr = page_num << 12;

            // Get or create page
            let page = self.pages.entry(page_num).or_insert_with(|| {
                MemoryPage::new(page_addr, Permission::RW)
            });

            // Modify in place (COW handled by bitmap allocation in mark_symbolic)
            page.mark_symbolic(offset, 1);
        }
    }

    /// Get the symbolic object at an address if it exists.
    pub fn get_symbolic_object(&self, addr: u64) -> Option<&RustBV> {
        self.symbolic_objects.get(&addr)
    }

    /// Check if there are any symbolic objects in memory.
    pub fn has_symbolic_objects(&self) -> bool {
        !self.symbolic_objects.is_empty()
    }

    /// Get the count of symbolic objects.
    pub fn symbolic_object_count(&self) -> usize {
        self.symbolic_objects.len()
    }

    /// Check if an address was imported from Python.
    pub fn is_imported_addr(&self, addr: u64) -> bool {
        self.imported_addrs.contains(&addr)
    }

    /// Iterate over all symbolic objects in memory.
    pub fn symbolic_objects_iter(&self) -> impl Iterator<Item = (&u64, &RustBV)> {
        self.symbolic_objects.iter()
    }

    /// Clear all symbolic objects (used when resetting state).
    pub fn clear_symbolic_objects(&mut self) {
        self.symbolic_objects.clear();
        self.symbolic_spans.clear();
    }

    /// Add a lazy region where pages can be fetched on-demand.
    ///
    /// When a load hits an unmapped page within this region, the memory
    /// will return `UnmappedPageInRegion` instead of `Unmapped`, signaling
    /// to the interpreter that it should fetch the page from Python.
    ///
    /// # Arguments
    /// * `start_addr` - Start address of the region (will be page-aligned down)
    /// * `size` - Size of the region in bytes
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        let start_page = start_addr >> 12;
        let end_page = (start_addr + size + PAGE_SIZE - 1) >> 12;
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
    pub fn is_addr_in_lazy_region(&self, addr: u64) -> bool {
        self.is_in_lazy_region(addr >> 12)
    }

    /// Clear all lazy regions.
    pub fn clear_lazy_regions(&mut self) {
        self.lazy_regions.clear();
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
    pub fn get_region_prefetch_list(&self, trigger_page_addr: u64, max_pages: usize) -> Option<Vec<u64>> {
        let trigger_page_num = trigger_page_addr >> 12;

        // Find the lazy region containing this page
        let region = self.lazy_regions.iter()
            .find(|&&(start, end)| trigger_page_num >= start && trigger_page_num < end)?;

        let (region_start, region_end) = *region;

        // Collect all unmapped pages in this region
        let mut pages_to_fetch = Vec::new();

        for page_num in region_start..region_end {
            if !self.pages.contains_key(&page_num) {
                pages_to_fetch.push(page_num << 12);  // Convert to page address
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
        trigger_page_addr: u64,
        count_before: u64,
        count_after: u64,
    ) -> Vec<u64> {
        let trigger_page_num = trigger_page_addr >> 12;
        let mut pages_to_fetch = Vec::new();

        // Check pages before the trigger
        for i in 1..=count_before {
            if let Some(page_num) = trigger_page_num.checked_sub(i) {
                if self.is_in_lazy_region(page_num) && !self.pages.contains_key(&page_num) {
                    pages_to_fetch.push(page_num << 12);
                }
            }
        }

        // Add the trigger page itself if not mapped
        if !self.pages.contains_key(&trigger_page_num) {
            pages_to_fetch.push(trigger_page_addr);
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
    pub fn map_page(&mut self, page_addr: u64, data: Vec<u8>, permissions: Permission) {
        let page_num = page_addr >> 12;
        let page = MemoryPage::from_data(page_addr, data, permissions);
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
    pub fn auto_map_zero_page(&mut self, addr: u64) -> bool {
        let page_num = addr >> 12;

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

    /// Load from a concrete address with lazy region support.
    ///
    /// # Deprecation Warning
    ///
    /// This function previously auto-mapped zero pages for unmapped regions,
    /// but that behavior caused state divergence with Python's actual backer
    /// data. Now it propagates the UnmappedPageInRegion error so callers can
    /// fall back to Python callbacks to get correct data.
    ///
    /// If you need auto-mapping behavior for internal Rust operations that
    /// don't involve Python state, use `load_concrete_automap_internal`.
    pub fn load_concrete_automap(
        &mut self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let base = self.load_concrete_lazy_inner(addr, size, ctx)?;
        Ok(self.apply_pending_writes_concrete(addr, size, base, ctx))
    }

    /// Load from a concrete address with internal auto-mapping.
    ///
    /// This is for internal Rust operations that don't involve Python state.
    /// For interpreter callbacks, use `load_concrete_automap` which propagates
    /// errors so Python can provide correct backer data.
    pub fn load_concrete_automap_internal(
        &mut self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // First try normal load
        let base = match self.load_concrete_lazy_inner(addr, size, ctx) {
            Ok(v) => v,
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Auto-map the missing page
                self.auto_map_zero_page(page_addr);
                // Retry the load
                self.load_concrete_lazy_inner(addr, size, ctx)?
            }
            Err(e) => return Err(e),
        };
        Ok(self.apply_pending_writes_concrete(addr, size, base, ctx))
    }

    /// Apply pending writes that might overlap a concrete load address.
    /// Returns the value with ITE chains for any matching pending writes.
    fn apply_pending_writes_concrete(
        &self,
        _addr: u64,
        _size: u32,
        base_value: RustBV,
        _ctx: &SymContext,
    ) -> RustBV {
        // Pending writes overlay is disabled during execution.
        // Stores go through the eager concretize+ITE path.
        // Pending writes are only used for deferred flushing on export.
        base_value
    }

    /// Apply pending writes that might overlap a symbolic load address.
    /// Returns the value with ITE chains for any matching pending writes.
    fn apply_pending_writes_symbolic(
        &self,
        _addr: &RustBV,
        _size: u32,
        base_value: RustBV,
        _ctx: &SymContext,
    ) -> RustBV {
        // Pending writes overlay is disabled during execution.
        // See apply_pending_writes_concrete for rationale.
        base_value
    }

    /// Load from a concrete address, returning UnmappedPageInRegion for lazy regions.
    ///
    /// This is similar to load_concrete but distinguishes between:
    /// - Unmapped page in a lazy region (can be fetched)
    /// - Totally unmapped memory (error)
    pub fn load_concrete_lazy(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let base = self.load_concrete_lazy_inner(addr, size, ctx)?;
        Ok(self.apply_pending_writes_concrete(addr, size, base, ctx))
    }

    /// Internal implementation of load_concrete_lazy.
    fn load_concrete_lazy_inner(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Check for stored symbolic object first
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            if sym.width() == size * 8 {
                return Ok(sym.clone());
            }
        }

        let start_page = addr >> 12;
        let end_page = (addr + size as u64 - 1) >> 12;

        self.check_perms_range(start_page, end_page, Permission::R)?;

        let mut bytes;
        let mut has_symbolic = false;

        if start_page == end_page {
            // Fast path: single page
            let page = match self.pages.get(&start_page) {
                Some(p) => p,
                None => {
                    if self.is_in_lazy_region(start_page) {
                        return Err(MemoryError::UnmappedPageInRegion {
                            page_addr: start_page << 12,
                        });
                    } else {
                        return Err(MemoryError::Unmapped {
                            addr: start_page << 12,
                            size: PAGE_SIZE,
                        });
                    }
                }
            };
            let offset = (addr & PAGE_MASK) as u16;
            bytes = page.load_concrete(offset, size as u16);
            for i in 0..size as u16 {
                if page.is_symbolic(offset + i) {
                    has_symbolic = true;
                    break;
                }
            }
        } else {
            // Slow path: cross-page load
            bytes = Vec::with_capacity(size as usize);
            for i in 0..size {
                let byte_addr = addr + i as u64;
                let page_num = byte_addr >> 12;
                let offset = (byte_addr & PAGE_MASK) as u16;

                if let Some(page) = self.pages.get(&page_num) {
                    if page.is_symbolic(offset) {
                        has_symbolic = true;
                    }
                    let byte = page.load_concrete(offset, 1);
                    bytes.push(byte.get(0).copied().unwrap_or(0));
                } else {
                    if self.is_in_lazy_region(page_num) {
                        return Err(MemoryError::UnmappedPageInRegion {
                            page_addr: page_num << 12,
                        });
                    } else {
                        return Err(MemoryError::Unmapped {
                            addr: byte_addr,
                            size: 1,
                        });
                    }
                }
            }
        }

        if has_symbolic {
            // Return stored symbolic object if available and width matches
            if let Some(sym) = self.symbolic_objects.get(&addr) {
                if sym.width() == size * 8 {
                    return Ok(sym.clone());
                }
            }

            // Try to combine individual byte objects into a multi-byte value
            // This handles the case where hooks write byte-by-byte
            let mut all_bytes_have_objects = true;
            let mut byte_objects: Vec<RustBV> = Vec::with_capacity(size as usize);
            for i in 0..size {
                let byte_addr = addr + i as u64;
                if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                    if sym.width() == 8 {
                        byte_objects.push(sym.clone());
                    } else {
                        all_bytes_have_objects = false;
                        break;
                    }
                } else {
                    all_bytes_have_objects = false;
                    break;
                }
            }

            if all_bytes_have_objects && byte_objects.len() == size as usize {
                // Combine bytes into a single value using Concat
                // For little-endian, the first byte is the LSB
                match self.endness {
                    Endness::Little => {
                        // Start with the MSB (last byte) and concat towards LSB
                        let mut result = byte_objects.pop().expect("byte_objects non-empty when size > 0");
                        while let Some(byte) = byte_objects.pop() {
                            result = result.concat(&byte, ctx);
                        }
                        return Ok(result);
                    }
                    Endness::Big => {
                        // Start with the MSB (first byte) and concat towards LSB
                        let mut result = byte_objects.remove(0);
                        for byte in byte_objects {
                            result = result.concat(&byte, ctx);
                        }
                        return Ok(result);
                    }
                }
            }

            // Try to extract from a wider symbolic object that contains our range
            for (&sym_addr, sym_val) in &self.symbolic_objects {
                let sym_size = sym_val.width() / 8;
                if sym_addr <= addr && addr + size as u64 <= sym_addr + sym_size as u64 {
                    let byte_offset = (addr - sym_addr) as u32;
                    let high = (byte_offset + size) * 8 - 1;
                    let low = byte_offset * 8;
                    return Ok(sym_val.extract(high, low, ctx));
                }
            }

            // Cannot reconstruct - return error for Python fallback
            return Err(MemoryError::SymbolicAddress {
                description: "symbolic bytes not fully tracked".to_string(),
            });
        }

        // Convert bytes to value based on endianness
        let value = match self.endness {
            Endness::Little => {
                let mut v: u128 = 0;
                for (i, &byte) in bytes.iter().enumerate() {
                    v |= (byte as u128) << (i * 8);
                }
                v
            }
            Endness::Big => {
                let mut v: u128 = 0;
                for &byte in &bytes {
                    v = (v << 8) | (byte as u128);
                }
                v
            }
        };

        Ok(RustBV::concrete(value, size * 8))
    }

    /// Store to a concrete address, returning UnmappedPageInRegion for lazy regions.
    pub fn store_concrete_lazy(
        &mut self,
        addr: u64,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        // Check if pages are mapped
        let start_page = addr >> 12;
        let end_page = (addr + size as u64 + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                // Page not mapped - check if it's in a lazy region
                if self.is_in_lazy_region(page_num) {
                    return Err(MemoryError::UnmappedPageInRegion {
                        page_addr: page_num << 12,
                    });
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_num << 12,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // Permission checks live in store_concrete; this wrapper only adds
        // lazy-region detection for unmapped pages.
        self.store_concrete(addr, value)
    }

    /// Store to a concrete address with lazy region support.
    ///
    /// # Deprecation Warning
    ///
    /// This function previously auto-mapped zero pages for unmapped regions,
    /// but that behavior caused state divergence with Python's actual backer
    /// data. Now it returns UnmappedPageInRegion error so callers can fall
    /// back to Python callbacks to handle the store correctly.
    ///
    /// If you need auto-mapping behavior for internal Rust operations that
    /// don't involve Python state, use `store_concrete_automap_internal`.
    pub fn store_concrete_automap(
        &mut self,
        addr: u64,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        let start_page = addr >> 12;
        let end_page = (addr + size as u64 + PAGE_SIZE - 1) >> 12;

        // Check all pages are mapped - do NOT auto-map
        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                let page_addr = page_num << 12;
                if self.is_in_lazy_region(page_num) {
                    // Return error so caller can fall back to Python
                    return Err(MemoryError::UnmappedPageInRegion { page_addr });
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_addr,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // All pages mapped, proceed with store
        self.store_concrete(addr, value)
    }

    /// Store to a concrete address with internal auto-mapping.
    ///
    /// This is for internal Rust operations that don't involve Python state.
    /// For interpreter callbacks, use `store_concrete_automap` which propagates
    /// errors so Python can handle the store correctly.
    pub fn store_concrete_automap_internal(
        &mut self,
        addr: u64,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        let start_page = addr >> 12;
        let end_page = (addr + size as u64 + PAGE_SIZE - 1) >> 12;

        // Auto-map any missing pages in lazy regions
        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                let page_addr = page_num << 12;
                if self.is_in_lazy_region(page_num) {
                    self.auto_map_zero_page(page_addr);
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_addr,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // All pages now mapped, proceed with store
        self.store_concrete(addr, value)
    }

    /// Merge another memory into this one using a merge condition.
    ///
    /// For each byte that differs between `self` and `other`, the merged
    /// value is `ITE(merge_cond_other, other_byte, self_byte)`.
    ///
    /// Returns true if any memory was actually merged (values differed).
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
        let all_pages: std::collections::HashSet<u64> = self_pages.union(&other_pages).copied().collect();

        // Collect merge operations first to avoid borrow conflicts
        let mut merge_ops: Vec<(u64, u64, RustBV)> = Vec::new(); // (page_num, addr, ite_val)
        let mut pages_to_add: Vec<(u64, MemoryPage)> = Vec::new();

        for &page_num in &all_pages {
            let self_page = self.pages.get(&page_num);
            let other_page = other.pages.get(&page_num);

            match (self_page, other_page) {
                (Some(sp), Some(op)) => {
                    // Both have this page — compare concrete data
                    let s_data = sp.load_concrete(0, PAGE_SIZE as u16);
                    let o_data = op.load_concrete(0, PAGE_SIZE as u16);

                    if s_data == o_data && !sp.has_symbolic() && !op.has_symbolic() {
                        continue;
                    }

                    let base_addr = page_num << 12;
                    for i in 0..PAGE_SIZE as usize {
                        let s_byte = s_data[i];
                        let o_byte = o_data[i];

                        let s_sym = sp.has_symbolic() && sp.is_symbolic(i as u16);
                        let o_sym = op.has_symbolic() && op.is_symbolic(i as u16);

                        if !s_sym && !o_sym && s_byte == o_byte {
                            continue;
                        }

                        let addr = base_addr + i as u64;
                        let self_val = if s_sym {
                            self.symbolic_objects
                                .get(&addr)
                                .cloned()
                                .unwrap_or_else(|| RustBV::concrete(s_byte as u128, 8))
                        } else {
                            RustBV::concrete(s_byte as u128, 8)
                        };

                        let other_val = if o_sym {
                            other
                                .symbolic_objects
                                .get(&addr)
                                .cloned()
                                .unwrap_or_else(|| RustBV::concrete(o_byte as u128, 8))
                        } else {
                            RustBV::concrete(o_byte as u128, 8)
                        };

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
            let offset_in_page = (addr & (PAGE_SIZE as u64 - 1)) as u16;
            if let Some(page) = self.pages.get_mut(&page_num) {
                page.mark_symbolic(offset_in_page, 1);
            }
            merged = true;
        }

        for (page_num, page) in pages_to_add {
            self.pages.insert(page_num, page);
            merged = true;
        }

        // Merge symbolic objects from other that aren't page-based
        for (&addr, other_obj) in &other.symbolic_objects {
            if !self.symbolic_objects.contains_key(&addr) {
                self.symbolic_objects.insert(addr, other_obj.clone());
                self.symbolic_spans.insert(addr, (addr, other_obj.width()));
                merged = true;
            }
        }

        // Merge pending writes
        if !other.pending_writes.is_empty() {
            self.pending_writes.extend(other.pending_writes.clone());
            merged = true;
        }

        merged
    }
}

impl Clone for SymbolicMemory {
    fn clone(&self) -> Self {
        self.fork()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_concrete() {
        let ctx = SymContext::new_mock();

        let mut mem = SymbolicMemory::new(Endness::Little);

        // Map a page
        mem.map(0x1000, 0x1000, Permission::RWX);

        // Store and load
        let addr = RustBV::concrete(0x1000, 64);
        let value = RustBV::concrete(0x12345678, 32);
        mem.store(addr.clone(), value.clone(), &ctx).unwrap();

        let loaded = mem.load(addr, 4, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_memory_endianness() {
        let ctx = SymContext::new_mock();

        // Little-endian memory
        let mut mem_le = SymbolicMemory::new(Endness::Little);
        mem_le.map(0x1000, 0x1000, Permission::RWX);
        mem_le
            .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
            .unwrap();

        // First byte should be 0x78 (low byte)
        let byte0 = mem_le.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0x78));

        // Big-endian memory
        let mut mem_be = SymbolicMemory::new(Endness::Big);
        mem_be.map(0x1000, 0x1000, Permission::RWX);
        mem_be
            .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
            .unwrap();

        // First byte should be 0x12 (high byte)
        let byte0 = mem_be.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0x12));
    }

    #[test]
    fn test_memory_fork() {
        let ctx = SymContext::new_mock();

        let mut mem1 = SymbolicMemory::new(Endness::Little);
        mem1.map(0x1000, 0x1000, Permission::RWX);
        mem1.store_concrete(0x1000, RustBV::concrete(0xAAAA, 16))
            .unwrap();

        // Fork
        let mut mem2 = mem1.fork();

        // Modify mem2
        mem2.store_concrete(0x1000, RustBV::concrete(0xBBBB, 16))
            .unwrap();

        // mem1 should still have original value
        let val1 = mem1.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(val1.as_u64(), Some(0xAAAA));

        // mem2 should have new value
        let val2 = mem2.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(val2.as_u64(), Some(0xBBBB));
    }

    #[test]
    fn test_memory_map_data() {
        let ctx = SymContext::new_mock();

        let mut mem = SymbolicMemory::new(Endness::Little);

        // Map data
        let data: Vec<u8> = (0..=255u8).collect();
        mem.map_data(0x1000, &data, Permission::RX);

        // Check some bytes
        let byte0 = mem.load_concrete(0x1000, 1, &ctx).unwrap();
        assert_eq!(byte0.as_u64(), Some(0));

        let byte100 = mem.load_concrete(0x1064, 1, &ctx).unwrap();
        assert_eq!(byte100.as_u64(), Some(100));
    }

    #[test]
    fn test_symbolic_load_concrete_fast_path() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();

        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RWX);
        mem.store_concrete(0x1000, RustBV::concrete(0x12345678, 32)).unwrap();

        // Load with concrete address should work
        let addr = RustBV::concrete(0x1000, 64);
        let loaded = mem.load_symbolic(addr, 4, &ctx, &concretizer).unwrap();
        assert_eq!(loaded.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_symbolic_store_concrete_fast_path() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();

        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RWX);

        // Store with concrete address should work
        let addr = RustBV::concrete(0x1000, 64);
        let value = RustBV::concrete(0xDEADBEEF, 32);
        mem.store_symbolic(addr, value, &ctx, &concretizer).unwrap();

        let loaded = mem.load_concrete(0x1000, 4, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_permission_enforcement_disabled_by_default() {
        let ctx = SymContext::new_mock();
        let mut mem = SymbolicMemory::new(Endness::Little);
        // Read-only page; without enforcement, stores should still succeed.
        mem.map(0x1000, 0x1000, Permission::R);
        assert!(!mem.enforce_permissions());

        mem.store_concrete(0x1000, RustBV::concrete(0xCAFE, 16)).unwrap();
        let loaded = mem.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(0xCAFE));
    }

    #[test]
    fn test_permission_enforcement_blocks_write_to_readonly() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::R);
        mem.set_enforce_permissions(true);

        let err = mem
            .store_concrete(0x1000, RustBV::concrete(0xCAFE, 16))
            .unwrap_err();
        match err {
            MemoryError::Permission { addr, required, actual } => {
                assert_eq!(addr, 0x1000);
                assert!(required.write);
                assert!(!actual.write);
                assert!(actual.read);
            }
            other => panic!("expected Permission error, got {:?}", other),
        }
    }

    #[test]
    fn test_permission_enforcement_blocks_read_from_writeonly() {
        let ctx = SymContext::new_mock();
        let mut mem = SymbolicMemory::new(Endness::Little);
        // Write-only page (unusual but exercises the R check independently).
        mem.map(0x1000, 0x1000, Permission::W);
        mem.set_enforce_permissions(true);

        let err = mem.load_concrete(0x1000, 2, &ctx).unwrap_err();
        match err {
            MemoryError::Permission { addr, required, actual } => {
                assert_eq!(addr, 0x1000);
                assert!(required.read);
                assert!(!actual.read);
            }
            other => panic!("expected Permission error, got {:?}", other),
        }
    }

    #[test]
    fn test_permission_enforcement_allows_rwx() {
        let ctx = SymContext::new_mock();
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RWX);
        mem.set_enforce_permissions(true);

        mem.store_concrete(0x1000, RustBV::concrete(0xBEEF, 16)).unwrap();
        let loaded = mem.load_concrete(0x1000, 2, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(0xBEEF));
    }

    #[test]
    fn test_permission_enforcement_cross_page_write() {
        // First page RW, second page R. A 4-byte store straddling the
        // boundary at 0x1ffe should fail because the second page is R-only.
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RW);
        mem.map(0x2000, 0x1000, Permission::R);
        mem.set_enforce_permissions(true);

        let err = mem
            .store_concrete(0x1ffe, RustBV::concrete(0x11223344, 32))
            .unwrap_err();
        assert!(matches!(err, MemoryError::Permission { .. }));
    }

    #[test]
    fn test_permission_enforcement_propagates_through_fork() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::R);
        mem.set_enforce_permissions(true);

        let mut forked = mem.fork();
        assert!(forked.enforce_permissions());
        let err = forked
            .store_concrete(0x1000, RustBV::concrete(0xDEAD, 16))
            .unwrap_err();
        assert!(matches!(err, MemoryError::Permission { .. }));
    }

    #[test]
    fn test_check_executable_rejects_non_x_page() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RW);
        mem.set_enforce_permissions(true);
        let err = mem.check_executable(0x1234).unwrap_err();
        match err {
            MemoryError::Permission { addr, required, actual } => {
                assert_eq!(addr, 0x1234);
                assert_eq!(required, Permission::X);
                assert_eq!(actual, Permission::RW);
            }
            _ => panic!("expected Permission error, got {:?}", err),
        }
    }

    #[test]
    fn test_check_executable_allows_x_page() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x2000, 0x1000, Permission::RX);
        mem.set_enforce_permissions(true);
        mem.check_executable(0x2010).unwrap();
        // RWX also allows execute.
        mem.map(0x3000, 0x1000, Permission::RWX);
        mem.check_executable(0x3000).unwrap();
    }

    #[test]
    fn test_check_executable_skips_unmapped() {
        // Unmapped pages must NOT error here — callers (native lift /
        // Python lift_block) handle resolution. Only mapped-without-X
        // is a violation.
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.set_enforce_permissions(true);
        mem.check_executable(0xdeadbeef).unwrap();
    }

    #[test]
    fn test_check_executable_noop_when_disabled() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RW);
        // enforce_permissions defaults to false: even RW page must pass.
        mem.check_executable(0x1000).unwrap();
    }

    /// angr-wyxb: when two symbolic stores partially overlap, the address
    /// constraint on each store's address expression and any value
    /// constraints must remain in the solver after the stores complete.
    #[test]
    fn test_symbolic_store_partial_overlap_constraint_propagation() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RWX);

        // Symbolic addresses, each pinned to a specific value via a
        // constraint added to the solver up-front.
        let addr1 = RustBV::symbolic(&ctx, "addr1".to_string(), 64);
        let addr2 = RustBV::symbolic(&ctx, "addr2".to_string(), 64);
        ctx.assume_true(&addr1.eq(&RustBV::concrete(0x1000, 64), &ctx));
        ctx.assume_true(&addr2.eq(&RustBV::concrete(0x1004, 64), &ctx));

        // Two 64-bit symbolic values; sym1 carries an additional constraint
        // (a specific u128 value) so we can verify that this value-side
        // constraint also survives the partial overlap.
        let sym1 = RustBV::symbolic(&ctx, "sym1".to_string(), 64);
        let sym2 = RustBV::symbolic(&ctx, "sym2".to_string(), 64);
        let pinned_sym1: u128 = 0xDEAD_BEEF_F00D_BABE;
        ctx.assume_true(&sym1.eq(&RustBV::concrete(pinned_sym1, 64), &ctx));

        // Partial overlap: sym1 covers [0x1000, 0x1008); sym2 covers
        // [0x1004, 0x100C). Bytes [0x1004, 0x1008) are written by both.
        mem.store_symbolic(addr1.clone(), sym1.clone(), &ctx, &concretizer)
            .expect("store_symbolic addr1 must succeed");
        mem.store_symbolic(addr2.clone(), sym2.clone(), &ctx, &concretizer)
            .expect("store_symbolic addr2 must succeed");

        // The base context must remain satisfiable.
        assert!(
            ctx.is_sat(),
            "context must stay SAT after partial-overlap stores"
        );

        // addr1's solution constraint (== 0x1000) must survive: probing
        // an alternative value in a forked context must be UNSAT.
        let probe_addr = ctx.fork();
        probe_addr.assume_true(
            &addr1.eq(&RustBV::concrete(0x2000, 64), &probe_addr),
        );
        assert!(
            !probe_addr.is_sat(),
            "addr1==0x1000 must survive partial-overlap stores; \
             probing addr1==0x2000 was unexpectedly SAT"
        );

        // sym1's value constraint must survive: probing sym1 == 0 must
        // be UNSAT (sym1 is pinned to 0xDEAD_BEEF_F00D_BABE).
        let probe_sym1 = ctx.fork();
        probe_sym1
            .assume_true(&sym1.eq(&RustBV::concrete(0, 64), &probe_sym1));
        assert!(
            !probe_sym1.is_sat(),
            "sym1's value constraint must survive partial-overlap stores; \
             probing sym1==0 was unexpectedly SAT"
        );

        // Loading from 0x1000 must still produce a satisfiable expression
        // and stay consistent with the surviving value-side constraints.
        let loaded = mem
            .load_concrete(0x1000, 4, &ctx)
            .expect("load[0x1000:4] must succeed after partial-overlap stores");
        assert!(
            ctx.is_sat(),
            "context must remain SAT after the post-store load"
        );
        // Eval should produce *some* concrete model — addr/value constraints
        // narrow the model space but do not make it UNSAT.
        assert!(
            ctx.eval(&loaded).is_some(),
            "loaded value must be evaluable under the preserved constraints"
        );
    }

    /// angr-syf4: 128-bit symbolic store to big-endian memory must lay out
    /// bytes MSB-first (byte at addr+0 = MSB, byte at addr+15 = LSB) and
    /// sub-word loads must extract the corresponding lanes.
    ///
    /// Regression for the wide-symbolic-object byte-reversal class of bugs
    /// described in project_endianness_bug.
    #[test]
    fn test_big_endian_128bit_wide_symbolic_store() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();
        let mut mem = SymbolicMemory::new(Endness::Big);
        mem.map(0x1000, 0x1000, Permission::RWX);

        // 128-bit symbolic value pinned to a known constant so we can
        // predict every byte. MSB byte = 0x10, LSB byte = 0x1F.
        let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
        let sym = RustBV::symbolic(&ctx, "wide128".to_string(), 128);
        ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

        let addr = RustBV::concrete(0x1000, 64);
        mem.store_symbolic(addr, sym.clone(), &ctx, &concretizer)
            .expect("store_symbolic must succeed");
        assert!(ctx.is_sat(), "context must remain SAT after store");

        // Exact 16-byte load returns the full symbolic value.
        let full = mem
            .load_concrete(0x1000, 16, &ctx)
            .expect("16-byte load must succeed");
        assert_eq!(
            ctx.eval(&full),
            Some(pinned),
            "exact-width load must round-trip the pinned u128"
        );

        // Per-byte BE layout: byte at addr+i corresponds to byte position
        // (15 - i) when the value is interpreted MSB-first.
        for i in 0u64..16 {
            let byte_bv = mem
                .load_concrete(0x1000 + i, 1, &ctx)
                .expect("single-byte load must succeed");
            let expected: u128 = (pinned >> ((15 - i) * 8)) & 0xff;
            assert_eq!(
                ctx.eval(&byte_bv),
                Some(expected),
                "BE byte at offset {} expected 0x{:02x}",
                i, expected
            );
        }

        // 4-byte load at offset 4 should return bytes [4..8) of the BE
        // layout (bits [95:64] of the original u128 = 0x14151617).
        let word = mem
            .load_concrete(0x1004, 4, &ctx)
            .expect("4-byte load must succeed");
        let expected_word: u128 = (pinned >> 64) & 0xFFFF_FFFF;
        assert_eq!(
            ctx.eval(&word),
            Some(expected_word),
            "BE 4-byte load at offset 4 expected 0x{:08x}",
            expected_word
        );

        // 8-byte halves: high half at offset 0, low half at offset 8.
        let qhi = mem
            .load_concrete(0x1000, 8, &ctx)
            .expect("hi qword load must succeed");
        let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
        assert_eq!(
            ctx.eval(&qhi),
            Some(expected_qhi),
            "BE high qword expected 0x{:016x}",
            expected_qhi
        );

        let qlo = mem
            .load_concrete(0x1008, 8, &ctx)
            .expect("lo qword load must succeed");
        let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
        assert_eq!(
            ctx.eval(&qlo),
            Some(expected_qlo),
            "BE low qword expected 0x{:016x}",
            expected_qlo
        );
    }

    /// angr-jdz9: wide symbolic load whose two pinned solutions each
    /// straddle a different 4 KiB page boundary.
    ///
    /// Setup: addr ∈ {0x1FFC, 0x2FFC}; both pages flanking each boundary
    /// are mapped with distinct concrete bytes near the seam.
    /// - 0x1FFC..0x2003 → page 0x1000 last 4 bytes + page 0x2000 first 4
    /// - 0x2FFC..0x3003 → page 0x2000 last 4 bytes + page 0x3000 first 4
    ///
    /// Each cross-page slice yields a distinct 8-byte value. After
    /// `load_symbolic_unified`, evaluating the result under each pinned
    /// addr (in a forked context to isolate from the multi-solution
    /// constraint) must reproduce the LE concatenation. If the engine
    /// were to apply a permission check or page lookup for only one
    /// branch — or were to materialise the ITE only against the first
    /// page's bytes — the eval under the *other* solution would diverge
    /// from the expected value.
    ///
    /// Locks down current behaviour for angr-jdz9. Two pinned solutions
    /// with stride 0x1000 hit the Strided concretization branch in
    /// `load_symbolic_unified`, so this also covers the strided ITE
    /// path's per-leaf cross-page handling.
    #[test]
    fn test_symbolic_load_cross_page_multiple_solutions() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();

        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x1000, 0x1000, Permission::RWX);
        mem.map(0x2000, 0x1000, Permission::RWX);
        mem.map(0x3000, 0x1000, Permission::RWX);

        // Distinct bytes around each page boundary so the two straddling
        // 8-byte slices yield distinguishable LE values.
        // 0x1FFC..0x1FFF on page 0x1000:
        mem.store_concrete(0x1FFC, RustBV::concrete(0x44_33_22_11, 32))
            .expect("store 0x1FFC");
        // 0x2000..0x2003 on page 0x2000:
        mem.store_concrete(0x2000, RustBV::concrete(0x88_77_66_55, 32))
            .expect("store 0x2000");
        // 0x2FFC..0x2FFF on page 0x2000:
        mem.store_concrete(0x2FFC, RustBV::concrete(0xCC_BB_AA_99, 32))
            .expect("store 0x2FFC");
        // 0x3000..0x3003 on page 0x3000:
        mem.store_concrete(0x3000, RustBV::concrete(0x00_FF_EE_DD, 32))
            .expect("store 0x3000");

        // Symbolic 64-bit address constrained to {0x1FFC, 0x2FFC} via
        // `(addr == a) | (addr == b)`. Sat solver enumeration in the
        // concretizer should expose both solutions.
        let addr = RustBV::symbolic(&ctx, "load_addr".to_string(), 64);
        let a = RustBV::concrete(0x1FFC, 64);
        let b = RustBV::concrete(0x2FFC, 64);
        let eq_a = addr.eq(&a, &ctx);
        let eq_b = addr.eq(&b, &ctx);
        ctx.assume_true(&eq_a.or(&eq_b, &ctx));
        assert!(ctx.is_sat(), "two-solution constraint must be SAT");

        // Wide load with the symbolic address.
        let loaded = mem
            .load_symbolic_unified(addr.clone(), 8, &ctx, &concretizer)
            .expect("symbolic load must succeed when both solutions are mapped");
        assert!(ctx.is_sat(), "context must remain SAT after the load");

        let expected_at_1ffc: u128 = 0x88_77_66_55_44_33_22_11;
        let expected_at_2ffc: u128 = 0x00_FF_EE_DD_CC_BB_AA_99;
        assert_ne!(
            expected_at_1ffc, expected_at_2ffc,
            "test fixture: pinned values must differ to detect ITE collapse"
        );

        // Pin addr to 0x1FFC in a forked context and evaluate the load.
        // The eval must reproduce the LE concatenation of the bytes
        // straddling page 0x1000 / page 0x2000. If the loaded ITE was
        // built only against page 0x2000's bytes (collapsing the cross-
        // page slice), the eval would be wrong here.
        let probe_a = ctx.fork();
        probe_a.assume_true(&addr.eq(&RustBV::concrete(0x1FFC, 64), &probe_a));
        assert!(probe_a.is_sat(), "addr == 0x1FFC must remain SAT");
        assert_eq!(
            probe_a.eval(&loaded),
            Some(expected_at_1ffc),
            "load under addr==0x1FFC: expected LE of page-0x1000 last 4 \
             bytes followed by page-0x2000 first 4 bytes"
        );

        // Pin addr to 0x2FFC; the eval must reproduce the slice across
        // page 0x2000 / page 0x3000. If the ITE branch for the second
        // solution were missing (e.g. permission check applied only to
        // the first concretized address), this eval would diverge.
        let probe_b = ctx.fork();
        probe_b.assume_true(&addr.eq(&RustBV::concrete(0x2FFC, 64), &probe_b));
        assert!(probe_b.is_sat(), "addr == 0x2FFC must remain SAT");
        assert_eq!(
            probe_b.eval(&loaded),
            Some(expected_at_2ffc),
            "load under addr==0x2FFC: expected LE of page-0x2000 last 4 \
             bytes followed by page-0x3000 first 4 bytes"
        );

        // Sanity: both solutions are reachable from the loaded value
        // (the solver must see two distinct results).
        let solutions = ctx.eval_upto(&loaded, 4);
        assert!(
            solutions.contains(&expected_at_1ffc),
            "solver must enumerate the 0x1FFC slice in loaded value; \
             got {:?}",
            solutions
        );
        assert!(
            solutions.contains(&expected_at_2ffc),
            "solver must enumerate the 0x2FFC slice in loaded value; \
             got {:?}",
            solutions
        );
    }
}
