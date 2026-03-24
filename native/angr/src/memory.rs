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

/// Page size in bytes (4KB).
pub const PAGE_SIZE: u64 = 4096;

/// Page mask for address calculation.
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Errors from memory operations.
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Unmapped memory access.
    Unmapped { addr: u64, size: u64 },
    /// Unmapped page in a mapped region (can be fetched on-demand).
    UnmappedPageInRegion { page_addr: u64 },
    /// Permission violation.
    Permission {
        addr: u64,
        required: Permission,
        actual: Permission,
    },
    /// Unresolvable symbolic address.
    SymbolicAddress { description: String },
    /// Out of bounds access.
    OutOfBounds { addr: u64, size: u64 },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::Unmapped { addr, size } => {
                write!(f, "unmapped memory at 0x{:x} (size {})", addr, size)
            }
            MemoryError::UnmappedPageInRegion { page_addr } => {
                write!(f, "unmapped page at 0x{:x} in mapped region", page_addr)
            }
            MemoryError::Permission {
                addr,
                required,
                actual,
            } => {
                write!(
                    f,
                    "permission violation at 0x{:x}: required {:?}, have {:?}",
                    addr, required, actual
                )
            }
            MemoryError::SymbolicAddress { description } => {
                write!(f, "symbolic address: {}", description)
            }
            MemoryError::OutOfBounds { addr, size } => {
                write!(f, "out of bounds access at 0x{:x} (size {})", addr, size)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

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
    /// Symbolic bytes (offset -> marker).
    /// We store these separately from the concrete data.
    symbolic: Arc<HashMap<u16, u8>>,
}

impl MemoryPage {
    /// Create a new zeroed page.
    pub fn new(base_addr: u64, permissions: Permission) -> Self {
        MemoryPage {
            data: Arc::new(vec![0u8; PAGE_SIZE as usize]),
            permissions,
            base_addr,
            symbolic: Arc::new(HashMap::new()),
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
            symbolic: Arc::new(HashMap::new()),
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
    pub fn has_symbolic(&self) -> bool {
        !self.symbolic.is_empty()
    }

    /// Get the symbolic byte offsets.
    pub fn symbolic_offsets(&self) -> Vec<u16> {
        self.symbolic.keys().copied().collect()
    }

    /// Load bytes from this page (concrete only).
    pub fn load_concrete(&self, offset: u16, size: u16) -> Vec<u8> {
        let start = offset as usize;
        let end = (start + size as usize).min(PAGE_SIZE as usize);
        self.data[start..end].to_vec()
    }

    /// Store bytes to this page (concrete).
    pub fn store_concrete(&mut self, offset: u16, bytes: &[u8]) {
        // Copy-on-write: if shared, make a unique copy
        let data = Arc::make_mut(&mut self.data);
        let start = offset as usize;
        let end = (start + bytes.len()).min(PAGE_SIZE as usize);
        data[start..end].copy_from_slice(&bytes[..(end - start)]);

        // Clear symbolic markers for overwritten bytes
        let sym = Arc::make_mut(&mut self.symbolic);
        for i in offset..(offset + bytes.len() as u16) {
            sym.remove(&i);
        }
    }

    /// Mark bytes as symbolic.
    pub fn mark_symbolic(&mut self, offset: u16, size: u16) {
        let sym = Arc::make_mut(&mut self.symbolic);
        for i in offset..(offset + size) {
            sym.insert(i, 1);
        }
    }

    /// Check if a byte is symbolic.
    pub fn is_symbolic(&self, offset: u16) -> bool {
        self.symbolic.contains_key(&offset)
    }

    /// Fork this page (O(1) via Arc).
    pub fn fork(&self) -> Self {
        MemoryPage {
            data: Arc::clone(&self.data),
            permissions: self.permissions,
            base_addr: self.base_addr,
            symbolic: Arc::clone(&self.symbolic),
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
        }
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
        let start_page = addr >> 12;
        let offset_in_page = addr & PAGE_MASK;

        let mut remaining = data;
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr >> 12;
            let page_offset = (current_addr & PAGE_MASK) as usize;
            let bytes_in_page = (PAGE_SIZE as usize - page_offset).min(remaining.len());

            // Get or create page
            let page = self.pages.entry(page_num).or_insert_with(|| {
                MemoryPage::new(page_num << 12, permissions)
            });

            // Write data
            let mut page = page.clone();
            page.store_concrete(page_offset as u16, &remaining[..bytes_in_page]);
            self.pages.insert(page_num, page);

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
        // Check for stored symbolic object first
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            if sym.width() == size * 8 {
                return Ok(sym.clone());
            }
        }

        let start_page = addr >> 12;
        let end_page = (addr + size as u64 - 1) >> 12;

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

        // If symbolic, store in symbolic_objects
        if value.is_symbolic() {
            self.symbolic_objects.insert(addr, value.clone());
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

            if let Some(page) = self.pages.get(&page_num) {
                let mut page = page.clone();
                page.store_concrete(page_offset, &remaining[..bytes_in_page]);
                self.pages.insert(page_num, page);
                // Mark page as dirty
                self.dirty_pages.insert(page_num);
            }

            remaining = &remaining[bytes_in_page..];
            current_addr += bytes_in_page as u64;
        }

        // Clear any symbolic object at this address
        self.symbolic_objects.remove(&addr);

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

        // Try to concretize the address
        match concretizer.concretize(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_lazy(concrete_addr, size, ctx)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                // Use balanced ITE tree for strided access - O(log N) depth vs O(N)
                self.load_strided_balanced(&addr, base, stride, count, size, ctx)
            }
            ConcretizationResult::Multiple(addrs) => {
                // Build balanced ITE tree for better solver performance
                self.build_balanced_ite_load(&addr, &addrs, size, ctx)
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

        // Try to concretize the address
        match concretizer.concretize(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(concrete_addr, value)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                // Handle strided access pattern
                self.store_strided(&addr, &value, base, stride, count, ctx)
            }
            ConcretizationResult::Multiple(addrs) => {
                // For each candidate address, perform a conditional store:
                // mem[candidate] = If(addr == candidate, new_value, mem[candidate])
                let size = value.width() / 8;

                for &candidate in &addrs {
                    // Build condition: addr == candidate
                    let addr_const = RustBV::concrete(candidate as u128, addr.width());
                    let cond = addr.eq(&addr_const, ctx);

                    // Load current value at candidate address
                    let current = self.load_concrete_lazy(candidate, size, ctx)?;

                    // Build conditional value
                    let conditional_value = cond.ite(&value, &current, ctx);

                    // Store the conditional value
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
    pub fn prepare_addresses_for_ite(&mut self, addrs: &[u64], size: u32) -> Vec<u64> {
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
                // Generate a unique name for the unconstrained memory read
                *counter += 1;
                RustBV::symbolic(ctx, &format!("unc_mem_{:x}_{}", addr, counter), size * 8)
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

        // Try to concretize the address
        match concretizer.concretize(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_automap(concrete_addr, size, ctx)
            }
            ConcretizationResult::Multiple(addrs) => {
                // Prepare addresses by auto-mapping unmapped pages in lazy regions
                // This mutates self, so we do it first
                let ready_addrs = self.prepare_addresses_for_ite(&addrs, size);

                if ready_addrs.is_empty() {
                    // All addresses unmapped - return unconstrained
                    return Ok(RustBV::symbolic(
                        ctx,
                        &format!("mem_all_unmapped_{}", size),
                        size * 8,
                    ));
                }

                // Now that preparation is done, build the ITE tree (immutable borrow)
                // Clone ready_addrs to own the data
                let addr_clone = addr.clone();
                self.build_balanced_ite_load_after_prep(&addr_clone, &ready_addrs, size, ctx)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                // Prepare strided region for access (mutates self)
                self.prepare_strided_region(base, stride, count, size);
                // Use the existing balanced ITE builder (immutable borrow)
                self.load_strided_balanced(&addr, base, stride, count, size, ctx)
            }
            ConcretizationResult::TooLarge { .. } => {
                // Address range too large - return unconstrained symbolic value
                // This is better than failing, as it preserves soundness
                Ok(RustBV::symbolic(
                    ctx,
                    &format!("mem_unbounded_{}", size),
                    size * 8,
                ))
            }
            ConcretizationResult::Failed(reason) => {
                Err(MemoryError::SymbolicAddress { description: reason })
            }
        }
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
                &format!("mem_empty_ite_{}", size),
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
    ) -> Result<(), MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.store_concrete_automap(concrete_addr, value);
        }

        // Try to concretize the address
        match concretizer.concretize(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(concrete_addr, value)
            }
            ConcretizationResult::Multiple(addrs) => {
                // Prepare addresses by auto-mapping
                let ready_addrs = self.prepare_addresses_for_ite(&addrs, value.width() / 8);

                // Perform conditional stores for each ready address
                self.store_conditional_multiple(&addr, &value, &ready_addrs, ctx)
            }
            ConcretizationResult::Strided { base, stride, count } => {
                // Prepare strided region
                self.prepare_strided_region(base, stride, count, value.width() / 8);
                // Use existing strided store
                self.store_strided(&addr, &value, base, stride, count, ctx)
            }
            ConcretizationResult::TooLarge { .. } => {
                // For truly unbounded addresses, we can't do much.
                // Log a warning and treat as a no-op to avoid unsoundness.
                // This is better than crashing or corrupting state.
                // A more sophisticated approach would track symbolic writes.
                Ok(())
            }
            ConcretizationResult::Failed(reason) => {
                Err(MemoryError::SymbolicAddress { description: reason })
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
        }
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
        // Store in symbolic_objects for lookup
        self.symbolic_objects.insert(addr, value.clone());

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

            // Clone and modify
            let mut page = page.clone();
            page.mark_symbolic(offset, 1);
            self.pages.insert(page_num, page);
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

    /// Clear all symbolic objects (used when resetting state).
    pub fn clear_symbolic_objects(&mut self) {
        self.symbolic_objects.clear();
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
        // Use the lazy inner function which returns UnmappedPageInRegion
        // for unmapped pages in lazy regions. Caller should handle this
        // by falling back to Python callback.
        self.load_concrete_lazy_inner(addr, size, ctx)
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
        match self.load_concrete_lazy_inner(addr, size, ctx) {
            Ok(v) => Ok(v),
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Auto-map the missing page
                self.auto_map_zero_page(page_addr);
                // Retry the load
                self.load_concrete_lazy_inner(addr, size, ctx)
            }
            Err(e) => Err(e),
        }
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
        self.load_concrete_lazy_inner(addr, size, ctx)
    }

    /// Internal implementation of load_concrete_lazy.
    fn load_concrete_lazy_inner(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let mut bytes = Vec::with_capacity(size as usize);
        let mut has_symbolic = false;

        // Check for stored symbolic object first
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            if sym.width() == size * 8 {
                return Ok(sym.clone());
            }
        }

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
                // Page not mapped - check if it's in a lazy region
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
                        let mut result = byte_objects.pop().unwrap();
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

        // All pages mapped, proceed with store
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
}
