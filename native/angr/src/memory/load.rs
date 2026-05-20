//! Load operations for `SymbolicMemory`.
//!
//! Extracted from `memory/mod.rs` (angr-0lre). Holds the load_*/apply_pending_writes_*
//! family in a single file. Multiple `impl SymbolicMemory` blocks across files are
//! fine — Rust permits inherent impls to be split.

use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::symbolic::{
    RustBV, SymContext, record_mem_lazy_page_fault, record_mem_load,
    record_mem_load_symbolic_addr,
};
use crate::vex::Endness;

use super::page::{PAGE_MASK, PAGE_SIZE, Permission};
use super::{MemoryError, SymbolicMemory};

impl SymbolicMemory {
    /// Load bytes from memory as a RustBV.
    pub fn load(&self, addr: RustBV, size: u32, ctx: &SymContext) -> Result<RustBV, MemoryError> {
        // Symbolic-addr subcounter: only visible here (post-eval the addr is
        // a concrete u64 indistinguishable from a load via `load_concrete`).
        // The base load_count / load_bytes counters live in `load_concrete`
        // so they catch the state.rs hot path too.
        if addr.as_u64().is_none() {
            record_mem_load_symbolic_addr();
        }
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

        let result = self.load_concrete(concrete_addr, size, ctx);
        if let Err(MemoryError::UnmappedPageInRegion { .. }) = &result {
            record_mem_lazy_page_fault();
        }
        result
    }

    /// Load from a concrete address.
    pub fn load_concrete(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        record_mem_load(size as u64);
        // angr-3zhl: a later partial store that begins inside [addr+1, addr+size)
        // overwrites trailing bytes of an earlier wider object at `addr`. The
        // exact-address and span fast paths below would return the stale wider
        // value, ignoring the overwrite. Detect by scanning for any
        // symbolic_objects entry whose start lies strictly inside our range,
        // and fall back to a per-byte merge (which uses both indices).
        let has_inner_overlap =
            (1..size as u64).any(|i| self.symbolic_objects.contains_key(&(addr + i)));
        if has_inner_overlap {
            if let Some(merged) = self.try_byte_merge_load(addr, size, ctx) {
                return Ok(merged);
            }
            // Byte-merge fell through (e.g. unmapped/non-symbolic byte gap);
            // continue to the page-scan path below.
        } else {
            // Check for stored symbolic object at exact address first
            if let Some(sym) = self.symbolic_objects.get(&addr) {
                if sym.width() == size * 8 {
                    return Ok(sym.clone());
                }
                // Partial read from a wider symbolic object
                // e.g., reading 1 byte from a 128-byte BVS
                if sym.width() > size * 8 {
                    let total_bits = sym.width();
                    // Endianness determines which bits correspond to byte 0.
                    // BE: byte 0 = MSB → hi = total_bits-1, lo = total_bits-size*8.
                    // LE: byte 0 = LSB → hi = size*8-1, lo = 0.
                    let (hi, lo) = match self.endness {
                        Endness::Big => (total_bits - 1, total_bits - size * 8),
                        Endness::Little => (size * 8 - 1, 0),
                    };
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
                        let off_bits = base_offset as u32 * 8;
                        // BE: bytes [off, off+size) of the wide BV occupy bits
                        //     [total-1-off_bits : total-off_bits-size*8].
                        // LE: same byte range occupies bits
                        //     [off_bits+size*8-1 : off_bits].
                        let (hi, lo) = match self.endness {
                            Endness::Big => {
                                (total_bits - off_bits - 1, total_bits - off_bits - size * 8)
                            }
                            Endness::Little => (off_bits + size * 8 - 1, off_bits),
                        };
                        return Ok(sym.extract(hi, lo, ctx));
                    }
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
                // Concatenate bytes: first byte is at lowest address.
                // LE: byte 0 = LSB → reverse so parts[N-1] is high.
                // BE: byte 0 = MSB → already high-to-low.
                // angr-kg58: balanced fold gives an O(log N) AST instead
                // of O(N) left-skewed chain.
                let result = match self.endness {
                    Endness::Little => {
                        let high_to_low: Vec<RustBV> = parts.iter().rev().cloned().collect();
                        RustBV::concat_balanced(&high_to_low, ctx)
                    }
                    Endness::Big => RustBV::concat_balanced(&parts, ctx),
                };
                return Ok(result);
            }
            // Check for wider symbolic objects that contain our range
            for (&sym_addr, sym_val) in &self.symbolic_objects {
                let sym_size = sym_val.width() / 8;
                if sym_addr <= addr && addr + size as u64 <= sym_addr + sym_size as u64 {
                    let total_bits = sym_val.width();
                    let off_bits = (addr - sym_addr) as u32 * 8;
                    // Mirror of fast-path angr-v1q2 fix: the wide BV's byte
                    // layout depends on memory endianness.
                    // BE: bytes [off, off+size) occupy bits
                    //     [total-1-off_bits : total-off_bits-size*8].
                    // LE: same byte range occupies bits [off_bits+size*8-1 : off_bits].
                    let (hi, lo) = match self.endness {
                        Endness::Big => {
                            (total_bits - off_bits - 1, total_bits - off_bits - size * 8)
                        }
                        Endness::Little => (off_bits + size * 8 - 1, off_bits),
                    };
                    return Ok(sym_val.extract(hi, lo, ctx));
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
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => self.load_strided_balanced(&addr, base, stride, count, size, ctx)?,
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
                let value =
                    self.build_balanced_ite_load_after_prep(&addr_clone, &ready_addrs, size, ctx)?;
                // angr-62li: hoist the addr-domain disjunction over the
                // post-prep set (addresses that actually had data; matches
                // the set Z3 ITE-loads against, and is the soundest
                // restriction we can communicate without re-running the
                // concretizer over unmapped pages).
                Self::assert_address_disjunction(&addr, &ready_addrs, ctx);
                value
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
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
                return Err(MemoryError::SymbolicAddress {
                    description: reason,
                });
            }
        };

        // Apply any pending writes that might overlap this symbolic load
        Ok(self.apply_pending_writes_symbolic(&addr, size, base_value, ctx))
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
    pub(super) fn apply_pending_writes_concrete(
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
    pub(super) fn apply_pending_writes_symbolic(
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
    pub(super) fn load_concrete_lazy_inner(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Phase 1.2 (angr-n082): Multi-cell lazy load. When any byte in
        // [addr, addr+size) carries lazy alternatives, fall through to a
        // per-byte reconstruction that folds the alternatives into an
        // ITE chain. Multi supersedes plain Symbolic per
        // memory `invariant-multi-vs-symbolic-cell-states`, so this
        // check runs BEFORE the symbolic_objects fast path.
        if !self.multi_objects.is_empty()
            && (0..size as u64).any(|i| self.multi_objects.contains_key(&(addr + i)))
        {
            let start_page = addr >> 12;
            let end_page = (addr + size as u64 - 1) >> 12;
            self.check_perms_range(start_page, end_page, Permission::R)?;
            return self.assemble_load_with_multi(addr, size, ctx);
        }

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
                // Combine bytes into a single value using a balanced Concat
                // tree (angr-kg58). byte_objects[0] is the byte at the
                // lowest address.
                let result = match self.endness {
                    Endness::Little => {
                        // LE: byte 0 = LSB → reverse so high byte is first
                        byte_objects.reverse();
                        RustBV::concat_balanced(&byte_objects, ctx)
                    }
                    Endness::Big => {
                        // BE: byte 0 = MSB → already high-to-low
                        RustBV::concat_balanced(&byte_objects, ctx)
                    }
                };
                return Ok(result);
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

    /// Per-byte reconstruction for loads that touch at least one Multi cell
    /// (Phase 1.2 of angr-czph, bead angr-n082). For each byte in
    /// `[addr, addr + size)`:
    ///   * Multi byte: right-fold the payload's `(cond, value)` pairs into
    ///     an ITE chain whose final `else` is the page's concrete byte.
    ///     Calls `crate::symbolic::record_mem_ite_depth(payload.len())`
    ///     per memory `invariant-mem-ite-depth-counter`.
    ///   * Plain Symbolic byte: extract from `symbolic_objects` /
    ///     `symbolic_spans` mirroring `try_byte_merge_load`.
    ///   * Concrete byte: read the page byte into an 8-bit `RustBV`.
    /// The per-byte parts are concatenated endianness-correctly to match
    /// `try_byte_merge_load`. The caller is responsible for permission
    /// checks; this helper only handles unmapped pages by returning the
    /// usual `MemoryError`.
    pub(super) fn assemble_load_with_multi(
        &self,
        addr: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Phase 4.1 (angr-mmdh.1): wider-load collapse cache. When a load
        // spans the same Multi/Concrete byte mix as a previous load and
        // none of those bytes have changed (per-byte versions for Multi,
        // concrete byte values for Concrete), reuse the assembled BV
        // directly — skipping the N page lookups + N collapse + N concat
        // sequence that dominates gate-on cost for sym-write. Loads
        // touching plain Symbolic bytes are not cached.
        //
        // The cache is bypassed for size=1 because there is no concat to
        // skip — the cost is exactly one `MultiPayload::collapse` call,
        // which Phase 3 already memoizes.
        let fingerprint = if size > 1 {
            self.compute_wider_load_fingerprint(addr, size)
        } else {
            None
        };
        if let Some(fp) = &fingerprint {
            if let Some(cached) = self.wider_load_cache.borrow().get(&(addr, size)) {
                if cached.byte_fingerprints == *fp {
                    // invariant-mem-ite-depth-counter: replay the same
                    // count the per-byte miss path would have recorded.
                    if cached.total_ite_depth > 0 {
                        crate::symbolic::record_mem_ite_depth(cached.total_ite_depth);
                    }
                    return Ok(cached.bv.clone());
                }
            }
        }

        let mut total_ite_depth: u32 = 0;
        let mut byte_parts: Vec<RustBV> = Vec::with_capacity(size as usize);
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr >> 12;
            let offset = (byte_addr & PAGE_MASK) as u16;
            let page = match self.pages.get(&page_num) {
                Some(p) => p,
                None => {
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
            };

            let part = if let Some(payload) = self.multi_objects.get(&byte_addr) {
                // Phase 3 (angr-j0n4): MultiPayload::collapse memoizes the
                // right-folded ITE keyed on the page's concrete default byte.
                // Cache hits skip the alt-by-alt Z3 ITE build, eliminating
                // the per-load cost that gated Phase 2.
                let concrete_byte = page
                    .load_concrete(offset, 1)
                    .first()
                    .copied()
                    .unwrap_or(0);
                let depth = payload.len() as u32;
                total_ite_depth = total_ite_depth.saturating_add(depth);
                crate::symbolic::record_mem_ite_depth(depth);
                payload.collapse(concrete_byte, ctx)
            } else if page.is_symbolic(offset) {
                if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                    Self::extract_byte_lane(sym, 0, self.endness, ctx).ok_or_else(|| {
                        MemoryError::SymbolicAddress {
                            description: "symbolic byte lane out of range".to_string(),
                        }
                    })?
                } else if let Some(&(base_addr, base_width)) =
                    self.symbolic_spans.get(&byte_addr)
                {
                    let sym = self.symbolic_objects.get(&base_addr).ok_or(
                        MemoryError::SymbolicAddress {
                            description: "stale symbolic span".to_string(),
                        },
                    )?;
                    if sym.width() != base_width {
                        return Err(MemoryError::SymbolicAddress {
                            description: "symbolic span width mismatch".to_string(),
                        });
                    }
                    let off_in_sym = (byte_addr - base_addr) as u32;
                    Self::extract_byte_lane(sym, off_in_sym, self.endness, ctx).ok_or_else(
                        || MemoryError::SymbolicAddress {
                            description: "symbolic span byte offset out of range".to_string(),
                        },
                    )?
                } else {
                    return Err(MemoryError::SymbolicAddress {
                        description: "symbolic byte without tracked object".to_string(),
                    });
                }
            } else {
                let concrete_byte = page.load_concrete(offset, 1);
                RustBV::concrete(concrete_byte.first().copied().unwrap_or(0) as u128, 8)
            };
            byte_parts.push(part);
        }

        // Concatenate per endianness with a balanced tree (angr-kg58):
        //   LE: byte[0] is the LSB → reverse so high byte is first.
        //   BE: byte[0] is the MSB → already high-to-low.
        let result = match self.endness {
            Endness::Little => {
                byte_parts.reverse();
                RustBV::concat_balanced(&byte_parts, ctx)
            }
            Endness::Big => RustBV::concat_balanced(&byte_parts, ctx),
        };

        // Phase 4.1: cache the assembled BV when the load is fully covered
        // by Multi + Concrete bytes (fingerprint is Some). The hit path
        // above already returned for size==1, so insertion here is only
        // for size>1.
        if let Some(fp) = fingerprint {
            self.insert_wider_load_cache(
                (addr, size),
                crate::memory::CachedWiderLoad {
                    byte_fingerprints: fp,
                    total_ite_depth,
                    bv: result.clone(),
                },
            );
        }

        Ok(result)
    }

    /// Reconstruct a load by per-byte lookup against `symbolic_objects` and
    /// `symbolic_spans`, producing the correct interleaving when an earlier
    /// wider symbolic value was partially overwritten by a later store
    /// (angr-3zhl).
    ///
    /// For each byte in `[addr, addr + size)`:
    /// - if `symbolic_objects[byte_addr]` is set, this byte is the start
    ///   of a stored symbolic value — its byte-0 lane is used;
    /// - otherwise `symbolic_spans[byte_addr]` is consulted, and the
    ///   referenced wider value's byte at the matching offset is used;
    /// - any other case (concrete bytes, gaps, stale-span mismatch)
    ///   short-circuits to `None` so the caller can fall through to the
    ///   page-scan path.
    ///
    /// Returns the byte-merged bitvector, or `None` if a fully-symbolic
    /// reconstruction was not possible.
    fn try_byte_merge_load(&self, addr: u64, size: u32, ctx: &SymContext) -> Option<RustBV> {
        let mut byte_parts: Vec<RustBV> = Vec::with_capacity(size as usize);
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let part = if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                Self::extract_byte_lane(sym, 0, self.endness, ctx)?
            } else if let Some(&(base_addr, base_width)) = self.symbolic_spans.get(&byte_addr) {
                let sym = self.symbolic_objects.get(&base_addr)?;
                if sym.width() != base_width {
                    return None;
                }
                let offset = (byte_addr - base_addr) as u32;
                Self::extract_byte_lane(sym, offset, self.endness, ctx)?
            } else {
                return None;
            };
            byte_parts.push(part);
        }
        // Concatenate per endianness with a balanced tree (angr-kg58):
        //   LE: byte[0] is the LSB → reverse so high byte is first.
        //   BE: byte[0] is the MSB → already high-to-low.
        let result = match self.endness {
            Endness::Little => {
                byte_parts.reverse();
                RustBV::concat_balanced(&byte_parts, ctx)
            }
            Endness::Big => RustBV::concat_balanced(&byte_parts, ctx),
        };
        Some(result)
    }

    /// Extract the byte at `byte_offset` from a wider symbolic value,
    /// honouring memory endianness:
    ///   LE: byte 0 is the LSB → bits `[off*8+7 : off*8]`.
    ///   BE: byte 0 is the MSB → bits `[total-off*8-1 : total-off*8-8]`.
    /// Returns `None` if the offset is out of range.
    pub(super) fn extract_byte_lane(
        sym: &RustBV,
        byte_offset: u32,
        endness: Endness,
        ctx: &SymContext,
    ) -> Option<RustBV> {
        let total_bits = sym.width();
        if (byte_offset + 1) * 8 > total_bits {
            return None;
        }
        Some(match endness {
            Endness::Little => sym.extract(byte_offset * 8 + 7, byte_offset * 8, ctx),
            Endness::Big => sym.extract(
                total_bits - byte_offset * 8 - 1,
                total_bits - byte_offset * 8 - 8,
                ctx,
            ),
        })
    }
}
