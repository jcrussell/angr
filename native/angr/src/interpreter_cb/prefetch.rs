use super::helpers::bytes_to_bv;
use super::*;

impl<'a> CallbackInterpreter<'a> {
    /// Enable or disable load prefetching.
    pub fn set_load_prefetch(&mut self, enabled: bool) {
        self.use_load_prefetch = enabled;
    }

    /// Set the number of pages to prefetch when fetching a page.
    ///
    /// When a page needs to be fetched from Python, this many additional pages
    /// will be fetched in each direction (before and after) to improve locality.
    /// Set to 0 to disable page prefetching.
    pub fn set_page_prefetch_count(&mut self, count: u32) {
        self.page_prefetch_count = count;
    }

    /// Fetch a page from Python and map it in Rust memory.
    ///
    /// This is called when a load/store encounters an unmapped page in a lazy region.
    /// The page is fetched via Python callback and added to rust_memory.
    ///
    /// Returns true if the page was successfully fetched and mapped.
    pub fn fetch_page(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addr: u64,
    ) -> Result<bool, CbExecutionError> {
        // Check if we have the callback
        if !callbacks.has_fetch_page() {
            return Ok(false);
        }

        // Call Python to fetch the page
        let (data, permissions, is_mapped) = callbacks
            .call_fetch_page(py, page_addr)
            .map_err(|e| CbExecutionError::Callback(format!("fetch_page failed: {}", e)))?;

        if !is_mapped {
            // Page doesn't exist in Python memory either
            return Ok(false);
        }

        // Ensure we have Rust memory enabled
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Convert permission bits to Permission struct
            let perm = Permission::from_bits(permissions);

            // Map the page in Rust memory
            rust_mem.map_page(page_addr, data, perm);

            Ok(true)
        } else {
            // Rust memory not enabled - shouldn't happen but handle gracefully
            Ok(false)
        }
    }

    /// Fetch multiple pages from Python in a batch.
    ///
    /// This is more efficient than fetching pages one at a time.
    /// Returns the number of pages successfully fetched.
    pub fn fetch_pages_batch(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addrs: &[u64],
    ) -> Result<usize, CbExecutionError> {
        if page_addrs.is_empty() {
            return Ok(0);
        }

        // Call Python to fetch pages in batch
        let results = callbacks
            .call_batch_fetch_pages(py, page_addrs)
            .map_err(|e| CbExecutionError::Callback(format!("batch_fetch_pages failed: {}", e)))?;

        let mut fetched = 0;

        if let Some(ref mut rust_mem) = self.rust_memory {
            for (i, (data, permissions, is_mapped)) in results.into_iter().enumerate() {
                if is_mapped {
                    let page_addr = page_addrs[i];
                    let perm = Permission::from_bits(permissions);
                    rust_mem.map_page(page_addr, data, perm);
                    fetched += 1;
                }
            }
        }

        Ok(fetched)
    }

    /// Fetch a page and prefetch nearby pages for better locality.
    ///
    /// This is an optimization that reduces future FFI calls by speculatively
    /// fetching pages around the accessed address. Useful for sequential access
    /// patterns (like stack frames, arrays, etc.).
    ///
    /// When `enable_eager_prefetch` is set in config, this will fetch all
    /// unmapped pages in the lazy region containing the page. Otherwise,
    /// it fetches `prefetch_count` pages in each direction.
    ///
    /// Args:
    ///     prefetch_count: Number of pages to prefetch in each direction (0 = disabled)
    ///
    /// Returns true if the main page was successfully fetched.
    pub fn fetch_page_with_prefetch(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addr: u64,
        prefetch_count: u32,
    ) -> Result<bool, CbExecutionError> {
        if prefetch_count == 0 && !self.config.enable_eager_prefetch {
            // No prefetching, just fetch the single page
            return self.fetch_page(py, callbacks, page_addr);
        }

        // Build list of pages to fetch
        let pages_to_fetch = if self.config.enable_eager_prefetch {
            // Eager region prefetch: fetch all unmapped pages in the region
            self.get_eager_prefetch_list(page_addr)
        } else {
            // Nearby prefetch: fetch pages before/after the trigger
            self.get_nearby_prefetch_list(page_addr, prefetch_count)
        };

        if pages_to_fetch.is_empty() {
            // No pages to fetch (shouldn't happen, but handle gracefully)
            return self.fetch_page(py, callbacks, page_addr);
        }

        // Fetch all pages in one batch
        let fetched = self.fetch_pages_batch(py, callbacks, &pages_to_fetch)?;

        // Return true if at least the main page was fetched
        if let Some(ref rust_mem) = self.rust_memory {
            Ok(rust_mem.is_mapped(page_addr))
        } else {
            Ok(fetched > 0)
        }
    }

    /// Get pages to fetch for eager region prefetch.
    ///
    /// Returns all unmapped pages in the lazy region containing `page_addr`,
    /// up to `max_prefetch_batch` pages.
    fn get_eager_prefetch_list(&self, page_addr: u64) -> Vec<u64> {
        if let Some(ref rust_mem) = self.rust_memory {
            if let Some(pages) =
                rust_mem.get_region_prefetch_list(page_addr, self.config.max_prefetch_batch)
            {
                return pages;
            }
        }
        // Fallback to just the main page
        vec![page_addr]
    }

    /// Get the current stack pointer value (architecture-aware).
    fn get_stack_pointer(&self) -> Option<u64> {
        let arch = self.registers.arch();
        let offset = arch.sp_offset();
        let size = arch.bytes();
        self.registers.get(offset, size, self.ctx).as_u64()
    }

    /// Check if an address is in the stack region (near current RSP).
    /// Stack typically grows downward, so we check if addr is below RSP + some margin.
    fn is_stack_region(&self, addr: u64) -> bool {
        if let Some(sp) = self.get_stack_pointer() {
            // Stack region: addresses from RSP - 1MB to RSP + 64KB
            // (stack grows down, but we allow some upward margin for locals)
            let stack_base = sp.saturating_sub(1024 * 1024); // 1MB below RSP
            let stack_limit = sp.saturating_add(64 * 1024); // 64KB above RSP
            addr >= stack_base && addr <= stack_limit
        } else {
            false
        }
    }

    /// Get pages to fetch for nearby prefetch (stack-aware).
    ///
    /// Returns `prefetch_count` unmapped pages, prioritizing stack growth direction
    /// (downward) when the access is in the stack region.
    fn get_nearby_prefetch_list(&self, page_addr: u64, prefetch_count: u32) -> Vec<u64> {
        let page_size = 0x1000u64;
        let mut pages_to_fetch = Vec::with_capacity(1 + 2 * prefetch_count as usize);

        // Add main page first
        pages_to_fetch.push(page_addr);

        // Determine if this is a stack access
        let is_stack = self.is_stack_region(page_addr);

        // For stack accesses, prioritize downward prefetch (stack grows down)
        // For non-stack, use balanced bidirectional prefetch
        let (down_count, up_count) = if is_stack {
            // Stack: 3x more pages downward than upward
            let down = (prefetch_count * 3).min(16);
            let up = prefetch_count.min(4);
            (down, up)
        } else {
            // Non-stack: equal in both directions
            (prefetch_count, prefetch_count)
        };

        // Add pages before (lower addresses - stack growth direction)
        for i in 1..=down_count {
            if let Some(addr) = page_addr.checked_sub(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory {
                    if !rust_mem.is_mapped(addr) && rust_mem.is_addr_in_lazy_region(addr) {
                        pages_to_fetch.push(addr);
                    }
                }
            }
        }

        // Add pages after (higher addresses)
        for i in 1..=up_count {
            if let Some(addr) = page_addr.checked_add(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory {
                    if !rust_mem.is_mapped(addr) && rust_mem.is_addr_in_lazy_region(addr) {
                        pages_to_fetch.push(addr);
                    }
                }
            }
        }

        pages_to_fetch
    }

    /// Clear the load prefetch cache.
    ///
    /// This should be called after stores to invalidate potentially stale values.
    pub fn clear_prefetch_cache(&mut self) {
        self.load_prefetch_cache.clear();
    }

    /// Check if a load result is in the prefetch cache.
    #[inline]
    pub fn get_prefetched_load(&self, addr: u64, size: usize) -> Option<&PrefetchedLoad> {
        self.load_prefetch_cache.get(&(addr, size))
    }

    /// Scan an IRSB for Load expressions with concrete addresses.
    ///
    /// This collects (address, size) pairs for loads that can be prefetched.
    /// Only loads with concrete addresses (computed from temps/constants) are collected.
    /// Writes into `loads` without clearing — the caller is responsible for clearing
    /// when reusing a scratch buffer.
    fn scan_loads_in_irsb(&self, irsb: &IRSB, loads: &mut Vec<(u64, usize)>) {
        for stmt in &irsb.statements {
            self.scan_loads_in_stmt(stmt, irsb, loads);
        }

        // Also scan the next expression
        self.scan_loads_in_expr(&irsb.next, irsb, loads);
    }

    /// Scan a statement for Load expressions.
    fn scan_loads_in_stmt(&self, stmt: &IRStmt, irsb: &IRSB, loads: &mut Vec<(u64, usize)>) {
        match stmt {
            IRStmt::WrTmp { data, .. } => {
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Put { data, .. } => {
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Store { addr, data, .. } => {
                self.scan_loads_in_expr(addr, irsb, loads);
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Exit { guard, .. } => {
                self.scan_loads_in_expr(guard, irsb, loads);
            }
            _ => {}
        }
    }

    /// Scan an expression for Load expressions.
    fn scan_loads_in_expr(&self, expr: &IRExpr, irsb: &IRSB, loads: &mut Vec<(u64, usize)>) {
        match expr {
            IRExpr::Load { addr, ty, .. } => {
                let size = ty.bytes() as usize;
                // Try to evaluate the address to a concrete value
                if let Some(addr_val) = self.try_eval_expr_concrete(addr, &irsb.tyenv) {
                    // Check if this address is NOT already in cached concrete memory
                    // (no point prefetching what we can already read locally)
                    if self.try_read_concrete_memory(addr_val, size).is_none() {
                        loads.push((addr_val, size));
                    }
                }
                // Also scan the address expression itself
                self.scan_loads_in_expr(addr, irsb, loads);
            }
            IRExpr::Unop { arg, .. } => {
                self.scan_loads_in_expr(arg, irsb, loads);
            }
            IRExpr::Binop { left, right, .. } => {
                self.scan_loads_in_expr(left, irsb, loads);
                self.scan_loads_in_expr(right, irsb, loads);
            }
            IRExpr::ITE {
                cond,
                iftrue,
                iffalse,
                ..
            } => {
                self.scan_loads_in_expr(cond, irsb, loads);
                self.scan_loads_in_expr(iftrue, irsb, loads);
                self.scan_loads_in_expr(iffalse, irsb, loads);
            }
            IRExpr::CCall { args, .. } => {
                for arg in args {
                    self.scan_loads_in_expr(arg, irsb, loads);
                }
            }
            _ => {}
        }
    }

    /// Try to evaluate an expression to a concrete u64 value (for prefetching).
    ///
    /// This is a simplified evaluation that only handles constants and simple
    /// operations. It doesn't evaluate temps since we're scanning before execution.
    fn try_eval_expr_concrete(&self, expr: &IRExpr, _tyenv: &TypeEnv) -> Option<u64> {
        match expr {
            IRExpr::Const(c) => match c {
                IRConst::U8(v) => Some(*v as u64),
                IRConst::U16(v) => Some(*v as u64),
                IRConst::U32(v) => Some(*v as u64),
                IRConst::U64(v) => Some(*v),
                _ => None,
            },
            IRExpr::Get { offset, ty } => {
                // Try to get a concrete register value
                let size = ty.bytes();
                let reg_val = self.registers.get(*offset, size, self.ctx);
                reg_val.as_u64()
            }
            // For more complex expressions (binops, etc.), we could evaluate them
            // but for simplicity we skip them - they'll be handled by the regular path
            _ => None,
        }
    }

    /// Prefetch loads for a block using batch callback.
    ///
    /// This scans the IRSB for Load expressions with concrete addresses,
    /// batches them into a single callback, and populates the prefetch cache.
    pub(super) fn prefetch_loads_for_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        if !self.use_load_prefetch {
            return Ok(());
        }

        // Clear previous prefetch cache
        self.load_prefetch_cache.clear();

        // Reuse scratch buffers across blocks to avoid per-block allocator churn.
        // Take buffers out of `self` so the borrow checker permits `&self` calls
        // (scan_loads_in_irsb / try_eval_expr_concrete) while we're writing.
        let mut loads = std::mem::take(&mut self.prefetch_loads_scratch);
        loads.clear();
        self.scan_loads_in_irsb(irsb, &mut loads);

        if loads.is_empty() {
            self.prefetch_loads_scratch = loads;
            return Ok(());
        }

        // Deduplicate loads (same address+size only needs to be fetched once).
        let unique_loads = &mut self.prefetch_unique_scratch;
        let seen = &mut self.prefetch_dedup_scratch;
        unique_loads.clear();
        seen.clear();
        for &load in &loads {
            if seen.insert(load) {
                unique_loads.push(load);
            }
        }
        self.prefetch_loads_scratch = loads;

        // Convert to callback format: (addr, size as u32).
        let callback_loads = &mut self.prefetch_callback_scratch;
        callback_loads.clear();
        callback_loads.extend(unique_loads.iter().map(|&(addr, size)| (addr, size as u32)));

        // Call batch callback
        let results = callbacks
            .call_memory_load_batch(py, callback_loads)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Populate prefetch cache
        for (i, (addr, size)) in unique_loads.iter().enumerate() {
            if let Some((data, is_symbolic, symbolic_ast)) = results.get(i) {
                let value = if *is_symbolic {
                    // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);

                        // Fast path: check for RustBVHandle first
                        if let Some(ref table) = self.symbol_table {
                            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                bv
                            } else if is_claripy_ast(&ast) {
                                // Slow path: claripy AST conversion
                                match claripy_to_rustbv(py, &ast, self.ctx) {
                                    Ok(bv) => bv,
                                    Err(_) => {
                                        // Fallback to fresh symbolic
                                        RustBV::symbolic(
                                            self.ctx,
                                            &format!("prefetch_{:x}_{}", addr, size),
                                            (size * 8) as u32,
                                        )
                                    }
                                }
                            } else {
                                RustBV::symbolic(
                                    self.ctx,
                                    &format!("prefetch_{:x}_{}", addr, size),
                                    (size * 8) as u32,
                                )
                            }
                        } else if is_claripy_ast(&ast) {
                            match claripy_to_rustbv(py, &ast, self.ctx) {
                                Ok(bv) => bv,
                                Err(_) => {
                                    // Fallback to fresh symbolic
                                    RustBV::symbolic(
                                        self.ctx,
                                        &format!("prefetch_{:x}_{}", addr, size),
                                        (size * 8) as u32,
                                    )
                                }
                            }
                        } else {
                            RustBV::symbolic(
                                self.ctx,
                                &format!("prefetch_{:x}_{}", addr, size),
                                (size * 8) as u32,
                            )
                        }
                    } else {
                        RustBV::symbolic(
                            self.ctx,
                            &format!("prefetch_{:x}_{}", addr, size),
                            (size * 8) as u32,
                        )
                    }
                } else {
                    bytes_to_bv(data, (size * 8) as u32)
                };

                self.load_prefetch_cache.insert(
                    (*addr, *size),
                    PrefetchedLoad {
                        value,
                        is_symbolic: *is_symbolic,
                    },
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vex::ir::{Endness, IRType};

    fn new_interp(ctx: &SymContext) -> CallbackInterpreter<'_> {
        CallbackInterpreter::new(VexArch::AMD64, ctx)
    }

    fn make_irsb_load(addr: u64, load_addr: u64, size: IRType) -> IRSB {
        let mut irsb = IRSB::new(addr, VexArch::AMD64);
        irsb.statements.push(IRStmt::IMark {
            addr,
            len: 4,
            delta: 0,
        });
        // Allocate a temp for the load destination.
        let tmp = irsb.tyenv.new_temp(size);
        irsb.statements.push(IRStmt::WrTmp {
            tmp,
            data: IRExpr::Load {
                addr: Box::new(IRExpr::Const(IRConst::U64(load_addr))),
                ty: size,
                endness: Endness::Little,
            },
        });
        irsb
    }

    #[test]
    fn set_load_prefetch_toggles_flag() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        assert!(!interp.use_load_prefetch);
        interp.set_load_prefetch(true);
        assert!(interp.use_load_prefetch);
        interp.set_load_prefetch(false);
        assert!(!interp.use_load_prefetch);
    }

    #[test]
    fn set_page_prefetch_count_overrides_default() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // Default established in CallbackInterpreter::new.
        let original = interp.page_prefetch_count;
        interp.set_page_prefetch_count(8);
        assert_eq!(interp.page_prefetch_count, 8);
        assert_ne!(interp.page_prefetch_count, original);
    }

    #[test]
    fn clear_prefetch_cache_empties_map() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.load_prefetch_cache.insert(
            (0x4000, 4),
            PrefetchedLoad {
                value: RustBV::concrete(0xaa, 32),
                is_symbolic: false,
            },
        );
        assert!(interp.get_prefetched_load(0x4000, 4).is_some());
        interp.clear_prefetch_cache();
        assert!(interp.get_prefetched_load(0x4000, 4).is_none());
    }

    #[test]
    fn get_prefetched_load_returns_value_on_hit() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.load_prefetch_cache.insert(
            (0x8000, 8),
            PrefetchedLoad {
                value: RustBV::concrete(0xbb, 64),
                is_symbolic: false,
            },
        );
        let hit = interp.get_prefetched_load(0x8000, 8).expect("hit");
        assert!(!hit.is_symbolic);
        assert_eq!(hit.value.as_u64(), Some(0xbb));
        // Miss on different size.
        assert!(interp.get_prefetched_load(0x8000, 4).is_none());
    }

    #[test]
    fn scan_loads_in_irsb_finds_concrete_load() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let irsb = make_irsb_load(0x1000, 0x4000, IRType::I32);
        let mut loads = Vec::new();
        interp.scan_loads_in_irsb(&irsb, &mut loads);
        assert!(loads.contains(&(0x4000, 4)));
    }

    #[test]
    fn scan_loads_skips_addresses_in_concrete_memory() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // Cache 0x4000 in concrete memory so the scanner skips it.
        interp.add_concrete_memory(0x4000, vec![0u8; 0x100]);
        let irsb = make_irsb_load(0x1000, 0x4000, IRType::I32);
        let mut loads = Vec::new();
        interp.scan_loads_in_irsb(&irsb, &mut loads);
        assert!(!loads.iter().any(|&(a, _)| a == 0x4000));
    }

    #[test]
    fn try_eval_expr_concrete_handles_const_u64() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let expr = IRExpr::Const(IRConst::U64(0xdead));
        assert_eq!(interp.try_eval_expr_concrete(&expr, &env), Some(0xdead));
    }

    #[test]
    fn try_eval_expr_concrete_handles_concrete_register() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
        let env = TypeEnv::new();
        // AMD64 RSP = offset 48
        let expr = IRExpr::Get {
            offset: 48,
            ty: IRType::I64,
        };
        assert_eq!(interp.try_eval_expr_concrete(&expr, &env), Some(0x7fff_0000));
    }

    #[test]
    fn try_eval_expr_concrete_returns_none_for_symbolic_register() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let sym = RustBV::symbolic(&ctx, "rsp_sym", 64);
        interp.registers.put_reg("rsp", sym);
        let env = TypeEnv::new();
        let expr = IRExpr::Get {
            offset: 48,
            ty: IRType::I64,
        };
        assert!(interp.try_eval_expr_concrete(&expr, &env).is_none());
    }

    #[test]
    fn try_eval_expr_concrete_returns_none_for_unop() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let expr = IRExpr::RdTmp(0);
        // RdTmp isn't handled by simplified concrete eval.
        assert_eq!(interp.try_eval_expr_concrete(&expr, &env), None);
    }

    #[test]
    fn get_stack_pointer_returns_rsp_value() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rsp", RustBV::concrete(0x1234_5678, 64));
        assert_eq!(interp.get_stack_pointer(), Some(0x1234_5678));
    }

    #[test]
    fn is_stack_region_below_rsp_within_window() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
        // Address 64KB below RSP — well within the 1MB window.
        assert!(interp.is_stack_region(0x7fff_0000 - 0x10000));
        // Address 2MB below RSP — outside the window.
        assert!(!interp.is_stack_region(0x7fff_0000 - 0x200000));
    }

    #[test]
    fn is_stack_region_above_rsp_within_window() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
        // Address 16KB above RSP — within 64KB upward margin.
        assert!(interp.is_stack_region(0x7fff_0000 + 0x4000));
        // Address 128KB above RSP — outside upward margin.
        assert!(!interp.is_stack_region(0x7fff_0000 + 0x20000));
    }

    #[test]
    fn get_nearby_prefetch_list_main_page_only_without_rust_memory() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        // Without rust_memory the inner is_mapped check skips siblings,
        // leaving only the main page in the list.
        let pages = interp.get_nearby_prefetch_list(0x10000, 4);
        assert_eq!(pages, vec![0x10000]);
    }
}
