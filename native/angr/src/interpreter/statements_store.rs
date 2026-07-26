use super::helpers::bv_to_bytes;
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Attempt to store via Rust-native memory. Returns Ok(true) if the store
    /// was handled, Ok(false) if the caller should fall back to the Python path.
    pub(super) fn try_rust_memory_store(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
        store_start: Option<Instant>,
    ) -> Result<bool, CbExecutionError> {
        // AVOID_MULTIVALUED_WRITES: silently drop symbolic-addr stores.
        // Mirrors `address_concretization_mixin.py:327-329`.
        if self.concretizer.should_avoid_multivalued_write(addr_val) {
            profile_add!(store_start, self.stats.store_stmt_time_ns);
            return Ok(true);
        }
        // angr-vfst: address_concretization BP_BEFORE for the store path. Only
        // dispatches when addr is symbolic (concrete-addr stores have nothing
        // to concretize). Gated on bit 17 inside the helper.
        if !addr_val.is_concrete() {
            self.dispatch_address_concretization_inspect(
                callbacks, addr_val, "store", "before", None,
            );
        }
        // Concretize for write with per-block cache (avoids redundant Z3 calls)
        let conc_result = self.concretize_cached_write(addr_val);
        if !addr_val.is_concrete() {
            let result_addrs = conc_result.addresses();
            self.dispatch_address_concretization_inspect(
                callbacks,
                addr_val,
                "store",
                "after",
                result_addrs,
            );
        }

        // Attempt store using pre-computed concretization
        let first_result = match self.rust_memory.as_mut() {
            Some(rust_mem) => rust_mem.store_with_concretization(
                addr_val,
                data_val.clone(),
                &conc_result,
                self.ctx,
            ),
            None => return Ok(false),
        };

        match first_result {
            Ok(()) => {
                self.invalidate_code_on_store(&conc_result, data_size);
                // Rust owns memory — no need to sync stores to Python.
                profile_add!(store_start, self.stats.store_stmt_time_ns);
                Ok(true)
            }
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                let prefetch_count = self.page_prefetch_count;
                let page_fetched =
                    self.fetch_page_with_prefetch(callbacks, page_addr, prefetch_count)?;

                if page_fetched {
                    // Page was fetched - retry store using cached concretization
                    if let Some(ref mut rust_mem) = self.rust_memory {
                        match rust_mem.store_with_concretization(
                            addr_val,
                            data_val.clone(),
                            &conc_result,
                            self.ctx,
                        ) {
                            Ok(()) => {
                                self.invalidate_code_on_store(&conc_result, data_size);
                                return Ok(true);
                            }
                            Err(_e) => {
                                // Still failed - fall through to Python callback
                            }
                        }
                    }
                }
                Ok(false)
            }
            Err(MemoryError::Unmapped {
                addr,
                size: unmapped_size,
            }) => {
                log::debug!(
                    "Unmapped memory store at 0x{addr:x} (size={unmapped_size}), falling back to Python"
                );
                Ok(false)
            }
            Err(MemoryError::SymbolicAddress { description }) => {
                // Address range too large or symbolic — Python's memory model handles natively
                log::debug!("Symbolic address store: {description}, falling back to Python");
                Ok(false)
            }
            Err(e) => Err(CbExecutionError::Memory(e)),
        }
    }

    /// Invalidate cached IRSBs after a successful Rust-native store when the
    /// store hits a loaded binary region (self-modifying code support).
    /// Single-address writes invalidate the overlapping block; other
    /// concretization shapes defer to `invalidate_code_for_concretization`.
    pub(super) fn invalidate_code_on_store(
        &mut self,
        conc_result: &ConcretizationResult,
        data_size: usize,
    ) {
        match conc_result {
            ConcretizationResult::Single(addr_concrete) => {
                if self.is_in_binary(*addr_concrete) {
                    self.invalidate_code_at(*addr_concrete, data_size);
                }
            }
            _ => {
                self.invalidate_code_for_concretization(conc_result, data_size);
            }
        }
    }

    /// Invalidate cached IRSBs for a multi-address concretization result whose
    /// solutions hit loaded binary regions. Conservative: any in-binary
    /// solution triggers a per-address invalidation; ranges/Any clear the
    /// cache outright since the affected bytes are unbounded.
    pub(super) fn invalidate_code_for_concretization(
        &mut self,
        conc_result: &ConcretizationResult,
        data_size: usize,
    ) {
        match conc_result {
            ConcretizationResult::Single(_) => {} // handled by caller
            ConcretizationResult::Multiple(addrs) => {
                for &addr in addrs.iter() {
                    if self.is_in_binary(addr) {
                        self.invalidate_code_at(addr, data_size);
                    }
                }
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                for i in 0..*count {
                    let addr = base.saturating_add(i.saturating_mul(*stride));
                    if self.is_in_binary(addr) {
                        self.invalidate_code_at(addr, data_size);
                    }
                }
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Range too large to enumerate. If the [min, max] range
                // intersects any loaded binary region, drop the entire
                // block cache to be safe.
                let intersects_binary = self.concrete_memory.iter().any(|region| {
                    let region_end = region.base + region.size;
                    *min < region_end && *max >= region.base
                });
                if intersects_binary {
                    self.block_cache.clear();
                    // Mark all binary pages as dirtied so native lift skips
                    // them until they're re-lifted via Python.
                    let pages: Vec<u64> = self
                        .concrete_memory
                        .iter()
                        .flat_map(|region| {
                            let first = region.base >> 12;
                            let last = (region.base + region.size - 1) >> 12;
                            first..=last
                        })
                        .collect();
                    for page in pages {
                        self.dirtied_code_pages.insert(page);
                    }
                }
            }
            ConcretizationResult::Failed(_) => {
                // Concretization failed; addresses are unknown. Be safe.
                self.block_cache.clear();
            }
        }
    }

    /// Fall back to the Python callback path for a store. Splits on
    /// concrete-vs-symbolic address; the concrete branch invalidates cached
    /// code then dispatches via `handle_concrete_store`, the symbolic
    /// branch goes through `handle_symbolic_store`.
    pub(super) fn fallback_to_python_store(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        if let Some(addr_concrete) = addr_val.as_u64() {
            self.invalidate_code_at_store(addr_concrete, data_size);
            self.handle_concrete_store(callbacks, addr_concrete, data_val, data_size)
        } else {
            self.handle_symbolic_store(callbacks, addr_val, &data_val, data_size)
        }
    }

    /// Drop any cached IRSB whose bytes overlap the store (self-modifying
    /// code support). Concrete-address stores only.
    pub(super) fn invalidate_code_at_store(&mut self, addr: u64, data_size: usize) {
        if self.is_in_binary(addr) {
            self.invalidate_code_at(addr, data_size);
        }
    }

    /// Concrete-address store path: chooses between the
    /// `memory_store_symbolic_value` callback (32-bit non-stack only) and
    /// the buffered `pending_stores` fast path. The 32-bit heuristic exists
    /// to keep flareon2015_5 working while avoiding the false-positive cost
    /// of routing every 64-bit symbolic store through Python.
    pub(super) fn handle_concrete_store(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        data_val: RustBV,
        _data_size: usize,
    ) -> Result<(), CbExecutionError> {
        let use_sym_store = if self.arch.pointer_size() == 32 {
            let is_stack = self.registers.get_sp_value().is_some_and(|sp_val| {
                // Non-wrapping distance check
                let dist = addr_concrete.abs_diff(sp_val);
                dist <= 0x10000
            });
            !is_stack
        } else {
            false // Skip for 64-bit — too expensive
        };

        if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() && use_sym_store {
            // Try symbolic store callback (preserves expression tree)
            self.buffer_store_for_rust_memory(callbacks, addr_concrete, &data_val);
            let sym_ok = (|| -> Result<(), CbExecutionError> {
                self.flush_stores(callbacks)?;
                callbacks
                    .call_memory_store_symbolic_value(addr_concrete, &data_val)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))
            })();
            if sym_ok.is_err() {
                // Symbolic store callback failed — evaluate to concrete and
                // store directly via Python callback (not pending_stores).
                // pending_stores would pollute all_flushed_stores with zeros
                // since bv_to_bytes returns zeros for symbolic expressions.
                let concrete_val = self.ctx.eval(&data_val).unwrap_or(0);
                let size_bytes = (data_val.width() / 8) as usize;
                let mut data_bytes = vec![0u8; size_bytes];
                for (i, b) in data_bytes.iter_mut().enumerate() {
                    *b = (concrete_val >> (i * 8)) as u8;
                }
                let _ = callbacks.call_memory_store(addr_concrete, &data_bytes);
            }
            Ok(())
        } else if data_val.is_symbolic() {
            // Symbolic fast path: record the value for exact + overlap load
            // forwarding. Do NOT push zero placeholder bytes into
            // pending_stores — bv_to_bytes yields all-zeros for symbolic data
            // (see the comment in store_concrete), which would make an offset
            // load inside the store return concrete 0 (angr-ofyh). The symbolic
            // maps (incl. overlap) are consulted before the concrete buffer on
            // load.
            self.pending_symbolic_stores.insert(addr_concrete, data_val);
            if self.pending_symbolic_stores.len() >= self.max_pending_stores {
                self.flush_stores(callbacks)?;
            }
            Ok(())
        } else {
            // Concrete fast path: evict any stale symbolic shadow this store
            // overwrites (angr-ofyh) so a later load can't return the old
            // symbolic value, then buffer the bytes for batch processing.
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr_concrete, size_bytes);
            let data_bytes = bv_to_bytes(&data_val);
            self.pending_stores.push(addr_concrete, data_bytes);

            if self.pending_stores.len() >= self.max_pending_stores {
                self.flush_stores(callbacks)?;
            }
            Ok(())
        }
    }

    /// Buffer a store that is *also* being dispatched to a Python memory
    /// callback, for the case where that callback cannot absorb it.
    ///
    /// Under the callback-memory-proxy gate the `memory_store` /
    /// `memory_store_symbolic_value` callbacks are deliberate no-ops
    /// (`state.memory` *is* Rust memory there, and re-entering `run()` would
    /// double-borrow), so a callback-only store is dropped outright:
    /// flareon2015_5's base64 encoder wrote its output through the
    /// symbolic-address path and the found state read back concrete zeros
    /// (angr-5rjbq). Buffering it here gives the value a home in
    /// `rust_memory`.
    ///
    /// Ungated, the Python shadow really does absorb the store, so this extra
    /// write would be pure cost — four fast-tier benches regressed 15-25% when
    /// it ran unconditionally. Hence the `memory_is_rust_proxy` guard.
    pub(super) fn buffer_store_for_rust_memory(
        &mut self,
        callbacks: &PythonCallbacks,
        addr: u64,
        data_val: &RustBV,
    ) {
        if !self.use_rust_memory || !callbacks.memory_is_rust_proxy() {
            return;
        }
        if data_val.is_symbolic() {
            self.pending_symbolic_stores.insert(addr, data_val.clone());
        } else {
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr, size_bytes);
            self.pending_stores.push(addr, bv_to_bytes(data_val));
        }
    }

    /// Remove symbolic store entries (pending and flushed) whose byte range
    /// overlaps `[addr, addr + size)`. A concrete store overwriting (part of) a
    /// prior symbolic store must drop the symbolic shadow, otherwise a later
    /// load at the same address would return the stale symbolic value instead
    /// of the concrete bytes (angr-ofyh). Short-circuits when no symbolic
    /// stores are buffered, so the all-concrete hot path pays nothing.
    pub(super) fn evict_overlapping_symbolic_stores(&mut self, addr: u64, size: usize) {
        if self.pending_symbolic_stores.is_empty() && self.all_flushed_symbolic_stores.is_empty() {
            return;
        }
        let lo = addr;
        let hi = addr.saturating_add(size as u64);
        let overlaps = |s_addr: u64, bv: &RustBV| {
            let s_hi = s_addr.saturating_add((bv.width() / 8) as u64);
            s_addr < hi && lo < s_hi
        };
        self.pending_symbolic_stores
            .retain(|&s_addr, bv| !overlaps(s_addr, bv));
        self.all_flushed_symbolic_stores
            .retain(|&s_addr, bv| !overlaps(s_addr, bv));
    }

    /// Symbolic-address store path. Clears the prefetch cache, flushes the
    /// pending-store buffer, then dispatches on the 5 ConcretizationResult
    /// shapes returned by `concretize_cached_write`. Single → direct callback;
    /// Multiple/Strided → `dispatch_multi_store`; TooLarge → full symbolic
    /// callback or Unsupported; Failed → `fallback_store_symbolic_full`.
    ///
    /// Invalidates any stale cached IRSB at the concretized target(s) before
    /// dispatching the store (self-modifying-code support) via
    /// `invalidate_code_on_store` — the same dispatcher `try_rust_memory_store`
    /// and `cas_store_symbolic_data` use, so Multiple/Strided/TooLarge/Failed
    /// get the same per-address (or defensive block-cache-clear) treatment as
    /// the Single case, instead of only Single being covered (angr-srk4b).
    pub(super) fn handle_symbolic_store(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        // AVOID_MULTIVALUED_WRITES: silently drop. Do not touch pending
        // stores — the address was never resolved, so no mutation propagates.
        if self.concretizer.should_avoid_multivalued_write(addr_val) {
            return Ok(());
        }
        // Touched addresses are unknown — flush pending stores before Python
        // sees the symbolic write.
        self.flush_stores(callbacks)?;

        let concret_result = self.concretize_cached_write(addr_val);
        // Invalidate before dispatching the store so Python (or the in-Rust
        // ITE path) never observes a write that landed without evicting the
        // stale lifted block first — mirrors the Single-arm ordering below.
        self.invalidate_code_on_store(&concret_result, data_size);
        match &*concret_result {
            ConcretizationResult::Single(addr_concrete) => {
                let addr_concrete = *addr_concrete;
                self.buffer_store_for_rust_memory(callbacks, addr_concrete, data_val);
                if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                    callbacks
                        .call_memory_store_symbolic_value(addr_concrete, data_val)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                } else {
                    let data_bytes = bv_to_bytes(data_val);
                    callbacks
                        .call_memory_store(addr_concrete, &data_bytes)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                }
                Ok(())
            }
            ConcretizationResult::Multiple(addrs) => {
                self.dispatch_multi_store(callbacks, addrs, addr_val, data_val)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let addrs: Vec<u64> = (0..*count).map(|i| base + i * stride).collect();
                self.dispatch_multi_store(callbacks, &addrs, addr_val, data_val)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let (min, max) = (*min, *max);
                // Range too large to enumerate — delegate to Python's memory
                // model (which has access to angr's address concretization
                // strategies) via the full symbolic callback.
                if callbacks.has_memory_store_symbolic_full() {
                    callbacks
                        .call_memory_store_symbolic_full(addr_val, data_val)
                        .map_err(|e| {
                            CbExecutionError::Callback(format!(
                                "symbolic store full callback failed at 0x{min:x}-0x{max:x}: {e}"
                            ))
                        })?;
                    Ok(())
                } else {
                    Err(CbExecutionError::Unsupported(format!(
                        "symbolic store with too-large address range 0x{min:x}-0x{max:x}: \
                         no memory_store_symbolic_full callback"
                    )))
                }
            }
            ConcretizationResult::Failed(reason) => {
                // Concretization failed entirely. Try the full symbolic
                // store callback so Python's memory model can still resolve
                // the address; only error out if the callback isn't wired up.
                let descr = format!("concretize failed: {reason}");
                self.fallback_store_symbolic_full(callbacks, addr_val, data_val, "store", &descr)
            }
        }
    }

    /// Shared dispatch for Multiple/Strided concretization results: build an
    /// in-Rust ITE chain when ≤16 addrs and the symbolic-value callback is
    /// available, otherwise hand the full address list to Python.
    pub(super) fn dispatch_multi_store(
        &self,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_val: &RustBV,
        data_val: &RustBV,
    ) -> Result<(), CbExecutionError> {
        if addrs.len() <= 16 && callbacks.has_memory_store_symbolic_value() {
            self.build_ite_store_from_callbacks(callbacks, addrs, addr_val, data_val)
        } else {
            callbacks
                .call_memory_store_symbolic(addrs, data_val, addr_val)
                .map_err(|e| CbExecutionError::Callback(e.to_string()))
        }
    }
}
