use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs blocks in a loop, using Python callbacks for memory access,
    /// until it hits a condition that requires Python-side handling.
    ///
    /// Returns a tuple of (result, blocks_executed, deferred_forks).
    pub fn run_until_event(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        max_blocks: u32,
    ) -> (RunResult, u32, Vec<DeferredFork>) {
        let total_start = profile_start!(self);
        let mut blocks_executed = 0u32;

        // Sort concrete memory regions for binary search if needed
        self.sort_concrete_memory();

        // Clear any previous deferred forks and reset per-step limit
        self.deferred_forks.clear();
        self.deferred_fork_this_step = false;

        for _ in 0..max_blocks {
            // Check for hook at current PC
            if self.is_hooked(self.pc) {
                let forks = self.take_deferred_forks();
                // Check if this is a registered SimProcedure with known args
                if let Some(info) = self.simprocedure_registry.get(&self.pc).cloned() {
                    // Get return address if available
                    let return_addr = self.get_return_addr().unwrap_or(0);
                    return (
                        RunResult::SimProcedure {
                            addr: self.pc,
                            name: info.name,
                            num_args: info.num_args,
                            return_addr,
                        },
                        blocks_executed,
                        forks,
                    );
                }
                // Fall back to generic Hook for unregistered hooks
                return (RunResult::Hook { addr: self.pc }, blocks_executed, forks);
            }

            // Check if we've hit the deferred forks limit
            if self.config.use_deferred_forks
                && self.deferred_forks.len() >= self.config.max_deferred_forks as usize
            {
                let forks = self.take_deferred_forks();
                return (
                    RunResult::MaxDeferredForks { pc: self.pc },
                    blocks_executed,
                    forks,
                );
            }

            // Try to get or lift the block
            let irsb = match self.get_or_lift_block(py, callbacks, self.pc) {
                Ok(irsb) => irsb,
                Err(e) => {
                    let forks = self.take_deferred_forks();
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        blocks_executed,
                        forks,
                    );
                }
            };

            // Execute the block
            let block_start = profile_start!(self);
            match self.execute_block_with_callbacks(py, callbacks, &irsb) {
                Ok(result) => {
                    blocks_executed += 1;
                    profile_add!(block_start, self.stats.block_exec_time_ns);

                    match result {
                        BlockResult::Continue { next_addr } => {
                            self.pc = next_addr;
                            // Continue to next block
                        }
                        BlockResult::BlockEnd {
                            next_addr,
                            jumpkind,
                        } => {
                            // Track call stack before updating PC
                            if jumpkind.is_call() {
                                let sp_val = self
                                    .registers
                                    .get(
                                        self.registers.arch().sp_offset(),
                                        self.calling_convention.pointer_size(),
                                        self.ctx,
                                    )
                                    .as_u64()
                                    .unwrap_or(0);
                                let ret_addr = self.get_return_addr().unwrap_or(0);
                                self.call_stack.push(crate::state::CallStackEntry {
                                    call_site_addr: self.current_insn_addr,
                                    callee_addr: next_addr,
                                    return_addr: ret_addr,
                                    stack_ptr: sp_val,
                                });
                            } else if jumpkind.is_ret() {
                                self.call_stack.pop();
                            }

                            // Record detailed history entry
                            self.detailed_history.push(crate::state::HistoryEntry {
                                addr: self.pc, // block that just executed
                                jumpkind: crate::state::HistoryEntry::jumpkind_from_vex(&jumpkind),
                                jump_target: next_addr,
                            });

                            self.pc = next_addr;
                            // Return for jumpkinds that need Python handling
                            if jumpkind.is_syscall() {
                                let syscall_num = self.get_syscall_num();
                                let forks = self.take_deferred_forks();
                                return (
                                    RunResult::Syscall {
                                        num: syscall_num,
                                        pc: next_addr,
                                    },
                                    blocks_executed,
                                    forks,
                                );
                            }
                            // For Call/Ret, we might want to return for SimProcedures
                            if self.is_hooked(next_addr) {
                                let forks = self.take_deferred_forks();
                                // Update PC before extracting args (so SP/ret addr are correct)
                                self.pc = next_addr;
                                // Check if this is a registered SimProcedure
                                if let Some(info) =
                                    self.simprocedure_registry.get(&next_addr).cloned()
                                {
                                    let return_addr = self.get_return_addr().unwrap_or(0);
                                    return (
                                        RunResult::SimProcedure {
                                            addr: next_addr,
                                            name: info.name,
                                            num_args: info.num_args,
                                            return_addr,
                                        },
                                        blocks_executed,
                                        forks,
                                    );
                                }
                                return (
                                    RunResult::Hook { addr: next_addr },
                                    blocks_executed,
                                    forks,
                                );
                            }
                            // Otherwise, continue execution
                        }
                        BlockResult::Syscall { num } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::Syscall { num, pc: self.pc },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::SymbolicBranch {
                            condition_id,
                            true_target,
                            false_target,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::SymbolicBranch {
                                    condition_id,
                                    true_target,
                                    false_target,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::Hook { addr } => {
                            let forks = self.take_deferred_forks();
                            // Check if this is a registered SimProcedure
                            if let Some(info) = self.simprocedure_registry.get(&addr).cloned() {
                                let return_addr = self.get_return_addr().unwrap_or(0);
                                return (
                                    RunResult::SimProcedure {
                                        addr,
                                        name: info.name,
                                        num_args: info.num_args,
                                        return_addr,
                                    },
                                    blocks_executed,
                                    forks,
                                );
                            }
                            return (RunResult::Hook { addr }, blocks_executed, forks);
                        }
                        BlockResult::Error { message } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::Error {
                                    message,
                                    addr: self.pc,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::SymbolicJumpTarget {
                            targets,
                            condition_id,
                            target_expr: _,
                            jumpkind,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::SymbolicJumpTarget {
                                    targets,
                                    condition_id,
                                    jumpkind: format!("{:?}", jumpkind),
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::UnconstrainedJump {
                            min_target,
                            max_target,
                            limit,
                            jumpkind,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::UnconstrainedJump {
                                    min_target,
                                    max_target,
                                    limit,
                                    jumpkind: format!("{:?}", jumpkind),
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::UnmodeledCall {
                            addr,
                            return_addr,
                            symbol_name,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::UnmodeledCall {
                                    addr,
                                    return_addr,
                                    symbol_name,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                    }
                }
                Err(e) => {
                    let forks = self.take_deferred_forks();
                    // Dispatch via the variant's declared FallbackStrategy.
                    // Adding a CbExecutionError variant requires choosing a
                    // strategy in mod.rs; that decision lands here.
                    let result = match e.strategy() {
                        FallbackStrategy::PythonCallback => RunResult::NeedPythonVEX {
                            addr: self.pc,
                            reason: e.to_string(),
                        },
                        FallbackStrategy::Panic => RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                    };
                    return (result, blocks_executed, forks);
                }
            }
        }

        // Reached max blocks
        let forks = self.take_deferred_forks();
        if self.profiling_enabled {
            self.stats.blocks_executed += blocks_executed as u64;
        }
        profile_add!(total_start, self.stats.total_time_ns);
        (RunResult::MaxBlocks { pc: self.pc }, blocks_executed, forks)
    }

    /// Get or lift a block at the given address.
    ///
    /// This tries the following in order:
    /// 1. Check the block cache
    /// 2. Try native lifting via libpyvex (if feature enabled and bytes available)
    /// 3. Fall back to Python callback for lifting
    fn get_or_lift_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr: u64,
    ) -> Result<Arc<IRSB>, CbExecutionError> {
        // Enforce execute permission before any cache hit / lift work so a
        // page whose X bit was stripped after the IRSB was first cached
        // still rejects on re-entry. No-op when enforce_permissions is off
        // or the page is unmapped (existing lift paths handle that).
        if let Some(ref mem) = self.rust_memory {
            mem.check_executable(addr)
                .map_err(|e| CbExecutionError::Memory(e.to_string()))?;
        }

        // Check cache first - Arc clone is O(1)
        if let Some(irsb) = self.block_cache.get(&addr) {
            if self.profiling_enabled {
                self.stats.cache_hit_count += 1;
            }
            return Ok(Arc::clone(irsb));
        }

        if self.profiling_enabled {
            self.stats.cache_miss_count += 1;
        }

        let lift_start = profile_start!(self);

        // Try native lifting if available
        #[cfg(feature = "native-lift")]
        {
            if crate::vex::libpyvex_ffi::is_vex_initialized() {
                // SMC fast path: when the page has been overwritten via a
                // store, `concrete_memory` is stale. Try to read fresh bytes
                // from `rust_memory` (which carries the post-write state)
                // and lift those instead. Falls through to the Python lift
                // path only if rust_memory has no concrete bytes available.
                if self.is_code_range_dirtied(addr, 4096) {
                    if let Some(rust_mem) = self.rust_memory.as_ref() {
                        let mut max_bytes = 4096usize;
                        for &hook_addr in self.hook_addrs.iter() {
                            if hook_addr > addr && hook_addr < addr + max_bytes as u64 {
                                let limit = (hook_addr - addr) as usize;
                                if limit > 0 && limit < max_bytes {
                                    max_bytes = limit;
                                }
                            }
                        }
                        if let Some(bytes) = rust_mem.read_concrete_bytes_for_lift(addr, max_bytes)
                        {
                            let native_opt_level = self
                                .vex_opt_level_overrides
                                .get(&addr)
                                .copied()
                                .or(self.vex_opt_level)
                                .unwrap_or(1);
                            match crate::vex::libpyvex_ffi::lift_native(
                                &bytes,
                                addr,
                                self.arch,
                                99,
                                bytes.len() as u32,
                                native_opt_level,
                            ) {
                                Ok(irsb) => {
                                    log::trace!(
                                        "Native lift from rust_memory (SMC) at 0x{:x}",
                                        addr
                                    );
                                    profile_add!(lift_start, self.stats.lift_time_ns);
                                    let arc_irsb = Arc::new(irsb);
                                    self.block_cache.put(addr, Arc::clone(&arc_irsb));
                                    return Ok(arc_irsb);
                                }
                                Err(_e) => {
                                    // Fall through to Python lift.
                                }
                            }
                        }
                    }
                    log::trace!(
                        "Native lift skipped at 0x{:x}: page dirtied by SMC, falling back to Python",
                        addr
                    );
                } else {
                    // Try to get bytes from concrete memory for native lifting
                    // Look for a region containing this address with enough bytes
                    for region in self.concrete_memory.iter() {
                        if addr >= region.base && addr < region.base + region.size {
                            let offset = (addr - region.base) as usize;
                            let available = region.size as usize - offset;
                            // Limit block size to stop at hook/avoid/find addresses.
                            // Without this, blocks can span past these addresses,
                            // and the hook check at block boundaries misses them.
                            let mut max_bytes = available.min(4096);
                            for &hook_addr in self.hook_addrs.iter() {
                                if hook_addr > addr && hook_addr < addr + max_bytes as u64 {
                                    let limit = (hook_addr - addr) as usize;
                                    if limit > 0 && limit < max_bytes {
                                        max_bytes = limit;
                                    }
                                }
                            }
                            if max_bytes >= 1 {
                                let bytes = &region.data[offset..offset + max_bytes];
                                // Resolve VEX opt_level for native lifting
                                let native_opt_level = self
                                    .vex_opt_level_overrides
                                    .get(&addr)
                                    .copied()
                                    .or(self.vex_opt_level)
                                    .unwrap_or(1); // pyvex default is 1
                                match crate::vex::libpyvex_ffi::lift_native(
                                    bytes,
                                    addr,
                                    self.arch,
                                    99, // max_insns
                                    max_bytes as u32,
                                    native_opt_level,
                                ) {
                                    Ok(irsb) => {
                                        // Native lift succeeded!
                                        log::trace!("Native lift succeeded at 0x{:x}", addr);
                                        profile_add!(lift_start, self.stats.lift_time_ns);
                                        let arc_irsb = Arc::new(irsb);
                                        self.block_cache.put(addr, Arc::clone(&arc_irsb));
                                        return Ok(arc_irsb);
                                    }
                                    Err(e) => {
                                        log::trace!("Native lift failed at 0x{:x}: {}", addr, e);
                                        // Fall through to Python callback
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Fall back to lifting via Python callback
        let callback_start = profile_start!(self);
        // Resolve VEX opt_level: per-address override > global > None (pyvex default)
        let opt_level = self
            .vex_opt_level_overrides
            .get(&addr)
            .copied()
            .or(self.vex_opt_level);

        // SMC: when this lift range overlaps a dirtied page, the cle binary
        // bytes that the Python lifter would normally read are stale. Try
        // to read fresh bytes from rust_memory and pass them via byte_string=
        // so the Python lift sees the post-store program.
        let dirty_bytes: Option<Vec<u8>> = if self.is_code_range_dirtied(addr, 4096) {
            self.rust_memory.as_ref().and_then(|rust_mem| {
                let mut max_bytes = 4096usize;
                for &hook_addr in self.hook_addrs.iter() {
                    if hook_addr > addr && hook_addr < addr + max_bytes as u64 {
                        let limit = (hook_addr - addr) as usize;
                        if limit > 0 && limit < max_bytes {
                            max_bytes = limit;
                        }
                    }
                }
                rust_mem.read_concrete_bytes_for_lift(addr, max_bytes)
            })
        } else {
            None
        };

        let irsb_json = callbacks
            .call_lift_block(py, addr, opt_level, dirty_bytes.as_deref())
            .map_err(|e| CbExecutionError::LiftError(format!("lift callback failed: {}", e)))?;
        if self.profiling_enabled {
            self.stats.python_callback_count += 1;
        }
        profile_add!(callback_start, self.stats.python_callback_time_ns);

        let irsb = deserialize_irsb(&irsb_json).map_err(|e| {
            CbExecutionError::LiftError(format!("IRSB deserialization failed: {}", e))
        })?;

        profile_add!(lift_start, self.stats.lift_time_ns);

        // Cache it - Arc allows O(1) cloning
        let arc_irsb = Arc::new(irsb);
        self.block_cache.put(addr, Arc::clone(&arc_irsb));

        Ok(arc_irsb)
    }

    /// Execute a single block using the given callbacks.
    ///
    /// Public entry point used by the [`crate::engine::execute_irsb_for_test`]
    /// helper to drive the interpreter from unit tests without a full
    /// `run_until_event` loop. The standard exploration path still goes
    /// through [`Self::run_until_event`] which calls
    /// `execute_block_with_callbacks` internally.
    pub fn execute_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        self.execute_block_with_callbacks(py, callbacks, irsb)
    }

    /// Execute a block using Python callbacks for memory access.
    fn execute_block_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        // Reset incremental branch solver state for this block
        self.block_solver_pushed = false;
        self.block_forks_asserted = 0;

        // Clear per-block concretization cache (constraints don't change within a block)
        self.concretize_cache.clear();

        // Reset temps for this block - reuse allocation instead of creating new vec
        let needed_temps = irsb.tyenv.types.len();
        self.temps.clear();
        if self.temps.capacity() < needed_temps {
            self.temps.reserve(needed_temps - self.temps.capacity());
        }
        self.temps.resize(needed_temps, None);
        self.current_insn_addr = irsb.addr;

        // Prefetch loads for this block (reduces individual FFI calls)
        let prefetch_start = profile_start!(self);
        self.prefetch_loads_for_block(py, callbacks, irsb)?;
        profile_add!(prefetch_start, self.stats.prefetch_time_ns);

        // state.inspect irsb event — fires `when='before'` at block entry,
        // before any statement runs. Bit 7 in the inspect-enabled bitmask.
        if callbacks.inspect_event_enabled(7) {
            let _ = callbacks.call_inspect_irsb(py, self.current_state_id, "before", irsb.addr);
        }

        // Execute statements
        let mut stmt_total_ns: u64 = 0;
        for stmt in &irsb.statements {
            if self.profiling_enabled {
                self.stats.stmt_count += 1;
            }
            let stmt_start = profile_start!(self);
            match self.execute_stmt_with_callbacks(py, callbacks, stmt, irsb)? {
                StmtResult::Continue => {
                    if let Some(start) = stmt_start {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        stmt_total_ns += elapsed;
                        #[cfg(debug_assertions)]
                        if elapsed > 50_000_000 {
                            // >50ms
                            log::warn!(
                                "  SLOW STMT at 0x{:x}: {}ms {:?}",
                                self.current_insn_addr,
                                elapsed / 1_000_000,
                                stmt
                            );
                        }
                    }
                    continue;
                }
                StmtResult::Exit { target, jumpkind } => {
                    // Flush pending stores before returning
                    self.flush_stores(py, callbacks)?;
                    self.pop_block_solver_if_pushed();
                    return Ok(self.handle_exit(target, jumpkind));
                }
                StmtResult::SymbolicBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    // Flush pending stores before returning
                    self.flush_stores(py, callbacks)?;
                    self.pop_block_solver_if_pushed();

                    // Generate unique condition ID and store condition for later retrieval
                    let cond_id = self.next_cond_id();
                    self.stored_conditions.insert(cond_id, condition.clone());

                    // Also store for callers using take_last_branch_condition
                    self.last_branch_condition = Some(condition);

                    return Ok(BlockResult::SymbolicBranch {
                        condition_id: cond_id, // Fixed: use proper unique ID
                        true_target,
                        false_target,
                    });
                }
            }
        }

        if self.profiling_enabled {
            self.stats.run_loop_time_ns += stmt_total_ns;
            #[cfg(debug_assertions)]
            if stmt_total_ns > 100_000_000 {
                // >100ms
                log::warn!(
                    "SLOW BLOCK at 0x{:x}: {}ms for {} stmts",
                    irsb.addr,
                    stmt_total_ns / 1_000_000,
                    irsb.statements.len()
                );
            }
        }

        // Flush pending stores at block end
        self.flush_stores(py, callbacks)?;

        // Pop incremental branch solver context if pushed
        self.pop_block_solver_if_pushed();

        // Handle default exit
        self.handle_default_exit(irsb)
    }

    /// Pop the solver if we pushed for incremental branch constraint tracking.
    #[inline]
    fn pop_block_solver_if_pushed(&mut self) {
        if self.block_solver_pushed {
            self.ctx.pop();
            self.block_solver_pushed = false;
            self.block_forks_asserted = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
        VEXInterpreter::new(VexArch::AMD64, ctx)
    }

    fn make_irsb(addr: u64, len_bytes: u32) -> IRSB {
        let mut irsb = IRSB::new(addr, VexArch::AMD64);
        irsb.statements.push(IRStmt::IMark {
            addr,
            len: len_bytes,
            delta: 0,
        });
        irsb
    }

    #[test]
    fn sort_concrete_memory_orders_by_base() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.add_concrete_memory(0x3000, vec![0u8; 0x10]);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x10]);
        interp.add_concrete_memory(0x2000, vec![0u8; 0x10]);
        assert!(!interp.concrete_memory_sorted);
        interp.sort_concrete_memory();
        assert!(interp.concrete_memory_sorted);
        let bases: Vec<u64> = interp.concrete_memory.iter().map(|r| r.base).collect();
        assert_eq!(bases, vec![0x1000, 0x2000, 0x3000]);
    }

    #[test]
    fn sort_concrete_memory_is_idempotent() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x10]);
        interp.add_concrete_memory(0x2000, vec![0u8; 0x10]);
        interp.sort_concrete_memory();
        // Second call is a no-op (already sorted) but should not panic or change order.
        interp.sort_concrete_memory();
        assert_eq!(interp.concrete_memory[0].base, 0x1000);
        assert_eq!(interp.concrete_memory[1].base, 0x2000);
    }

    #[test]
    fn sort_concrete_memory_noop_below_two_regions() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.add_concrete_memory(0x4000, vec![0u8; 0x10]);
        interp.sort_concrete_memory();
        // Single-region case does not flip the sorted flag.
        assert!(!interp.concrete_memory_sorted);
    }

    #[test]
    fn block_cache_round_trip() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb(0x4000, 4);
        assert!(!interp.has_cached_block(0x4000));
        interp.cache_block(0x4000, irsb);
        assert!(interp.has_cached_block(0x4000));
        let got = interp.get_cached_block(0x4000).expect("cached IRSB");
        assert_eq!(got.addr, 0x4000);
    }

    #[test]
    fn pop_block_solver_if_pushed_is_no_op_when_not_pushed() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        assert!(!interp.block_solver_pushed);
        // Should not panic / not double-pop.
        interp.pop_block_solver_if_pushed();
        assert!(!interp.block_solver_pushed);
    }

    #[test]
    fn next_cond_id_is_monotonic() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let a = interp.next_cond_id();
        let b = interp.next_cond_id();
        let c = interp.next_cond_id();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
    }

    #[test]
    fn deferred_forks_take_clears_state() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        assert_eq!(interp.num_deferred_forks(), 0);
        // The take_/clear_ APIs should not panic on empty state.
        let taken = interp.take_deferred_forks();
        assert!(taken.is_empty());
        interp.clear_deferred_forks();
        assert_eq!(interp.num_deferred_forks(), 0);
    }

    #[test]
    fn stored_conditions_take_and_lookup() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // Insert a stored condition via the private map; verify take/get behaviour.
        let cond = RustBV::symbolic(&ctx, "cond", 1);
        interp.stored_conditions.insert(42, cond.clone());
        assert!(interp.get_stored_condition(42).is_some());
        let taken = interp.take_stored_conditions();
        assert_eq!(taken.len(), 1);
        // After take, the map should be empty.
        assert!(interp.get_stored_condition(42).is_none());
    }

    #[test]
    fn take_last_branch_condition_returns_none_initially() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        assert!(interp.take_last_branch_condition().is_none());
    }

    #[test]
    fn take_last_branch_condition_returns_then_clears() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let cond = RustBV::symbolic(&ctx, "b", 1);
        interp.last_branch_condition = Some(cond);
        assert!(interp.take_last_branch_condition().is_some());
        assert!(interp.take_last_branch_condition().is_none());
    }

    #[test]
    fn swap_block_cache_exchanges_caches() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.cache_block(0x4000, make_irsb(0x4000, 4));
        // Swap in an empty cache — should get back the populated one.
        let fresh = LruCache::new(NonZeroUsize::new(4096).expect("nonzero"));
        let old = interp.swap_block_cache(fresh);
        assert!(old.contains(&0x4000));
        assert!(!interp.has_cached_block(0x4000));
    }
}
