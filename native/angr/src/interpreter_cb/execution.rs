use super::*;

impl<'a> CallbackInterpreter<'a> {
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
        let total_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
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
                return (RunResult::MaxDeferredForks { pc: self.pc }, blocks_executed, forks);
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
            let block_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
            match self.execute_block_with_callbacks(py, callbacks, &irsb) {
                Ok(result) => {
                    blocks_executed += 1;
                    if let Some(start) = block_start {
                        self.stats.block_exec_time_ns += start.elapsed().as_nanos() as u64;
                    }

                    match result {
                        BlockResult::Continue { next_addr } => {
                            self.pc = next_addr;
                            // Continue to next block
                        }
                        BlockResult::BlockEnd { next_addr, jumpkind } => {
                            // Track call stack before updating PC
                            if jumpkind.is_call() {
                                let sp_val = self.registers.get(
                                    self.registers.arch().sp_offset(),
                                    self.calling_convention.pointer_size(),
                                    self.ctx,
                                ).as_u64().unwrap_or(0);
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
                                if let Some(info) = self.simprocedure_registry.get(&next_addr).cloned() {
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
                                return (RunResult::Hook { addr: next_addr }, blocks_executed, forks);
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
                        FallbackStrategy::PythonCallback => {
                            RunResult::NeedPythonVEX { addr: self.pc, reason: e.to_string() }
                        }
                        FallbackStrategy::Panic => RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        FallbackStrategy::Silent => {
                            // Unreachable today: no CbExecutionError variant
                            // is tagged Silent. If a future variant uses it,
                            // the interpreter loop must define a sound default
                            // *before* propagating Err here, so Silent
                            // surfacing past the dispatcher is a bug.
                            RunResult::Error {
                                message: format!("silent strategy unreachable: {}", e),
                                addr: self.pc,
                            }
                        }
                    };
                    return (result, blocks_executed, forks);
                }
            }
        }

        // Reached max blocks
        let forks = self.take_deferred_forks();
        if self.profiling_enabled {
            self.stats.blocks_executed += blocks_executed as u64;
            if let Some(start) = total_start {
                self.stats.total_time_ns += start.elapsed().as_nanos() as u64;
            }
        }
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

        let lift_start = if self.profiling_enabled { Some(Instant::now()) } else { None };

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
                        if let Some(bytes) = rust_mem.read_concrete_bytes_for_lift(addr, max_bytes) {
                            let native_opt_level = self.vex_opt_level_overrides.get(&addr).copied()
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
                                    log::trace!("Native lift from rust_memory (SMC) at 0x{:x}", addr);
                                    if let Some(start) = lift_start {
                                        self.stats.lift_time_ns += start.elapsed().as_nanos() as u64;
                                    }
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
                            let native_opt_level = self.vex_opt_level_overrides.get(&addr).copied()
                                .or(self.vex_opt_level)
                                .unwrap_or(1);  // pyvex default is 1
                            match crate::vex::libpyvex_ffi::lift_native(
                                bytes,
                                addr,
                                self.arch,
                                99,  // max_insns
                                max_bytes as u32,
                                native_opt_level,
                            ) {
                                Ok(irsb) => {
                                    // Native lift succeeded!
                                    log::trace!("Native lift succeeded at 0x{:x}", addr);
                                    if let Some(start) = lift_start {
                                        self.stats.lift_time_ns += start.elapsed().as_nanos() as u64;
                                    }
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
        let callback_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
        // Resolve VEX opt_level: per-address override > global > None (pyvex default)
        let opt_level = self.vex_opt_level_overrides.get(&addr).copied()
            .or(self.vex_opt_level);

        // SMC: when this lift range overlaps a dirtied page, the cle binary
        // bytes that the Python lifter would normally read are stale. Try
        // to read fresh bytes from rust_memory and pass them via byte_string=
        // so the Python lift sees the post-store program.
        let dirty_bytes: Option<Vec<u8>> =
            if self.is_code_range_dirtied(addr, 4096) {
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
        if let Some(start) = callback_start {
            self.stats.python_callback_count += 1;
            self.stats.python_callback_time_ns += start.elapsed().as_nanos() as u64;
        }

        let irsb = deserialize_irsb(&irsb_json)
            .map_err(|e| CbExecutionError::LiftError(format!("IRSB deserialization failed: {}", e)))?;

        if let Some(start) = lift_start {
            self.stats.lift_time_ns += start.elapsed().as_nanos() as u64;
        }

        // Cache it - Arc allows O(1) cloning
        let arc_irsb = Arc::new(irsb);
        self.block_cache.put(addr, Arc::clone(&arc_irsb));

        Ok(arc_irsb)
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
        let prefetch_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
        self.prefetch_loads_for_block(py, callbacks, irsb)?;
        if let Some(start) = prefetch_start {
            self.stats.prefetch_time_ns += start.elapsed().as_nanos() as u64;
        }

        // Execute statements
        let mut stmt_total_ns: u64 = 0;
        for stmt in &irsb.statements {
            if self.profiling_enabled { self.stats.stmt_count += 1; }
            let stmt_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
            match self.execute_stmt_with_callbacks(py, callbacks, stmt, irsb)? {
                StmtResult::Continue => {
                    if let Some(start) = stmt_start {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        stmt_total_ns += elapsed;
                        #[cfg(debug_assertions)]
                        if elapsed > 50_000_000 { // >50ms
                            log::warn!("  SLOW STMT at 0x{:x}: {}ms {:?}", self.current_insn_addr, elapsed / 1_000_000, stmt);
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
                        condition_id: cond_id,  // Fixed: use proper unique ID
                        true_target,
                        false_target,
                    });
                }
            }
        }

        if self.profiling_enabled {
            self.stats.run_loop_time_ns += stmt_total_ns;
            #[cfg(debug_assertions)]
            if stmt_total_ns > 100_000_000 { // >100ms
                log::warn!("SLOW BLOCK at 0x{:x}: {}ms for {} stmts", irsb.addr, stmt_total_ns / 1_000_000, irsb.statements.len());
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