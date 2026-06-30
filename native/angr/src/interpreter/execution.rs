use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs blocks in a loop, using Python callbacks for memory access,
    /// until it hits a condition that requires Python-side handling.
    ///
    /// Returns a tuple of (result, blocks_executed, deferred_forks).
    ///
    /// `stop_addrs` is the set of address-based find/avoid targets. The
    /// interpreter chains blocks internally; without breaking the chain when a
    /// chained block boundary lands on one of these, a state would execute
    /// right past an address-based find/avoid target and the step-boundary
    /// filter in `run_loop` would only ever observe the final pc. This mirrors
    /// the `steps_limit = 1` guard that already exists for *callable* find/avoid
    /// predicates (see `stepping.rs`). angr-027h: CADET easter-egg find at
    /// 0x804833E was silently skipped because the chain ran on to the
    /// overflow-corrupted ret and went unconstrained.
    ///
    /// `block_granular` (angr-bmyx) generalizes that: when `true`, the chain
    /// breaks at *every* block boundary, not just at `stop_addrs`, so each call
    /// executes exactly one block and every interior pc is observable at the
    /// step boundary — Python angr's block-granular `step()` semantics. The
    /// manager sets it for bare step-loops with no find target (CADET solve.py
    /// phase 3); it stays `false` for `explore()` so chaining keeps its
    /// throughput.
    pub fn run_until_event(
        &mut self,
        callbacks: &PythonCallbacks,
        max_blocks: u32,
        stop_addrs: &std::collections::HashSet<u64>,
        block_granular: bool,
    ) -> (RunResult, u32, Vec<DeferredFork>) {
        let total_start = profile_start!(self);
        let mut blocks_executed = 0u32;

        // Sort concrete memory regions for binary search if needed
        self.sort_concrete_memory();

        // Clear any previous deferred forks and reset per-step limit
        self.deferred_forks.clear();
        self.deferred_fork_this_step = false;

        for _ in 0..max_blocks {
            // Break the internal block chain at a block boundary so the
            // step-boundary find/avoid filter in run_loop observes this pc.
            // `blocks_executed > 0` so a state that STARTS at the boundary
            // (already checked at the prior step boundary) still makes forward
            // progress instead of stalling. Falls through to the MaxBlocks
            // return below, which materializes deferred forks. Two triggers:
            //   - block_granular (angr-bmyx): break at EVERY boundary, so each
            //     call advances exactly one block (Python-like step()).
            //   - stop_addrs (angr-027h): break only when the boundary lands on
            //     an address-based find/avoid target.
            if blocks_executed > 0
                && (block_granular || (!stop_addrs.is_empty() && stop_addrs.contains(&self.pc)))
            {
                break;
            }

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
            let irsb = match self.get_or_lift_block(callbacks, self.pc) {
                Ok(irsb) => irsb,
                Err(e) => {
                    let forks = self.take_deferred_forks();
                    let kind = e.run_error_kind();
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                            kind,
                        },
                        blocks_executed,
                        forks,
                    );
                }
            };

            // Execute the block
            let block_start = profile_start!(self);
            match self.execute_block_with_callbacks(callbacks, &irsb) {
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
                                let sp_offset = self.registers.arch().sp_offset();
                                let sp_val = self
                                    .registers
                                    .get_offset_u64(sp_offset, self.ctx)
                                    .unwrap_or(0);
                                let ret_addr = self.get_return_addr().unwrap_or(0);
                                // angr-4ai9: state.inspect `call` event —
                                // mirror Python callstack.py:386/419 (BEFORE
                                // the push with the target IP, AFTER the
                                // push). function_address is the resolved
                                // call target (`next_addr`).
                                self.dispatch_call_inspect(callbacks, next_addr, "before");
                                self.call_stack.push(crate::state::CallStackEntry {
                                    call_site_addr: self.current_insn_addr,
                                    callee_addr: next_addr,
                                    return_addr: ret_addr,
                                    stack_ptr: sp_val,
                                });
                                self.dispatch_call_inspect(callbacks, next_addr, "after");
                            } else if jumpkind.is_ret() {
                                // angr-4ai9: state.inspect `return` event —
                                // mirror Python callstack.py:430/432.
                                // function_address is the top frame's
                                // callee_addr (what we are about to return
                                // FROM). Snapshot before pop so the BEFORE
                                // callback sees a valid frame.
                                let popped_func_addr =
                                    self.call_stack.last().map(|f| f.callee_addr).unwrap_or(0);
                                self.dispatch_return_inspect(callbacks, popped_func_addr, "before");
                                self.call_stack.pop();
                                self.dispatch_return_inspect(callbacks, popped_func_addr, "after");
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
                                    kind: RunErrorKind::Fatal,
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
                            kind: e.run_error_kind(),
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

        // Check cache first - Arc clone is O(1).
        //
        // Hit/miss/eviction counters are always-on (not gated by
        // `profiling_enabled`): a `u64 += 1` per cache lookup is negligible
        // compared to the lift work it precedes, and these counters need to
        // be reliable for capacity-tuning regardless of profiling state.
        if let Some(irsb) = self.block_cache.get(&addr) {
            self.stats.cache_hit_count += 1;
            return Ok(Arc::clone(irsb));
        }

        self.stats.cache_miss_count += 1;

        let lift_start = profile_start!(self);

        // Lifting via Python callback
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
            .call_lift_block(addr, opt_level, dirty_bytes.as_deref())
            .map_err(|e| CbExecutionError::LiftError(format!("lift callback failed: {}", e)))?;
        if self.profiling_enabled {
            self.stats.python_callback_count += 1;
        }
        profile_add!(callback_start, self.stats.python_callback_time_ns);

        // `_cb_lift_block` returns the literal "{}" sentinel when the block
        // could not be lifted (SimEngineError/PyVEXError, e.g. "No bytes in
        // memory"). That is a designed graceful-deadend signal, not a malformed
        // IRSB — keep it as a LiftError so it routes to the deadended stash
        // (RunErrorKind::Deadend). A deserialize failure on any *other* JSON is
        // a genuinely malformed IRSB and maps to InvalidIR -> errored
        // (MalformedIRSB), consistent with the InvalidIR sites in statements.rs
        // (angr-zzju9).
        if irsb_json.trim() == "{}" {
            return Err(CbExecutionError::LiftError(
                "unliftable block: empty IRSB sentinel".to_string(),
            ));
        }
        let irsb = deserialize_irsb(&irsb_json).map_err(|e| {
            CbExecutionError::InvalidIR(format!("IRSB deserialization failed: {}", e))
        })?;

        profile_add!(lift_start, self.stats.lift_time_ns);

        // Cache it - Arc allows O(1) cloning. Post-miss: any returned Some
        // is an eviction (key was just confirmed not present).
        let arc_irsb = Arc::new(irsb);
        if self.block_cache.put(addr, Arc::clone(&arc_irsb)).is_some() {
            self.stats.cache_eviction_count += 1;
        }

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
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        self.execute_block_with_callbacks(callbacks, irsb)
    }

    /// Execute a block using Python callbacks for memory access.
    fn execute_block_with_callbacks(
        &mut self,
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
        self.prefetch_loads_for_block(callbacks, irsb)?;
        profile_add!(prefetch_start, self.stats.prefetch_time_ns);

        // state.inspect irsb event — fires `when='before'` at block entry,
        // before any statement runs. Bit 7 in the inspect-enabled bitmask.
        if callbacks.inspect_event_enabled(7) {
            let _ = callbacks.call_inspect_irsb(self.current_state_id, "before", irsb.addr);
        }

        // Execute statements
        let mut stmt_total_ns: u64 = 0;
        for (stmt_idx, stmt) in irsb.statements.iter().enumerate() {
            if self.profiling_enabled {
                self.stats.stmt_count += 1;
            }
            // state.inspect statement event — fires `when='before'` once per
            // VEX IR statement, with `stmt_idx` as the only attr. Bit 15 in
            // the inspect-enabled bitmask. The bitmask gate keeps the no-BP
            // cost at one `AtomicU32::load + AND` per statement.
            if callbacks.inspect_event_enabled(15) {
                let _ = callbacks.call_inspect_statement(
                    self.current_state_id,
                    "before",
                    stmt_idx as u32,
                );
            }
            let stmt_start = profile_start!(self);
            match self.execute_stmt_with_callbacks(callbacks, stmt, irsb)? {
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
                    self.flush_stores(callbacks)?;
                    self.pop_block_solver_if_pushed();
                    return Ok(self.handle_exit(target, jumpkind));
                }
                StmtResult::SymbolicBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    // Flush pending stores before returning
                    self.flush_stores(callbacks)?;
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
        self.flush_stores(callbacks)?;

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

    /// Fire a `call` inspect callback into Python for an Ijk_Call exit.
    /// Gated on bit 8 of the inspect-enabled bitmask (above the
    /// `InspectEvent` enum's 0..=5 range and the custom bits 6/7 used by
    /// instruction/irsb). `function_address` is the resolved call target.
    /// Mirrors Python `callstack.py:386` / `:419`.
    fn dispatch_call_inspect(
        &self,
        callbacks: &PythonCallbacks,
        function_address: u64,
        when: &str,
    ) {
        if !callbacks.inspect_event_enabled(8) {
            return;
        }
        let _ = callbacks.call_inspect_call(self.current_state_id, when, function_address);
    }

    /// Fire a `return` inspect callback into Python for an Ijk_Ret exit.
    /// Gated on bit 9 of the inspect-enabled bitmask. `function_address`
    /// is the callee address of the frame being popped (taken from
    /// `call_stack.last().callee_addr` before the pop, mirroring Python
    /// `callstack.py:430` which reads `rself.top.func_addr` prior to
    /// `pop()`). Falls back to 0 when the call stack is empty.
    fn dispatch_return_inspect(
        &self,
        callbacks: &PythonCallbacks,
        function_address: u64,
        when: &str,
    ) {
        if !callbacks.inspect_event_enabled(9) {
            return;
        }
        let _ = callbacks.call_inspect_return(self.current_state_id, when, function_address);
    }
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
