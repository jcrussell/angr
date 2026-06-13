use super::helpers::bv_to_bytes;
use super::*;

/// DCAS-only state bundled together so the single-CAS path can pass `None`
/// and the DCAS path can pass `Some(&DcasState)` through `cas_writeback` and
/// the oldHi temp-assignment in `execute_cas_stmt`.
struct DcasState<'a> {
    addr_hi_expr: IRExpr,
    data_hi_expr: &'a IRExpr,
    current_hi: RustBV,
    expd_hi_val: RustBV,
    data_hi_val: RustBV,
    old_hi_idx: u32,
}

impl<'a> VEXInterpreter<'a> {
    /// Execute a single statement using Python callbacks.
    pub(super) fn execute_stmt_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        stmt: &IRStmt,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark { addr, len, .. } => {
                self.current_insn_addr = *addr;
                self.current_insn_len = *len;
                self.dispatch_instruction_inspect(py, callbacks, *addr);
                // Check for hooks at this address
                if self.is_hooked(*addr) {
                    return Ok(StmtResult::Exit {
                        target: *addr,
                        jumpkind: JumpKind::Boring,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::AbiHint { .. } => Ok(StmtResult::Continue),

            IRStmt::Put { offset, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                let size = value.width().div_ceil(8);
                self.dispatch_reg_write_inspect(py, callbacks, *offset, size, &value);
                self.registers.put(*offset, value);
                self.mark_register_dirty(*offset);

                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                if (*tmp as usize) < self.temps.len() {
                    self.dispatch_tmp_write_inspect(py, callbacks, *tmp, &value);
                    self.temps[*tmp as usize] = Some(value);
                } else {
                    return Err(CbExecutionError::UnknownTemp(*tmp));
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::Store {
                addr,
                data,
                endness,
            } => {
                let store_start = profile_start!(self);
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                let data_size = data_val.width().div_ceil(8) as usize;
                if self.profiling_enabled {
                    self.stats.store_stmt_count += 1;
                }

                if self.use_rust_memory
                    && self.try_rust_memory_store(
                        py,
                        callbacks,
                        &addr_val,
                        &data_val,
                        data_size,
                        store_start,
                    )?
                {
                    // SymbolicMemory::store_concrete already bumped record_mem_store.
                    self.dispatch_mem_write_inspect(
                        py, callbacks, &addr_val, &data_val, data_size, *endness,
                    );
                    return Ok(StmtResult::Continue);
                }

                // angr-obrm: callback-path stores bypass SymbolicMemory, so
                // bump the global mem_store counter here for parity with
                // the Rust-memory path.
                record_mem_store(data_size as u64);
                self.fallback_to_python_store(
                    py,
                    callbacks,
                    &addr_val,
                    data_val.clone(),
                    data_size,
                )?;
                self.dispatch_mem_write_inspect(
                    py, callbacks, &addr_val, &data_val, data_size, *endness,
                );
                Ok(StmtResult::Continue)
            }

            IRStmt::Exit { guard, dst, jk, .. } => {
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;
                self.dispatch_exit_inspect(py, callbacks, *dst, *jk, &guard_val);

                // Check if guard is symbolic first (Constrained has concrete value but is still symbolic)
                if !guard_val.is_symbolic() {
                    // Truly concrete guard - simple check
                    if let Some(g) = guard_val.as_u64() {
                        if g != 0 {
                            return Ok(StmtResult::Exit {
                                target: *dst,
                                jumpkind: *jk,
                            });
                        }
                        return Ok(StmtResult::Continue);
                    }
                }

                // Guard is symbolic — handle based on deferred fork mode
                if !self.config.use_deferred_forks {
                    // Non-deferred mode: return to Python immediately for forking.
                    // Skip can_be_true/can_be_false Z3 checks — Python's
                    // resume_after_symbolic_branch will add constraints and the
                    // sat_cache optimization avoids redundant checks there.
                    let fallthrough = self.eval_next_addr(py, callbacks, irsb)?;
                    // Store condition for Rust-side constraint addition during resume
                    let cond_id = self.next_cond_id();
                    self.stored_conditions.insert(cond_id, guard_val.clone());
                    let result_cond = guard_val.clone();
                    self.last_branch_condition = Some(guard_val);
                    return Ok(StmtResult::SymbolicBranch {
                        condition: result_cond,
                        true_target: *dst,
                        false_target: fallthrough,
                    });
                }

                // Deferred fork mode: check feasibility to decide which paths to explore.
                // Determine branch feasibility. When lazy_solves is active,
                // skip Z3 queries entirely and assume both paths are feasible.
                let (can_be_true, can_be_false) = if self.lazy_solves {
                    (true, true)
                } else {
                    // Incremental assertion: push once per block, assert new fork
                    // conditions incrementally. This avoids re-asserting all N prior
                    // conditions for the N-th Exit (O(N) → O(1) per check).
                    if !self.deferred_forks.is_empty() {
                        if !self.block_solver_pushed {
                            // First time in this block with prior forks: push and assert all
                            self.ctx.push();
                            self.block_solver_pushed = true;
                            for prev_fork in &self.deferred_forks {
                                if let Some(cond) =
                                    self.stored_conditions.get(&prev_fork.condition_id)
                                {
                                    if prev_fork.path_taken {
                                        self.ctx.assume_true(cond);
                                    } else {
                                        self.ctx.assume_false(cond);
                                    }
                                }
                            }
                            self.block_forks_asserted = self.deferred_forks.len();
                        } else {
                            // Subsequent Exits: only assert NEW fork conditions
                            while self.block_forks_asserted < self.deferred_forks.len() {
                                let prev_fork = &self.deferred_forks[self.block_forks_asserted];
                                if let Some(cond) =
                                    self.stored_conditions.get(&prev_fork.condition_id)
                                {
                                    if prev_fork.path_taken {
                                        self.ctx.assume_true(cond);
                                    } else {
                                        self.ctx.assume_false(cond);
                                    }
                                }
                                self.block_forks_asserted += 1;
                            }
                        }
                    }
                    self.ctx.check_branch_feasibility(&guard_val)
                };
                if can_be_true && can_be_false {
                    // The unexplored path (guard=false) should resume at the
                    // next instruction after this conditional jump, NOT the
                    // block's default exit. When a VEX IRSB contains multiple
                    // Ist_Exit statements, using the block fallthrough would
                    // skip all code between this exit and the end of the block.
                    let false_target = self.current_insn_addr + self.current_insn_len as u64;

                    // Skip expensive rustbv_to_claripy conversion for the condition.
                    // The condition is stored in stored_conditions (below) as a RustBV,
                    // which is the primary lookup path in fork processing. The claripy
                    // AST was only a P11 fallback for missing stored_conditions entries.
                    let condition_ast: Option<Py<PyAny>> = None;

                    // Take the "true" path (jump to dst), defer the "false" path
                    let cond_id = self.next_cond_id();
                    // Store the Rust condition for later retrieval when processing forks
                    self.stored_conditions.insert(cond_id, guard_val.clone());

                    // Flush pending stores before snapshotting so the memory
                    // snapshot includes all writes up to this branch point.
                    self.flush_stores_to_rust_memory();

                    // Snapshot full state BEFORE adding the branch constraint.
                    // This enables correct alternate-path forking with solver,
                    // registers, and memory from the branch point.
                    self.fork_snapshots.insert(
                        cond_id,
                        BranchSnapshot {
                            solver: self.ctx.fork(),
                            registers: self.registers.fork(),
                            memory: self.rust_memory.as_ref().map(|m| m.fork()),
                        },
                    );

                    // Decide which path to take based on branch direction.
                    // For backward branches (loops), take the exit (loop back)
                    // and defer the fall-through (loop exit). For forward branches,
                    // take the fall-through and defer the exit. VEX often inverts
                    // forward branch conditions (e.g., `jne target` becomes
                    // `if (eq) goto exit; NEXT: target`), so fall-through follows
                    // the natural execution flow for forward branches.
                    let is_backward_branch = *dst < self.current_insn_addr;

                    let deferred = if is_backward_branch {
                        DeferredFork {
                            branch_addr: self.current_insn_addr,
                            path_taken: true, // we took the exit (guard=true) path
                            unexplored_target: false_target, // fall-through deferred
                            condition_id: cond_id,
                            push_level: self.push_level,
                            condition_ast,
                        }
                    } else {
                        DeferredFork {
                            branch_addr: self.current_insn_addr,
                            path_taken: false, // we took the fallthrough (guard=false) path
                            unexplored_target: *dst, // the exit target is deferred
                            condition_id: cond_id,
                            push_level: self.push_level,
                            condition_ast,
                        }
                    };
                    self.deferred_forks.push(deferred);
                    self.deferred_fork_this_step = true;

                    // NOTE: We intentionally do NOT call assume_true/false() permanently.
                    // The solver stays clean so snapshots capture unconstrained state.
                    // Taken-path constraints are temporarily added via push/pop for
                    // check_branch_feasibility() (above), then applied permanently
                    // during fork processing in exploration.rs after the step completes.

                    if is_backward_branch {
                        // Take the exit path (loop back to target)
                        return Ok(StmtResult::Exit {
                            target: *dst,
                            jumpkind: *jk,
                        });
                    } else {
                        // Continue execution on the fallthrough path
                        return Ok(StmtResult::Continue);
                    }
                } else if can_be_true {
                    return Ok(StmtResult::Exit {
                        target: *dst,
                        jumpkind: *jk,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::MBE(_) => Ok(StmtResult::Continue),

            IRStmt::PutI {
                descr,
                ix,
                bias,
                data,
            } => {
                // Evaluate the index expression
                let ix_val = self.eval_expr_with_callbacks(py, callbacks, ix, &irsb.tyenv)?;

                // PutI requires a concrete index to compute the register offset
                let idx = if let Some(idx) = ix_val.as_u64() {
                    idx
                } else {
                    // Symbolic index - concretize using solver
                    if let Some(concrete) = self.ctx.eval(&ix_val) {
                        concrete as u64
                    } else {
                        return Err(CbExecutionError::Unsupported(
                            "PutI index concretization failed".to_string(),
                        ));
                    }
                };

                // Calculate the rotating register offset:
                // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
                let elem_size = descr.elemTy.bytes();
                let index = ((idx as u32).wrapping_add(*bias)) % descr.nElems;
                let offset = descr.base + index * elem_size;

                // Evaluate the data to write
                let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;

                // Write to the register file
                self.registers.put(offset, data_val);
                self.mark_register_dirty(offset);

                Ok(StmtResult::Continue)
            }

            IRStmt::StoreG {
                guard, addr, data, ..
            } => {
                // Evaluate guard condition
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Check if guard is symbolic
                if guard_val.is_symbolic() {
                    // Symbolic guard: need to handle conditional store
                    // For now, check if guard can be true at all
                    if !self.ctx.can_be_true(&guard_val) {
                        // Guard is always false - skip store
                        return Ok(StmtResult::Continue);
                    }
                    if !self.ctx.can_be_false(&guard_val) {
                        // Guard is always true - perform store unconditionally
                        let addr_val =
                            self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                        let data_val =
                            self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                        let data_size = data_val.width().div_ceil(8) as usize;

                        if let Some(addr_concrete) = addr_val.as_u64() {
                            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            // Check if data is symbolic - use symbolic store callback
                            if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value()
                            {
                                self.flush_stores(py, callbacks)?;
                                callbacks
                                    .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            } else {
                                let data_bytes = bv_to_bytes(&data_val);
                                self.pending_stores.push(addr_concrete, data_bytes);
                                if self.pending_stores.len() >= self.max_pending_stores {
                                    self.flush_stores(py, callbacks)?;
                                }
                            }
                        }
                        return Ok(StmtResult::Continue);
                    }
                    // Both paths possible with symbolic guard - use ITE for conditional store
                    // Store ITE(guard, new_data, current_data)
                    let addr_val =
                        self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                    let data_val =
                        self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                    let data_size = data_val.width().div_ceil(8) as usize;

                    if let Some(addr_concrete) = addr_val.as_u64() {
                        // Load current value at address
                        let current =
                            self.load_from_callback(py, callbacks, addr_concrete, data_size)?;
                        // Create ITE: if guard then new_data else current
                        let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                        self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                        // ITE result is symbolic if guard or either operand is symbolic
                        if ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                            self.flush_stores(py, callbacks)?;
                            callbacks
                                .call_memory_store_symbolic_value(py, addr_concrete, &ite_result)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        } else {
                            let ite_bytes = bv_to_bytes(&ite_result);
                            self.pending_stores.push(addr_concrete, ite_bytes);
                            if self.pending_stores.len() >= self.max_pending_stores {
                                self.flush_stores(py, callbacks)?;
                            }
                        }
                    } else {
                        // Symbolic address with symbolic guard - concretize for write
                        match &*self.concretize_cached_write(&addr_val) {
                            ConcretizationResult::Single(addr_concrete) => {
                                let addr_concrete = *addr_concrete;
                                // Load current value and use ITE
                                let current = self.load_from_callback(
                                    py,
                                    callbacks,
                                    addr_concrete,
                                    data_size,
                                )?;
                                let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                                self.flush_stores(py, callbacks)?;
                                // ITE result is symbolic - use symbolic store callback
                                if ite_result.is_symbolic()
                                    && callbacks.has_memory_store_symbolic_value()
                                {
                                    callbacks
                                        .call_memory_store_symbolic_value(
                                            py,
                                            addr_concrete,
                                            &ite_result,
                                        )
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                } else {
                                    let ite_bytes = bv_to_bytes(&ite_result);
                                    callbacks
                                        .call_memory_store(py, addr_concrete, &ite_bytes)
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                }
                                // Invalidate any prefetched value at this address.
                                self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            }
                            _ => {
                                // Symbolic guard + non-Single address solutions
                                // (Multiple/Strided/TooLarge/Failed). Combining the
                                // guard-ITE with per-address ITEs requires a per-
                                // address load and is brittle, so delegate to Python's
                                // full symbolic store callback which has access to
                                // angr's address concretization strategies.
                                self.flush_stores(py, callbacks)?;
                                if callbacks.has_memory_store_symbolic_full() {
                                    callbacks
                                        .call_memory_store_symbolic_full(py, &addr_val, &data_val)
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                } else {
                                    return Err(CbExecutionError::Unsupported(
                                        "guarded store with symbolic address: \
                                         no memory_store_symbolic_full callback"
                                            .to_string(),
                                    ));
                                }
                                // Touched addresses are unknown, drop the whole cache.
                                self.load_prefetch_cache.clear();
                            }
                        }
                    }
                    return Ok(StmtResult::Continue);
                }

                // Concrete guard: simple check
                if let Some(g) = guard_val.as_u64()
                    && g != 0
                {
                    // Guard is true - perform the store
                    let addr_val =
                        self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                    let data_val =
                        self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                    let data_size = data_val.width().div_ceil(8) as usize;

                    if let Some(addr_concrete) = addr_val.as_u64() {
                        self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                        // Check if data is symbolic - use symbolic store callback
                        if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                            self.flush_stores(py, callbacks)?;
                            callbacks
                                .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        } else {
                            let data_bytes = bv_to_bytes(&data_val);
                            self.pending_stores.push(addr_concrete, data_bytes);
                            if self.pending_stores.len() >= self.max_pending_stores {
                                self.flush_stores(py, callbacks)?;
                            }
                        }
                    } else {
                        // Symbolic address with concrete guard - flush and use callback
                        self.flush_stores(py, callbacks)?;
                        // Check if data is symbolic - use symbolic store callback
                        if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                            // Concretize address for write (with cache)
                            let concret_result = self.concretize_cached_write(&addr_val);
                            match &*concret_result {
                                ConcretizationResult::Single(addr_concrete) => {
                                    let addr_concrete = *addr_concrete;
                                    callbacks
                                        .call_memory_store_symbolic_value(
                                            py,
                                            addr_concrete,
                                            &data_val,
                                        )
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                                }
                                ConcretizationResult::Multiple(addrs) => {
                                    // Symbolic data + multiple address solutions: prefer the full
                                    // symbolic store callback. Otherwise build an ITE chain in Rust
                                    // (mirrors fallback_to_python_store::Multiple) so all candidate
                                    // addresses are updated, not just the first one.
                                    if callbacks.has_memory_store_symbolic_full() {
                                        callbacks
                                            .call_memory_store_symbolic_full(
                                                py, &addr_val, &data_val,
                                            )
                                            .map_err(|e| {
                                                CbExecutionError::Callback(e.to_string())
                                            })?;
                                    } else if callbacks.has_memory_store_symbolic_value()
                                        && addrs.len() <= 16
                                    {
                                        self.build_ite_store_from_callbacks(
                                            py, callbacks, addrs, &addr_val, &data_val,
                                        )?;
                                    } else {
                                        return Err(CbExecutionError::Unsupported(
                                                "symbolic store with multiple address solutions: \
                                                 no memory_store_symbolic_full callback and ITE chain unavailable".to_string()
                                            ));
                                    }
                                    // Multiple candidate addresses written; drop the whole cache.
                                    self.load_prefetch_cache.clear();
                                }
                                _ => {
                                    // TooLarge or Failed - delegate to Python's full symbolic callback
                                    if callbacks.has_memory_store_symbolic_full() {
                                        callbacks
                                            .call_memory_store_symbolic_full(
                                                py, &addr_val, &data_val,
                                            )
                                            .map_err(|e| {
                                                CbExecutionError::Callback(e.to_string())
                                            })?;
                                    } else {
                                        return Err(CbExecutionError::Unsupported(
                                            "symbolic store with unconcretizable address"
                                                .to_string(),
                                        ));
                                    }
                                    // Touched addresses are unknown, drop the whole cache.
                                    self.load_prefetch_cache.clear();
                                }
                            }
                        } else {
                            let data_bytes = bv_to_bytes(&data_val);
                            callbacks
                                .call_memory_store(py, 0, &data_bytes)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                    }
                }
                // Guard is false - skip the store

                Ok(StmtResult::Continue)
            }

            IRStmt::LoadG {
                dst,
                guard,
                addr,
                alt,
                cvt,
                ..
            } => {
                // Evaluate guard condition
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Evaluate the alternative value (used when guard is false)
                let alt_val = self.eval_expr_with_callbacks(py, callbacks, alt, &irsb.tyenv)?;

                // Determine the load size from the destination temp type
                let dst_ty = irsb.tyenv.get(*dst).ok_or_else(|| {
                    CbExecutionError::InvalidIR(format!(
                        "LoadG destination temp {} not in tyenv",
                        dst
                    ))
                })?;
                let load_size = match cvt {
                    IRLoadGOp::Identity => dst_ty.bytes() as usize,
                    // The source width is carried in the cvt op, so an 8->32
                    // and a 16->32 widening load are correctly distinguished
                    // (the latter previously defaulted to a 1-byte load).
                    IRLoadGOp::WidenS { src_bits } | IRLoadGOp::WidenZ { src_bits } => {
                        (*src_bits / 8) as usize
                    }
                    IRLoadGOp::Unknown => {
                        return Err(CbExecutionError::InvalidIR(
                            "LoadG has an unrecognized conversion op (cvt)".to_string(),
                        ));
                    }
                };

                // Check if guard is symbolic
                if guard_val.is_symbolic() {
                    // Check if guard can be true/false
                    let can_be_true = self.ctx.can_be_true(&guard_val);
                    let can_be_false = self.ctx.can_be_false(&guard_val);

                    if can_be_true && !can_be_false {
                        // Guard is always true - perform load unconditionally
                        let addr_val =
                            self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                        let loaded = self.resolve_loadg_load(
                            py,
                            callbacks,
                            &addr_val,
                            load_size,
                            "LoadG (always-true guard)",
                        )?;

                        // Apply conversion
                        let result = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());

                        if (*dst as usize) < self.temps.len() {
                            self.temps[*dst as usize] = Some(result);
                        }
                        return Ok(StmtResult::Continue);
                    }

                    if !can_be_true && can_be_false {
                        // Guard is always false - use alt value
                        if (*dst as usize) < self.temps.len() {
                            self.temps[*dst as usize] = Some(alt_val);
                        }
                        return Ok(StmtResult::Continue);
                    }

                    // Both paths possible - evaluate address and load, then ITE
                    let addr_val =
                        self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                    let loaded = self.resolve_loadg_load(
                        py,
                        callbacks,
                        &addr_val,
                        load_size,
                        "LoadG (symbolic guard)",
                    )?;

                    // Apply conversion to loaded value
                    let converted = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());

                    // Create ITE: if guard then loaded else alt
                    let result = guard_val.ite(&converted, &alt_val, self.ctx);

                    if (*dst as usize) < self.temps.len() {
                        self.temps[*dst as usize] = Some(result);
                    }
                    return Ok(StmtResult::Continue);
                }

                // Concrete guard
                if let Some(g) = guard_val.as_u64() {
                    let result = if g != 0 {
                        // Guard is true - perform the load
                        let addr_val =
                            self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                        let loaded = self.resolve_loadg_load(
                            py,
                            callbacks,
                            &addr_val,
                            load_size,
                            "LoadG (concrete-true guard)",
                        )?;
                        self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits())
                    } else {
                        // Guard is false - use alternative value
                        alt_val
                    };

                    if (*dst as usize) < self.temps.len() {
                        self.temps[*dst as usize] = Some(result);
                    }
                } else {
                    // This shouldn't happen if guard_val is concrete
                    return Err(CbExecutionError::InvalidIR(
                        "LoadG guard evaluation failed".to_string(),
                    ));
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::CAS {
                old_hi,
                old_lo,
                addr,
                expdHi,
                expdLo,
                dataHi,
                dataLo,
                endness,
            } => self.execute_cas_stmt(
                py,
                callbacks,
                *old_hi,
                *old_lo,
                addr,
                expdHi.as_deref(),
                expdLo,
                dataHi.as_deref(),
                dataLo,
                *endness,
                irsb,
            ),

            IRStmt::LLSC {
                storedata,
                result,
                addr,
                endness,
            } => {
                match storedata {
                    None => {
                        // Load-linked: load value at addr, write to result temp.
                        let result_ty = irsb.tyenv.get(*result).ok_or_else(|| {
                            CbExecutionError::InvalidIR(format!(
                                "LLSC result temp {} not in tyenv",
                                result
                            ))
                        })?;
                        let load_expr = IRExpr::Load {
                            addr: addr.clone(),
                            ty: result_ty,
                            endness: *endness,
                        };
                        let value =
                            self.eval_expr_with_callbacks(py, callbacks, &load_expr, &irsb.tyenv)?;
                        if (*result as usize) < self.temps.len() {
                            self.temps[*result as usize] = Some(value);
                        } else {
                            return Err(CbExecutionError::UnknownTemp(*result));
                        }
                    }
                    Some(data_expr) => {
                        // Store-conditional: store data, write 1 (success) to result temp.
                        // Simplified non-atomic single-state model — store always succeeds.
                        let store_stmt = IRStmt::Store {
                            addr: (**addr).clone(),
                            data: (**data_expr).clone(),
                            endness: *endness,
                        };
                        self.execute_stmt_with_callbacks(py, callbacks, &store_stmt, irsb)?;
                        if (*result as usize) < self.temps.len() {
                            self.temps[*result as usize] = Some(RustBV::concrete(1, 1));
                        } else {
                            return Err(CbExecutionError::UnknownTemp(*result));
                        }
                    }
                }
                Ok(StmtResult::Continue)
            }
            IRStmt::Dirty(dirty) => {
                // Check guard if present
                if let Some(guard) = &dirty.guard {
                    let guard_val =
                        self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;
                    if guard_val.is_symbolic() {
                        // Symbolic guard: pick the taken branch if feasible,
                        // otherwise skip. We can't fork mid-block, so we
                        // concretize-to-taken (lossy but unblocks execution).
                        let (cb_true, cb_false) = self.ctx.check_branch_feasibility(&guard_val);
                        if !cb_true {
                            // Guard must be false — skip the dirty call.
                            return Ok(StmtResult::Continue);
                        }
                        if cb_false {
                            // Both feasible: pin guard true so the dirty call
                            // runs. Loses the not-taken branch but matches
                            // angr's existing dirty-helper concretization.
                            log::debug!(
                                "dirty call '{}': symbolic guard concretized to taken branch",
                                dirty.cee.name
                            );
                            self.ctx.assume_true(&guard_val);
                        }
                        // Fall through and execute the dirty call.
                    } else if let Some(g) = guard_val.as_u64()
                        && g == 0
                    {
                        // Guard is false - skip the dirty call
                        return Ok(StmtResult::Continue);
                    }
                }

                // Evaluate arguments. Eager-concretize symbolic args via the
                // solver so that native dispatch + Python callback (which both
                // expect concrete u64 args) can run; the equality constraint
                // is added so downstream branches stay consistent.
                let mut arg_vals: Vec<u64> = Vec::with_capacity(dirty.args.len());
                let mut all_args_concrete = true;
                for arg in &dirty.args {
                    let val = self.eval_expr_with_callbacks(py, callbacks, arg, &irsb.tyenv)?;
                    if let Some(concrete) = val.as_u64() {
                        arg_vals.push(concrete);
                    } else if let Some(concrete) = self.ctx.eval(&val) {
                        let conc_bv = RustBV::concrete(concrete, val.width());
                        let constraint = val.eq(&conc_bv, self.ctx);
                        self.ctx.assume_true(&constraint);
                        arg_vals.push(concrete as u64);
                    } else {
                        // Solver couldn't produce a concrete value (e.g. UNSAT
                        // path). Fall through to the Python/no-handler paths
                        // so they can apply their own fallback strategy.
                        all_args_concrete = false;
                        break;
                    }
                }

                // Determine return type bits
                let ret_ty_bits = if let Some(tmp) = dirty.tmp {
                    irsb.tyenv.get(tmp).map(|t| t.bits()).unwrap_or(64)
                } else {
                    0 // No return value
                };

                // Try native dirty helper dispatch first
                if all_args_concrete
                    && let Some(result) = self.dirty_dispatch.try_call(&dirty.cee.name, &arg_vals)
                {
                    // Native handler succeeded!
                    log::trace!(
                        "Native dirty call: {} (args: {:?})",
                        dirty.cee.name,
                        arg_vals
                    );

                    // Store result in temporary if specified
                    if let Some(tmp) = dirty.tmp
                        && let Some(return_value) = result.return_value
                    {
                        let value = RustBV::concrete(return_value as u128, ret_ty_bits);
                        if (tmp as usize) < self.temps.len() {
                            self.temps[tmp as usize] = Some(value);
                        }
                    }

                    // Apply any register writes from the helper
                    for (offset, value) in result.reg_writes {
                        // Convert u64 value to RustBV and store in register
                        let bv = RustBV::concrete(value as u128, 64);
                        self.registers.put(offset, bv);
                    }

                    return Ok(StmtResult::Continue);
                }

                // No native handler matched. If Python also has no callback
                // registered, treat the dirty call as a stub: write a fresh
                // symbolic value into the result tmp (if any) and continue.
                // This avoids hard-erroring on long-tail dirty helpers that
                // neither Rust nor Python explicitly model.
                if !callbacks.has_dirty_call() {
                    log::warn!(
                        "dirty call '{}': no native handler and no Python callback; \
                         stubbing with a fresh symbolic tmp",
                        dirty.cee.name
                    );
                    if let Some(tmp) = dirty.tmp {
                        let bits = if ret_ty_bits == 0 { 64 } else { ret_ty_bits };
                        let stub = RustBV::symbolic(
                            self.ctx,
                            format!("dirty_{}_stub", dirty.cee.name),
                            bits,
                        );
                        if (tmp as usize) < self.temps.len() {
                            self.temps[tmp as usize] = Some(stub);
                        }
                    }
                    return Ok(StmtResult::Continue);
                }

                if !all_args_concrete {
                    // First-pass loop bailed early because the solver could not
                    // produce a concrete value for one of the args. Try again,
                    // this time concretizing more aggressively; if any arg is
                    // still unrepresentable, surface a clear error.
                    arg_vals.clear();
                    for arg in &dirty.args {
                        let val = self.eval_expr_with_callbacks(py, callbacks, arg, &irsb.tyenv)?;
                        if let Some(concrete) = val.as_u64() {
                            arg_vals.push(concrete);
                        } else if let Some(concrete) = self.ctx.eval(&val) {
                            let conc_bv = RustBV::concrete(concrete, val.width());
                            let constraint = val.eq(&conc_bv, self.ctx);
                            self.ctx.assume_true(&constraint);
                            arg_vals.push(concrete as u64);
                        } else {
                            return Err(CbExecutionError::Unsupported(format!(
                                "dirty call '{}' arg unconcretizable",
                                dirty.cee.name
                            )));
                        }
                    }
                }

                // Call Python callback
                self.stats.python_dirty_call_count += 1;
                let (data, is_symbolic, _symbolic_ast) = callbacks
                    .call_dirty_call(py, &dirty.cee.name, &arg_vals, ret_ty_bits)
                    .map_err(|e| {
                        CbExecutionError::Callback(format!(
                            "dirty call {} failed: {}",
                            dirty.cee.name, e
                        ))
                    })?;

                // Store result in temporary if specified
                if let Some(tmp) = dirty.tmp {
                    let result = if is_symbolic {
                        // Create a symbolic value for the result
                        RustBV::symbolic(self.ctx, format!("dirty_{}", dirty.cee.name), ret_ty_bits)
                    } else {
                        // Convert bytes to concrete value
                        let mut value: u128 = 0;
                        for (i, &byte) in data.iter().enumerate() {
                            if (i * 8) as u32 >= ret_ty_bits {
                                break;
                            }
                            value |= (byte as u128) << (i * 8);
                        }
                        RustBV::concrete(value, ret_ty_bits)
                    };

                    if (tmp as usize) < self.temps.len() {
                        self.temps[tmp as usize] = Some(result);
                    }
                }

                Ok(StmtResult::Continue)
            }
        }
    }

    /// Attempt to store via Rust-native memory. Returns Ok(true) if the store
    /// was handled, Ok(false) if the caller should fall back to the Python path.
    fn try_rust_memory_store(
        &mut self,
        py: Python<'_>,
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
                py, callbacks, addr_val, "store", "before", None,
            );
        }
        // Concretize for write with per-block cache (avoids redundant Z3 calls)
        let conc_result = self.concretize_cached_write(addr_val);
        if !addr_val.is_concrete() {
            let result_addrs = match &*conc_result {
                ConcretizationResult::Single(a) => Some(vec![*a]),
                ConcretizationResult::Multiple(addrs) => Some(addrs.clone()),
                ConcretizationResult::Strided {
                    base,
                    stride,
                    count,
                } => Some((0..*count).map(|i| base + i * stride).collect()),
                ConcretizationResult::TooLarge { .. } | ConcretizationResult::Failed(_) => None,
            };
            self.dispatch_address_concretization_inspect(
                py,
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
                self.update_prefetch_on_store(addr_val, &conc_result, data_size);
                // Rust owns memory — no need to sync stores to Python.
                profile_add!(store_start, self.stats.store_stmt_time_ns);
                Ok(true)
            }
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                let prefetch_count = self.page_prefetch_count;
                let page_fetched =
                    self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

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
                                self.update_prefetch_on_store(addr_val, &conc_result, data_size);
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
                    "Unmapped memory store at 0x{:x} (size={}), falling back to Python",
                    addr,
                    unmapped_size
                );
                Ok(false)
            }
            Err(MemoryError::SymbolicAddress { description }) => {
                // Address range too large or symbolic — Python's memory model handles natively
                log::debug!(
                    "Symbolic address store: {}, falling back to Python",
                    description
                );
                Ok(false)
            }
            Err(e) => Err(CbExecutionError::Memory(e.to_string())),
        }
    }

    /// Update load-prefetch cache after a successful Rust-native store.
    /// Single-address writes invalidate that (addr, size) entry; otherwise
    /// the entire prefetch cache is dropped. Also invalidates cached IRSBs
    /// when the store hits a loaded binary region (self-modifying code
    /// support).
    fn update_prefetch_on_store(
        &mut self,
        _addr_val: &RustBV,
        conc_result: &ConcretizationResult,
        data_size: usize,
    ) {
        match conc_result {
            ConcretizationResult::Single(addr_concrete) => {
                self.load_prefetch_cache
                    .remove(&(*addr_concrete, data_size));
                if self.is_in_binary(*addr_concrete) {
                    self.invalidate_code_at(*addr_concrete, data_size);
                }
            }
            _ => {
                self.load_prefetch_cache.clear();
                self.invalidate_code_for_concretization(conc_result, data_size);
            }
        }
    }

    /// Invalidate cached IRSBs for a multi-address concretization result whose
    /// solutions hit loaded binary regions. Conservative: any in-binary
    /// solution triggers a per-address invalidation; ranges/Any clear the
    /// cache outright since the affected bytes are unbounded.
    fn invalidate_code_for_concretization(
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
    /// concrete-vs-symbolic address; the concrete branch invalidates load
    /// caches then dispatches via `handle_concrete_store`, the symbolic
    /// branch goes through `handle_symbolic_store`.
    fn fallback_to_python_store(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        if let Some(addr_concrete) = addr_val.as_u64() {
            self.invalidate_loads_at(addr_concrete, data_size);
            self.handle_concrete_store(py, callbacks, addr_concrete, data_val, data_size)
        } else {
            self.handle_symbolic_store(py, callbacks, addr_val, &data_val, data_size)
        }
    }

    /// Invalidate the load prefetch cache entry for `(addr, data_size)` and
    /// drop any cached IRSB whose bytes overlap the store (self-modifying
    /// code support). Concrete-address stores only; symbolic-address stores
    /// must clear the prefetch cache wholesale instead — see
    /// `handle_symbolic_store` and the `invariant-prefetch-cache-on-symbolic-store`
    /// memory.
    fn invalidate_loads_at(&mut self, addr: u64, data_size: usize) {
        self.load_prefetch_cache.remove(&(addr, data_size));
        if self.is_in_binary(addr) {
            self.invalidate_code_at(addr, data_size);
        }
    }

    /// Concrete-address store path: chooses between the
    /// `memory_store_symbolic_value` callback (32-bit non-stack only) and
    /// the buffered `pending_stores` fast path. The 32-bit heuristic exists
    /// to keep flareon2015_5 working while avoiding the false-positive cost
    /// of routing every 64-bit symbolic store through Python.
    fn handle_concrete_store(
        &mut self,
        py: Python<'_>,
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
            let sym_ok = (|| -> Result<(), CbExecutionError> {
                self.flush_stores(py, callbacks)?;
                callbacks
                    .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
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
                let _ = callbacks.call_memory_store(py, addr_concrete, &data_bytes);
            }
            Ok(())
        } else {
            // Fast path: buffer for batch processing
            let data_bytes = bv_to_bytes(&data_val);
            // Track symbolic values for load forwarding
            if data_val.is_symbolic() {
                self.pending_symbolic_stores.insert(addr_concrete, data_val);
            }
            self.pending_stores.push(addr_concrete, data_bytes);

            if self.pending_stores.len() >= self.max_pending_stores {
                self.flush_stores(py, callbacks)?;
            }
            Ok(())
        }
    }

    /// Symbolic-address store path. Clears the prefetch cache, flushes the
    /// pending-store buffer, then dispatches on the 5 ConcretizationResult
    /// shapes returned by `concretize_cached_write`. Single → direct callback;
    /// Multiple/Strided → `dispatch_multi_store`; TooLarge → full symbolic
    /// callback or Unsupported; Failed → `fallback_store_symbolic_full`.
    fn handle_symbolic_store(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        // AVOID_MULTIVALUED_WRITES: silently drop. Do not touch the prefetch
        // cache or pending stores — the address was never resolved, so no
        // mutation propagates.
        if self.concretizer.should_avoid_multivalued_write(addr_val) {
            return Ok(());
        }
        // Touched addresses are unknown — drop the entire prefetch cache
        // and flush pending stores before Python sees the symbolic write.
        self.load_prefetch_cache.clear();
        self.flush_stores(py, callbacks)?;

        let concret_result = self.concretize_cached_write(addr_val);
        match &*concret_result {
            ConcretizationResult::Single(addr_concrete) => {
                let addr_concrete = *addr_concrete;
                if self.is_in_binary(addr_concrete) {
                    self.invalidate_code_at(addr_concrete, data_size);
                }
                if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                    callbacks
                        .call_memory_store_symbolic_value(py, addr_concrete, data_val)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                } else {
                    let data_bytes = bv_to_bytes(data_val);
                    callbacks
                        .call_memory_store(py, addr_concrete, &data_bytes)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                }
                Ok(())
            }
            ConcretizationResult::Multiple(addrs) => {
                self.dispatch_multi_store(py, callbacks, addrs, addr_val, data_val)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let addrs: Vec<u64> = (0..*count).map(|i| base + i * stride).collect();
                self.dispatch_multi_store(py, callbacks, &addrs, addr_val, data_val)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let (min, max) = (*min, *max);
                // Range too large to enumerate — delegate to Python's memory
                // model (which has access to angr's address concretization
                // strategies) via the full symbolic callback.
                if callbacks.has_memory_store_symbolic_full() {
                    callbacks
                        .call_memory_store_symbolic_full(py, addr_val, data_val)
                        .map_err(|e| {
                            CbExecutionError::Callback(format!(
                                "symbolic store full callback failed at 0x{:x}-0x{:x}: {}",
                                min, max, e
                            ))
                        })?;
                    Ok(())
                } else {
                    Err(CbExecutionError::Unsupported(format!(
                        "symbolic store with too-large address range 0x{:x}-0x{:x}: \
                         no memory_store_symbolic_full callback",
                        min, max
                    )))
                }
            }
            ConcretizationResult::Failed(reason) => {
                // Concretization failed entirely. Try the full symbolic
                // store callback so Python's memory model can still resolve
                // the address; only error out if the callback isn't wired up.
                let descr = format!("concretize failed: {}", reason);
                self.fallback_store_symbolic_full(
                    py, callbacks, addr_val, data_val, "store", &descr,
                )
            }
        }
    }

    /// Shared dispatch for Multiple/Strided concretization results: build an
    /// in-Rust ITE chain when ≤16 addrs and the symbolic-value callback is
    /// available, otherwise hand the full address list to Python.
    fn dispatch_multi_store(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_val: &RustBV,
        data_val: &RustBV,
    ) -> Result<(), CbExecutionError> {
        if addrs.len() <= 16 && callbacks.has_memory_store_symbolic_value() {
            self.build_ite_store_from_callbacks(py, callbacks, addrs, addr_val, data_val)
        } else {
            callbacks
                .call_memory_store_symbolic(py, addrs, data_val, addr_val)
                .map_err(|e| CbExecutionError::Callback(e.to_string()))
        }
    }

    /// CAS handler — supports both single CAS and DCAS (double compare-and-swap,
    /// e.g. x86-64 cmpxchg16b). DCAS = oldHi/expdHi/dataHi all Some, single = all None.
    ///
    /// DCAS semantics (mirroring `_perform_vex_stmt_CAS` in
    /// angr/engines/vex/light/light.py): load both halves at `addr` and
    /// `addr + sizeof(expd_ty)`; compare both vs `(expd_hi, expd_lo)`; on match,
    /// write `(data_hi, data_lo)` back. Only little-endian is supported — the
    /// only real DCAS users (x86-64, ARM64) are LE; BE is rejected explicitly.
    #[allow(clippy::too_many_arguments)]
    fn execute_cas_stmt(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        old_hi: Option<u32>,
        old_lo: u32,
        addr: &IRExpr,
        expd_hi: Option<&IRExpr>,
        expd_lo: &IRExpr,
        data_hi: Option<&IRExpr>,
        data_lo: &IRExpr,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        let is_dcas = match (old_hi, expd_hi, data_hi) {
            (Some(_), Some(_), Some(_)) => true,
            (None, None, None) => false,
            _ => {
                return Err(CbExecutionError::InvalidIR(
                    "CAS: oldHi/expdHi/dataHi must be all-Some (DCAS) or all-None (single)"
                        .to_string(),
                ));
            }
        };

        // Half type comes from expdLo — both halves share it.
        let half_ty = expd_lo
            .get_type(&irsb.tyenv)
            .ok_or_else(|| CbExecutionError::InvalidIR("CAS expdLo has no type".to_string()))?;

        if is_dcas && endness == Endness::Big {
            // No real arch (x86-64 cmpxchg16b, ARM64 LDXP) is BE; if BE DCAS
            // ever shows up, defer to Python's full-CAS implementation rather
            // than guessing the address-of-Hi vs address-of-Lo convention.
            return Err(CbExecutionError::Unsupported(
                "DCAS with big-endian memory not supported".to_string(),
            ));
        }

        // Load current_lo at addr.
        let load_lo_expr = IRExpr::Load {
            addr: Box::new(addr.clone()),
            ty: half_ty,
            endness,
        };
        let current_lo =
            self.eval_expr_with_callbacks(py, callbacks, &load_lo_expr, &irsb.tyenv)?;
        let expd_lo_val = self.eval_expr_with_callbacks(py, callbacks, expd_lo, &irsb.tyenv)?;
        let data_lo_val = self.eval_expr_with_callbacks(py, callbacks, data_lo, &irsb.tyenv)?;

        // For DCAS, also load the high half at addr + sizeof(half).
        let dcas = if is_dcas {
            let addr_hi_expr = Self::cas_compute_addr_hi(addr, half_ty, irsb)?;
            let (current_hi, expd_hi_val, data_hi_val) = self.cas_load_dcas_high(
                py,
                callbacks,
                &addr_hi_expr,
                expd_hi.unwrap(),
                data_hi.unwrap(),
                half_ty,
                endness,
                irsb,
            )?;
            Some(DcasState {
                addr_hi_expr,
                data_hi_expr: data_hi.unwrap(),
                current_hi,
                expd_hi_val,
                data_hi_val,
                old_hi_idx: old_hi.unwrap(),
            })
        } else {
            None
        };

        // Combined cmp = (current_lo == expd_lo) & (current_hi == expd_hi)?
        let cmp_lo = current_lo.eq(&expd_lo_val, self.ctx);
        let cmp = if let Some(d) = &dcas {
            let cmp_hi = d.current_hi.eq(&d.expd_hi_val, self.ctx);
            cmp_lo.and(&cmp_hi, self.ctx)
        } else {
            cmp_lo
        };

        self.cas_writeback(
            py,
            callbacks,
            &cmp,
            addr,
            data_lo,
            &data_lo_val,
            &current_lo,
            dcas.as_ref(),
            endness,
            irsb,
        )?;

        // Write current values to oldLo (and oldHi for DCAS).
        if (old_lo as usize) >= self.temps.len() {
            return Err(CbExecutionError::UnknownTemp(old_lo));
        }
        self.temps[old_lo as usize] = Some(current_lo);
        if let Some(d) = dcas {
            if (d.old_hi_idx as usize) >= self.temps.len() {
                return Err(CbExecutionError::UnknownTemp(d.old_hi_idx));
            }
            self.temps[d.old_hi_idx as usize] = Some(d.current_hi);
        }

        Ok(StmtResult::Continue)
    }

    /// Compute the address of the DCAS high half: `addr + sizeof(half_ty)`.
    /// Synthesised as a fresh `IRExpr::Binop(Add)` rather than mutating the
    /// per-IRSB tyenv — see the `dcas-irexpr-binop-addr-hi` invariant.
    fn cas_compute_addr_hi(
        addr: &IRExpr,
        half_ty: IRType,
        irsb: &IRSB,
    ) -> Result<IRExpr, CbExecutionError> {
        let addr_ty = addr
            .get_type(&irsb.tyenv)
            .ok_or_else(|| CbExecutionError::InvalidIR("CAS addr has no type".to_string()))?;
        let half_bytes = half_ty.bytes() as u64;
        let offset_const = match addr_ty {
            IRType::I32 => IRExpr::Const(IRConst::U32(half_bytes as u32)),
            IRType::I64 => IRExpr::Const(IRConst::U64(half_bytes)),
            _ => {
                return Err(CbExecutionError::InvalidIR(
                    "CAS addr must be I32 or I64".to_string(),
                ));
            }
        };
        Ok(IRExpr::Binop {
            op: IROp::Add(addr_ty),
            left: Box::new(addr.clone()),
            right: Box::new(offset_const),
        })
    }

    /// Load and evaluate the DCAS high half. Returns
    /// `(current_hi, expd_hi_val, data_hi_val)`.
    #[allow(clippy::too_many_arguments)]
    fn cas_load_dcas_high(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_hi_expr: &IRExpr,
        expd_hi: &IRExpr,
        data_hi: &IRExpr,
        half_ty: IRType,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<(RustBV, RustBV, RustBV), CbExecutionError> {
        let load_hi_expr = IRExpr::Load {
            addr: Box::new(addr_hi_expr.clone()),
            ty: half_ty,
            endness,
        };
        let current_hi =
            self.eval_expr_with_callbacks(py, callbacks, &load_hi_expr, &irsb.tyenv)?;
        let expd_hi_val = self.eval_expr_with_callbacks(py, callbacks, expd_hi, &irsb.tyenv)?;
        let data_hi_val = self.eval_expr_with_callbacks(py, callbacks, data_hi, &irsb.tyenv)?;
        Ok((current_hi, expd_hi_val, data_hi_val))
    }

    /// Perform the CAS writeback. Three cases on `cmp`:
    ///   - concrete false: no store.
    ///   - concrete true:  store `data` (reusing the original IRExpr).
    ///   - symbolic:       store `ITE(cmp, data, current)` — the deferred-fork
    ///     branch, where both outcomes are encoded into a single
    ///     state via ITE rather than splitting into two states.
    #[allow(clippy::too_many_arguments)]
    fn cas_writeback(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        cmp: &RustBV,
        addr: &IRExpr,
        data_lo_expr: &IRExpr,
        data_lo_val: &RustBV,
        current_lo: &RustBV,
        dcas: Option<&DcasState>,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        match cmp.as_u64() {
            Some(0) => Ok(()),
            Some(_) => {
                self.cas_dispatch_store(
                    py,
                    callbacks,
                    addr,
                    data_lo_expr,
                    data_lo_val,
                    endness,
                    irsb,
                )?;
                if let Some(d) = dcas {
                    self.cas_dispatch_store(
                        py,
                        callbacks,
                        &d.addr_hi_expr,
                        d.data_hi_expr,
                        &d.data_hi_val,
                        endness,
                        irsb,
                    )?;
                }
                Ok(())
            }
            None => {
                let store_lo = cmp.ite(data_lo_val, current_lo, self.ctx);
                self.cas_store_symbolic_data(py, callbacks, addr, &store_lo, irsb)?;
                if let Some(d) = dcas {
                    let store_hi = cmp.ite(&d.data_hi_val, &d.current_hi, self.ctx);
                    self.cas_store_symbolic_data(py, callbacks, &d.addr_hi_expr, &store_hi, irsb)?;
                }
                Ok(())
            }
        }
    }

    /// Dispatch a CAS store: if the precomputed RustBV is concrete, synthesize
    /// an IRStmt::Store that reuses the full Store path (which re-evaluates the
    /// data IRExpr); if symbolic, route through `cas_store_symbolic_data`.
    #[allow(clippy::too_many_arguments)]
    fn cas_dispatch_store(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_expr: &IRExpr,
        data_expr: &IRExpr,
        data_bv: &RustBV,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        if data_bv.is_symbolic() {
            self.cas_store_symbolic_data(py, callbacks, addr_expr, data_bv, irsb)
        } else {
            let store_stmt = IRStmt::Store {
                addr: addr_expr.clone(),
                data: data_expr.clone(),
                endness,
            };
            self.execute_stmt_with_callbacks(py, callbacks, &store_stmt, irsb)?;
            Ok(())
        }
    }

    /// Store a symbolic data value at an address. Used by CAS when the value
    /// to store is a computed RustBV (e.g. ITE) that cannot be wrapped back
    /// into an IRExpr — see the `cas-llsc-recursion-limit` invariant.
    /// Routes through `memory_store_symbolic_value` for concrete addresses
    /// and `memory_store_symbolic_full` for symbolic addresses.
    fn cas_store_symbolic_data(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_expr: &IRExpr,
        data_bv: &RustBV,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr_expr, &irsb.tyenv)?;
        let data_size = data_bv.width().div_ceil(8) as usize;
        if let Some(addr_concrete) = addr_val.as_u64() {
            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
            if callbacks.has_memory_store_symbolic_value() {
                self.flush_stores(py, callbacks)?;
                callbacks
                    .call_memory_store_symbolic_value(py, addr_concrete, data_bv)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
            } else {
                self.pending_symbolic_stores
                    .insert(addr_concrete, data_bv.clone());
                let data_bytes = bv_to_bytes(data_bv);
                self.pending_stores.push(addr_concrete, data_bytes);
                if self.pending_stores.len() >= self.max_pending_stores {
                    self.flush_stores(py, callbacks)?;
                }
            }
        } else {
            self.flush_stores(py, callbacks)?;
            if callbacks.has_memory_store_symbolic_full() {
                callbacks
                    .call_memory_store_symbolic_full(py, &addr_val, data_bv)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
            } else {
                return Err(CbExecutionError::Unsupported(
                    "CAS with symbolic address: \
                     no memory_store_symbolic_full callback"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Fire a `mem_write` inspect callback into Python for this store.
    ///
    /// Gated on `inspect_event_enabled(MemWrite)` so the common case
    /// (no breakpoints) is a single bitmask test per Store. Symbolic
    /// addresses are skipped for the MVP (uq4n.4) — only concrete
    /// addresses dispatch; symbolic-address dispatch is a follow-up.
    /// The `when='after'` event is fired once the underlying memory
    /// write has completed; the BP receives the stored value AST as
    /// `mem_write_expr`. Errors from the Python callback are swallowed
    /// and logged on the Python side; we do not surface them up the
    /// interpreter stack so a user BP error cannot halt exploration.
    fn dispatch_mem_write_inspect(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
        endness: Endness,
    ) {
        // MemWrite = InspectEvent variant 1 — see crate::state::InspectEvent.
        if !callbacks.inspect_event_enabled(1) {
            return;
        }
        let Some(addr_u64) = addr_val.as_u64() else {
            return;
        };
        let endness_str = match endness {
            Endness::Little => "Iend_LE",
            Endness::Big => "Iend_BE",
        };
        let claripy_mod = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let value_ast = match crate::claripy_bridge::rustbv_to_claripy(py, data_val, &claripy_mod) {
            Ok(v) => v,
            Err(_) => return,
        };
        // Swallow errors — the Python dispatcher logs BP failures itself.
        let _ = callbacks.call_inspect_mem_write(
            py,
            self.current_state_id,
            "after",
            addr_u64,
            data_size as u32,
            Some(&value_ast),
            endness_str,
        );
    }

    /// Fire a `reg_write` inspect callback into Python for a VEX `Put`.
    /// Gated on `inspect_event_enabled(RegWrite)`. Dispatches `when='after'`
    /// with the stored value as `reg_write_expr`.
    fn dispatch_reg_write_inspect(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        offset: u32,
        size: u32,
        value: &RustBV,
    ) {
        // RegWrite = InspectEvent variant 3.
        if !callbacks.inspect_event_enabled(3) {
            return;
        }
        let claripy_mod = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let value_ast = match crate::claripy_bridge::rustbv_to_claripy(py, value, &claripy_mod) {
            Ok(v) => v,
            Err(_) => return,
        };
        let _ = callbacks.call_inspect_reg_write(
            py,
            self.current_state_id,
            "after",
            offset,
            size,
            Some(&value_ast),
        );
    }

    /// Fire a `tmp_write` inspect callback for a VEX `WrTmp` (angr-64pi).
    ///
    /// Gated on `inspect_event_enabled(14)` so the no-breakpoint case is
    /// one bitmask test per `WrTmp`. Dispatches `when='after'` with the
    /// written value as `tmp_write_expr`. Fires before the slot mutation
    /// only when a BP is registered; the mutation itself happens in the
    /// caller after this returns so the slot is consistent post-dispatch.
    fn dispatch_tmp_write_inspect(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        tmp_num: u32,
        value: &RustBV,
    ) {
        // TmpWrite bit assigned in _INSPECT_EVENT_SPECS.
        if !callbacks.inspect_event_enabled(14) {
            return;
        }
        let claripy_mod = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let value_ast = match crate::claripy_bridge::rustbv_to_claripy(py, value, &claripy_mod) {
            Ok(v) => v,
            Err(_) => return,
        };
        let _ = callbacks.call_inspect_tmp_write(
            py,
            self.current_state_id,
            "after",
            tmp_num,
            Some(&value_ast),
        );
    }

    /// Fire an `instruction` inspect callback into Python for a VEX `IMark`.
    /// Gated on bit 6 of the inspect-enabled bitmask. Bits 0..=5 mirror
    /// `crate::state::InspectEvent`; bit 6 is custom for the `instruction`
    /// event (no `InspectEvent` slot — angr Python exposes it but the Rust
    /// `InspectionManager` enum doesn't track it). Dispatches `when='before'`.
    fn dispatch_instruction_inspect(&self, py: Python<'_>, callbacks: &PythonCallbacks, addr: u64) {
        // Instruction = bit 6 (custom — not in the Rust InspectEvent enum).
        if !callbacks.inspect_event_enabled(6) {
            return;
        }
        let _ = callbacks.call_inspect_instruction(py, self.current_state_id, "before", addr);
    }

    /// Fire an `exit` inspect callback into Python for a VEX conditional `Exit`.
    /// Gated on the Exit bit (InspectEvent::Exit = 5). Dispatches `when='before'`
    /// with the branch target, jumpkind name (`Ijk_*`), and guard AST.
    fn dispatch_exit_inspect(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        target: u64,
        jk: JumpKind,
        guard: &RustBV,
    ) {
        // Exit = InspectEvent variant 5.
        if !callbacks.inspect_event_enabled(5) {
            return;
        }
        let claripy_mod = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let guard_ast = match crate::claripy_bridge::rustbv_to_claripy(py, guard, &claripy_mod) {
            Ok(v) => v,
            Err(_) => return,
        };
        let _ = callbacks.call_inspect_exit(
            py,
            self.current_state_id,
            "before",
            target,
            jk.ijk_name(),
            Some(&guard_ast),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vex::ir::{Endness, IRType, MBusEvent};

    fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
        VEXInterpreter::new(VexArch::AMD64, ctx)
    }

    fn make_irsb_with_temps(addr: u64, temp_types: &[IRType]) -> IRSB {
        let mut irsb = IRSB::new(addr, VexArch::AMD64);
        irsb.statements.push(IRStmt::IMark {
            addr,
            len: 4,
            delta: 0,
        });
        for ty in temp_types {
            irsb.tyenv.new_temp(*ty);
        }
        irsb
    }

    /// Initialize Python once for tests that need to call execute_stmt_with_callbacks.
    fn with_python<F, R>(f: F) -> R
    where
        F: FnOnce(Python<'_>, &PythonCallbacks) -> R,
    {
        Python::initialize();
        let callbacks = PythonCallbacks::new();
        Python::attach(|py| f(py, &callbacks))
    }

    #[test]
    fn noop_returns_continue() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        with_python(|py, cb| {
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &IRStmt::NoOp, &irsb)
                .expect("noop");
            assert!(matches!(res, StmtResult::Continue));
        });
    }

    #[test]
    fn imark_updates_current_insn() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x2000, &[]);
        with_python(|py, cb| {
            let stmt = IRStmt::IMark {
                addr: 0x2004,
                len: 4,
                delta: 0,
            };
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("imark");
            assert!(matches!(res, StmtResult::Continue));
            assert_eq!(interp.current_insn_addr, 0x2004);
            assert_eq!(interp.current_insn_len, 4);
        });
    }

    #[test]
    fn imark_at_hooked_address_returns_exit() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.add_hook(0x3000);
        let irsb = make_irsb_with_temps(0x3000, &[]);
        with_python(|py, cb| {
            let stmt = IRStmt::IMark {
                addr: 0x3000,
                len: 1,
                delta: 0,
            };
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("imark");
            match res {
                StmtResult::Exit { target, jumpkind } => {
                    assert_eq!(target, 0x3000);
                    assert!(matches!(jumpkind, JumpKind::Boring));
                }
                _ => panic!("expected Exit for hooked IMark"),
            }
        });
    }

    #[test]
    fn abihint_is_no_op() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::AbiHint {
            base: Box::new(IRExpr::Const(IRConst::U64(0))),
            len: 0,
            nia: Box::new(IRExpr::Const(IRConst::U64(0x1004))),
        };
        with_python(|py, cb| {
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("abihint");
            assert!(matches!(res, StmtResult::Continue));
        });
    }

    #[test]
    fn mbe_fence_is_continue() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::MBE(MBusEvent::Fence);
        with_python(|py, cb| {
            // MBE may fall through to default arm but should not error.
            let res = interp.execute_stmt_with_callbacks(py, cb, &stmt, &irsb);
            assert!(res.is_ok(), "MBE should not error");
        });
    }

    #[test]
    fn put_concrete_writes_register() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        // Write 0xcafe to RAX (offset 16 on AMD64).
        let stmt = IRStmt::Put {
            offset: 16,
            data: IRExpr::Const(IRConst::U64(0xcafe)),
        };
        with_python(|py, cb| {
            interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("put");
        });
        let val = interp.registers.get(16, 8, &ctx);
        assert_eq!(val.as_u64(), Some(0xcafe));
    }

    #[test]
    fn put_marks_register_dirty() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        assert_eq!(interp.dirty_registers, 0);
        let stmt = IRStmt::Put {
            offset: 16, // RAX -> bit index 4
            data: IRExpr::Const(IRConst::U64(1)),
        };
        with_python(|py, cb| {
            interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("put");
        });
        assert_ne!(interp.dirty_registers, 0);
        assert_eq!(interp.dirty_registers & (1u128 << 4), 1u128 << 4);
    }

    #[test]
    fn wrtmp_concrete_writes_temp() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
        interp.temps.resize(irsb.tyenv.types.len(), None);
        let stmt = IRStmt::WrTmp {
            tmp: 0,
            data: IRExpr::Const(IRConst::U32(0x1234)),
        };
        with_python(|py, cb| {
            interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("wrtmp");
        });
        let val = interp.temps[0].as_ref().expect("temp written");
        assert_eq!(val.as_u64(), Some(0x1234));
    }

    #[test]
    fn wrtmp_unknown_temp_errors() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // No temps allocated, but writing tmp 5.
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::WrTmp {
            tmp: 5,
            data: IRExpr::Const(IRConst::U32(0)),
        };
        with_python(|py, cb| {
            let res = interp.execute_stmt_with_callbacks(py, cb, &stmt, &irsb);
            assert!(res.is_err(), "should error on unknown temp");
            if let Err(err) = res {
                assert!(matches!(err, CbExecutionError::UnknownTemp(5)));
            }
        });
    }

    #[test]
    fn exit_with_concrete_false_guard_continues() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::Exit {
            guard: IRExpr::Const(IRConst::U1(false)),
            dst: 0x9000,
            jk: JumpKind::Boring,
            offsIP: 184,
        };
        with_python(|py, cb| {
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("exit");
            assert!(matches!(res, StmtResult::Continue));
        });
    }

    #[test]
    fn exit_with_concrete_true_guard_takes_branch() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::Exit {
            guard: IRExpr::Const(IRConst::U1(true)),
            dst: 0x9000,
            jk: JumpKind::Boring,
            offsIP: 184,
        };
        with_python(|py, cb| {
            let res = interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("exit");
            match res {
                StmtResult::Exit { target, .. } => assert_eq!(target, 0x9000),
                _ => panic!("expected Exit for true guard"),
            }
        });
    }

    #[test]
    fn store_to_concrete_addr_buffers_pending_store() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // Without rust_memory the fallback path takes a fast buffer route for
        // concrete addresses + concrete data: just append to pending_stores
        // (no callback invoked until flush).
        assert!(!interp.use_rust_memory);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        let stmt = IRStmt::Store {
            addr: IRExpr::Const(IRConst::U64(0x4000)),
            data: IRExpr::Const(IRConst::U32(0xdead_beef)),
            endness: Endness::Little,
        };
        with_python(|py, cb| {
            interp
                .execute_stmt_with_callbacks(py, cb, &stmt, &irsb)
                .expect("store should buffer without erroring");
        });
        // Buffered into pending_stores keyed by address.
        let data = interp
            .pending_stores
            .try_load(0x4000, 4)
            .expect("pending store at 0x4000");
        assert_eq!(data, &[0xef, 0xbe, 0xad, 0xde]);
    }

    #[test]
    fn loadg_unknown_cvt_surfaces_invalid_ir() {
        use crate::vex::ir::IRLoadGOp;
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // dst temp t0 is I32.
        let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
        let stmt = IRStmt::LoadG {
            dst: 0,
            addr: Box::new(IRExpr::Const(IRConst::U64(0x4000))),
            alt: Box::new(IRExpr::Const(IRConst::U32(0))),
            guard: Box::new(IRExpr::Const(IRConst::U8(1))),
            cvt: IRLoadGOp::Unknown,
            endness: Endness::Little,
        };
        with_python(|py, cb| {
            let res = interp.execute_stmt_with_callbacks(py, cb, &stmt, &irsb);
            // An unrecognized cvt must error rather than silently load+Identity.
            assert!(
                matches!(res, Err(CbExecutionError::InvalidIR(_))),
                "Unknown LoadG cvt should surface InvalidIR"
            );
        });
    }

    #[test]
    fn put_at_high_offset_does_not_overflow_dirty_mask() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x1000, &[]);
        // Offset 600 -> bit index 150, exceeds 128-bit mask. Should NOT panic.
        let stmt = IRStmt::Put {
            offset: 600,
            data: IRExpr::Const(IRConst::U8(0xaa)),
        };
        with_python(|py, cb| {
            let res = interp.execute_stmt_with_callbacks(py, cb, &stmt, &irsb);
            assert!(res.is_ok(), "high-offset Put should not overflow");
        });
    }
}
