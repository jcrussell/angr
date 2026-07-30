use super::helpers::bv_to_bytes;
use super::statements_cas::CasArgs;
use super::*;

/// The `IRStmt::LoadG` operands, bundled so the handler takes one borrow of the
/// statement's fields instead of five positional params.
pub(super) struct LoadGArgs<'s> {
    pub(super) dst: &'s u32,
    pub(super) guard: &'s IRExpr,
    pub(super) addr: &'s IRExpr,
    pub(super) alt: &'s IRExpr,
    pub(super) cvt: &'s IRLoadGOp,
}

impl<'a> VEXInterpreter<'a> {
    /// Execute a single statement using Python callbacks.
    pub(super) fn execute_stmt_with_callbacks(
        &mut self,
        callbacks: &PythonCallbacks,
        stmt: &IRStmt,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark { addr, len, .. } => {
                self.current_insn_addr = *addr;
                self.current_insn_len = *len;
                self.dispatch_instruction_inspect(callbacks, *addr);
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
                let value = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
                let size = value.width().div_ceil(8);
                self.dispatch_reg_write_inspect(callbacks, *offset, size, &value);
                self.registers.put(*offset, value);
                self.mark_register_dirty(*offset);

                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
                if (*tmp as usize) < self.temps.len() {
                    self.dispatch_tmp_write_inspect(callbacks, *tmp, &value);
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
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let mut data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
                let data_size = data_val.width().div_ceil(8) as usize;
                if self.profiling_enabled {
                    self.stats.store_stmt_count += 1;
                }

                // angr-inh0: fire mem_write BP_BEFORE so a user BP overriding
                // state.inspect.mem_write_expr injects the value pre-commit.
                // The width-guarded override (if any) replaces data_val before
                // the store; data_size is unchanged because the guard rejects a
                // width mismatch.
                if let Some(injected) = self.dispatch_mem_write_inspect(
                    callbacks, &addr_val, &data_val, data_size, *endness, "before",
                ) {
                    data_val = injected;
                }

                if self.use_rust_memory
                    && self.try_rust_memory_store(
                        callbacks,
                        &addr_val,
                        &data_val,
                        data_size,
                        store_start,
                    )?
                {
                    // SymbolicMemory::store_concrete already bumped record_mem_store.
                    self.dispatch_mem_write_inspect(
                        callbacks, &addr_val, &data_val, data_size, *endness, "after",
                    );
                    return Ok(StmtResult::Continue);
                }

                // angr-obrm: callback-path stores bypass SymbolicMemory, so
                // bump the global mem_store counter here for parity with
                // the Rust-memory path.
                record_mem_store(data_size as u64);
                self.fallback_to_python_store(callbacks, &addr_val, data_val.clone(), data_size)?;
                self.dispatch_mem_write_inspect(
                    callbacks, &addr_val, &data_val, data_size, *endness, "after",
                );
                Ok(StmtResult::Continue)
            }

            IRStmt::Exit { guard, dst, jk, .. } => {
                let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
                self.dispatch_exit_inspect(callbacks, *dst, *jk, &guard_val);

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
                    let fallthrough = self.eval_next_addr(callbacks, irsb)?;
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
                        // These prev-fork assumes exist ONLY to make
                        // `check_branch_feasibility` accurate inside the bare
                        // block-solver `push()` scope. Their Z3 assertions are
                        // discarded by the matching `pop()` at block end, but
                        // `pop()` does not truncate the `assumed` export log, so
                        // without an explicit truncate they would leak into the
                        // log that a wave migration re-asserts verbatim — the
                        // ype54 poison (angr-ype54). Bracket every assume with a
                        // savepoint + truncate so the log stays clean while the
                        // Z3 solver still sees the constraints for feasibility.
                        #[cfg(feature = "vex-engine-z3")]
                        let assumed_savepoint = self.ctx.assumed_local_len();
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
                        #[cfg(feature = "vex-engine-z3")]
                        self.ctx.truncate_assumed_local(assumed_savepoint);
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
                    self.stored_conditions.insert(cond_id, guard_val);

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
                            memory: self
                                .rust_memory
                                .as_ref()
                                .map(super::super::memory::SymbolicMemory::fork),
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
                let ix_val = self.eval_expr_with_callbacks(callbacks, ix, &irsb.tyenv)?;

                // PutI requires a concrete index to compute the register offset
                let idx = if let Some(idx) = ix_val.as_u64() {
                    idx
                } else {
                    // Symbolic index - concretize using the solver and pin the
                    // choice with an equality constraint so a later solve cannot
                    // pick a different index, which would make this register
                    // write inconsistent with the path constraints (unsound).
                    self.concretize_and_pin(&ix_val).ok_or_else(|| {
                        CbExecutionError::Unsupported(
                            "PutI index concretization failed".to_string(),
                        )
                    })?
                };

                // Calculate the rotating register offset:
                // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
                let elem_size = descr.elemTy.bytes();
                let index = ((idx as u32).wrapping_add(*bias)) % descr.nElems;
                let offset = descr.base + index * elem_size;

                // Evaluate the data to write
                let data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;

                // Write to the register file
                self.registers.put(offset, data_val);
                self.mark_register_dirty(offset);

                Ok(StmtResult::Continue)
            }

            IRStmt::StoreG {
                guard, addr, data, ..
            } => self.handle_storeg(callbacks, guard, addr, data, irsb),

            IRStmt::LoadG {
                dst,
                guard,
                addr,
                alt,
                cvt,
                ..
            } => self.handle_loadg(
                callbacks,
                LoadGArgs {
                    dst,
                    guard,
                    addr,
                    alt,
                    cvt,
                },
                irsb,
            ),

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
                callbacks,
                &CasArgs {
                    old_hi: *old_hi,
                    old_lo: *old_lo,
                    addr,
                    expd_hi: expdHi.as_deref(),
                    expd_lo: expdLo,
                    data_hi: dataHi.as_deref(),
                    data_lo: dataLo,
                    endness: *endness,
                },
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
                                "LLSC result temp {result} not in tyenv"
                            ))
                        })?;
                        let load_expr = IRExpr::Load {
                            addr: addr.clone(),
                            ty: result_ty,
                            endness: *endness,
                        };
                        let value =
                            self.eval_expr_with_callbacks(callbacks, &load_expr, &irsb.tyenv)?;
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
                        self.execute_stmt_with_callbacks(callbacks, &store_stmt, irsb)?;
                        if (*result as usize) < self.temps.len() {
                            self.temps[*result as usize] = Some(RustBV::concrete(1, 1));
                        } else {
                            return Err(CbExecutionError::UnknownTemp(*result));
                        }
                    }
                }
                Ok(StmtResult::Continue)
            }
            IRStmt::Dirty(dirty) => self.handle_dirty_call(callbacks, dirty, irsb),
        }
    }

    /// Execute an `IRStmt::StoreG` (guarded store): guard tristate handling
    /// (symbolic always-false/always-true/both via ITE, plus concrete guard),
    /// with symbolic-address concretization fallbacks. Extracted verbatim from
    /// `execute_stmt_with_callbacks` (cudgw.18).
    fn handle_storeg(
        &mut self,
        callbacks: &PythonCallbacks,
        guard: &IRExpr,
        addr: &IRExpr,
        data: &IRExpr,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        // Evaluate guard condition
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;

        // Check if guard is symbolic
        if guard_val.is_symbolic() {
            // Symbolic guard: need to handle conditional store
            // For now, check if guard can be true at all
            if !self.ctx.can_be_true(&guard_val) {
                // Guard is always false - skip store
                return Ok(StmtResult::Continue);
            }
            if !self.ctx.can_be_false(&guard_val) {
                // Guard is always true - perform store unconditionally, which is
                // semantically a plain Store. Route through the shared
                // dispatcher so it gets code-cache invalidation and symbolic-
                // shadow eviction (and handles symbolic addresses, which the old
                // inline path silently dropped) — angr-myzjx.26.
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
                let data_size = data_val.width().div_ceil(8) as usize;
                self.store_value(callbacks, &addr_val, data_val, data_size)?;
                return Ok(StmtResult::Continue);
            }
            // Both paths possible with symbolic guard - use ITE for conditional store
            // Store ITE(guard, new_data, current_data)
            let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
            let data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
            let data_size = data_val.width().div_ceil(8) as usize;

            if let Some(addr_concrete) = addr_val.as_u64() {
                // Load current value at address
                let current = self.load_from_callback(callbacks, addr_concrete, data_size)?;
                // Create ITE: if guard then new_data else current
                let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                // The ITE captured `current` above; now that we're about to
                // overwrite this range, invalidate stale cached code and evict
                // overlapping symbolic shadows (angr-myzjx.26).
                self.invalidate_and_evict_concrete_store(addr_concrete, data_size);
                // ITE result is symbolic if guard or either operand is symbolic
                if ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                    self.flush_stores(callbacks)?;
                    callbacks
                        .call_memory_store_symbolic_value(addr_concrete, &ite_result)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                } else {
                    let ite_bytes = bv_to_bytes(&ite_result);
                    self.pending_stores.push(addr_concrete, ite_bytes);
                    if self.pending_stores.len() >= self.max_pending_stores {
                        self.flush_stores(callbacks)?;
                    }
                }
            } else {
                // Symbolic address with symbolic guard - concretize for write.
                // Invalidate cached code at the concretized target(s) before
                // dispatching (self-modifying-code support), mirroring
                // handle_symbolic_store (angr-myzjx.26).
                let concret_result = self.concretize_cached_write(&addr_val);
                self.invalidate_code_on_store(&concret_result, data_size);
                match &*concret_result {
                    ConcretizationResult::Single(addr_concrete) => {
                        let addr_concrete = *addr_concrete;
                        // Load current value and use ITE
                        let current =
                            self.load_from_callback(callbacks, addr_concrete, data_size)?;
                        let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                        // ITE captured `current`; evict stale overlapping
                        // symbolic shadows before storing the new value.
                        self.evict_overlapping_symbolic_stores(addr_concrete, data_size);
                        self.flush_stores(callbacks)?;
                        // ITE result is symbolic - use symbolic store callback
                        if ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                            callbacks
                                .call_memory_store_symbolic_value(addr_concrete, &ite_result)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        } else {
                            let ite_bytes = bv_to_bytes(&ite_result);
                            callbacks
                                .call_memory_store(addr_concrete, &ite_bytes)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                    }
                    _ => {
                        // Symbolic guard + non-Single address solutions
                        // (Multiple/Strided/TooLarge/Failed). Combining the
                        // guard-ITE with per-address ITEs requires a per-
                        // address load and is brittle, so delegate to Python's
                        // full symbolic store callback which has access to
                        // angr's address concretization strategies.
                        self.flush_stores(callbacks)?;
                        if callbacks.has_memory_store_symbolic_full() {
                            callbacks
                                .call_memory_store_symbolic_full(&addr_val, &data_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        } else {
                            return Err(CbExecutionError::Unsupported(
                                "guarded store with symbolic address: \
                                         no memory_store_symbolic_full callback"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
            return Ok(StmtResult::Continue);
        }

        // Concrete guard: simple check
        if let Some(g) = guard_val.as_u64()
            && g != 0
        {
            // Guard is true - the store is semantically identical to a plain
            // IRStmt::Store of data_val at addr_val. Route through the shared
            // dispatcher (angr-myzjx.26): this replaces a near-verbatim clone of
            // handle_symbolic_store's concretization ladder that (a) never
            // invalidated the code cache or evicted overlapping symbolic
            // shadows, and (b) stored concrete data at a symbolic address via
            // call_memory_store(0, ..) — a hard-coded address 0.
            let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
            let data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
            let data_size = data_val.width().div_ceil(8) as usize;
            self.store_value(callbacks, &addr_val, data_val, data_size)?;
        }
        // Guard is false - skip the store

        Ok(StmtResult::Continue)
    }

    /// Execute an `IRStmt::LoadG` (guarded load): guard tristate handling
    /// (symbolic always-true/always-false/both via ITE, plus concrete guard),
    /// with cvt-based widening conversions. Extracted verbatim from
    /// `execute_stmt_with_callbacks` (cudgw.18).
    fn handle_loadg(
        &mut self,
        callbacks: &PythonCallbacks,
        args: LoadGArgs<'_>,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        let LoadGArgs {
            dst,
            guard,
            addr,
            alt,
            cvt,
        } = args;
        // Evaluate guard condition
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;

        // Evaluate the alternative value (used when guard is false)
        let alt_val = self.eval_expr_with_callbacks(callbacks, alt, &irsb.tyenv)?;

        // Determine the load size from the destination temp type
        let dst_ty = irsb.tyenv.get(*dst).ok_or_else(|| {
            CbExecutionError::InvalidIR(format!("LoadG destination temp {dst} not in tyenv"))
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
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let loaded = self.resolve_loadg_load(
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
            let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
            let loaded =
                self.resolve_loadg_load(callbacks, &addr_val, load_size, "LoadG (symbolic guard)")?;

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
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let loaded = self.resolve_loadg_load(
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

    /// Execute an `IRStmt::Dirty` call: guard handling, native dirty-helper
    /// dispatch, and the Python-callback / fresh-symbolic fallbacks. Extracted
    /// verbatim from `execute_stmt_with_callbacks` (cudgw.18).
    fn handle_dirty_call(
        &mut self,
        callbacks: &PythonCallbacks,
        dirty: &crate::vex::ir::IRDirty,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        // Check guard if present
        if let Some(guard) = &dirty.guard {
            let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
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
            let val = self.eval_expr_with_callbacks(callbacks, arg, &irsb.tyenv)?;
            if let Some(concrete) = val.as_u64() {
                arg_vals.push(concrete);
            } else if let Some(concrete) = self.concretize_and_pin(&val) {
                arg_vals.push(concrete);
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
                let stub =
                    RustBV::symbolic(self.ctx, format!("dirty_{}_stub", dirty.cee.name), bits);
                if (tmp as usize) < self.temps.len() {
                    self.temps[tmp as usize] = Some(stub);
                }
            }
            return Ok(StmtResult::Continue);
        }

        if !all_args_concrete {
            // The arg-concretization loop above bailed early because the
            // solver could not produce a concrete value for one of the
            // args (UNSAT). `concretize_and_pin` only fails on an empty
            // model, and the pins added for the preceding args merely
            // tighten the constraint set — re-running the loop would fail
            // on the same arg. So there is nothing more aggressive to try;
            // surface a clear error and let Python apply its own fallback.
            return Err(CbExecutionError::Unsupported(format!(
                "dirty call '{}' arg unconcretizable",
                dirty.cee.name
            )));
        }

        // Call Python callback
        self.stats.python_dirty_call_count += 1;
        let (data, is_symbolic, _symbolic_ast) = callbacks
            .call_dirty_call(&dirty.cee.name, &arg_vals, ret_ty_bits)
            .map_err(|e| {
                CbExecutionError::Callback(format!("dirty call {} failed: {}", dirty.cee.name, e))
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

#[cfg(test)]
#[path = "statements_tests.rs"]
mod statements_tests;
