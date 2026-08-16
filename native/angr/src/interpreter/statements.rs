//! `IRStmt` execution — the write half of VEX execution.
//!
//! `execute_stmt_with_callbacks` is the per-statement dispatch point;
//! everything reachable only from one arm of it either sits below
//! (`handle_exit_stmt`, `handle_storeg`, `handle_loadg`, `handle_dirty_call`)
//! or has its own file: `Ist_Store` in `statements_store.rs`, `Ist_CAS` in
//! `statements_cas.rs`, and the `state.inspect` write dispatchers in
//! `statements_inspect.rs`.
//!
//! The one type defined here rather than in `mod.rs` is [`GuardClass`], the
//! shared tristate the four guarded statements (`Exit`, `StoreG`, `LoadG`,
//! `Dirty`) classify their guard with; see its own docs for why `Exit` uses
//! the two decision rules but not the `classify_guard` wrapper.

use super::bv_utils::{bv_to_bytes, bytes_to_bv, reject_symbolic_byte_store};
use super::expressions::fabricate_unsupported_irop;
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

/// Tristate classification of a guarded statement's guard expression.
///
/// `Exit`, `StoreG`, `LoadG` and `Dirty` all ask the same two questions of
/// their guard — can it be true, can it be false — and each used to re-derive
/// the answer its own way (angr-12jjk.22). The rule now lives here once, so a
/// fix to it lands in every handler instead of in one copy out of four.
///
/// `Exit` shares the two decision rules (`GuardClass::concrete` /
/// `GuardClass::from_feasibility`) but not the `classify_guard` wrapper: it
/// must not query the solver at all in non-deferred mode, and in deferred mode
/// it needs the raw feasibility pair after its own incremental-assertion
/// preamble — see `handle_exit_stmt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardClass {
    /// The guard cannot be true: a concrete zero, or a symbolic guard the
    /// solver proves always-false. A doubly-infeasible guard (neither
    /// direction sat, i.e. an already-unsat state) also lands here — the
    /// state is dead, so the arm it picks is immaterial.
    Never,
    /// The guard must be true: a concrete non-zero, or a symbolic guard the
    /// solver proves always-true. The statement runs unconditionally, exactly
    /// as its unguarded counterpart would.
    Always,
    /// Both directions are feasible — the guard is genuinely symbolic and the
    /// handler must model both outcomes (an ITE, or a fork).
    Symbolic,
}

impl GuardClass {
    /// Decide a guard without consulting the solver, or `None` when it is
    /// genuinely symbolic and only the solver can answer.
    pub(super) fn concrete(guard_val: &RustBV) -> Option<Self> {
        // A `Constrained` value is still `is_symbolic()` even when it carries a
        // concrete value, so test both before deciding.
        if guard_val.is_symbolic() {
            return None;
        }
        guard_val
            .as_u64()
            .map(|g| if g != 0 { Self::Always } else { Self::Never })
    }

    /// Fold a `(can_be_true, can_be_false)` feasibility pair into the tristate.
    pub(super) fn from_feasibility(can_be_true: bool, can_be_false: bool) -> Self {
        match (can_be_true, can_be_false) {
            (false, _) => Self::Never,
            (true, false) => Self::Always,
            (true, true) => Self::Symbolic,
        }
    }
}

impl<'a> VEXInterpreter<'a> {
    /// Classify a guard value into [`GuardClass`], asking the solver only when
    /// the guard is not concretely decidable.
    ///
    /// The solver query is `check_branch_feasibility`, which answers both
    /// directions under a single solver acquisition — cheaper than the
    /// `can_be_true()` + `can_be_false()` pair the guarded-load/store handlers
    /// used to call (each of which is itself a full `check_branch_feasibility`).
    pub(super) fn classify_guard(&self, guard_val: &RustBV) -> GuardClass {
        if let Some(class) = GuardClass::concrete(guard_val) {
            return class;
        }
        let (can_be_true, can_be_false) = self.ctx.check_branch_feasibility(guard_val);
        GuardClass::from_feasibility(can_be_true, can_be_false)
    }

    /// Write `value` into VEX temp slot `tmp`, erroring when the index is out
    /// of range for the temps vector sized from this IRSB's tyenv.
    ///
    /// The single write path for the statement handlers (`WrTmp`, `LLSC`,
    /// `LoadG`, `CAS`, `Dirty`) — every one of them previously hand-wrote its
    /// own `if (tmp as usize) < self.temps.len()` guard, and they had drifted
    /// into two behaviours: erroring vs. silently dropping the computed value
    /// (angr-12jjk.18 / angr-sqfj8.69). An out-of-range slot means the lifted
    /// IR disagrees with its own tyenv, which is a lifter trust-boundary bug,
    /// so the one policy here is to fail loud with `UnknownTemp`. A future
    /// caller that genuinely wants to drop the write must say so explicitly
    /// with a `SILENT(cat-x)` tag rather than by omitting an `else` arm.
    pub(super) fn write_tmp(&mut self, tmp: u32, value: RustBV) -> Result<(), CbExecutionError> {
        let slot = self
            .temps
            .get_mut(tmp as usize)
            .ok_or(CbExecutionError::UnknownTemp(tmp))?;
        *slot = Some(value);
        Ok(())
    }

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

                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
                self.write_tmp(*tmp, value)?;
                // Dispatched after the store (the callback's `when` is
                // "after") and by re-borrowing the slot rather than cloning
                // `value`, which would cost a refcount bump on the hottest
                // statement in the IR.
                if let Some(written) = &self.temps[*tmp as usize] {
                    self.dispatch_tmp_write_inspect(callbacks, *tmp, written);
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
                    if self.profiling_enabled {
                        self.stats.rust_memory_store_count += 1;
                    }
                    self.dispatch_mem_write_inspect(
                        callbacks, &addr_val, &data_val, data_size, *endness, "after",
                    );
                    return Ok(StmtResult::Continue);
                }

                if self.profiling_enabled {
                    self.stats.fallback_memory_store_count += 1;
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
                let exit_start = profile_start!(self);
                if self.profiling_enabled {
                    self.stats.exit_stmt_count += 1;
                }
                let result = self.handle_exit_stmt(callbacks, irsb, guard, *dst, *jk);
                profile_add!(exit_start, self.stats.exit_stmt_time_ns);
                result
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
                let (offset, _elem_size) = self.regarray_offset(descr, &ix_val, *bias, "PutI")?;

                // Evaluate the data to write
                let data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;

                // Write to the register file
                self.registers.put(offset, data_val);

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
                        self.write_tmp(*result, value)?;
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
                        self.write_tmp(*result, RustBV::concrete(1, 1))?;
                    }
                }
                Ok(StmtResult::Continue)
            }
            IRStmt::Dirty(dirty) => self.handle_dirty_call(callbacks, dirty, irsb),
        }
    }

    /// `IRStmt::Exit` — the conditional-branch statement.
    ///
    /// Split out of `execute_stmt_with_callbacks`'s match arm (angr-sqfj8.64)
    /// so the caller can bracket the whole handler with `profile_start!` /
    /// `profile_add!` and finally feed `exit_stmt_time_ns`. The handler has
    /// six `return` sites plus a `?`, so timing it in place would have meant a
    /// `profile_add!` before every one of them.
    fn handle_exit_stmt(
        &mut self,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
        guard: &IRExpr,
        dst: u64,
        jk: JumpKind,
    ) -> Result<StmtResult, CbExecutionError> {
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
        self.dispatch_exit_inspect(callbacks, dst, jk, &guard_val);

        // Decide the guard without the solver where possible (note that a
        // Constrained value has a concrete value but is still symbolic, so
        // `GuardClass::concrete` declines it).
        match GuardClass::concrete(&guard_val) {
            Some(GuardClass::Always) => {
                return Ok(StmtResult::Exit {
                    target: dst,
                    jumpkind: jk,
                });
            }
            Some(GuardClass::Never) => return Ok(StmtResult::Continue),
            // Not concretely decidable — fall through to the symbolic path.
            Some(GuardClass::Symbolic) | None => {}
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
                true_target: dst,
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
                // block-solver `push()` scope. The matching `pop()` at
                // block end does discard both their Z3 assertions and
                // their `assumed` export-log entries (`scope_savepoint_pop`
                // truncates `local.assumed`, angr-ph300.41/.42) — but that
                // pop only runs at block teardown, while the `self.ctx.fork()`
                // taken for the `BranchSnapshot` below happens mid-scope,
                // right here. A snapshot forked with these assumes still in
                // the log would carry them into a wave migration that
                // re-asserts the log verbatim — the ype54 poison
                // (angr-ype54). Bracket every assume with a savepoint +
                // truncate so the log is clean before that fork while the
                // Z3 solver still sees the constraints for feasibility.
                #[cfg(feature = "vex-engine-z3")]
                let assumed_savepoint = self.ctx.assumed_local_len();
                if !self.block_solver_pushed {
                    // First time in this block with prior forks: push and assert all
                    self.ctx.push();
                    self.block_solver_pushed = true;
                    for prev_fork in &self.deferred_forks {
                        if let Some(cond) = self.stored_conditions.get(&prev_fork.condition_id) {
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
                        if let Some(cond) = self.stored_conditions.get(&prev_fork.condition_id) {
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
        match GuardClass::from_feasibility(can_be_true, can_be_false) {
            GuardClass::Symbolic => {
                // The unexplored path (guard=false) should resume at the
                // next instruction after this conditional jump, NOT the
                // block's default exit. When a VEX IRSB contains multiple
                // Ist_Exit statements, using the block fallthrough would
                // skip all code between this exit and the end of the block.
                let false_target = self
                    .current_insn_addr
                    .wrapping_add(self.current_insn_len as u64);

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
                let is_backward_branch = dst < self.current_insn_addr;

                let deferred = if is_backward_branch {
                    DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: true, // we took the exit (guard=true) path
                        unexplored_target: false_target, // fall-through deferred
                        condition_id: cond_id,
                        condition_ast,
                    }
                } else {
                    DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: false, // we took the fallthrough (guard=false) path
                        unexplored_target: dst, // the exit target is deferred
                        condition_id: cond_id,
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
                    Ok(StmtResult::Exit {
                        target: dst,
                        jumpkind: jk,
                    })
                } else {
                    // Continue execution on the fallthrough path
                    Ok(StmtResult::Continue)
                }
            }
            // Guard must be true — take the exit unconditionally.
            GuardClass::Always => Ok(StmtResult::Exit {
                target: dst,
                jumpkind: jk,
            }),
            // Guard must be false — fall through past the exit.
            GuardClass::Never => Ok(StmtResult::Continue),
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

        match self.classify_guard(&guard_val) {
            // Guard is false - skip the store.
            GuardClass::Never => return Ok(StmtResult::Continue),
            GuardClass::Always => {
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
                return Ok(StmtResult::Continue);
            }
            GuardClass::Symbolic => {
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
                        reject_symbolic_byte_store(&ite_result, addr_concrete, "StoreG")?;
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
                            if ite_result.is_symbolic()
                                && callbacks.has_memory_store_symbolic_value()
                            {
                                callbacks
                                    .call_memory_store_symbolic_value(addr_concrete, &ite_result)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            } else {
                                reject_symbolic_byte_store(
                                    &ite_result,
                                    addr_concrete,
                                    "StoreG (concretized addr)",
                                )?;
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
            }
        }

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

        let result = match self.classify_guard(&guard_val) {
            // Guard is false - the load never happens; take the alt value.
            GuardClass::Never => alt_val,
            GuardClass::Always => {
                // Guard is true - perform the load unconditionally.
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let loaded =
                    self.resolve_loadg_load(callbacks, &addr_val, load_size, "LoadG (true guard)")?;
                self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits())
            }
            GuardClass::Symbolic => {
                // Both paths possible - evaluate address and load, then ITE
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let loaded = self.resolve_loadg_load(
                    callbacks,
                    &addr_val,
                    load_size,
                    "LoadG (symbolic guard)",
                )?;

                // Apply conversion to loaded value
                let converted = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());

                // Create ITE: if guard then loaded else alt
                guard_val.ite(&converted, &alt_val, self.ctx)
            }
        };

        self.write_tmp(*dst, result)?;

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
            match self.classify_guard(&guard_val) {
                // Guard is false - skip the dirty call.
                GuardClass::Never => return Ok(StmtResult::Continue),
                // Guard must be true - fall through and execute it.
                GuardClass::Always => {}
                GuardClass::Symbolic => {
                    // Both feasible: we can't fork mid-block, so pin the guard
                    // true and run the call. Loses the not-taken branch but
                    // matches angr's existing dirty-helper concretization.
                    log::debug!(
                        "dirty call '{}': symbolic guard concretized to taken branch",
                        dirty.cee.name
                    );
                    self.ctx.assume_true(&guard_val);
                }
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

        // Determine return type bits. A dirty call that names a result temp
        // must have that temp in the block's tyenv; a missing entry is
        // malformed IR, not a "assume 64" case. `ret_ty_bits` feeds both the
        // native-handler write path (`RustBV::concrete` below) and the
        // Python-callback path (`bytes_to_bv`), so defaulting silently
        // produces a wrong-width tmp that propagates instead of erroring.
        // Fail loud, matching the `IRStmt::LLSC` arm of
        // `execute_stmt_with_callbacks`, which does the same lookup for the
        // same condition (angr-03vl4.33).
        let ret_ty_bits = if let Some(tmp) = dirty.tmp {
            irsb.tyenv
                .get(tmp)
                .ok_or_else(|| {
                    CbExecutionError::InvalidIR(format!(
                        "dirty call '{}': result temp {tmp} not in tyenv",
                        dirty.cee.name
                    ))
                })?
                .bits()
        } else {
            0 // No return value
        };

        // Try native dirty helper dispatch first
        if all_args_concrete
            && let Some(result) = self.dirty_dispatch.try_call(
                &mut self.dirty_helper_state,
                &dirty.cee.name,
                &arg_vals,
            )
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
                self.write_tmp(tmp, value)?;
            }

            // Apply any register writes from the helper
            for (offset, value) in result.reg_writes {
                // Convert u64 value to RustBV and store in register
                let bv = RustBV::concrete(value as u128, 64);
                self.registers.put(offset, bv);
            }

            return Ok(StmtResult::Continue);
        }

        // No native handler matched and Python has no `dirty_call` callback
        // registered either, so nobody in this process models the helper.
        // Follow the same policy as `vex_op_fallback` / `eval_ccall`
        // (angr-c7xno.44): route the block to Python's VEX engine by default,
        // and only fabricate a fresh unconstrained symbolic under the opt-in
        // `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` gate. Fabricating by default
        // silently diverges — an unconstrained tmp makes every downstream
        // condition over it explore both branches regardless of what the
        // helper really computes.
        if !callbacks.has_dirty_call() {
            if !fabricate_unsupported_irop() {
                return Err(CbExecutionError::NeedPythonFallback(format!(
                    "dirty call '{}': no native handler and no Python callback",
                    dirty.cee.name
                )));
            }
            log::warn!(
                "dirty call '{}': no native handler and no Python callback; \
                         stubbing with a fresh symbolic tmp \
                         (ANGR_RUST_FABRICATE_UNSUPPORTED_IROP)",
                dirty.cee.name
            );
            self.stats.vex_bypass_fabricate_count += 1;
            if let Some(tmp) = dirty.tmp {
                // `ret_ty_bits` is the tyenv width of exactly this temp, and
                // `IRType::bits` never yields 0, so the 0 sentinel means
                // "no result temp" and cannot be reached inside this arm.
                let stub = RustBV::symbolic(
                    self.ctx,
                    format!("dirty_{}_stub", dirty.cee.name),
                    ret_ty_bits,
                );
                self.write_tmp(tmp, stub)?;
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
                // Convert the little-endian callback bytes to a concrete value.
                bytes_to_bv(&data, ret_ty_bits)
            };

            self.write_tmp(tmp, result)?;
        }

        Ok(StmtResult::Continue)
    }
}

test_submod!("statements_tests.rs" => statements_tests);
