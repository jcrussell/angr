//! `IRStmt` execution — the write half of VEX execution.
//!
//! `execute_stmt_with_callbacks` is the per-statement dispatch point;
//! everything reachable only from one arm of it either sits below
//! (`handle_exit_stmt`, `handle_storeg`, `handle_loadg`) or has its own file:
//! `Ist_Store` in `statements_store.rs`, `Ist_CAS` in `statements_cas.rs`,
//! `Ist_Dirty` in `statements_dirty.rs`, and the `state.inspect` write
//! dispatchers in `statements_inspect.rs`.
//!
//! The one type defined here rather than in `mod.rs` is [`GuardClass`], the
//! shared tristate the four guarded statements (`Exit`, `StoreG`, `LoadG`,
//! `Dirty`) classify their guard with; see its own docs for why `Exit` uses
//! the two decision rules but not the `classify_guard` wrapper.

use super::statements_cas::CasArgs;
use super::statements_store::GuardedStoreAddr;
use super::*;

/// The `IRStmt::LoadG` operands, bundled so the handler takes one borrow of the
/// statement's fields instead of five positional params.
pub(super) struct LoadGArgs<'s> {
    pub(super) dst: &'s u32,
    pub(super) guard: &'s IRExpr,
    pub(super) addr: &'s IRExpr,
    pub(super) alt: &'s IRExpr,
    pub(super) cvt: &'s IRLoadGOp,
    pub(super) endness: Endness,
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
        // Only a `Concrete` guard decides here; an `Ast` is `is_symbolic()`.
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
    /// with a tagged discard rather than by omitting an `else` arm.
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
        callbacks: &dyn EngineHost,
        stmt: &IRStmt,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark { addr, len, .. } => {
                self.current_insn_addr = *addr;
                self.current_insn_len = *len;
                self.dispatch_instruction_inspect(callbacks, *addr);
                // Check for hooks at this address. The skipped hook's own
                // instruction runs once; past it the hook is live again.
                if self.skip_hook_once == Some(*addr) {
                    self.skip_hook_once = None;
                } else if self.is_hooked(*addr)
                    || (*addr != irsb.addr && self.lift_cuts.contains(addr))
                {
                    // A lift cut past the block's start ends it there, for a
                    // block the lift window could not cut (a Python-callback
                    // lift) (angr-mx00p.132.60).
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
                    callbacks, &addr_val, &data_val, data_size, *endness, None, "before",
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
                        callbacks, &addr_val, &data_val, data_size, *endness, None, "after",
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
                    callbacks, &addr_val, &data_val, data_size, *endness, None, "after",
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
                guard,
                addr,
                data,
                endness,
            } => self.handle_storeg(callbacks, guard, addr, data, *endness, irsb),

            IRStmt::LoadG {
                dst,
                guard,
                addr,
                alt,
                cvt,
                endness,
            } => self.handle_loadg(
                callbacks,
                LoadGArgs {
                    dst,
                    guard,
                    addr,
                    alt,
                    cvt,
                    endness: *endness,
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
        callbacks: &dyn EngineHost,
        irsb: &IRSB,
        guard: &IRExpr,
        dst: u64,
        jk: JumpKind,
    ) -> Result<StmtResult, CbExecutionError> {
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
        self.dispatch_exit_inspect(callbacks, dst, jk, &guard_val);

        // Decide the guard without the solver where possible.
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
        let trap_side = TrapSide::of(jk, self.lazy_solves);
        if !self.config.use_deferred_forks {
            // The eager branch event carries no jumpkind, so its resume would
            // mint the trap side as an ordinary path (angr-vmi2l.160). Python's
            // VEX engine applies the trap rules itself.
            if trap_side.is_some() {
                return Err(CbExecutionError::NeedPythonFallback(format!(
                    "symbolic-guard {} exit under eager forking",
                    jk.ijk_name()
                )));
            }
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
            // Prior forks' taken-path conditions are not yet permanent (fork
            // processing applies them after the step), so assert them in a
            // scope that brackets only this check. The scope must close
            // before any later statement runs: a pin or other permanent
            // constraint added inside it would be discarded by the pop while
            // the access it pinned stays done (angr-vmi2l.161). The pop also
            // truncates the scope's `assumed` export-log entries before the
            // `BranchSnapshot` fork below (angr-ype54).
            if !self.deferred_forks.is_empty() {
                self.ctx.push();
                for prev_fork in &self.deferred_forks {
                    if let Some(cond) = self.stored_conditions.get(&prev_fork.condition_id) {
                        if prev_fork.path_taken {
                            self.ctx.assume_true(cond);
                        } else {
                            self.ctx.assume_false(cond);
                        }
                    }
                }
                let feasibility = self.ctx.check_branch_feasibility(&guard_val);
                self.ctx.pop();
                feasibility
            } else {
                self.ctx.check_branch_feasibility(&guard_val)
            }
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

                // Take the "true" path (jump to dst), defer the "false" path
                let cond_id = self.next_cond_id();
                // Store the Rust condition for later retrieval when processing forks
                self.stored_conditions.insert(cond_id, guard_val);

                // Flush pending stores before snapshotting so the memory
                // snapshot includes all writes up to this branch point.
                self.flush_stores_to_rust_memory();

                // Decide which path to take based on branch direction.
                // For backward branches (loops), take the exit (loop back)
                // and defer the fall-through (loop exit). For forward branches,
                // take the fall-through and defer the exit. VEX often inverts
                // forward branch conditions (e.g., `jne target` becomes
                // `if (eq) goto exit; NEXT: target`), so fall-through follows
                // the natural execution flow for forward branches.
                // A trap side is always the deferred one: only the
                // fall-through can continue (angr-vmi2l.160).
                let is_backward_branch = dst < self.current_insn_addr && trap_side.is_none();
                let (unexplored_target, unexplored_jumpkind) = if is_backward_branch {
                    (false_target, crate::state::HistoryEntry::JK_BORING)
                } else if trap_side.is_some() {
                    // The byte encoding has no trap kinds; the export reads the
                    // exact one from the sibling's pending exit.
                    (dst, crate::state::HistoryEntry::JK_OTHER)
                } else {
                    (dst, crate::state::HistoryEntry::jumpkind_from_vex(&jk))
                };
                // The unexplored side's history: this step's entries so far,
                // then the edge out of this block (angr-mx00p.132.44).
                let mut detailed_history = self.detailed_history.clone();
                detailed_history.push(crate::state::HistoryEntry {
                    addr: irsb.addr,
                    jumpkind: unexplored_jumpkind,
                    jump_target: unexplored_target,
                });

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
                        call_stack: self.call_stack.clone(),
                        detailed_history,
                    },
                );

                let deferred = if is_backward_branch {
                    DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: true,  // we took the exit (guard=true) path
                        unexplored_target, // fall-through deferred
                        condition_id: cond_id,
                        trap_side: None,
                    }
                } else {
                    DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: false, // we took the fallthrough (guard=false) path
                        unexplored_target, // the exit target is deferred
                        condition_id: cond_id,
                        trap_side,
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
    ///
    /// A store whose guard is not provably false fires `mem_write` before and
    /// after like a plain `Store`, with the guard as `mem_write_condition`,
    /// as Python's `memory.store(.., condition=guard)` does (angr-vmi2l.337).
    fn handle_storeg(
        &mut self,
        callbacks: &dyn EngineHost,
        guard: &IRExpr,
        addr: &IRExpr,
        data: &IRExpr,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        // Evaluate guard condition
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
        let guard_class = self.classify_guard(&guard_val);
        // Guard is false - skip the store.
        if guard_class == GuardClass::Never {
            return Ok(StmtResult::Continue);
        }
        let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
        let mut data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
        let data_size = data_val.width().div_ceil(8) as usize;
        if let Some(injected) = self.dispatch_mem_write_inspect(
            callbacks,
            &addr_val,
            &data_val,
            data_size,
            endness,
            Some(&guard_val),
            "before",
        ) {
            data_val = injected;
        }

        if guard_class == GuardClass::Always {
            // Guard is true - the store is semantically identical to a plain
            // IRStmt::Store of data_val at addr_val. Route through the shared
            // dispatcher (angr-myzjx.26): this replaces a near-verbatim clone of
            // handle_symbolic_store's concretization ladder that (a) never
            // invalidated the code cache or evicted overlapping symbolic
            // shadows, and (b) stored concrete data at a symbolic address via
            // call_memory_store(0, ..) — a hard-coded address 0.
            self.store_value(callbacks, &addr_val, data_val.clone(), data_size)?;
        } else if let Some(addr_concrete) = addr_val.as_u64() {
            // Both paths possible with symbolic guard - use ITE for conditional store
            // Store ITE(guard, new_data, current_data)
            self.store_guarded_ite(
                callbacks,
                addr_concrete,
                &guard_val,
                &data_val,
                data_size,
                GuardedStoreAddr::Literal,
            )?;
        } else {
            self.store_guarded_symbolic_addr(
                callbacks,
                &addr_val,
                &guard_val,
                &data_val,
                &data_val,
                "guarded store",
            )?;
        }

        self.dispatch_mem_write_inspect(
            callbacks,
            &addr_val,
            &data_val,
            data_size,
            endness,
            Some(&guard_val),
            "after",
        );
        Ok(StmtResult::Continue)
    }

    /// Execute an `IRStmt::LoadG` (guarded load): guard tristate handling
    /// (symbolic always-true/always-false/both via ITE, plus concrete guard),
    /// with cvt-based widening conversions. Extracted verbatim from
    /// `execute_stmt_with_callbacks` (cudgw.18).
    fn handle_loadg(
        &mut self,
        callbacks: &dyn EngineHost,
        args: LoadGArgs<'_>,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        let LoadGArgs {
            dst,
            guard,
            addr,
            alt,
            cvt,
            endness,
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
            guard_class => {
                // The guard can be true: load, and fire `mem_read` with the
                // guard as `mem_read_condition`, as Python's guarded load
                // does (angr-vmi2l.337).
                let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
                let mut loaded = self.load_layered(callbacks, &addr_val, load_size, None)?;
                if let Some(injected) = self.dispatch_mem_read_inspect(
                    callbacks,
                    &addr_val,
                    &loaded,
                    load_size,
                    endness,
                    Some(&guard_val),
                ) {
                    loaded = injected;
                }
                let converted = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());
                if guard_class == GuardClass::Always {
                    converted
                } else {
                    // Both paths possible: if guard then loaded else alt
                    guard_val.ite(&converted, &alt_val)
                }
            }
        };

        self.write_tmp(*dst, result)?;

        Ok(StmtResult::Continue)
    }
}

test_submod!("statements_tests.rs" => statements_tests);
