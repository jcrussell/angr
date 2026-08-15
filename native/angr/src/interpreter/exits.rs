//! VEX exit / jump-target resolution: where a block goes next.
//!
//! Covers both ends of a block's control flow — `eval_next_addr` and
//! `eval_next_addr_concretized` turn `irsb.next` into concrete target(s)
//! (routing symbolic targets through the shared per-block cache in
//! [`super::concretize_cache`]), while `handle_default_exit` / `handle_exit`
//! turn a resolved target plus its [`JumpKind`] into the [`BlockResult`] that
//! either chains to the next block or hands control back to Python.
//! `get_syscall_num` reads the syscall number for the `Ijk_Sys_*` kinds.

use super::bv_utils::extract_ite_targets;
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Evaluate the next address from an IRSB.
    /// Used for Exit statements where we still need callbacks for complex expressions.
    pub(super) fn eval_next_addr(
        &mut self,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<u64, CbExecutionError> {
        let next_val = self.eval_expr_with_callbacks(callbacks, &irsb.next, &irsb.tyenv)?;
        // Check for symbolic addresses FIRST - Constrained BV has concrete value but is still symbolic
        if next_val.is_symbolic() {
            // Try to concretize to a single value. Routed through the shared
            // per-block cache (`concretize_cache.rs::concretize_cached_jump`)
            // so a block that re-enters this path — every symbolic `Ist_Exit`
            // evaluates the same `irsb.next` fallthrough — pays one solver
            // query, not one per exit.
            match &*self.concretize_cached_jump(&next_val) {
                &ConcretizationResult::Single(addr) => {
                    // Add constraint that target == addr
                    let concrete = RustBV::concrete(addr as u128, next_val.width());
                    let constraint = next_val.eq(&concrete, self.ctx);
                    self.ctx.assume_true(&constraint);
                    return Ok(addr);
                }
                _ => {
                    // For Exit statements mid-block, we can't easily fork
                    // Return error to fall back to Python handling
                    return Err(CbExecutionError::Unsupported(
                        "symbolic next address".to_string(),
                    ));
                }
            }
        }
        next_val
            .as_u64()
            .ok_or_else(|| CbExecutionError::Unsupported("non-concrete next address".to_string()))
    }

    /// Evaluate and concretize the jump target for the default exit.
    ///
    /// This method handles symbolic jump targets (e.g., ret instructions with symbolic
    /// return addresses) by concretizing them to a bounded set of concrete values.
    fn eval_next_addr_concretized(
        &mut self,
        irsb: &IRSB,
    ) -> Result<ConcretizedJump, CbExecutionError> {
        let next_val = self.eval_expr_simple(&irsb.next, &irsb.tyenv)?;

        // Fast path: concrete address
        if let Some(addr) = next_val.as_u64()
            && !next_val.is_symbolic()
        {
            return Ok(ConcretizedJump::Single(addr));
        }

        // NO_IP_CONCRETIZATION (engines/successors.py:292-296) and
        // NO_SYMBOLIC_JUMP_RESOLUTION (engines/successors.py:234-239) both
        // route a symbolic jump target straight to the unconstrained stash
        // without enumeration. The Python checks live at different layers
        // (jump-resolution fires earlier in the elif chain than ip-concret)
        // but for the Rust engine — which reaches this site only after the
        // concrete-IP fast path — they produce the same outcome, so we OR
        // them at one short-circuit.
        if self.no_ip_concretization || self.no_symbolic_jump_resolution {
            return Ok(ConcretizedJump::TooMany {
                min: 0,
                max: 0,
                limit: 0,
            });
        }

        // ITE fast path: extract concrete targets from nested ITE chains
        // without solver queries. Pattern: if(c1, addr1, if(c2, addr2, ...))
        if let Some(targets) = extract_ite_targets(&next_val, self.config.max_symbolic_ip_targets) {
            if targets.len() == 1 {
                // KEEP_IP_SYMBOLIC: stash the original ITE so the manager can
                // restore it to the IP register after `set_pc(targets[0])`.
                if self.keep_ip_symbolic {
                    self.symbolic_ip_at_exit = Some(next_val.clone());
                }
                return Ok(ConcretizedJump::Single(targets[0]));
            }
            return Ok(ConcretizedJump::Multiple {
                targets,
                expr: next_val,
            });
        }

        // Symbolic address - use AddressConcretizer
        match self.concretizer.concretize(&next_val, self.ctx) {
            ConcretizationResult::Single(addr) => {
                // KEEP_IP_SYMBOLIC: do NOT pin `next_val == addr` (mirrors
                // engines/successors.py:327-328, which skips
                // `add_constraints(cond)` when the option is set). Stash the
                // symbolic expression so the manager can write it back to the
                // IP register after `set_pc`.
                if self.keep_ip_symbolic {
                    self.symbolic_ip_at_exit = Some(next_val.clone());
                } else {
                    let concrete = RustBV::concrete(addr as u128, next_val.width());
                    let constraint = next_val.eq(&concrete, self.ctx);
                    self.ctx.assume_true(&constraint);
                }
                Ok(ConcretizedJump::Single(addr))
            }
            ConcretizationResult::Multiple(addrs) => {
                // Check if we exceed max_symbolic_ip_targets
                if addrs.len() > self.config.max_symbolic_ip_targets {
                    let min = *addrs.first().unwrap_or(&0);
                    let max = *addrs.last().unwrap_or(&0);
                    Ok(ConcretizedJump::TooMany {
                        min,
                        max,
                        limit: self.config.max_symbolic_ip_targets,
                    })
                } else {
                    Ok(ConcretizedJump::Multiple {
                        targets: addrs,
                        expr: next_val,
                    })
                }
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                // Convert strided to explicit list, but check limit first
                let num_targets = count as usize;
                if num_targets > self.config.max_symbolic_ip_targets {
                    // The address arithmetic wraps, matching `strided_addrs`
                    // in the sibling arm below.
                    // overflow-ok: `count >= 1` here — this arm is only
                    // reached when `num_targets` exceeds an unsigned limit.
                    let max = base.wrapping_add((count - 1).wrapping_mul(stride));
                    Ok(ConcretizedJump::TooMany {
                        min: base,
                        max,
                        limit: self.config.max_symbolic_ip_targets,
                    })
                } else {
                    let targets = strided_addrs(base, stride, count);
                    Ok(ConcretizedJump::Multiple {
                        targets,
                        expr: next_val,
                    })
                }
            }
            ConcretizationResult::TooLarge { min, max, limit: _ } => Ok(ConcretizedJump::TooMany {
                min,
                max,
                limit: self.config.max_symbolic_ip_targets,
            }),
            ConcretizationResult::Failed(msg) => Err(CbExecutionError::Unsupported(format!(
                "jump target concretization failed: {msg}"
            ))),
        }
    }

    /// Handle the default exit (end of block).
    pub(super) fn handle_default_exit(
        &mut self,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        let concretized = self.eval_next_addr_concretized(irsb)?;
        match concretized {
            ConcretizedJump::Single(addr) => Ok(self.handle_exit(addr, irsb.jumpkind)),
            ConcretizedJump::Multiple { targets, expr } => {
                // Store the expression for constraint addition later
                let condition_id = self.next_condition_id;
                self.next_condition_id += 1;
                self.stored_conditions.insert(condition_id, expr);

                Ok(BlockResult::SymbolicJumpTarget {
                    targets,
                    condition_id,
                    jumpkind: irsb.jumpkind,
                })
            }
            ConcretizedJump::TooMany { min, max, limit } => Ok(BlockResult::UnconstrainedJump {
                min_target: min,
                max_target: max,
                limit,
                jumpkind: irsb.jumpkind,
            }),
        }
    }

    /// Handle an exit (update PC, return result).
    pub(super) fn handle_exit(&mut self, target: u64, jumpkind: JumpKind) -> BlockResult {
        self.set_pc(target);
        // Trace removed after debugging

        if jumpkind.is_syscall() {
            // angr-gffd: keep `num: Option<u64>` honest about symbolicness.
            // A symbolic syscall register MUST NOT silently dispatch to a
            // native handler — on amd64 the previous `.unwrap_or(0)` routed
            // every symbolic syscall to NativeReadSyscall. Caller forces the
            // Python fallback when `num` is None, where Python's
            // engines/successors.py::_resolve_syscall handles both
            // enumeration and NO_SYMBOLIC_SYSCALL_RESOLUTION.
            let syscall_num = self.get_syscall_num();
            return BlockResult::Syscall { num: syscall_num };
        }

        if self.is_hooked(target) {
            return BlockResult::Hook { addr: target };
        }

        // For CALL instructions to external code, ask Python to resolve
        if jumpkind.is_call() && !self.is_in_binary(target) {
            let return_addr =
                self.get_return_addr_or_log("BlockResult::UnmodeledCall (call to external code)");
            return BlockResult::UnmodeledCall {
                addr: target,
                return_addr,
                symbol_name: None, // Symbol lookup done by Python
            };
        }

        // Ijk_Ret with an empty call stack means we are returning from a
        // function we never entered (typical for blank_state at a `ret`
        // instruction). The popped IP was lazy-filled by Python's symbolic
        // stack but materialized to a concrete value during Python->Rust
        // sync (rust_state_sync.py:_sync_stack_page). Match Python's
        // _eval_target_brutal behaviour (engines/successors.py:308-323):
        // when the IP has no meaningful constraint pinning it, route to
        // the unconstrained stash instead of treating it as a call to
        // address 0. See angr-3uye.
        if jumpkind.is_ret() && self.call_stack.is_empty() && !self.is_in_binary(target) {
            return BlockResult::UnconstrainedJump {
                min_target: target,
                max_target: target,
                limit: self.config.max_symbolic_ip_targets,
                jumpkind,
            };
        }

        // For jumps/returns to external addresses that are NOT hooked,
        // treat as UnmodeledCall so Python can handle them properly.
        // This includes:
        // - angr's internal continuation addresses (0x700000+)
        // - extern stubs and SimProcedure return points
        // - dynamically registered hooks that weren't synced yet
        if !self.is_in_binary(target) {
            let return_addr = self.get_return_addr_or_log(
                "BlockResult::UnmodeledCall (unhooked jump/return to external addr)",
            );
            return BlockResult::UnmodeledCall {
                addr: target,
                return_addr,
                symbol_name: Some("__extern_addr__".to_string()),
            };
        }

        BlockResult::BlockEnd {
            next_addr: target,
            jumpkind,
        }
    }

    /// Get the syscall number from the appropriate register.
    ///
    /// Returns `Some(n)` when the syscall register holds a concrete value,
    /// `None` when the register is symbolic (caller must route to Python so
    /// `engines/successors.py::_resolve_syscall` can enumerate or honor
    /// `NO_SYMBOLIC_SYSCALL_RESOLUTION`). Also returns `None` when the
    /// architecture has no syscall-num register, matching the prior
    /// `unwrap_or(0)` fallback in spirit but without silently dispatching
    /// to syscall 0.
    pub(super) fn get_syscall_num(&self) -> Option<u64> {
        let offset = self.registers.arch().syscall_num_offset()?;
        self.registers.get_offset_u64(offset, self.ctx)
    }
}

test_submod!("exits_tests.rs" => exits_tests);
