use super::helpers::extract_ite_targets;
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Evaluate the next address from an IRSB.
    /// Used for Exit statements where we still need callbacks for complex expressions.
    pub(super) fn eval_next_addr(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<u64, CbExecutionError> {
        let next_val = self.eval_expr_with_callbacks(py, callbacks, &irsb.next, &irsb.tyenv)?;
        // Check for symbolic addresses FIRST - Constrained BV has concrete value but is still symbolic
        if next_val.is_symbolic() {
            // Try to concretize to a single value
            match self.concretizer.concretize(&next_val, self.ctx) {
                ConcretizationResult::Single(addr) => {
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
                    let max = base + (count - 1) * stride;
                    Ok(ConcretizedJump::TooMany {
                        min: base,
                        max,
                        limit: self.config.max_symbolic_ip_targets,
                    })
                } else {
                    let targets: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
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
                "jump target concretization failed: {}",
                msg
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
                self.stored_conditions.insert(condition_id, expr.clone());

                Ok(BlockResult::SymbolicJumpTarget {
                    targets,
                    condition_id,
                    target_expr: expr,
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

    /// Simple expression evaluation (no callbacks, for already-evaluated temps).

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
            let return_addr = self.get_return_addr().unwrap_or(0);
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
            let return_addr = self.get_return_addr().unwrap_or(0);
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
        let arch = self.registers.arch();
        let offset = arch.syscall_num_offset()?;
        let size = arch.bytes();
        self.registers.get(offset, size, self.ctx).as_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CallStackEntry;

    fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
        VEXInterpreter::new(VexArch::AMD64, ctx)
    }

    fn add_internal_region(interp: &mut VEXInterpreter<'_>) {
        // Define a "binary" region so is_in_binary returns true for 0x1000..0x2000.
        interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
    }

    #[test]
    fn handle_exit_updates_pc_on_boring_internal_jump() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        let result = interp.handle_exit(0x1500, JumpKind::Boring);
        assert_eq!(interp.get_pc(), 0x1500);
        matches!(
            result,
            BlockResult::BlockEnd {
                next_addr: 0x1500,
                jumpkind: JumpKind::Boring
            }
        );
    }

    #[test]
    fn handle_exit_syscall_returns_syscall_variant() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        // Place syscall number into RAX via register file's put_reg helper.
        interp.registers.put_reg("rax", RustBV::concrete(60, 64));
        let result = interp.handle_exit(0x1234, JumpKind::Sys_syscall);
        match result {
            BlockResult::Syscall { num } => assert_eq!(num, Some(60)),
            other => panic!("expected Syscall, got {:?}", other),
        }
        // PC is updated even for syscalls.
        assert_eq!(interp.get_pc(), 0x1234);
    }

    #[test]
    fn handle_exit_hooked_address_returns_hook() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        interp.add_hook(0x1500);
        match interp.handle_exit(0x1500, JumpKind::Boring) {
            BlockResult::Hook { addr } => assert_eq!(addr, 0x1500),
            other => panic!("expected Hook, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_call_to_external_returns_unmodeled_call() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        // 0x9000 is not in [0x1000, 0x2000) — external.
        match interp.handle_exit(0x9000, JumpKind::Call) {
            BlockResult::UnmodeledCall {
                addr, symbol_name, ..
            } => {
                assert_eq!(addr, 0x9000);
                // Calls don't get __extern_addr__ tag — that's the post-call path.
                assert_eq!(symbol_name, None);
            }
            other => panic!("expected UnmodeledCall, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_ret_with_empty_call_stack_returns_unconstrained() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        assert!(interp.call_stack.is_empty());
        // External target + Ret + empty stack -> UnconstrainedJump (see angr-3uye).
        match interp.handle_exit(0x9000, JumpKind::Ret) {
            BlockResult::UnconstrainedJump {
                min_target,
                max_target,
                jumpkind,
                ..
            } => {
                assert_eq!(min_target, 0x9000);
                assert_eq!(max_target, 0x9000);
                assert!(jumpkind.is_ret());
            }
            other => panic!("expected UnconstrainedJump, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_ret_to_internal_address_is_block_end() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        // Returning to an internal address is a normal block end even with empty stack.
        match interp.handle_exit(0x1800, JumpKind::Ret) {
            BlockResult::BlockEnd {
                next_addr,
                jumpkind,
            } => {
                assert_eq!(next_addr, 0x1800);
                assert!(jumpkind.is_ret());
            }
            other => panic!("expected BlockEnd, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_external_non_call_non_ret_returns_unmodeled_call() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        // A jump (Boring) to a non-hooked external address is tagged __extern_addr__.
        match interp.handle_exit(0x9000, JumpKind::Boring) {
            BlockResult::UnmodeledCall {
                addr, symbol_name, ..
            } => {
                assert_eq!(addr, 0x9000);
                assert_eq!(symbol_name.as_deref(), Some("__extern_addr__"));
            }
            other => panic!("expected UnmodeledCall, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_internal_block_end_carries_jumpkind() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        // Internal call (target in binary) — falls through to BlockEnd.
        match interp.handle_exit(0x1234, JumpKind::Call) {
            BlockResult::BlockEnd {
                next_addr,
                jumpkind,
            } => {
                assert_eq!(next_addr, 0x1234);
                assert!(jumpkind.is_call());
            }
            other => panic!("expected BlockEnd, got {:?}", other),
        }
    }

    #[test]
    fn handle_exit_ret_with_nonempty_call_stack_to_external_is_unmodeled() {
        // When a Ret targets external memory but we DO have a call frame, the
        // empty-stack guard should not fire — the fall-through path (external,
        // non-hooked) emits UnmodeledCall with the __extern_addr__ tag.
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        add_internal_region(&mut interp);
        interp.call_stack.push(CallStackEntry {
            call_site_addr: 0x1100,
            callee_addr: 0x1200,
            return_addr: 0x1108,
            stack_ptr: 0x7ffe_0000,
        });
        match interp.handle_exit(0x9000, JumpKind::Ret) {
            BlockResult::UnmodeledCall {
                addr, symbol_name, ..
            } => {
                assert_eq!(addr, 0x9000);
                assert_eq!(symbol_name.as_deref(), Some("__extern_addr__"));
            }
            other => panic!("expected UnmodeledCall, got {:?}", other),
        }
    }

    #[test]
    fn get_syscall_num_reads_rax_on_amd64() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rax", RustBV::concrete(231, 64)); // exit_group
        assert_eq!(interp.get_syscall_num(), Some(231));
    }

    #[test]
    fn get_syscall_num_returns_none_for_symbolic_rax() {
        // angr-gffd: symbolic syscall register must produce None so the
        // caller routes to Python instead of dispatching to a native handler.
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let sym = RustBV::symbolic(&ctx, "rax_sym", 64);
        interp.registers.put_reg("rax", sym);
        assert_eq!(interp.get_syscall_num(), None);
    }
}
