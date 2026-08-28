//! The Python-bouncing arm of the single-threaded step driver.
//!
//! Extracted from `stepping.rs` (angr-5mnx3.71).
//! [`RustExplorationManager::dispatch_bounce`] turns a [`PendingBounce`] — the
//! `NeedsPython` outcome the core hands back — into the exact
//! [`PendingCallback`] the caller must service (SimProcedure, hook, syscall,
//! symbolic branch, error). `handle_unmodeled_call` /
//! `unmodeled_call_generic_skip` cover the unresolved-call arm, which is the
//! only one that runs a `&mut self` resolve step of its own. The matching
//! re-entry points live in `resume.rs`.

use super::core_outcome::{BounceKind, PendingBounce};
use super::*;

#[allow(clippy::result_large_err)]
impl RustExplorationManager {
    /// Run the legacy Python-bouncing arm for a `NeedsPython` core outcome. The
    /// core already did any native dispatch and recorded its counters; these
    /// tails build the exact `PendingCallback` (and, for `UnmodeledCall`, run the
    /// `&mut self` resolve path) the single-threaded engine produced inline.
    pub(crate) fn dispatch_bounce(
        &mut self,
        callbacks: &PythonCallbacks,
        bounce: PendingBounce,
    ) -> Result<Vec<RustSimState>, StepError> {
        let PendingBounce {
            kind,
            mut state,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        } = bounce;

        match kind {
            BounceKind::Hook { addr } => {
                state.set_pc(addr);
                // History BEFORE callback so Python can read recent_bbl_addrs[-1].
                state.add_to_history(addr);
                let hook_fork_start = if self.profiling.profiling_enabled {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let pre_callback_snapshot = if !deferred_forks.is_empty() {
                    Some(state.fork())
                } else {
                    None
                };
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                if let Some(start) = hook_fork_start {
                    let fork_count = u64::from(pre_callback_snapshot.is_some());
                    self.profiling.accumulated_stats.solver_fork_time_ns +=
                        crate::elapsed_ns(start);
                    self.profiling.accumulated_stats.solver_fork_count += fork_count;
                }
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    pre_callback_snapshot,
                    CallbackReason::SimProcedure {
                        addr,
                        name: "unknown".to_string(),
                        num_args: 0,
                        return_addr: 0,
                    },
                    JumpKind::Boring.ijk_name(),
                    Some(shared_ctx),
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
                )))
            }
            BounceKind::SimProcedurePython {
                addr,
                name,
                num_args,
                return_addr,
            } => {
                state.set_pc(addr);
                state.add_to_history(addr);
                let (pre_callback_snapshot, shared_ctx) =
                    super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    pre_callback_snapshot,
                    CallbackReason::SimProcedure {
                        addr,
                        name,
                        num_args,
                        return_addr,
                    },
                    "Ijk_Call",
                    Some(shared_ctx),
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
                )))
            }
            BounceKind::SyscallPython { num } => {
                let (pre_callback_snapshot, shared_ctx) =
                    super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    pre_callback_snapshot,
                    CallbackReason::Syscall { num },
                    "Ijk_Sys_syscall",
                    Some(shared_ctx),
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
                )))
            }
            BounceKind::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            } => self.handle_unmodeled_call(
                callbacks,
                state,
                addr,
                return_addr,
                symbol_name,
                ForkBundle {
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                },
            ),
            BounceKind::PythonVEXFallback { addr, reason } => {
                // state.set_pc(addr) was applied by the core before bouncing.
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::PythonVEXFallback { addr, reason },
                    JumpKind::Boring.ijk_name(),
                    None,
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
                )))
            }
        }
    }

    /// Handle UnmodeledCall: try to resolve via Python callback. Resolved
    /// functions are registered as SimProcedures and dispatched via callback;
    /// unresolved calls use P21 generic skip (set return register to 0,
    /// continue at return address) instead of deadending.
    fn handle_unmodeled_call(
        &mut self,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        addr: u64,
        return_addr: u64,
        symbol_name: Option<String>,
        forks: ForkBundle,
    ) -> Result<Vec<RustSimState>, StepError> {
        let ForkBundle {
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        } = forks;

        // Unhooked CALL target - try to resolve via Python callback
        state.set_pc(addr);
        // Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
        state.add_to_history(addr);

        // Try to resolve the function via callback
        if callbacks.has_resolve_function() {
            match callbacks.call_resolve_function(addr, symbol_name.as_deref()) {
                Ok(Some((name, num_args, no_return))) => {
                    // Function resolved! Register it and return to Python for execution
                    log::debug!(
                        "Resolved unmodeled call at 0x{addr:x} -> {name} (args={num_args}, no_return={no_return})"
                    );

                    // Register the procedure so future calls are hooked
                    self.hooks.insert(addr);
                    self.simprocedures
                        .insert(addr, (name.clone(), num_args, no_return));

                    let (pre_callback_snapshot, shared_ctx) =
                        super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);

                    // Return to Python for SimProcedure execution
                    Err(StepError::NeedCallback(PendingCallback::with_context(
                        state,
                        pre_callback_snapshot,
                        CallbackReason::SimProcedure {
                            addr,
                            name,
                            num_args,
                            return_addr,
                        },
                        "Ijk_Call",
                        Some(shared_ctx),
                        ForkBundle {
                            deferred_forks,
                            stored_conditions,
                            fork_snapshots,
                        },
                    )))
                }
                Ok(None) => {
                    // P21: Function could not be resolved - use generic skip instead of deadending
                    self.unmodeled_call_generic_skip(
                        state,
                        addr,
                        return_addr,
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    )
                }
                Err(e) => {
                    // Callback error - treat as execution error
                    log::warn!("resolve_function callback error at 0x{addr:x}: {e}");
                    Err(StepError::Error(
                        state,
                        format!("resolve_function error: {e}"),
                    ))
                }
            }
        } else {
            // P21: No resolve_function callback - use generic skip instead of deadending
            self.unmodeled_call_generic_skip(
                state,
                addr,
                return_addr,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            )
        }
    }

    /// P21 generic skip for unmodeled calls: set return register to 0,
    /// continue at return_addr, and process any deferred forks. Used both
    /// when resolve_function returns None and when no callback is registered.
    fn unmodeled_call_generic_skip(
        &mut self,
        mut state: RustSimState,
        addr: u64,
        return_addr: u64,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        log::debug!(
            "P21: Unmodeled call at 0x{addr:x} - generic skip (ret=0) to return_addr=0x{return_addr:x}"
        );

        // Set return register to 0 (symbolic unconstrained would be better but
        // concrete 0 is simpler and often sufficient)
        let ret_reg_offset = self.environment.calling_convention.return_register();
        let ptr_size = self.environment.calling_convention.pointer_size();
        let zero_val = RustBV::zero(ptr_size * 8);
        state.set_register_by_offset(ret_reg_offset, zero_val);

        // Continue at return address
        state.set_pc(return_addr);

        // Process any deferred forks from the interpreter step
        let mut successors = vec![state];
        self.process_deferred_forks_into(
            &mut successors,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
        );
        Ok(successors)
    }
}
