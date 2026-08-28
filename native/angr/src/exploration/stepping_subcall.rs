//! Native sub-call setup: the `ProcOutcome::CallAndResume` path.
//!
//! Extracted from `stepping.rs` (angr-5mnx3.71). Holds the ABI-agnostic
//! [`setup_native_subcall_with_abi`] plus its [`SubcallAbi`] borrow and
//! [`SubcallSetupError`] failure enum, the manager-side thin adapter
//! [`RustExplorationManager::setup_native_subcall`], and the `#[cfg(test)]`
//! `handle_native_resume` twin. The parallel path reaches the same helper
//! through `CcSnapshot::setup_native_subcall` in `core_outcome_cc.rs`.
//!
//! `stepping.rs` re-exports the three free items, so the cross-module import
//! path `super::stepping::{SubcallAbi, ...}` in `core_outcome.rs` is unchanged.

use super::core_outcome::NativeSubcall;
use super::*;

/// Why a native sub-call could not be set up; the dispatcher falls back to the
/// Python SimProcedure path on any of these. Fields are carried for the
/// `Debug` diagnostic in the fallback log line (dead-code analysis ignores
/// `Debug`-only reads, hence the allow).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum SubcallSetupError {
    /// More guest arguments than the ABI exposes in registers. Stack-spilled
    /// guest args are not yet supported (S2 scope; defers to Python).
    TooManyArgs { requested: usize, available: usize },
    /// Stack pointer is symbolic — cannot place the sentinel return slot.
    SpSymbolic,
    /// Writing the sentinel return slot to the stack failed (unmapped / perms).
    Memory(crate::memory::MemoryError),
    /// Link-register ABI with no `link_register()` wired up — cannot redirect
    /// the guest routine's return to the sentinel.
    UnsupportedAbi,
}

/// The ABI facts [`setup_native_subcall_with_abi`] needs, decoupled from where
/// they came from.
///
/// The single-threaded path reads them off the manager's
/// `Box<dyn CallingConvention>`; the parallel post-step path reads them off the
/// scalar `CcSnapshot` (the trait object is not `Clone`). Both funnel through
/// this borrow so the dispatch logic exists once (angr-sqfj8.41).
pub(crate) struct SubcallAbi<'a> {
    pub(crate) arg_registers: &'a [u32],
    pub(crate) pointer_size: u32,
    pub(crate) pops_return_addr: bool,
    pub(crate) link_register: Option<u32>,
}

/// Set up a native sub-call (`ProcOutcome::CallAndResume`): make the guest
/// routine `sub.target` run with `sub.sub_args`, then return to the resume
/// sentinel so the proc's continuation re-enters via `handle_native_resume`.
///
/// All feasibility checks happen before any state mutation, so on `Err` the
/// caller can cleanly fall back to the Python SimProcedure path. `S2`,
/// bead angr-5gf0s. See `tools/decisions/native_subcall_dispatcher_design.md`.
///
/// Stack-return ABI (x86/amd64): at proc entry `[sp]` holds the caller's
/// return address; we overwrite it with the sentinel so the guest `ret`
/// lands on the sentinel (SP unchanged here — the guest's own `ret` advances
/// it). The original caller address rides in the frame's
/// `caller_return_addr`, not the stack. Link-register ABI: write the
/// sentinel into the link register (requires `SubcallAbi::link_register`).
pub(crate) fn setup_native_subcall_with_abi(
    abi: &SubcallAbi<'_>,
    state: &mut RustSimState,
    sub: NativeSubcall,
) -> Result<(), SubcallSetupError> {
    let NativeSubcall {
        proc_name,
        saved_args,
        caller_return_addr,
        target,
        sub_args,
        resume_tag,
    } = sub;
    let arg_regs = abi.arg_registers;
    if sub_args.len() > arg_regs.len() {
        return Err(SubcallSetupError::TooManyArgs {
            requested: sub_args.len(),
            available: arg_regs.len(),
        });
    }
    let ptr_bits = abi.pointer_size * 8;
    let sentinel = crate::procedures::native_resume_sentinel(abi.pointer_size);

    // --- feasibility checks (no mutation yet) ---
    let lr_offset = if abi.pops_return_addr {
        None
    } else {
        Some(abi.link_register.ok_or(SubcallSetupError::UnsupportedAbi)?)
    };
    let sp_val = if abi.pops_return_addr {
        Some(
            state
                .get_sp()
                .as_u64()
                .ok_or(SubcallSetupError::SpSymbolic)?,
        )
    } else {
        None
    };

    // --- mutation: redirect the guest routine's return to the sentinel ---
    if let Some(sp) = sp_val {
        // Overwrite the caller return slot at [sp] with the sentinel. This
        // is the only fallible mutation; do it first so an unmapped stack
        // leaves the state untouched for the Python fallback.
        state
            .memory_mut()
            .store_concrete(sp, RustBV::concrete(sentinel as u128, ptr_bits))
            .map_err(SubcallSetupError::Memory)?;
    } else if let Some(lr) = lr_offset {
        state.set_register_by_offset(lr, RustBV::concrete(sentinel as u128, ptr_bits));
    }

    // --- record the continuation and enter the guest routine ---
    state.push_native_resume_frame(crate::state::NativeResumeFrame {
        proc_name,
        resume_tag,
        saved_args,
        caller_return_addr,
    });
    for (reg, val) in arg_regs.iter().zip(sub_args.into_iter()) {
        state.set_register_by_offset(*reg, val);
    }
    state.set_pc(target);
    Ok(())
}

#[allow(clippy::result_large_err)]
impl RustExplorationManager {
    /// Set up a native sub-call (`ProcOutcome::CallAndResume`) from the
    /// manager's live calling convention.
    ///
    /// Thin adapter over [`setup_native_subcall_with_abi`], which holds the
    /// dispatch logic and documents the per-ABI behavior; the parallel path's
    /// `CcSnapshot::setup_native_subcall` adapts the same helper from its
    /// scalar snapshot.
    pub(crate) fn setup_native_subcall(
        &self,
        state: &mut RustSimState,
        sub: NativeSubcall,
    ) -> Result<(), SubcallSetupError> {
        let cc = &self.environment.calling_convention;
        let abi = SubcallAbi {
            arg_registers: cc.arg_registers(),
            pointer_size: cc.pointer_size(),
            pops_return_addr: cc.pops_return_addr(),
            link_register: cc.link_register(),
        };
        setup_native_subcall_with_abi(&abi, state, sub)
    }

    /// Re-enter a native proc's continuation after a `CallAndResume` sub-call
    /// returns to the resume sentinel. Pops the top resume frame, calls the
    /// proc's [`crate::procedures::NativeSimProcedure::resume`], and applies the
    /// resulting [`ProcOutcome`]. `S2`, bead angr-5gf0s.
    ///
    /// Retained as a focused direct-call test harness (`subcall_tests.rs`); the
    /// production single-threaded path now resumes via
    /// `core_outcome::handle_native_resume_core` (angr-vh834). Gated to test
    /// builds — its only caller is the `#[cfg(test)]` `subcall_tests` module —
    /// so it carries no `dead_code` allow (angr-0mqkc.2). The production twin
    /// has its own direct coverage in `core_outcome_tests/`
    /// (`native_resume_core_*`, angr-c7xno.32), so the two can no longer drift
    /// unobserved.
    #[cfg(test)]
    fn handle_native_resume(
        &mut self,
        mut state: RustSimState,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        let frame = match state.pop_native_resume_frame() {
            Some(f) => f,
            None => {
                // Sentinel reached with no pending frame: a corrupt state we
                // cannot resume. Deadend defensively rather than guess a PC.
                log::error!("native resume sentinel hit with empty resume stack; deadending");
                return Err(StepError::Deadended(state));
            }
        };

        // Run the continuation. The `self.native_procedures` borrow is released
        // once `outcome` is bound, freeing `&mut self` for the sub-call setup.
        let outcome = match self.native_procedures.get(&frame.proc_name) {
            Some(proc) => proc.resume(&mut state, frame.resume_tag, &frame.saved_args),
            None => {
                log::error!(
                    "native resume: proc {} not in registry; deadending",
                    frame.proc_name
                );
                return Err(StepError::Deadended(state));
            }
        };

        match outcome {
            Ok(ProcOutcome::Return(ret_val)) => {
                if let Some(rv) = ret_val {
                    let ret_reg = self.environment.calling_convention.return_register();
                    state.set_register_by_offset(ret_reg, rv);
                }
                // Resume the original caller. The guest routine's `ret` already
                // consumed the sentinel return slot (stack-return ABI advances
                // SP), so unlike the fresh-entry return path we do NOT adjust SP.
                state.set_pc(frame.caller_return_addr);
            }
            Ok(ProcOutcome::CallAndResume {
                target,
                args: sub_args,
                resume_tag,
            }) => {
                // Nested sub-call: the original caller and saved args carry
                // forward so the final return still lands at `caller_return_addr`.
                if let Err(e) = self.setup_native_subcall(
                    &mut state,
                    NativeSubcall {
                        proc_name: frame.proc_name.clone(),
                        saved_args: frame.saved_args.clone(),
                        caller_return_addr: frame.caller_return_addr,
                        target,
                        sub_args,
                        resume_tag,
                    },
                ) {
                    log::error!("native resume nested sub-call setup failed ({e:?}); deadending");
                    return Err(StepError::Deadended(state));
                }
            }
            Err(e) => {
                // resume() should never fail when reached via the sentinel (the
                // proc opted into sub-calls). Deadend defensively.
                log::error!(
                    "native resume: {} resume() failed: {:?}; deadending",
                    frame.proc_name,
                    e
                );
                return Err(StepError::Deadended(state));
            }
        }

        // Deferred-fork handling identical to the native return path.
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

test_submod!("subcall_tests.rs" => subcall_tests);
