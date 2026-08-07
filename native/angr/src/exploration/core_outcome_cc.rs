//! Calling-convention snapshot marshalling for the post-step arms.
//!
//! Extracted from `core_outcome.rs` (angr-nbim4.3). Holds [`CcSnapshot`] and
//! its argument/return marshalling helpers; the post-step handler arms in the
//! sibling `handlers` module call these via `ctx.cc`.

use super::*;

/// Scalar / vec snapshot of the manager's calling convention.
///
/// The `Box<dyn CallingConvention>` trait object is not `Clone`, so the post-step
/// native arms snapshot exactly the scalar and vec data they read (mirrors the
/// CC reads at the legacy `stepping.rs` sites). Built once per step by
/// `RustExplorationManager::step_context`.
#[derive(Clone)]
pub(crate) struct CcSnapshot {
    pub(crate) arg_registers: Vec<u32>,
    pub(crate) syscall_arg_registers: Vec<u32>,
    pub(crate) return_register: u32,
    pub(crate) link_register: Option<u32>,
    pub(crate) pops_return_addr: bool,
    pub(crate) pointer_size: u32,
    pub(crate) stack_arg_offset: u64,
    pub(crate) syscall_stack_arg_offset: Option<u64>,
    pub(crate) syscall_error_register: Option<(u32, i64)>,
}

impl CcSnapshot {
    /// Write a syscall's return value into the ABI's return register, splitting
    /// off the errno flag on ABIs that carry one in a second register.
    ///
    /// On a `syscall_error_register` ABI (MIPS: `$a3`), a return at or above
    /// `errno_start` (unsigned compare) means failure: the error register is
    /// set to all-ones and
    /// the return register to the negated value. Everything is expressed as
    /// `ite` over the symbolic return, so a return value the solver has not
    /// pinned stays symbolic in both registers.
    ///
    /// The sole implementation — unlike its `extract_*_args` /
    /// `setup_native_subcall` siblings there is no live-CC twin to share with;
    /// the single-threaded path routes syscall returns through the same post-step
    /// handlers in `core_outcome_handlers.rs`.
    pub(crate) fn write_syscall_return(&self, state: &mut RustSimState, ret_reg: u32, ret: RustBV) {
        let Some((err_reg, errno_start)) = self.syscall_error_register else {
            state.set_register_by_offset(ret_reg, ret);
            return;
        };

        let bits = ret.width();
        let (ret_val, err_val) = {
            let ctx = state.solver().borrow();
            let errno_start_bv = RustBV::concrete(errno_start as u128, bits);
            let error_cond = ret.uge(&errno_start_bv, &ctx);
            let err_val = error_cond.ite(&RustBV::ones(bits), &RustBV::zero(bits), &ctx);
            let ret_val = error_cond.ite(&ret.neg(&ctx), &ret, &ctx);
            (ret_val, err_val)
        };
        state.set_register_by_offset(ret_reg, ret_val);
        state.set_register_by_offset(err_reg, err_val);
    }

    /// Extract procedure arguments from this scalar snapshot.
    ///
    /// Thin adapter over [`extract_args_with_abi`], which holds the extraction
    /// logic and documents the error semantics; the single-threaded
    /// `RustExplorationManager::extract_procedure_args` adapts the same helper
    /// from the live `Box<dyn CallingConvention>`.
    pub(crate) fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        let abi = ArgExtractAbi {
            arg_registers: &self.arg_registers,
            pointer_size: self.pointer_size,
            stack_arg_offset: Some(self.stack_arg_offset),
        };
        extract_args_with_abi(&abi, state, num_args)
    }

    /// Extract syscall arguments from this scalar snapshot.
    ///
    /// Same adapter shape as [`CcSnapshot::extract_procedure_args`], but over
    /// the syscall register window and its (usually absent) stack window — see
    /// [`extract_args_with_abi`] and the twin
    /// `RustExplorationManager::extract_syscall_args`.
    pub(crate) fn extract_syscall_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        let abi = ArgExtractAbi {
            arg_registers: &self.syscall_arg_registers,
            pointer_size: self.pointer_size,
            stack_arg_offset: self.syscall_stack_arg_offset,
        };
        extract_args_with_abi(&abi, state, num_args)
    }

    /// Set up a native sub-call from this scalar snapshot (no `&self` on the
    /// manager, so the parallel post-step path can run it off-thread).
    ///
    /// Thin adapter over [`setup_native_subcall_with_abi`], which holds the
    /// dispatch logic and documents the per-ABI behavior; the single-threaded
    /// `RustExplorationManager::setup_native_subcall` adapts the same helper
    /// from the live `Box<dyn CallingConvention>`.
    pub(crate) fn setup_native_subcall(
        &self,
        state: &mut RustSimState,
        sub: NativeSubcall,
    ) -> Result<(), SubcallSetupError> {
        let abi = SubcallAbi {
            arg_registers: &self.arg_registers,
            pointer_size: self.pointer_size,
            pops_return_addr: self.pops_return_addr,
            link_register: self.link_register,
        };
        setup_native_subcall_with_abi(&abi, state, sub)
    }
}
