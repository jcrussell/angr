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
    /// Mirror of `RustExplorationManager::write_syscall_return`.
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

    /// Mirror of `RustExplorationManager::extract_procedure_args`.
    pub(crate) fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        use crate::arch::ExtractionError;
        let ptr_size = self.pointer_size;
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        for &offset in self.arg_registers.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + self.stack_arg_offset;
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        drop(ctx);
        Ok(args)
    }

    /// Mirror of `RustExplorationManager::extract_syscall_args`.
    pub(crate) fn extract_syscall_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        use crate::arch::ExtractionError;
        let ptr_size = self.pointer_size;
        let mut args = Vec::with_capacity(num_args);
        for &offset in self.syscall_arg_registers.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let stack_offset =
                self.syscall_stack_arg_offset
                    .ok_or(ExtractionError::RegisterOverflow {
                        requested: num_args,
                        available: self.syscall_arg_registers.len(),
                    })?;
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + stack_offset;
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        Ok(args)
    }

    /// Mirror of `RustExplorationManager::setup_native_subcall` (no `&self`).
    pub(crate) fn setup_native_subcall(
        &self,
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
        let arg_regs = &self.arg_registers;
        if sub_args.len() > arg_regs.len() {
            return Err(SubcallSetupError::TooManyArgs {
                requested: sub_args.len(),
                available: arg_regs.len(),
            });
        }
        let ptr_bits = self.pointer_size * 8;
        let sentinel = native_resume_sentinel(self.pointer_size);

        // --- feasibility checks (no mutation yet) ---
        let lr_offset = if self.pops_return_addr {
            None
        } else {
            Some(
                self.link_register
                    .ok_or(SubcallSetupError::UnsupportedAbi)?,
            )
        };
        let sp_val = if self.pops_return_addr {
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
            state
                .memory_mut()
                .store_concrete(sp, RustBV::concrete(sentinel as u128, ptr_bits))
                .map_err(SubcallSetupError::Memory)?;
        } else if let Some(lr) = lr_offset {
            state.set_register_by_offset(lr, RustBV::concrete(sentinel as u128, ptr_bits));
        }

        // --- record the continuation and enter the guest routine ---
        state.push_native_resume_frame(NativeResumeFrame {
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
}
