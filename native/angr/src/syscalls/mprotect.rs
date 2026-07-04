//! amd64 mprotect syscall handler.
//!
//! Mirrors `procedures/linux_kernel/mprotect.py`:
//!   1. All three args (addr, length, prot) must be concrete; symbolic
//!      args fall back to the Python path (which itself raises
//!      `SimValueError`, matching prior behavior).
//!   2. If `addr & 0xFFF != 0`, return -1 (alignment).
//!   3. If any page in `[addr, page_end)` is unmapped, return -1.
//!   4. Otherwise set permissions on every page to `prot & 7` and
//!      return 0.
//!
//! Permission encoding gotcha: Linux mprotect uses `PROT_READ=0x1`,
//! `PROT_WRITE=0x2`, `PROT_EXEC=0x4`, but `Permission::from_bits` uses
//! `read=0x4, write=0x2, execute=0x1` (reversed). We translate the
//! Linux bits explicitly here rather than going through `from_bits`.

use super::page::{PAGE_MASK, PAGE_SIZE};
use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub struct NativeMprotectSyscall;

fn linux_prot_to_permission(prot: u64) -> Permission {
    Permission {
        read: prot & 0x1 != 0,
        write: prot & 0x2 != 0,
        execute: prot & 0x4 != 0,
    }
}

impl NativeSyscall for NativeMprotectSyscall {
    fn name(&self) -> &'static str {
        "mprotect"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 3 {
            return Err(SyscallError::Other(format!(
                "mprotect expected 3 args, got {}",
                args.len()
            )));
        }
        let addr = extract_concrete_arg(&args[0], "mprotect addr")?;
        let length = extract_concrete_arg(&args[1], "mprotect length")?;
        let prot = extract_concrete_arg(&args[2], "mprotect prot")?;

        // Linux: misaligned addr → EINVAL. Python returns -1 here too.
        if addr & PAGE_MASK != 0 {
            return Ok(SyscallOutcome::Continue { ret: u64::MAX });
        }

        // Length 0 is a no-op in Linux mprotect (returns 0 without
        // touching any pages). Mirror that.
        if length == 0 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        // page_end matches Python: ((addr + length - 1) & ~0xFFF) + 0x1000
        // i.e. one past the last page touched. Use checked arithmetic to
        // avoid overflowing u64 on absurd inputs.
        let last_byte = match addr.checked_add(length).and_then(|v| v.checked_sub(1)) {
            Some(v) => v,
            None => return Ok(SyscallOutcome::Continue { ret: u64::MAX }),
        };
        let page_end = (last_byte & !PAGE_MASK).saturating_add(PAGE_SIZE);

        let memory = state.memory();
        let mut page_addr = addr;
        while page_addr < page_end {
            let page_num = page_addr >> 12;
            if memory.page_permissions(page_num).is_none() {
                return Ok(SyscallOutcome::Continue { ret: u64::MAX });
            }
            page_addr += PAGE_SIZE;
        }

        let new_perm = linux_prot_to_permission(prot & 7);
        let memory = state.memory_mut();
        let mut page_addr = addr;
        while page_addr < page_end {
            let page_num = page_addr >> 12;
            // Pages confirmed mapped above; assert via debug_assert and
            // tolerate concurrent unmaps (unlikely but cheap to handle).
            let updated = memory.set_page_permissions(page_num, new_perm);
            debug_assert!(updated, "mprotect: page {page_addr:#x} disappeared");
            page_addr += PAGE_SIZE;
        }

        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
#[path = "mprotect_tests.rs"]
mod mprotect_tests;
