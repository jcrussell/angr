//! amd64 brk syscall handler.
//!
//! Mirrors `procedures/linux_kernel/brk.py` (which calls
//! `state.posix.set_brk`):
//!   1. Symbolic new_brk → fall back to Python (the warning + If(...) path
//!      in Python; rare in practice).
//!   2. Concrete `new_brk < current_brk` → no-op, return current brk.
//!      This includes `brk(0)` → return current brk (Linux convention),
//!      since the default brk (0x1B00000) is always > 0.
//!   3. Otherwise: set posix_brk = new_brk; if it grew across a page
//!      boundary, map the new pages with RWX permissions; return
//!      new_brk.
//!
//! Collision handling: Python's `set_brk` catches a `SimMemoryError`
//! from `map_region` and uses `e.args[1]` as the alternate brk. Rust's
//! `SymbolicMemory::map` is idempotent (silently leaves already-mapped
//! pages alone) and has no equivalent collision signal, so when we
//! detect that any page in the to-be-mapped range is already mapped
//! we fall back to the Python path to preserve semantics.

use super::page::{PAGE_MASK, PAGE_SIZE};
use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub struct NativeBrkSyscall;

impl NativeSyscall for NativeBrkSyscall {
    fn name(&self) -> &'static str {
        "brk"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.is_empty() {
            return Err(SyscallError::Other("brk expected 1 arg, got 0".into()));
        }
        let new_brk = extract_concrete_arg(&args[0], "brk new_brk")?;

        let current = state.posix_brk();

        // brk(addr) where addr < current is a no-op that returns the
        // current break. brk(0) is the canonical "query" form and is
        // covered here since posix_brk default 0x1B00000 > 0.
        if new_brk < current {
            return Ok(SyscallOutcome::Continue { ret: current });
        }

        // Compute the page range that needs mapping. Python checks
        // `((conc_start - 1) ^ (conc_end - 1)) & ~0xFFF`, i.e. did the
        // last-byte-before-the-break cross a page boundary. Equivalent:
        // align both to page floor and see if they differ.
        let need_map = if current == new_brk {
            false
        } else {
            // last byte before each break (subtract 1) — matches Python's
            // logic that "the break is the address of the first unmapped
            // byte." current==0 is impossible (default is 0x1B00000), but
            // we guard with checked_sub for safety.
            let cur_floor = current.saturating_sub(1) & !PAGE_MASK;
            let new_floor = new_brk.saturating_sub(1) & !PAGE_MASK;
            cur_floor != new_floor
        };

        if need_map {
            // Align up: pages [aligned_start, aligned_end).
            let aligned_start = (current + PAGE_MASK) & !PAGE_MASK;
            let aligned_end = (new_brk + PAGE_MASK) & !PAGE_MASK;

            // Collision check: if any page in the new range is already
            // mapped, fall back to Python so its SimMemoryError-driven
            // fixup logic runs.
            let memory = state.memory();
            let mut page_addr = aligned_start;
            while page_addr < aligned_end {
                let page_num = page_addr >> 12;
                if memory.page_permissions(page_num).is_some() {
                    return Err(SyscallError::Other(format!(
                        "brk: page {page_addr:#x} already mapped (collision)"
                    )));
                }
                page_addr += PAGE_SIZE;
            }

            // Map the new pages with RWX (matches Python `map_region(..., 7)`).
            let memory = state.memory_mut();
            memory.map(aligned_start, aligned_end - aligned_start, Permission::RWX);
        }

        state.set_posix_brk(new_brk);
        Ok(SyscallOutcome::Continue { ret: new_brk })
    }
}

#[cfg(test)]
#[path = "brk_tests.rs"]
mod brk_tests;
