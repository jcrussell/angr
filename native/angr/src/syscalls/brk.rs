//! amd64 brk syscall handler.
//!
//! Mirrors `procedures/linux_kernel/brk.py` (which calls
//! `state.posix.set_brk`):
//!   1. Symbolic new_brk → fall back to Python (the warning + If(...) path
//!      in Python; rare in practice).
//!   2. Concrete `new_brk < current_brk` → no-op, return current brk.
//!      This includes `brk(0)` → return current brk (Linux convention),
//!      since the default brk (0x1B00000) is always > 0.
//!   3. Growth beyond `MAX_MAP_SIZE` → refuse and return the *current*
//!      break unchanged, matching how Linux signals a failed brk. This
//!      is a host-safety bound, not a Python-parity rule; see
//!      `MAX_MAP_SIZE` in `syscalls::mod` (angr-c7xno.80).
//!   4. Otherwise: set posix_brk = new_brk; if it grew across a page
//!      boundary, map the new pages with RWX permissions; return
//!      new_brk.
//!
//! Collision handling: Python's `set_brk` catches a `SimMemoryError`
//! from `map_region` and uses `e.args[1]` as the alternate brk. Rust's
//! `SymbolicMemory::map` is idempotent (silently leaves already-mapped
//! pages alone) and has no equivalent collision signal, so when we
//! detect that any page in the to-be-mapped range is already mapped
//! we fall back to the Python path to preserve semantics.

use super::page::{PAGE_MASK, PAGE_SIZE, PageIndex};
use super::require_syscall_args;
use super::{
    BoundedArg, MAX_MAP_SIZE as MAX_BRK_GROWTH, NativeSyscall, SyscallError, SyscallOutcome,
    bounded_value, extract_concrete_arg,
};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub(crate) struct NativeBrkSyscall;

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
        require_syscall_args!(self, args);
        let new_brk = extract_concrete_arg(&args[0], "brk new_brk")?;

        let current = state.posix_brk();

        // brk(addr) where addr < current is a no-op that returns the
        // current break. brk(0) is the canonical "query" form and is
        // covered here since posix_brk default 0x1B00000 > 0.
        if new_brk < current {
            return Ok(SyscallOutcome::Continue { ret: current });
        }

        // Host-safety cap on the *growth*, not the absolute break: a guest
        // can pass any concrete new_brk, and each new page costs a real eager
        // heap allocation (`MemoryPage::new`), with the collision scan below
        // walking the range page-by-page before that. Refuse an absurd jump
        // instead of OOM-ing (or hanging) the host process. Linux signals brk
        // failure by leaving the break where it was and returning it, which is
        // exactly the `new_brk < current` no-op above; falling back to Python
        // would only move the same unbounded `map_region` there. See
        // `MAX_MAP_SIZE` in `syscalls::mod` (angr-c7xno.80). The bounded
        // quantity is derived, not a raw argument, so this goes through
        // `bounded_value` rather than `extract_bounded_concrete_arg` — the
        // `BoundedArg` match is what makes the refusal arm mandatory
        // (angr-91vj9.1).
        // overflow-ok: new_brk < current returned above, so new_brk >= current.
        match bounded_value("brk growth", new_brk - current, MAX_BRK_GROWTH) {
            BoundedArg::Within(_) => {}
            BoundedArg::Exceeds => return Ok(SyscallOutcome::Continue { ret: current }),
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
            // Align up: pages [aligned_start, aligned_end). wrapping_add since
            // new_brk is a guest-controlled concrete value bounded only by
            // MAX_BRK_GROWTH relative to `current`, not by an absolute cap.
            let aligned_start = current.wrapping_add(PAGE_MASK) & !PAGE_MASK;
            let aligned_end = new_brk.wrapping_add(PAGE_MASK) & !PAGE_MASK;

            // Collision check: if any page in the new range is already
            // mapped, fall back to Python so its SimMemoryError-driven
            // fixup logic runs.
            let memory = state.memory();
            let mut page_addr = aligned_start;
            while page_addr < aligned_end {
                // `.get()` is the one boundary crossing: `page_permissions` is
                // part of the raw-`u64` page API inside `memory/`. Open-coding
                // the shift is the drift `PageIndex` exists to prevent — see
                // its doc in `memory/page.rs`, and the identical conversion in
                // sibling `mprotect.rs`.
                let page_num = PageIndex::of(page_addr).get();
                if memory.page_permissions(page_num).is_some() {
                    return Err(SyscallError::Other(format!(
                        "brk: page {page_addr:#x} already mapped (collision)"
                    )));
                }
                page_addr += PAGE_SIZE;
            }

            // Map the new pages with RWX (matches Python `map_region(..., 7)`).
            let memory = state.memory_mut();
            memory.map(
                aligned_start,
                aligned_end.wrapping_sub(aligned_start),
                Permission::RWX,
            );
        }

        state.set_posix_brk(new_brk);
        Ok(SyscallOutcome::Continue { ret: new_brk })
    }
}

test_submod!("brk_tests.rs" => brk_tests);
