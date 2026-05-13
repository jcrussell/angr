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

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

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
        let new_brk = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("brk new_brk".into()))?;

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
                        "brk: page {:#x} already mapped (collision)",
                        page_addr
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
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    const DEFAULT_BRK: u64 = 0x1B0_0000;

    fn fresh_state() -> RustSimState {
        RustSimState::new("amd64").expect("amd64 state")
    }

    #[test]
    fn default_brk_is_0x1b00000() {
        let state = fresh_state();
        assert_eq!(state.posix_brk(), DEFAULT_BRK);
    }

    #[test]
    fn brk_zero_returns_current_brk() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_BRK),
            _ => panic!("expected Continue"),
        }
        assert_eq!(
            state.posix_brk(),
            DEFAULT_BRK,
            "posix_brk unchanged on query"
        );
    }

    #[test]
    fn brk_below_current_is_noop() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        // Set brk to something larger first, then ask for a smaller value.
        state.set_posix_brk(0x1B0_4000);
        let outcome = h
            .call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_4000),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.posix_brk(), 0x1B0_4000, "posix_brk unchanged");
    }

    #[test]
    fn brk_grow_within_same_page_does_not_map() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        // Default is 0x1B00000 (page-aligned). Grow within the same
        // page (offset 0..0x800).
        state.set_posix_brk(0x1B0_0010);
        let outcome = h
            .call(&mut state, &[RustBV::concrete(0x1B0_0800, 64)])
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_0800),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.posix_brk(), 0x1B0_0800);
        // No new pages should have been mapped (since both bumps live in
        // page 0x1B00 and we never touched it before).
        assert!(state.memory().page_permissions(DEFAULT_BRK >> 12).is_none());
    }

    #[test]
    fn brk_grow_across_page_boundary_maps_pages() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        // current = default (0x1B00000); grow to 0x1B03000 → maps pages
        // 0x1B00, 0x1B01, 0x1B02 (3 new pages).
        let outcome = h
            .call(&mut state, &[RustBV::concrete(0x1B0_3000, 64)])
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_3000),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.posix_brk(), 0x1B0_3000);
        for pn in [0x1B00, 0x1B01, 0x1B02] {
            assert_eq!(
                state.memory().page_permissions(pn),
                Some(Permission::RWX),
                "page_num {:#x} should be mapped RWX",
                pn,
            );
        }
        // Page after the new break must remain unmapped.
        assert!(state.memory().page_permissions(0x1B03).is_none());
    }

    #[test]
    fn brk_grow_then_grow_only_maps_new_pages() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();

        // First grow: 0x1B00000 → 0x1B01000 (maps page 0x1B00).
        h.call(&mut state, &[RustBV::concrete(0x1B0_1000, 64)])
            .expect("ok");
        // Mutate the freshly-mapped page perms to a sentinel so we can
        // detect if the second grow accidentally re-maps over it.
        state
            .memory_mut()
            .set_page_permissions(0x1B00, Permission::R);

        // Second grow: 0x1B01000 → 0x1B02000 (maps page 0x1B01).
        h.call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
            .expect("ok");

        // Page 0x1B00 should still have its sentinel R perms (not re-mapped).
        assert_eq!(
            state.memory().page_permissions(0x1B00),
            Some(Permission::R),
            "first page must not be re-mapped",
        );
        assert_eq!(
            state.memory().page_permissions(0x1B01),
            Some(Permission::RWX),
            "second page should be newly mapped RWX",
        );
    }

    #[test]
    fn brk_collision_falls_back_to_python() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        // Pre-map a page in the path of the brk grow so the handler
        // detects the collision and returns Err.
        state.map_memory(0x1B0_1000, 0x1000, Permission::RW);

        let err = h
            .call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
        // posix_brk MUST NOT have changed on fallback (Python will run).
        assert_eq!(state.posix_brk(), DEFAULT_BRK);
        // Pre-existing mapping must be intact.
        assert_eq!(
            state.memory().page_permissions(0x1B01),
            Some(Permission::RW),
        );
    }

    #[test]
    fn symbolic_arg_falls_back() {
        let h = NativeBrkSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "new_brk", 64);
        let err = h.call(&mut state, &[sym]).expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("new_brk"),
                "message should name the symbolic arg, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        assert_eq!(state.posix_brk(), DEFAULT_BRK);
    }

    #[test]
    fn handler_metadata() {
        let h = NativeBrkSyscall;
        assert_eq!(h.name(), "brk");
        assert_eq!(h.num_args(), 1);
    }

    #[test]
    fn fork_preserves_posix_brk() {
        let mut state = fresh_state();
        state.set_posix_brk(0x1B0_5000);
        let forked = state.fork();
        assert_eq!(forked.posix_brk(), 0x1B0_5000);
        // Mutating the parent must not affect the fork.
        state.set_posix_brk(0x1B0_9000);
        assert_eq!(forked.posix_brk(), 0x1B0_5000);
    }
}
