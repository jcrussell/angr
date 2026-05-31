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

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

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
            debug_assert!(updated, "mprotect: page {:#x} disappeared", page_addr);
            page_addr += PAGE_SIZE;
        }

        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::RustBV;

    fn mk_state_with_page(addr: u64, perm: Permission) -> RustSimState {
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory(addr, 0x1000, perm);
        state
    }

    #[test]
    fn linux_prot_bits_translate_correctly() {
        assert_eq!(linux_prot_to_permission(0), Permission::NONE);
        assert_eq!(linux_prot_to_permission(0x1), Permission::R);
        assert_eq!(linux_prot_to_permission(0x3), Permission::RW);
        assert_eq!(linux_prot_to_permission(0x5), Permission::RX);
        assert_eq!(linux_prot_to_permission(0x7), Permission::RWX);
        assert_eq!(linux_prot_to_permission(0x4), Permission::X);
    }

    #[test]
    fn unaligned_addr_returns_neg_one() {
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let args = vec![
            RustBV::concrete(0x1001, 64), // misaligned
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x5, 64), // PROT_READ | PROT_EXEC
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn unmapped_page_returns_neg_one() {
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let args = vec![
            RustBV::concrete(0x2000, 64), // unmapped
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x7, 64),
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn aligned_mapped_succeeds_and_updates_perms() {
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        // Sanity check pre-state.
        assert_eq!(
            state.memory().page_permissions(0x1000 >> 12),
            Some(Permission::RW)
        );

        let args = vec![
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x5, 64), // PROT_READ | PROT_EXEC
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        assert_eq!(
            state.memory().page_permissions(0x1000 >> 12),
            Some(Permission::RX),
        );
    }

    #[test]
    fn multi_page_updates_all_pages() {
        let h = NativeMprotectSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory(0x1000, 0x3000, Permission::RW); // 3 pages: 0x1000..0x4000

        let args = vec![
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x3000, 64),
            RustBV::concrete(0x1, 64), // PROT_READ
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        for pn in [0x1, 0x2, 0x3] {
            assert_eq!(
                state.memory().page_permissions(pn),
                Some(Permission::R),
                "page_num {}",
                pn,
            );
        }
    }

    #[test]
    fn span_crossing_unmapped_page_fails() {
        let h = NativeMprotectSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        // Map page 1 and page 3, leave page 2 (at 0x2000) unmapped.
        state.map_memory(0x1000, 0x1000, Permission::RW);
        state.map_memory(0x3000, 0x1000, Permission::RW);

        let args = vec![
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x3000, 64),
            RustBV::concrete(0x7, 64),
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
        // Permissions on the mapped pages must NOT have been touched on failure.
        assert_eq!(state.memory().page_permissions(0x1), Some(Permission::RW));
        assert_eq!(state.memory().page_permissions(0x3), Some(Permission::RW));
    }

    #[test]
    fn symbolic_addr_falls_back() {
        use crate::symbolic::SymContext;
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let ctx = SymContext::new();
        let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
        let args = vec![
            sym_addr,
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x5, 64),
        ];
        let err = h.call(&mut state, &args).expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("addr"),
                "message should name the symbolic arg, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        // Pre-call perms must be intact (no partial mutation on Err).
        assert_eq!(state.memory().page_permissions(0x1), Some(Permission::RW));
    }

    #[test]
    fn symbolic_length_falls_back() {
        use crate::symbolic::SymContext;
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let ctx = SymContext::new();
        let sym_length = RustBV::symbolic(&ctx, "length", 64);
        let args = vec![
            RustBV::concrete(0x1000, 64),
            sym_length,
            RustBV::concrete(0x5, 64),
        ];
        let err = h.call(&mut state, &args).expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("length"),
                "message should name the symbolic arg, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        assert_eq!(state.memory().page_permissions(0x1), Some(Permission::RW));
    }

    #[test]
    fn symbolic_prot_falls_back() {
        use crate::symbolic::SymContext;
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let ctx = SymContext::new();
        let sym_prot = RustBV::symbolic(&ctx, "prot", 64);
        let args = vec![
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x1000, 64),
            sym_prot,
        ];
        let err = h.call(&mut state, &args).expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("prot"),
                "message should name the symbolic arg, got {msg:?}",
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        assert_eq!(state.memory().page_permissions(0x1), Some(Permission::RW));
    }

    #[test]
    fn zero_length_is_noop() {
        let h = NativeMprotectSyscall;
        let mut state = mk_state_with_page(0x1000, Permission::RW);
        let args = vec![
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0x5, 64),
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        // Permissions unchanged.
        assert_eq!(state.memory().page_permissions(0x1), Some(Permission::RW));
    }

    #[test]
    fn handler_metadata() {
        let h = NativeMprotectSyscall;
        assert_eq!(h.name(), "mprotect");
        assert_eq!(h.num_args(), 3);
    }
}
