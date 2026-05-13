//! amd64 mmap syscall handler.
//!
//! Mirrors `procedures/posix/mmap.py`. Handles the common case
//! (anonymous, concrete args, fd=-1); falls back to Python for the
//! cases that require sim_fd, page-collision retry loops, or symbolic
//! reasoning that the Python procedure already covers.
//!
//! Argument order matches the Linux amd64 syscall ABI (see
//! `invariant-syscall-arg-extraction`):
//!   rdi=addr, rsi=length, rdx=prot, r10=flags, r8=fd, r9=offset.
//!
//! Native subset:
//! 1. All six args are concrete; flags are valid (exactly one of
//!    `MAP_SHARED`/`MAP_PRIVATE`).
//! 2. fd == (u32)-1 (anonymous mapping). Non-anonymous (file-backed)
//!    requires sim_fd plumbing — out of scope, falls back.
//! 3. addr == 0: pick from `mmap_base` and bump (matches
//!    `mmap.allocate_memory`). If the candidate range collides with
//!    already-mapped pages, fall back so Python's loop runs.
//! 4. addr != 0 && !MAP_FIXED: try the requested addr first; on
//!    collision, fall back so Python's loop finds a different addr.
//! 5. addr != 0 && MAP_FIXED: collision → fall back (Python returns -1
//!    in that path; we let it run for fidelity).
//!
//! Bad-flags fast path: when `(flags & (MAP_SHARED|MAP_PRIVATE)) == 0`
//! or both bits are set, Python returns -1 outright. Mirror that here
//! so the syscall completes natively (still in the concrete-args path)
//! instead of routing back to Python.
//!
//! Cross-engine sync: native `mmap_base` advances are NOT propagated
//! back to Python's `state.heap.mmap_base` today (same drift risk as
//! `posix_brk`). This is acceptable while syscall fallbacks rarely
//! interleave with successful native calls; future cross-engine sync
//! work (see angr-0z34 / state-cache sync) should address both.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

const MAP_SHARED: u64 = 0x01;
const MAP_PRIVATE: u64 = 0x02;
/// Linux MAP_FIXED. Not consulted by the native fast path (collisions
/// fall back to Python regardless), but kept here for the spec and used
/// by the `map_fixed_collision_falls_back` test below.
#[allow(dead_code)]
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;

/// Translate Linux PROT bits (0x1=R, 0x2=W, 0x4=X) into the internal
/// `Permission` struct. Mirrors `mprotect::linux_prot_to_permission`.
fn linux_prot_to_permission(prot: u64) -> Permission {
    Permission {
        read: prot & 0x1 != 0,
        write: prot & 0x2 != 0,
        execute: prot & 0x4 != 0,
    }
}

/// Returns true if any page in `[addr, addr+size)` is already mapped.
/// `addr` and `size` may be unaligned; the check covers every page
/// the request would touch.
fn range_collides(state: &RustSimState, addr: u64, size: u64) -> bool {
    if size == 0 {
        return false;
    }
    let start_page = addr >> 12;
    // last byte covered, then its page; matches Python's
    // `((addr + length - 1) & ~0xFFF) + 0x1000` end calculation.
    let last_byte = match addr.checked_add(size - 1) {
        Some(v) => v,
        None => return true, // overflow → treat as collision (fall back)
    };
    let end_page = (last_byte >> 12) + 1;
    let memory = state.memory();
    for page_num in start_page..end_page {
        if memory.page_permissions(page_num).is_some() {
            return true;
        }
    }
    false
}

pub struct NativeMmapSyscall;

impl NativeSyscall for NativeMmapSyscall {
    fn name(&self) -> &'static str {
        "mmap"
    }

    fn num_args(&self) -> usize {
        6
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 6 {
            return Err(SyscallError::Other(format!(
                "mmap expected 6 args, got {}",
                args.len()
            )));
        }
        let addr = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap addr".into()))?;
        let length = args[1]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap length".into()))?;
        let prot = args[2]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap prot".into()))?;
        let flags = args[3]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap flags".into()))?;
        let fd_full = args[4]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap fd".into()))?;
        let _offset = args[5]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("mmap offset".into()))?;

        // Bad-flags fast path: Python's mmap returns BVV(-1, bits) when
        // exactly-one-of(MAP_SHARED, MAP_PRIVATE) doesn't hold.
        let shared = flags & MAP_SHARED != 0;
        let private = flags & MAP_PRIVATE != 0;
        if shared == private {
            return Ok(SyscallOutcome::Continue { ret: u64::MAX });
        }

        // File-backed mappings need sim_fd. Python compares `fd[31:0] == -1`
        // — i.e. only the low 32 bits matter. Anonymous: fd == 0xFFFFFFFF.
        let fd_low32 = fd_full & 0xFFFF_FFFF;
        let is_anonymous = (flags & MAP_ANONYMOUS) != 0;
        if !is_anonymous || fd_low32 != 0xFFFF_FFFF {
            return Err(SyscallError::Other(
                "mmap: file-backed (fd != -1) — fall back".into(),
            ));
        }

        // Length 0: Python's posix mmap.py runs the loop with size=0;
        // map_region(.., 0, ..) on the page system maps zero pages, so
        // Python effectively returns the candidate addr without mapping.
        // Match that here — return the candidate addr (mmap_base if
        // addr=0, else addr) without any mapping. No collision check
        // needed since size=0 covers no pages.
        if length == 0 {
            let ret = if addr == 0 { state.mmap_base() } else { addr };
            return Ok(SyscallOutcome::Continue { ret });
        }

        // Choose a candidate address. addr=0 ⇒ use mmap_base.
        let candidate = if addr == 0 { state.mmap_base() } else { addr };

        // For both addr=0 and addr!=0 (with or without MAP_FIXED), if the
        // requested range collides with an existing mapping, fall back to
        // Python. Reasons:
        //   - addr=0 + collision: Python's allocate_memory loop would try
        //     a different mmap_base. We can't reproduce that loop here
        //     without potentially repeated collisions, and it's rare.
        //   - addr!=0 without MAP_FIXED + collision: Python loops; same.
        //   - addr!=0 + MAP_FIXED + collision: Python returns -1; defer
        //     for fidelity (Python prints a warning on map_region's
        //     SimMemoryError).
        if range_collides(state, candidate, length) {
            return Err(SyscallError::Other(format!(
                "mmap: collision at {:#x}+{:#x} — fall back",
                candidate, length,
            )));
        }

        // All checks passed: map the region.
        let perm = linux_prot_to_permission(prot & 0x7);
        state.memory_mut().map(candidate, length, perm);

        // Bump mmap_base on addr=0 (kernel chooses): align next base up
        // to the next page if the chosen region didn't end on a page
        // boundary. Mirrors mmap.allocate_memory in Python.
        if addr == 0 {
            let new_base = candidate.wrapping_add(length);
            let aligned = if new_base & PAGE_MASK != 0 {
                (new_base & !PAGE_MASK) + PAGE_SIZE
            } else {
                new_base
            };
            state.set_mmap_base(aligned);
        }

        Ok(SyscallOutcome::Continue { ret: candidate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    const DEFAULT_MMAP_BASE: u64 = 0xC100_0000;
    const ANON_FD: u64 = 0xFFFF_FFFF;
    const ANON_PRIVATE: u64 = MAP_ANONYMOUS | MAP_PRIVATE;
    const ANON_SHARED: u64 = MAP_ANONYMOUS | MAP_SHARED;

    fn fresh_state() -> RustSimState {
        RustSimState::new("amd64").expect("amd64 state")
    }

    fn args(addr: u64, length: u64, prot: u64, flags: u64, fd: u64, offset: u64) -> Vec<RustBV> {
        vec![
            RustBV::concrete(addr as u128, 64),
            RustBV::concrete(length as u128, 64),
            RustBV::concrete(prot as u128, 64),
            RustBV::concrete(flags as u128, 64),
            RustBV::concrete(fd as u128, 64),
            RustBV::concrete(offset as u128, 64),
        ]
    }

    #[test]
    fn default_mmap_base_is_0xc1000000() {
        let state = fresh_state();
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn anonymous_addr_zero_allocates_from_mmap_base() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &args(0, 0x1000, 0x3 /* RW */, ANON_PRIVATE, ANON_FD, 0),
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
        assert_eq!(
            state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
            Some(Permission::RW),
        );
    }

    #[test]
    fn anon_unaligned_length_bumps_to_next_page() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // 0x800 bytes — not page-aligned. Python's allocate_memory
        // bumps mmap_base by size, then aligns up to next page.
        h.call(&mut state, &args(0, 0x800, 0x3, ANON_PRIVATE, ANON_FD, 0))
            .expect("ok");
        // First call: allocated at DEFAULT_MMAP_BASE, returned that.
        // mmap_base should now be DEFAULT_MMAP_BASE + 0x1000 (aligned up).
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
        // The second call should land at the next page.
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, DEFAULT_MMAP_BASE + 0x1000);
            }
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x2000);
    }

    #[test]
    fn anon_concrete_addr_succeeds_when_range_unmapped() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let target = 0x4000_0000;
        let outcome = h
            .call(
                &mut state,
                &args(target, 0x2000, 0x5 /* RX */, ANON_PRIVATE, ANON_FD, 0),
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
            _ => panic!("expected Continue"),
        }
        // mmap_base does NOT advance for an explicit-addr request.
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
        assert_eq!(
            state.memory().page_permissions(target >> 12),
            Some(Permission::RX),
        );
        assert_eq!(
            state.memory().page_permissions((target >> 12) + 1),
            Some(Permission::RX),
        );
    }

    #[test]
    fn map_shared_anonymous_also_works() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_SHARED, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn bad_flags_returns_neg_one_no_shared_no_private() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // flags has neither MAP_SHARED nor MAP_PRIVATE → Python returns -1.
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, MAP_ANONYMOUS, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
        // No region mapped, mmap_base unchanged.
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
        assert!(
            state
                .memory()
                .page_permissions(DEFAULT_MMAP_BASE >> 12)
                .is_none()
        );
    }

    #[test]
    fn bad_flags_returns_neg_one_both_shared_and_private() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &args(
                    0,
                    0x1000,
                    0x3,
                    MAP_ANONYMOUS | MAP_SHARED | MAP_PRIVATE,
                    ANON_FD,
                    0,
                ),
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn file_backed_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // fd = 3 (a real file descriptor) + no MAP_ANONYMOUS → Python.
        let err = h
            .call(&mut state, &args(0, 0x1000, 0x3, MAP_PRIVATE, 3, 0))
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn anonymous_with_real_fd_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // MAP_ANONYMOUS but fd=3 (Linux ignores fd in this case but Python
        // checks `fd[31:0] != -1` for sim_fd lookup; mirror Python's path
        // by falling back so the (rare) sim_fd resolution runs there).
        let err = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, 3, 0))
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn collision_at_explicit_addr_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let target = 0x4000_0000;
        // Pre-map one page in the path.
        state.map_memory(target, 0x1000, Permission::RW);

        let err = h
            .call(
                &mut state,
                &args(target, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0),
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
        // Pre-existing mapping intact (no partial mutation).
        assert_eq!(
            state.memory().page_permissions(target >> 12),
            Some(Permission::RW),
        );
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn map_fixed_collision_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let target = 0x4000_0000;
        state.map_memory(target, 0x1000, Permission::RW);

        let err = h
            .call(
                &mut state,
                &args(target, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn collision_at_mmap_base_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // Pre-map at the default mmap_base — pathological but possible
        // if a binary explicitly mapped at 0xC1000000.
        state.map_memory(DEFAULT_MMAP_BASE, 0x1000, Permission::RW);

        let err = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
        // mmap_base must NOT advance on fallback.
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn symbolic_addr_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
        a[0] = RustBV::symbolic(&ctx, "addr", 64);
        let err = h.call(&mut state, &a).expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("addr"), "got {msg:?}")
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn symbolic_length_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
        a[1] = RustBV::symbolic(&ctx, "length", 64);
        let err = h.call(&mut state, &a).expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn symbolic_flags_falls_back() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
        a[3] = RustBV::symbolic(&ctx, "flags", 64);
        let err = h.call(&mut state, &a).expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn prot_zero_maps_no_access_pages() {
        // Python passes `prot[2:0]` to map_region, so prot=0 produces a
        // page with no read/write/execute. Mirror that.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0, ANON_PRIVATE, ANON_FD, 0))
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
        assert_eq!(
            state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
            Some(Permission::NONE),
        );
    }

    #[test]
    fn prot_high_bits_ignored() {
        // mmap.py extracts prot[2:0] (low 3 bits) before mapping. We do
        // `prot & 0x7` to match.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        // prot = 0x107 → keeps low 3 bits = 0x7 (RWX).
        let outcome = h
            .call(
                &mut state,
                &args(0, 0x1000, 0x107, ANON_PRIVATE, ANON_FD, 0),
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
        assert_eq!(
            state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
            Some(Permission::RWX),
        );
    }

    #[test]
    fn length_zero_returns_candidate_no_mapping() {
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0, 0x3, ANON_PRIVATE, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
            _ => panic!("expected Continue"),
        }
        // mmap_base unchanged for size=0 (matches python's allocate_memory
        // which would have left it where it is after aligning new_base=base).
        // Our impl skips set_mmap_base in the size==0 short-circuit.
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
        assert!(
            state
                .memory()
                .page_permissions(DEFAULT_MMAP_BASE >> 12)
                .is_none()
        );
    }

    #[test]
    fn handler_metadata() {
        let h = NativeMmapSyscall;
        assert_eq!(h.name(), "mmap");
        assert_eq!(h.num_args(), 6);
    }

    #[test]
    fn fork_preserves_mmap_base() {
        let mut state = fresh_state();
        state.set_mmap_base(0xC100_5000);
        let forked = state.fork();
        assert_eq!(forked.mmap_base(), 0xC100_5000);
        // Mutating parent does not affect fork.
        state.set_mmap_base(0xC100_9000);
        assert_eq!(forked.mmap_base(), 0xC100_5000);
    }
}
