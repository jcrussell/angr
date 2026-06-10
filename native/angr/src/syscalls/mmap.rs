//! mmap syscall handlers.
//!
//! Three flavors share a `do_mmap` core:
//!
//! * `NativeMmapSyscall` — modern 6-register form (amd64 syscall 9,
//!   ARM64 syscall 222). Mirrors `procedures/posix/mmap.py`. Argument
//!   order matches the Linux amd64 syscall ABI (see
//!   `invariant-syscall-arg-extraction`):
//!   rdi=addr, rsi=length, rdx=prot, r10=flags, r8=fd, r9=offset.
//! * `NativeOldMmapSyscall` — legacy struct-arg form (i386 syscall 90,
//!   ARM EABI 90, MIPS32 4090). One pointer argument; the handler reads
//!   six 32-bit fields (addr, length, prot, flags, fd, offset) from
//!   guest memory and dispatches into `do_mmap`. Mirrors
//!   `procedures/linux_kernel/mmap.py::old_mmap`.
//! * `NativeMmap2Syscall` — 6-register form with page-unit offset
//!   (i386 syscall 192, ARM EABI 192). The offset argument is in
//!   page-size units; the handler scales it by PAGE_SIZE before
//!   dispatching to `do_mmap`. Mirrors
//!   `procedures/linux_kernel/mmap.py::mmap2`.
//!
//! Handles the common case (anonymous, concrete args, fd=-1); falls
//! back to Python for the cases that require sim_fd, page-collision
//! retry loops, or symbolic reasoning that the Python procedure
//! already covers.
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
//! 5. addr != 0 && MAP_FIXED: POSIX semantics — atomically unmap any
//!    colliding pages in `[addr, addr+length)` and remap with the
//!    requested perms. This diverges from `procedures/posix/mmap.py`
//!    (which returns -1 on collision) but matches real Linux mmap(2):
//!    "If the memory region specified by addr and length overlaps
//!    pages of any existing mapping(s), then the overlapped part of
//!    the existing mapping(s) will be discarded." See angr-ttr7.
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

use super::page::{PAGE_MASK, PAGE_SIZE};
use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAP_SHARED: u64 = 0x01;
const MAP_PRIVATE: u64 = 0x02;
/// Linux MAP_FIXED. When set, the request must be honored at exactly
/// `addr`; any colliding pages are atomically discarded and remapped.
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

/// Shared mmap implementation, called by all three flavors.
///
/// `addr`/`length`/`prot`/`flags`/`fd_full`/`_offset` are the already-extracted
/// concrete syscall args. `_offset` is unused (mmap2/old_mmap callers
/// handle the page-unit scaling before reaching here, matching Python's
/// `posix/mmap.py::run` which also ignores offset for anonymous mappings).
fn do_mmap(
    state: &mut RustSimState,
    addr: u64,
    length: u64,
    prot: u64,
    flags: u64,
    fd_full: u64,
    _offset: u64,
) -> Result<SyscallOutcome, SyscallError> {
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
    let is_fixed = addr != 0 && (flags & MAP_FIXED) != 0;

    // Collision policy:
    //   - addr=0 + collision: fall back. Python's allocate_memory loop
    //     would try a different mmap_base; we can't reproduce that loop
    //     here without potentially repeated collisions, and it's rare.
    //   - addr!=0 && !MAP_FIXED + collision: fall back. Python loops.
    //   - addr!=0 && MAP_FIXED: POSIX says discard colliding pages and
    //     remap at the requested address — handled below (no fallback).
    if !is_fixed && range_collides(state, candidate, length) {
        return Err(SyscallError::Other(format!(
            "mmap: collision at {:#x}+{:#x} — fall back",
            candidate, length,
        )));
    }

    // MAP_FIXED with collision: atomically unmap the colliding range.
    // `unmap` is page-granular and tolerates unmapped pages within
    // the range (it just removes whatever is there), so it's safe to
    // call unconditionally on the requested range. We do it only when
    // MAP_FIXED is set so the non-fixed path stays a pure map().
    let perm = linux_prot_to_permission(prot & 0x7);
    if is_fixed {
        state.memory_mut().unmap(candidate, length);
    }
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
        let addr = extract_concrete_arg(&args[0], "mmap addr")?;
        let length = extract_concrete_arg(&args[1], "mmap length")?;
        let prot = extract_concrete_arg(&args[2], "mmap prot")?;
        let flags = extract_concrete_arg(&args[3], "mmap flags")?;
        let fd_full = extract_concrete_arg(&args[4], "mmap fd")?;
        let offset = extract_concrete_arg(&args[5], "mmap offset")?;
        do_mmap(state, addr, length, prot, flags, fd_full, offset)
    }
}

/// Legacy struct-arg mmap (i386 syscall 90, ARM EABI 90, MIPS32 4090).
///
/// One pointer argument; the kernel reads six 32-bit fields from
/// `[arg, arg+24)`:
///
/// ```c
/// struct mmap_arg_struct {
///     unsigned long addr;
///     unsigned long len;
///     unsigned long prot;
///     unsigned long flags;
///     unsigned long fd;
///     unsigned long offset;  // byte offset, not page units
/// };
/// ```
///
/// Mirrors `procedures/linux_kernel/mmap.py::old_mmap` which dispatches
/// to the base mmap proc after reading 6 dwords. The reads use the
/// memory's natural endianness, so this handler is correct for both
/// LE and BE arches as long as `state.memory()` is configured to match
/// the binary.
pub struct NativeOldMmapSyscall;

impl NativeSyscall for NativeOldMmapSyscall {
    fn name(&self) -> &'static str {
        "old_mmap"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let ptr = extract_concrete_arg(&args[0], "old_mmap arg_struct ptr")?;

        // Read six 32-bit fields. If any byte is symbolic, fall back to
        // Python with a SymbolicArgument error so the syscall completes
        // through the normal callback path.
        let mut fields = [0u64; 6];
        for (i, field) in fields.iter_mut().enumerate() {
            let bv = state
                .memory_load(ptr.wrapping_add(i as u64 * 4), 4)
                .map_err(|e| {
                    SyscallError::Other(format!("old_mmap arg_struct field {i}: {e:?}"))
                })?;
            *field = bv.as_u64().ok_or_else(|| {
                SyscallError::SymbolicArgument(format!("old_mmap arg_struct field {i}"))
            })?;
        }
        let [addr, length, prot, flags, fd_full, offset] = fields;
        do_mmap(state, addr, length, prot, flags, fd_full, offset)
    }
}

/// `mmap2` (i386 syscall 192, ARM EABI 192).
///
/// Same 6-register signature as modern mmap, but the offset is in
/// page-size units rather than bytes — multiplied by `PAGE_SIZE` here
/// before dispatch. Mirrors `procedures/linux_kernel/mmap.py::mmap2`.
///
/// Note: not registered for MIPS32 O32. The O32 ABI passes args 5-6
/// on the stack and `extract_syscall_args` does not currently traverse
/// it; mmap2 falls back to Python on MIPS32 for that reason.
pub struct NativeMmap2Syscall;

impl NativeSyscall for NativeMmap2Syscall {
    fn name(&self) -> &'static str {
        "mmap2"
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
                "mmap2 expected 6 args, got {}",
                args.len()
            )));
        }
        let addr = extract_concrete_arg(&args[0], "mmap2 addr")?;
        let length = extract_concrete_arg(&args[1], "mmap2 length")?;
        let prot = extract_concrete_arg(&args[2], "mmap2 prot")?;
        let flags = extract_concrete_arg(&args[3], "mmap2 flags")?;
        let fd_full = extract_concrete_arg(&args[4], "mmap2 fd")?;
        // Python's mmap2 zero-extends a 32-bit offset to 64 bits before
        // scaling by PAGE_SIZE; the syscall ABI already delivers a 32-bit
        // unsigned value zero-extended into the 64-bit register on our
        // side, so wrapping_mul is enough.
        let offset_pages = extract_concrete_arg(&args[5], "mmap2 offset")?;
        let offset = offset_pages.wrapping_mul(PAGE_SIZE);
        do_mmap(state, addr, length, prot, flags, fd_full, offset)
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
    fn map_fixed_collision_unmaps_and_remaps_natively() {
        // POSIX MAP_FIXED: collision is not an error — the kernel
        // discards the colliding pages and maps the new range at the
        // requested address. angr-ttr7.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let target = 0x4000_0000;
        // Pre-map one page R/W.
        state.map_memory(target, 0x1000, Permission::RW);
        assert_eq!(
            state.memory().page_permissions(target >> 12),
            Some(Permission::RW),
        );

        let outcome = h
            .call(
                &mut state,
                // New mapping: R/X (0x5). Verifies that the prior
                // perms (RW) are replaced, not merged.
                &args(target, 0x1000, 0x5, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
            )
            .expect("MAP_FIXED collision must succeed natively");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
            _ => panic!("expected Continue, got {:?}", outcome),
        }
        // The page is now RX (the new perms), not RW (the old perms).
        assert_eq!(
            state.memory().page_permissions(target >> 12),
            Some(Permission::RX),
        );
        // mmap_base must NOT advance for an explicit-addr request.
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    }

    #[test]
    fn map_fixed_multi_page_collision_discards_all_overlapped_pages() {
        // MAP_FIXED with a 3-page request that overlaps a single
        // pre-existing page in the middle. POSIX semantics: every
        // overlapped page is discarded and remapped with the new perms.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let base = 0x4000_0000;
        // Pre-map only the middle page.
        state.map_memory(base + 0x1000, 0x1000, Permission::RW);

        let outcome = h
            .call(
                &mut state,
                &args(base, 0x3000, 0x7 /* RWX */, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
            )
            .expect("MAP_FIXED multi-page collision must succeed natively");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, base),
            _ => panic!("expected Continue, got {:?}", outcome),
        }
        // All three pages now carry the new RWX perms.
        for i in 0..3 {
            assert_eq!(
                state.memory().page_permissions((base >> 12) + i),
                Some(Permission::RWX),
                "page {i} after MAP_FIXED remap",
            );
        }
    }

    #[test]
    fn map_fixed_clean_addr_still_maps_without_unmap_noise() {
        // MAP_FIXED on a clean address: no collision, plain map.
        // Regression guard for the is_fixed shortcut path.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();
        let target = 0x4000_0000;

        let outcome = h
            .call(
                &mut state,
                &args(target, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
            )
            .expect("MAP_FIXED no-collision must succeed");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
            _ => panic!("expected Continue"),
        }
        assert_eq!(
            state.memory().page_permissions(target >> 12),
            Some(Permission::RW),
        );
    }

    #[test]
    fn map_fixed_with_addr_zero_does_not_engage_fixed_path() {
        // `is_fixed` requires addr != 0 — MAP_FIXED with addr=0 is a
        // nonsensical combination (Linux treats it as a portable
        // suggestion, mapping wherever convenient). Our native path
        // routes through the normal allocate_from_mmap_base flow:
        // if mmap_base is clean, it succeeds without unmapping.
        let h = NativeMmapSyscall;
        let mut state = fresh_state();

        let outcome = h
            .call(
                &mut state,
                &args(0, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
            )
            .expect("MAP_FIXED with addr=0 falls through to allocate");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
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

    // --- Legacy mmap (struct-arg) tests -----------------------------------

    /// Write six little-endian 32-bit fields into memory at `ptr`,
    /// matching the layout `mmap_arg_struct` reads.
    #[allow(clippy::too_many_arguments)]
    fn write_struct_le(
        state: &mut RustSimState,
        ptr: u64,
        addr: u32,
        length: u32,
        prot: u32,
        flags: u32,
        fd: u32,
        offset: u32,
    ) {
        let fields = [addr, length, prot, flags, fd, offset];
        for (i, f) in fields.iter().enumerate() {
            state
                .memory_store(ptr + (i as u64) * 4, RustBV::concrete(*f as u128, 32))
                .expect("store");
        }
    }

    #[test]
    fn old_mmap_dispatches_anonymous_struct_call() {
        let h = NativeOldMmapSyscall;
        let mut state = fresh_state();
        let ptr: u64 = 0x4000;
        state.map_memory(ptr, 0x1000, Permission::RW);
        // Anonymous private, RW perms, addr=0 (kernel chooses).
        write_struct_le(
            &mut state,
            ptr,
            0,        // addr
            0x1000,   // length
            0x3,      // prot RW
            ANON_PRIVATE as u32,
            ANON_FD as u32,
            0,        // offset
        );

        let outcome = h
            .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
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
    fn old_mmap_bad_flags_returns_neg_one() {
        let h = NativeOldMmapSyscall;
        let mut state = fresh_state();
        let ptr: u64 = 0x4000;
        state.map_memory(ptr, 0x1000, Permission::RW);
        // MAP_ANONYMOUS only (no SHARED/PRIVATE) — Python returns -1.
        write_struct_le(
            &mut state,
            ptr,
            0,
            0x1000,
            0x3,
            MAP_ANONYMOUS as u32,
            ANON_FD as u32,
            0,
        );
        let outcome = h
            .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn old_mmap_file_backed_falls_back() {
        let h = NativeOldMmapSyscall;
        let mut state = fresh_state();
        let ptr: u64 = 0x4000;
        state.map_memory(ptr, 0x1000, Permission::RW);
        // fd=3 + no MAP_ANONYMOUS → Python.
        write_struct_le(
            &mut state,
            ptr,
            0,
            0x1000,
            0x3,
            MAP_PRIVATE as u32,
            3,
            0,
        );
        let err = h
            .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
            .expect_err("fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn old_mmap_symbolic_field_falls_back() {
        let h = NativeOldMmapSyscall;
        let mut state = fresh_state();
        let ptr: u64 = 0x4000;
        state.map_memory(ptr, 0x1000, Permission::RW);
        // Initial concrete struct, then poison the `length` field with a
        // symbolic BV so the field read errors with SymbolicArgument.
        write_struct_le(
            &mut state,
            ptr,
            0,
            0x1000,
            0x3,
            ANON_PRIVATE as u32,
            ANON_FD as u32,
            0,
        );
        let ctx = SymContext::new();
        state
            .memory_store(ptr + 4, RustBV::symbolic(&ctx, "len", 32))
            .expect("store sym");
        let err = h
            .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
            .expect_err("fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("field 1"), "got {msg:?}")
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn old_mmap_handler_metadata() {
        let h = NativeOldMmapSyscall;
        assert_eq!(h.name(), "old_mmap");
        assert_eq!(h.num_args(), 1);
    }

    // --- mmap2 (page-offset 6-reg) tests ----------------------------------

    #[test]
    fn mmap2_dispatches_with_zero_offset() {
        let h = NativeMmap2Syscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
    }

    #[test]
    fn mmap2_offset_is_in_page_units() {
        // Anonymous mappings ignore offset, but the multiplication must
        // not overflow or otherwise crash for a non-zero page offset.
        // Page offset 0x20 → byte offset 0x20_000.
        let h = NativeMmap2Syscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0x20))
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
    }

    #[test]
    fn mmap2_bad_flags_returns_neg_one() {
        let h = NativeMmap2Syscall;
        let mut state = fresh_state();
        let outcome = h
            .call(&mut state, &args(0, 0x1000, 0x3, MAP_ANONYMOUS, ANON_FD, 0))
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn mmap2_file_backed_falls_back() {
        let h = NativeMmap2Syscall;
        let mut state = fresh_state();
        let err = h
            .call(&mut state, &args(0, 0x1000, 0x3, MAP_PRIVATE, 3, 0))
            .expect_err("fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn mmap2_symbolic_offset_falls_back() {
        let h = NativeMmap2Syscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
        a[5] = RustBV::symbolic(&ctx, "offset", 64);
        let err = h.call(&mut state, &a).expect_err("fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn mmap2_handler_metadata() {
        let h = NativeMmap2Syscall;
        assert_eq!(h.name(), "mmap2");
        assert_eq!(h.num_args(), 6);
    }
}
