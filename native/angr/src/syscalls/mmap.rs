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
//!    A misaligned addr is only a hint here, so it is accepted and
//!    the page-granular `Memory::map` rounds it down to its containing
//!    page (matching Python).
//! 5. addr != 0 && MAP_FIXED: POSIX semantics — atomically unmap any
//!    colliding pages in `[addr, addr+length)` and remap with the
//!    requested perms. This diverges from `procedures/posix/mmap.py`
//!    (which returns -1 on collision) but matches real Linux mmap(2):
//!    "If the memory region specified by addr and length overlaps
//!    pages of any existing mapping(s), then the overlapped part of
//!    the existing mapping(s) will be discarded." See angr-ttr7.
//!    A *misaligned* MAP_FIXED addr is rejected with -1 (EINVAL),
//!    also matching Linux and again diverging from Python, which
//!    would silently map the containing page instead. See
//!    angr-sqfj8.110 and the check in `do_mmap`.
//!
//! Oversized-request fast path: a `length` above `MAX_MAP_SIZE` is
//! rejected with -1 (MAP_FAILED) rather than mapped or bounced to
//! Python. Host-safety bound, not Python parity — see `MAX_MAP_SIZE`
//! in `syscalls::mod` and the check in `do_mmap` (angr-c7xno.80).
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

use super::page::{PAGE_MASK, PAGE_SIZE, linux_prot_to_permission};
use super::require_syscall_args;
use super::{
    BoundedArg, MAX_MAP_SIZE as MAX_MMAP_SIZE, NativeSyscall, SyscallError, SyscallOutcome,
    bounded_value, extract_concrete_arg,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAP_SHARED: u64 = 0x01;
const MAP_PRIVATE: u64 = 0x02;
/// Linux MAP_FIXED. When set, the request must be honored at exactly
/// `addr`; any colliding pages are atomically discarded and remapped.
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;

/// Returns true if any page in `[addr, addr+size)` is already mapped.
/// `addr` and `size` may be unaligned; the check covers every page
/// the request would touch.
fn range_collides(state: &RustSimState, addr: u64, size: u64) -> bool {
    if size == 0 {
        return false;
    }
    // Overflow → treat as collision (fall back). Checked here rather than
    // leaning on `range_covering`'s saturating add: a wrapping request is a
    // caller error, not a range clamped to the top page.
    // overflow-ok: size == 0 returned above, so size >= 1 here.
    if addr.checked_add(size - 1).is_none() {
        return true;
    }
    // `range_covering` is inclusive of the page holding the last byte, matching
    // Python's `((addr + length - 1) & ~0xFFF) + 0x1000` end calculation.
    let memory = state.memory();
    crate::memory::PageIndex::range_covering(addr, size)
        // `.get()` is the one boundary crossing: `page_permissions` is part of
        // the raw-`u64` page API inside `memory/`.
        .any(|page| memory.page_permissions(page.get()).is_some())
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

    // Host-safety cap. `length` is fully guest-controlled and every page of
    // the request costs a real eager heap allocation (`MemoryPage::new`), so
    // an absurd size must be refused *before* the collision scan — that loop
    // is itself O(pages) and would hang long before the OOM. Linux rejects
    // oversized requests too, with MAP_FAILED/-ENOMEM; return -1 rather than
    // falling back, since Python's `map_region` would allocate just as
    // eagerly. See `MAX_MAP_SIZE` in `syscalls::mod` (angr-c7xno.80). All
    // three flavors funnel through here, and `old_mmap` reads its length out
    // of a guest struct rather than a register, so the check goes through
    // `bounded_value` on the already-extracted length; the `BoundedArg` match
    // is what makes the refusal arm mandatory (angr-91vj9.1).
    match bounded_value("mmap length", length, MAX_MMAP_SIZE) {
        BoundedArg::Within(_) => {}
        BoundedArg::Exceeds => return Ok(SyscallOutcome::Continue { ret: u64::MAX }),
    }

    // Choose a candidate address. addr=0 ⇒ use mmap_base.
    let candidate = if addr == 0 { state.mmap_base() } else { addr };
    let is_fixed = addr != 0 && (flags & MAP_FIXED) != 0;

    // MAP_FIXED demands the mapping land at exactly `addr`, so a
    // misaligned `addr` has no correct answer: `Memory::map`/`unmap`
    // are page-granular (they floor `addr` to its containing page), so
    // honoring the request would silently map a *different* range than
    // the caller named. Linux rejects this outright
    // (`do_mmap`: `if (flags & MAP_FIXED) { if (offset_in_page(addr))
    // return -EINVAL; }`), so return -1 rather than round down. This
    // is a deliberate divergence from `procedures/posix/mmap.py`, which
    // never checks alignment — same class of divergence as the
    // MAP_FIXED collision policy above (angr-ttr7). Non-MAP_FIXED
    // `addr` is only a hint, so it keeps the round-down behavior.
    if is_fixed && addr & PAGE_MASK != 0 {
        return Ok(SyscallOutcome::Continue { ret: u64::MAX });
    }

    // MAP_FIXED skips `range_collides` below, and with it that function's
    // `checked_add` overflow guard — so a page-aligned `addr` near the top of
    // the address space used to fall straight through to `unmap`/`map`, whose
    // page-range helpers reject the wrapping range and skip silently, leaving
    // `do_mmap` to report success for a mapping that was never made
    // (angr-03vl4.68). Mirror `range_collides`'s guard exactly: the *last
    // byte* is what must be representable, so a region ending precisely at
    // `2^64` (e.g. mapping the final page) stays legal, matching
    // `Memory::end_page_exclusive`. `length` is nonzero here — the `length ==
    // 0` fast path returned above. Linux rejects an unrepresentable fixed
    // range with -EINVAL, so return -1 rather than falling back to Python,
    // whose `map_region` would wrap just as silently.
    // overflow-ok: the length == 0 fast path returned above, so length >= 1.
    if is_fixed && candidate.checked_add(length - 1).is_none() {
        return Ok(SyscallOutcome::Continue { ret: u64::MAX });
    }

    // Collision policy:
    //   - addr=0 + collision: fall back. Python's allocate_memory loop
    //     would try a different mmap_base; we can't reproduce that loop
    //     here without potentially repeated collisions, and it's rare.
    //   - addr!=0 && !MAP_FIXED + collision: fall back. Python loops.
    //   - addr!=0 && MAP_FIXED: POSIX says discard colliding pages and
    //     remap at the requested address — handled below (no fallback).
    if !is_fixed && range_collides(state, candidate, length) {
        return Err(SyscallError::Other(format!(
            "mmap: collision at {candidate:#x}+{length:#x} — fall back",
        )));
    }

    // Bump mmap_base on addr=0 (kernel chooses): align next base up
    // to the next page if the chosen region didn't end on a page
    // boundary. Mirrors mmap.allocate_memory in Python.
    //
    // Computed *before* the map so the refusal below leaves the state
    // untouched. `candidate` is `mmap_base`, which nothing clamps
    // (`RustSimState::set_mmap_base`, and every snapshot/fork/merge path that
    // carries it), so the region can end inside — or exactly at the top of —
    // the final page, and there is then no representable next base. The bare
    // `+` this replaces wrapped `mmap_base` to near-zero under the shipped
    // release profile, which aliases the next addr=0 allocation onto already
    // mapped memory; an allocator cursor is an identity, so refuse rather
    // than saturate (bd `invariant-overflow-fix-refuse-not-saturate-identities`)
    // and let Python's `allocate_memory` own the exhausted-address-space case
    // (angr-5mnx3.53). The `is_fixed` sibling guard above cannot cover this:
    // it only runs on the MAP_FIXED branch, and its `length - 1` bound admits
    // a region whose *end* is unrepresentable.
    let next_mmap_base = if addr == 0 {
        let aligned = candidate.checked_add(length).and_then(|end| {
            if end & PAGE_MASK != 0 {
                (end & !PAGE_MASK).checked_add(PAGE_SIZE)
            } else {
                Some(end)
            }
        });
        match aligned {
            Some(base) => Some(base),
            None => {
                return Err(SyscallError::Other(format!(
                    "mmap: next mmap_base past {candidate:#x}+{length:#x} \
                     is unrepresentable — fall back",
                )));
            }
        }
    } else {
        None
    };

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

    if let Some(base) = next_mmap_base {
        state.set_mmap_base(base);
    }

    Ok(SyscallOutcome::Continue { ret: candidate })
}

pub(crate) struct NativeMmapSyscall;

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
        require_syscall_args!(self, args);
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
pub(crate) struct NativeOldMmapSyscall;

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
/// Registered for MIPS32 O32 (4210) as well as i386/ARM (192). O32 passes
/// args 5-6 on the stack at [sp+16]; `extract_syscall_args` traverses that
/// window for concrete SP (angr-tvod), so mmap2 dispatches natively there.
pub(crate) struct NativeMmap2Syscall;

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
        require_syscall_args!(self, args);
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

test_submod!("mmap_tests.rs" => mmap_tests);
