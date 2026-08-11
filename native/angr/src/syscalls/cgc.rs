//! DECREE CGC syscall ABI handlers (x86 only).
//!
//! The CGC platform defines seven syscalls with their own numbers; these
//! collide with Linux i386 numbers (1=exit, 2=fork, ...) so the
//! dispatcher routes CGC binaries through a separate `"CGC"` table
//! based on `ExecutionEnvironment::os_name`. The arch under DECREE is
//! always 32-bit x86; the CGC syscall calling convention matches the
//! Linux i386 ABI (eax=number, ebx/ecx/edx/esi/edi as arg1..5).
//!
//! # Coverage status
//!
//! * `1 _terminate` — fully native, reuses [`exit::NativeExitSyscall`]
//!   semantics (deadend; ignore exit code).
//! * `2 transmit(fd, buf, count, tx_bytes)` — writes `count` concrete
//!   bytes from `[buf, buf+count)` to `fd` and stores `count` (as 32-bit
//!   LE) at `*tx_bytes` when non-zero. Returns 0 on success. Falls back
//!   to Python for symbolic args / bytes, or when `fd` is not a stdout/
//!   stderr clone — matching the Python `transmit` SimProcedure's
//!   simple-fd case.
//! * `3 receive(fd, buf, count, rx_bytes)` — writes `count` fresh
//!   symbolic bytes (tracked in `state.stdin_symbols`) into `[buf, ...)`
//!   when `fd==0`. Stores `count` (32-bit LE) at `*rx_bytes`. Returns 0
//!   on success. Falls back to Python for symbolic args, other fds, or
//!   large counts. Mirrors the Python `receive` SimProcedure's
//!   stdin-only happy path.
//! * `4 fdwait(nfds, readfds, writefds, timeout, readyfds)` — stub that
//!   models the `CGC_NON_BLOCKING_FDS` mode: every fd is "ready", and
//!   the readfds/writefds bitmasks are filled with concrete 1-bits up
//!   to `min(nfds, 32)`. Returns 0. Matches the Python proc with
//!   `angr.options.CGC_NON_BLOCKING_FDS` set; falls back to Python when
//!   the option is unset (the Python proc then makes each ready bit
//!   unconstrained, which the stub cannot express).
//! * `7 random(buf, count, rnd_bytes)` — writes `count` fresh symbolic
//!   bytes into `[buf, ...)` (no fd) and stores `count` at `*rnd_bytes`.
//!   Returns 0. Mirrors `random` SimProcedure for non-fastpath mode.
//!
//! * `5 allocate(length, is_x, addr)` — simple bump allocator that
//!   mirrors `procedures/cgc/allocate.py`. Uses the state's
//!   `cgc_allocation_base` high-water and `cgc_sinkholes` freelist;
//!   error codes match the Python plugin (EINVAL=3, EFAULT=2). Falls
//!   back to Python on symbolic args or when the bump would collide
//!   with the loader/flag-page region (Rust syscall has no access to
//!   `project.loader`, so the safe path is to defer).
//! * `6 deallocate(addr, length)` — page-aligned unmap with sinkhole
//!   bookkeeping, mirroring `procedures/cgc/deallocate.py`. Walks
//!   consecutive mapped pages, unmaps them, and adds the run to the
//!   sinkhole freelist on success.

use std::sync::atomic::AtomicU64;

use super::require_syscall_args;
use super::{
    MAX_IO_SIZE as MAX_CGC_BYTES, NativeSyscall, SyscallError, SyscallOutcome, exit,
    extract_concrete_arg, fresh_byte_names, mint_symbolic_bytes,
};
use crate::memory::Permission;
use crate::procedures::stdin_common::mint_stdin_bytes;
use crate::procedures::strings::write_bv_bytes;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

// CGC error codes — see `state_plugins/cgc.py::SimStateCGC`. The Python
// plugin defines six error codes but only EINVAL and EFAULT are
// reachable from the allocate / deallocate happy paths.
const CGC_EFAULT: u64 = 2;
const CGC_EINVAL: u64 = 3;

/// Upper bound on a single `allocate` request (`state.cgc.max_allocation`,
/// see `state_plugins/cgc.py`). Requests larger than this return EINVAL
/// without touching the allocation base.
const CGC_MAX_ALLOCATION: u64 = 0x1000_0000;

/// CGC flag-page sentinel (`cgc_flag_page_start_addr` in
/// `procedures/cgc/allocate.py`). The simple Rust bump allocator falls
/// back to Python rather than risk landing on or straddling the flag
/// page — the proper overlap-handling lives in the Python procedure.
const CGC_FLAG_PAGE_START: u64 = 0x4347_C000;
const CGC_FLAG_PAGE_END: u64 = CGC_FLAG_PAGE_START + 0x1000;

/// Re-export so `syscalls::cgc::NativeTerminateSyscall` reads naturally
/// alongside the other CGC handlers.
pub(crate) use exit::NativeExitSyscall as NativeTerminateSyscall;

// The transmit / receive / random byte cap is the shared `MAX_IO_SIZE`
// (aliased above as `MAX_CGC_BYTES`) — see `syscalls::mod`. It keeps a
// runaway concrete count from filling memory before the dispatcher can react,
// and sharing the constant stops these handlers desyncing from read/write/fd_io
// if the cap ever moves (angr-9ke6b.157).

/// Unique-name counters; one per name-minting syscall family. Each mints
/// `(family)_{id}_{i}` for byte `i` of read `id`. `transmit` has no counter
/// because it only reads concrete bytes out — it never mints symbolic names.
static RECEIVE_COUNTER: AtomicU64 = AtomicU64::new(0);
static RANDOM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Store a 32-bit value at `addr` in little-endian, leaving memory at
/// `[addr, addr+4)` holding `value`. Returns Ok(()) on success.
///
/// CGC is x86-only, which is always little-endian; the helper writes a
/// single 32-bit BV via `memory_store` so the existing endianness path
/// in `RustSimState::memory_store` lays out bytes correctly.
fn store_u32_le(state: &mut RustSimState, addr: u64, value: u32) -> Result<(), SyscallError> {
    state
        .memory_store(addr, RustBV::concrete(value as u128, 32))
        .map_err(SyscallError::Memory)
}

// =====================================================================
// 1: _terminate — see `pub use exit::NativeExitSyscall` above.
// =====================================================================

// =====================================================================
// 2: transmit(fd, buf, count, tx_bytes) -> int
// =====================================================================
pub(crate) struct NativeTransmitSyscall;

impl NativeSyscall for NativeTransmitSyscall {
    fn name(&self) -> &'static str {
        "transmit"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let fd = match extract_concrete_arg(&args[0], "transmit fd") {
            Ok(fd) => fd,
            Err(e) => {
                // Symbolic fd on a write path: any bounded symbolic file
                // could be the target — hand them all to Python before the
                // fallback (angr-0xyq2 A4; O(1) when none attached).
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
        let buf = extract_concrete_arg(&args[1], "transmit buf")?;
        let count = extract_concrete_arg(&args[2], "transmit count")?;
        let tx_bytes = extract_concrete_arg(&args[3], "transmit tx_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "transmit count {count} exceeds limit"
            )));
        }

        // Only fds open in the Rust FileSystem (stdout=1, stderr=2,
        // pre-registered) are handled natively. fd=0 (stdin) is
        // nonsensical for transmit and we let Python reject it.
        let fd_u32 = fd as u32;
        if fd == 0 || !state.file_system_ref().is_open(fd_u32) {
            return Err(SyscallError::Other(format!(
                "transmit fd={fd} falls back to Python"
            )));
        }

        // Read the bytes — symbolic bytes fall back to Python where the
        // proper claripy data is written through simfd.write_data.
        let mut bytes = Vec::with_capacity(count as usize);
        for i in 0..count {
            let bv = state.memory_load(buf.wrapping_add(i), 1)?;
            match bv.as_u64() {
                Some(v) => bytes.push(v as u8),
                None => {
                    return Err(SyscallError::SymbolicArgument(format!(
                        "transmit symbolic byte at buf+{i}"
                    )));
                }
            }
        }

        // A refusal means the fd carried bounded symbolic content (now
        // demoted) — bounce to Python (angr-0xyq2 Phase 2 choke point; see
        // FileSystem::write).
        if !state.write_fd(fd_u32, &bytes) {
            return Err(SyscallError::Other(format!(
                "transmit to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        if tx_bytes != 0 {
            store_u32_le(state, tx_bytes, count as u32)?;
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 3: receive(fd, buf, count, rx_bytes) -> int
// =====================================================================
pub(crate) struct NativeReceiveSyscall;

impl NativeSyscall for NativeReceiveSyscall {
    fn name(&self) -> &'static str {
        "receive"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let fd = extract_concrete_arg(&args[0], "receive fd")?;
        let buf = extract_concrete_arg(&args[1], "receive buf")?;
        let count = extract_concrete_arg(&args[2], "receive count")?;
        let rx_bytes = extract_concrete_arg(&args[3], "receive rx_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "receive count {count} exceeds limit"
            )));
        }
        // fd validation precedes every count-based short-circuit, matching
        // both `NativeTransmitSyscall::call` and the Python reference
        // (procedures/cgc/receive.py::run), which resolves `posix.get_fd(fd)`
        // and returns -1 before it ever looks at `count`. Checking `count == 0`
        // first would report success for a never-opened fd (angr-fs583).
        if fd != 0 {
            // Non-stdin receive: defer to Python's symbolic-file model.
            return Err(SyscallError::Other(format!(
                "receive fd={fd} falls back to Python"
            )));
        }

        if count == 0 {
            if rx_bytes != 0 {
                store_u32_le(state, rx_bytes, 0)?;
            }
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        // Write fresh symbolic bytes into [buf, buf+count). Each byte
        // gets a unique name and is tracked under state.stdin_symbols
        // so a downstream `posix.dumps(0)` over the Python state can
        // recover the model — same precedent as `NativeReadSyscall` for
        // Linux stdin.
        let names = fresh_byte_names("cgc_receive", &RECEIVE_COUNTER, count);
        // Not `mint_symbolic_bytes`: `mint_stdin_bytes` binds each leaf to a
        // harness-seeded fd-0 byte when the harness pre-filled
        // `posix.stdin.content`, and records the unseeded ones under
        // state.stdin_symbols (in read order) so the Python-side
        // `_inject_rust_stdin` can evaluate them via the Rust solver and feed
        // posix.dumps(0) — same precedent as `read_stdin_symbolic` for Linux
        // stdin, which shares the helper. Only the naming is shared.
        let sym_bytes: Vec<RustBV> = mint_stdin_bytes(state, &names);
        write_bv_bytes(state, buf, sym_bytes)?;

        if rx_bytes != 0 {
            store_u32_le(state, rx_bytes, count as u32)?;
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 4: fdwait(nfds, readfds, writefds, timeout, readyfds) -> int
// =====================================================================
//
// Stubbed to the `CGC_NON_BLOCKING_FDS` behavior: every queried fd is
// reported ready in both the read and write masks. `timeout` is
// ignored (no per-state CGC time accounting in RustSimState). The
// total-ready count written at `*readyfds` is always `min(nfds, 32) * 2`,
// independent of whether the mask pointers are null — matching the
// Python proc, which accumulates its count across both fd loops
// unconditionally and guards only the mask *stores* on a non-null
// pointer. States without the option defer to Python.
pub(crate) struct NativeFdwaitSyscall;

impl NativeSyscall for NativeFdwaitSyscall {
    fn name(&self) -> &'static str {
        "fdwait"
    }

    fn num_args(&self) -> usize {
        5
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        // The concrete-ready stub below IS the `CGC_NON_BLOCKING_FDS`
        // behavior. Without the option the Python proc fills the masks
        // with *unconstrained* per-fd ready bits, so a binary that
        // branches on FD readiness explores both arms; running the stub
        // anyway would silently prune those paths. Defer to Python
        // instead — that keeps the option honored in both directions
        // rather than only when it is set (angr-op0dn.14.8).
        if !state.has_option("CGC_NON_BLOCKING_FDS") {
            return Err(SyscallError::Other(
                "fdwait without CGC_NON_BLOCKING_FDS needs symbolic ready bits; falls back to Python"
                    .to_string(),
            ));
        }
        let nfds = extract_concrete_arg(&args[0], "fdwait nfds")?;
        let readfds = extract_concrete_arg(&args[1], "fdwait readfds")?;
        let writefds = extract_concrete_arg(&args[2], "fdwait writefds")?;
        let _timeout = extract_concrete_arg(&args[3], "fdwait timeout")?;
        let readyfds = extract_concrete_arg(&args[4], "fdwait readyfds")?;

        // CGC fdwait operates on 32-fd words; the Python proc walks fds
        // 0..32 in 8-fd groups and sets one bit per fd. We mirror that
        // bound to keep the buffer layout identical: 32 bits = 4 bytes.
        let queried = nfds.min(32) as u32;

        // Build a 32-bit mask with the low `queried` bits set, then
        // store it (LE) at readfds/writefds if non-null. The Python
        // proc stores the bits in big-endian-bit order within each byte,
        // but the byte-order itself is the native (little-endian on
        // x86) memory layout — so a 32-bit LE store with the low
        // `queried` bits set matches the resulting byte pattern that
        // the binary reads back via FD_ISSET.
        let mask: u32 = if queried >= 32 {
            u32::MAX
        } else if queried == 0 {
            0
        } else {
            (1u32 << queried) - 1
        };

        // `total_ready` counts each queried fd once per mask -- and does so
        // whether or not the mask pointer is null. That looks wrong, but it
        // is exactly what `procedures/cgc/fdwait.py` does: it accumulates
        // `total_ready` across both fd loops unconditionally and only the
        // *stores* are guarded by `condition=readfds != 0`. Counting only
        // the non-null masks here would under-report by `min(nfds, 32)` per
        // null pointer and diverge from Python (angr-cslvl).
        let total_ready: u32 = queried * 2;
        if readfds != 0 {
            store_u32_le(state, readfds, mask)?;
        }
        if writefds != 0 {
            store_u32_le(state, writefds, mask)?;
        }
        if readyfds != 0 {
            store_u32_le(state, readyfds, total_ready)?;
        }

        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 7: random(buf, count, rnd_bytes) -> int
// =====================================================================
pub(crate) struct NativeRandomSyscall;

impl NativeSyscall for NativeRandomSyscall {
    fn name(&self) -> &'static str {
        "random"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let buf = extract_concrete_arg(&args[0], "random buf")?;
        let count = extract_concrete_arg(&args[1], "random count")?;
        let rnd_bytes = extract_concrete_arg(&args[2], "random rnd_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "random count {count} exceeds limit"
            )));
        }
        if count == 0 {
            if rnd_bytes != 0 {
                store_u32_le(state, rnd_bytes, 0)?;
            }
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        mint_symbolic_bytes(state, buf, count, "cgc_random", &RANDOM_COUNTER)?;

        if rnd_bytes != 0 {
            store_u32_le(state, rnd_bytes, count as u32)?;
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 5: allocate(length, is_x, addr) -> int
// =====================================================================
//
// Mirrors `procedures/cgc/allocate.py` for the concrete-arg happy path:
// validate (length / is_x / addr), align length up to a page, try to
// satisfy from the sinkhole freelist (first-fit, highest-address),
// otherwise bump `cgc_allocation_base` down, map the region, and write
// the chosen address back to `*addr` as a 32-bit LE word.
//
// Returns 0 on success. Returns EINVAL (3) for length == 0 or
// length > max_allocation. Returns EFAULT (2) when `addr` is null.
// Falls back to Python when the bump would touch the loader-owned
// region (Rust has no access to `project.loader.max_addr` here) or the
// CGC flag page — the Python procedure has the proper overlap-handling
// path for both cases.
pub(crate) struct NativeAllocateSyscall;

impl NativeSyscall for NativeAllocateSyscall {
    fn name(&self) -> &'static str {
        "allocate"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let length = extract_concrete_arg(&args[0], "allocate length")?;
        let is_x = extract_concrete_arg(&args[1], "allocate is_x")?;
        let addr_ptr = extract_concrete_arg(&args[2], "allocate addr")?;

        // Error preconditions — match `allocate.py`'s claripy.ite_cases
        // chain in evaluation order: length, max_allocation, addr_ptr.
        if length == 0 || length > CGC_MAX_ALLOCATION {
            return Ok(SyscallOutcome::Continue { ret: CGC_EINVAL });
        }
        if addr_ptr == 0 {
            return Ok(SyscallOutcome::Continue { ret: CGC_EFAULT });
        }

        let aligned_length = length.div_ceil(0x1000) * 0x1000;

        // First-fit over the sinkhole freelist; bump otherwise.
        let chosen = if let Some(addr) = state.cgc_take_max_sinkhole(aligned_length) {
            addr
        } else {
            let base = state.cgc_allocation_base();
            // Reject obvious underflow before touching state.
            let next_base = match base.checked_sub(aligned_length) {
                Some(b) => b,
                None => {
                    return Err(SyscallError::Other(format!(
                        "allocate length {aligned_length:#x} underflows allocation_base {base:#x}",
                    )));
                }
            };
            // Defer to Python if the bump would straddle the CGC flag
            // page. The Python procedure splits the range and routes the
            // request to a fresh region above the flag page; replicating
            // that here would force us to mirror the project loader,
            // which is out of scope for this slice.
            let region_start = next_base;
            let region_end = base;
            if region_start < CGC_FLAG_PAGE_END && region_end > CGC_FLAG_PAGE_START {
                return Err(SyscallError::Other(
                    "allocate region overlaps CGC flag page; falling back to Python".into(),
                ));
            }
            state.set_cgc_allocation_base(next_base);
            next_base
        };

        // Map the region. Permissions match `allocate.py`: RW always,
        // X conditional on the `is_x` arg.
        let perm = Permission {
            read: true,
            write: true,
            execute: is_x != 0,
        };
        state.map_memory(chosen, aligned_length, perm);

        // Store the chosen address back into `*addr` as a 32-bit LE word.
        store_u32_le(state, addr_ptr, chosen as u32)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 6: deallocate(addr, length) -> int
// =====================================================================
//
// Mirrors `procedures/cgc/deallocate.py`: validate alignment / length /
// addr, walk the consecutive mapped page run starting at `addr`, unmap
// what we can, and record the unmapped run as a sinkhole. Returns 0 on
// success, EINVAL on validation failure. Matches the Python procedure's
// quirk that a wholly-unmapped `addr` still returns 0 with no work.
pub(crate) struct NativeDeallocateSyscall;

impl NativeSyscall for NativeDeallocateSyscall {
    fn name(&self) -> &'static str {
        "deallocate"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let addr = extract_concrete_arg(&args[0], "deallocate addr")?;
        let length = extract_concrete_arg(&args[1], "deallocate length")?;

        // Validation order matches deallocate.py's claripy.ite_cases:
        // alignment, length, addr != 0, addr + length != 0.
        if addr & 0xFFF != 0 || length == 0 || addr == 0 {
            return Ok(SyscallOutcome::Continue { ret: CGC_EINVAL });
        }
        let end_excl = match addr.checked_add(length) {
            Some(v) => v,
            None => return Ok(SyscallOutcome::Continue { ret: CGC_EINVAL }),
        };
        if end_excl == 0 {
            return Ok(SyscallOutcome::Continue { ret: CGC_EINVAL });
        }

        let aligned_length = length.div_ceil(0x1000) * 0x1000;

        // Walk consecutive mapped pages starting at `addr` up to
        // `aligned_length`. Python's procedure stops at the first
        // unmapped page in the run and returns 0 with `allowed_pages == 0`
        // as a no-op.
        let mut allowed = 0u64;
        while allowed < aligned_length {
            let probe = addr + allowed;
            if !state.memory().is_mapped(probe) {
                break;
            }
            allowed += 0x1000;
        }

        if allowed == 0 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        state.cgc_add_sinkhole(addr, allowed);
        state.memory_mut().unmap(addr, allowed);
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

test_submod!("cgc_tests.rs" => cgc_tests);
