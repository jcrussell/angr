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
//!   `angr.options.CGC_NON_BLOCKING_FDS` set.
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

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, exit, extract_concrete_arg};
use crate::memory::Permission;
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
pub use exit::NativeExitSyscall as NativeTerminateSyscall;

/// Cap on transmit / receive / random byte counts. Same value as the
/// per-syscall MAX in `read.rs` / `write.rs`; keeps a runaway concrete
/// count from filling memory before the dispatcher can react.
const MAX_CGC_BYTES: u64 = 4096;

/// Unique-name counters; one per syscall family. Each syscall mints
/// `(family)_{id}_{i}` for byte `i` of read `id`.
static TRANSMIT_COUNTER: AtomicU64 = AtomicU64::new(0);
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
pub struct NativeTransmitSyscall;

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
        if args.len() < 4 {
            return Err(SyscallError::Other(format!(
                "transmit expected 4 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "transmit fd")?;
        let buf = extract_concrete_arg(&args[1], "transmit buf")?;
        let count = extract_concrete_arg(&args[2], "transmit count")?;
        let tx_bytes = extract_concrete_arg(&args[3], "transmit tx_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "transmit count {} exceeds limit",
                count
            )));
        }

        // Only fds open in the Rust FileSystem (stdout=1, stderr=2,
        // pre-registered) are handled natively. fd=0 (stdin) is
        // nonsensical for transmit and we let Python reject it.
        let fd_u32 = fd as u32;
        if fd == 0 || !state.file_system_ref().is_open(fd_u32) {
            return Err(SyscallError::Other(format!(
                "transmit fd={} falls back to Python",
                fd
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

        state.write_fd(fd_u32, &bytes);
        // Bump the unique-name counter so Python-side correlation IDs
        // never collide if a Python transmit also runs (e.g. after a
        // fallback to allocate/deallocate then back to native).
        let _ = TRANSMIT_COUNTER.fetch_add(1, Ordering::Relaxed);

        if tx_bytes != 0 {
            store_u32_le(state, tx_bytes, count as u32)?;
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

// =====================================================================
// 3: receive(fd, buf, count, rx_bytes) -> int
// =====================================================================
pub struct NativeReceiveSyscall;

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
        if args.len() < 4 {
            return Err(SyscallError::Other(format!(
                "receive expected 4 args, got {}",
                args.len()
            )));
        }
        let fd = extract_concrete_arg(&args[0], "receive fd")?;
        let buf = extract_concrete_arg(&args[1], "receive buf")?;
        let count = extract_concrete_arg(&args[2], "receive count")?;
        let rx_bytes = extract_concrete_arg(&args[3], "receive rx_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "receive count {} exceeds limit",
                count
            )));
        }
        if count == 0 {
            if rx_bytes != 0 {
                store_u32_le(state, rx_bytes, 0)?;
            }
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        if fd != 0 {
            // Non-stdin receive: defer to Python's symbolic-file model.
            return Err(SyscallError::Other(format!(
                "receive fd={} falls back to Python",
                fd
            )));
        }

        // Write fresh symbolic bytes into [buf, buf+count). Each byte
        // gets a unique name and is tracked under state.stdin_symbols
        // so a downstream `posix.dumps(0)` over the Python state can
        // recover the model — same precedent as `NativeReadSyscall` for
        // Linux stdin.
        let read_id = RECEIVE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let names: Vec<String> = (0..count)
            .map(|i| format!("cgc_receive_{read_id}_{i}"))
            .collect();
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            names
                .iter()
                .map(|name| RustBV::symbolic(&ctx, name, 8))
                .collect()
        };
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state
                .memory_store(buf.wrapping_add(i as u64), sym_byte)
                .map_err(SyscallError::Memory)?;
        }
        // Record each fresh symbolic byte under state.stdin_symbols (in read
        // order) so the Python-side `_inject_rust_stdin` can evaluate them via
        // the Rust solver and feed posix.dumps(0) — same precedent as
        // `NativeReadSyscall::read_stdin_symbolic` for Linux stdin.
        for name in names {
            state.record_stdin_symbol(name, 8);
        }

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
// total-ready count written at `*readyfds` is `min(nfds, 32) * 2`
// when both mask pointers are non-zero, else half that — matching the
// Python proc's "1 bit per fd per mask" counting under
// `CGC_NON_BLOCKING_FDS`.
pub struct NativeFdwaitSyscall;

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
        if args.len() < 5 {
            return Err(SyscallError::Other(format!(
                "fdwait expected 5 args, got {}",
                args.len()
            )));
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

        let mut total_ready: u32 = 0;
        if readfds != 0 {
            store_u32_le(state, readfds, mask)?;
            total_ready += queried;
        }
        if writefds != 0 {
            store_u32_le(state, writefds, mask)?;
            total_ready += queried;
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
pub struct NativeRandomSyscall;

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
        if args.len() < 3 {
            return Err(SyscallError::Other(format!(
                "random expected 3 args, got {}",
                args.len()
            )));
        }
        let buf = extract_concrete_arg(&args[0], "random buf")?;
        let count = extract_concrete_arg(&args[1], "random count")?;
        let rnd_bytes = extract_concrete_arg(&args[2], "random rnd_bytes")?;

        if count > MAX_CGC_BYTES {
            return Err(SyscallError::Other(format!(
                "random count {} exceeds limit",
                count
            )));
        }
        if count == 0 {
            if rnd_bytes != 0 {
                store_u32_le(state, rnd_bytes, 0)?;
            }
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        let read_id = RANDOM_COUNTER.fetch_add(1, Ordering::Relaxed);
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            (0..count)
                .map(|i| {
                    let name = format!("cgc_random_{read_id}_{i}");
                    RustBV::symbolic(&ctx, &name, 8)
                })
                .collect()
        };
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state
                .memory_store(buf.wrapping_add(i as u64), sym_byte)
                .map_err(SyscallError::Memory)?;
        }

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
pub struct NativeAllocateSyscall;

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
        if args.len() < 3 {
            return Err(SyscallError::Other(format!(
                "allocate expected 3 args, got {}",
                args.len()
            )));
        }
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
pub struct NativeDeallocateSyscall;

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
        if args.len() < 2 {
            return Err(SyscallError::Other(format!(
                "deallocate expected 2 args, got {}",
                args.len()
            )));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    fn x86_state_with_buf() -> RustSimState {
        let mut state = RustSimState::new("x86").expect("x86 state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
    }

    #[test]
    fn terminate_returns_exit_outcome() {
        let h = NativeTerminateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h.call(&mut state, &[]).expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Exit));
    }

    #[test]
    fn transmit_writes_to_stdout_and_stores_count() {
        let h = NativeTransmitSyscall;
        let mut state = x86_state_with_buf();
        state.map_memory_data(0x3000, b"hello", Permission::RWX);

        // fd=1 (stdout), buf=0x3000, count=5, tx_bytes=0x2000.
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0x3000, 32),
                    RustBV::concrete(5, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        assert_eq!(state.stdout_buffer(), b"hello");
        let stored = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(stored, Some(5));
    }

    #[test]
    fn transmit_with_null_tx_bytes_skips_store() {
        let h = NativeTransmitSyscall;
        let mut state = x86_state_with_buf();
        state.map_memory_data(0x3000, b"x", Permission::RWX);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0x3000, 32),
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        assert_eq!(state.stdout_buffer(), b"x");
    }

    #[test]
    fn transmit_symbolic_byte_falls_back() {
        let h = NativeTransmitSyscall;
        let mut state = x86_state_with_buf();
        state.map_memory(0x3000, 0x1000, Permission::RWX);
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "b", 8);
        state.memory_store(0x3000, sym).expect("store");
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0x3000, 32),
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0, 32),
                ],
            )
            .expect_err("fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
        // No partial transmit.
        assert_eq!(state.stdout_buffer(), b"");
    }

    #[test]
    fn transmit_unknown_fd_falls_back() {
        let h = NativeTransmitSyscall;
        let mut state = x86_state_with_buf();
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0, 32),
                ],
            )
            .expect_err("fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn receive_stdin_writes_symbolic_and_stores_count() {
        let h = NativeReceiveSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(4, 32),
                    RustBV::concrete(0x2800, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        for i in 0..4u64 {
            let byte = state.memory_load(0x2000 + i, 1).expect("load");
            assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
        }
        let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
        assert_eq!(stored, Some(4));
    }

    #[test]
    fn receive_stdin_records_stdin_symbols() {
        // Regression (angr-vx8p.3): each fresh symbolic byte from a CGC
        // stdin `receive` must be tracked under state.stdin_symbols so the
        // Python-side _inject_rust_stdin can feed posix.dumps(0). Mirrors
        // read.rs::test_read_records_stdin_symbols.
        let h = NativeReceiveSyscall;
        let mut state = x86_state_with_buf();
        assert!(!state.has_stdin_symbols());
        let _ = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(4, 32),
                    RustBV::concrete(0x2800, 32),
                ],
            )
            .expect("ok");
        let symbols = state.stdin_symbols();
        assert_eq!(symbols.len(), 4);
        for (name, bits) in symbols {
            assert_eq!(*bits, 8);
            assert!(name.starts_with("cgc_receive_"), "got {name}");
        }
    }

    #[test]
    fn receive_non_stdin_fd_records_no_stdin_symbols() {
        // A non-stdin receive falls back to Python and must NOT pollute
        // stdin_symbols.
        let h = NativeReceiveSyscall;
        let mut state = x86_state_with_buf();
        let _ = h.call(
            &mut state,
            &[
                RustBV::concrete(1, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(4, 32),
                RustBV::concrete(0x2800, 32),
            ],
        );
        assert!(!state.has_stdin_symbols());
    }

    #[test]
    fn receive_zero_count_writes_zero_count_and_returns() {
        let h = NativeReceiveSyscall;
        let mut state = x86_state_with_buf();
        state.map_memory_data(0x2000, b"abcd", Permission::RWX);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2800, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        // The buffer at 0x2000 is untouched.
        for (i, &b) in b"abcd".iter().enumerate() {
            let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
            assert_eq!(byte.as_u64(), Some(b as u64));
        }
        let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
        assert_eq!(stored, Some(0));
    }

    #[test]
    fn receive_non_stdin_fd_falls_back() {
        let h = NativeReceiveSyscall;
        let mut state = x86_state_with_buf();
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(4, 32),
                    RustBV::concrete(0, 32),
                ],
            )
            .expect_err("fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn fdwait_sets_masks_and_total() {
        let h = NativeFdwaitSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(4, 32),      // nfds
                    RustBV::concrete(0x2000, 32), // readfds
                    RustBV::concrete(0x2100, 32), // writefds
                    RustBV::concrete(0, 32),      // timeout (null)
                    RustBV::concrete(0x2200, 32), // readyfds
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(rd, Some(0b1111));
        let wr = state.memory_load(0x2100, 4).expect("load").as_u64();
        assert_eq!(wr, Some(0b1111));
        let total = state.memory_load(0x2200, 4).expect("load").as_u64();
        assert_eq!(total, Some(8));
    }

    #[test]
    fn fdwait_nfds_zero_yields_zero_mask() {
        let h = NativeFdwaitSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2200, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(rd, Some(0));
        let total = state.memory_load(0x2200, 4).expect("load").as_u64();
        assert_eq!(total, Some(0));
    }

    #[test]
    fn fdwait_clamps_to_32_fds() {
        let h = NativeFdwaitSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(100, 32),
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2200, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(rd, Some(u32::MAX as u64));
        let total = state.memory_load(0x2200, 4).expect("load").as_u64();
        assert_eq!(total, Some(32));
    }

    #[test]
    fn random_writes_symbolic_and_stores_count() {
        let h = NativeRandomSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(8, 32),
                    RustBV::concrete(0x2800, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        for i in 0..8u64 {
            let byte = state.memory_load(0x2000 + i, 1).expect("load");
            assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
        }
        let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
        assert_eq!(stored, Some(8));
    }

    #[test]
    fn allocate_zero_length_returns_einval() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
        ));
    }

    #[test]
    fn allocate_oversize_length_returns_einval() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(CGC_MAX_ALLOCATION as u128 + 1, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
        ));
    }

    #[test]
    fn allocate_null_addr_returns_efault() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EFAULT
        ));
    }

    #[test]
    fn allocate_bumps_and_maps_region() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let base_before = state.cgc_allocation_base();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        // allocation_base shrinks by one page.
        assert_eq!(state.cgc_allocation_base(), base_before - 0x1000);
        let chosen = base_before - 0x1000;
        // The chosen address is written back to *addr.
        let stored = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(stored, Some(chosen));
        // The chosen page is now mapped (sanity-check via a store).
        state
            .memory_store(chosen, RustBV::concrete(42, 8))
            .expect("store on freshly allocated page");
    }

    #[test]
    fn allocate_with_is_x_grants_execute() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        h.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(1, 32), // is_x = true
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
        let chosen = 0xB800_0000 - 0x1000;
        let page_num = chosen >> 12;
        let perm = state.memory().page_permissions(page_num).expect("mapped");
        assert!(perm.execute);
        assert!(perm.read);
        assert!(perm.write);
    }

    #[test]
    fn allocate_round_up_length_to_page() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let base_before = state.cgc_allocation_base();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 32), // 1 byte → 1 page
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        assert_eq!(state.cgc_allocation_base(), base_before - 0x1000);
    }

    #[test]
    fn allocate_reuses_sinkhole_first_fit() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        // Pre-seed two sinkholes; first-fit picks the highest-addr one.
        state.cgc_add_sinkhole(0x9000_0000, 0x2000);
        state.cgc_add_sinkhole(0xA000_0000, 0x2000);
        let base_before = state.cgc_allocation_base();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        // allocation_base unchanged (came from sinkhole).
        assert_eq!(state.cgc_allocation_base(), base_before);
        // chosen is HIGH end of highest-addr sinkhole: 0xA0000000 + 0x1000.
        let chosen = state.memory_load(0x2000, 4).expect("load").as_u64();
        assert_eq!(chosen, Some(0xA000_1000));
        // The remaining 0x1000 of the 0xA000_0000 sinkhole stays in the list.
        let sinks = state.cgc_sinkholes();
        let has_lo_a = sinks.iter().any(|&(a, l)| a == 0xA000_0000 && l == 0x1000);
        let has_lo_9 = sinks.iter().any(|&(a, l)| a == 0x9000_0000 && l == 0x2000);
        assert!(has_lo_a, "remainder of A sinkhole should be retained");
        assert!(has_lo_9, "untouched 9 sinkhole should be retained");
    }

    #[test]
    fn allocate_symbolic_length_falls_back() {
        let h = NativeAllocateSyscall;
        let mut state = x86_state_with_buf();
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "len", 32);
        let err = h
            .call(
                &mut state,
                &[sym, RustBV::concrete(0, 32), RustBV::concrete(0x2000, 32)],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn deallocate_unaligned_addr_returns_einval() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x1001, 32), RustBV::concrete(0x1000, 32)],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
        ));
    }

    #[test]
    fn deallocate_zero_length_returns_einval() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 32), RustBV::concrete(0, 32)],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
        ));
    }

    #[test]
    fn deallocate_null_addr_returns_einval() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0, 32), RustBV::concrete(0x1000, 32)],
            )
            .expect("ok");
        assert!(matches!(
            outcome,
            SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
        ));
    }

    #[test]
    fn deallocate_unmaps_region_and_records_sinkhole() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        // x86_state_with_buf maps [0x2000, 0x3000) RWX; deallocate it.
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 32), RustBV::concrete(0x1000, 32)],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        // Page is gone.
        assert!(!state.memory().is_mapped(0x2000u64));
        // Sinkhole records the freed run.
        let sinks = state.cgc_sinkholes();
        assert!(sinks.iter().any(|&(a, l)| a == 0x2000 && l == 0x1000));
    }

    #[test]
    fn deallocate_unmapped_region_is_noop_with_zero_ret() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        // 0x9000 is not mapped — Python procedure returns 0 with no work.
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x9000, 32), RustBV::concrete(0x1000, 32)],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        // No sinkhole entry added.
        assert!(state.cgc_sinkholes().is_empty());
    }

    #[test]
    fn deallocate_partial_run_unmaps_what_it_can() {
        let h = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        // Map a single page at 0x4000, ask to deallocate two pages — the
        // procedure should only free the one that's mapped.
        state.map_memory(0x4000, 0x1000, Permission::RWX);
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x4000, 32), RustBV::concrete(0x2000, 32)],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        assert!(!state.memory().is_mapped(0x4000u64));
        let sinks = state.cgc_sinkholes();
        assert!(sinks.iter().any(|&(a, l)| a == 0x4000 && l == 0x1000));
    }

    #[test]
    fn deallocate_then_allocate_reuses_freed_region() {
        let alloc = NativeAllocateSyscall;
        let dealloc = NativeDeallocateSyscall;
        let mut state = x86_state_with_buf();
        // First allocate to bump the high-water down by a page.
        alloc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("alloc ok");
        let first_chosen = state
            .memory_load(0x2000, 4)
            .expect("load")
            .as_u64()
            .expect("concrete");
        // Deallocate that page.
        dealloc
            .call(
                &mut state,
                &[
                    RustBV::concrete(first_chosen as u128, 32),
                    RustBV::concrete(0x1000, 32),
                ],
            )
            .expect("dealloc ok");
        // Allocate again — should hand back the same page from the sinkhole.
        alloc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2000, 32),
                ],
            )
            .expect("realloc ok");
        let second_chosen = state
            .memory_load(0x2000, 4)
            .expect("load")
            .as_u64()
            .expect("concrete");
        assert_eq!(first_chosen, second_chosen, "sinkhole should be reused");
    }

    #[test]
    fn random_zero_count_writes_zero_and_returns() {
        let h = NativeRandomSyscall;
        let mut state = x86_state_with_buf();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 32),
                    RustBV::concrete(0, 32),
                    RustBV::concrete(0x2800, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
        assert_eq!(stored, Some(0));
    }
}
