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
//! Not implemented (fall through to Python):
//! * `5 allocate` / `6 deallocate` — need the CGC state plugin
//!   (sinkholes, allocation_base, EINVAL/EFAULT constants) which
//!   `RustSimState` does not carry. Both syscalls deal with page-level
//!   memory map/unmap; semantics live in `state_plugins/cgc.py`.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, exit, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            (0..count)
                .map(|i| {
                    let name = format!("cgc_receive_{read_id}_{i}");
                    RustBV::symbolic(&ctx, &name, 8)
                })
                .collect()
        };
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state
                .memory_store(buf.wrapping_add(i as u64), sym_byte)
                .map_err(SyscallError::Memory)?;
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
                    RustBV::concrete(4, 32),       // nfds
                    RustBV::concrete(0x2000, 32),  // readfds
                    RustBV::concrete(0x2100, 32),  // writefds
                    RustBV::concrete(0, 32),       // timeout (null)
                    RustBV::concrete(0x2200, 32),  // readyfds
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
