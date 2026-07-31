//! Process/startup syscalls that previously round-tripped to Python
//! (angr-6ylm): uname / set_tid_address / set_robust_list / getrandom.
//!
//! These are hit during libc / dynamic-loader startup on essentially every
//! Linux binary:
//!   - `uname(buf)` mirrors `linux_kernel/uname.py`: writes a fixed
//!     `struct utsname` (five 65-byte NUL-padded fields) and returns 0.
//!   - `set_tid_address(tidptr)` mirrors `linux_kernel/set_tid_address.py`:
//!     single-threaded model, returns tid 1. The `tidptr` is ignored (we do
//!     not maintain a clear-child-tid futex).
//!   - `set_robust_list(head, len)` has no Python `SimProcedure`; the kernel
//!     returns 0 on success and the futex robust-list is irrelevant to
//!     single-threaded symbolic execution, so we stub it to 0.
//!   - `getrandom(buf, buflen, flags)` has no Python `SimProcedure` (the
//!     stub path mints a symbolic return). We do better: fill `buf` with
//!     `buflen` fresh symbolic bytes (the randomness) and return `buflen`,
//!     so a downstream read of the buffer sees unconstrained bytes rather
//!     than concrete zeros.
//!
//! Symbolic / oversize args fall back to Python per the usual pattern.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `struct utsname` field width used by Linux/glibc and by
/// `linux_kernel/uname.py`.
const UTSNAME_FIELD: u64 = 65;

/// Cap on `getrandom` buflen handled natively; larger falls back.
const MAX_GETRANDOM: u64 = 4096;

/// Counter for unique getrandom byte names (see `read::SYS_READ_COUNTER`).
static SYS_GETRANDOM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Store `val` NUL-padded to `UTSNAME_FIELD` bytes at `buf + off`; returns
/// the next offset. Mirrors `uname.py::_store`.
fn store_field(
    state: &mut RustSimState,
    buf: u64,
    off: u64,
    val: &[u8],
) -> Result<u64, SyscallError> {
    for i in 0..UTSNAME_FIELD {
        let byte = val.get(i as usize).copied().unwrap_or(0);
        state.memory_store(buf.wrapping_add(off + i), RustBV::concrete(byte as u128, 8))?;
    }
    Ok(off + UTSNAME_FIELD)
}

/// `uname(buf)` — fill a fixed `struct utsname`.
pub(crate) struct NativeUnameSyscall;

impl NativeSyscall for NativeUnameSyscall {
    fn name(&self) -> &'static str {
        "uname"
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
            return Err(SyscallError::Other(
                "uname expected 1 arg, got 0".to_string(),
            ));
        }
        let buf = extract_concrete_arg(&args[0], "uname buf")?;
        let machine: &[u8] = if state.arch().bits() == 64 {
            b"x86_64"
        } else {
            b"x86"
        };
        let mut off = store_field(state, buf, 0, b"Linux")?;
        off = store_field(state, buf, off, b"localhost")?;
        off = store_field(state, buf, off, b"4.0.0")?;
        off = store_field(state, buf, off, b"#1 SMP Mon Jan 01 00:00:00 GMT 1970")?;
        store_field(state, buf, off, machine)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `set_tid_address(tidptr)` — single-threaded, returns tid 1.
pub(crate) struct NativeSetTidAddressSyscall;

impl NativeSyscall for NativeSetTidAddressSyscall {
    fn name(&self) -> &'static str {
        "set_tid_address"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Continue { ret: 1 })
    }
}

/// `set_robust_list(head, len)` — no-op success.
pub(crate) struct NativeSetRobustListSyscall;

impl NativeSyscall for NativeSetRobustListSyscall {
    fn name(&self) -> &'static str {
        "set_robust_list"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `getrandom(buf, buflen, flags)` — fill `buf` with symbolic randomness.
pub(crate) struct NativeGetrandomSyscall;

impl NativeSyscall for NativeGetrandomSyscall {
    fn name(&self) -> &'static str {
        "getrandom"
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
                "getrandom expected 3 args, got {}",
                args.len()
            )));
        }
        let buf = extract_concrete_arg(&args[0], "getrandom buf")?;
        let buflen = extract_concrete_arg(&args[1], "getrandom buflen")?;
        // flags (args[2]) ignored: GRND_NONBLOCK/GRND_RANDOM do not change the
        // symbolic-byte model.
        if buflen > MAX_GETRANDOM {
            return Err(SyscallError::Other(format!(
                "getrandom buflen {buflen} exceeds limit"
            )));
        }
        if buflen == 0 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        let read_id = SYS_GETRANDOM_COUNTER.fetch_add(1, Ordering::Relaxed);
        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            (0..buflen)
                .map(|i| RustBV::symbolic(&ctx, format!("sys_getrandom_{read_id}_{i}"), 8))
                .collect()
        };
        for (i, b) in sym_bytes.into_iter().enumerate() {
            state.memory_store(buf.wrapping_add(i as u64), b)?;
        }
        Ok(SyscallOutcome::Continue { ret: buflen })
    }
}

#[cfg(test)]
#[path = "startup_tests.rs"]
mod startup_tests;
