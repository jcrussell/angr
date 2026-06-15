//! Directory / file-name syscall handlers (angr-0hif.2).
//!
//! Covers `chdir`, `fchdir`, `getcwd`, `mkdir`, `mkdirat`, `rmdir`,
//! `unlink`, `unlinkat`, `rename`, `renameat`, `renameat2`.
//!
//! ## Coverage shape
//!
//! * **`chdir` / `getcwd`** — back the per-state `cwd: Vec<u8>` on
//!   `FileSystem` added for this subset. `chdir` reads the concrete
//!   C-string at `path_addr` and assigns it raw (no `_normalize_path`,
//!   matching `procedures/linux_kernel/cwd.py::chdir` which also does
//!   `self.state.fs.cwd = cwd`). `getcwd` writes `cwd + b"\0"` to
//!   `buf` and returns the byte count, or `-ERANGE` when `size` is too
//!   small (matches `procedures/linux_kernel/cwd.py::getcwd`).
//! * **`fchdir`** — no Python `SimProcedure`. The unhandled-syscall
//!   path falls through to `procedures/stubs/syscall_stub.py::syscall`,
//!   which returns `ReturnUnconstrained`. The native handler mirrors
//!   that via `SyscallOutcome::ContinueSymbolic`. (We do not look up
//!   the fd → directory mapping because no directory-table model is
//!   wired up; see follow-up note below.)
//! * **`mkdir` / `mkdirat` / `rmdir` / `rename` / `renameat` /
//!   `renameat2`** — no Python proc either; same stub shape.
//! * **`unlink` / `unlinkat`** — `procedures/linux_kernel/unlink.py`
//!   defines real semantics (`state.fs.delete(path)`), but `FileSystem`
//!   in Rust is purely fd-keyed (no path → SimFile map yet). The
//!   handlers below mirror `syscall_stub` for now; angr's real `unlink`
//!   path stays accessible to callers that route through the Python
//!   manager directly. This is the same trade-off bd-k3ol calls out
//!   for the `open` / `close` family.
//!
//! ## What is intentionally NOT covered yet
//!
//! State-attached directory table: the bd description says
//! "mkdir/rmdir/unlink/rename mutate a state-attached directory
//! table." We deliberately stop at the stub shape because the
//! Python `state.fs._files` mapping holds `SimFile` objects with
//! their own symbolic byte-stream model, which has no Rust analogue
//! yet (the FD content in `FileDescriptor::content` is a `Vec<u8>`,
//! not a `SimFile`). Wiring the directory table without that
//! foundation would let the Rust state report "deleted" when the
//! Python proc would still find the file via `state.fs.get(path)`.
//! Tracked as a follow-up bd alongside `angr-k3ol`.

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `-EFAULT` as a 64-bit two's-complement value (kernel-ABI negative errno).
const NEG_EFAULT: u64 = (-14_i64) as u64;
/// `-ERANGE` as a 64-bit two's-complement value (kernel-ABI negative errno).
const NEG_ERANGE: u64 = (-34_i64) as u64;

/// Upper bound on path lengths we walk from memory. Matches the
/// `PATH_MAX` Linux constant. A symbolic byte mid-scan terminates the
/// walk by falling back to `SymbolicArgument(...)` — same shape as
/// `procedures/strtol.rs::read_bytes_until_null`.
const PATH_MAX: usize = 4096;

/// Read a concrete C-string starting at `addr`, stopping at the first
/// NUL byte (exclusive of NUL). Returns `SyscallError::SymbolicArgument`
/// if a symbolic byte is encountered before the NUL, or
/// `SyscallError::Memory(...)` if the load itself fails.
///
/// We need *concrete* path bytes because the Python proc does the same
/// (`self.state.mem[buf].string.concrete`); a symbolic path would never
/// be representable as a single Rust `Vec<u8>` cwd.
fn read_concrete_cstring(
    state: &RustSimState,
    addr: u64,
    max_len: usize,
) -> Result<Vec<u8>, SyscallError> {
    let mut out = Vec::with_capacity(max_len.min(64));
    for i in 0..max_len {
        let byte = state.memory_load(addr.wrapping_add(i as u64), 1)?;
        match byte.as_u64() {
            Some(b) if (b as u8) == 0 => return Ok(out),
            Some(b) => out.push(b as u8),
            None => {
                return Err(SyscallError::SymbolicArgument(format!(
                    "path byte at offset {i} is symbolic"
                )));
            }
        }
    }
    // Hit the PATH_MAX cap without a NUL; treat as the full slice
    // (Linux truncates path lookups at PATH_MAX too).
    Ok(out)
}

// fchdir(fd) → int — no Python SimProcedure; falls through to syscall_stub.
stub_syscall!(NativeFchdirSyscall, "fchdir", "syscall_stub_fchdir", 1);
// mkdir(path, mode) → int — no Python SimProcedure.
stub_syscall!(NativeMkdirSyscall, "mkdir", "syscall_stub_mkdir", 2);
// mkdirat(dfd, path, mode) → int — no Python SimProcedure.
stub_syscall!(NativeMkdiratSyscall, "mkdirat", "syscall_stub_mkdirat", 3);
// rmdir(path) → int — no Python SimProcedure.
stub_syscall!(NativeRmdirSyscall, "rmdir", "syscall_stub_rmdir", 1);
// unlink(path) → int — Python proc exists but needs SimFile plumbing;
// stub for now (see module doc).
stub_syscall!(NativeUnlinkSyscall, "unlink", "syscall_stub_unlink", 1);
// unlinkat(dfd, path, flag) → int — Python lacks a dedicated proc.
stub_syscall!(
    NativeUnlinkatSyscall,
    "unlinkat",
    "syscall_stub_unlinkat",
    3
);
// rename(oldpath, newpath) → int — no Python SimProcedure.
stub_syscall!(NativeRenameSyscall, "rename", "syscall_stub_rename", 2);
// renameat(olddfd, oldpath, newdfd, newpath) → int — no Python proc.
stub_syscall!(
    NativeRenameatSyscall,
    "renameat",
    "syscall_stub_renameat",
    4
);
// renameat2(olddfd, oldpath, newdfd, newpath, flags) → int.
stub_syscall!(
    NativeRenameat2Syscall,
    "renameat2",
    "syscall_stub_renameat2",
    5
);

/// `chdir(path) → 0` — set the per-state cwd to the concrete C-string
/// at `path`. Mirrors `procedures/linux_kernel/cwd.py::chdir`:
/// assigns the raw bytes (no `_normalize_path`).
///
/// `SyscallError::SymbolicArgument` falls back to Python so the proc
/// can apply its `solver.eval_one` concretization path.
pub struct NativeChdirSyscall;

impl NativeSyscall for NativeChdirSyscall {
    fn name(&self) -> &'static str {
        "chdir"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let path_addr = extract_concrete_arg(&args[0], "path")?;
        let path = read_concrete_cstring(state, path_addr, PATH_MAX)?;
        state.file_system().set_cwd(path);
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// `getcwd(buf, size) → len` — write the per-state cwd plus a NUL
/// terminator to `buf`. Returns the byte count (cwd len + 1), or
/// `-ERANGE` if `size` is too small to hold it. Matches
/// `procedures/linux_kernel/cwd.py::getcwd`.
pub struct NativeGetcwdSyscall;

impl NativeSyscall for NativeGetcwdSyscall {
    fn name(&self) -> &'static str {
        "getcwd"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let buf_addr = extract_concrete_arg(&args[0], "buf")?;
        let size = extract_concrete_arg(&args[1], "size")?;

        // cwd + NUL — clone the bytes before any mutable state.file_system()
        // call so the borrow does not extend across memory_store.
        let mut payload = state.file_system_ref().cwd().to_vec();
        payload.push(0u8);

        if (payload.len() as u64) > size {
            return Ok(SyscallOutcome::Continue { ret: NEG_ERANGE });
        }

        // Write one byte at a time — matches the existing
        // memory_store(addr, RustBV::concrete(b, 8)) shape used by the
        // strstr / scanf procs; avoids needing a multi-byte BV builder.
        for (i, b) in payload.iter().enumerate() {
            let dst = buf_addr.wrapping_add(i as u64);
            if let Err(e) = state.memory_store(dst, RustBV::concrete(*b as u128, 8)) {
                // EFAULT mirrors the SimSegfaultException branch of the
                // Python proc. We do not surface SyscallError here
                // because the proc already encodes EFAULT as a negative
                // return rather than a syscall-level error.
                let _ = e;
                return Ok(SyscallOutcome::Continue { ret: NEG_EFAULT });
            }
        }

        Ok(SyscallOutcome::Continue {
            ret: payload.len() as u64,
        })
    }
}

#[cfg(test)]
#[path = "directory_tests.rs"]
mod directory_tests;
