//! File-path syscall handlers.
//!
//! Two cohorts live here:
//!
//! ## Symbolic-return stubs (angr-0hif.1)
//!
//! `lstat`, `newfstatat`, `readlink`, `readlinkat`, `faccessat` — none
//! have a dedicated Python `SimProcedure` in
//! `angr/procedures/linux_kernel/`, so the Python path falls through to
//! `procedures/stubs/syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//! The native stubs match: ignore args, emit a fresh `RustBV::symbolic`
//! of width `arch().bits()`, routed through
//! `SyscallOutcome::ContinueSymbolic`.
//!
//! ## FD-allocating handlers (angr-k3ol.1)
//!
//! `open`, `openat`, `close` allocate / release FDs against
//! `RustSimState::file_system()`, mirroring the existing
//! `procedures/fileops::NativeOpen` / `NativeClose` libc procs (which
//! are already wired through `state.file_system()` and therefore
//! diverge from Python `state.posix.fd` in the same way — see the
//! `syscall-vs-procedure-dispatch` bd memory). The Python proc
//! `procedures/posix/open.py` returns `-1` when `state.fs.get(path)`
//! is `None` and creation flags are absent; we always allocate a fresh
//! fd since the Rust-side `FileSystem` does not mirror Python's
//! `state.fs`. This is the same trade-off the libc procedure made.
//!
//! ## `access` (angr-k3ol.2)
//!
//! `access(pathname, mode) → 0 | -1` queries
//! `RustSimState::file_system().is_path_known(path)`, which mirrors the
//! Python proc `procedures/linux_kernel/access.py::run` (returns `-1`
//! when `state.fs.get(path)` is `None`, else `0`). The path set is
//! populated by `open` / `open_with_content` calls — pre-populated
//! Python `state.fs` entries are NOT mirrored unless an explicit
//! `register_known_path` call is made on the Rust state. Same
//! trade-off as `open` / `openat` here.
//!
//! ## What is intentionally NOT covered here (separate subtasks under
//! angr-k3ol):
//!
//! * `stat`, `fstat` — need per-arch `struct stat` field layouts and
//!   `state.posix.fstat_with_result`. Falls back to Python.
//!
//! On those, the unhandled-syscall path continues to dispatch to the
//! Python `_handle_syscall_callback`, preserving full semantics.

use super::{
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall,
};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

// lstat(pathname, statbuf) → long
stub_syscall!(NativeLstatSyscall, "lstat", "syscall_stub_lstat", 2);
// newfstatat(dfd, filename, statbuf, flag) → long
stub_syscall!(NativeNewfstatatSyscall, "newfstatat", "syscall_stub_newfstatat", 4);
// readlink(path, buf, bufsiz) → long
stub_syscall!(NativeReadlinkSyscall, "readlink", "syscall_stub_readlink", 3);
// readlinkat(dfd, path, buf, bufsiz) → long
stub_syscall!(NativeReadlinkatSyscall, "readlinkat", "syscall_stub_readlinkat", 4);
// faccessat(dfd, filename, mode) → long
stub_syscall!(NativeFaccessatSyscall, "faccessat", "syscall_stub_faccessat", 3);

/// Upper bound on the NUL-terminated path we will read from memory.
/// Matches `procedures/fileops.rs::MAX_FOPEN_PATH_LEN` (256 bytes).
const MAX_PATH_LEN: u64 = 256;

/// `AT_FDCWD` in unsigned 32-bit form (-100 reinterpreted). Linux's
/// `openat(2)` treats this as "use the current working directory" for
/// relative paths. `procedures/linux_kernel/openat.py` also matches
/// against this exact unsigned value.
const AT_FDCWD_UNSIGNED: u64 = 4_294_967_196;

/// `-1` (as `u64`) — kernel ABI failure return for `open` / `openat` /
/// `close` mirroring `procedures/posix/open.py::run` (`return -1`).
/// The dispatcher truncates to `arch().bits()` when writing the return
/// register.
const NEG_ONE: u64 = u64::MAX;

/// Read a NUL-terminated path from memory at `addr`, up to
/// `MAX_PATH_LEN`. Returns `SymbolicArgument` on the first symbolic
/// byte (the syscall then falls back to Python). Errors out with
/// `Other` if no NUL is seen within the limit.
fn read_path(state: &RustSimState, addr: u64, label: &str) -> Result<String, SyscallError> {
    let mut bytes: Vec<u8> = Vec::new();
    for i in 0..MAX_PATH_LEN {
        let bv = state.memory_load(addr.wrapping_add(i), 1)?;
        let v = bv
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument(format!("{label} path byte")))?;
        if v == 0 {
            return Ok(String::from_utf8_lossy(&bytes).to_string());
        }
        bytes.push(v as u8);
    }
    Err(SyscallError::Other(format!(
        "{label} path not NUL-terminated within {MAX_PATH_LEN} bytes"
    )))
}

/// `open(pathname, flags, mode) → fd` — allocate a fresh fd in the
/// Rust `FileSystem` keyed by `pathname`. Mirrors
/// `procedures/fileops::NativeOpen`; the only divergence from
/// `procedures/posix/open.py` is that we do not consult Python's
/// `state.fs` map (which is not mirrored into Rust state), so we never
/// return `-1` for "file doesn't exist". Empty path → `-1` (same as
/// the Python proc).
pub struct NativeOpenSyscall;

impl NativeSyscall for NativeOpenSyscall {
    fn name(&self) -> &'static str {
        "open"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let pathname_addr = extract_concrete_arg(&args[0], "open pathname")?;
        let flags = extract_concrete_arg(&args[1], "open flags")?;
        // args[2] = mode — irrelevant in Rust's FileSystem model.
        let _ = args.get(2);

        let path = read_path(state, pathname_addr, "open")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32));
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `openat(dirfd, pathname, flags, mode) → fd` — like `open`, plus
/// the `dirfd` arg for relative paths. We only handle absolute paths
/// and the `AT_FDCWD` sentinel (mirroring
/// `procedures/linux_kernel/openat.py`, which returns `-1` for any
/// other dirfd). The relative-path-from-dirfd case is not modeled in
/// Python either.
pub struct NativeOpenatSyscall;

impl NativeSyscall for NativeOpenatSyscall {
    fn name(&self) -> &'static str {
        "openat"
    }

    fn num_args(&self) -> usize {
        4
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let dirfd = extract_concrete_arg(&args[0], "openat dirfd")?;
        let pathname_addr = extract_concrete_arg(&args[1], "openat pathname")?;
        let flags = extract_concrete_arg(&args[2], "openat flags")?;
        // args[3] = mode — irrelevant in Rust's FileSystem model.
        let _ = args.get(3);

        let path = read_path(state, pathname_addr, "openat")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let absolute = path.starts_with('/');
        if !absolute && dirfd != AT_FDCWD_UNSIGNED {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let fd = state
            .file_system()
            .open(path, FdFlags::from_posix(flags as u32));
        Ok(SyscallOutcome::Continue { ret: fd as u64 })
    }
}

/// `close(fd) → 0 | -1` — mark `fd` closed in Rust's `FileSystem`.
/// Returns `-1` if the fd was never opened by Rust (mirrors
/// `state.posix.close` returning falsy).
pub struct NativeCloseSyscall;

impl NativeSyscall for NativeCloseSyscall {
    fn name(&self) -> &'static str {
        "close"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let fd = extract_concrete_arg(&args[0], "close fd")?;
        let ret = if state.file_system().close(fd as u32) {
            0
        } else {
            NEG_ONE
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

/// `access(pathname, mode) → 0 | -1` — return `0` if the path has been
/// registered as known (via prior `open` / `openat` or
/// `FileSystem::register_known_path`), `-1` otherwise. Mirrors
/// `procedures/linux_kernel/access.py::run`. The `mode` arg
/// (`F_OK` / `R_OK` / ...) is ignored — the Python proc also ignores it.
/// Empty path → `-1` (defensive: Python would also miss in `state.fs`).
pub struct NativeAccessSyscall;

impl NativeSyscall for NativeAccessSyscall {
    fn name(&self) -> &'static str {
        "access"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        let pathname_addr = extract_concrete_arg(&args[0], "access pathname")?;
        // args[1] = mode — Python proc ignores it; so do we.
        let _ = args.get(1);

        let path = read_path(state, pathname_addr, "access")?;
        if path.is_empty() {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }
        let ret = if state.file_system_ref().is_path_known(&path) {
            0
        } else {
            NEG_ONE
        };
        Ok(SyscallOutcome::Continue { ret })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};
    use crate::syscalls::{NativeSyscall, SyscallOutcome};

    /// Map an RW page at 0x2000 and stage a NUL-terminated path byte-by-byte.
    fn stage_path(state: &mut RustSimState, addr: u64, path: &[u8]) {
        state.map_memory(addr & !0xfff, 0x1000, Permission::RWX);
        for (i, b) in path.iter().enumerate() {
            state
                .memory_store(addr + i as u64, RustBV::concrete(*b as u128, 8))
                .expect("store path byte");
        }
        state
            .memory_store(addr + path.len() as u64, RustBV::concrete(0, 8))
            .expect("store NUL");
    }

    /// Sweep all five stub handlers across every supported arch and
    /// verify they return a fresh `RustBV::symbolic` of width
    /// `arch().bits()`. Successive invocations must yield distinct
    /// fresh symbols (different `RustBV::Symbolic.id`), matching
    /// `syscall_stub.py::syscall` semantics where each call gets a
    /// new `Unconstrained` BV.
    #[test]
    fn stub_handlers_return_fresh_symbolic_on_all_arches() {
        // (handler, expected name, arity)
        let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
            (&NativeLstatSyscall, "lstat", 2),
            (&NativeNewfstatatSyscall, "newfstatat", 4),
            (&NativeReadlinkSyscall, "readlink", 3),
            (&NativeReadlinkatSyscall, "readlinkat", 4),
            (&NativeFaccessatSyscall, "faccessat", 3),
        ];

        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();

            for &(handler, label, nargs) in cases {
                assert_eq!(handler.name(), label);
                assert_eq!(handler.num_args(), nargs, "{label} arity");

                let args: Vec<RustBV> =
                    (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
                let outcome = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label} errored: {e:?}"));
                let ret = match outcome {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} expected ContinueSymbolic, got {other:?}"
                    ),
                };
                assert_eq!(ret.width(), bits, "{arch} {label} width");
                assert!(!ret.is_concrete(), "{arch} {label} should be symbolic");

                // Second call: must produce a *distinct* fresh symbol.
                let outcome2 = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label} 2nd call errored: {e:?}"));
                let ret2 = match outcome2 {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} 2nd: expected ContinueSymbolic, got {other:?}"
                    ),
                };
                let (id1, id2) = match (&ret, &ret2) {
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
                    _ => panic!("{arch} {label} returns must be Symbolic"),
                };
                assert_ne!(
                    id1, id2,
                    "{arch} {label} successive calls must yield distinct fresh symbols",
                );
            }
        }
    }

    #[test]
    fn open_allocates_fresh_fd_and_records_name() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"/tmp/example.txt");

        let outcome = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64), // O_RDONLY
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("open ok");
        let fd = match outcome {
            SyscallOutcome::Continue { ret } => ret,
            other => panic!("expected Continue, got {other:?}"),
        };
        assert_eq!(fd, 3, "first allocated fd should be 3");
        assert!(state.file_system_ref().is_open(fd as u32));
        let (name, _, _, _, _) = state.file_system_ref().fd_info(fd as u32).unwrap();
        assert_eq!(name, "/tmp/example.txt");
    }

    #[test]
    fn open_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Just stage a NUL at addr 0x2000.
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store nul");

        let out = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn open_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(
                    msg.contains("open"),
                    "expected message to mention 'open', got {msg:?}",
                );
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn open_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeOpenSyscall
            .call(
                &mut state,
                &[sym_ptr, RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("pathname"), "got {msg:?}");
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn openat_absolute_path_allocates_fd_ignoring_dirfd() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"/etc/hosts");

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(99, 64), // arbitrary dirfd — ignored for absolute paths
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        match outcome {
            SyscallOutcome::Continue { ret } => {
                assert_eq!(ret, 3);
                assert!(state.file_system_ref().is_open(ret as u32));
            }
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn openat_relative_path_with_at_fdcwd_allocates_fd() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"flag.txt");

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(AT_FDCWD_UNSIGNED as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn openat_relative_path_without_at_fdcwd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"flag.txt");
        let prev_next_fd = state.file_system_ref().next_fd();

        let outcome = NativeOpenatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(7, 64), // arbitrary dirfd ≠ AT_FDCWD
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("openat ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            other => panic!("expected Continue, got {other:?}"),
        }
        // No fd should have been allocated.
        assert_eq!(state.file_system_ref().next_fd(), prev_next_fd);
    }

    #[test]
    fn close_open_fd_returns_zero_and_marks_closed() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Allocate a fresh fd via the file system directly.
        let fd = state
            .file_system()
            .open("/tmp/x".into(), FdFlags::ReadOnly);
        assert!(state.file_system_ref().is_open(fd));

        let outcome = NativeCloseSyscall
            .call(&mut state, &[RustBV::concrete(fd as u128, 64)])
            .unwrap();
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            other => panic!("expected Continue, got {other:?}"),
        }
        assert!(!state.file_system_ref().is_open(fd));
    }

    #[test]
    fn close_unknown_fd_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        // fd=99 was never allocated.
        let outcome = NativeCloseSyscall
            .call(&mut state, &[RustBV::concrete(99, 64)])
            .unwrap();
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn close_symbolic_fd_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let ctx = SymContext::new();
        let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
        let err = NativeCloseSyscall
            .call(&mut state, &[sym_fd])
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("fd"), "got {msg:?}");
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn access_unknown_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"/no/such/file");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn access_after_open_returns_zero() {
        let mut state = RustSimState::new("amd64").expect("state");
        stage_path(&mut state, 0x2000, b"/tmp/exists.txt");

        // Open registers the path.
        NativeOpenSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("open ok");

        // Re-stage path at a different addr to prove access reads it
        // fresh (not relying on caller-side cached state).
        stage_path(&mut state, 0x3000, b"/tmp/exists.txt");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x3000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn access_registered_path_returns_zero() {
        let mut state = RustSimState::new("amd64").expect("state");
        // Mirror the Python state.fs.insert path: register without
        // allocating an fd.
        state
            .file_system()
            .register_known_path("/etc/passwd".to_string());
        stage_path(&mut state, 0x2000, b"/etc/passwd");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn access_empty_path_returns_minus_one() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
            .memory_store(0x2000, RustBV::concrete(0, 8))
            .expect("store NUL");

        let out = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect("access ok");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn access_symbolic_path_byte_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        let sym_byte = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "first_path_byte", 8)
        };
        state.memory_store(0x2000, sym_byte).unwrap();

        let err = NativeAccessSyscall
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("access"), "got {msg:?}");
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn access_symbolic_pathname_addr_falls_back() {
        let mut state = RustSimState::new("amd64").expect("state");
        let sym_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "pathname_ptr", 64)
        };
        let err = NativeAccessSyscall
            .call(&mut state, &[sym_ptr, RustBV::concrete(0, 64)])
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => {
                assert!(msg.contains("pathname"), "got {msg:?}");
            }
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn access_round_trip_sweeps_supported_arches() {
        // AArch64 has no legacy `access` syscall (asm-generic only ships
        // faccessat), but the handler itself is arch-agnostic — exercise
        // it from each `RustSimState::new(...)` arch to confirm the
        // path-read + lookup path is independent of pointer width.
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/access-rt");

            // Unknown → -1.
            let out = NativeAccessSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("access");
            match out {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, NEG_ONE, "{arch} pre-open")
                }
                other => panic!("{arch} pre-open: got {other:?}"),
            }

            // Register, then known → 0.
            state
                .file_system()
                .register_known_path("/tmp/access-rt".to_string());
            let out2 = NativeAccessSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("access");
            match out2 {
                SyscallOutcome::Continue { ret } => {
                    assert_eq!(ret, 0, "{arch} post-register")
                }
                other => panic!("{arch} post-register: got {other:?}"),
            }
        }
    }

    #[test]
    fn open_close_round_trip_sweeps_supported_arches() {
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            stage_path(&mut state, 0x2000, b"/tmp/roundtrip");

            let fd_out = NativeOpenSyscall
                .call(
                    &mut state,
                    &[
                        RustBV::concrete(0x2000, bits),
                        RustBV::concrete(0, bits),
                        RustBV::concrete(0, bits),
                    ],
                )
                .expect("open");
            let fd = match fd_out {
                SyscallOutcome::Continue { ret } => ret,
                other => panic!("{arch} open: expected Continue, got {other:?}"),
            };
            assert!(state.file_system_ref().is_open(fd as u32), "{arch}");

            let close_out = NativeCloseSyscall
                .call(&mut state, &[RustBV::concrete(fd as u128, bits)])
                .expect("close");
            match close_out {
                SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "{arch}"),
                other => panic!("{arch} close: expected Continue, got {other:?}"),
            }
            assert!(!state.file_system_ref().is_open(fd as u32), "{arch}");
        }
    }
}
