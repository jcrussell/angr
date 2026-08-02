//! Native syscall handlers.
//!
//! Mirrors `procedures/` in spirit: a registry of handlers keyed by
//! `(arch_name, syscall_num)` that can short-circuit the Python
//! `_handle_syscall_callback` round-trip for ubiquitous syscalls
//! (currently exit / exit_group on amd64).
//!
//! When `dispatch` returns `Some(SyscallOutcome)`, the caller in
//! `exploration::stepping` skips creating a `PendingCallback` and
//! either continues execution (writing the return value to the
//! return register) or routes the state to `STASH_DEADENDED`.
//!
//! Architecture note (amd64 syscall ABI):
//! * args: rdi, rsi, rdx, r10, r8, r9
//! * return: rax
//! * The dispatcher uses `extract_syscall_args` (true syscall ABI), so
//!   handlers see `r10` at index 3, not `rcx`. This matters for any
//!   handler with ≥4 args (e.g. rt_sigaction).

pub(crate) mod arch_prctl;
pub(crate) mod brk;
pub(crate) mod cgc;
pub(crate) mod concurrency;
pub(crate) mod directory;
pub(crate) mod exit;
pub(crate) mod fd_io;
pub(crate) mod file_descriptor;
pub(crate) mod file_path;
pub(crate) mod identity;
pub(crate) mod memory_extras;
pub(crate) mod mmap;
pub(crate) mod mprotect;
pub(crate) mod munmap;
pub(crate) mod page;
pub(crate) mod read;
pub(crate) mod rlimit;
pub(crate) mod sigaction;
pub(crate) mod signals;
pub(crate) mod sim_time;
pub(crate) mod startup;
pub(crate) mod write;

use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::MemoryError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Shared cap (bytes) on the concrete byte count a native IO syscall handler
/// will service before falling back to Python. `read`/`write` treat it as the
/// whole-request limit, as do `cgc` (transmit/receive/random) and `startup`
/// (getrandom); `fd_io` (readv/writev) applies it per segment. Kept as a single
/// source of truth so the handlers can't desync (angr-myzjx.18, angr-9ke6b.157)
/// — importers alias it locally (`MAX_IO_SIZE as MAX_READ_SIZE`) so each call
/// site still reads in its own vocabulary.
pub(crate) const MAX_IO_SIZE: u64 = 4096;

/// Failure during native syscall dispatch.
///
/// Returning `Err` falls back to the Python `_handle_syscall_callback`
/// path so semantics remain identical to angr's existing behavior.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants; intra-crate matches must include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum SyscallError {
    #[error("symbolic argument: {0}")]
    SymbolicArgument(String),
    /// Memory operation failed. Carries the structured `MemoryError`
    /// so callers can pattern-match on the underlying cause
    /// (unmapped page, permission violation, symbolic address, ...).
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    #[error("{0}")]
    Other(String),
}

/// Extract a concrete u64 value from a syscall argument, or return
/// `SymbolicArgument(name)`. Mirrors `procedures::extract_concrete_arg`
/// to centralize the symbolic-fallthrough error context across handlers.
pub(crate) fn extract_concrete_arg(arg: &RustBV, name: &str) -> Result<u64, SyscallError> {
    arg.as_u64()
        .ok_or_else(|| SyscallError::SymbolicArgument(name.to_string()))
}

/// Mint a fresh symbolic bitvector with a per-invocation unique name.
///
/// Native syscall handlers that return (or store) symbolic values must not
/// reuse a fixed name. `RustBV::symbolic` calls
/// `z3::ast::BV::new_const(name, sort)`, and two `new_const` calls with the
/// same name+sort alias to the **same** Z3 constant. That would make two
/// invocations of e.g. `time()` solver-equal (`ret1 == ret2` unsatisfiable),
/// unlike claripy's unique-suffixed `BVS`. Appending
/// `procedures::symbol_counter(prefix)` gives each mint a distinct name and
/// therefore a distinct Z3 term — mirroring the fgets/scanf/rand procedures.
pub(crate) fn fresh_symbolic(
    ctx: &crate::symbolic::SymContext,
    prefix: &'static str,
    width: u32,
) -> RustBV {
    let id = crate::procedures::symbol_counter(prefix);
    RustBV::symbolic(ctx, format!("{prefix}_{id}"), width)
}

/// Allocate the `<prefix>_<id>_<i>` names for a batch of `count` fresh
/// symbolic bytes, bumping `counter` once for the whole batch.
///
/// Split out from [`mint_symbolic_bytes`] because the CGC `receive` path needs
/// the names *before* the bytes exist: it hands them to
/// `procedures::stdin_common::mint_stdin_bytes`, which binds each leaf to a
/// harness-seeded fd-0 byte instead of minting a plain constant. Every caller
/// shares this one naming scheme so per-handler drift (see `fresh_symbolic`
/// for why the uniquifying `id` is load-bearing) is impossible.
pub(crate) fn fresh_byte_names(
    prefix: &str,
    counter: &std::sync::atomic::AtomicU64,
    count: u64,
) -> Vec<String> {
    let id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (0..count).map(|i| format!("{prefix}_{id}_{i}")).collect()
}

/// Mint `count` fresh symbolic bytes named `<prefix>_<id>_<i>` and store them
/// one 8-bit store per byte into `[base, base + count)`.
///
/// The single implementation of the "fill this buffer with fresh symbolic
/// bytes" pattern shared by `read`, `readv`, `getrandom` and CGC `random`
/// (angr-9ke6b.158). `counter` is the caller's per-handler `AtomicU64`, which
/// keeps repeat invocations of the *same* callsite uniquely named; `prefix`
/// disambiguates callsites (and, for the fd-keyed readers, streams).
pub(crate) fn mint_symbolic_bytes(
    state: &mut RustSimState,
    base: u64,
    count: u64,
    prefix: &str,
    counter: &std::sync::atomic::AtomicU64,
) -> Result<(), SyscallError> {
    let names = fresh_byte_names(prefix, counter, count);
    let sym_bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        names
            .iter()
            .map(|name| RustBV::symbolic(&ctx, name, 8))
            .collect()
    };
    crate::procedures::strings::write_bv_bytes(state, base, sym_bytes)?;
    Ok(())
}

/// What the dispatcher should do after a syscall handler runs.
#[derive(Debug)]
pub(crate) enum SyscallOutcome {
    /// State should continue at PC. `ret` is written to the return
    /// register (rax on amd64).
    Continue { ret: u64 },
    /// State should continue at PC with a symbolic return value. The BV
    /// is written directly to the return register (rax on amd64). Used by
    /// syscalls like time(2) that return a fresh symbolic value.
    ContinueSymbolic { ret: RustBV },
    /// State should be deadended (used by exit / exit_group).
    Exit,
}

pub(crate) trait NativeSyscall: Send + Sync {
    /// Human-readable handler label (e.g. `"read"`, `"getuid"`).
    ///
    /// The dispatcher (`NativeSyscallRegistry::get`) keys on `(arch, num)`, so
    /// this label is purely diagnostic: `handle_syscall_core` logs it on both
    /// Python-fallback paths (arg extraction failed / handler declined), which
    /// are otherwise indistinguishable from "no native handler registered" in
    /// a debug log. Together with the `stub_syscall!` / `constant_syscall!`
    /// `$label` argument it is also the only in-tree mapping from handler type
    /// to syscall name (angr-9ke6b.218 item 6).
    fn name(&self) -> &'static str;
    fn num_args(&self) -> usize;
    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError>;
}

/// Registry of native syscall handlers, keyed by arch name then syscall num.
///
/// Two-level map (`arch -> num -> handler`) so `get` resolves a runtime
/// `&str` arch against the `&'static str` outer keys via `Borrow<str>` and
/// then does a second hash lookup on the number — both O(1). A flat
/// `(&'static str, u64)` key cannot be looked up from a runtime `&str`
/// without interning, which previously forced a linear `iter().find()` over
/// all 432 rows on every executed syscall (see angr-k95c).
pub(crate) struct NativeSyscallRegistry {
    handlers: HashMap<&'static str, HashMap<u64, Arc<dyn NativeSyscall>>>,
}

/// Register a batch of `(syscall_number, handler)` rows for one arch.
///
/// Wraps each handler in `Arc::new(...)` and inserts into the registry.
/// Each row is `(num, expr)` where `expr` constructs a `NativeSyscall`
/// (e.g. `read::NativeReadSyscall` or `exit::NativeExitSyscall`).
///
/// Per-arch syscall numbers diverge across Linux ABIs, so each arch keeps
/// its own table — the macro collapses the boilerplate (one `r.register(...)`
/// + `Arc::new(...)` line) but does not merge tables.
macro_rules! register_syscalls {
    ($r:expr, $arch:literal, [ $( ($num:expr, $handler:expr) ),* $(,)? ]) => {
        $(
            $r.register($arch, $num, Arc::new($handler));
        )*
    };
}

/// Declare a stub syscall handler with arbitrary arity.
///
/// Generates a unit struct implementing `NativeSyscall` whose `call`
/// returns a fresh `RustBV::symbolic` of width `arch().bits()` (matching
/// Python's `procedures/stubs/syscall_stub.py::syscall` for syscalls with
/// no dedicated `SimProcedure`). Args are intentionally ignored.
///
/// Re-exported `pub(crate)` so individual syscall modules can `use
/// super::stub_syscall;` instead of redeclaring the same boilerplate.
macro_rules! stub_syscall {
    ($ty:ident, $label:expr, $sym_name:expr, $nargs:expr) => {
        pub(crate) struct $ty;

        impl $crate::syscalls::NativeSyscall for $ty {
            fn name(&self) -> &'static str {
                $label
            }

            fn num_args(&self) -> usize {
                $nargs
            }

            fn call(
                &self,
                state: &mut $crate::state::RustSimState,
                _args: &[$crate::symbolic::RustBV],
            ) -> Result<$crate::syscalls::SyscallOutcome, $crate::syscalls::SyscallError> {
                let bits = state.arch().bits();
                let ret = {
                    let ctx = state.solver().borrow();
                    $crate::syscalls::fresh_symbolic(&ctx, $sym_name, bits)
                };
                Ok($crate::syscalls::SyscallOutcome::ContinueSymbolic { ret })
            }
        }
    };
}
pub(crate) use stub_syscall;

impl Default for NativeSyscallRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeSyscallRegistry {
    pub(crate) fn new() -> Self {
        let mut r = NativeSyscallRegistry {
            handlers: HashMap::new(),
        };

        // ===== amd64 (asm/unistd_64.h) =====
        // read (0): stdin (fd=0) only; symbolic bytes mirror NativeRead.
        // write (1): stdout (fd=1) and stderr (fd=2); concrete bytes only.
        // lseek (8), readv (19), writev (20): FD I/O (angr-6ylm). lseek
        //   mirrors NativeLseek; readv/writev walk a struct iovec[] and
        //   reuse the read/write logic (writev is glibc stdio's flush
        //   path). See `syscalls/fd_io.rs`.
        // pread64 (17), pwrite64 (18): positioned I/O (angr-dbb1) that does
        //   not move the fd position. pread64 serves concrete content via
        //   FileSystem::read_at; pwrite64 overwrites at the offset via
        //   FileSystem::write_at (offset-honoring, unlike append-only write).
        //   Symbolic offset / data fall back to Python.
        // uname (63), set_tid_address (218), set_robust_list (273),
        //   getrandom (318): libc/loader startup stubs (angr-6ylm). See
        //   `syscalls/startup.rs`; uname/set_tid_address mirror the
        //   linux_kernel procs, getrandom fills buf with symbolic bytes.
        // exit (60), exit_group (231): deadend.
        // mprotect (10): set page perms; -1 on misalign / unmapped.
        // brk (12): grow/query the program break.
        // munmap (11): Python implementation is a no-op return 0.
        // mmap (9): anonymous concrete-args fast path; falls back to Python
        //   for symbolic / file-backed / collision cases.
        // rt_sigaction (13): Python is essentially a no-op return 0 (with
        //   -EINVAL for signum 33).
        // gettimeofday (96): writes fresh symbolic timeval; -1 on null.
        // arch_prctl (158): fs_const/gs_const set/get (amd64-only).
        // time (201): fresh symbolic time_t in rax; monotonic via
        //   state.last_time; stores at *pointer if non-null.
        // clock_gettime (228): CLOCK_REALTIME-only; -1 on null; other clocks
        //   fall back to Python's SimProcedureError path.
        // getpid (39), getppid (110), gettid (186): identity getters
        //   returning angr's posix defaults (pid=1337, ppid=1336).
        // getuid (102), geteuid (107), getgid (104), getegid (108): return
        //   1000 (matches angr Python proc default).
        // setuid (105), setgid (106): return fresh symbolic BV (matches
        //   Python `syscall_stub.py::syscall` ReturnUnconstrained fallback —
        //   angr has no dedicated SimProcedure for these).
        // mremap (25), msync (26), madvise (28), mlock (149), munlock (150),
        //   mlockall (151), munlockall (152): same stub-symbolic semantics —
        //   no Python SimProcedure, falls through to syscall_stub.
        // rt_sigreturn (15), pause (34), alarm (37), kill (62), tgkill (234):
        //   signals + process control (angr-0hif.6). kill/rt_sigreturn/pause/
        //   alarm have no Python SimProcedure and mirror syscall_stub;
        //   tgkill returns concrete 0 (matches procedures/linux_kernel/
        //   tgkill.py exactly). rt_sigprocmask (14) is intentionally NOT
        //   registered — its Python impl mutates state.posix.sigmask which
        //   RustSimState does not carry; falls back to Python for parity.
        // getrlimit (97), setrlimit (160), prlimit64 (302): resource limits
        //   (angr-0hif.7). getrlimit mirrors procedures/linux_kernel/
        //   getrlimit.py — RLIMIT_STACK writes 8388608 + symbolic rlim_max,
        //   other resources return fresh symbolic. setrlimit / prlimit64
        //   have no Python SimProcedure and mirror syscall_stub.
        // futex (202): mirrors procedures/linux_kernel/futex.py — op&1
        //   (FUTEX_WAKE family) returns 0, else fresh symbolic.
        // eventfd (284), eventfd2 (290), epoll_create (213), epoll_create1
        //   (291), epoll_ctl (233), epoll_wait (232): concurrency primitives
        //   with no Python SimProcedure; native handlers mirror syscall_stub.
        //   angr is single-threaded symex so blocking is never modeled.
        //   epoll_ctl_old (214) / epoll_wait_old (215) intentionally NOT
        //   registered — pre-2.6 legacy that no current binary uses.
        // readlink (89), readlinkat (267): consult the FileSystem symlink
        //   table (angr-wv38, angr-11djq.6.2). A path registered via
        //   FileSystem::add_symlink writes min(target_len, bufsiz) target
        //   bytes (no NUL) and returns that count; anything else returns
        //   -1 with the buffer untouched. The table is empty by default,
        //   so an unconfigured state sees -1 for every path. readlinkat
        //   reads the dirfd and applies the openat policy. See
        //   file_path.rs module doc.
        // faccessat (269): native FileSystem::is_path_known query
        //   (angr-6009). Clone of NativeAccessSyscall with dirfd
        //   handling. Absolute paths and AT_FDCWD return 0/-1 from
        //   the FileSystem; relative paths with non-AT_FDCWD dirfd
        //   return -1 (we do not model directory fds — matches
        //   NativeOpenatSyscall's policy).
        // lstat (6), newfstatat (262): stat()-shaped clones promoted
        //   from stubs in angr-poao. See file_path.rs module doc.
        // open (2), openat (257), close (3): FD-allocating syscalls
        //   (angr-k3ol.1). Mutate `RustSimState::file_system()` directly,
        //   mirroring the existing `procedures/fileops::NativeOpen` /
        //   `NativeClose` libc procs. Python `state.posix.fd` / `state.fs`
        //   are NOT mirrored — same trade-off as the libc procs and
        //   `dup`/`dup2`.
        // access (21): file-existence check (angr-k3ol.2). Queries the
        //   `FileSystem::known_paths` set populated by `open` /
        //   `openat` / `open_with_content`. Pre-populated Python
        //   `state.fs._files` entries are NOT mirrored — same trade-off
        //   as the FD-allocating handlers.
        // fstat (5): native struct-stat write (angr-k3ol.3). Reads
        //   content_len from FileSystem::effective_size(fd) and writes a
        //   per-arch `struct stat` (AMD64/ARM64 use the 64-bit struct;
        //   i386/ARM/MIPS32 use the LFS struct stat64 via the `*64` numbers,
        //   mirroring `fstat64.py`). st_mode is the concrete S_IFREG|0o755
        //   instead of a fresh symbolic BVS (Python's `fstat_with_result`
        //   mints one) — except MIPS32, whose `_store_mips32` omits st_mode.
        // stat (4): native path-keyed struct-stat write (angr-k3ol.4).
        //   Resolves pathname via read_path, returns -1 for empty /
        //   unknown paths, otherwise reuses `write_amd64_stat` with
        //   `FileSystem::content_size_for_path(path)` as the size
        //   field. AMD64 only — ARM64's asm-generic ABI dropped legacy
        //   `stat` (only newfstatat 79, already a stub). Diverges from
        //   `procedures/linux_kernel/stat.py`'s open→fstat→close in
        //   that the Rust path never mutates the fd table.
        // fcntl (72), ioctl (16), pipe (22), pipe2 (293): FD-control
        //   syscalls that fall through to syscall_stub (angr-0hif.5
        //   stub-fallthrough subset). posix/fcntl.py defines a fcntl
        //   SimProcedure but linux_kernel.py does NOT bind it into the
        //   kernel library — so the syscall path is pure stub.
        // dup (32), dup2 (33), dup3 (292): FD-table syscalls
        //   (angr-vp19). Mutate `RustSimState::file_system()` directly,
        //   matching the precedent set by `procedures/fileops::
        //   NativeDup` / `NativeDup2`. Python `state.posix.fd` is NOT
        //   kept in sync — same as the libc procs — so any callback
        //   reading `state.posix.fd` after a Rust dup will not see the
        //   newly allocated slot. See `syscalls/file_descriptor.rs`
        //   doc comment for the lowest-free-vs-monotonic divergence.
        // chdir (80), fchdir (81), getcwd (79), mkdir (83), mkdirat (258),
        //   rmdir (84), unlink (87), unlinkat (263), rename (82),
        //   renameat (264), renameat2 (316): directory syscalls
        //   (angr-0hif.2). chdir / getcwd back the per-state
        //   `FileSystem::cwd` field added for this subset; the rest
        //   mirror `procedures/stubs/syscall_stub.py::syscall` — no
        //   Python `SimProcedure` exists for fchdir/mkdir/rmdir/rename
        //   etc., and unlink's Python proc needs the path → SimFile
        //   plumbing not yet in `RustSimState` (same trade-off as
        //   angr-k3ol for open/close). See `syscalls/directory.rs` doc
        //   comment for the rationale on the stub subset.
        register_syscalls!(
            r,
            "AMD64",
            [
                (0, read::NativeReadSyscall),
                (1, write::NativeWriteSyscall),
                (2, file_path::NativeOpenSyscall),
                (3, file_path::NativeCloseSyscall),
                (4, file_path::NativeStatSyscall),
                (5, file_path::NativeFstatSyscall),
                (6, file_path::NativeLstatSyscall),
                (9, mmap::NativeMmapSyscall),
                (10, mprotect::NativeMprotectSyscall),
                (11, munmap::NativeMunmapSyscall),
                (12, brk::NativeBrkSyscall),
                (13, sigaction::NativeRtSigactionSyscall),
                (15, signals::NativeRtSigreturnSyscall),
                (8, fd_io::NativeLseekSyscall),
                (16, file_descriptor::NativeIoctlSyscall),
                (17, fd_io::NativePread64Syscall),
                (18, fd_io::NativePwrite64Syscall),
                (19, fd_io::NativeReadvSyscall),
                (20, fd_io::NativeWritevSyscall),
                (21, file_path::NativeAccessSyscall),
                (22, file_descriptor::NativePipeSyscall),
                (25, memory_extras::NativeMremapSyscall),
                (26, memory_extras::NativeMsyncSyscall),
                (28, memory_extras::NativeMadviseSyscall),
                (32, file_descriptor::NativeDupSyscall),
                (33, file_descriptor::NativeDup2Syscall),
                (34, signals::NativePauseSyscall),
                (37, signals::NativeAlarmSyscall),
                (39, identity::NativeGetpidSyscall),
                (60, exit::NativeExitSyscall),
                (62, signals::NativeKillSyscall),
                (63, startup::NativeUnameSyscall),
                (72, file_descriptor::NativeFcntlSyscall),
                (79, directory::NativeGetcwdSyscall),
                (80, directory::NativeChdirSyscall),
                (81, directory::NativeFchdirSyscall),
                (82, directory::NativeRenameSyscall),
                (83, directory::NativeMkdirSyscall),
                (84, directory::NativeRmdirSyscall),
                (87, directory::NativeUnlinkSyscall),
                (89, file_path::NativeReadlinkSyscall),
                (96, sim_time::NativeGettimeofdaySyscall),
                (97, rlimit::NativeGetrlimitSyscall),
                (102, identity::NativeGetuidSyscall),
                (104, identity::NativeGetgidSyscall),
                (105, identity::NativeSetuidSyscall),
                (106, identity::NativeSetgidSyscall),
                (107, identity::NativeGeteuidSyscall),
                (108, identity::NativeGetegidSyscall),
                (110, identity::NativeGetppidSyscall),
                (149, memory_extras::NativeMlockSyscall),
                (150, memory_extras::NativeMunlockSyscall),
                (151, memory_extras::NativeMlockallSyscall),
                (152, memory_extras::NativeMunlockallSyscall),
                (158, arch_prctl::NativeArchPrctlSyscall),
                (160, rlimit::NativeSetrlimitSyscall),
                (186, identity::NativeGettidSyscall),
                (201, sim_time::NativeTimeSyscall),
                (202, concurrency::NativeFutexSyscall),
                (213, concurrency::NativeEpollCreateSyscall),
                (218, startup::NativeSetTidAddressSyscall),
                (228, sim_time::NativeClockGettimeSyscall),
                (231, exit::NativeExitSyscall),
                (232, concurrency::NativeEpollWaitSyscall),
                (233, concurrency::NativeEpollCtlSyscall),
                (234, signals::NativeTgkillSyscall),
                (257, file_path::NativeOpenatSyscall),
                (258, directory::NativeMkdiratSyscall),
                (262, file_path::NativeNewfstatatSyscall),
                (263, directory::NativeUnlinkatSyscall),
                (264, directory::NativeRenameatSyscall),
                (267, file_path::NativeReadlinkatSyscall),
                (269, file_path::NativeFaccessatSyscall),
                (273, startup::NativeSetRobustListSyscall),
                (281, concurrency::NativeEpollPwaitSyscall),
                (284, concurrency::NativeEventfdSyscall),
                (290, concurrency::NativeEventfd2Syscall),
                (291, concurrency::NativeEpollCreate1Syscall),
                (292, file_descriptor::NativeDup3Syscall),
                (293, file_descriptor::NativePipe2Syscall),
                (302, rlimit::NativePrlimit64Syscall),
                (316, directory::NativeRenameat2Syscall),
                (318, startup::NativeGetrandomSyscall),
            ]
        );

        // ===== Per-arch registrations (angr-7xms) =====
        //
        // Handlers themselves are arch-agnostic (they take RustBV args from
        // the dispatcher, which uses each arch's CallingConvention to extract
        // the right registers). The only per-arch knob is the syscall number,
        // so the tables below map numbers from <asm/unistd_*.h> for each
        // Linux ABI to the same set of handlers.
        //
        // Notes on what is *not* registered here:
        //  - arch_prctl is amd64-only (no equivalent on other Linux arches).
        //  - On x86/ARM, EAX/R0 also hold the return value, so the
        //    Cdecl/ARMEABI return_register matches the kernel ABI.
        //
        // i386/ARM mmap family (angr-6gmc): the legacy struct-arg form
        // `old_mmap` (90 on both) and `mmap2` (192 on both) are now
        // registered. `old_mmap` reads six 32-bit fields from a
        // `struct mmap_arg_struct *` and dispatches to `do_mmap`;
        // `mmap2` takes 6 register args with offset in page units and
        // scales by PAGE_SIZE. MIPS32 O32 registers both `old_mmap`
        // (4090) and `mmap2` (4210): O32 passes args 5-6 on the stack at
        // [sp+16], which `extract_syscall_args` now traverses for concrete
        // SP (angr-tvod).

        // Linux i386 (asm/unistd_32.h).
        //
        // Identity getters: i386 has both the legacy 16-bit-uid_t variants
        // (numbers 20/24/47/49/50/64) and the LFS 32-bit-uid_t variants
        // (199-202). Both alias to the same handler since angr returns the
        // same constant 1000 for both forms.
        register_syscalls!(
            r,
            "X86",
            [
                (1, exit::NativeExitSyscall),
                (3, read::NativeReadSyscall),
                (4, write::NativeWriteSyscall),
                (5, file_path::NativeOpenSyscall),
                (6, file_path::NativeCloseSyscall),
                (10, directory::NativeUnlinkSyscall),
                (12, directory::NativeChdirSyscall),
                (13, sim_time::NativeTimeSyscall),
                (20, identity::NativeGetpidSyscall),
                (23, identity::NativeSetuidSyscall),
                (24, identity::NativeGetuidSyscall),
                (27, signals::NativeAlarmSyscall),
                (29, signals::NativePauseSyscall),
                (33, file_path::NativeAccessSyscall),
                (37, signals::NativeKillSyscall),
                (38, directory::NativeRenameSyscall),
                (39, directory::NativeMkdirSyscall),
                (40, directory::NativeRmdirSyscall),
                (41, file_descriptor::NativeDupSyscall),
                (42, file_descriptor::NativePipeSyscall),
                (45, brk::NativeBrkSyscall),
                (46, identity::NativeSetgidSyscall),
                (47, identity::NativeGetgidSyscall),
                (49, identity::NativeGeteuidSyscall),
                (50, identity::NativeGetegidSyscall),
                (54, file_descriptor::NativeIoctlSyscall),
                (55, file_descriptor::NativeFcntlSyscall),
                (63, file_descriptor::NativeDup2Syscall),
                (64, identity::NativeGetppidSyscall),
                (75, rlimit::NativeSetrlimitSyscall),
                (76, rlimit::NativeGetrlimitSyscall),
                (78, sim_time::NativeGettimeofdaySyscall),
                (85, file_path::NativeReadlinkSyscall),
                (90, mmap::NativeOldMmapSyscall),
                (91, munmap::NativeMunmapSyscall),
                // Legacy 106/107/108 (old 32-bit struct stat) have no
                // Rust writer — modern 32-bit glibc emits the LFS `*64`
                // variants below (struct stat64, `write_i386_stat`), whose
                // field offsets differ from the pre-LFS layout. angr's i386
                // map does define all three, so they fall back to Python
                // rather than getting a wrong-offset struct (angr-11djq.5.1).
                (125, mprotect::NativeMprotectSyscall),
                (133, directory::NativeFchdirSyscall),
                (144, memory_extras::NativeMsyncSyscall),
                (150, memory_extras::NativeMlockSyscall),
                (151, memory_extras::NativeMunlockSyscall),
                (152, memory_extras::NativeMlockallSyscall),
                (153, memory_extras::NativeMunlockallSyscall),
                (163, memory_extras::NativeMremapSyscall),
                (173, signals::NativeRtSigreturnSyscall),
                (174, sigaction::NativeRtSigactionSyscall),
                (183, directory::NativeGetcwdSyscall),
                // 191 = ugetrlimit (LFS uid_t variant) aliases to getrlimit.
                (191, rlimit::NativeGetrlimitSyscall),
                (192, mmap::NativeMmap2Syscall),
                // LFS stat family — struct stat64 (`write_i386_stat`).
                (195, file_path::NativeStatSyscall),
                (196, file_path::NativeLstatSyscall),
                (197, file_path::NativeFstatSyscall),
                (199, identity::NativeGetuidSyscall),
                (200, identity::NativeGetgidSyscall),
                (201, identity::NativeGeteuidSyscall),
                (202, identity::NativeGetegidSyscall),
                (213, identity::NativeSetuidSyscall),
                (214, identity::NativeSetgidSyscall),
                (219, memory_extras::NativeMadviseSyscall),
                // 221 = fcntl64 (LFS-style 64-bit offset variant).
                (221, file_descriptor::NativeFcntl64Syscall),
                (224, identity::NativeGettidSyscall),
                (240, concurrency::NativeFutexSyscall),
                (252, exit::NativeExitSyscall),
                (254, concurrency::NativeEpollCreateSyscall),
                (255, concurrency::NativeEpollCtlSyscall),
                (256, concurrency::NativeEpollWaitSyscall),
                (265, sim_time::NativeClockGettimeSyscall),
                (270, signals::NativeTgkillSyscall),
                (295, file_path::NativeOpenatSyscall),
                (296, directory::NativeMkdiratSyscall),
                (301, directory::NativeUnlinkatSyscall),
                (302, directory::NativeRenameatSyscall),
                // newfstatat absent on i386 — Linux 32-bit uses fstatat64
                // (327 in angr's i386 map), wired to the same handler.
                (327, file_path::NativeNewfstatatSyscall),
                (305, file_path::NativeReadlinkatSyscall),
                (307, file_path::NativeFaccessatSyscall),
                (319, concurrency::NativeEpollPwaitSyscall),
                (323, concurrency::NativeEventfdSyscall),
                (328, concurrency::NativeEventfd2Syscall),
                (329, concurrency::NativeEpollCreate1Syscall),
                (330, file_descriptor::NativeDup3Syscall),
                (331, file_descriptor::NativePipe2Syscall),
                (340, rlimit::NativePrlimit64Syscall),
                (353, directory::NativeRenameat2Syscall),
                // angr-dbb1: FD I/O + libc-startup handlers shared with AMD64,
                // i386 numbers from <asm/unistd_32.h>. getrandom (355) is
                // present in angr's i386 table (absent on ARM/ARM64/MIPS).
                (19, fd_io::NativeLseekSyscall),
                (122, startup::NativeUnameSyscall),
                (145, fd_io::NativeReadvSyscall),
                (146, fd_io::NativeWritevSyscall),
                (180, fd_io::NativePread64Syscall),
                (181, fd_io::NativePwrite64Syscall),
                (258, startup::NativeSetTidAddressSyscall),
                (311, startup::NativeSetRobustListSyscall),
                (355, startup::NativeGetrandomSyscall),
            ]
        );

        // Linux ARM EABI (arm/asm/unistd-eabi.h). Identity getters share
        // numbering with i386 (both legacy 16-bit and 32-bit variants).
        // `old_mmap` (90) and `mmap2` (192) handled by the same handlers as
        // i386 — see angr-6gmc.
        register_syscalls!(
            r,
            "ARM",
            [
                (1, exit::NativeExitSyscall),
                (3, read::NativeReadSyscall),
                (4, write::NativeWriteSyscall),
                (5, file_path::NativeOpenSyscall),
                (6, file_path::NativeCloseSyscall),
                (10, directory::NativeUnlinkSyscall),
                (12, directory::NativeChdirSyscall),
                (13, sim_time::NativeTimeSyscall),
                (20, identity::NativeGetpidSyscall),
                (23, identity::NativeSetuidSyscall),
                (24, identity::NativeGetuidSyscall),
                (27, signals::NativeAlarmSyscall),
                (29, signals::NativePauseSyscall),
                (33, file_path::NativeAccessSyscall),
                (37, signals::NativeKillSyscall),
                (38, directory::NativeRenameSyscall),
                (39, directory::NativeMkdirSyscall),
                (40, directory::NativeRmdirSyscall),
                (41, file_descriptor::NativeDupSyscall),
                (42, file_descriptor::NativePipeSyscall),
                (45, brk::NativeBrkSyscall),
                (46, identity::NativeSetgidSyscall),
                (47, identity::NativeGetgidSyscall),
                (49, identity::NativeGeteuidSyscall),
                (50, identity::NativeGetegidSyscall),
                (54, file_descriptor::NativeIoctlSyscall),
                (55, file_descriptor::NativeFcntlSyscall),
                (63, file_descriptor::NativeDup2Syscall),
                (64, identity::NativeGetppidSyscall),
                (75, rlimit::NativeSetrlimitSyscall),
                (76, rlimit::NativeGetrlimitSyscall),
                (78, sim_time::NativeGettimeofdaySyscall),
                (85, file_path::NativeReadlinkSyscall),
                (90, mmap::NativeOldMmapSyscall),
                (91, munmap::NativeMunmapSyscall),
                // Legacy 106/107/108 (old 32-bit struct stat) have no Rust
                // writer — `write_arm_stat` emits the LFS `struct stat64`
                // layout used by 195/196/197, whose field offsets differ
                // (st_size at 0x30, 64-bit st_ino at 0x0C). angr's "arm" map
                // does define all three, so they fall back to Python rather
                // than getting a wrong-offset struct (angr-9ke6b.226); same
                // policy as i386 106/107/108 and MIPS32 4106/4107/4108.
                (125, mprotect::NativeMprotectSyscall),
                (133, directory::NativeFchdirSyscall),
                (144, memory_extras::NativeMsyncSyscall),
                (150, memory_extras::NativeMlockSyscall),
                (151, memory_extras::NativeMunlockSyscall),
                (152, memory_extras::NativeMlockallSyscall),
                (153, memory_extras::NativeMunlockallSyscall),
                (163, memory_extras::NativeMremapSyscall),
                (173, signals::NativeRtSigreturnSyscall),
                (174, sigaction::NativeRtSigactionSyscall),
                (183, directory::NativeGetcwdSyscall),
                // 191 = ugetrlimit (LFS uid_t variant) aliases to getrlimit.
                (191, rlimit::NativeGetrlimitSyscall),
                (192, mmap::NativeMmap2Syscall),
                // LFS stat family — struct stat64 (`write_arm_stat`). ARM
                // EABI numbers match i386 (195/196/197) per angr's "arm"
                // map in linux_kernel.py.
                (195, file_path::NativeStatSyscall),
                (196, file_path::NativeLstatSyscall),
                (197, file_path::NativeFstatSyscall),
                (199, identity::NativeGetuidSyscall),
                (200, identity::NativeGetgidSyscall),
                (201, identity::NativeGeteuidSyscall),
                (202, identity::NativeGetegidSyscall),
                (213, identity::NativeSetuidSyscall),
                (214, identity::NativeSetgidSyscall),
                (220, memory_extras::NativeMadviseSyscall),
                // 221 = fcntl64 (LFS-style 64-bit offset variant).
                (221, file_descriptor::NativeFcntl64Syscall),
                (224, identity::NativeGettidSyscall),
                (240, concurrency::NativeFutexSyscall),
                (248, exit::NativeExitSyscall),
                (250, concurrency::NativeEpollCreateSyscall),
                (251, concurrency::NativeEpollCtlSyscall),
                (252, concurrency::NativeEpollWaitSyscall),
                (263, sim_time::NativeClockGettimeSyscall),
                (268, signals::NativeTgkillSyscall),
                (322, file_path::NativeOpenatSyscall),
                (323, directory::NativeMkdiratSyscall),
                (328, directory::NativeUnlinkatSyscall),
                (329, directory::NativeRenameatSyscall),
                // newfstatat absent on ARM EABI — uses fstatat64 (327 in
                // angr's "arm" map), wired to the same handler as i386.
                // renameat2 absent in angr's ARM EABI table.
                (327, file_path::NativeNewfstatatSyscall),
                (332, file_path::NativeReadlinkatSyscall),
                (334, file_path::NativeFaccessatSyscall),
                (346, concurrency::NativeEpollPwaitSyscall),
                (351, concurrency::NativeEventfdSyscall),
                (356, concurrency::NativeEventfd2Syscall),
                (357, concurrency::NativeEpollCreate1Syscall),
                (358, file_descriptor::NativeDup3Syscall),
                (359, file_descriptor::NativePipe2Syscall),
                (369, rlimit::NativePrlimit64Syscall),
                // angr-dbb1: FD I/O + libc-startup handlers shared with AMD64,
                // ARM EABI numbers from <arm/asm/unistd-eabi.h>. getrandom is
                // absent from angr's ARM table, so it is not registered.
                (19, fd_io::NativeLseekSyscall),
                (122, startup::NativeUnameSyscall),
                (145, fd_io::NativeReadvSyscall),
                (146, fd_io::NativeWritevSyscall),
                (180, fd_io::NativePread64Syscall),
                (181, fd_io::NativePwrite64Syscall),
                (256, startup::NativeSetTidAddressSyscall),
                (338, startup::NativeSetRobustListSyscall),
            ]
        );

        // Linux AArch64 (asm-generic/unistd.h). Uses the asm-generic ABI:
        // mmap takes the modern 6-register form with byte offset, so the
        // existing NativeMmapSyscall works as-is.
        // ARM64 uses asm-generic numbering. Note: asm-generic does NOT
        // define `pause` (29 on i386/arm) or `alarm` (27 on i386/arm) —
        // glibc on aarch64 emulates them via setitimer / rt_sigtimedwait.
        // So only kill, tgkill, and rt_sigreturn from angr-0hif.6 land here.
        // ARM64 asm-generic note: epoll_create (legacy) and epoll_wait are
        //   absent; binaries use epoll_create1 (20) and epoll_pwait (22).
        //   Likewise the legacy 1-arg eventfd is absent (only eventfd2 at 19).
        register_syscalls!(
            r,
            "ARM64",
            [
                (19, concurrency::NativeEventfd2Syscall),
                (20, concurrency::NativeEpollCreate1Syscall),
                (21, concurrency::NativeEpollCtlSyscall),
                (22, concurrency::NativeEpollPwaitSyscall),
                // asm-generic ABI omits legacy `pipe` (only pipe2 at 59),
                // legacy `fcntl64` (unified fcntl at 25 since asm-generic
                // is 64-bit-oriented), and the older epoll/eventfd variants.
                // Likewise legacy `dup2` is dropped — only `dup` (23) and
                // `dup3` (24) exist in asm-generic.
                (17, directory::NativeGetcwdSyscall),
                (23, file_descriptor::NativeDupSyscall),
                (24, file_descriptor::NativeDup3Syscall),
                (25, file_descriptor::NativeFcntlSyscall),
                (29, file_descriptor::NativeIoctlSyscall),
                // asm-generic ABI dropped legacy `lstat` and `readlink` — only
                // the *at variants exist here. faccessat (48), readlinkat (78),
                // newfstatat (79). Likewise legacy `mkdir`/`rmdir`/`unlink`/
                // `rename` are absent; only the *at variants exist.
                (34, directory::NativeMkdiratSyscall),
                (35, directory::NativeUnlinkatSyscall),
                (38, directory::NativeRenameatSyscall),
                (48, file_path::NativeFaccessatSyscall),
                (49, directory::NativeChdirSyscall),
                (50, directory::NativeFchdirSyscall),
                (56, file_path::NativeOpenatSyscall),
                (57, file_path::NativeCloseSyscall),
                (59, file_descriptor::NativePipe2Syscall),
                (63, read::NativeReadSyscall),
                (64, write::NativeWriteSyscall),
                (78, file_path::NativeReadlinkatSyscall),
                (79, file_path::NativeNewfstatatSyscall),
                (80, file_path::NativeFstatSyscall),
                (93, exit::NativeExitSyscall),
                (94, exit::NativeExitSyscall),
                (98, concurrency::NativeFutexSyscall),
                (113, sim_time::NativeClockGettimeSyscall),
                (129, signals::NativeKillSyscall),
                (131, signals::NativeTgkillSyscall),
                (134, sigaction::NativeRtSigactionSyscall),
                (139, signals::NativeRtSigreturnSyscall),
                (144, identity::NativeSetgidSyscall),
                (146, identity::NativeSetuidSyscall),
                (163, rlimit::NativeGetrlimitSyscall),
                (164, rlimit::NativeSetrlimitSyscall),
                (169, sim_time::NativeGettimeofdaySyscall),
                (172, identity::NativeGetpidSyscall),
                (173, identity::NativeGetppidSyscall),
                (174, identity::NativeGetuidSyscall),
                (175, identity::NativeGeteuidSyscall),
                (176, identity::NativeGetgidSyscall),
                (177, identity::NativeGetegidSyscall),
                (178, identity::NativeGettidSyscall),
                (214, brk::NativeBrkSyscall),
                (215, munmap::NativeMunmapSyscall),
                (216, memory_extras::NativeMremapSyscall),
                (222, mmap::NativeMmapSyscall),
                (226, mprotect::NativeMprotectSyscall),
                (227, memory_extras::NativeMsyncSyscall),
                (228, memory_extras::NativeMlockSyscall),
                (229, memory_extras::NativeMunlockSyscall),
                (230, memory_extras::NativeMlockallSyscall),
                (231, memory_extras::NativeMunlockallSyscall),
                (233, memory_extras::NativeMadviseSyscall),
                (261, rlimit::NativePrlimit64Syscall),
                // angr-dbb1: FD I/O + libc-startup handlers shared with AMD64,
                // asm-generic numbers from <asm-generic/unistd.h>. getrandom
                // is absent from angr's AArch64 table, so it is not registered.
                (62, fd_io::NativeLseekSyscall),
                (65, fd_io::NativeReadvSyscall),
                (66, fd_io::NativeWritevSyscall),
                (67, fd_io::NativePread64Syscall),
                (68, fd_io::NativePwrite64Syscall),
                (96, startup::NativeSetTidAddressSyscall),
                (99, startup::NativeSetRobustListSyscall),
                (160, startup::NativeUnameSyscall),
            ]
        );

        // Linux MIPS32 O32 (asm/unistd_o32.h). Numbers start at 4000.
        // `old_mmap` (4090) registered — one struct-pointer arg, fits in
        // $a0 (angr-6gmc). `mmap2` (4210) is a 6-arg syscall; O32 passes
        // args 1-4 in $a0-$a3 and args 5-6 on the stack at [sp+16]. Since
        // angr-tvod taught extract_syscall_args to traverse the O32 stack
        // save area (CallingConvention::syscall_stack_arg_offset), it now
        // dispatches natively when SP is concrete.
        register_syscalls!(
            r,
            "MIPS32",
            [
                (4001, exit::NativeExitSyscall),
                (4003, read::NativeReadSyscall),
                (4004, write::NativeWriteSyscall),
                (4005, file_path::NativeOpenSyscall),
                (4006, file_path::NativeCloseSyscall),
                (4010, directory::NativeUnlinkSyscall),
                (4012, directory::NativeChdirSyscall),
                (4013, sim_time::NativeTimeSyscall),
                (4020, identity::NativeGetpidSyscall),
                (4023, identity::NativeSetuidSyscall),
                (4024, identity::NativeGetuidSyscall),
                (4027, signals::NativeAlarmSyscall),
                (4029, signals::NativePauseSyscall),
                (4033, file_path::NativeAccessSyscall),
                (4037, signals::NativeKillSyscall),
                (4038, directory::NativeRenameSyscall),
                (4039, directory::NativeMkdirSyscall),
                (4040, directory::NativeRmdirSyscall),
                (4041, file_descriptor::NativeDupSyscall),
                (4042, file_descriptor::NativePipeSyscall),
                (4045, brk::NativeBrkSyscall),
                (4046, identity::NativeSetgidSyscall),
                (4047, identity::NativeGetgidSyscall),
                (4049, identity::NativeGeteuidSyscall),
                (4050, identity::NativeGetegidSyscall),
                (4054, file_descriptor::NativeIoctlSyscall),
                (4055, file_descriptor::NativeFcntlSyscall),
                (4063, file_descriptor::NativeDup2Syscall),
                (4064, identity::NativeGetppidSyscall),
                (4075, rlimit::NativeSetrlimitSyscall),
                (4076, rlimit::NativeGetrlimitSyscall),
                (4078, sim_time::NativeGettimeofdaySyscall),
                (4085, file_path::NativeReadlinkSyscall),
                (4090, mmap::NativeOldMmapSyscall),
                // angr-tvod: 6-arg, args 5-6 from [sp+16] via O32 stack path.
                (4210, mmap::NativeMmap2Syscall),
                (4091, munmap::NativeMunmapSyscall),
                // Legacy 4106/4107/4108 (old 32-bit struct stat) have no Rust
                // writer — angr's `fstat.py` only defines MIPS64, not MIPS32,
                // so legacy stat-family on MIPS32 errors in Python too. Modern
                // O32 glibc emits the LFS `*64` variants below (struct stat64,
                // `write_mips32_stat`), so leave the legacy numbers to Python
                // (angr-11djq.5.3).
                // LFS stat family — struct stat64 (`write_mips32_stat`).
                // MIPS-O32 numbers from angr's `mips-o32` map: stat64=4213,
                // lstat64=4214, fstat64=4215, fstatat64=4293 (no newfstatat
                // on O32 — uses fstatat64, wired to the same handler).
                (4213, file_path::NativeStatSyscall),
                (4214, file_path::NativeLstatSyscall),
                (4215, file_path::NativeFstatSyscall),
                (4293, file_path::NativeNewfstatatSyscall),
                (4125, mprotect::NativeMprotectSyscall),
                (4133, directory::NativeFchdirSyscall),
                (4144, memory_extras::NativeMsyncSyscall),
                (4154, memory_extras::NativeMlockSyscall),
                (4155, memory_extras::NativeMunlockSyscall),
                (4156, memory_extras::NativeMlockallSyscall),
                (4157, memory_extras::NativeMunlockallSyscall),
                (4167, memory_extras::NativeMremapSyscall),
                (4193, signals::NativeRtSigreturnSyscall),
                (4194, sigaction::NativeRtSigactionSyscall),
                (4203, directory::NativeGetcwdSyscall),
                (4218, memory_extras::NativeMadviseSyscall),
                // 4220 = fcntl64 (LFS-style 64-bit offset variant).
                (4220, file_descriptor::NativeFcntl64Syscall),
                (4222, identity::NativeGettidSyscall),
                (4238, concurrency::NativeFutexSyscall),
                (4246, exit::NativeExitSyscall),
                (4248, concurrency::NativeEpollCreateSyscall),
                (4249, concurrency::NativeEpollCtlSyscall),
                (4250, concurrency::NativeEpollWaitSyscall),
                (4263, sim_time::NativeClockGettimeSyscall),
                (4266, signals::NativeTgkillSyscall),
                // newfstatat absent on MIPS32 O32 — uses fstatat64 (4293).
                // renameat2 absent in angr's MIPS-O32 table.
                (4288, file_path::NativeOpenatSyscall),
                (4289, directory::NativeMkdiratSyscall),
                (4294, directory::NativeUnlinkatSyscall),
                (4295, directory::NativeRenameatSyscall),
                (4298, file_path::NativeReadlinkatSyscall),
                (4300, file_path::NativeFaccessatSyscall),
                (4313, concurrency::NativeEpollPwaitSyscall),
                (4319, concurrency::NativeEventfdSyscall),
                (4325, concurrency::NativeEventfd2Syscall),
                (4326, concurrency::NativeEpollCreate1Syscall),
                (4327, file_descriptor::NativeDup3Syscall),
                (4328, file_descriptor::NativePipe2Syscall),
                (4338, rlimit::NativePrlimit64Syscall),
                // angr-dbb1: FD I/O + libc-startup handlers shared with AMD64,
                // MIPS32 O32 numbers from <asm/unistd_o32.h> (4000-based).
                // getrandom is absent from angr's MIPS-O32 table.
                (4019, fd_io::NativeLseekSyscall),
                (4122, startup::NativeUnameSyscall),
                (4145, fd_io::NativeReadvSyscall),
                (4146, fd_io::NativeWritevSyscall),
                (4200, fd_io::NativePread64Syscall),
                (4201, fd_io::NativePwrite64Syscall),
                (4252, startup::NativeSetTidAddressSyscall),
                (4309, startup::NativeSetRobustListSyscall),
            ]
        );

        // Linux MIPS64 N64 (asm/unistd_n64.h). Numbers start at 5000.
        // angr-smtv: mirrors the MIPS32 dispatch with N64 numbering from
        // angr's `mips-n64` syscall table (linux_kernel.py). N64 uses 8
        // register args (a0-a7), so the modern 6-arg `mmap` (5009) works
        // directly — no `old_mmap` / `mmap2` needed. N64 omits the legacy
        // 32-bit `time` syscall (use `gettimeofday` 5094) and lacks a
        // distinct `fcntl64` since it is already 64-bit (`fcntl` 5070
        // covers it). `newfstatat` (5252) exists here unlike MIPS32 O32
        // which uses `fstatat64`.
        register_syscalls!(
            r,
            "MIPS64",
            [
                (5000, read::NativeReadSyscall),
                (5001, write::NativeWriteSyscall),
                (5002, file_path::NativeOpenSyscall),
                (5003, file_path::NativeCloseSyscall),
                (5006, file_path::NativeLstatSyscall),
                (5009, mmap::NativeMmapSyscall),
                (5010, mprotect::NativeMprotectSyscall),
                (5011, munmap::NativeMunmapSyscall),
                (5012, brk::NativeBrkSyscall),
                (5013, sigaction::NativeRtSigactionSyscall),
                (5015, file_descriptor::NativeIoctlSyscall),
                (5020, file_path::NativeAccessSyscall),
                (5021, file_descriptor::NativePipeSyscall),
                (5024, memory_extras::NativeMremapSyscall),
                (5025, memory_extras::NativeMsyncSyscall),
                (5027, memory_extras::NativeMadviseSyscall),
                (5031, file_descriptor::NativeDupSyscall),
                (5032, file_descriptor::NativeDup2Syscall),
                (5033, signals::NativePauseSyscall),
                (5037, signals::NativeAlarmSyscall),
                (5038, identity::NativeGetpidSyscall),
                (5058, exit::NativeExitSyscall),
                (5060, signals::NativeKillSyscall),
                (5070, file_descriptor::NativeFcntlSyscall),
                (5077, directory::NativeGetcwdSyscall),
                (5078, directory::NativeChdirSyscall),
                (5079, directory::NativeFchdirSyscall),
                (5080, directory::NativeRenameSyscall),
                (5081, directory::NativeMkdirSyscall),
                (5082, directory::NativeRmdirSyscall),
                (5085, directory::NativeUnlinkSyscall),
                (5087, file_path::NativeReadlinkSyscall),
                (5094, sim_time::NativeGettimeofdaySyscall),
                (5095, rlimit::NativeGetrlimitSyscall),
                (5100, identity::NativeGetuidSyscall),
                (5102, identity::NativeGetgidSyscall),
                (5103, identity::NativeSetuidSyscall),
                (5104, identity::NativeSetgidSyscall),
                (5105, identity::NativeGeteuidSyscall),
                (5106, identity::NativeGetegidSyscall),
                (5108, identity::NativeGetppidSyscall),
                (5146, memory_extras::NativeMlockSyscall),
                (5147, memory_extras::NativeMunlockSyscall),
                (5148, memory_extras::NativeMlockallSyscall),
                (5149, memory_extras::NativeMunlockallSyscall),
                (5155, rlimit::NativeSetrlimitSyscall),
                (5178, identity::NativeGettidSyscall),
                (5194, concurrency::NativeFutexSyscall),
                (5205, exit::NativeExitSyscall),
                (5207, concurrency::NativeEpollCreateSyscall),
                (5208, concurrency::NativeEpollCtlSyscall),
                (5209, concurrency::NativeEpollWaitSyscall),
                (5211, signals::NativeRtSigreturnSyscall),
                (5222, sim_time::NativeClockGettimeSyscall),
                (5225, signals::NativeTgkillSyscall),
                (5247, file_path::NativeOpenatSyscall),
                (5248, directory::NativeMkdiratSyscall),
                (5252, file_path::NativeNewfstatatSyscall),
                (5253, directory::NativeUnlinkatSyscall),
                (5254, directory::NativeRenameatSyscall),
                (5257, file_path::NativeReadlinkatSyscall),
                (5259, file_path::NativeFaccessatSyscall),
                (5272, concurrency::NativeEpollPwaitSyscall),
                (5278, concurrency::NativeEventfdSyscall),
                (5284, concurrency::NativeEventfd2Syscall),
                (5285, concurrency::NativeEpollCreate1Syscall),
                (5286, file_descriptor::NativeDup3Syscall),
                (5287, file_descriptor::NativePipe2Syscall),
                (5297, rlimit::NativePrlimit64Syscall),
                // angr-dbb1: FD I/O + libc-startup handlers shared with AMD64,
                // MIPS64 N64 numbers from <asm/unistd_n64.h> (5000-based).
                // getrandom is absent from angr's MIPS-N64 table.
                (5008, fd_io::NativeLseekSyscall),
                (5016, fd_io::NativePread64Syscall),
                (5017, fd_io::NativePwrite64Syscall),
                (5018, fd_io::NativeReadvSyscall),
                (5019, fd_io::NativeWritevSyscall),
                (5061, startup::NativeUnameSyscall),
                (5212, startup::NativeSetTidAddressSyscall),
                (5268, startup::NativeSetRobustListSyscall),
            ]
        );

        // ===== DECREE CGC ABI (angr-krp1, angr-rdgs) =====
        //
        // CGC binaries run on x86 but use a custom syscall ABI whose
        // numbers collide with Linux i386 (1=exit/_terminate,
        // 2=fork/transmit, ...). The dispatcher in `stepping.rs` selects
        // this table by checking `ExecutionEnvironment::os_name == "cgc"`
        // and using "CGC" as the registry key instead of the arch name.
        // All seven CGC syscalls land natively in `syscalls/cgc.rs`;
        // `allocate` / `deallocate` carry their own freelist / bump
        // allocator on the new `RustSimState::cgc_*` fields (added in
        // angr-rdgs).
        register_syscalls!(
            r,
            "CGC",
            [
                (1, cgc::NativeTerminateSyscall),
                (2, cgc::NativeTransmitSyscall),
                (3, cgc::NativeReceiveSyscall),
                (4, cgc::NativeFdwaitSyscall),
                (5, cgc::NativeAllocateSyscall),
                (6, cgc::NativeDeallocateSyscall),
                (7, cgc::NativeRandomSyscall),
            ]
        );

        r
    }

    pub(crate) fn register(
        &mut self,
        arch: &'static str,
        num: u64,
        syscall: Arc<dyn NativeSyscall>,
    ) {
        self.handlers.entry(arch).or_default().insert(num, syscall);
    }

    pub(crate) fn get(&self, arch: &str, num: u64) -> Option<&Arc<dyn NativeSyscall>> {
        // Two O(1) hash lookups: arch (`&'static str` keys borrow as `str`,
        // so a runtime `&str` resolves directly) then syscall number.
        self.handlers.get(arch)?.get(&num)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
