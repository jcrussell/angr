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

pub mod arch_prctl;
pub mod brk;
pub mod concurrency;
pub mod exit;
pub mod file_path;
pub mod identity;
pub mod memory_extras;
pub mod mmap;
pub mod mprotect;
pub mod munmap;
pub mod read;
pub mod rlimit;
pub mod sigaction;
pub mod signals;
pub mod sim_time;
pub mod write;

use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::MemoryError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Failure during native syscall dispatch.
///
/// Returning `Err` falls back to the Python `_handle_syscall_callback`
/// path so semantics remain identical to angr's existing behavior.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SyscallError {
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
pub fn extract_concrete_arg(arg: &RustBV, name: &str) -> Result<u64, SyscallError> {
    arg.as_u64()
        .ok_or_else(|| SyscallError::SymbolicArgument(name.to_string()))
}

/// What the dispatcher should do after a syscall handler runs.
#[derive(Debug)]
pub enum SyscallOutcome {
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

pub trait NativeSyscall: Send + Sync {
    fn name(&self) -> &'static str;
    fn num_args(&self) -> usize;
    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError>;
}

/// Registry of native syscall handlers, keyed by `(arch_name, num)`.
pub struct NativeSyscallRegistry {
    handlers: HashMap<(&'static str, u64), Arc<dyn NativeSyscall>>,
    enabled: bool,
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

impl Default for NativeSyscallRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeSyscallRegistry {
    pub fn new() -> Self {
        let mut r = NativeSyscallRegistry {
            handlers: HashMap::new(),
            enabled: true,
        };

        // ===== amd64 (asm/unistd_64.h) =====
        // read (0): stdin (fd=0) only; symbolic bytes mirror NativeRead.
        // write (1): stdout (fd=1) and stderr (fd=2); concrete bytes only.
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
        // lstat (6), readlink (89), newfstatat (262), readlinkat (267),
        //   faccessat (269): file-path syscalls without a Python
        //   SimProcedure (angr-0hif.1, symbolic-return subset). Mirror
        //   syscall_stub. The companion path-aware syscalls (open, openat,
        //   close, stat, fstat, access) all have Python procs that touch
        //   state.posix.fd / state.fs and are intentionally NOT covered
        //   here — they fall back to Python until the FD/FS plumbing is
        //   ported into RustSimState.
        register_syscalls!(r, "AMD64", [
            (0, read::NativeReadSyscall),
            (1, write::NativeWriteSyscall),
            (6, file_path::NativeLstatSyscall),
            (9, mmap::NativeMmapSyscall),
            (10, mprotect::NativeMprotectSyscall),
            (11, munmap::NativeMunmapSyscall),
            (12, brk::NativeBrkSyscall),
            (13, sigaction::NativeRtSigactionSyscall),
            (15, signals::NativeRtSigreturnSyscall),
            (25, memory_extras::NativeMremapSyscall),
            (26, memory_extras::NativeMsyncSyscall),
            (28, memory_extras::NativeMadviseSyscall),
            (34, signals::NativePauseSyscall),
            (37, signals::NativeAlarmSyscall),
            (39, identity::NativeGetpidSyscall),
            (60, exit::NativeExitSyscall),
            (62, signals::NativeKillSyscall),
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
            (228, sim_time::NativeClockGettimeSyscall),
            (231, exit::NativeExitSyscall),
            (232, concurrency::NativeEpollWaitSyscall),
            (233, concurrency::NativeEpollCtlSyscall),
            (234, signals::NativeTgkillSyscall),
            (262, file_path::NativeNewfstatatSyscall),
            (267, file_path::NativeReadlinkatSyscall),
            (269, file_path::NativeFaccessatSyscall),
            (284, concurrency::NativeEventfdSyscall),
            (290, concurrency::NativeEventfd2Syscall),
            (291, concurrency::NativeEpollCreate1Syscall),
            (302, rlimit::NativePrlimit64Syscall),
        ]);

        // ===== Per-arch registrations (angr-7xms) =====
        //
        // Handlers themselves are arch-agnostic (they take RustBV args from
        // the dispatcher, which uses each arch's CallingConvention to extract
        // the right registers). The only per-arch knob is the syscall number,
        // so the tables below map numbers from <asm/unistd_*.h> for each
        // Linux ABI to the same set of handlers.
        //
        // Notes on what is *not* registered here:
        //  - x86/ARM/MIPS32 mmap is the legacy struct-arg form (Iop_mmap on
        //    these archs takes one pointer arg, not six). mmap2 takes six
        //    register args but uses page offsets, not byte offsets — needs a
        //    distinct handler. Skipped to avoid silent semantic drift.
        //  - arch_prctl is amd64-only (no equivalent on other Linux arches).
        //  - On x86/ARM, EAX/R0 also hold the return value, so the
        //    Cdecl/ARMEABI return_register matches the kernel ABI.

        // Linux i386 (asm/unistd_32.h). Skipped: mmap (90, legacy struct-arg
        // form) and mmap2 (192, uses page-offset semantics — needs a distinct
        // handler).
        //
        // Identity getters: i386 has both the legacy 16-bit-uid_t variants
        // (numbers 20/24/47/49/50/64) and the LFS 32-bit-uid_t variants
        // (199-202). Both alias to the same handler since angr returns the
        // same constant 1000 for both forms.
        register_syscalls!(r, "X86", [
            (1, exit::NativeExitSyscall),
            (3, read::NativeReadSyscall),
            (4, write::NativeWriteSyscall),
            (13, sim_time::NativeTimeSyscall),
            (20, identity::NativeGetpidSyscall),
            (23, identity::NativeSetuidSyscall),
            (24, identity::NativeGetuidSyscall),
            (27, signals::NativeAlarmSyscall),
            (29, signals::NativePauseSyscall),
            (37, signals::NativeKillSyscall),
            (45, brk::NativeBrkSyscall),
            (46, identity::NativeSetgidSyscall),
            (47, identity::NativeGetgidSyscall),
            (49, identity::NativeGeteuidSyscall),
            (50, identity::NativeGetegidSyscall),
            (64, identity::NativeGetppidSyscall),
            (75, rlimit::NativeSetrlimitSyscall),
            (76, rlimit::NativeGetrlimitSyscall),
            (78, sim_time::NativeGettimeofdaySyscall),
            (85, file_path::NativeReadlinkSyscall),
            (91, munmap::NativeMunmapSyscall),
            (107, file_path::NativeLstatSyscall),
            (125, mprotect::NativeMprotectSyscall),
            (144, memory_extras::NativeMsyncSyscall),
            (150, memory_extras::NativeMlockSyscall),
            (151, memory_extras::NativeMunlockSyscall),
            (152, memory_extras::NativeMlockallSyscall),
            (153, memory_extras::NativeMunlockallSyscall),
            (163, memory_extras::NativeMremapSyscall),
            (173, signals::NativeRtSigreturnSyscall),
            (174, sigaction::NativeRtSigactionSyscall),
            // 191 = ugetrlimit (LFS uid_t variant) aliases to getrlimit.
            (191, rlimit::NativeGetrlimitSyscall),
            (199, identity::NativeGetuidSyscall),
            (200, identity::NativeGetgidSyscall),
            (201, identity::NativeGeteuidSyscall),
            (202, identity::NativeGetegidSyscall),
            (213, identity::NativeSetuidSyscall),
            (214, identity::NativeSetgidSyscall),
            (219, memory_extras::NativeMadviseSyscall),
            (224, identity::NativeGettidSyscall),
            (240, concurrency::NativeFutexSyscall),
            (252, exit::NativeExitSyscall),
            (254, concurrency::NativeEpollCreateSyscall),
            (255, concurrency::NativeEpollCtlSyscall),
            (256, concurrency::NativeEpollWaitSyscall),
            (265, sim_time::NativeClockGettimeSyscall),
            (270, signals::NativeTgkillSyscall),
            // newfstatat absent on i386 — Linux 32-bit uses fstatat64 (327).
            (305, file_path::NativeReadlinkatSyscall),
            (307, file_path::NativeFaccessatSyscall),
            (323, concurrency::NativeEventfdSyscall),
            (328, concurrency::NativeEventfd2Syscall),
            (329, concurrency::NativeEpollCreate1Syscall),
            (340, rlimit::NativePrlimit64Syscall),
        ]);

        // Linux ARM EABI (arm/asm/unistd-eabi.h). Skipped: mmap (90, legacy
        // form) and mmap2 (192, page-offset semantics). Identity getters
        // share numbering with i386 (both legacy 16-bit and 32-bit variants).
        register_syscalls!(r, "ARM", [
            (1, exit::NativeExitSyscall),
            (3, read::NativeReadSyscall),
            (4, write::NativeWriteSyscall),
            (13, sim_time::NativeTimeSyscall),
            (20, identity::NativeGetpidSyscall),
            (23, identity::NativeSetuidSyscall),
            (24, identity::NativeGetuidSyscall),
            (27, signals::NativeAlarmSyscall),
            (29, signals::NativePauseSyscall),
            (37, signals::NativeKillSyscall),
            (45, brk::NativeBrkSyscall),
            (46, identity::NativeSetgidSyscall),
            (47, identity::NativeGetgidSyscall),
            (49, identity::NativeGeteuidSyscall),
            (50, identity::NativeGetegidSyscall),
            (64, identity::NativeGetppidSyscall),
            (75, rlimit::NativeSetrlimitSyscall),
            (76, rlimit::NativeGetrlimitSyscall),
            (78, sim_time::NativeGettimeofdaySyscall),
            (85, file_path::NativeReadlinkSyscall),
            (91, munmap::NativeMunmapSyscall),
            (107, file_path::NativeLstatSyscall),
            (125, mprotect::NativeMprotectSyscall),
            (144, memory_extras::NativeMsyncSyscall),
            (150, memory_extras::NativeMlockSyscall),
            (151, memory_extras::NativeMunlockSyscall),
            (152, memory_extras::NativeMlockallSyscall),
            (153, memory_extras::NativeMunlockallSyscall),
            (163, memory_extras::NativeMremapSyscall),
            (173, signals::NativeRtSigreturnSyscall),
            (174, sigaction::NativeRtSigactionSyscall),
            // 191 = ugetrlimit (LFS uid_t variant) aliases to getrlimit.
            (191, rlimit::NativeGetrlimitSyscall),
            (199, identity::NativeGetuidSyscall),
            (200, identity::NativeGetgidSyscall),
            (201, identity::NativeGeteuidSyscall),
            (202, identity::NativeGetegidSyscall),
            (213, identity::NativeSetuidSyscall),
            (214, identity::NativeSetgidSyscall),
            (220, memory_extras::NativeMadviseSyscall),
            (224, identity::NativeGettidSyscall),
            (240, concurrency::NativeFutexSyscall),
            (248, exit::NativeExitSyscall),
            (250, concurrency::NativeEpollCreateSyscall),
            (251, concurrency::NativeEpollCtlSyscall),
            (252, concurrency::NativeEpollWaitSyscall),
            (263, sim_time::NativeClockGettimeSyscall),
            (268, signals::NativeTgkillSyscall),
            // newfstatat absent on ARM EABI — uses fstatat64 (327).
            (332, file_path::NativeReadlinkatSyscall),
            (334, file_path::NativeFaccessatSyscall),
            (351, concurrency::NativeEventfdSyscall),
            (356, concurrency::NativeEventfd2Syscall),
            (357, concurrency::NativeEpollCreate1Syscall),
            (369, rlimit::NativePrlimit64Syscall),
        ]);

        // Linux AArch64 (asm-generic/unistd.h). Uses the asm-generic ABI:
        // mmap takes the modern 6-register form with byte offset, so the
        // existing NativeMmapSyscall works as-is.
        // ARM64 uses asm-generic numbering. Note: asm-generic does NOT
        // define `pause` (29 on i386/arm) or `alarm` (27 on i386/arm) —
        // glibc on aarch64 emulates them via setitimer / rt_sigtimedwait.
        // So only kill, tgkill, and rt_sigreturn from angr-0hif.6 land here.
        // ARM64 asm-generic note: epoll_create (legacy) and epoll_wait are
        //   absent; binaries use epoll_create1 (20) and epoll_pwait (22).
        //   epoll_pwait is intentionally NOT registered here — bd-0hif.7
        //   scoped epoll_wait specifically, and a stub for epoll_pwait is
        //   a small follow-up. Likewise the legacy 1-arg eventfd is absent
        //   (only eventfd2 at 19).
        register_syscalls!(r, "ARM64", [
            (19, concurrency::NativeEventfd2Syscall),
            (20, concurrency::NativeEpollCreate1Syscall),
            (21, concurrency::NativeEpollCtlSyscall),
            // asm-generic ABI dropped legacy `lstat` and `readlink` — only
            // the *at variants exist here. faccessat (48), readlinkat (78),
            // newfstatat (79).
            (48, file_path::NativeFaccessatSyscall),
            (63, read::NativeReadSyscall),
            (64, write::NativeWriteSyscall),
            (78, file_path::NativeReadlinkatSyscall),
            (79, file_path::NativeNewfstatatSyscall),
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
        ]);

        // Linux MIPS32 O32 (asm/unistd_o32.h). Numbers start at 4000.
        // mmap (4090, legacy) and mmap2 (4210) skipped — mmap2 takes 6 args
        // but O32 only passes 4 in registers ($a0-$a3); arg 5+ live on the
        // stack and our extract_syscall_args does not currently traverse it.
        register_syscalls!(r, "MIPS32", [
            (4001, exit::NativeExitSyscall),
            (4003, read::NativeReadSyscall),
            (4004, write::NativeWriteSyscall),
            (4013, sim_time::NativeTimeSyscall),
            (4020, identity::NativeGetpidSyscall),
            (4023, identity::NativeSetuidSyscall),
            (4024, identity::NativeGetuidSyscall),
            (4027, signals::NativeAlarmSyscall),
            (4029, signals::NativePauseSyscall),
            (4037, signals::NativeKillSyscall),
            (4045, brk::NativeBrkSyscall),
            (4046, identity::NativeSetgidSyscall),
            (4047, identity::NativeGetgidSyscall),
            (4049, identity::NativeGeteuidSyscall),
            (4050, identity::NativeGetegidSyscall),
            (4064, identity::NativeGetppidSyscall),
            (4075, rlimit::NativeSetrlimitSyscall),
            (4076, rlimit::NativeGetrlimitSyscall),
            (4078, sim_time::NativeGettimeofdaySyscall),
            (4085, file_path::NativeReadlinkSyscall),
            (4091, munmap::NativeMunmapSyscall),
            (4107, file_path::NativeLstatSyscall),
            (4125, mprotect::NativeMprotectSyscall),
            (4144, memory_extras::NativeMsyncSyscall),
            (4154, memory_extras::NativeMlockSyscall),
            (4155, memory_extras::NativeMunlockSyscall),
            (4156, memory_extras::NativeMlockallSyscall),
            (4157, memory_extras::NativeMunlockallSyscall),
            (4167, memory_extras::NativeMremapSyscall),
            (4193, signals::NativeRtSigreturnSyscall),
            (4194, sigaction::NativeRtSigactionSyscall),
            (4218, memory_extras::NativeMadviseSyscall),
            (4222, identity::NativeGettidSyscall),
            (4238, concurrency::NativeFutexSyscall),
            (4246, exit::NativeExitSyscall),
            (4248, concurrency::NativeEpollCreateSyscall),
            (4249, concurrency::NativeEpollCtlSyscall),
            (4250, concurrency::NativeEpollWaitSyscall),
            (4263, sim_time::NativeClockGettimeSyscall),
            (4266, signals::NativeTgkillSyscall),
            // newfstatat absent on MIPS32 O32 — uses fstatat64 (4293).
            (4298, file_path::NativeReadlinkatSyscall),
            (4300, file_path::NativeFaccessatSyscall),
            (4319, concurrency::NativeEventfdSyscall),
            (4325, concurrency::NativeEventfd2Syscall),
            (4326, concurrency::NativeEpollCreate1Syscall),
            (4338, rlimit::NativePrlimit64Syscall),
        ]);

        r
    }

    pub fn empty() -> Self {
        NativeSyscallRegistry {
            handlers: HashMap::new(),
            enabled: true,
        }
    }

    pub fn register(&mut self, arch: &'static str, num: u64, syscall: Arc<dyn NativeSyscall>) {
        self.handlers.insert((arch, num), syscall);
    }

    pub fn get(&self, arch: &str, num: u64) -> Option<&Arc<dyn NativeSyscall>> {
        if !self.enabled {
            return None;
        }
        // The HashMap key is &'static str; lookup needs to compare arch by value.
        // Iterate is fine: the registry is small (typically <20 entries).
        self.handlers
            .iter()
            .find(|((a, n), _)| *a == arch && *n == num)
            .map(|(_, h)| h)
    }

    pub fn enable_all(&mut self) {
        self.enabled = true;
    }

    pub fn disable_all(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Number of registered handlers (for diagnostics / tests).
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_amd64_exit_handlers() {
        let r = NativeSyscallRegistry::new();
        assert!(r.get("AMD64", 0).is_some(), "read (0) should be registered");
        assert!(
            r.get("AMD64", 1).is_some(),
            "write (1) should be registered"
        );
        assert!(
            r.get("AMD64", 60).is_some(),
            "exit (60) should be registered"
        );
        assert!(
            r.get("AMD64", 231).is_some(),
            "exit_group (231) should be registered"
        );
        assert!(
            r.get("AMD64", 10).is_some(),
            "mprotect (10) should be registered"
        );
        assert!(
            r.get("AMD64", 12).is_some(),
            "brk (12) should be registered"
        );
        assert!(
            r.get("AMD64", 11).is_some(),
            "munmap (11) should be registered"
        );
        assert!(r.get("AMD64", 9).is_some(), "mmap (9) should be registered");
        assert!(
            r.get("AMD64", 13).is_some(),
            "rt_sigaction (13) should be registered"
        );
        assert!(
            r.get("AMD64", 96).is_some(),
            "gettimeofday (96) should be registered"
        );
        assert!(
            r.get("AMD64", 158).is_some(),
            "arch_prctl (158) should be registered"
        );
        assert!(
            r.get("AMD64", 201).is_some(),
            "time (201) should be registered"
        );
        assert!(
            r.get("AMD64", 228).is_some(),
            "clock_gettime (228) should be registered"
        );
        assert!(
            r.get("X86", 60).is_none(),
            "amd64 numbers don't apply to x86"
        );
    }

    #[test]
    fn disable_blocks_lookup() {
        let mut r = NativeSyscallRegistry::new();
        assert!(r.is_enabled());
        r.disable_all();
        assert!(r.get("AMD64", 60).is_none());
        r.enable_all();
        assert!(r.get("AMD64", 60).is_some());
    }

    #[test]
    fn empty_registry_has_no_handlers() {
        let r = NativeSyscallRegistry::empty();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.get("AMD64", 60).is_none());
    }

    #[test]
    fn exit_handler_returns_exit_outcome() {
        use crate::state::RustSimState;
        let h = exit::NativeExitSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let outcome = h.call(&mut state, &[]).expect("exit handler succeeds");
        assert!(matches!(outcome, SyscallOutcome::Exit));
        assert_eq!(h.name(), "exit");
        assert_eq!(h.num_args(), 0);
    }

    // ======================================================================
    // Per-arch dispatch tests (angr-7xms)
    // ======================================================================
    //
    // These verify that the Linux syscall numbers for each arch route to a
    // registered handler, and that arch isolation holds (an x86 number does
    // not collide with the AMD64 table). Each arch test enumerates the
    // representative syscalls registered in `register_<arch>` so adding /
    // removing one without updating the test fails noisily.

    #[test]
    fn x86_syscall_numbers_route_to_handlers() {
        let r = NativeSyscallRegistry::new();
        for (num, label) in [
            (1, "exit"),
            (3, "read"),
            (4, "write"),
            (13, "time"),
            (20, "getpid"),
            (24, "getuid"),
            (45, "brk"),
            (47, "getgid"),
            (49, "geteuid"),
            (50, "getegid"),
            (64, "getppid"),
            (78, "gettimeofday"),
            (91, "munmap"),
            (125, "mprotect"),
            (174, "rt_sigaction"),
            (199, "getuid32"),
            (200, "getgid32"),
            (201, "geteuid32"),
            (202, "getegid32"),
            (224, "gettid"),
            (252, "exit_group"),
            (265, "clock_gettime"),
        ] {
            assert!(
                r.get("X86", num).is_some(),
                "X86 syscall {num} ({label}) should be registered",
            );
        }
        // x86 has no native mmap/mmap2 handler (legacy struct-arg / page-offset).
        assert!(r.get("X86", 90).is_none(), "x86 mmap (90) intentionally absent");
        assert!(r.get("X86", 192).is_none(), "x86 mmap2 (192) intentionally absent");
    }

    #[test]
    fn arm_syscall_numbers_route_to_handlers() {
        let r = NativeSyscallRegistry::new();
        for (num, label) in [
            (1, "exit"),
            (3, "read"),
            (4, "write"),
            (13, "time"),
            (20, "getpid"),
            (24, "getuid"),
            (45, "brk"),
            (47, "getgid"),
            (49, "geteuid"),
            (50, "getegid"),
            (64, "getppid"),
            (78, "gettimeofday"),
            (91, "munmap"),
            (125, "mprotect"),
            (174, "rt_sigaction"),
            (199, "getuid32"),
            (200, "getgid32"),
            (201, "geteuid32"),
            (202, "getegid32"),
            (224, "gettid"),
            (248, "exit_group"),
            (263, "clock_gettime"),
        ] {
            assert!(
                r.get("ARM", num).is_some(),
                "ARM syscall {num} ({label}) should be registered",
            );
        }
        assert!(r.get("ARM", 192).is_none(), "ARM mmap2 (192) intentionally absent");
    }

    #[test]
    fn arm64_syscall_numbers_route_to_handlers() {
        let r = NativeSyscallRegistry::new();
        for (num, label) in [
            (63, "read"),
            (64, "write"),
            (93, "exit"),
            (94, "exit_group"),
            (113, "clock_gettime"),
            (134, "rt_sigaction"),
            (169, "gettimeofday"),
            (172, "getpid"),
            (173, "getppid"),
            (174, "getuid"),
            (175, "geteuid"),
            (176, "getgid"),
            (177, "getegid"),
            (178, "gettid"),
            (214, "brk"),
            (215, "munmap"),
            (222, "mmap"),
            (226, "mprotect"),
        ] {
            assert!(
                r.get("ARM64", num).is_some(),
                "ARM64 syscall {num} ({label}) should be registered",
            );
        }
    }

    #[test]
    fn mips32_syscall_numbers_route_to_handlers() {
        let r = NativeSyscallRegistry::new();
        for (num, label) in [
            (4001, "exit"),
            (4003, "read"),
            (4004, "write"),
            (4013, "time"),
            (4020, "getpid"),
            (4024, "getuid"),
            (4045, "brk"),
            (4047, "getgid"),
            (4049, "geteuid"),
            (4050, "getegid"),
            (4064, "getppid"),
            (4078, "gettimeofday"),
            (4091, "munmap"),
            (4125, "mprotect"),
            (4194, "rt_sigaction"),
            (4222, "gettid"),
            (4246, "exit_group"),
            (4263, "clock_gettime"),
        ] {
            assert!(
                r.get("MIPS32", num).is_some(),
                "MIPS32 syscall {num} ({label}) should be registered",
            );
        }
        // MIPS32 mmap2 (4210) takes 6 args but O32 only passes 4 in regs.
        assert!(
            r.get("MIPS32", 4210).is_none(),
            "MIPS32 mmap2 (4210) intentionally absent — args 5+ live on stack"
        );
    }

    #[test]
    fn arch_namespaces_are_isolated() {
        // Same numeric value must NOT collide across arches. e.g. X86 syscall
        // 1 = exit, ARM 1 = exit, but ARM64 1 is unassigned. Each lookup
        // must respect the arch key.
        let r = NativeSyscallRegistry::new();
        assert!(r.get("ARM64", 1).is_none(), "ARM64 has no syscall 1");
        assert!(r.get("AMD64", 1).is_some(), "AMD64 syscall 1 = write");
        // 60 = AMD64 exit, but X86's exit is 1, not 60.
        assert!(r.get("X86", 60).is_none(), "X86 has no syscall 60");
        // 4001 = MIPS32 exit. Other arches should not see it.
        assert!(r.get("AMD64", 4001).is_none(), "AMD64 has no syscall 4001");
        assert!(r.get("X86", 4001).is_none(), "X86 has no syscall 4001");
    }

    #[test]
    fn identity_syscalls_registered_on_all_arches() {
        // angr-0hif.3: getpid / getppid / gettid / getuid / geteuid /
        // getgid / getegid should be hookable on every supported arch.
        // arch → (getpid, getppid, gettid, getuid, geteuid, getgid, getegid)
        let r = NativeSyscallRegistry::new();
        for (arch, nums) in [
            ("AMD64", (39, 110, 186, 102, 107, 104, 108)),
            ("X86", (20, 64, 224, 24, 49, 47, 50)),
            ("ARM", (20, 64, 224, 24, 49, 47, 50)),
            ("ARM64", (172, 173, 178, 174, 175, 176, 177)),
            ("MIPS32", (4020, 4064, 4222, 4024, 4049, 4047, 4050)),
        ] {
            let (pid, ppid, tid, uid, euid, gid, egid) = nums;
            for (num, label) in [
                (pid, "getpid"),
                (ppid, "getppid"),
                (tid, "gettid"),
                (uid, "getuid"),
                (euid, "geteuid"),
                (gid, "getgid"),
                (egid, "getegid"),
            ] {
                let h = r.get(arch, num).unwrap_or_else(|| {
                    panic!("{arch} {label} ({num}) handler missing")
                });
                assert_eq!(
                    h.name(),
                    label,
                    "{arch} syscall {num} should be {label}"
                );
                assert_eq!(h.num_args(), 0, "{arch} {label} takes 0 args");
            }
        }
    }

    #[test]
    fn setuid_setgid_registered_on_all_arches() {
        // angr-pqgu: setuid / setgid have no Python SimProcedure, so the
        // unhandled-syscall path falls through to syscall_stub which
        // returns a fresh symbolic value. The native handlers mirror
        // that via SyscallOutcome::ContinueSymbolic; they should be
        // registered (with name() == "setuid"/"setgid", 1 arg) on every
        // supported arch. Legacy and LFS variants on x86/ARM both alias
        // to the same handler — the syscall semantics are identical
        // and angr's prototype is also the same.
        let r = NativeSyscallRegistry::new();
        // arch → (setuid_num, setgid_num)
        for (arch, setuid, setgid) in [
            ("AMD64", 105u64, 106u64),
            ("X86", 23, 46),
            ("ARM", 23, 46),
            ("ARM64", 146, 144),
            ("MIPS32", 4023, 4046),
        ] {
            let u = r
                .get(arch, setuid)
                .unwrap_or_else(|| panic!("{arch} setuid ({setuid}) missing"));
            assert_eq!(u.name(), "setuid", "{arch} {setuid} should be setuid");
            assert_eq!(u.num_args(), 1, "{arch} setuid takes 1 arg");
            let g = r
                .get(arch, setgid)
                .unwrap_or_else(|| panic!("{arch} setgid ({setgid}) missing"));
            assert_eq!(g.name(), "setgid", "{arch} {setgid} should be setgid");
            assert_eq!(g.num_args(), 1, "{arch} setgid takes 1 arg");
        }
        // x86/ARM LFS variants (32-bit uid_t/gid_t) share the handler.
        for arch in ["X86", "ARM"] {
            let u = r
                .get(arch, 213)
                .unwrap_or_else(|| panic!("{arch} setuid32 (213) missing"));
            assert_eq!(u.name(), "setuid");
            let g = r
                .get(arch, 214)
                .unwrap_or_else(|| panic!("{arch} setgid32 (214) missing"));
            assert_eq!(g.name(), "setgid");
        }
    }

    #[test]
    fn memory_extras_registered_on_all_arches() {
        // angr-0hif.4: madvise / mremap / msync / mlock / munlock /
        // mlockall / munlockall have no Python SimProcedure, so the
        // unhandled-syscall path falls through to `syscall_stub` which
        // returns a fresh symbolic. The native handlers mirror that via
        // SyscallOutcome::ContinueSymbolic; they must be registered with
        // the right name + arity on every supported arch.
        let r = NativeSyscallRegistry::new();
        // arch -> (madvise, mremap, msync, mlock, munlock, mlockall, munlockall)
        let table: &[(&str, [u64; 7])] = &[
            ("AMD64", [28, 25, 26, 149, 150, 151, 152]),
            ("X86", [219, 163, 144, 150, 151, 152, 153]),
            ("ARM", [220, 163, 144, 150, 151, 152, 153]),
            ("ARM64", [233, 216, 227, 228, 229, 230, 231]),
            ("MIPS32", [4218, 4167, 4144, 4154, 4155, 4156, 4157]),
        ];
        let labels = [
            ("madvise", 3usize),
            ("mremap", 5),
            ("msync", 3),
            ("mlock", 2),
            ("munlock", 2),
            ("mlockall", 1),
            ("munlockall", 0),
        ];
        for (arch, nums) in table {
            for (i, (label, nargs)) in labels.iter().enumerate() {
                let num = nums[i];
                let h = r.get(arch, num).unwrap_or_else(|| {
                    panic!("{arch} {label} ({num}) handler missing")
                });
                assert_eq!(
                    h.name(),
                    *label,
                    "{arch} syscall {num} should be {label}"
                );
                assert_eq!(
                    h.num_args(),
                    *nargs,
                    "{arch} {label} should take {nargs} args"
                );
            }
        }
    }

    #[test]
    fn signals_registered_on_all_arches() {
        // angr-0hif.6: kill / tgkill / rt_sigreturn / pause / alarm.
        // kill, rt_sigreturn, pause, alarm have no Python SimProcedure;
        // tgkill is concrete-0 (matches procedures/linux_kernel/tgkill.py).
        // ARM64 does NOT define `pause` (29 on i386/arm) or `alarm` (27)
        // in asm-generic, so only kill/tgkill/rt_sigreturn land there.
        let r = NativeSyscallRegistry::new();

        // (arch, kill, tgkill, rt_sigreturn, pause-or-None, alarm-or-None)
        let table: &[(&str, u64, u64, u64, Option<u64>, Option<u64>)] = &[
            ("AMD64", 62, 234, 15, Some(34), Some(37)),
            ("X86", 37, 270, 173, Some(29), Some(27)),
            ("ARM", 37, 268, 173, Some(29), Some(27)),
            ("ARM64", 129, 131, 139, None, None),
            ("MIPS32", 4037, 4266, 4193, Some(4029), Some(4027)),
        ];

        for &(arch, kill_n, tgkill_n, rtret_n, pause_n, alarm_n) in table {
            let kill = r
                .get(arch, kill_n)
                .unwrap_or_else(|| panic!("{arch} kill ({kill_n}) missing"));
            assert_eq!(kill.name(), "kill");
            assert_eq!(kill.num_args(), 2);

            let tg = r
                .get(arch, tgkill_n)
                .unwrap_or_else(|| panic!("{arch} tgkill ({tgkill_n}) missing"));
            assert_eq!(tg.name(), "tgkill");
            assert_eq!(tg.num_args(), 3);

            let rtret = r
                .get(arch, rtret_n)
                .unwrap_or_else(|| panic!("{arch} rt_sigreturn ({rtret_n}) missing"));
            assert_eq!(rtret.name(), "rt_sigreturn");
            assert_eq!(rtret.num_args(), 0);

            if let Some(n) = pause_n {
                let p = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} pause ({n}) missing"));
                assert_eq!(p.name(), "pause");
                assert_eq!(p.num_args(), 0);
            }
            if let Some(n) = alarm_n {
                let a = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} alarm ({n}) missing"));
                assert_eq!(a.name(), "alarm");
                assert_eq!(a.num_args(), 1);
            }
        }
    }

    #[test]
    fn rlimit_registered_on_all_arches() {
        // angr-0hif.7: getrlimit / setrlimit / prlimit64.
        // getrlimit mirrors procedures/linux_kernel/getrlimit.py — the
        // RLIMIT_STACK branch writes 8388608 to *rlim and returns 0;
        // other resources return a fresh symbolic. setrlimit/prlimit64
        // have no Python SimProcedure and fall through to syscall_stub.
        // x86 and ARM also expose `ugetrlimit` (191), aliased to the
        // same handler since Python defines ugetrlimit(getrlimit).
        let r = NativeSyscallRegistry::new();

        // (arch, getrlimit, setrlimit, prlimit64, ugetrlimit-or-None)
        let table: &[(&str, u64, u64, u64, Option<u64>)] = &[
            ("AMD64", 97, 160, 302, None),
            ("X86", 76, 75, 340, Some(191)),
            ("ARM", 76, 75, 369, Some(191)),
            ("ARM64", 163, 164, 261, None),
            ("MIPS32", 4076, 4075, 4338, None),
        ];

        for &(arch, get_n, set_n, pr_n, uget_n) in table {
            let g = r
                .get(arch, get_n)
                .unwrap_or_else(|| panic!("{arch} getrlimit ({get_n}) missing"));
            assert_eq!(g.name(), "getrlimit");
            assert_eq!(g.num_args(), 2);

            let s = r
                .get(arch, set_n)
                .unwrap_or_else(|| panic!("{arch} setrlimit ({set_n}) missing"));
            assert_eq!(s.name(), "setrlimit");
            assert_eq!(s.num_args(), 2);

            let p = r
                .get(arch, pr_n)
                .unwrap_or_else(|| panic!("{arch} prlimit64 ({pr_n}) missing"));
            assert_eq!(p.name(), "prlimit64");
            assert_eq!(p.num_args(), 4);

            if let Some(un) = uget_n {
                let u = r
                    .get(arch, un)
                    .unwrap_or_else(|| panic!("{arch} ugetrlimit ({un}) missing"));
                // Alias: ugetrlimit shares the getrlimit handler.
                assert_eq!(u.name(), "getrlimit", "{arch} ugetrlimit must alias to getrlimit");
            }
        }
    }

    #[test]
    fn concurrency_registered_on_all_arches() {
        // angr-0hif.7: futex + eventfd/eventfd2 + epoll_create/_create1/
        // _ctl/_wait. ARM64 (asm-generic) drops legacy epoll_create,
        // epoll_wait, and 1-arg eventfd — only the *_create1 / epoll_ctl
        // / epoll_pwait / eventfd2 variants exist.
        let r = NativeSyscallRegistry::new();

        // (arch, futex, eventfd-or-None, eventfd2, epoll_create-or-None,
        //  epoll_create1, epoll_ctl, epoll_wait-or-None)
        #[allow(clippy::type_complexity)]
        let table: &[(
            &str,
            u64,
            Option<u64>,
            u64,
            Option<u64>,
            u64,
            u64,
            Option<u64>,
        )] = &[
            ("AMD64", 202, Some(284), 290, Some(213), 291, 233, Some(232)),
            ("X86", 240, Some(323), 328, Some(254), 329, 255, Some(256)),
            ("ARM", 240, Some(351), 356, Some(250), 357, 251, Some(252)),
            ("ARM64", 98, None, 19, None, 20, 21, None),
            (
                "MIPS32",
                4238,
                Some(4319),
                4325,
                Some(4248),
                4326,
                4249,
                Some(4250),
            ),
        ];

        for &(arch, futex_n, evfd_n, evfd2_n, ec_n, ec1_n, ectl_n, ewait_n) in table {
            let f = r
                .get(arch, futex_n)
                .unwrap_or_else(|| panic!("{arch} futex ({futex_n}) missing"));
            assert_eq!(f.name(), "futex");
            assert_eq!(f.num_args(), 6);

            let e2 = r
                .get(arch, evfd2_n)
                .unwrap_or_else(|| panic!("{arch} eventfd2 ({evfd2_n}) missing"));
            assert_eq!(e2.name(), "eventfd2");
            assert_eq!(e2.num_args(), 2);

            let ec1 = r
                .get(arch, ec1_n)
                .unwrap_or_else(|| panic!("{arch} epoll_create1 ({ec1_n}) missing"));
            assert_eq!(ec1.name(), "epoll_create1");
            assert_eq!(ec1.num_args(), 1);

            let ectl = r
                .get(arch, ectl_n)
                .unwrap_or_else(|| panic!("{arch} epoll_ctl ({ectl_n}) missing"));
            assert_eq!(ectl.name(), "epoll_ctl");
            assert_eq!(ectl.num_args(), 4);

            if let Some(n) = evfd_n {
                let e = r.get(arch, n).unwrap_or_else(|| panic!("{arch} eventfd ({n}) missing"));
                assert_eq!(e.name(), "eventfd");
                assert_eq!(e.num_args(), 1);
            }
            if let Some(n) = ec_n {
                let h = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} epoll_create ({n}) missing"));
                assert_eq!(h.name(), "epoll_create");
                assert_eq!(h.num_args(), 1);
            }
            if let Some(n) = ewait_n {
                let h = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} epoll_wait ({n}) missing"));
                assert_eq!(h.name(), "epoll_wait");
                assert_eq!(h.num_args(), 4);
            }
        }
    }

    #[test]
    fn file_path_stubs_registered_on_all_arches() {
        // angr-0hif.1 symbolic-return subset: lstat / newfstatat /
        // readlink / readlinkat / faccessat. None have a Python
        // SimProcedure; the unhandled-syscall path falls through to
        // `syscall_stub`. The native handlers mirror that via
        // SyscallOutcome::ContinueSymbolic.
        //
        // Per-arch availability:
        //   * AArch64 asm-generic ABI dropped legacy `lstat` and `readlink`
        //     (only *at variants exist).
        //   * 32-bit Linux i386 / ARM EABI / MIPS32 O32 use `fstatat64`
        //     instead of `newfstatat`; absent here.
        let r = NativeSyscallRegistry::new();

        // (arch, lstat-or-None, newfstatat-or-None, readlink-or-None,
        //  readlinkat, faccessat)
        let table: &[(&str, Option<u64>, Option<u64>, Option<u64>, u64, u64)] = &[
            ("AMD64", Some(6), Some(262), Some(89), 267, 269),
            ("X86", Some(107), None, Some(85), 305, 307),
            ("ARM", Some(107), None, Some(85), 332, 334),
            ("ARM64", None, Some(79), None, 78, 48),
            ("MIPS32", Some(4107), None, Some(4085), 4298, 4300),
        ];

        for &(arch, lstat_n, nfstatat_n, readlink_n, readlinkat_n, faccessat_n) in table {
            let rla = r
                .get(arch, readlinkat_n)
                .unwrap_or_else(|| panic!("{arch} readlinkat ({readlinkat_n}) missing"));
            assert_eq!(rla.name(), "readlinkat");
            assert_eq!(rla.num_args(), 4);

            let fa = r
                .get(arch, faccessat_n)
                .unwrap_or_else(|| panic!("{arch} faccessat ({faccessat_n}) missing"));
            assert_eq!(fa.name(), "faccessat");
            assert_eq!(fa.num_args(), 3);

            if let Some(n) = lstat_n {
                let h = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} lstat ({n}) missing"));
                assert_eq!(h.name(), "lstat");
                assert_eq!(h.num_args(), 2);
            }
            if let Some(n) = nfstatat_n {
                let h = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} newfstatat ({n}) missing"));
                assert_eq!(h.name(), "newfstatat");
                assert_eq!(h.num_args(), 4);
            }
            if let Some(n) = readlink_n {
                let h = r
                    .get(arch, n)
                    .unwrap_or_else(|| panic!("{arch} readlink ({n}) missing"));
                assert_eq!(h.name(), "readlink");
                assert_eq!(h.num_args(), 3);
            }
        }
    }

    #[test]
    fn all_arches_have_exit_handler() {
        // Sanity smoke: every supported arch can deadend on exit. Catches
        // accidental dropping of the exit registration during a refactor.
        let r = NativeSyscallRegistry::new();
        // arch_name, exit number
        for (arch, num) in [("AMD64", 60), ("X86", 1), ("ARM", 1), ("ARM64", 93), ("MIPS32", 4001)]
        {
            let h = r
                .get(arch, num)
                .unwrap_or_else(|| panic!("{arch} exit ({num}) handler missing"));
            assert_eq!(h.name(), "exit", "{arch} syscall {num} should be exit");
            assert_eq!(h.num_args(), 0, "{arch} exit takes 0 args");
        }
    }
}
