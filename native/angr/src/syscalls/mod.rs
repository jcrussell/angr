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
pub mod exit;
pub mod identity;
pub mod mmap;
pub mod mprotect;
pub mod munmap;
pub mod read;
pub mod sigaction;
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
        // setuid / setgid intentionally NOT registered: angr has no Python
        //   SimProcedure for them, so the unhandled-syscall path returns a
        //   fresh symbolic value — a constant-success native return would
        //   diverge.
        register_syscalls!(r, "AMD64", [
            (0, read::NativeReadSyscall),
            (1, write::NativeWriteSyscall),
            (9, mmap::NativeMmapSyscall),
            (10, mprotect::NativeMprotectSyscall),
            (11, munmap::NativeMunmapSyscall),
            (12, brk::NativeBrkSyscall),
            (13, sigaction::NativeRtSigactionSyscall),
            (39, identity::NativeGetpidSyscall),
            (60, exit::NativeExitSyscall),
            (96, sim_time::NativeGettimeofdaySyscall),
            (102, identity::NativeGetuidSyscall),
            (104, identity::NativeGetgidSyscall),
            (107, identity::NativeGeteuidSyscall),
            (108, identity::NativeGetegidSyscall),
            (110, identity::NativeGetppidSyscall),
            (158, arch_prctl::NativeArchPrctlSyscall),
            (186, identity::NativeGettidSyscall),
            (201, sim_time::NativeTimeSyscall),
            (228, sim_time::NativeClockGettimeSyscall),
            (231, exit::NativeExitSyscall),
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
            (24, identity::NativeGetuidSyscall),
            (45, brk::NativeBrkSyscall),
            (47, identity::NativeGetgidSyscall),
            (49, identity::NativeGeteuidSyscall),
            (50, identity::NativeGetegidSyscall),
            (64, identity::NativeGetppidSyscall),
            (78, sim_time::NativeGettimeofdaySyscall),
            (91, munmap::NativeMunmapSyscall),
            (125, mprotect::NativeMprotectSyscall),
            (174, sigaction::NativeRtSigactionSyscall),
            (199, identity::NativeGetuidSyscall),
            (200, identity::NativeGetgidSyscall),
            (201, identity::NativeGeteuidSyscall),
            (202, identity::NativeGetegidSyscall),
            (224, identity::NativeGettidSyscall),
            (252, exit::NativeExitSyscall),
            (265, sim_time::NativeClockGettimeSyscall),
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
            (24, identity::NativeGetuidSyscall),
            (45, brk::NativeBrkSyscall),
            (47, identity::NativeGetgidSyscall),
            (49, identity::NativeGeteuidSyscall),
            (50, identity::NativeGetegidSyscall),
            (64, identity::NativeGetppidSyscall),
            (78, sim_time::NativeGettimeofdaySyscall),
            (91, munmap::NativeMunmapSyscall),
            (125, mprotect::NativeMprotectSyscall),
            (174, sigaction::NativeRtSigactionSyscall),
            (199, identity::NativeGetuidSyscall),
            (200, identity::NativeGetgidSyscall),
            (201, identity::NativeGeteuidSyscall),
            (202, identity::NativeGetegidSyscall),
            (224, identity::NativeGettidSyscall),
            (248, exit::NativeExitSyscall),
            (263, sim_time::NativeClockGettimeSyscall),
        ]);

        // Linux AArch64 (asm-generic/unistd.h). Uses the asm-generic ABI:
        // mmap takes the modern 6-register form with byte offset, so the
        // existing NativeMmapSyscall works as-is.
        register_syscalls!(r, "ARM64", [
            (63, read::NativeReadSyscall),
            (64, write::NativeWriteSyscall),
            (93, exit::NativeExitSyscall),
            (94, exit::NativeExitSyscall),
            (113, sim_time::NativeClockGettimeSyscall),
            (134, sigaction::NativeRtSigactionSyscall),
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
            (222, mmap::NativeMmapSyscall),
            (226, mprotect::NativeMprotectSyscall),
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
            (4024, identity::NativeGetuidSyscall),
            (4045, brk::NativeBrkSyscall),
            (4047, identity::NativeGetgidSyscall),
            (4049, identity::NativeGeteuidSyscall),
            (4050, identity::NativeGetegidSyscall),
            (4064, identity::NativeGetppidSyscall),
            (4078, sim_time::NativeGettimeofdaySyscall),
            (4091, munmap::NativeMunmapSyscall),
            (4125, mprotect::NativeMprotectSyscall),
            (4194, sigaction::NativeRtSigactionSyscall),
            (4222, identity::NativeGettidSyscall),
            (4246, exit::NativeExitSyscall),
            (4263, sim_time::NativeClockGettimeSyscall),
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
    fn setuid_setgid_intentionally_absent() {
        // angr has no Python SimProcedure for setuid / setgid — the
        // unhandled-syscall path returns a fresh symbolic value, so
        // registering a constant-success native handler would diverge.
        // Document the intentional gap on the arches that have explicit
        // setuid/setgid numbers.
        let r = NativeSyscallRegistry::new();
        // AMD64: setuid=105, setgid=106.
        assert!(r.get("AMD64", 105).is_none(), "AMD64 setuid intentionally absent");
        assert!(r.get("AMD64", 106).is_none(), "AMD64 setgid intentionally absent");
        // X86/ARM: setuid=23, setgid=46 (legacy); setuid32=213, setgid32=214.
        assert!(r.get("X86", 23).is_none(), "X86 setuid intentionally absent");
        assert!(r.get("X86", 46).is_none(), "X86 setgid intentionally absent");
        assert!(r.get("ARM", 23).is_none(), "ARM setuid intentionally absent");
        assert!(r.get("ARM", 46).is_none(), "ARM setgid intentionally absent");
        // ARM64: setuid=146, setgid=144.
        assert!(r.get("ARM64", 146).is_none(), "ARM64 setuid intentionally absent");
        assert!(r.get("ARM64", 144).is_none(), "ARM64 setgid intentionally absent");
        // MIPS-O32: setuid=4023, setgid=4046.
        assert!(r.get("MIPS32", 4023).is_none(), "MIPS32 setuid intentionally absent");
        assert!(r.get("MIPS32", 4046).is_none(), "MIPS32 setgid intentionally absent");
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
