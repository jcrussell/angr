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
pub mod mmap;
pub mod mprotect;
pub mod munmap;
pub mod sigaction;
pub mod sim_time;

use std::collections::HashMap;
use std::sync::Arc;

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
    #[error("{0}")]
    Other(String),
}

/// What the dispatcher should do after a syscall handler runs.
#[derive(Debug)]
pub enum SyscallOutcome {
    /// State should continue at PC. `ret` is written to the return
    /// register (rax on amd64).
    Continue { ret: u64 },
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
        // amd64: exit (60), exit_group (231) -> deadend.
        r.register("AMD64", 60, Arc::new(exit::NativeExitSyscall { name: "exit" }));
        r.register("AMD64", 231, Arc::new(exit::NativeExitSyscall { name: "exit_group" }));
        // amd64: mprotect (10) — set page perms; -1 on misalign / unmapped.
        r.register("AMD64", 10, Arc::new(mprotect::NativeMprotectSyscall));
        // amd64: brk (12) — grow/query the program break.
        r.register("AMD64", 12, Arc::new(brk::NativeBrkSyscall));
        // amd64: munmap (11) — Python implementation is a no-op return 0.
        r.register("AMD64", 11, Arc::new(munmap::NativeMunmapSyscall));
        // amd64: mmap (9) — anonymous concrete-args fast path; falls back
        // to Python for symbolic / file-backed / collision cases.
        r.register("AMD64", 9, Arc::new(mmap::NativeMmapSyscall));
        // amd64: rt_sigaction (13) — Python is essentially a no-op return 0
        // (with -EINVAL for signum 33).
        r.register("AMD64", 13, Arc::new(sigaction::NativeRtSigactionSyscall));
        // amd64: gettimeofday (96) — writes fresh symbolic timeval; -1 on null.
        r.register("AMD64", 96, Arc::new(sim_time::NativeGettimeofdaySyscall));
        // amd64: arch_prctl (158) — fs_const/gs_const set/get.
        r.register("AMD64", 158, Arc::new(arch_prctl::NativeArchPrctlSyscall));
        // amd64: clock_gettime (228) — CLOCK_REALTIME-only; -1 on null;
        // other clocks fall back to Python's SimProcedureError path.
        r.register("AMD64", 228, Arc::new(sim_time::NativeClockGettimeSyscall));
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
        assert!(r.get("AMD64", 60).is_some(), "exit (60) should be registered");
        assert!(r.get("AMD64", 231).is_some(), "exit_group (231) should be registered");
        assert!(r.get("AMD64", 10).is_some(), "mprotect (10) should be registered");
        assert!(r.get("AMD64", 12).is_some(), "brk (12) should be registered");
        assert!(r.get("AMD64", 11).is_some(), "munmap (11) should be registered");
        assert!(r.get("AMD64", 9).is_some(), "mmap (9) should be registered");
        assert!(r.get("AMD64", 13).is_some(), "rt_sigaction (13) should be registered");
        assert!(r.get("AMD64", 96).is_some(), "gettimeofday (96) should be registered");
        assert!(r.get("AMD64", 158).is_some(), "arch_prctl (158) should be registered");
        assert!(r.get("AMD64", 228).is_some(), "clock_gettime (228) should be registered");
        assert!(r.get("AMD64", 0).is_none(), "read (0) is intentionally unregistered");
        assert!(r.get("X86", 60).is_none(), "amd64 numbers don't apply to x86");
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
        let h = exit::NativeExitSyscall { name: "exit" };
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let outcome = h.call(&mut state, &[]).expect("exit handler succeeds");
        assert!(matches!(outcome, SyscallOutcome::Exit));
        assert_eq!(h.name(), "exit");
        assert_eq!(h.num_args(), 0);
    }
}

