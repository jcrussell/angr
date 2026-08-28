//! Shared runtime utilities for the native syscall handlers.
//!
//! Split out of [`crate::syscalls`] (angr-5mnx3.58), which otherwise bundled these
//! with the ~800-line arch-keyed registration table in
//! [`NativeSyscallRegistry::new`](crate::syscalls::NativeSyscallRegistry::new). Nothing
//! here is registration data: it is the vocabulary every `syscalls/*.rs`
//! handler file is written against — the [`SyscallError`] / [`SyscallOutcome`]
//! result types, the [`NativeSyscall`] trait itself, the concrete-arg
//! extraction and host-safety bounding helpers, the fresh-symbolic minting
//! helpers, and the [`stub_syscall!`] / [`require_syscall_args!`] macros.
//!
//! Every item is re-exported from [`crate::syscalls`], so handlers keep spelling their
//! imports `crate::syscalls::X` / `use super::X;` — the split is internal
//! organization, not a path change.

use crate::memory::MemoryError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Shared cap (bytes) on the concrete byte count a native IO handler will
/// service before falling back to Python. `read`/`write` treat it as the
/// whole-request limit, as do `cgc` (transmit/receive/random) and `startup`
/// (getrandom); `fd_io` (readv/writev) applies it per segment. The libc-hook
/// twins in `procedures` — [`crate::procedures::read`] and
/// [`crate::procedures::write`] — share it too, since they cap the same
/// request the syscall handlers do (angr-03vl4.49), as do the stdio
/// request-size caps `fread`/`fwrite`/`fgets`, which had each kept their own
/// local `4096` (angr-0jh0j.45). Kept as a single source of
/// truth so the handlers can't desync (angr-myzjx.18, angr-9ke6b.157)
/// — importers alias it locally (`MAX_IO_SIZE as MAX_READ_SIZE`) so each call
/// site still reads in its own vocabulary.
pub(crate) const MAX_IO_SIZE: u64 = 4096;

/// Shared cap (bytes) on how much address space a single native syscall will
/// *map* before refusing the request (angr-c7xno.80).
///
/// Unlike `MAX_IO_SIZE` this is a host-safety bound, not a fallback threshold:
/// `Memory::map` walks the requested range page-by-page and `MemoryPage::new`
/// eagerly heap-allocates a full `PAGE_SIZE` buffer per page, so a guest
/// calling `mmap(NULL, 0x7fffffff000, ...)` (or an equally large `brk` jump)
/// with *concrete* args would drive the angr host process into OOM — or spin
/// for minutes in the collision scan first. Bouncing to Python does not help;
/// Python's `map_region` has the same shape. So the handlers reject the
/// request the way Linux does (`mmap` → MAP_FAILED, `brk` → break unchanged)
/// instead of falling back.
///
/// 256 MiB matches `syscalls::cgc`'s `CGC_MAX_ALLOCATION`, which already
/// applied this exact defense to `allocate`; that is ~65536 pages, large
/// enough that no realistic analysis workload hits it.
pub(crate) const MAX_MAP_SIZE: u64 = 0x1000_0000;

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

/// A guest-controlled integer paired with the cap it was checked against.
///
/// Handlers that size an allocation or a page-walk from a syscall argument get
/// this instead of a bare `u64`, so the over-cap case cannot be forgotten: the
/// value is only reachable through a `match`, which means a new bounded call
/// site does not compile until its author writes the refusal arm. Six round-3
/// findings (angr-c7xno.67/.80/.81/.92/.93/.94) shared one root cause — the
/// extraction and the bound check were two separate, independently-forgettable
/// steps (angr-91vj9.1).
///
/// The cap itself is a host-safety bound, not a Python-parity rule: see
/// [`MAX_MAP_SIZE`] for why refusing beats falling back to Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) enum BoundedArg {
    /// Concrete and within the cap; carries the value.
    Within(u64),
    /// Concrete but above the cap. The caller must refuse the request with
    /// whatever failure value its syscall uses (`mmap`/`mprotect` → -1, `brk`
    /// → the unchanged break). Deferring to Python is *not* a fix — the Python
    /// handler has the same unbounded shape. `bounded_value` has already
    /// logged the offending value, so the arm does not need to log again.
    Exceeds,
}

/// Check an already-extracted value against `max`, logging the refusal.
///
/// Split out from [`extract_bounded_concrete_arg`] for the handlers whose
/// bounded quantity is *derived* rather than a raw argument — `brk` caps the
/// growth `new_brk - current`, and `mmap`'s three flavors funnel a length that
/// `old_mmap` reads out of a guest struct rather than a register.
pub(crate) fn bounded_value(name: &str, value: u64, max: u64) -> BoundedArg {
    if value > max {
        log::warn!("{name}: {value:#x} exceeds the {max:#x} host-safety cap — refusing");
        BoundedArg::Exceeds
    } else {
        BoundedArg::Within(value)
    }
}

/// Extract a concrete u64 syscall argument *and* bound it in one step.
///
/// The bound is a required parameter, so unlike `extract_concrete_arg` +
/// a follow-up `if value > max` there is no shape in which the check is
/// silently absent. Symbolic args still fall back to Python via
/// `SymbolicArgument`, exactly as [`extract_concrete_arg`] does.
pub(crate) fn extract_bounded_concrete_arg(
    arg: &RustBV,
    name: &str,
    max: u64,
) -> Result<BoundedArg, SyscallError> {
    Ok(bounded_value(name, extract_concrete_arg(arg, name)?, max))
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

/// Build the "give up and return a fresh symbolic of register width" outcome
/// that every stub-style handler produces.
///
/// The three steps — read `arch().bits()`, borrow the solver context just long
/// enough to mint via [`fresh_symbolic`], wrap in
/// [`SyscallOutcome::ContinueSymbolic`] — used to be spelled out at each site,
/// so a change to how symbolic fallback returns are constructed had to be
/// applied in four places (`stub_syscall!` plus three hand-written copies) and
/// missing one would silently keep the old behaviour in just that handler.
/// `stub_syscall!` expands to a call to this, as do the partial handlers that
/// only fall back on their unhandled branches (`futex` wait, `getrlimit` for
/// resources other than `RLIMIT_STACK`, `fcntl`/`ioctl` on an unknown `cmd`).
pub(crate) fn symbolic_outcome(state: &RustSimState, prefix: &'static str) -> SyscallOutcome {
    let bits = state.arch().bits();
    let ret = {
        let ctx = state.solver().borrow();
        fresh_symbolic(&ctx, prefix, bits)
    };
    SyscallOutcome::ContinueSymbolic { ret }
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
                Ok($crate::syscalls::symbolic_outcome(state, $sym_name))
            }
        }
    };
}
pub(crate) use stub_syscall;

/// Bail out of a handler's `call` when the arg slice is shorter than the
/// handler's own declared `num_args()`.
///
/// The dispatcher (`exploration::core_outcome_handlers::handle_syscall_core`)
/// always extracts exactly `num_args()` args before calling, so this never
/// fires in production — it is defense-in-depth for direct calls (unit tests,
/// and any future handler that delegates to a sibling the way
/// `procedures/fortify_*` do).
///
/// Deriving both the arity and the label from `num_args()` / `name()` is the
/// whole point: this replaced 26 hand-written `if args.len() < N` guards that
/// each re-stated `N` and the handler name as literals, which could silently
/// drift from the trait methods when an arity changed (angr-12jjk.17). Use it
/// instead of hand-rolling a new guard; a handler that genuinely needs a
/// *weaker* bound than its declared arity should say so explicitly with a
/// comment rather than copy the old shape back in.
macro_rules! require_syscall_args {
    ($self:ident, $args:ident) => {
        if $args.len() < $self.num_args() {
            return Err($crate::syscalls::SyscallError::Other(format!(
                "{} expected {} args, got {}",
                $self.name(),
                $self.num_args(),
                $args.len()
            )));
        }
    };
}
pub(crate) use require_syscall_args;
