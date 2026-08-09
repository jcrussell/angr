//! Error taxonomy for callback-based VEX execution.
//!
//! Split out of `interpreter/mod.rs` (angr-9ke6b.91). Holds
//! [`CbExecutionError`], the [`FallbackStrategy`] each variant maps to, and the
//! reason-marker constants the exploration manager string-matches on when it
//! attributes a `PythonVEXFallback` event to a specific cause.

use crate::callbacks::RunErrorKind;
use crate::memory::MemoryError;
use crate::vex::ops::OpError;

/// Reason string used by the CAS handler when it sees a double-CAS (cmpxchg16b).
/// Shared with `exploration::mod` so the manager can identify DCAS in
/// `PythonVEXFallback` events and bump a dedicated visibility counter.
pub(crate) const DCAS_UNSUPPORTED_REASON: &str = "double compare-and-swap";

/// Reason marker used by the VECRET/GSPTR fallback site
/// (`expressions.rs::eval_expr_with_callbacks`). The manager scans for this
/// substring in `PythonVEXFallback` reasons and bumps
/// `vecret_gsptr_fallback_count` so we can measure how often the corpus
/// actually exercises these vector-call/global-state pointer holders.
/// See bd `angr-2iow` — prevalence drives whether to implement natively or
/// document as a corpus-absent limitation.
pub(crate) const VECRET_GSPTR_REASON: &str = "VECRET/GSPTR";

/// Reason marker for the three dispatch-fabricate binop families
/// (`Iop_Perm{8,32}x*` => `VPerm`, `Iop_Pclmul*`, `Iop_Crc32C`). These parse to a
/// concrete IROp but have no native dispatch arm, so `VEXOps::binop` returns
/// `OpError::NotBinary`. Rather than fabricate a wrong fresh symbolic
/// (`eval_binop`'s BYPASS arm), `eval_binop` routes them to Python's VEX engine
/// — deterministic ops Python models exactly. See bd `angr-s6miz` and the
/// "Parse-succeeds / dispatch-fabricates (silent BYPASS)" section of
/// `docs/extending-angr/rust_vex_ops.rst`.
pub(crate) const DISPATCH_FABRICATE_REASON: &str = "dispatch-fabricate bypass";

/// How an error variant should be handled by the top-level interpreter loop.
///
/// Every [`CbExecutionError`] variant maps to one of these via
/// [`CbExecutionError::strategy`]. Adding a new variant requires an explicit
/// strategy decision — there is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackStrategy {
    /// Hand the failing block to Python's VEX engine and resume from there.
    /// Used for VEX features the Rust interpreter doesn't model
    /// (e.g. unsupported CCalls, VECRET/GSPTR, oversized symbolic addresses).
    PythonCallback,
    /// Surface the error to the caller as `RunResult::Error`. The state
    /// normally moves to the errored stash; no recovery is attempted. Used
    /// for genuine bugs (TypeMismatch, UnknownTemp, InvalidIR, lifter errors,
    /// callback-side failures). "Normally" because the final stash is picked
    /// by `RunErrorKind::route`, not by this strategy — see
    /// [`CbExecutionError::run_error_kind`] for the two exceptions.
    ///
    /// NOTE: `Op` / `TypeMismatch` / `InvalidIR` are exactly the failures
    /// that Python's `HeavyResilienceMixin` would catch and substitute a
    /// default for when `BYPASS_ERRORED_IROP` / `_IRCCALL` / `_IRSTMT` is
    /// set. Because Rust terminates here instead of handing the block to
    /// Python, those bypasses cannot fire — a silent divergence. Rather
    /// than diverge silently, the Python wrapper raises `NotImplementedError`
    /// at manager construction if any `BYPASS_ERRORED_*` option is set (see
    /// `_RAISE_OPTION_NAMES` in `angr/exploration/rust_manager.py`). Wiring a
    /// real bypass would mean routing these variants through `PythonCallback`
    /// (and re-classifying them as recoverable) so the resilience mixin can
    /// act; until then the raise is the contract.
    Panic,
}

/// Errors during callback-based VEX execution.
///
/// Each variant has a documented [`FallbackStrategy`]. The dispatcher in
/// `execution.rs::run` consults [`Self::strategy`] to decide whether the
/// error becomes `RunResult::NeedPythonVEX` (recoverable) or
/// `RunResult::Error` (terminal).
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum CbExecutionError {
    /// Memory error from callback. Strategy: [`FallbackStrategy::Panic`].
    /// These come from underlying memory-model failures (unmapped, perms,
    /// solver timeout) that the interpreter can't paper over.
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Operation error. Strategy: [`FallbackStrategy::Panic`].
    /// VEX op execution failed in a non-recoverable way; lifting to Python
    /// would just rerun the same op.
    #[error("operation error: {0}")]
    Op(#[from] OpError),
    /// Invalid VEX IR. Strategy: [`FallbackStrategy::Panic`].
    #[error("invalid VEX IR: {0}")]
    InvalidIR(String),
    /// Unsupported feature. Strategy: [`FallbackStrategy::PythonCallback`].
    /// Triggered when the Rust interpreter encounters VEX it doesn't model
    /// (e.g. complex symbolic memory operations, certain DirtyHelpers,
    /// symbolic exit targets mid-block).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Unknown temporary variable. Strategy: [`FallbackStrategy::Panic`].
    #[error("unknown temporary t{0}")]
    UnknownTemp(u32),
    /// Python callback error. Strategy: [`FallbackStrategy::Panic`].
    /// The Python side already had its chance and raised; rerunning the
    /// block via the VEX engine would not help.
    #[error("callback error: {0}")]
    Callback(String),
    /// Block lifting error. Strategy: [`FallbackStrategy::Panic`].
    #[error("lift error: {0}")]
    LiftError(String),
    /// Needs Python fallback for special expressions.
    /// Strategy: [`FallbackStrategy::PythonCallback`]. Distinct from
    /// `Unsupported` so call sites can request fallback explicitly without
    /// having to invent a "feature missing" message (e.g. VECRET/GSPTR,
    /// non-eflags CCalls).
    #[error("need Python fallback: {0}")]
    NeedPythonFallback(String),
}

impl CbExecutionError {
    /// Map this error to its declared [`FallbackStrategy`]. The match is
    /// exhaustive so adding a variant forces an explicit strategy choice.
    pub(crate) fn strategy(&self) -> FallbackStrategy {
        match self {
            CbExecutionError::Unsupported(_) | CbExecutionError::NeedPythonFallback(_) => {
                FallbackStrategy::PythonCallback
            }
            CbExecutionError::Memory(_)
            | CbExecutionError::Op(_)
            | CbExecutionError::InvalidIR(_)
            | CbExecutionError::UnknownTemp(_)
            | CbExecutionError::Callback(_)
            | CbExecutionError::LiftError(_) => FallbackStrategy::Panic,
        }
    }

    /// Classify this error for the exploration stepping loop (angr-zzju9).
    ///
    /// `LiftError` is the designed signal that a block could not be lifted —
    /// the Python lift callback returned the empty-IRSB sentinel (e.g. on
    /// `SimEngineError: No bytes in memory`) or the callback itself failed.
    /// Such states gracefully deadend, matching the vanilla Python engine.
    /// Every other `Panic`-strategy variant — including `InvalidIR`, which a
    /// genuinely malformed IRSB now maps to — is a real error that moves the
    /// state to the errored stash.
    ///
    /// This kind is not the last word on the stash, though: `RunErrorKind`
    /// alone does not decide routing. `RunErrorKind::route` also takes the pc
    /// the error was reported at, and downgrades a `Fatal` at pc 0 to
    /// `ErrorRoute::NullAddressDeadend` — a deadend, not an errored state —
    /// regardless of which variant produced the `Fatal` (angr-c7xno.30).
    /// See the `ErrorRoute` docs for that carve-out's rationale.
    pub(crate) fn run_error_kind(&self) -> RunErrorKind {
        match self {
            CbExecutionError::LiftError(_) => RunErrorKind::Deadend,
            _ => RunErrorKind::Fatal,
        }
    }
}
