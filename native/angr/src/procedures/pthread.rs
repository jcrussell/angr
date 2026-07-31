//! Native pthread mutex no-ops.
//!
//! Under single-path symbolic execution there is no real concurrency, so the
//! common locking primitives are pure no-ops that report success. These mirror
//! angr's Python `pthread_mutex_lock` / `pthread_mutex_unlock` SimProcedures
//! (both `return 0`), eliminating a ~100us Python fallback per call on
//! lock-heavy real binaries (measured: pthread_mutex_lock×4 + unlock×3 in the
//! bounded xmllint_getenv slice — see bd memory T2-MEASURE / angr-11djq.4).
//!
//! The mutex pointer argument is declared `bv` (cloned, never inspected) so a
//! symbolic pointer does NOT trigger a Python fallback — it is ignored either
//! way. The `0` (SUCCESS) return is word-width, matching the calling
//! convention's return register, consistent with the ctype procedures.
//!
//! `pthread_once` IS implemented natively below via the sub-call mechanism
//! (`ProcOutcome::CallAndResume`, bead angr-5gf0s / xxukz). NOT implemented
//! natively: `pthread_create` (spawns a symbolic branch); it correctly falls
//! through to its Python SimProcedure.

use crate::procedures::{NativeSimProcedure, ProcOutcome, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// `int pthread_mutex_lock(pthread_mutex_t *mutex);` — always succeeds (0).
    name = "pthread_mutex_lock",
    struct = NativePthreadMutexLock,
    args = [_mutex: bv],
    call |state| {
        Ok(Some(RustBV::concrete(0, state.arch().bits())))
    }
}

crate::declare_proc! {
    /// `int pthread_mutex_unlock(pthread_mutex_t *mutex);` — always succeeds (0).
    name = "pthread_mutex_unlock",
    struct = NativePthreadMutexUnlock,
    args = [_mutex: bv],
    call |state| {
        Ok(Some(RustBV::concrete(0, state.arch().bits())))
    }
}

/// `int pthread_once(pthread_once_t *control, void (*init_routine)(void));`
///
/// Mirrors angr's Python `pthread_once` (`angr/procedures/posix/pthread.py`):
/// read the once-guard byte at `control`; if the "done" bit (value `2`) is set,
/// return `0` without running `init_routine`; otherwise set the bit and sub-call
/// `init_routine` (no args), resuming to return `0`.
///
/// Setting the bit *before* the sub-call is deliberate (matches POSIX/glibc and
/// the Python model): a recursive `pthread_once(control)` inside `init_routine`
/// then sees the bit already set and returns immediately, rather than
/// re-invoking the routine and diverging.
///
/// Falls back to the Python SimProcedure on a symbolic `control` pointer, a
/// symbolic guard byte, or a symbolic stack pointer (the sub-call needs a
/// concrete SP to place the resume sentinel) — see the guard-before-mutate note
/// in `call_ex`.
pub(crate) struct NativePthreadOnce;

impl NativeSimProcedure for NativePthreadOnce {
    fn name(&self) -> &'static str {
        "pthread_once"
    }

    fn num_args(&self) -> usize {
        2
    }

    /// Return-only entry is unsupported — `pthread_once` needs a sub-call, which
    /// only [`Self::call_ex`] can express. Erroring here makes any dispatch path
    /// that still calls `call` (rather than `call_ex`) fall back to the Python
    /// SimProcedure: correct behaviour, just without the native speedup.
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Err(ProcedureError::NotImplemented)
    }

    fn call_ex(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<ProcOutcome, ProcedureError> {
        let control = extract_concrete_arg(&args[0], "pthread_once control")?;
        let func = extract_concrete_arg(&args[1], "pthread_once func")?;
        let bits = state.arch().bits();

        // Read the 1-byte once-guard. A symbolic byte falls back to Python,
        // matching Python raising SimProcedureError on a symbolic control word.
        let guard = state.memory_load(control, 1)?.as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("pthread_once control word".to_string())
        })?;

        // Already initialised: return 0 without invoking func.
        if guard & 2 != 0 {
            return Ok(ProcOutcome::Return(Some(RustBV::concrete(0, bits))));
        }

        // Guard-before-mutate: the sub-call setup writes the resume sentinel to
        // [sp], so it needs a concrete SP. Bail *before* writing the guard bit —
        // otherwise a symbolic-SP fallback to Python would leave the bit set and
        // Python would skip `func` entirely (bead xxukz correctness note).
        if state.get_sp().as_u64().is_none() {
            return Err(ProcedureError::SymbolicArgument(
                "pthread_once: symbolic SP".to_string(),
            ));
        }

        // Set the "done" bit (preserving the other bits), then sub-call func.
        state
            .memory_mut()
            .store_concrete(control, RustBV::concrete((guard | 2) as u128, 8))?;
        Ok(ProcOutcome::CallAndResume {
            target: func,
            args: vec![],
            resume_tag: 0,
        })
    }

    /// `retsite` in the Python proc: once `func` returns, `pthread_once` yields 0.
    fn resume(
        &self,
        state: &mut RustSimState,
        _resume_tag: u32,
        _saved_args: &[RustBV],
    ) -> Result<ProcOutcome, ProcedureError> {
        Ok(ProcOutcome::Return(Some(RustBV::concrete(
            0,
            state.arch().bits(),
        ))))
    }
}

#[cfg(test)]
#[path = "pthread_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
