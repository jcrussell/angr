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
//! NOT implemented natively: `pthread_once` (calls the init routine via
//! `self.call(func, ...)` — needs ADDS_EXITS sub-call machinery the native
//! dispatcher lacks) and `pthread_create` (spawns a symbolic branch). Both
//! correctly fall through to their Python SimProcedures.

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

#[cfg(test)]
#[path = "pthread_tests.rs"]
mod tests;
