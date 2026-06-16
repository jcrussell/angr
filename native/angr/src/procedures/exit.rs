//! Native exit/abort implementations.
//!
//! These are no-return procedures that terminate the current state.
//! They don't need Python callbacks since they just deadend the state.

use super::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Native exit implementation.
///
/// ```c
/// void exit(int status);
/// ```
pub struct NativeExit;

impl NativeSimProcedure for NativeExit {
    fn name(&self) -> &'static str {
        "exit"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn no_return(&self) -> bool {
        true
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // No-return: just return None to signal state should be deadended
        Ok(None)
    }
}

/// Native _exit implementation (same as exit for our purposes).
pub struct NativeUnderscoreExit;

impl NativeSimProcedure for NativeUnderscoreExit {
    fn name(&self) -> &'static str {
        "_exit"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn no_return(&self) -> bool {
        true
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(None)
    }
}

/// Native abort implementation.
pub struct NativeAbort;

impl NativeSimProcedure for NativeAbort {
    fn name(&self) -> &'static str {
        "abort"
    }

    fn num_args(&self) -> usize {
        0
    }

    fn no_return(&self) -> bool {
        true
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(None)
    }
}

/// Native `__stack_chk_fail` implementation.
///
/// Stack-protector failure handler emitted by GCC/Clang. Like abort, it
/// never returns — the state is deadended by the dispatcher when
/// `no_return()` is true. Already recognized as a terminal in the Python
/// callback dispatcher's `_SIMPROC_NO_RET_TERMINAL` set, so the native
/// path keeps semantics identical while skipping the Python round-trip.
pub struct NativeStackChkFail;

impl NativeSimProcedure for NativeStackChkFail {
    fn name(&self) -> &'static str {
        "__stack_chk_fail"
    }

    fn num_args(&self) -> usize {
        0
    }

    fn no_return(&self) -> bool {
        true
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        Ok(None)
    }
}

#[cfg(test)]
#[path = "exit_tests.rs"]
mod tests;
