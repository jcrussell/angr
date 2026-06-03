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
mod tests {
    use super::*;

    #[test]
    fn test_exit_no_return() {
        let proc = NativeExit;
        assert!(proc.no_return());
        assert_eq!(proc.num_args(), 1);
        assert_eq!(proc.name(), "exit");
    }

    #[test]
    fn test_exit_returns_none() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeExit
            .call(&mut state, &[RustBV::concrete(0, 32)])
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_underscore_exit() {
        let proc = NativeUnderscoreExit;
        assert!(proc.no_return());
        assert_eq!(proc.name(), "_exit");
        let mut state = RustSimState::new("amd64").unwrap();
        let result = proc.call(&mut state, &[RustBV::concrete(1, 32)]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_abort() {
        let proc = NativeAbort;
        assert!(proc.no_return());
        assert_eq!(proc.num_args(), 0);
        assert_eq!(proc.name(), "abort");
        let mut state = RustSimState::new("amd64").unwrap();
        let result = proc.call(&mut state, &[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_stack_chk_fail() {
        let proc = NativeStackChkFail;
        assert!(proc.no_return());
        assert_eq!(proc.num_args(), 0);
        assert_eq!(proc.name(), "__stack_chk_fail");
        let mut state = RustSimState::new("amd64").unwrap();
        let result = proc.call(&mut state, &[]).unwrap();
        assert!(result.is_none());
    }
}
