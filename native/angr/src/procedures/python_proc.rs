//! Python-callable wrapper for the native SimProcedure registry.
//!
//! `PythonNativeProcedure` lets a Python callable be registered as if it
//! were a Rust-side native procedure. The wrapper extracts concrete u64
//! arguments before invoking the callable; symbolic arguments cause a
//! `SymbolicArgument` error so the dispatcher falls back to Python's
//! regular SimProcedure path. The callable returns `Optional[int]` for
//! the procedure's return value (None = no return value, or pair with
//! `no_return=True` for terminal procedures like exit/abort).
//!
//! Contract:
//! ```python
//! def my_proc(args: list[int]) -> Optional[int]:
//!     # args are concrete u64 values
//!     # return None = no return value
//!     # return int = use as return value (wrapped in arch-bits BV)
//!     return 42
//! ```

use pyo3::prelude::*;

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Wraps a Python callable so it can act as a native SimProcedure.
pub struct PythonNativeProcedure {
    name: String,
    num_args: usize,
    no_return: bool,
    callable: Py<PyAny>,
}

impl PythonNativeProcedure {
    pub fn new(name: String, num_args: usize, no_return: bool, callable: Py<PyAny>) -> Self {
        Self {
            name,
            num_args,
            no_return,
            callable,
        }
    }

    /// Leaked &'static str for the trait's name() method. Stored once at
    /// registration time so successive calls return the same pointer.
    fn leaked_name(&self) -> &'static str {
        // Box the name into a heap allocation that lives for the rest of
        // the process. The registry holds an Arc<Self> for the lifetime
        // of the manager; leaking once per registered procedure is OK.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

impl NativeSimProcedure for PythonNativeProcedure {
    fn name(&self) -> &'static str {
        self.leaked_name()
    }

    fn num_args(&self) -> usize {
        self.num_args
    }

    fn no_return(&self) -> bool {
        self.no_return
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Extract concrete arg values. Symbolic args fall back to Python.
        let mut concrete_args: Vec<u64> = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            let v = extract_concrete_arg(arg, &format!("arg{i}"))?;
            concrete_args.push(v);
        }

        let arch_bits = state.arch().bits();

        Python::attach(|py| -> Result<Option<RustBV>, ProcedureError> {
            let result = self.callable.call1(py, (concrete_args,)).map_err(|e| {
                ProcedureError::Other(format!("python procedure '{}' raised: {}", self.name, e))
            })?;

            if result.is_none(py) {
                return Ok(None);
            }

            let ret_int: u64 = result.extract(py).map_err(|e| {
                ProcedureError::Other(format!(
                    "python procedure '{}' returned non-int: {}",
                    self.name, e
                ))
            })?;

            Ok(Some(RustBV::concrete(ret_int as u128, arch_bits)))
        })
    }
}

#[cfg(test)]
#[path = "python_proc_tests.rs"]
mod tests;
