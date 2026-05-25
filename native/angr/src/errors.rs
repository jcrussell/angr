//! Unified Rust execution error type + Python-side exception classes.
//!
//! Provides typed exceptions Python can `pytest.raises` against:
//!
//! ```text
//! Exception
//!   └── RustExecutionError              (base)
//!         ├── RustMalformedIRSBError    (lifter / IR validation failure)
//!         ├── RustUnsupportedSyscallError
//!         ├── RustUnsupportedVexOpError (e.g. NEON op not yet implemented)
//!         ├── RustZ3Error
//!         └── RustOomError
//! ```
//!
//! See angr-tkbr.3. tkbr.1/tkbr.2 land the remaining panic→typed-error
//! conversions across syscalls/ and vex/ops.rs.
//!
//! The new public types are `#[non_exhaustive]` per angr-irwe so future
//! variants can land in minor versions without breaking downstream code.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use thiserror::Error;

create_exception!(
    angr.rustylib.vex_engine,
    RustExecutionError,
    PyException,
    "Base class for typed errors raised by the Rust execution engine."
);

create_exception!(
    angr.rustylib.vex_engine,
    RustMalformedIRSBError,
    RustExecutionError,
    "VEX IRSB failed lift/validation."
);

create_exception!(
    angr.rustylib.vex_engine,
    RustUnsupportedSyscallError,
    RustExecutionError,
    "Syscall is not implemented by a native Rust handler."
);

create_exception!(
    angr.rustylib.vex_engine,
    RustUnsupportedVexOpError,
    RustExecutionError,
    "VEX op is not implemented by the Rust interpreter."
);

create_exception!(
    angr.rustylib.vex_engine,
    RustZ3Error,
    RustExecutionError,
    "Z3 solver returned an error."
);

create_exception!(
    angr.rustylib.vex_engine,
    RustOomError,
    RustExecutionError,
    "Rust engine ran out of memory."
);

/// Categorized execution failures surfaced to Python as typed exceptions.
///
/// Each variant maps to one of the [`create_exception!`] classes above via
/// [`From<RustExecError> for PyErr`].
#[non_exhaustive]
#[derive(Debug, Clone, Error)]
pub enum RustExecError {
    #[error("malformed VEX IRSB at {addr:#x}: {reason}")]
    MalformedIRSB { addr: u64, reason: String },

    #[error("unsupported syscall {name} (num={num}) on {arch}: {reason}")]
    UnsupportedSyscall {
        name: String,
        num: u64,
        arch: String,
        reason: String,
    },

    #[error("unsupported VEX op {op_name} on {arch}")]
    UnsupportedVexOp { op_name: String, arch: String },

    #[error("Z3 error: {0}")]
    Z3(String),

    #[error("out of memory: {0}")]
    Oom(String),

    #[error("execution error: {0}")]
    Other(String),
}

impl From<RustExecError> for PyErr {
    fn from(err: RustExecError) -> PyErr {
        let msg = err.to_string();
        match err {
            RustExecError::MalformedIRSB { .. } => RustMalformedIRSBError::new_err(msg),
            RustExecError::UnsupportedSyscall { .. } => RustUnsupportedSyscallError::new_err(msg),
            RustExecError::UnsupportedVexOp { .. } => RustUnsupportedVexOpError::new_err(msg),
            RustExecError::Z3(_) => RustZ3Error::new_err(msg),
            RustExecError::Oom(_) => RustOomError::new_err(msg),
            RustExecError::Other(_) => RustExecutionError::new_err(msg),
        }
    }
}

/// Register the typed exception classes on the rustylib vex_engine module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("RustExecutionError", py.get_type::<RustExecutionError>())?;
    m.add(
        "RustMalformedIRSBError",
        py.get_type::<RustMalformedIRSBError>(),
    )?;
    m.add(
        "RustUnsupportedSyscallError",
        py.get_type::<RustUnsupportedSyscallError>(),
    )?;
    m.add(
        "RustUnsupportedVexOpError",
        py.get_type::<RustUnsupportedVexOpError>(),
    )?;
    m.add("RustZ3Error", py.get_type::<RustZ3Error>())?;
    m.add("RustOomError", py.get_type::<RustOomError>())?;
    Ok(())
}
