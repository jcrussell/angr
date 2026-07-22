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
//!
//! # Test-only taxonomy vs. the live exploration path (angr-ghwsd.3)
//!
//! The variant→subclass mapping in [`From<RustExecError> for PyErr`] is
//! reached **only** from the test-only `#[pyfunction]` hooks
//! `engine::execute_irsb_for_test` and `engine::_raise_typed_test_error`.
//! Those are the only callers of `cb_execution_error_to_typed` /
//! `op_error_to_typed`, so a `pytest.raises(RustUnsupportedVexOpError)`
//! only matches when driving one of those test entry points.
//!
//! During real exploration the typed `CbExecutionError` is **stringified
//! at the interpreter boundary** and never reaches this enum: a failing
//! step produces `RunResult::Error { message: e.to_string(), addr }`
//! (`interpreter/execution.rs`), which `stepping::try_step` collapses into
//! `StepError::Error(state, String)` and `run_loop` pushes into the
//! errored stash as a `(pc, message, state_id)` record. The concrete
//! variant (UnsupportedVexOp vs Z3 vs Oom …) is therefore lost the moment
//! a live step fails — only the formatted message survives, readable off
//! the errored-stash record (`state.error`). The errored-stash string is
//! the production contract; the typed subclasses are a test-harness
//! affordance. Carrying the typed error through the live path is the
//! deferred option (b) on angr-ghwsd.3.
//!
//! # Internal error-typing convention (angr-0mqkc.10)
//!
//! `RustExecError` above is the **user-facing** taxonomy — it exists to
//! give Python typed exceptions to `pytest.raises` against, and (per the
//! note above) is reached only from the test-only hooks. It is **not** the
//! type internal fallible functions should thread through. The crate-wide
//! convention for everything *below* the PyO3 boundary is:
//!
//! 1. **`PyResult<T>` lives strictly at the boundary.** Only `#[pymethods]`
//!    and `#[pyfunction]` entry points return `PyResult`; they are the sole
//!    place a `PyErr` is constructed. An internal helper that never touches
//!    the interpreter/Python bridge should not return `PyResult`.
//!
//! 2. **Internal functions return `Result<T, DomainError>` with a
//!    subsystem-local, `thiserror`-derived enum.** Each subsystem owns its
//!    error type and composes upward via `#[from]`. The canonical set:
//!    `CbExecutionError` (`interpreter/mod.rs`), `StepError` /
//!    `SubcallSetupError` (`exploration/stepping.rs`), `MemoryError`
//!    (`memory/mod.rs`), `OpError` (`vex/ops.rs`), `ProcedureError`
//!    (`procedures/mod.rs`), `SyscallError` (`syscalls/mod.rs`),
//!    `BridgeError` (`claripy_bridge/mod.rs`), `LiftError`
//!    (`vex/lifter.rs`), and friends. Prefer growing/reusing one of these
//!    over inventing an ad-hoc type.
//!
//! 3. **No stringly-typed errors and no `anyhow`.** `Result<_, String>`
//!    loses the variant a caller needs to branch on; `anyhow` is
//!    deliberately not a dependency. Errors collapse to a `String` (or a
//!    `PyErr`) *only* at the boundary — the errored-stash record and the
//!    `From<RustExecError> for PyErr` map below are the two sanctioned
//!    collapse points.
//!
//! `interpreter/` and `exploration/` already follow this end-to-end (no
//! `Result<_, String>`, no `anyhow`); new code in those subsystems — and
//! ideally the rest of the crate — should keep to it. The mirror of this
//! note for doc readers lives in the "Internal error convention"
//! subsection of `docs/advanced-topics/rust_engine.rst`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use thiserror::Error;

create_exception!(
    angr.rustylib.vex_engine,
    RustExecutionError,
    PyException,
    "Base class for typed errors raised by the Rust execution engine.\n\n\
     See the 'User-facing error taxonomy' section of\n\
     docs/advanced-topics/rust_engine.rst for the full hierarchy,\n\
     per-variant trigger conditions, and known incompatibilities\n\
     (e.g. Oppologist's except SimError clause does not catch these)."
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

/// Map a typed [`RustExecError`] to its Python exception subclass.
///
/// Reached only from the test-only `#[pyfunction]` hooks (see the
/// "Test-only taxonomy vs. the live exploration path" section in the
/// module docs). The live exploration path stringifies the error into
/// the errored stash and never constructs a `RustExecError`.
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

#[cfg(test)]
#[path = "errors_tests.rs"]
mod tests;

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
