//! Unit tests for the user-facing error taxonomy (angr-ph300.5).
//!
//! Two contracts are pinned here:
//!
//! 1. **Display** — the `thiserror` format strings are what the errored-stash
//!    record and the Python exception message are built from (see the
//!    "Test-only taxonomy vs. the live exploration path" note in `errors.rs`),
//!    so a silent reformat is a user-visible change.
//! 2. **Variant → subclass** — `From<RustExecError> for PyErr` must land each
//!    variant on its own `create_exception!` class, and every class must stay a
//!    subclass of `RustExecutionError` so `except RustExecutionError` keeps
//!    catching the whole family.

use super::*;

fn malformed() -> RustExecError {
    RustExecError::MalformedIRSB {
        addr: 0x400_123,
        reason: "no statements".to_string(),
    }
}

fn unsupported_syscall() -> RustExecError {
    RustExecError::UnsupportedSyscall {
        name: "sendmmsg".to_string(),
        num: 307,
        arch: "AMD64".to_string(),
        reason: "no native handler".to_string(),
    }
}

fn unsupported_vex_op() -> RustExecError {
    RustExecError::UnsupportedVexOp {
        op_name: "Iop_Add64Fx2".to_string(),
        arch: "ARM64".to_string(),
    }
}

#[test]
fn display_malformed_irsb_renders_addr_in_hex() {
    // `{addr:#x}` — a switch to decimal would silently break message scrapers.
    assert_eq!(
        malformed().to_string(),
        "malformed VEX IRSB at 0x400123: no statements"
    );
}

#[test]
fn display_unsupported_syscall_names_number_and_arch() {
    assert_eq!(
        unsupported_syscall().to_string(),
        "unsupported syscall sendmmsg (num=307) on AMD64: no native handler"
    );
}

#[test]
fn display_unsupported_vex_op_names_op_and_arch() {
    // angr-tkbr.3 acceptance: the message must name the arch as well as the op.
    assert_eq!(
        unsupported_vex_op().to_string(),
        "unsupported VEX op Iop_Add64Fx2 on ARM64"
    );
}

#[test]
fn display_string_variants() {
    assert_eq!(
        RustExecError::Z3("timeout".to_string()).to_string(),
        "Z3 error: timeout"
    );
    assert_eq!(
        RustExecError::Oom("page table".to_string()).to_string(),
        "out of memory: page table"
    );
    assert_eq!(
        RustExecError::Other("boom".to_string()).to_string(),
        "execution error: boom"
    );
}

/// Assert `err` converts to a `PyErr` of exactly `T`, that it is also a
/// `RustExecutionError` (the family base), and that the Python-visible message
/// is the `Display` rendering verbatim.
fn assert_maps_to<T>(err: RustExecError)
where
    T: pyo3::type_object::PyTypeInfo,
{
    let expected = err.to_string();
    let py_err: PyErr = err.into();
    Python::attach(|py| {
        assert!(
            py_err.is_instance_of::<T>(py),
            "wrong subclass for {expected:?}: got {py_err:?}"
        );
        assert!(
            py_err.is_instance_of::<RustExecutionError>(py),
            "{expected:?} escaped the RustExecutionError family"
        );
        assert_eq!(py_err.value(py).to_string(), expected);
    });
}

#[test]
fn every_variant_maps_to_its_own_subclass() {
    Python::initialize();
    assert_maps_to::<RustMalformedIRSBError>(malformed());
    assert_maps_to::<RustUnsupportedSyscallError>(unsupported_syscall());
    assert_maps_to::<RustUnsupportedVexOpError>(unsupported_vex_op());
    assert_maps_to::<RustZ3Error>(RustExecError::Z3("timeout".to_string()));
    assert_maps_to::<RustOomError>(RustExecError::Oom("page table".to_string()));
    assert_maps_to::<RustExecutionError>(RustExecError::Other("boom".to_string()));
}

#[test]
fn subclasses_do_not_alias_each_other() {
    Python::initialize();
    let py_err: PyErr = RustExecError::Z3("timeout".to_string()).into();
    Python::attach(|py| {
        assert!(!py_err.is_instance_of::<RustOomError>(py));
        assert!(!py_err.is_instance_of::<RustMalformedIRSBError>(py));
        assert!(!py_err.is_instance_of::<RustUnsupportedVexOpError>(py));
        assert!(!py_err.is_instance_of::<RustUnsupportedSyscallError>(py));
    });
}
