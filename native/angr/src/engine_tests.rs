//! Unit tests for the Python-facing engine module (angr-ph300.5).
//!
//! Covers the two pieces of `engine.rs` that carry real logic rather than
//! PyO3 plumbing: the `RUST_LOG`-style log-filter wiring behind
//! `set_rust_log_level`, and the exhaustive `CbExecutionError` / `OpError` →
//! [`RustExecError`] triage that guards the Python exception boundary.

use super::*;

use crate::memory::MemoryError;
use crate::vex::ir::IROp;
use log::{Level, LevelFilter};

fn metadata(target: &str, level: Level) -> log::Metadata<'_> {
    log::Metadata::builder().target(target).level(level).build()
}

// ---------------------------------------------------------------------------
// log-filter spec parsing
// ---------------------------------------------------------------------------

#[test]
fn build_filter_accepts_a_bare_level() {
    let f = build_filter("debug");
    assert_eq!(f.filter(), LevelFilter::Debug);
    assert!(f.enabled(&metadata("rustylib::stash", Level::Debug)));
    assert!(!f.enabled(&metadata("rustylib::stash", Level::Trace)));
}

#[test]
fn build_filter_honors_a_per_module_spec() {
    // The spec documented in CLAUDE.md: raise one module, silence everything else.
    let f = build_filter("rustylib::stash=warn,off");
    assert!(f.enabled(&metadata("rustylib::stash", Level::Warn)));
    assert!(!f.enabled(&metadata("rustylib::stash", Level::Info)));
    assert!(!f.enabled(&metadata("rustylib::exploration", Level::Error)));
    // `max_level` must cover the loudest directive or `log!` short-circuits
    // before the filter is ever consulted.
    assert!(f.filter() >= LevelFilter::Warn);
}

#[test]
fn build_filter_off_disables_every_level() {
    let f = build_filter("off");
    assert_eq!(f.filter(), LevelFilter::Off);
    assert!(!f.enabled(&metadata("rustylib::stash", Level::Error)));
}

#[test]
fn set_rust_log_level_rejects_a_typoed_single_word() {
    // A bare word that is not a level is a typo, not a filter spec — it must
    // raise rather than silently parse to "off".
    let err = set_rust_log_level("invalid").unwrap_err();
    Python::initialize();
    Python::attach(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.value(py).to_string().contains("invalid log level"));
    });
}

#[test]
fn set_rust_log_level_accepts_levels_and_specs() {
    for spec in [
        "error",
        "warn",
        "warning",
        "info",
        "debug",
        "trace",
        "OFF",
        // Anything containing `=` or `,` is handed to env_logger unvalidated,
        // including a malformed directive — env_logger ignores it, no panic.
        "rustylib::stash=warn,off",
        "not_a_level=lolwut,off",
    ] {
        assert!(set_rust_log_level(spec).is_ok(), "rejected {spec:?}");
    }
    // Leave the process-global logger quiet for the rest of the suite.
    set_rust_log_level("off").unwrap();
    assert_eq!(log::max_level(), LevelFilter::Off);
}

// ---------------------------------------------------------------------------
// error triage at the Python boundary
// ---------------------------------------------------------------------------

#[test]
fn invalid_ir_becomes_malformed_irsb_carrying_the_addr() {
    let typed = cb_execution_error_to_typed(
        CbExecutionError::InvalidIR("no statements".to_string()),
        0x400_123,
        "amd64",
    );
    match typed {
        RustExecError::MalformedIRSB { addr, ref reason } => {
            assert_eq!(addr, 0x400_123);
            assert_eq!(reason, "no statements");
        }
        other => panic!("expected MalformedIRSB, got {other:?}"),
    }
}

#[test]
fn op_errors_route_through_op_error_to_typed_with_the_arch() {
    let typed = cb_execution_error_to_typed(
        CbExecutionError::Op(OpError::UnsupportedNeon {
            name: "Iop_Add64Fx2",
        }),
        0x1000,
        "arm64",
    );
    match typed {
        RustExecError::UnsupportedVexOp { op_name, arch } => {
            assert_eq!(op_name, "Iop_Add64Fx2");
            assert_eq!(arch, "arm64");
        }
        other => panic!("expected UnsupportedVexOp, got {other:?}"),
    }
}

#[test]
fn unmapped_opcode_and_vector_op_also_become_unsupported_vex_op() {
    for err in [
        OpError::UnsupportedVectorOp("Iop_QNarrowBin32Sto16Sx8".to_string()),
        OpError::UnsupportedVexOp {
            op_name: "Iop_Bogus".to_string(),
        },
    ] {
        let expected_msg = err.to_string();
        match op_error_to_typed(err, "amd64") {
            RustExecError::UnsupportedVexOp { op_name, arch } => {
                assert!(expected_msg.contains(&op_name), "op name lost: {op_name}");
                assert_eq!(arch, "amd64");
            }
            other => panic!("expected UnsupportedVexOp, got {other:?}"),
        }
    }
}

#[test]
fn deliberately_untyped_variants_collapse_to_other_preserving_the_message() {
    // These are the variants `cb_execution_error_to_typed` lists by name so a
    // new variant fails to compile; assert they still stringify losslessly.
    for err in [
        CbExecutionError::Memory(MemoryError::Unmapped { addr: 0, size: 0 }),
        CbExecutionError::UnknownTemp(7),
        CbExecutionError::Callback("python raised".to_string()),
        CbExecutionError::LiftError("bad bytes".to_string()),
        CbExecutionError::Unsupported("symbolic exit".to_string()),
    ] {
        let expected = err.to_string();
        match cb_execution_error_to_typed(err, 0x1000, "amd64") {
            RustExecError::Other(msg) => assert_eq!(msg, expected),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}

#[test]
fn non_op_shaped_op_errors_collapse_to_other() {
    let err = OpError::NotUnary(IROp::Unmapped("Iop_Bogus"));
    let expected = err.to_string();
    match op_error_to_typed(err, "amd64") {
        RustExecError::Other(msg) => assert_eq!(msg, expected),
        other => panic!("expected Other, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// arch introspection helpers (the engine dispatcher's routing oracle)
// ---------------------------------------------------------------------------

#[test]
fn arch_helpers_agree_with_arch_from_name() {
    assert!(arch_supported("AMD64"));
    assert!(arch_supported("amd64"), "arch names are case-insensitive");
    assert!(
        !arch_supported("ppc32"),
        "unimplemented arch must route to Python"
    );

    assert_eq!(register_size_for_arch("AMD64", "rax"), Some(8));
    assert_eq!(register_size_for_arch("AMD64", "not_a_register"), None);
    assert_eq!(register_size_for_arch("ppc32", "r0"), None);

    let names = register_names_for_arch("AMD64");
    assert!(names.iter().any(|n| n == "rax"));
    assert!(
        register_names_for_arch("ppc32").is_empty(),
        "unknown arch must yield an empty name list, not a panic"
    );
}

#[test]
#[cfg(feature = "libvex-ffi")]
fn libvex_ffi_enabled_reports_true_when_feature_is_on() {
    assert!(libvex_ffi_enabled());
}

#[test]
#[cfg(not(feature = "libvex-ffi"))]
fn libvex_ffi_enabled_reports_false_when_feature_is_off() {
    assert!(!libvex_ffi_enabled());
}
