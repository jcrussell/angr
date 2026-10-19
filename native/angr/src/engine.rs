//! Python-facing VEX execution engine.
//!
//! This module provides the PyO3 bindings for the Rust VEX execution engine,
//! allowing it to be used as an alternative engine in angr.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::arch::arch_from_name;
use crate::callbacks::{ExecutionConfig, PythonCallbacks};
use crate::errors::RustExecError;
use crate::interpreter::CbExecutionError;
use crate::solver::RustSolverContext;
use crate::symbolic::SymContext;
use crate::vex::ops::OpError;

/// Map a [`CbExecutionError`] from the callback-driven VEX interpreter to
/// a typed [`RustExecError`] for surfacing as a Python exception.
///
/// `arch` is the engine's `arch_name`; carried into `UnsupportedVexOp` so
/// the Python message can name the arch as well as the op (acceptance
/// criterion for angr-tkbr.3).
///
/// The match is fully exhaustive (no `_` wildcard) so adding a new
/// `CbExecutionError` or `OpError` variant fails to compile until each
/// new case is explicitly triaged here — protecting the Python boundary
/// from silently degrading new failures to `RustExecError::Other`.
fn cb_execution_error_to_typed(err: CbExecutionError, addr: u64, arch: &str) -> RustExecError {
    match err {
        CbExecutionError::InvalidIR(reason) => RustExecError::MalformedIRSB { addr, reason },
        CbExecutionError::Op(op_err) => op_error_to_typed(op_err, arch),
        // Variants that intentionally collapse to `Other`. Listed by name
        // (no `_` wildcard) so adding a CbExecutionError variant produces
        // a compile error and forces a decision instead of silently
        // surfacing as `Other`.
        e @ (CbExecutionError::Memory(_)
        | CbExecutionError::UnknownTemp(_)
        | CbExecutionError::Callback(_)
        | CbExecutionError::LiftError(_)
        | CbExecutionError::Unsupported(_)
        | CbExecutionError::NeedPythonFallback(_)) => RustExecError::Other(e.to_string()),
    }
}

/// Map a VEX [`OpError`] to a typed [`RustExecError`].
///
/// Exhaustive (no `_` wildcard) — see [`cb_execution_error_to_typed`] for
/// the rationale.
fn op_error_to_typed(err: OpError, arch: &str) -> RustExecError {
    match err {
        OpError::UnsupportedNeon { name } => RustExecError::UnsupportedVexOp {
            op_name: name.to_string(),
            arch: arch.to_string(),
        },
        OpError::UnsupportedVectorOp(op_name) => RustExecError::UnsupportedVexOp {
            op_name,
            arch: arch.to_string(),
        },
        // angr-tkbr.2: unmapped pyvex opcode (no entry in parse_opcode).
        // The op_name was captured at parse time via IROp::Unmapped(name).
        OpError::UnsupportedVexOp { op_name } => RustExecError::UnsupportedVexOp {
            op_name,
            arch: arch.to_string(),
        },
        // angr-fs8kb.95: a width-mismatched operand is malformed IR, not an
        // op the engine lacks — `RustExecError::UnsupportedVexOp` would name
        // the whole rejection message as the "op name". `Other` carries the
        // `what: actual vs expected` text `VEXOps::require_operand_width` built.
        e @ (OpError::NotUnary(_)
        | OpError::NotBinary(_)
        | OpError::NotQuaternary(_)
        | OpError::TypeMismatch { .. }
        | OpError::OperandWidthMismatch { .. }
        | OpError::InvalidFloatType(_)
        | OpError::RawOpcode(_)) => RustExecError::Other(e.to_string()),
    }
}

/// Run a single VEX block through `VEXInterpreter` for unit tests.
///
/// Driven directly from Python tests that exercise the typed-error mapping
/// (e.g. NEON / unmapped-opcode / unhandled-CCall → typed exception).
/// Uses a fresh mock [`SymContext`] and an empty [`PythonCallbacks`] —
/// any IRSB that needs real memory / register callbacks will surface a
/// callback or unsupported error.
///
/// `irsb` is a pyvex `IRSB`, `arch_name` is e.g. `"amd64"`
/// or `"arm64"` (case-insensitive, matches [`crate::arch::arch_from_name`]).
///
/// `little_endian` is the target's byte order. It defaults to true because
/// every caller today is on a fixed-little-endian arch, but MIPS32/MIPS64/ARM
/// are either-endian families and `arch_name` cannot answer for them; pass
/// `False` for a big-endian target so a sub-register Get/Put picks the same
/// half angr's Python engine does (see `VEXInterpreter::with_config_endian`).
#[pyfunction]
#[pyo3(signature = (irsb, arch_name, little_endian = true))]
pub(crate) fn execute_irsb_for_test(
    irsb: &Bound<'_, PyAny>,
    arch_name: &str,
    little_endian: bool,
) -> PyResult<()> {
    let arch = arch_from_name(arch_name)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {arch_name}")))?;
    let vex_arch = arch.vex_arch();

    let irsb = crate::vex::pyvex_reader::read_irsb(irsb)?;

    let ctx = SymContext::new_mock();
    let callbacks = PythonCallbacks::new();
    let mut interp = crate::interpreter::VEXInterpreter::with_config_endian(
        vex_arch,
        &ctx,
        crate::callbacks::ExecutionConfig::default(),
        little_endian,
    );

    let addr = irsb.addr;
    match interp.execute_block(&callbacks, &irsb) {
        Ok(_) => Ok(()),
        Err(e) => Err(cb_execution_error_to_typed(e, addr, arch_name).into()),
    }
}

/// Set a Z3 module-level parameter via `Z3_global_param_set`.
///
/// Module-level params (e.g. `smt.random_seed`, `sat.random_seed`,
/// `parallel.enable`) are process-global and read by every
/// `z3::Solver` constructed AFTER the call. Solvers that already
/// exist are not affected.
///
/// This is the only path that takes effect for keys like
/// `smt.random_seed` — the per-solver `Z3_solver_set_params` route
/// is empirically broken for those keys (see angr-iaol.1 memory
/// `iaol1-seed-pin-empirically-broken` and the
/// `build_solver_params` docstring in `symbolic/context.rs`).
///
/// Used by `RustExplorationManager(deterministic=True)` (angr-iaol.2)
/// to pin `smt.random_seed` + `sat.random_seed` before the first
/// `Solver::new`. Z3 4.13 still reserves variable / restart
/// heuristic latitude that is not bounded by these seeds, so the
/// flag narrows but does not close residual model variation.
///
/// AVOID setting `parallel.enable=true` — see
/// `avoid-z3-parallel-enable` memory.
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn set_z3_global_param(key: &str, value: &str) -> PyResult<()> {
    z3::set_global_param(key, value);
    Ok(())
}

/// Read a Z3 module-level parameter back via `Z3_global_param_get`.
///
/// The inverse of [`set_z3_global_param`]. Returns `Some(value)` when Z3
/// recognises the key (whether or not it was explicitly set — Z3 reports
/// its current/default value), or `None` when the key does not exist.
///
/// The high-level `z3` crate exposes `set_global_param` but no getter, so
/// this calls the raw `z3_sys::Z3_global_param_get` directly. Z3 writes the
/// value into a thread-local buffer that the next call overwrites, so we
/// copy it out immediately. Used by the round-trip test that pins down
/// `deterministic=True` seed wiring (angr-a116s.11).
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn get_z3_global_param(key: &str) -> PyResult<Option<String>> {
    use std::ffi::{CStr, CString};

    let c_key = CString::new(key).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid Z3 param key: {e}"))
    })?;
    let mut out: z3_sys::Z3_string = std::ptr::null();
    // SAFETY: `c_key` is a live NUL-terminated CString for the whole call, and
    // `out` is a valid writable slot for the out-pointer. Z3_global_param_get
    // is context-free (it reads the process-global param table), so no Z3
    // context needs to be alive here.
    let found = unsafe { z3_sys::Z3_global_param_get(c_key.as_ptr(), &mut out) };
    if !found || out.is_null() {
        return Ok(None);
    }
    // SAFETY: Z3 returned true and a non-null `out` (both checked above), so it
    // points at a NUL-terminated string in Z3's thread-local buffer. That buffer
    // stays valid until the next Z3_global_param_get on this thread, and we copy
    // it into an owned String before returning — so the borrow never escapes.
    let value = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    Ok(Some(value))
}

/// Stderr logger backed by an [`env_logger::filter::Filter`].
///
/// Output format: `[rust:LEVEL] target: msg` (matches the pre-env_logger
/// hand-rolled logger so any external scrapers stay valid). Filter decisions
/// (per-module level, plain level, full RUST_LOG-style spec) are delegated
/// to env_logger's parser — we just install / replace the inner `Filter`
/// when `set_rust_log_level` is called.
struct StderrLogger {
    filter: parking_lot::RwLock<env_logger::filter::Filter>,
}

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.filter.read().enabled(metadata)
    }
    fn log(&self, record: &log::Record) {
        let filter = self.filter.read();
        if filter.matches(record) {
            eprintln!(
                "[rust:{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }
    fn flush(&self) {}
}

static LOGGER: std::sync::OnceLock<StderrLogger> = std::sync::OnceLock::new();

fn build_filter(spec: &str) -> env_logger::filter::Filter {
    let mut b = env_logger::filter::Builder::new();
    b.parse(spec);
    b.build()
}

/// Set the Rust log level from Python.
///
/// Accepts either:
///   - a single level word: `error` / `warn` / `info` / `debug` / `trace` / `off`
///   - a full RUST_LOG-style filter spec, e.g.
///     `angr::stepping=debug,angr::interpreter=info,warn` (per-module filters
///     plus a fallback level).
///
/// Initializes a stderr logger on first call; subsequent calls swap the
/// active filter in place (no double-install of the global logger).
#[pyfunction]
#[pyo3(signature = (level="info"))]
fn set_rust_log_level(level: &str) -> PyResult<()> {
    // Reject typo'd single-word levels (preserves the pre-existing error for
    // calls like `set_rust_log_level("invalid")`). A spec with `=` or `,` is
    // treated as a full RUST_LOG-style filter and validated by env_logger
    // (parse() silently ignores malformed directives but still returns a
    // usable Filter — same behavior as standard env_logger initialization).
    let is_simple_word = !level.contains('=') && !level.contains(',');
    if is_simple_word {
        match level.to_lowercase().as_str() {
            "error" | "warn" | "warning" | "info" | "debug" | "trace" | "off" => {}
            _ => {
                return Err(PyValueError::new_err(format!(
                    "invalid log level '{level}': use error/warn/info/debug/trace/off, \
                     or a RUST_LOG-style spec like 'angr::stepping=debug'"
                )));
            }
        }
    }

    let new_filter = build_filter(level);
    let max_level = new_filter.filter();

    let logger = LOGGER.get_or_init(|| StderrLogger {
        filter: parking_lot::RwLock::new(build_filter("off")),
    });
    *logger.filter.write() = new_filter;
    // SILENT(cat-a): log::set_logger errors on second call — fine, the first install wins
    // and from then on we only swap the filter inside our Logger.
    drop(log::set_logger(logger));
    log::set_max_level(max_level);
    Ok(())
}

/// Get the size in BYTES of a register on the given architecture.
///
/// Returns `None` for unknown arch names or registers not modelled by Rust.
/// Arch name follows the same case-insensitive conventions as
/// `RustSimState::new` (e.g. "AMD64", "aarch64", "armel", "mips32"). Used by
/// the Python `RustRegisterProxy` to derive register widths from the single
/// Rust source of truth instead of hardcoding prefix-based heuristics.
#[pyfunction]
fn register_size_for_arch(arch_name: &str, reg_name: &str) -> Option<u32> {
    arch_from_name(arch_name).and_then(|a| a.register_size(reg_name))
}

/// Get the canonical register name list for the given architecture.
///
/// Returns an empty vec for unknown arch names. Names are the canonical
/// Rust-side identifiers (e.g. "rax", "x0", "v0") — the same set the
/// interpreter uses. Callers that need a narrower sync subset (e.g.
/// excluding XMM/CC flags) must filter further.
#[pyfunction]
fn register_names_for_arch(arch_name: &str) -> Vec<String> {
    arch_from_name(arch_name)
        .map(|a| {
            a.register_names()
                .iter()
                .map(std::string::ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The engine's register table for `arch_name`, or `None` for an arch the
/// engine does not implement.
///
/// Rust's `arch/*.rs` tables are the engine's source of truth for register
/// names, offsets and sizes (checked against archinfo's VEX guest layout by
/// `tests/engines/rust/test_arch_offset_parity.py`); Python code that
/// describes what crosses the Rust boundary reads them here rather than
/// keeping its own copy. Keys:
///
/// - `registers`: `[(name, offset, size, canonical)]`, every name
///   `register_offset` resolves; `canonical` is false for an `ALIASES` row.
/// - `exported`: `Arch::register_names`, the names that cross the Python
///   boundary.
/// - `sp`, `ip`, `bp`, `syscall_num`: register names (`bp` / `syscall_num`
///   may be `None`).
/// - `state_size`: the guest-state buffer size in bytes.
/// - `cc_args`, `cc_syscall_args`, `cc_ret`, `cc_fp_ret`: the default calling
///   convention's argument and return registers, by name.
#[pyfunction]
pub(crate) fn arch_register_info<'py>(
    py: Python<'py>,
    arch_name: &str,
) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
    let Some(arch) = arch_from_name(arch_name) else {
        return Ok(None);
    };
    let cc = crate::arch::default_cc_for_arch(arch.name());
    let name_of = |offset: u32| arch.register_name(offset);
    let names_of = |offsets: &[u32]| offsets.iter().map(|&o| name_of(o)).collect::<Vec<_>>();
    let registers: Vec<(&str, u32, u32, bool)> = arch
        .canonical_registers()
        .iter()
        .map(|&(name, offset, size)| (name, offset, size, true))
        .chain(
            arch.alias_registers()
                .iter()
                .map(|&(name, offset, size)| (name, offset, size, false)),
        )
        .collect();
    let info = pyo3::types::PyDict::new(py);
    info.set_item("registers", registers)?;
    info.set_item("exported", arch.register_names().to_vec())?;
    info.set_item("sp", name_of(arch.sp_offset()))?;
    info.set_item("ip", name_of(arch.ip_offset()))?;
    info.set_item("bp", arch.bp_offset().and_then(name_of))?;
    info.set_item("syscall_num", arch.syscall_num_offset().and_then(name_of))?;
    info.set_item("state_size", arch.state_size())?;
    info.set_item("cc_args", names_of(cc.arg_registers()))?;
    info.set_item("cc_syscall_args", names_of(cc.syscall_arg_registers()))?;
    info.set_item("cc_ret", name_of(cc.return_register()))?;
    info.set_item("cc_fp_ret", cc.fp_return_register().and_then(name_of))?;
    Ok(Some(info))
}

/// Report whether the Rust engine implements the named architecture.
///
/// Delegates to [`arch_from_name`] so the answer cannot drift from the
/// arches the interpreter actually has. The engine dispatcher in
/// `AngrObjectFactory.simulation_manager` calls this to route a project
/// on an unimplemented arch (PPC32/PPC64/S390X) to the Python engine
/// instead of letting `RustExplorationManager` construction raise.
#[pyfunction]
fn arch_supported(arch_name: &str) -> bool {
    arch_from_name(arch_name).is_some()
}

/// Raise a typed [`RustExecError`] for unit tests.
///
/// Lets the pytest.raises tests cover exception variants whose real-path
/// trigger is gated behind sibling beads (e.g. the syscall panics in
/// angr-tkbr.1). When tkbr.1 lands, syscall handlers raise these directly
/// and this helper becomes redundant — keep it as the API smoke test.
///
/// `kind` selects the variant: "malformed_irsb" | "unsupported_syscall"
/// | "unsupported_vex_op" | "z3" | "oom" | "other". `name` carries the
/// op/syscall name for the relevant variants.
#[pyfunction]
#[pyo3(signature = (kind, name=None, num=None, arch=None, message=None))]
fn _raise_typed_test_error(
    kind: &str,
    name: Option<&str>,
    num: Option<u64>,
    arch: Option<&str>,
    message: Option<&str>,
) -> PyResult<()> {
    let err = match kind {
        "malformed_irsb" => RustExecError::MalformedIRSB {
            addr: num.unwrap_or(0),
            reason: message.unwrap_or("test").to_string(),
        },
        "unsupported_syscall" => RustExecError::UnsupportedSyscall {
            name: name.unwrap_or("unknown").to_string(),
            num: num.unwrap_or(0),
            arch: arch.unwrap_or("AMD64").to_string(),
            reason: message.unwrap_or("no native handler").to_string(),
        },
        "unsupported_vex_op" => RustExecError::UnsupportedVexOp {
            op_name: name.unwrap_or("unknown").to_string(),
            arch: arch.unwrap_or("AMD64").to_string(),
        },
        "z3" => RustExecError::Z3(message.unwrap_or("z3 test").to_string()),
        "oom" => RustExecError::Oom(message.unwrap_or("oom test").to_string()),
        _ => RustExecError::Other(message.unwrap_or("test").to_string()),
    };
    Err(err.into())
}

/// Lift `data` at `addr` with the native libVEX lifter for `arch` (a Python
/// `archinfo.Arch`), at `opt_level` (`None`: the default), and return the
/// IRSB's `{:#?}` rendering. Test-only: the native-vs-pyvex parity check in
/// `test_native_lift_parity.py` compares it with [`irsb_debug_for_test`].
#[pyfunction]
#[pyo3(signature = (arch, data, addr, opt_level = None))]
pub(crate) fn native_lift_debug_for_test(
    arch: &Bound<'_, PyAny>,
    data: &[u8],
    addr: u64,
    opt_level: Option<i32>,
) -> PyResult<String> {
    let lifter = crate::vex::libvex_lifter::NativeLibVEXLifter::from_py_arch(arch)?;
    let name: String = arch.getattr("name")?.extract()?;
    let vex_arch = arch_from_name(&name)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {name}")))?
        .vex_arch();
    let irsb = lifter
        .lift_at(data, addr, vex_arch, opt_level)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(format!("{irsb:#?}"))
}

/// A pyvex `IRSB` as `pyvex_reader` reads it, rendered `{:#?}` (or `{:?}`
/// with `pretty=False`). Test-only: the pyvex side of
/// [`native_lift_debug_for_test`], and the subject of
/// `test_pyvex_reader.py`.
#[pyfunction]
#[pyo3(signature = (irsb, pretty = true))]
pub(crate) fn irsb_debug_for_test(irsb: &Bound<'_, PyAny>, pretty: bool) -> PyResult<String> {
    let irsb = crate::vex::pyvex_reader::read_irsb(irsb)?;
    Ok(if pretty {
        format!("{irsb:#?}")
    } else {
        format!("{irsb:?}")
    })
}

/// Whether some thread holds the engine's libVEX lift lock, probed from a
/// fresh thread so the caller's own (reentrant) hold counts. Test-only: the
/// lift-lock regression test in `test_native_lift_parity.py` calls it from
/// inside the Python `lift_block` callback (angr-mx00p.45).
#[pyfunction]
pub(crate) fn lift_lock_is_held_for_test(py: Python<'_>) -> bool {
    py.detach(|| {
        std::thread::spawn(crate::vex::libvex_lifter::lift_lock_is_free_for_test)
            .join()
            .map(|free| !free)
            .unwrap_or(false)
    })
}

/// The native syscall registrations as `[(angr syscall name, [angr ABI
/// names])]`: the rows `NativeSyscallRegistry::from_angr_tables` resolves
/// angr's numbering through. `test_native_syscall_parity.py` checks each name
/// exists in angr's table for some ABI in its scope.
#[pyfunction]
pub(crate) fn native_syscall_rows_for_test() -> Vec<(&'static str, Vec<&'static str>)> {
    crate::syscalls::NativeSyscallRegistry::by_name_rows()
}

/// `(enum tags, struct layouts)` as returned by [`native_lift_tables_for_test`].
type NativeLiftTables = (Vec<(&'static str, u32)>, Vec<(&'static str, usize)>);

/// The native lifter's enum tags and the `ailment::vex_ffi` struct layouts,
/// for `test_native_lift_parity.py` to check against pyvex's cffi library.
#[pyfunction]
pub(crate) fn native_lift_tables_for_test() -> NativeLiftTables {
    (
        crate::vex::libvex_lifter::enum_tables_for_test(),
        crate::vex::libvex_lifter::layouts_for_test(),
    )
}

test_submod!("engine_tests.rs" => tests);

/// Register the VEX engine module with Python.
pub(crate) fn vex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Typed exception classes (angr-tkbr.3).
    crate::errors::register(m)?;
    m.add_function(pyo3::wrap_pyfunction!(_raise_typed_test_error, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(execute_irsb_for_test, m)?)?;
    m.add_class::<PythonCallbacks>()?;
    m.add_class::<RustSolverContext>()?;
    // Deferred fork types (`DeferredFork` itself is Rust-internal — angr-03vl4.6)
    m.add_class::<ExecutionConfig>()?;
    // Rust-first state
    m.add_class::<crate::state::PyRustSimState>()?;
    // State snapshot for exploration export
    m.add_class::<crate::state::ExplorationStateSnapshot>()?;
    // Exploration manager
    crate::exploration::register_exploration(m)?;
    // Z3 global parameters
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(set_z3_global_param, m)?)?;
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(get_z3_global_param, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_rust_log_level, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_size_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_names_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(arch_supported, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(arch_register_info, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(native_lift_debug_for_test, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(irsb_debug_for_test, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(native_lift_tables_for_test, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lift_lock_is_held_for_test, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(native_syscall_rows_for_test, m)?)?;
    // Memory layout constants (single source of truth; Python imports these
    // rather than redeclaring 0x1000 etc.).
    m.add("PAGE_SIZE", crate::memory::PAGE_SIZE)?;
    m.add("PAGE_MASK", crate::memory::PAGE_MASK)?;
    // Engine version, sourced from native/angr/Cargo.toml at compile time.
    // Re-exported by Python as `angr.exploration.__rust_engine_version__` so
    // user code can branch on engine API version independently of
    // `angr.__version__`.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
