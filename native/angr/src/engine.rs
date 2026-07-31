//! Python-facing VEX execution engine.
//!
//! This module provides the PyO3 bindings for the Rust VEX execution engine,
//! allowing it to be used as an alternative engine in angr.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::arch::arch_from_name;
use crate::callbacks::{
    BranchPolicy, DeferredFork, ExecutionConfig, LoopExecutionEvent, PythonCallbacks,
};
use crate::errors::RustExecError;
use crate::interpreter::CbExecutionError;
use crate::solver::RustSolverContext;
use crate::symbolic::SymContext;
use crate::vex::deserialize_irsb;
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
        e @ (OpError::NotUnary(_)
        | OpError::NotBinary(_)
        | OpError::NotQuaternary(_)
        | OpError::TypeMismatch { .. }
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
/// `irsb_json` is a serialized pyvex IRSB, `arch_name` is e.g. `"amd64"`
/// or `"arm64"` (case-insensitive, matches [`crate::arch::arch_from_name`]).
#[pyfunction]
pub(crate) fn execute_irsb_for_test(irsb_json: &str, arch_name: &str) -> PyResult<()> {
    let arch = arch_from_name(arch_name)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {arch_name}")))?;
    let vex_arch = arch.vex_arch();

    let irsb = deserialize_irsb(irsb_json)
        .map_err(|e| PyValueError::new_err(format!("Failed to deserialize IRSB: {e}")))?;

    let ctx = SymContext::new_mock();
    let callbacks = PythonCallbacks::new();
    let mut interp = crate::interpreter::VEXInterpreter::new(vex_arch, &ctx);

    let addr = irsb.addr;
    match interp.execute_block(&callbacks, &irsb) {
        Ok(_) => Ok(()),
        Err(e) => Err(cb_execution_error_to_typed(e, addr, arch_name).into()),
    }
}

/// Set the Rust Z3 thread-local context to share Python's Z3 context.
///
/// This enables Rust and Python to share Z3 ASTs without translation.
/// The pointer must be a valid Z3_context created by Python's z3 module.
/// Call this once at startup, before creating any Rust solver contexts.
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn set_shared_z3_context(py_z3_ctx_ptr: usize) -> PyResult<bool> {
    if py_z3_ctx_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Z3 context pointer is null",
        ));
    }

    // Create a z3-rs Context from the raw pointer.
    // SAFETY: The pointer comes from Python's z3.main_ctx().ctx.value which
    // is a valid Z3_context created by Z3_mk_context_rc(). Python owns this
    // context and will delete it at process exit. The Context::from_raw
    // constructor marks it as "borrowed" so ContextInternal::drop will NOT
    // call Z3_del_context — Python retains ownership.
    unsafe {
        let raw_ctx = std::ptr::NonNull::new_unchecked(py_z3_ctx_ptr as *mut _);
        let ctx = z3::Context::from_raw(raw_ctx);
        z3::Context::set_thread_local(&ctx);
        // No need to forget — from_raw marks the context as borrowed,
        // so Z3_del_context is never called from Rust's side.
    }

    Ok(true)
}

/// Reset the Rust thread-local Z3 context to a fresh Rust-owned context.
/// Call this before Python's Z3 context is freed (e.g., via atexit) to
/// prevent use-after-free during process shutdown.
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn reset_shared_z3_context() -> PyResult<()> {
    // Replace the thread-local with a fresh Rust-owned context.
    // The old thread-local (pointing to Python's context) has a leaked Rc
    // (refcount stays at 1 after this, ContextInternal::drop never called).
    let fresh = z3::Context::new(&z3::Config::new());
    z3::Context::set_thread_local(&fresh);
    Ok(())
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
    // log::set_logger errors on second call — fine, the first install wins
    // and from then on we only swap the filter inside our Logger.
    let _ = log::set_logger(logger);
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

/// Clear the Rust-side thread-local claripy AST translation caches.
///
/// Drops the two LRU caches in `claripy_bridge` (AST_CACHE,
/// EXPRESSION_BY_OPERANDS_PTR).
/// Used by `RustExplorationManager.cleanup()` to bound per-process
/// growth in Callable-heavy workloads where many short-lived managers
/// share the same thread (e.g. mma_howtouse's 45 invocations).
///
/// Does NOT clear the global `SymbolicIdentityRegistry` because that
/// is shared across all managers in the process; clearing it from one
/// manager would invalidate live symbol IDs held by another.
#[pyfunction]
fn clear_ast_cache() {
    crate::claripy_bridge::clear_ast_cache();
}

/// Raise a typed [`RustExecError`] for unit tests.
///
/// Lets the pytest.raises tests cover exception variants whose real-path
/// trigger is gated behind sibling beads (e.g. the 48 syscall panics in
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

/// Report whether the `libvex-ffi` build feature was compiled in.
///
/// Python wiring (`rust_manager.py`) gates the native cold-block lifting
/// flag on this so it only calls `set_native_lift_enabled(True)` on a build
/// that actually carries the FFI lifter (z087y Stage-2). Returns `false` for
/// the default build, where the interpreter setter is a no-op stub anyway.
#[pyfunction]
fn libvex_ffi_enabled() -> bool {
    cfg!(feature = "libvex-ffi")
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;

/// Register the VEX engine module with Python.
pub(crate) fn vex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Typed exception classes (angr-tkbr.3).
    crate::errors::register(m)?;
    m.add_function(pyo3::wrap_pyfunction!(_raise_typed_test_error, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(execute_irsb_for_test, m)?)?;
    m.add_class::<PythonCallbacks>()?;
    m.add_class::<LoopExecutionEvent>()?;
    m.add_class::<RustSolverContext>()?;
    // Handle-based API for claripy bypass
    m.add_class::<crate::symbolic::RustBVHandle>()?;
    // Deferred fork types
    m.add_class::<DeferredFork>()?;
    m.add_class::<BranchPolicy>()?;
    m.add_class::<ExecutionConfig>()?;
    // Rust-first state
    m.add_class::<crate::state::PyRustSimState>()?;
    // State snapshot for exploration export
    m.add_class::<crate::state::ExplorationStateSnapshot>()?;
    // Exploration manager
    crate::exploration::register_exploration(m)?;
    // Z3 context sharing
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(set_shared_z3_context, m)?)?;
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(reset_shared_z3_context, m)?)?;
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(set_z3_global_param, m)?)?;
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(get_z3_global_param, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_rust_log_level, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_size_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_names_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(arch_supported, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(clear_ast_cache, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(libvex_ffi_enabled, m)?)?;
    // Memory layout constants (single source of truth; Python imports these
    // rather than redeclaring 0x1000 etc.).
    m.add("PAGE_SIZE", crate::memory::PAGE_SIZE)?;
    m.add("PAGE_MASK", crate::memory::PAGE_MASK)?;
    // Engine version, sourced from native/angr/Cargo.toml at compile time.
    // Re-exported by Python as `angr.exploration.__rust_engine_version__` so
    // user code can branch on engine API version independently of
    // `angr.__version__`. See policy in
    // `docs/advanced-topics/rust_engine.rst` (:ref:`rust-engine-api-stability`).
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
