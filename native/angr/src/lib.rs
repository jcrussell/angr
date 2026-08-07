//! `rustylib` — the native core of angr's Rust symbolic-execution engine.
//!
//! This crate is compiled twice by one `[lib]` entry: as a `cdylib` it is the
//! CPython extension module imported as `angr.rustylib`, and as an `rlib` it is
//! linked by the in-repo `benches/`, `examples/`, `tests/` targets and the
//! `fuzz/` cargo-fuzz project. (The crate is *named* `angr` in `Cargo.toml`
//! but produces a library named `rustylib` so the Python import path works.)
//!
//! Python drives it through `angr.exploration.RustExplorationManager` (or
//! `use_rust_engine=True`); the Rust side then lifts, interprets and solves
//! without returning to the interpreter per basic block. Z3 is shared with
//! claripy — the same `libz3.so` is loaded by both sides so ASTs pass through
//! by handle rather than being re-parsed.
//!
//! # Entry points
//!
//! `rustylib()` is the `#[pymodule]` initializer. It registers the
//! per-subsystem submodules via `import_submodule`:
//!
//! - `angr.rustylib.vex_engine` — the engine proper (`engine::vex_engine`),
//!   including `RustExplorationManager`, `RustSimState` and `RustSolverContext`
//! - `angr.rustylib.segmentlist` — the CFG's `SegmentList`/`Segment`
//! - `angr.rustylib.automaton` — DFA/NFA support for
//!   `angr.analyses.typehoon.dfa`
//! - `angr.rustylib.fuzzer`, `angr.rustylib.icicle` — optional fuzzing backends
//!
//! # Module map
//!
//! - [`vex`] / `interpreter` — IRSB lifting and VEX statement/expression
//!   execution
//! - [`state`] / [`memory`] / [`stash`] — simulation state, the lazy
//!   copy-on-write memory model, and the exploration stashes
//! - [`symbolic`] / `solver` / `claripy_bridge` — the Z3 context, constraint
//!   solving, and AST import/export across the Python boundary
//! - `procedures` / `syscalls` — native SimProcedures and syscall handlers,
//!   each falling back to Python via `ProcedureError`/`SyscallError`
//! - `exploration` — the run loop, work-stealing scheduler and step core
//! - [`concretize`] — concretization strategies for symbolic addresses
//!
//! # Feature flags
//!
//! `default = ["vex-engine", "vex-engine-z3", "automaton"]`; `setup.py`
//! additionally appends `libvex-ffi`, so a stock `pip install -e .` builds with
//! native cold-block lifting.
//!
//! - `vex-engine` — the interpreter, state model and native procedures
//! - `vex-engine-z3` — adds the Z3-backed solver and the parallel scheduler
//! - `automaton` — the DFA/NFA module
//! - `libvex-ffi` — lift cold blocks through libVEX directly instead of pyvex
//! - `fuzzer` — icicle/libafl backends; `fuzzing` — widen hostile-input parsers
//!   to a crate-`pub` surface for the cargo-fuzz targets (see `fuzz_api`)
//!
//! # Further reading
//!
//! `docs/advanced-topics/rust_engine.rst` is the authoritative architecture and
//! usage overview (support matrix, `SimOption` coverage, known-slow benches).
//! See also `rust_z3_sharing.rst`, `rust_lazy_memory_design.rst`,
//! `rust_parallel_design.rst` and `rust_libvex_ffi.rst` in the same directory,
//! plus `docs/extending-angr/rust_vex_ops.rst` for adding a VEX op.

// clippy::unwrap_used/expect_used (workspace lint, angr-9ke6b guardrail) is
// scoped to production code. `cfg(test)` applies to the WHOLE crate when
// compiled as the test harness, not just individual `#[test]` fns, so this
// allow covers every `#[cfg(test)] mod tests` block and *_tests.rs file
// crate-wide. Production code still gets the lint's full force: `--all-targets`
// also compiles the plain (non-test) lib target, where `cfg(test)` is false.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
// Keep the crate's `pub` surface honest (angr-9ke6b.214).
//
// `[lib] crate-type = ["cdylib", "rlib"]` means rustc treats every `pub` item
// reachable from the crate root as externally consumable, so `dead_code` stays
// silent for all of them — it measured 0 hits across the whole tree while 468
// `unreachable_pub` sites existed. Dropping `rlib` is not an option (`benches/`,
// `examples/`, `tests/` and the `fuzz/` cargo-fuzz project all link it), so the
// lever is the other direction: only the modules an external target actually
// imports stay `pub mod` (`automaton`, `concretize`, `fuzz_api`, `memory`,
// `stash`, `state`, `symbolic`, `vex` — re-derive with
// `grep -rhoE "rustylib::[a-z_]+" tests benches examples fuzz`), everything else
// is `pub(crate) mod`. Narrowing the items inside those modules is what lets
// `dead_code` see them at all; this warn keeps a new over-broad `pub` from
// silently re-opening the hole.
//
// Caveat (angr-9ke6b.50): pyo3's non-`multiple-pymethods` codegen *masked* this
// lint on every `#[pymethods]` method and on the `#[pyclass]` structs those
// blocks belong to. Turning `multiple-pymethods` on unmasks 391 such sites
// crate-wide. Those items are reached from Python, not from Rust, so their Rust
// visibility is decorative — each affected `#[pymethods]`/`#[pyclass]` carries a
// scoped `#[allow(unreachable_pub, reason = ...)]` that restores exactly the
// pre-feature coverage rather than silencing the lint crate-wide.
#![warn(unreachable_pub)]

// Dev-only public surface for the cargo-fuzz targets under `fuzz/`
// (angr-qwyti.9). Re-exports the pure hostile-input parsers so a fuzz binary
// can reach them without depending on `pub(super)` internals. Gated behind
// `fuzzing`, so a stock build never sees it.
#[cfg(feature = "fuzzing")]
pub mod fuzz_api {
    #[cfg(feature = "vex-engine")]
    pub use crate::procedures::format_common::{parse_length_modifier, parse_width_digits};
    #[cfg(feature = "vex-engine-z3")]
    pub use crate::symbolic::fuzz_exports::{
        parse_binary_to_bytes, parse_decimal_to_bytes, parse_hex_to_bytes,
        parse_wide_binary_low128, parse_wide_hex_low128,
    };
}

// Conditional compilation for fuzzer module (requires optional deps)
#[cfg(feature = "fuzzer")]
pub(crate) mod fuzzer;
#[cfg(feature = "fuzzer")]
pub(crate) mod icicle;

#[cfg(feature = "automaton")]
pub mod automaton;
pub(crate) mod segmentlist;

// VEX Engine modules (requires vex-engine feature, enabled by default)
#[cfg(feature = "vex-engine")]
pub(crate) mod arch;
#[cfg(feature = "vex-engine")]
pub(crate) mod callbacks;
#[cfg(feature = "vex-engine")]
pub(crate) mod claripy_bridge;
#[cfg(feature = "vex-engine")]
pub mod concretize;
#[cfg(feature = "vex-engine")]
pub(crate) mod engine;
#[cfg(feature = "vex-engine")]
pub(crate) mod errors;
#[cfg(feature = "vex-engine")]
pub(crate) mod exploration;
#[cfg(feature = "vex-engine")]
pub(crate) mod gil_profile;
#[cfg(feature = "vex-engine")]
pub(crate) mod interpreter;
#[cfg(feature = "vex-engine")]
pub mod memory;
pub(crate) mod migrate_phase_timers;
#[cfg(feature = "vex-engine")]
pub(crate) mod procedures;
#[cfg(feature = "vex-engine")]
pub(crate) mod solver;
#[cfg(feature = "vex-engine")]
pub mod stash;
#[cfg(feature = "vex-engine")]
pub mod state;
pub mod symbolic;
#[cfg(feature = "vex-engine")]
pub(crate) mod syscalls;
#[cfg(feature = "vex-engine")]
pub mod vex;

use pyo3::prelude::*;

/// Wrap a value in an `Arc` whose inner type is deliberately not `Send`/`Sync`.
///
/// The engine's core shared types (`RustBV`, Z3 AST handles, `FileDescriptor`)
/// are `!Send` by design, and that is what makes the suppression sound — NOT
/// single-threadedness. The engine does step states on real OS worker threads
/// (`scheduler::pool::worker_thread`, spawned as `angr-worker-N`); what keeps these
/// `Arc`s confined to one thread is that the only cross-thread transport for a
/// state is `StateMigrationPayload` (`state/migration.rs`), which is
/// `Send`-by-construction and guarded at compile time by an `assert_send::<..>()`
/// check. There is no `unsafe impl Send`/`Sync` in production code, so an
/// `arc_shared` `Arc` cannot escape its worker except by round-tripping through
/// `detach_for_migration`/`reattach`, which rebuilds the shared state on the
/// destination thread. Contributors adding a new cross-thread share must go
/// through that payload — do not widen this suppression to cover an `Arc` that
/// is genuinely handed between threads.
///
/// The `Arc`s over them are load-bearing: they give O(1) copy-on-write
/// sharing across `fork()`/snapshot siblings, which `Rc` could not without
/// leaking `!Send` through the public fork/snapshot API surface. Clippy's
/// `arc_with_non_send_sync` flags every such `Arc::new` as a design smell;
/// routing them through this generic helper documents the decision in ONE place
/// and collapses the 7 scattered per-fn `#[allow]`s into this single site.
/// The lone `#[allow]` below is the single, deliberate suppression that
/// replaces the former per-call-site cluster. See bd bead angr-inieg.1.
///
/// Retained without the engine: every caller lives under `vex-engine`
/// (angr-sqfj8.139).
#[cfg_attr(not(feature = "vex-engine"), allow(dead_code))]
#[allow(clippy::arc_with_non_send_sync)]
#[inline]
pub(crate) fn arc_shared<T>(value: T) -> std::sync::Arc<T> {
    std::sync::Arc::new(value)
}

/// Build and register a submodule under the `angr.rustylib` package.
fn import_submodule(
    m: &Bound<'_, PyModule>,
    name: &str,
    import_func: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = m.py();
    let submodule = PyModule::new(py, name)?;
    import_func(&submodule)?;

    let sys_modules = PyModule::import(py, "sys")?.getattr("modules")?;
    sys_modules.set_item(format!("angr.rustylib.{name}"), submodule.clone())?;

    m.add_submodule(&submodule)?;
    Ok(())
}

#[pymodule]
fn rustylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Fuzzer modules (optional)
    #[cfg(feature = "fuzzer")]
    {
        import_submodule(m, "fuzzer", fuzzer::fuzzer)?;
        import_submodule(m, "icicle", icicle::icicle)?;
    }

    // Segmentlist (always available)
    import_submodule(m, "segmentlist", segmentlist::segmentlist)?;
    #[cfg(feature = "automaton")]
    import_submodule(m, "automaton", automaton::automaton)?;
    m.add_class::<segmentlist::Segment>()?;
    m.add_class::<segmentlist::SegmentList>()?;

    // VEX Engine module (enabled by default)
    #[cfg(feature = "vex-engine")]
    import_submodule(m, "vex_engine", engine::vex_engine)?;

    Ok(())
}
