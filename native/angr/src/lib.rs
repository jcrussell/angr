//! `rustylib` — the native core of angr's Rust symbolic-execution engine.
//!
//! This crate is compiled twice by one `[lib]` entry: as a `cdylib` it is the
//! CPython extension module imported as `angr.rustylib`, and as an `rlib` it is
//! linked by the in-repo `benches/`, `examples/`, `tests/` targets and the
//! `fuzz/` cargo-fuzz project. (The crate is *named* `angr` in `Cargo.toml`
//! but produces a library named `rustylib` so the Python import path works.)
//!
//! Python drives it through `angr.exploration.RustExplorationManager` (or
//! `factory.simulation_manager(engine="rust")`); the Rust side then lifts,
//! interprets and solves without returning to the interpreter per basic block.
//! Z3 is shared with claripy — the same `libz3.so` is loaded by both sides so
//! ASTs pass through by handle rather than being re-parsed.
//!
//! # Entry points
//!
//! `rustylib()` is the `#[pymodule]` initializer. It registers the
//! per-subsystem submodules via `import_submodule`:
//!
//! - `angr.rustylib.vex_engine` — the engine proper (`engine::vex_engine`),
//!   including `RustExplorationManager`, `RustSimState` and `RustSolverContext`
//! - `angr.rustylib.vex_ccall` — the VEX clean-call helpers the Python
//!   engine's ccall dispatch tries before `ccall.py` (`vex_ccall`)
//! - `angr.rustylib.vex_irop` — the integer VEX ops the Python engine's op
//!   dispatch tries before `irop.py` (`vex_irop`)
//! - `angr.rustylib.segmentlist` — the CFG's `SegmentList`/`Segment`
//! - `angr.rustylib.automaton` — DFA/NFA support for
//!   `angr.analyses.typehoon.dfa`
//! - `angr.rustylib.fuzzer`, `angr.rustylib.icicle` — optional fuzzing backends
//!
//! # Module map
//!
//! - [`vex`] / `interpreter` — IRSB lifting and VEX statement/expression
//!   execution; `libvex_lifter` is the in-process libVEX FFI lifter the
//!   interpreter reaches through `vex::VEXLifter`
//! - [`state`] / [`memory`] / [`stash`] — simulation state, the lazy
//!   copy-on-write memory model, and the exploration stashes;
//!   `state_bridge` is the state's PyO3 face (`RustSimState`,
//!   `ExplorationStateSnapshot`)
//! - [`symbolic`] / `solver` / `claripy_bridge` — the Z3 context, constraint
//!   solving, and AST import/export across the Python boundary
//! - `procedures` / `syscalls` — native SimProcedures and syscall handlers,
//!   each falling back to Python via `ProcedureError`/`SyscallError`
//! - `exploration` — the run loop, work-stealing scheduler and step core
//! - [`symbolic::concretize`] — concretization strategies for symbolic addresses
//!
//! # Feature flags
//!
//! Everything is built unconditionally. The one feature, `fuzzing`, widens
//! the hostile-input parsers to a crate-`pub` surface for the cargo-fuzz
//! targets (see `fuzz_api`).
//!
//! # Panic policy
//!
//! The shipped `.so` is built with `[profile.release] panic = "abort"`
//! (workspace `Cargo.toml`, angr-1cue), and every module that handles guest
//! or Python-supplied data carries `#![deny(clippy::unwrap_used,
//! clippy::expect_used)]`. Three consequences follow, and the per-module
//! "Panic policy" notes point here rather than restating them:
//!
//! - **A poisoned lock always follows an earlier panic.** A lock is poisoned
//!   only when a thread *unwinds* out of a live `MutexGuard`, i.e. after a
//!   panic that has already failed its step. Under the shipped
//!   `panic = "abort"` that first panic ends the process, so every
//!   `.lock().expect("… poisoned")` is an invariant label, not an error path,
//!   and a worker thread cannot "die" mid-wave. In a `panic = "unwind"` build
//!   (cargo's default, and upstream's) the `expect` turns the poison into a
//!   second panic, which pyo3 raises as `PanicException`: still loud, still
//!   never a silent wrong answer.
//! - **A genuine bug surfaces as a process abort, by design.** There is no
//!   "surface as a Python exception" path to build for those guards. A site
//!   that *is* fallible on real input returns a domain error (`ProcedureError`,
//!   `SyscallError`, `MemoryError`, …) and falls back to Python; see the
//!   convention in `errors.rs`. Never trade a loud panic for a silent wrong
//!   answer: a guard that would otherwise turn into an early `None` and drop
//!   states stays a panic.
//! - **Under `cargo test` the argument does not hold, and that is expected.**
//!   Cargo forces `panic = "unwind"` on test binaries, so a panic inside a held
//!   guard poisons the lock and a *second*, cascaded `… poisoned` panic
//!   follows. Ignore the cascade and look for the earliest panic in the
//!   output.
//!
//! Each surviving `expect` sits under a narrow
//! `#[allow(clippy::expect_used, reason = "…")]` whose reason names the local
//! guard that proves it; if no such reason can be written, the code needs a
//! domain error instead.
//!
//! # Further reading
//!
//! `docs/advanced-topics/rust_engine.rst` is the authoritative usage
//! reference (SimOption coverage, inspect dispatch, techniques, errors);
//! `rust_engine_decisions.rst` records rejected alternatives and
//! `rust_build.rst` how to build, test and profile. See also
//! `rust_z3_sharing.rst`, `rust_lazy_memory_design.rst`,
//! `rust_parallel_design.rst` and `rust_libvex_ffi.rst` in the same directory,
//! plus `docs/extending-angr/rust_vex_ops.rst` for adding a VEX op and
//! `docs/extending-angr/simprocedures.rst` for a native SimProcedure.

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
// silent for all of them, however many `unreachable_pub` sites exist. Dropping `rlib` is not an option (`benches/`,
// `examples/`, `tests/` and the `fuzz/` cargo-fuzz project all link it), so the
// lever is the other direction: only the modules an external target actually
// imports stay `pub mod` (`automaton`, `fuzz_api`, `memory`,
// `stash`, `state`, `symbolic`, `vex` — re-derive with
// `grep -rhoE "rustylib::[a-z_]+" tests benches examples fuzz`), everything else
// is `pub(crate) mod`. Narrowing the items inside those modules is what lets
// `dead_code` see them at all; this warn keeps a new over-broad `pub` from
// silently re-opening the hole.
//
// Caveat (angr-9ke6b.50): pyo3's non-`multiple-pymethods` codegen *masked* this
// lint on every `#[pymethods]` method and on the `#[pyclass]` structs those
// blocks belong to; with `multiple-pymethods` on, the lint sees them. Those items are reached from Python, not from Rust, so their Rust
// visibility is decorative — each affected `#[pymethods]`/`#[pyclass]` carries a
// scoped `#[allow(unreachable_pub, reason = ...)]` that restores exactly the
// pre-feature coverage rather than silencing the lint crate-wide.
#![warn(unreachable_pub)]

// The VEX layer is the `angr_vex` crate, re-exported as `vex` so
// `crate::vex::` and `rustylib::vex::` paths name its items. It also
// `#[macro_export]`s `test_submod!` and `silent_default!` (their docs live in
// native/angr-vex/src/lib.rs); `#[macro_use]` ahead of the `mod` declarations
// keeps every invocation unqualified. One `pub extern crate` does both: a
// `pub use` of a `#[macro_use]` extern crate is rejected (E0365).
#[macro_use]
pub extern crate angr_vex as vex;

// The crate-root helpers live in angr-symex-core (angr-mx00p.51.11); these
// re-exports keep `crate::arc_shared` and friends resolving.
pub(crate) use angr_symex_core::{arc_shared, duration_ns, elapsed_ns};

// Dev-only public surface for the cargo-fuzz targets under `fuzz/`
// (angr-qwyti.9). Re-exports the pure hostile-input parsers so a fuzz binary
// can reach them without depending on `pub(super)` internals. Gated behind
// `fuzzing`, so a stock build never sees it.
#[cfg(feature = "fuzzing")]
pub mod fuzz_api {
    pub use crate::procedures::format_common::{parse_length_modifier, parse_width_digits};
}

// Modules shared with upstream angr (always built). They are kept
// byte-identical to upstream, which enables neither the fork's stricter
// clippy lints nor its rustdoc gate, so both are relaxed for them here.
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod ailment;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod automaton;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod cfg_graph;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod claripy;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod function_graph;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod fuzzer;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod icicle;
#[allow(
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::let_underscore_must_use,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::redundant_explicit_links,
    rustdoc::invalid_html_tags
)]
pub mod segmentlist;

// VEX Engine modules
pub(crate) mod arch;
pub(crate) mod callbacks;
pub(crate) mod claripy_bridge;
pub(crate) mod engine;
pub(crate) mod errors;
pub(crate) mod exploration;
pub(crate) mod gil_profile;
pub(crate) mod interpreter;
pub(crate) mod libvex_lifter;
pub mod memory;
pub(crate) mod procedures;
pub(crate) mod solver;
pub mod stash;
pub mod state;
pub(crate) mod state_bridge;
pub use angr_symex_core::symbolic;
pub(crate) mod syscalls;
pub(crate) mod vex_ccall;
pub(crate) mod vex_irop;

// The boundary-value tables live in angr-symex-core; see its
// `test_boundary_values` module doc.
#[cfg(test)]
pub(crate) use angr_symex_core::test_boundary_values;

use pyo3::prelude::*;

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

// `gil_used = true`: pyo3 0.28 flipped the default to "safe on free-threaded
// Python". This extension is not: every `SymContext` captures a thread-local
// Z3 context, `RustSolverContext` is `#[pyclass(unsendable)]`, and the
// scheduler serializes state across threads by design
// (docs/advanced-topics/rust_parallel_design.rst). On a free-threaded
// interpreter this makes CPython re-enable the GIL for the process instead of
// letting two Python threads race into one Z3 context.
#[pymodule(gil_used = true)]
fn rustylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    import_submodule(m, "fuzzer", fuzzer::fuzzer)?;
    import_submodule(m, "icicle", icicle::icicle)?;
    import_submodule(m, "segmentlist", segmentlist::segmentlist)?;
    import_submodule(m, "automaton", automaton::automaton)?;
    import_submodule(m, "ailment", ailment::ailment)?;
    import_submodule(m, "cfg_graph", cfg_graph::cfg_graph)?;
    import_submodule(m, "function_graph", function_graph::function_graph)?;

    // claripy registers its own `angr.rustylib.claripy.*` submodule tree.
    let py = m.py();
    let claripy_module = PyModule::new(py, "angr.rustylib.claripy")?;
    claripy::claripy(py, &claripy_module)?;
    PyModule::import(py, "sys")?
        .getattr("modules")?
        .set_item("angr.rustylib.claripy", claripy_module.clone())?;
    m.add("claripy", claripy_module)?;

    m.add_class::<segmentlist::Segment>()?;
    m.add_class::<segmentlist::SegmentList>()?;

    // VEX Engine module (enabled by default)
    import_submodule(m, "vex_engine", engine::vex_engine)?;
    // The VEX ccall helpers behind the Python engine's ccall dispatch.
    import_submodule(m, "vex_ccall", vex_ccall::vex_ccall)?;
    // The integer VEX ops behind the Python engine's op dispatch.
    import_submodule(m, "vex_irop", vex_irop::vex_irop)?;

    Ok(())
}
