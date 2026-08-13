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
// Test-support rot in the no-z3 combos (angr-c7xno.99). Dozens of `#[test]` fns
// are individually `#[cfg(feature = "vex-engine-z3")]`-gated because they drive
// `add_constraint` / `to_z3_ast`; compiling the test harness *without* z3
// therefore strands their shared helpers, fixture consts and imports as
// unused. Those are not real rot — they are live in every build anyone ships
// or benchmarks — so gating each helper individually would be ~30 `cfg`s that
// must be re-audited whenever a test moves. Scope the two lints off for
// test-cfg no-z3 builds only: the default (z3) build, which is what CI's
// `rust_check` clippy gate and `cargo test` run, keeps their full force over
// the exact same code.
#![cfg_attr(
    all(test, not(feature = "vex-engine-z3")),
    allow(dead_code, unused_imports)
)]
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

/// Declare a test-only submodule that lives in a sibling file.
///
/// Every `#[cfg(test)] mod tests;` in this crate points at an out-of-line
/// `*_tests.rs` (or a `*_tests/` directory of scoped test files) and has to
/// re-`allow` the two test-only clippy lints, because each test module's own
/// `deny` overrides the crate-wide `cfg_attr(test, allow(..))` at the top of
/// this file. That is four attributes of identical boilerplate per site,
/// repeated ~120 times crate-wide (bd angr-12jjk.16); this macro is the single
/// place the justification wording lives.
///
/// ```ignore
/// test_submod!("stash_tests.rs" => tests);        // explicit `#[path]`
/// test_submod!(tests_core);                       // default `tests_core.rs`
/// // z3-only test files, so the no-z3 `cargo test` combos still compile:
/// test_submod!(z3 "context_tests/constraints.rs" => context_tests_constraints);
/// test_submod!(z3 tests_int_arith);
/// ```
///
/// Defined before the crate's `mod` declarations so legacy textual macro scope
/// reaches every module; invoke it unqualified.
///
/// It is the convention for **every** out-of-line test submodule, not just the
/// ones where the `allow` is load-bearing. The payload only bites under a host
/// module's own `#![deny(clippy::unwrap_used, clippy::expect_used)]` (today:
/// `exploration::shadow_probe`), so the hand-written spelling compiles fine
/// everywhere else — which is exactly why three audit rounds each found a
/// two-site subset of the same ~60-site divergence and called it fixed
/// (angr-c7xno.48, angr-03vl4.8, angr-03vl4.89). The last of those swept the
/// remainder, so the invariant a future audit checks is now a simple one:
/// outside this macro's own definition, `native/angr/src/**` contains no
/// `#[cfg(test)] #[path = ...] mod ...;`.
macro_rules! test_submod {
    ($file:literal => $name:ident) => {
        #[cfg(test)]
        #[path = $file]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
    ($name:ident) => {
        #[cfg(test)]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
    // Gated on vex-engine-z3 (bd angr-cagbn): test files that drive
    // `SymContext::add_constraint` / Z3AstPtr, which only exist with z3.
    (z3 $file:literal => $name:ident) => {
        #[cfg(all(test, feature = "vex-engine-z3"))]
        #[path = $file]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
    (z3 $name:ident) => {
        #[cfg(all(test, feature = "vex-engine-z3"))]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
}

/// Take a fallible value's success payload, or log the failure and fall back to
/// `$default` — the one-line spelling of the `SILENT(cat-x)` convention.
///
/// CLAUDE.md's silent-fallback rules say a site that discards an error and
/// continues with a degraded result must carry a `// SILENT(cat-a|b|c)` tag,
/// and that `cat-c` (wrong-answer risk) must *also* `log::warn!`. Writing that
/// out by hand is four lines of `unwrap_or_else` boilerplate, so the tempting
/// silent form (`.unwrap_or(0)`, `let _ = ...`) keeps winning — three fresh
/// instances landed in one review round (angr-91vj9.6). This macro makes the
/// correct form the shorter one: the category picks the log level, and the
/// invocation itself is the greppable tag (`tools/audit_silent_fallback.py`
/// accepts `silent_default!(cat_x, ...)` in place of the comment).
///
/// `cat_b` logs at `debug` (fallback with loss), `cat_c` at `warn`
/// (wrong-answer risk). There is deliberately no `cat_a` arm: expected control
/// flow needs no log, so it stays a plain `match`/`unwrap_or` with a comment.
/// An unrecognized category is a compile error rather than a silent downgrade.
///
/// ```ignore
/// // Option<T>: `None` is the failure, and there is nothing to `Display`.
/// silent_default!(cat_c, maybe_symbolic_addr, 0, "address unavailable ({context})");
///
/// // Result<T, E>: `|err|` names a binding for the error, in scope for the
/// // message. Not mentioning it is an unused-variable error under the crate's
/// // `-D warnings`, so the cause cannot be dropped. (The binder must be
/// // written by the caller rather than injected by the macro: macro_rules
/// // hygiene would put a macro-defined `err` out of the message's scope.)
/// silent_default!(cat_b, parse(s), Default::default(), |err| "bad input: {err}");
/// ```
///
/// Defined before the crate's `mod` declarations so legacy textual macro scope
/// reaches every module; invoke it unqualified.
#[allow(
    unused_macros,
    reason = "every current call site sits behind `vex-engine`, so the \
              `--no-default-features` combos `make check-no-z3` gates compile \
              the definition with no user. Defining it unconditionally keeps \
              the convention reachable from feature-independent modules too"
)]
macro_rules! silent_default {
    // `Result<T, E>` form. Listed first: its `|err|` prefix would otherwise be
    // swallowed by the `$($msg:tt)+` of the `Option` arm below.
    ($cat:ident, $fallible:expr, $default:expr, |$err:ident| $($msg:tt)+) => {
        match $fallible {
            Ok(value) => value,
            Err($err) => {
                silent_default!(@log $cat, $($msg)+);
                $default
            }
        }
    };
    // `Option<T>` form.
    ($cat:ident, $fallible:expr, $default:expr, $($msg:tt)+) => {
        match $fallible {
            Some(value) => value,
            None => {
                silent_default!(@log $cat, $($msg)+);
                $default
            }
        }
    };
    // Internal: the category selects the log level. Keep in sync with the
    // cat-a/b/c definitions in CLAUDE.md's "Silent-fallback tagging" section.
    (@log cat_b, $($msg:tt)+) => { log::debug!($($msg)+) };
    (@log cat_c, $($msg:tt)+) => { log::warn!($($msg)+) };
}

test_submod!("silent_default_tests.rs" => silent_default_tests);

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

/// Shared boundary-value tables for the integer-overflow/wraparound proactive
/// test sweep (Harness 6, see the module doc for the full rationale). Not
/// tied to any one subsystem — call sites span `memory`, `state::filesystem`,
/// `syscalls`, `procedures`, `interpreter` and `symbolic`, all of which sit
/// behind `vex-engine` — so it is declared once here at the crate root
/// rather than nested under any of them.
#[cfg(all(test, feature = "vex-engine"))]
pub(crate) mod test_boundary_values;

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
