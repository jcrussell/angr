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
// imports stay `pub mod` (`automaton`, `concretize`, `fuzz_api`, `memory`,
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

/// Declare a test-only submodule that lives in a sibling file.
///
/// Every `#[cfg(test)] mod tests;` in this crate points at an out-of-line
/// `*_tests.rs` (or a `*_tests/` directory of scoped test files) and has to
/// re-`allow` the two test-only clippy lints, because each test module's own
/// `deny` overrides the crate-wide `cfg_attr(test, allow(..))` at the top of
/// this file. That is four attributes of identical boilerplate per site,
/// repeated at every out-of-line test module (angr-12jjk.16); this macro is
/// the single place the justification wording lives.
///
/// ```ignore
/// test_submod!("stash_tests.rs" => tests);        // explicit `#[path]`
/// test_submod!(tests_core);                       // default `tests_core.rs`
/// test_submod!("context_tests/constraints.rs" => context_tests_constraints);
/// test_submod!(tests_int_arith);
/// ```
///
/// Defined before the crate's `mod` declarations so legacy textual macro scope
/// reaches every module; invoke it unqualified.
///
/// It is the convention for **every** out-of-line test submodule, not just the
/// ones where the `allow` is load-bearing. The payload only bites under a host
/// module's own `#![deny(clippy::unwrap_used, clippy::expect_used)]` (for
/// example `exploration::shadow_probe`), so the hand-written spelling compiles fine
/// everywhere else — which is exactly why a partial sweep keeps looking
/// complete (angr-c7xno.48, angr-03vl4.8, angr-03vl4.89). The invariant a
/// future audit checks is therefore a simple one:
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
}

/// Take a fallible value's success payload, or log the failure and fall back to
/// `$default` — the one-line spelling of the `SILENT(cat-x)` convention.
///
/// The silent-fallback rules (`tools/audit_silent_fallback.py`, which gates
/// them in CI) say a site that discards an error and
/// continues with a degraded result must carry a `// SILENT(cat-a|b|c)` tag,
/// and that `cat-c` (wrong-answer risk) must *also* `log::warn!`. Writing that
/// out by hand is four lines of `unwrap_or_else` boilerplate, so the tempting
/// silent form (`.unwrap_or(0)`, `let _ = ...`) keeps winning (angr-91vj9.6).
/// This macro makes the
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
    // cat-a/b/c definitions in the `tools/audit_silent_fallback.py` docstring.
    (@log cat_b, $($msg:tt)+) => { log::debug!($($msg)+) };
    (@log cat_c, $($msg:tt)+) => { log::warn!($($msg)+) };
}

test_submod!("silent_default_tests.rs" => silent_default_tests);

/// Nanoseconds in `d`, saturating at `u64::MAX` instead of wrapping.
///
/// Every timing counter in this crate is a `u64` (they accumulate in
/// `AtomicU64`/`u64` fields and are exported to Python through `stats()`), but
/// `Duration::as_nanos` returns `u128`. The obvious `as u64` narrowing is
/// silent even under `[profile.release-checked]` — `as` never panics — so a
/// span past `u64::MAX` ns would report a near-zero duration rather than a huge
/// one. Saturating keeps the impossible case implausible-looking instead of
/// wrong, and gives the narrowing one documented home rather than one
/// undocumented cast per timer (angr-0jh0j.80). Per the overflow rules
/// (`invariant-overflow-fix-refuse-not-saturate-identities`) a *size*-like
/// quantity must refuse rather than
/// saturate; a monotonic timer has no such identity to corrupt, and a pinned
/// `u64::MAX` is the loudest thing a counter can say.
///
/// The bound is ~584 years, so no real process reaches it; this is
/// defensiveness and documentation, not a live bug fix.
pub(crate) fn duration_ns(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Nanoseconds elapsed since `start`, saturating at `u64::MAX`.
///
/// The `Instant` spelling of [`duration_ns`], which carries the rationale.
/// Timing sites should call this rather than narrowing `as_nanos()`'s `u128`
/// with a bare `as u64`.
pub(crate) fn elapsed_ns(start: std::time::Instant) -> u64 {
    duration_ns(start.elapsed())
}

// Dev-only public surface for the cargo-fuzz targets under `fuzz/`
// (angr-qwyti.9). Re-exports the pure hostile-input parsers so a fuzz binary
// can reach them without depending on `pub(super)` internals. Gated behind
// `fuzzing`, so a stock build never sees it.
#[cfg(feature = "fuzzing")]
pub mod fuzz_api {
    pub use crate::procedures::format_common::{parse_length_modifier, parse_width_digits};
    pub use crate::procedures::getopt::fuzz_exports::parse_optstring;
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
pub mod concretize;
pub(crate) mod engine;
pub(crate) mod errors;
pub(crate) mod exploration;
pub(crate) mod gil_profile;
pub(crate) mod interpreter;
pub mod memory;
pub(crate) mod procedures;
pub(crate) mod solver;
pub mod stash;
pub mod state;
pub mod symbolic;
pub(crate) mod syscalls;
pub mod vex;

/// Shared boundary-value tables for the integer-overflow/wraparound proactive
/// test sweep (Harness 6, see the module doc for the full rationale). Not
/// tied to any one subsystem — call sites span `memory`, `state::filesystem`,
/// `syscalls`, `procedures`, `interpreter`, `symbolic` and `segmentlist` — so
/// it is declared once here at the crate root rather than nested under any of
/// them.
///
/// `segmentlist`'s own tests sweep `boundary_addresses` over
/// `SegmentList::{occupy,release}` too (angr-fs8kb.87).
#[cfg(test)]
pub(crate) mod test_boundary_values;

use pyo3::prelude::*;

/// Wrap a value in an `Arc` whose inner type is deliberately not `Send`/`Sync`.
///
/// The engine's core shared types (`RustBV`, Z3 AST handles, `FileDescriptor`)
/// are `!Send` by design, and that is what makes the suppression sound — NOT
/// single-threadedness. The engine does step states on real OS worker threads
/// (`exploration::scheduler_pool::worker_thread`, spawned as `angr-worker-N`);
/// what keeps these `Arc`s confined to one thread is that the only cross-thread
/// transport for a state is `StateMigrationPayload` (`state/migration.rs`), which is
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

    Ok(())
}
