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
pub mod fuzzer;
#[cfg(feature = "fuzzer")]
pub mod icicle;

#[cfg(feature = "automaton")]
pub mod automaton;
pub mod segmentlist;

// VEX Engine modules (requires vex-engine feature, enabled by default)
#[cfg(feature = "vex-engine")]
pub mod arch;
#[cfg(feature = "vex-engine")]
pub mod callbacks;
#[cfg(feature = "vex-engine")]
pub mod claripy_bridge;
#[cfg(feature = "vex-engine")]
pub mod concretize;
#[cfg(feature = "vex-engine")]
pub mod engine;
#[cfg(feature = "vex-engine")]
pub mod errors;
#[cfg(feature = "vex-engine")]
pub mod exploration;
#[cfg(feature = "vex-engine")]
pub mod gil_profile;
#[cfg(feature = "vex-engine")]
pub mod interpreter;
#[cfg(feature = "vex-engine")]
pub mod memory;
pub mod migrate_phase_timers;
#[cfg(feature = "vex-engine")]
pub mod procedures;
#[cfg(feature = "vex-engine")]
pub mod solver;
#[cfg(feature = "vex-engine")]
pub mod stash;
#[cfg(feature = "vex-engine")]
pub mod state;
pub mod symbolic;
#[cfg(feature = "vex-engine")]
pub mod syscalls;
#[cfg(feature = "vex-engine")]
pub mod vex;

use pyo3::prelude::*;

/// Wrap a value in an `Arc` whose inner type is deliberately not `Send`/`Sync`.
///
/// The engine's core shared types (`RustBV`, Z3 AST handles, `FileDescriptor`)
/// are `!Send` by design, and that is what makes the suppression sound — NOT
/// single-threadedness. The engine does step states on real OS worker threads
/// (`scheduler::worker_thread`, spawned as `angr-worker-N`); what keeps these
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
