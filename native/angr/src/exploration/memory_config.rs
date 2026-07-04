//! Memory and VEX configuration for `RustExplorationManager`.
//!
//! Groups memory-init and VEX-lifting knobs that the manager propagates
//! to per-state and per-interpreter contexts on each step:
//!
//! * `zero_fill_unconstrained` — fill unconstrained reads with zero.
//! * `concretizer_config` — address concretization strategies.
//! * `vex_opt_level` — global VEX optimization level (None = pyvex default).
//! * `vex_opt_level_overrides` — per-address optimization overrides.
//!
//! Same pattern as `ProfilingCollector` / `ConstraintSolver`: `pub(crate)`
//! direct field access by design — callers read/write inner fields through
//! a thin delegation. Do NOT add helper methods as a separate cleanup; the
//! parent angr-4j5u was deferred multiple times for cosmetic gains.

use rustc_hash::FxHashMap;

/// Memory and VEX-lifting knobs propagated from the manager to each state
/// and interpreter on construction / per-step.
#[derive(Debug, Default)]
pub(crate) struct MemoryConfiguration {
    /// When true, fill unconstrained memory reads with zero instead of
    /// fresh symbolic values.
    pub(crate) zero_fill_unconstrained: bool,
    /// Address concretization configuration, propagated to each
    /// engine/interpreter via `set_concretizer`.
    pub(crate) concretizer_config: crate::concretize::AddressConcretizer,
    /// VEX optimization level (0-3). None = use pyvex default (typically 1).
    pub(crate) vex_opt_level: Option<i32>,
    /// Per-address VEX optimization level overrides.
    pub(crate) vex_opt_level_overrides: FxHashMap<u64, i32>,
    /// Enable native (in-process) libVEX cold-block lifting on each
    /// interpreter. Only meaningful on a `libvex-ffi` build — the default
    /// build's `VEXInterpreter::set_native_lift_enabled` is a no-op stub.
    /// Python gates this on `libvex_ffi_enabled()` + AMD64 (z087y Stage-2).
    pub(crate) native_lift_enabled: bool,
}
