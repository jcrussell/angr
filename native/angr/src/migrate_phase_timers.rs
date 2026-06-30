//! Phase-attribution timers for cross-Z3-context state migration
//! (Phase 0, bead angr-t3l5o).
//!
//! Env-gated (`ANGR_MIGRATE_PHASE_TIMERS`) transient sub-timers wrapped
//! around the individual phases of [`crate::state::RustSimState::to_serialized`]
//! / [`crate::state::RustSimState::from_serialized`] and the SMT-LIB2 emit /
//! parse helpers in `symbolic/snapshot_fork_ops.rs`. Each phase accumulates
//! into a process-global atomic so the round-trip can be attributed across
//! its phases *without changing migration behaviour*.
//!
//! The counters are intentionally **process-global** (not per-manager): the
//! detach side (`to_serialized`) runs on the producer thread while the
//! reattach side (`from_serialized`) runs on a *different* thread (the
//! shadow-probe scratch thread, or a steal worker). Globals let the two
//! halves land in the same accumulator without threading a handle across the
//! channel.
//!
//! **Zero overhead when off.** The env var is read exactly once (`OnceLock`);
//! when unset, [`time_phase`] reduces to calling the closure directly (no
//! `Instant::now`, no atomic write). The gate is a single relaxed atomic load.
//!
//! These counters are surfaced through `exploration::stats_api` so
//! `run_single.py --counters-json` (with `RUST_PARALLEL_SHADOW_PROBE=1`)
//! reports them alongside `parallel_shadow_migration_ns`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cumulative ns spent emitting the solver SMT-LIB2 text dump
/// (`SymContext::dump_solver_smtlib2`) on the detach side.
pub static MIGRATE_SMTLIB2_EMIT_NS: AtomicU64 = AtomicU64::new(0);
/// Cumulative ns spent re-parsing the SMT-LIB2 text (`from_string` +
/// `get_assertions` + re-add loop) on the reattach side.
pub static MIGRATE_SMTLIB2_PARSE_NS: AtomicU64 = AtomicU64::new(0);
/// Cumulative ns spent in pure serde (json encode on detach + json decode on
/// reattach), excluding the SMT-LIB2 emit/parse which are timed separately.
pub static MIGRATE_SERDE_NS: AtomicU64 = AtomicU64::new(0);
/// Cumulative ns spent rebuilding the symbolic memory pages on reattach
/// (`SymbolicMemory::from_snapshot`) — the non-solver "leaf rebuild" work.
pub static MIGRATE_LEAF_REBUILD_NS: AtomicU64 = AtomicU64::new(0);
/// Cumulative count of *residual* (no-`RustBV`) solver assertions observed
/// across migrated states, approximated as
/// `z3_assertion_count - assumed_constraints.len()` per snapshot. Summed over
/// all migrated states; divide by `parallel_shadow_migration_states` for a
/// per-state average. ≈ 0 means the SMT-LIB2 text round-trip is pure overhead
/// (the Phase-1 lever is clean).
pub static MIGRATE_RAW_CONSTRAINT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Cumulative wall ns of every migration `to_serialized` + `from_serialized`
/// (env-gated). This is the **self-consistent denominator** for the per-phase
/// fractions: it covers exactly the same call set as the phase sub-timers,
/// unlike `parallel_shadow_migration_ns` which only sums the shadow-probe
/// round-trips (the process also serializes a handful of non-probe states —
/// found/teardown — that the phase timers see but the shadow counter does
/// not). Reported as `migrate_roundtrip_ns`.
pub static MIGRATE_ROUNDTRIP_NS: AtomicU64 = AtomicU64::new(0);

/// Read `ANGR_MIGRATE_PHASE_TIMERS` exactly once. Any non-empty value (the
/// var merely being present) enables the timers.
#[inline]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ANGR_MIGRATE_PHASE_TIMERS").is_some())
}

thread_local! {
    /// Set true for the duration of a migration `to_serialized` /
    /// `from_serialized`. Lets the SMT-LIB2 emit timer (which lives in
    /// `SymContext::to_snapshot`, a helper also reached by *non-migration*
    /// callers such as `StashManager` snapshots) fire ONLY for migration
    /// transport, keeping the per-phase fractions comparable against the
    /// shadow-probe round-trip total.
    static SERIALIZING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the current thread as in/out of a migration serialize/deserialize.
/// Returns the previous value so callers can restore it (nesting-safe).
#[inline]
pub fn set_serializing(on: bool) -> bool {
    SERIALIZING.with(|c| c.replace(on))
}

/// True when the current thread is inside a migration `to_serialized` /
/// `from_serialized`.
#[inline]
pub fn serializing() -> bool {
    SERIALIZING.with(|c| c.get())
}

/// Run `f`; when phase timers are enabled AND the current thread is inside a
/// migration serialize/deserialize ([`serializing`]), add its wall time (ns)
/// to `counter`. Scoping on `serializing()` keeps non-migration callers of the
/// shared `to_snapshot` / `from_snapshot` helpers (e.g. `StashManager`
/// snapshots) out of the attribution. When disabled this is a direct tail
/// call to `f` (no clock read, no atomic store).
#[inline]
pub fn time_phase<R>(counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
    if enabled() && serializing() {
        let t0 = std::time::Instant::now();
        let r = f();
        counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    } else {
        f()
    }
}

/// Add `n` residual (no-`RustBV`) assertions to the running count, gated on
/// the env flag so it is zero-cost when off.
#[inline]
pub fn add_raw_constraint_count(n: u64) {
    if enabled() && serializing() {
        MIGRATE_RAW_CONSTRAINT_COUNT.fetch_add(n, Ordering::Relaxed);
    }
}

/// True when both the env flag is set and we are inside a migration
/// serialize/deserialize — the precondition for the residual-count probe
/// (which calls the lazy-solver-materializing `z3_assertion_count`).
#[inline]
pub fn count_armed() -> bool {
    enabled() && serializing()
}

/// Time the whole-`f` migration round-trip half (`to_serialized` or
/// `from_serialized`) into [`MIGRATE_ROUNDTRIP_NS`] when enabled. Returns
/// `f`'s result. Pairs with the inner phase timers to give a self-consistent
/// denominator over the same call set.
#[inline]
pub fn time_roundtrip_half<R>(f: impl FnOnce() -> R) -> R {
    if enabled() {
        let t0 = std::time::Instant::now();
        let r = f();
        MIGRATE_ROUNDTRIP_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    } else {
        f()
    }
}

/// Snapshot of all phase counters (for `stats_api` surfacing):
/// `(emit_ns, parse_ns, serde_ns, leaf_rebuild_ns, raw_count, roundtrip_ns)`.
pub fn snapshot() -> (u64, u64, u64, u64, u64, u64) {
    (
        MIGRATE_SMTLIB2_EMIT_NS.load(Ordering::Relaxed),
        MIGRATE_SMTLIB2_PARSE_NS.load(Ordering::Relaxed),
        MIGRATE_SERDE_NS.load(Ordering::Relaxed),
        MIGRATE_LEAF_REBUILD_NS.load(Ordering::Relaxed),
        MIGRATE_RAW_CONSTRAINT_COUNT.load(Ordering::Relaxed),
        MIGRATE_ROUNDTRIP_NS.load(Ordering::Relaxed),
    )
}
