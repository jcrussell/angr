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
    SERIALIZING.with(std::cell::Cell::get)
}

/// Run `f`; when phase timers are enabled AND the current thread is inside a
/// migration serialize/deserialize ([`serializing`]), add its wall time (ns)
/// to `counter`. Scoping on `serializing()` keeps non-migration callers of the
/// shared `to_snapshot` / `from_snapshot` helpers (e.g. `StashManager`
/// snapshots) out of the attribution. When disabled this is a direct tail
/// call to `f` (no clock read, no atomic store).
#[inline]
pub fn time_phase<R>(counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
    time_phase_armed(enabled() && serializing(), counter, f)
}

/// Pure timing core: when `armed`, add `f`'s wall time (ns) to `counter`;
/// otherwise a direct tail call. Split out from [`time_phase`] so the
/// firing/no-firing decision can be exercised deterministically without the
/// process-global env gate (mirrors `gil_profile`'s `enabled: bool` guards).
#[inline]
fn time_phase_armed<R>(armed: bool, counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
    if armed {
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
    add_raw_constraint_count_armed(enabled() && serializing(), n);
}

/// Pure core of [`add_raw_constraint_count`], split out so the gate can be
/// driven directly in tests.
#[inline]
fn add_raw_constraint_count_armed(armed: bool, n: u64) {
    if armed {
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
    time_roundtrip_half_armed(enabled(), f)
}

/// Pure core of [`time_roundtrip_half`], split out so the gate can be driven
/// directly in tests.
#[inline]
fn time_roundtrip_half_armed<R>(armed: bool, f: impl FnOnce() -> R) -> R {
    if armed {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that assert *exact* deltas on the process-global
    /// counters — `cargo test` runs test fns on parallel threads, so two
    /// tests mutating the same static would race each other's before/after
    /// arithmetic. Tests using only local atomics or `>`-comparisons don't
    /// need it.
    static GLOBALS_LOCK: Mutex<()> = Mutex::new(());

    /// Spin until the monotonic clock advances by at least `min_ns`, so a
    /// timed closure banks a strictly-positive delta regardless of clock
    /// resolution (mirrors `gil_profile::tests::busy_ns`).
    fn busy_ns(min_ns: u64) {
        let t0 = std::time::Instant::now();
        let mut spins: u64 = 0;
        while (t0.elapsed().as_nanos() as u64) < min_ns {
            spins = spins.wrapping_add(1);
            std::hint::black_box(spins);
        }
    }

    #[test]
    fn set_serializing_is_nesting_safe_and_returns_prev() {
        // Baseline off.
        assert!(!serializing());
        let prev_outer = set_serializing(true);
        assert!(!prev_outer, "outer previous should be the false baseline");
        assert!(serializing());
        // Nested arm returns the outer's `true` so it can restore correctly.
        let prev_inner = set_serializing(true);
        assert!(prev_inner);
        set_serializing(prev_inner);
        assert!(
            serializing(),
            "restoring the nested arm keeps us serializing"
        );
        // Restore the outer baseline; other tests must not see us serializing.
        set_serializing(prev_outer);
        assert!(!serializing());
    }

    #[test]
    fn time_phase_armed_banks_only_when_armed() {
        // Local counter → no cross-test interference on the process globals.
        let counter = AtomicU64::new(0);

        // Disarmed: pure pass-through, no bank.
        let r = time_phase_armed(false, &counter, || {
            busy_ns(1_000);
            42
        });
        assert_eq!(r, 42, "closure result must pass through");
        assert_eq!(counter.load(Ordering::Relaxed), 0, "disarmed must not bank");

        // Armed: closure result passes through AND wall time banks (> 0).
        let r = time_phase_armed(true, &counter, || {
            busy_ns(50_000);
            7
        });
        assert_eq!(r, 7);
        assert!(
            counter.load(Ordering::Relaxed) > 0,
            "armed must bank a positive ns delta"
        );
    }

    #[test]
    fn time_roundtrip_half_armed_banks_only_when_armed() {
        let _g = GLOBALS_LOCK.lock().unwrap();
        let before = MIGRATE_ROUNDTRIP_NS.load(Ordering::Relaxed);
        let r = time_roundtrip_half_armed(false, || {
            busy_ns(1_000);
            "off"
        });
        assert_eq!(r, "off");
        assert_eq!(
            MIGRATE_ROUNDTRIP_NS.load(Ordering::Relaxed),
            before,
            "disarmed roundtrip must not move the global"
        );

        let r = time_roundtrip_half_armed(true, || {
            busy_ns(50_000);
            "on"
        });
        assert_eq!(r, "on");
        assert!(
            MIGRATE_ROUNDTRIP_NS.load(Ordering::Relaxed) > before,
            "armed roundtrip must advance the global"
        );
    }

    #[test]
    fn add_raw_constraint_count_armed_gates_on_flag() {
        let _g = GLOBALS_LOCK.lock().unwrap();
        let before = MIGRATE_RAW_CONSTRAINT_COUNT.load(Ordering::Relaxed);
        add_raw_constraint_count_armed(false, 100);
        assert_eq!(
            MIGRATE_RAW_CONSTRAINT_COUNT.load(Ordering::Relaxed),
            before,
            "disarmed count must not accumulate"
        );
        add_raw_constraint_count_armed(true, 5);
        assert_eq!(
            MIGRATE_RAW_CONSTRAINT_COUNT.load(Ordering::Relaxed),
            before + 5,
            "armed count must add exactly n"
        );
    }

    #[test]
    fn count_armed_matches_env_and_serializing() {
        // `count_armed()` is `enabled() && serializing()`. `enabled()` is the
        // process-global env gate (fixed for the test binary); regardless of
        // its value, count_armed must be false while not serializing.
        let prev = set_serializing(false);
        assert!(!count_armed(), "not serializing => never armed");
        // When serializing, count_armed tracks enabled() exactly.
        set_serializing(true);
        assert_eq!(count_armed(), enabled());
        set_serializing(prev);
    }

    #[test]
    fn snapshot_reports_globals_in_documented_order() {
        let _g = GLOBALS_LOCK.lock().unwrap();
        // Field order is (emit, parse, serde, leaf_rebuild, raw_count,
        // roundtrip). Bump each global by a distinct delta and confirm the
        // matching tuple field moved by exactly that amount.
        let s0 = snapshot();
        MIGRATE_SMTLIB2_EMIT_NS.fetch_add(1, Ordering::Relaxed);
        MIGRATE_SMTLIB2_PARSE_NS.fetch_add(2, Ordering::Relaxed);
        MIGRATE_SERDE_NS.fetch_add(4, Ordering::Relaxed);
        MIGRATE_LEAF_REBUILD_NS.fetch_add(8, Ordering::Relaxed);
        MIGRATE_RAW_CONSTRAINT_COUNT.fetch_add(16, Ordering::Relaxed);
        MIGRATE_ROUNDTRIP_NS.fetch_add(32, Ordering::Relaxed);
        let s1 = snapshot();
        assert_eq!(s1.0 - s0.0, 1);
        assert_eq!(s1.1 - s0.1, 2);
        assert_eq!(s1.2 - s0.2, 4);
        assert_eq!(s1.3 - s0.3, 8);
        assert_eq!(s1.4 - s0.4, 16);
        assert_eq!(s1.5 - s0.5, 32);
    }
}
