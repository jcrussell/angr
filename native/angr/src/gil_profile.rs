//! Thread-local GIL-work timer for the parallel-exploration spike (angr-1ilq.7).
//!
//! Measures, on a single worker thread, the wall-clock fraction of stepping
//! that is spent *inside Python-touching code* (the "GIL-held" work that a
//! lazy-GIL worker could NOT run with the GIL released). The complement is the
//! pure-Rust+Z3 work that IS parallelizable. This number feeds the A-vs-B
//! GIL-strategy decision via Amdahl's law (see the bead + the rust_engine docs).
//!
//! Design constraints that shaped this module (from adversarial peer review of
//! the naive "bracket every site and sum" approach):
//!
//! * **Re-entrancy depth guard.** Python-touch sites nest (a `PythonCallbacks`
//!   dispatch method calls `rustbv_to_claripy`) and recurse (`claripy_to_rustbv`
//!   self-recurses per operand). A per-call timer would sum descendant elapsed
//!   and super-linearly over-count. Instead a thread-local depth counter starts
//!   the clock only on the outermost `0 -> 1` entry and banks elapsed only on
//!   the matching `1 -> 0` exit. Nested/recursive guards are timing no-ops, so
//!   each disjoint GIL region is counted exactly once.
//! * **No `self` borrow.** The run loop holds `&mut self` across the whole
//!   stepping loop, so the accumulators cannot live behind a `&mut self` guard.
//!   They are thread-locals; the run-loop wall guard ([`RunLoopWallGuard`])
//!   banks elapsed on `Drop`, capturing every exit path (early `return`, `?`,
//!   panic) without touching `self`.
//! * **Off by default.** Gated on a thread-local `ENABLED` flag set from the
//!   manager's `profiling_enabled` at run-loop entry; when off, every guard is a
//!   single `Cell` read and the hot path (bench-regression gate) is unaffected.
//!
//! Both accumulators are monotonic across run-loop invocations and are read
//! cumulatively by `stats()` on the same thread. [`reset`] clears them at
//! `set_profiling(true)` so a fresh manager on a reused thread starts clean.
//!
//! **Coherence by construction.** GIL timing is gated on an `ACTIVE` flag that
//! is set only while a profiled [`RunLoopWallGuard`] is live. The bridge
//! functions and `PythonCallbacks` methods are also called from non-stepping
//! paths (state export to Python after `run()`, constraint export, proxy
//! reads); counting those in the numerator while the denominator measures only
//! the run loop would let `GIL_fraction` exceed 1. Gating on `ACTIVE` makes
//! every banked GIL region a subset of the run-loop wall window, so
//! `gil_work_ns() <= run_wall_ns()` always holds.

use std::cell::Cell;
use std::time::Instant;

thread_local! {
    /// True while a profiled run loop is executing on this thread. GIL regions
    /// are timed only while this is set (see module "Coherence" note).
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// Nesting depth of active [`GilWorkGuard`]s. The clock runs while > 0.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Start instant of the current outermost GIL region (`Some` while depth > 0).
    static REGION_START: Cell<Option<Instant>> = const { Cell::new(None) };
    /// Cumulative nanoseconds spent inside Python-touching code (numerator).
    static GIL_ACCUM_NS: Cell<u64> = const { Cell::new(0) };
    /// Cumulative run-loop wall-clock nanoseconds (denominator).
    static WALL_ACCUM_NS: Cell<u64> = const { Cell::new(0) };
}

/// Reset both accumulators and the depth/region state. Call when (re)enabling
/// profiling so a fresh manager on a reused thread does not inherit stale time.
pub fn reset() {
    ACTIVE.with(|a| a.set(false));
    DEPTH.with(|d| d.set(0));
    REGION_START.with(|s| s.set(None));
    GIL_ACCUM_NS.with(|a| a.set(0));
    WALL_ACCUM_NS.with(|w| w.set(0));
}

/// Cumulative GIL-work nanoseconds on this thread (the Amdahl numerator).
#[inline]
pub fn gil_work_ns() -> u64 {
    GIL_ACCUM_NS.with(std::cell::Cell::get)
}

/// Cumulative run-loop wall-clock nanoseconds on this thread (the denominator).
#[inline]
pub fn run_wall_ns() -> u64 {
    WALL_ACCUM_NS.with(std::cell::Cell::get)
}

/// RAII guard bracketing a Python-touching region. Only the outermost live
/// guard on the thread times; nested guards just balance the depth counter so
/// overlapping/recursive sites are never double-counted. A no-op when disabled.
pub struct GilWorkGuard {
    /// True only when this guard participates in depth counting (profiling was
    /// enabled at `enter()`). Disabled guards skip all `Drop` work.
    active: bool,
}

impl GilWorkGuard {
    #[inline]
    pub fn enter() -> Self {
        if !ACTIVE.with(std::cell::Cell::get) {
            return GilWorkGuard { active: false };
        }
        let prev = DEPTH.with(|d| {
            let n = d.get();
            d.set(n + 1);
            n
        });
        if prev == 0 {
            REGION_START.with(|s| s.set(Some(Instant::now())));
        }
        GilWorkGuard { active: true }
    }
}

impl Drop for GilWorkGuard {
    #[inline]
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let now = DEPTH.with(|d| {
            let n = d.get().saturating_sub(1);
            d.set(n);
            n
        });
        if now == 0
            && let Some(start) = REGION_START.with(std::cell::Cell::take)
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            GIL_ACCUM_NS.with(|a| a.set(a.get() + elapsed));
        }
    }
}

/// RAII guard for one run-loop invocation's wall time. While live (and enabled)
/// it sets the `ACTIVE` flag so [`GilWorkGuard`]s time, and banks elapsed wall
/// into the thread-local denominator on `Drop` — capturing every exit path
/// (early `return`, `?`, panic) without borrowing `self`. A no-op when disabled.
///
/// Run loops are not re-entrant on a worker thread, so a plain bool flag is
/// sufficient; the guard restores the prior `ACTIVE` value on drop defensively.
pub struct RunLoopWallGuard {
    start: Option<Instant>,
    prev_active: bool,
}

impl RunLoopWallGuard {
    #[inline]
    pub fn new(enabled: bool) -> Self {
        let prev_active = ACTIVE.with(std::cell::Cell::get);
        if enabled {
            ACTIVE.with(|a| a.set(true));
        }
        RunLoopWallGuard {
            start: if enabled { Some(Instant::now()) } else { None },
            prev_active,
        }
    }
}

impl Drop for RunLoopWallGuard {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed().as_nanos() as u64;
            WALL_ACCUM_NS.with(|w| w.set(w.get() + elapsed));
            ACTIVE.with(|a| a.set(self.prev_active));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy_ns(min_ns: u64) {
        // Spin until at least `min_ns` of wall time has elapsed so the timer
        // sees a non-zero region without relying on sleep (which the harness
        // forbids in some contexts).
        let start = Instant::now();
        while (start.elapsed().as_nanos() as u64) < min_ns {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn disabled_is_noop() {
        reset();
        // No active run loop -> GIL timing is off even though a guard is entered.
        {
            let _g = GilWorkGuard::enter();
            busy_ns(50_000);
        }
        assert_eq!(gil_work_ns(), 0, "no time should bank while inactive");
    }

    #[test]
    fn gil_outside_run_loop_is_ignored() {
        // A bridge/callback call made outside a profiled run loop (e.g. state
        // export after run()) must NOT inflate the numerator.
        reset();
        {
            let _w = RunLoopWallGuard::new(true); // run loop active
            let _g = GilWorkGuard::enter();
            busy_ns(50_000);
        }
        let in_loop = gil_work_ns();
        {
            // Outside any run loop now (ACTIVE restored to false).
            let _g = GilWorkGuard::enter();
            busy_ns(50_000);
        }
        assert_eq!(
            gil_work_ns(),
            in_loop,
            "GIL work outside the run loop must not be counted"
        );
    }

    #[test]
    fn outermost_only_times_once() {
        reset();
        let _w = RunLoopWallGuard::new(true);
        {
            let _outer = GilWorkGuard::enter();
            busy_ns(100_000);
            {
                // Nested + recursive guards must not add their own elapsed.
                let _inner = GilWorkGuard::enter();
                let _inner2 = GilWorkGuard::enter();
                busy_ns(100_000);
            }
        }
        let banked = gil_work_ns();
        // One contiguous region (~200us). It must NOT be ~400us+ (which is what
        // a naive per-call sum of the 3 nested guards would produce).
        assert!(banked >= 150_000, "region undercounted: {banked} ns");
        assert!(
            banked < 350_000,
            "nested guards double-counted: {banked} ns"
        );
    }

    #[test]
    fn wall_guard_banks_on_drop() {
        reset();
        {
            let _w = RunLoopWallGuard::new(true);
            busy_ns(100_000);
        }
        assert!(run_wall_ns() >= 80_000, "wall time undercounted");
    }

    #[test]
    fn gil_never_exceeds_wall() {
        // Coherence invariant the spike relies on: GIL work is a subset of
        // run-loop wall time, so the fraction is <= 1.
        reset();
        {
            let _w = RunLoopWallGuard::new(true);
            busy_ns(50_000);
            {
                let _g = GilWorkGuard::enter();
                busy_ns(100_000);
            }
            busy_ns(50_000);
        }
        assert!(
            gil_work_ns() <= run_wall_ns(),
            "gil {} > wall {}",
            gil_work_ns(),
            run_wall_ns()
        );
    }
}
