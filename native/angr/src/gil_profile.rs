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
    /// Class of the current outermost GIL region (meaningful while depth > 0).
    static REGION_CLASS: Cell<GilClass> = const { Cell::new(GilClass::Callback) };
    /// Per-class split of `GIL_ACCUM_NS`, indexed by `GilClass as usize`.
    static CLASS_ACCUM_NS: Cell<[u64; GilClass::COUNT]> = const { Cell::new([0; GilClass::COUNT]) };
    /// Dispatch site of the current outermost GIL region (meaningful while the
    /// region's class is [`GilClass::Callback`] and depth > 0).
    static REGION_SITE: Cell<CallbackSite> = const { Cell::new(CallbackSite::Other) };
    /// Per-site split of the `Callback` class, indexed by `CallbackSite as usize`.
    static SITE_ACCUM_NS: Cell<[u64; CallbackSite::COUNT]> =
        const { Cell::new([0; CallbackSite::COUNT]) };
    /// Start instant of an in-flight park-and-bounce excursion (`Some` between
    /// [`park_start`] and the matching [`park_end`] / [`park_cancel`]).
    static PARK_START: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Why the run loop is holding the GIL. Attributing the *outermost* region is
/// what makes this a partition: nested guards do not time (see [`GilWorkGuard`]),
/// so every banked nanosecond belongs to exactly one class and the classes sum
/// to [`gil_work_ns`].
///
/// The split exists because callback counters alone cannot explain the residual
/// GIL on the zero-bounce path (bd angr-gorvf.4): benches with two cheap `posix`
/// callbacks and nothing else were still spending 74% of the run loop under the
/// GIL — all of it in the claripy AST bridge and the fork-metadata attach, which
/// are Python touches that no `callback_*` counter tracks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GilClass {
    /// A `PythonCallbacks` dispatch (lift_block, posix, simprocedure, ...). The
    /// per-class detail lives in the `callback_*_total_ns` counters.
    Callback,
    /// `rustbv_to_claripy` — exporting a Rust AST into claripy.
    ClaripyExport,
    /// `claripy_to_rustbv` — importing a claripy AST into Rust.
    ClaripyImport,
    /// A state fork cloning its Python-side overlays. **Expected to stay 0**:
    /// since angr-gorvf.4.2 the overlays hold `SharedPyAst` (`Arc<Py<PyAny>>`),
    /// so `clone_py_metadata` bumps atomic refcounts instead of attaching to
    /// Python. The class is kept (rather than deleted) so the counter keeps
    /// *proving* that — a regression that reintroduces a GIL attach on the fork
    /// path shows up here as a nonzero `gil_work_ns_fork_metadata`.
    ForkMetadata,
    /// A **park-and-bounce excursion**: the run loop returned an
    /// `ExplorationEvent` with a pending callback, Python ran the handler (a
    /// SimProcedure / syscall / hook / symbolic-branch resolution), and then
    /// re-entered Rust through a `resume_after_*` method.
    ///
    /// This class exists because a bounce is *not* a [`GilWorkGuard`] region:
    /// no Rust frame is live while it runs, so the run loop's
    /// [`RunLoopWallGuard`] has already dropped and neither the numerator nor
    /// the denominator would otherwise see the excursion (bd angr-gorvf.8 —
    /// fauxware banked a 59.6ms Python `open` SimProcedure while reporting
    /// `gil_work_time_ns == 0`). [`park_start`] / [`park_end`] bracket the
    /// excursion at the pymethod boundary and bank it into *both* accumulators,
    /// so `gil_work_time_ns == 0` really does mean "no Python ran during
    /// exploration" and `gil_work_ns() <= run_wall_ns()` still holds.
    Bounce,
}

impl GilClass {
    const COUNT: usize = 5;

    /// Stable counter suffix, used to name the `gil_work_ns_*` stats keys.
    pub fn name(self) -> &'static str {
        match self {
            GilClass::Callback => "callback",
            GilClass::ClaripyExport => "claripy_export",
            GilClass::ClaripyImport => "claripy_import",
            GilClass::ForkMetadata => "fork_metadata",
            GilClass::Bounce => "bounce",
        }
    }

    pub fn all() -> [GilClass; GilClass::COUNT] {
        [
            GilClass::Callback,
            GilClass::ClaripyExport,
            GilClass::ClaripyImport,
            GilClass::ForkMetadata,
            GilClass::Bounce,
        ]
    }
}

/// Which `PythonCallbacks` entry point took the GIL. One variant per
/// `GilWorkGuard` site in `callbacks/dispatch.rs`, plus a bucket for the
/// `state.inspect` dispatch family (`callbacks/inspect.rs`) and an `Other`
/// catch-all.
///
/// This is a *sub*-partition of [`GilClass::Callback`]: a site is recorded only
/// when the outermost region on the thread is a `Callback` region, so the sites
/// sum exactly to `gil_class_ns(GilClass::Callback)`.
///
/// It exists because the Python-side `callback_*_total_ns` counters do not
/// explain the residual `Callback` GIL on the zero-bounce path (bd
/// angr-gorvf.4.1): `google2016_unbreakable_1` banks 723ms of `Callback` (74% of
/// the run-loop wall) while its only counted surface is two `posix` calls at
/// 0ms. Those counters are process-wide and also tick outside the profiled run
/// loop; these are thread-local, run-loop-gated, and complete by construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallbackSite {
    MemoryLoad,
    MemoryStore,
    MemoryStoreBatch,
    MemoryLoadBatch,
    MemoryLoadSymbolic,
    MemoryStoreSymbolic,
    MemoryStoreSymbolicValue,
    MemoryStoreSymbolicFull,
    MemoryLoadSymbolicFull,
    OnHook,
    OnSyscall,
    LiftBlock,
    GetRegister,
    PutRegister,
    DirtyCall,
    FetchPage,
    BatchFetchPages,
    ResolveFunction,
    Inspect,
    /// A `GilWorkGuard::enter()` with no site attached. Expected to stay 0 —
    /// a nonzero value means a new Python-touching callback was added without
    /// naming its site, and the sub-partition has an unexplained residual.
    Other,
}

impl CallbackSite {
    const COUNT: usize = 20;

    /// Stable counter suffix, used to name the `gil_work_ns_callback_*` keys.
    pub fn name(self) -> &'static str {
        match self {
            CallbackSite::MemoryLoad => "memory_load",
            CallbackSite::MemoryStore => "memory_store",
            CallbackSite::MemoryStoreBatch => "memory_store_batch",
            CallbackSite::MemoryLoadBatch => "memory_load_batch",
            CallbackSite::MemoryLoadSymbolic => "memory_load_symbolic",
            CallbackSite::MemoryStoreSymbolic => "memory_store_symbolic",
            CallbackSite::MemoryStoreSymbolicValue => "memory_store_symbolic_value",
            CallbackSite::MemoryStoreSymbolicFull => "memory_store_symbolic_full",
            CallbackSite::MemoryLoadSymbolicFull => "memory_load_symbolic_full",
            CallbackSite::OnHook => "on_hook",
            CallbackSite::OnSyscall => "on_syscall",
            CallbackSite::LiftBlock => "lift_block",
            CallbackSite::GetRegister => "get_register",
            CallbackSite::PutRegister => "put_register",
            CallbackSite::DirtyCall => "dirty_call",
            CallbackSite::FetchPage => "fetch_page",
            CallbackSite::BatchFetchPages => "batch_fetch_pages",
            CallbackSite::ResolveFunction => "resolve_function",
            CallbackSite::Inspect => "inspect",
            CallbackSite::Other => "other",
        }
    }

    pub fn all() -> [CallbackSite; CallbackSite::COUNT] {
        [
            CallbackSite::MemoryLoad,
            CallbackSite::MemoryStore,
            CallbackSite::MemoryStoreBatch,
            CallbackSite::MemoryLoadBatch,
            CallbackSite::MemoryLoadSymbolic,
            CallbackSite::MemoryStoreSymbolic,
            CallbackSite::MemoryStoreSymbolicValue,
            CallbackSite::MemoryStoreSymbolicFull,
            CallbackSite::MemoryLoadSymbolicFull,
            CallbackSite::OnHook,
            CallbackSite::OnSyscall,
            CallbackSite::LiftBlock,
            CallbackSite::GetRegister,
            CallbackSite::PutRegister,
            CallbackSite::DirtyCall,
            CallbackSite::FetchPage,
            CallbackSite::BatchFetchPages,
            CallbackSite::ResolveFunction,
            CallbackSite::Inspect,
            CallbackSite::Other,
        ]
    }
}

/// Cumulative GIL-work nanoseconds attributed to `class` on this thread.
#[inline]
pub fn gil_class_ns(class: GilClass) -> u64 {
    CLASS_ACCUM_NS.with(|c| c.get()[class as usize])
}

/// Cumulative `Callback`-class nanoseconds attributed to `site` on this thread.
#[inline]
pub fn callback_site_ns(site: CallbackSite) -> u64 {
    SITE_ACCUM_NS.with(|s| s.get()[site as usize])
}

/// Reset both accumulators and the depth/region state. Call when (re)enabling
/// profiling so a fresh manager on a reused thread does not inherit stale time.
pub fn reset() {
    ACTIVE.with(|a| a.set(false));
    DEPTH.with(|d| d.set(0));
    REGION_START.with(|s| s.set(None));
    GIL_ACCUM_NS.with(|a| a.set(0));
    WALL_ACCUM_NS.with(|w| w.set(0));
    CLASS_ACCUM_NS.with(|c| c.set([0; GilClass::COUNT]));
    REGION_SITE.with(|s| s.set(CallbackSite::Other));
    SITE_ACCUM_NS.with(|s| s.set([0; CallbackSite::COUNT]));
    PARK_START.with(|p| p.set(None));
}

/// Arm the park clock: the run loop is about to hand control back to Python
/// with a callback pending. A no-op when `enabled` is false.
///
/// The excursion is banked only if Python comes back through [`park_end`]
/// (a `resume_after_*` / `deadend_pending_callback`). If Python instead re-runs
/// the loop without resuming, [`park_cancel`] discards the clock — driver-loop
/// overhead between `run()` calls is not exploration-time Python work.
#[inline]
pub fn park_start(enabled: bool) {
    if enabled {
        PARK_START.with(|p| p.set(Some(Instant::now())));
    }
}

/// Bank an in-flight park excursion as [`GilClass::Bounce`] work. Called on
/// re-entry from the Python callback handler. A no-op when no park is armed.
#[inline]
pub fn park_end() {
    if let Some(start) = PARK_START.with(std::cell::Cell::take) {
        let elapsed = start.elapsed().as_nanos() as u64;
        GIL_ACCUM_NS.with(|a| a.set(a.get() + elapsed));
        CLASS_ACCUM_NS.with(|c| {
            let mut split = c.get();
            split[GilClass::Bounce as usize] += elapsed;
            c.set(split);
        });
        // The excursion happens with the run loop's wall guard dropped, so the
        // denominator must grow with the numerator or `gil <= wall` breaks.
        WALL_ACCUM_NS.with(|w| w.set(w.get() + elapsed));
    }
}

/// Discard an in-flight park excursion without banking it.
#[inline]
pub fn park_cancel() {
    PARK_START.with(|p| p.set(None));
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
    /// Enter an unattributed [`GilClass::Callback`] region. Prefer
    /// [`GilWorkGuard::enter_site`]: time banked here lands in
    /// [`CallbackSite::Other`], which is the sub-partition's residual bucket.
    #[inline]
    pub fn enter() -> Self {
        Self::enter_site(CallbackSite::Other)
    }

    /// Enter a [`GilClass::Callback`] region attributed to `site`.
    #[inline]
    pub fn enter_site(site: CallbackSite) -> Self {
        Self::enter_inner(GilClass::Callback, site)
    }

    /// Enter a region attributed to `class`. Only the outermost live guard on
    /// the thread times, so the class recorded is the *reason the GIL was first
    /// taken*, not whichever nested site happens to be innermost.
    #[inline]
    pub fn enter_as(class: GilClass) -> Self {
        Self::enter_inner(class, CallbackSite::Other)
    }

    #[inline]
    fn enter_inner(class: GilClass, site: CallbackSite) -> Self {
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
            REGION_CLASS.with(|c| c.set(class));
            REGION_SITE.with(|s| s.set(site));
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
            let class = REGION_CLASS.with(std::cell::Cell::get);
            CLASS_ACCUM_NS.with(|c| {
                let mut split = c.get();
                split[class as usize] += elapsed;
                c.set(split);
            });
            // The site split is a sub-partition of the `Callback` class only;
            // a claripy-bridge or fork-metadata region has no dispatch site.
            if class == GilClass::Callback {
                let site = REGION_SITE.with(std::cell::Cell::get);
                SITE_ACCUM_NS.with(|s| {
                    let mut split = s.get();
                    split[site as usize] += elapsed;
                    s.set(split);
                });
            }
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
    fn class_split_partitions_the_total() {
        reset();
        {
            let _w = RunLoopWallGuard::new(true);
            {
                let _g = GilWorkGuard::enter_as(GilClass::ClaripyExport);
                busy_ns(50_000);
            }
            {
                // A nested guard of a *different* class must not steal the
                // region: the outermost entry is what took the GIL.
                let _outer = GilWorkGuard::enter_as(GilClass::ForkMetadata);
                let _inner = GilWorkGuard::enter_as(GilClass::ClaripyImport);
                busy_ns(50_000);
            }
        }
        let total: u64 = GilClass::all().iter().map(|c| gil_class_ns(*c)).sum();
        assert_eq!(total, gil_work_ns(), "classes must partition the GIL total");
        assert!(gil_class_ns(GilClass::ClaripyExport) > 0);
        assert!(gil_class_ns(GilClass::ForkMetadata) > 0);
        assert_eq!(
            gil_class_ns(GilClass::ClaripyImport),
            0,
            "a nested guard must not be attributed"
        );
        assert_eq!(gil_class_ns(GilClass::Callback), 0);
    }

    #[test]
    fn sites_partition_the_callback_class() {
        reset();
        {
            let _w = RunLoopWallGuard::new(true);
            {
                let _g = GilWorkGuard::enter_site(CallbackSite::LiftBlock);
                busy_ns(50_000);
            }
            {
                // A nested site must not steal the region from the outermost one.
                let _outer = GilWorkGuard::enter_site(CallbackSite::MemoryLoad);
                let _inner = GilWorkGuard::enter_site(CallbackSite::GetRegister);
                busy_ns(50_000);
            }
            {
                // A non-Callback outermost region contributes no site time.
                let _g = GilWorkGuard::enter_as(GilClass::ClaripyExport);
                busy_ns(50_000);
            }
        }
        let sites: u64 = CallbackSite::all()
            .iter()
            .map(|s| callback_site_ns(*s))
            .sum();
        assert_eq!(
            sites,
            gil_class_ns(GilClass::Callback),
            "sites must partition the Callback class"
        );
        assert!(callback_site_ns(CallbackSite::LiftBlock) > 0);
        assert!(callback_site_ns(CallbackSite::MemoryLoad) > 0);
        assert_eq!(
            callback_site_ns(CallbackSite::GetRegister),
            0,
            "a nested site must not be attributed"
        );
        assert_eq!(
            callback_site_ns(CallbackSite::Other),
            0,
            "no unattributed callback GIL time"
        );
    }

    #[test]
    fn park_excursion_banks_into_gil_and_wall() {
        // The bounce runs with the run loop's wall guard dropped (Python owns
        // the thread), so it must grow both accumulators (bd angr-gorvf.8).
        reset();
        {
            let _w = RunLoopWallGuard::new(true);
            busy_ns(50_000);
        }
        let (gil_before, wall_before) = (gil_work_ns(), run_wall_ns());
        park_start(true);
        busy_ns(200_000);
        park_end();
        let banked = gil_work_ns() - gil_before;
        assert!(banked >= 150_000, "bounce undercounted: {banked} ns");
        assert_eq!(gil_class_ns(GilClass::Bounce), banked);
        assert_eq!(
            run_wall_ns() - wall_before,
            banked,
            "wall must track bounce"
        );
        assert!(
            gil_work_ns() <= run_wall_ns(),
            "gil <= wall must still hold"
        );
        let total: u64 = GilClass::all().iter().map(|c| gil_class_ns(*c)).sum();
        assert_eq!(
            total,
            gil_work_ns(),
            "classes must still partition the total"
        );
    }

    #[test]
    fn park_cancel_and_disabled_park_bank_nothing() {
        reset();
        // A cancelled park (Python re-ran the loop instead of resuming).
        park_start(true);
        busy_ns(50_000);
        park_cancel();
        park_end();
        assert_eq!(gil_work_ns(), 0, "cancelled park must not bank");
        // Profiling off: park_start is a no-op, so park_end has nothing to bank.
        park_start(false);
        busy_ns(50_000);
        park_end();
        assert_eq!(gil_work_ns(), 0, "disabled park must not bank");
        assert_eq!(run_wall_ns(), 0);
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
        // Time the whole nested region ourselves. The outermost guard's banked
        // value must track THIS wall span (single timing), not a per-guard sum
        // which for 3 nested guards would be ~2x the span. Comparing against
        // the measured span rather than a fixed ceiling keeps the test robust
        // under scheduler preemption: preemption inflates the observed span and
        // the guard's banked value equally, so their ratio stays stable even
        // when the whole suite runs in parallel and contends for CPU.
        let outer_start = Instant::now();
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
        let outer_span = outer_start.elapsed().as_nanos() as u64;
        let banked = gil_work_ns();
        // Lower bound: the ~200us of busy work must actually be counted.
        assert!(banked >= 150_000, "region undercounted: {banked} ns");
        // Upper bound: banked reflects one timing of the outermost region, so
        // it can never meaningfully exceed the wall span that contains it. A
        // naive per-guard sum would be ~2x the span; 1.5x cleanly separates the
        // correct (~1.0x) case from the double-counted one.
        assert!(
            banked * 2 <= outer_span * 3,
            "nested guards double-counted: banked {banked} ns vs span {outer_span} ns"
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
