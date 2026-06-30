//! Statistics-export methods.
//!
//! Bodies for the pyclass-exposed methods that aggregate counters and
//! state-population stats into Python dicts — `stats`, `get_fallback_stats`,
//! and `native_procedure_stats`. The pyclass-facing thin wrappers live in
//! `mod.rs` and forward to the `pub(crate)` bodies in this module.
//!
//! PyO3 0.27.2 in this project does not enable `multiple-pymethods`, so each
//! pyclass is limited to a single `#[pymethods]` impl block — see
//! `invariant-pyo3-single-pymethods-impl`. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` /
//! `pending_api.rs` / `state_api.rs` extension-impl pattern used elsewhere in
//! `exploration/`.

use super::*;

impl RustExplorationManager {
    pub(crate) fn _stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("steps", self.steps)?;
        dict.set_item("active", self.active_count())?;
        dict.set_item("found", self.found_count())?;
        dict.set_item("errors", self.errors.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("simprocedures", self.simprocedures.len())?;
        dict.set_item("find_addrs", self.find_addrs.len())?;
        dict.set_item("avoid_addrs", self.avoid_addrs.len())?;
        dict.set_item("block_cache_size", self.environment.block_cache.len())?;
        // IRSB block-cache hit/miss/eviction counters (always-on; merged from
        // the per-interpreter ExecutionStats on each `swap_block_cache`).
        // Use these to tune `BLOCK_CACHE_CAPACITY`: a high eviction-to-miss
        // ratio means the working set exceeds capacity; near-zero evictions
        // mean the cache is oversized for that workload.
        dict.set_item(
            "block_cache_hits",
            self.profiling.accumulated_stats.cache_hit_count,
        )?;
        dict.set_item(
            "block_cache_misses",
            self.profiling.accumulated_stats.cache_miss_count,
        )?;
        dict.set_item(
            "block_cache_evictions",
            self.profiling.accumulated_stats.cache_eviction_count,
        )?;
        dict.set_item(
            "native_proc_calls",
            self.profiling.native_proc_stats.native_calls,
        )?;
        dict.set_item(
            "native_proc_fallbacks",
            self.profiling.native_proc_stats.python_fallbacks,
        )?;
        // angr-ilsr: per-reason fallback breakdown. sum(symbolic +
        // not_implemented + other) == native_proc_fallbacks.
        let symbolic_by_name = PyDict::new(py);
        let mut symbolic_total: u64 = 0;
        for (name, count) in &self.profiling.native_proc_stats.symbolic_fallbacks_by_name {
            symbolic_by_name.set_item(name, *count)?;
            symbolic_total += *count;
        }
        dict.set_item("native_proc_symbolic_fallbacks_by_name", symbolic_by_name)?;
        dict.set_item("native_proc_symbolic_fallbacks", symbolic_total)?;

        let not_impl_by_name = PyDict::new(py);
        let mut not_impl_total: u64 = 0;
        for (name, count) in &self
            .profiling
            .native_proc_stats
            .not_implemented_fallbacks_by_name
        {
            not_impl_by_name.set_item(name, *count)?;
            not_impl_total += *count;
        }
        dict.set_item(
            "native_proc_not_implemented_fallbacks_by_name",
            not_impl_by_name,
        )?;
        dict.set_item("native_proc_not_implemented_fallbacks", not_impl_total)?;

        let other_by_name = PyDict::new(py);
        let mut other_total: u64 = 0;
        for (name, count) in &self.profiling.native_proc_stats.other_fallbacks_by_name {
            other_by_name.set_item(name, *count)?;
            other_total += *count;
        }
        dict.set_item("native_proc_other_fallbacks_by_name", other_by_name)?;
        dict.set_item("native_proc_other_fallbacks", other_total)?;
        dict.set_item("avoided_count", self.sm.avoided_count)?;
        dict.set_item("pruned_count", self.sm.pruned_count)?;
        dict.set_item("deadended_count", self.sm.deadended_count)?;
        dict.set_item("drop_terminal_states", self.sm.drop_terminal_states())?;
        dict.set_item("state_roots_size", self.sm.roots().len())?;
        dict.set_item("vex_fallback_count", self.vex_fallback_count)?;
        dict.set_item("vex_fallback_unique_addrs", self.vex_fallback_addrs.len())?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "vecret_gsptr_fallback_count",
            self.vecret_gsptr_fallback_count,
        )?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        let fallback_by_name = PyDict::new(py);
        for (name, count) in &self.simprocedure_fallback_by_name {
            fallback_by_name.set_item(name, *count)?;
        }
        dict.set_item("simprocedure_fallback_by_name", fallback_by_name)?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        let syscall_fallback_by_num = PyDict::new(py);
        for (num, count) in &self.syscall_python_fallback_by_num {
            syscall_fallback_by_num.set_item(*num, *count)?;
        }
        dict.set_item("syscall_python_fallback_by_num", syscall_fallback_by_num)?;
        dict.set_item("syscall_native_count", self.syscall_native_count)?;
        let syscall_native_by_num = PyDict::new(py);
        for (num, count) in &self.syscall_native_by_num {
            syscall_native_by_num.set_item(*num, *count)?;
        }
        dict.set_item("syscall_native_by_num", syscall_native_by_num)?;
        // DS-instr (angr-11djq.16): state-reconvergence counters. A collision
        // == two+ active states sharing a (pc, callstack) key at the same step.
        // `reconvergence_rate` = collision_states / active_observed over all
        // per-step samples (0.0 when nothing was sampled). High rate => lots of
        // reconvergence => directed-search pruning / state merging have
        // headroom; near-zero => they have nothing to merge.
        dict.set_item(
            "reconvergence_collision_states",
            self.reconvergence_collision_states,
        )?;
        dict.set_item(
            "reconvergence_active_observed",
            self.reconvergence_active_observed,
        )?;
        dict.set_item("reconvergence_samples", self.reconvergence_samples)?;
        dict.set_item("reconvergence_max_group", self.reconvergence_max_group)?;
        let reconvergence_rate = if self.reconvergence_active_observed > 0 {
            self.reconvergence_collision_states as f64 / self.reconvergence_active_observed as f64
        } else {
            0.0
        };
        dict.set_item("reconvergence_rate", reconvergence_rate)?;
        // angr-panhl.1 (Phase 0 kill-gate): work-stealing migration model.
        // `parallel_migrations` is the count of modelled cross-worker steals;
        // `parallel_tasks` the number of dispatched tasks (≈ steps); the
        // kill-gate compares migrations/bench (<10 PASS) and per-task duration
        // (wall_time / parallel_tasks > 100ms PASS), with
        // `parallel_max_active_width` the independent-state count over time and
        // `python_callback_time_ns` the GIL/callback fraction.
        dict.set_item("parallel_migrations", self.parallel_migrations)?;
        dict.set_item("parallel_tasks", self.parallel_tasks)?;
        dict.set_item("parallel_num_workers", self.parallel_num_workers)?;
        dict.set_item("parallel_real_workers", self.parallel_real_workers)?;
        dict.set_item("parallel_max_active_width", self.parallel_max_active_width)?;
        // angr-panhl.3 (concurrent-width audit): step-weighted width histogram
        // [width==1, ==2, 3–4, 5–8, ≥9]. Sustained width (the parallel-
        // favorability signal panhl.1 omitted) = 1 - hist[0]/sum(hist); the
        // fraction of steps with ≥3 concurrent states = (hist[2]+hist[3]+hist[4])
        // / sum(hist).
        dict.set_item("parallel_width_hist", self.parallel_width_hist.to_vec())?;
        // SI-B (angr-1ilq.3 increment 2b'): real state-migration serde tax,
        // measured by the opt-in shadow probe (RUST_PARALLEL_SHADOW_PROBE). All
        // zero unless the probe is on; feeds the 2b' overhead GO/NO-GO gate.
        // `parallel_shadow_migration_ns` is to_serialized (main thread) +
        // from_serialized (foreign Z3 context, scratch thread); per-state cost =
        // ns / states, per-state payload = bytes / states.
        dict.set_item(
            "parallel_shadow_migration_ns",
            self.parallel_shadow_migration_ns,
        )?;
        dict.set_item(
            "parallel_shadow_migration_states",
            self.parallel_shadow_migration_states,
        )?;
        dict.set_item(
            "parallel_shadow_migration_bytes",
            self.parallel_shadow_migration_bytes,
        )?;
        // angr-t3l5o Phase 0b: per-phase migration attribution (env-gated by
        // ANGR_MIGRATE_PHASE_TIMERS). Process-global accumulators, all zero
        // unless the env var is set. emit/parse/serde/leaf_rebuild are ns;
        // raw_constraint_count is the summed residual (no-RustBV) assertion
        // count across migrated states (divide by
        // parallel_shadow_migration_states for a per-state average). Surfaced
        // here so `run_single.py --counters-json` reports them alongside the
        // shadow-probe round-trip total.
        {
            let (emit_ns, parse_ns, serde_ns, leaf_rebuild_ns, raw_count, roundtrip_ns) =
                crate::migrate_phase_timers::snapshot();
            dict.set_item("migrate_smtlib2_emit_ns", emit_ns)?;
            dict.set_item("migrate_smtlib2_parse_ns", parse_ns)?;
            dict.set_item("migrate_serde_ns", serde_ns)?;
            dict.set_item("migrate_leaf_rebuild_ns", leaf_rebuild_ns)?;
            dict.set_item("migrate_raw_constraint_count", raw_count)?;
            // Self-consistent denominator: total wall of all migration
            // serialize+deserialize halves (same call set as the phase timers,
            // unlike parallel_shadow_migration_ns which omits non-probe states).
            dict.set_item("migrate_roundtrip_ns", roundtrip_ns)?;
        }
        dict.set_item(
            "python_callback_count",
            self.profiling.accumulated_stats.python_callback_count,
        )?;
        dict.set_item(
            "python_callback_time_ns",
            self.profiling.accumulated_stats.python_callback_time_ns,
        )?;
        // angr-1ilq.7 GIL-strategy spike: the depth-guarded total GIL-hold time
        // and the run-loop wall time, both read from the thread-local
        // accumulators (see `gil_profile`). `gil_work_time_ns` is the Amdahl
        // serial fraction for strategy A (lazy-GIL); the denominator is
        // `run_wall_time_ns`. By construction `gil_work_time_ns <=
        // run_wall_time_ns` (GIL regions are gated on an active run loop), so
        // `GIL_fraction <= 1`. Unlike `python_callback_time_ns` (lift-only),
        // this covers every Python-touch point: bridge round-trips, memory /
        // register / hook / syscall / dirty / fetch callbacks, inspect dispatch,
        // and per-fork metadata clone_ref.
        dict.set_item("gil_work_time_ns", crate::gil_profile::gil_work_ns())?;
        dict.set_item("run_wall_time_ns", crate::gil_profile::run_wall_ns())?;
        Ok(dict)
    }

    pub(crate) fn _get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("count", self.vex_fallback_count)?;
        let addrs = PyDict::new(py);
        for (&addr, reason) in &self.vex_fallback_addrs {
            addrs.set_item(format!("0x{:x}", addr), reason)?;
        }
        dict.set_item("addresses", addrs)?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "vecret_gsptr_fallback_count",
            self.vecret_gsptr_fallback_count,
        )?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        let fallback_by_name = PyDict::new(py);
        for (name, count) in &self.simprocedure_fallback_by_name {
            fallback_by_name.set_item(name, *count)?;
        }
        dict.set_item("simprocedure_fallback_by_name", fallback_by_name)?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        let syscall_fallback_by_num = PyDict::new(py);
        for (num, count) in &self.syscall_python_fallback_by_num {
            syscall_fallback_by_num.set_item(*num, *count)?;
        }
        dict.set_item("syscall_python_fallback_by_num", syscall_fallback_by_num)?;
        dict.set_item("syscall_native_count", self.syscall_native_count)?;
        let syscall_native_by_num = PyDict::new(py);
        for (num, count) in &self.syscall_native_by_num {
            syscall_native_by_num.set_item(*num, *count)?;
        }
        dict.set_item("syscall_native_by_num", syscall_native_by_num)?;
        Ok(dict)
    }

    pub(crate) fn _native_procedure_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let stats = &self.profiling.native_proc_stats;
        dict.set_item("native_calls", stats.native_calls)?;
        dict.set_item("python_fallbacks", stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        // Per-procedure fallback breakdown by reason. Sum across all three
        // maps equals `python_fallbacks`. Distinguishes "input was symbolic,
        // expected fallback" from "native impl missing this case".
        let symbolic = PyDict::new(py);
        let mut symbolic_total: u64 = 0;
        for (name, count) in &stats.symbolic_fallbacks_by_name {
            symbolic.set_item(name, *count)?;
            symbolic_total += *count;
        }
        dict.set_item("symbolic_fallbacks_by_name", symbolic)?;
        dict.set_item("symbolic_fallbacks", symbolic_total)?;

        let not_impl = PyDict::new(py);
        let mut not_impl_total: u64 = 0;
        for (name, count) in &stats.not_implemented_fallbacks_by_name {
            not_impl.set_item(name, *count)?;
            not_impl_total += *count;
        }
        dict.set_item("not_implemented_fallbacks_by_name", not_impl)?;
        dict.set_item("not_implemented_fallbacks", not_impl_total)?;

        let other = PyDict::new(py);
        let mut other_total: u64 = 0;
        for (name, count) in &stats.other_fallbacks_by_name {
            other.set_item(name, *count)?;
            other_total += *count;
        }
        dict.set_item("other_fallbacks_by_name", other)?;
        dict.set_item("other_fallbacks", other_total)?;

        Ok(dict)
    }
}
