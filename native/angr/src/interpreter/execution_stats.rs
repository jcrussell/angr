//! `ExecutionStats` — the interpreter's profiling counter block.
//!
//! Split out of `interpreter/mod.rs` (angr-9ke6b.91) so a counter addition and
//! a change to [`VEXInterpreter`](super::VEXInterpreter) no longer touch the
//! same file. The counters are folded into the manager-level dict by
//! `exploration::stats_api`; the `profile_start!` / `profile_add!` macros that
//! feed the `*_time_ns` fields stay in the module root next to their users.

use std::collections::HashMap;

/// Generates `ExecutionStats` plus its `to_hashmap` / `merge` impls from a
/// single field list. `: sum` accumulates on merge; `: snapshot` overwrites.
/// Adding a stat means editing only the invocation below.
macro_rules! define_execution_stats {
    (
        $(
            $(#[$attr:meta])*
            $field:ident: $mode:ident,
        )*
    ) => {
        /// Execution statistics for profiling.
        ///
        /// Tracks timing and counts for various operations during VEX execution.
        /// Times are in nanoseconds for precision.
        #[derive(Debug, Clone, Default)]
        pub(crate) struct ExecutionStats {
            $(
                $(#[$attr])*
                pub $field: u64,
            )*
        }

        impl ExecutionStats {
            /// Convert stats to a HashMap for Python exposure.
            pub(crate) fn to_hashmap(&self) -> HashMap<String, u64> {
                let mut map = HashMap::new();
                $(
                    map.insert(stringify!($field).to_string(), self.$field);
                )*
                map
            }

            /// Reset all statistics to zero.
            pub(crate) fn reset(&mut self) {
                *self = Self::default();
            }

            /// Merge another stats instance into this one.
            pub(crate) fn merge(&mut self, other: &ExecutionStats) {
                $(
                    define_execution_stats!(@merge_field self.$field, other.$field, $mode);
                )*
            }
        }
    };
    (@merge_field $self_field:expr, $other_field:expr, sum) => {
        $self_field += $other_field;
    };
    (@merge_field $self_field:expr, $other_field:expr, snapshot) => {
        $self_field = $other_field;
    };
}

define_execution_stats! {
    /// Number of load statements executed.
    load_stmt_count: sum,
    /// Time spent in load statements (nanoseconds).
    load_stmt_time_ns: sum,
    /// Number of store statements executed.
    store_stmt_count: sum,
    /// Time spent in store statements (nanoseconds).
    store_stmt_time_ns: sum,
    /// Number of `IRStmt::Exit` statements executed. Tallied by the `Exit` arm
    /// of `statements::VEXInterpreter::execute_stmt_with_callbacks`, counting
    /// every guarded exit *reached*, including the ones whose guard evaluates
    /// false and fall through.
    exit_stmt_count: sum,
    /// Time spent in exit statements (nanoseconds) — the whole of
    /// `statements::VEXInterpreter::handle_exit_stmt`, so it includes guard
    /// evaluation, `check_branch_feasibility` solver work, and the fork
    /// snapshot taken when both paths are satisfiable.
    exit_stmt_time_ns: sum,
    /// Number of Python callback invocations.
    python_callback_count: sum,
    /// Time spent in Python callbacks (nanoseconds).
    python_callback_time_ns: sum,
    /// Number of address concretizations performed.
    concretize_count: sum,
    /// Time spent in address concretization (nanoseconds).
    concretize_time_ns: sum,
    /// Number of IRSB cache hits.
    cache_hit_count: sum,
    /// Number of IRSB cache misses (lifts needed).
    cache_miss_count: sum,
    /// Number of IRSB cache evictions (capacity reached, LRU entry dropped).
    /// Tallied at every `block_cache.put()` site whose return value is `Some`
    /// AND the key was not already present (an overwrite, not an eviction).
    /// Together with `cache_hit_count`/`cache_miss_count` this gives the data
    /// needed to tune `BLOCK_CACHE_CAPACITY`: a high eviction-to-miss ratio
    /// indicates capacity pressure (working set exceeds cache); near-zero
    /// evictions on a benchmark mean the cache is oversized for that workload.
    cache_eviction_count: sum,
    /// Time spent lifting blocks (nanoseconds).
    lift_time_ns: sum,
    /// Number of loads served by the Rust-native memory layer — the
    /// `try_rust_memory_load` branch of `expressions::VEXInterpreter::load_layered`
    /// returning `Some`. Zero when `use_rust_memory` is off.
    rust_memory_load_count: sum,
    /// Number of loads that did NOT come from the Rust-native memory layer, i.e.
    /// the tail of `expressions::VEXInterpreter::load_layered`. Note this is
    /// wider than "Python callback": it also covers pending-store buffer hits,
    /// prefetch-cache hits and `concrete_memory` hits, which are the same
    /// layers `record_mem_load` is bumped for there. Together with
    /// `rust_memory_load_count` it partitions every `load_layered` call.
    fallback_memory_load_count: sum,
    /// Number of stores committed by the Rust-native memory layer — the
    /// `try_rust_memory_store` branch of the `Store` arm of
    /// `statements::VEXInterpreter::execute_stmt_with_callbacks` returning
    /// `true`. Zero when `use_rust_memory` is off. Only plain `IRStmt::Store`
    /// is counted; `StoreG` / `CAS` / `LLSC` go through their own handlers.
    rust_memory_store_count: sum,
    /// Number of `IRStmt::Store`s that fell through to
    /// `statements_store::VEXInterpreter::fallback_to_python_store`. As on the
    /// load side this is wider than "Python callback": that helper may instead
    /// park the write in the pending-store buffer
    /// (`handle_concrete_store` / `handle_symbolic_store`) to be flushed later.
    /// Together with `rust_memory_store_count` it partitions every plain
    /// `IRStmt::Store`.
    fallback_memory_store_count: sum,
    /// Number of expression evaluations.
    expr_eval_count: sum,
    /// Time spent evaluating expressions (nanoseconds).
    expr_eval_time_ns: sum,
    /// Number of blocks executed.
    blocks_executed: sum,
    /// Total execution time (nanoseconds).
    total_time_ns: sum,
    /// Time spent setting up interpreter per step (nanoseconds).
    step_setup_time_ns: sum,
    /// Number of exploration steps executed.
    step_count: sum,
    /// Number of solver satisfiability checks.
    solver_sat_count: sum,
    /// Time spent in solver satisfiability checks (nanoseconds).
    solver_sat_time_ns: sum,
    /// Time spent executing blocks (nanoseconds) — the inner VEX execution.
    block_exec_time_ns: sum,
    /// Number of statements executed.
    stmt_count: sum,
    /// Number of deferred forks PRESENTED to the exploration loop's
    /// post-block processing (input length of the `deferred_forks` Vec,
    /// summed across all four processing sites:
    /// `fork_materialize::materialize_deferred_forks` and its parallel
    /// mirror `core_outcome_handlers::materialize_deferred_forks_core`,
    /// plus `SimulationLoop::process_deferred_forks_into` and its mirror
    /// `core_outcome_handlers::process_deferred_forks_into_core`). NOT every
    /// entry produces a *tallied* solver clone: an entry whose condition is
    /// neither stored nor reconstructible is routed through a no-condition
    /// conservative `state.fork()` (which DOES clone the solver but is not
    /// tallied below), and neither `process_deferred_forks_into` variant
    /// tallies solver forks on either branch. This counter is therefore
    /// NOT directly comparable to `solver_fork_count` — they measure
    /// orthogonal but overlapping concepts. See angr-95up.2.
    deferred_fork_count: sum,
    /// Time spent processing deferred forks (nanoseconds).
    deferred_fork_time_ns: sum,
    /// Time spent in solver fork/clone operations (nanoseconds).
    solver_fork_time_ns: sum,
    /// Number of solver fork operations whose Z3-clone cost is timed by
    /// `solver_fork_time_ns`. Tallied at three sites: (1) the pre-callback
    /// state-snapshot fork taken in `SimulationLoop::dispatch_bounce`'s
    /// `BounceKind::Hook` arm before bouncing to a Python SimProcedure (NOT
    /// a deferred fork); (2) per-deferred-fork creation in
    /// `fork_materialize::materialize_deferred_forks` when a condition is
    /// available — either stored or reconstructed by
    /// `reconstruct_deferred_fork_condition`; (3) the parallel-scheduler
    /// mirror of (2) in
    /// `core_outcome_handlers::materialize_deferred_forks_core` (stored
    /// conditions only). NOT incremented by the no-condition
    /// conservative-fork fallback in either materializer, nor anywhere in
    /// `SimulationLoop::process_deferred_forks_into` /
    /// `core_outcome_handlers::process_deferred_forks_into_core`, which
    /// build their forks untimed on both branches. Because of (1), this
    /// counter generally exceeds the deferred-fork-with-condition subset of
    /// `deferred_fork_count`; because of the untallied paths, the
    /// relationship is not a simple sum. See angr-95up.2.
    solver_fork_count: sum,
    /// Number of active states at end of run.
    active_states_count: snapshot,
    /// Time spent in the main run() loop overhead (nanoseconds).
    run_loop_time_ns: sum,
    /// Number of dirty-helper invocations that fell back to the Python
    /// `call_dirty_call` callback (no native handler matched).
    python_dirty_call_count: sum,
    /// Number of VEX op evaluations (Unop/Binop/Triop/Qop) that reached
    /// `VEXInterpreter::vex_op_fallback` because `VEXOps::*` returned an
    /// `OpError` other than the explicitly-surfaced `UnsupportedNeon` /
    /// `UnsupportedVexOp` variants. By default the op is NOT evaluated
    /// natively at all: with a symbolic operand it returns
    /// `CbExecutionError::NeedPythonFallback` (the block IS re-run in
    /// Python), and with all-concrete operands it returns the typed
    /// `CbExecutionError::Op`, which routes the state to the errored
    /// stash. The silent fresh-symbolic synthesis this counter used to
    /// describe is now opt-in only, behind
    /// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` (off by default since
    /// angr-oyzvj) — see `vex_bypass_fabricate_count` for that subset.
    /// Also counts the three dispatch-fabricate families (VPerm/Pclmul*/
    /// Crc32C, angr-s6miz), which route to Python for concrete args too
    /// (angr-9ke6b.85).
    python_vex_op_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Unop`.
    python_vex_unop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Binop`,
    /// including the dispatch-fabricate families (angr-9ke6b.85).
    python_vex_binop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Triop`.
    python_vex_triop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Qop`.
    python_vex_qop_fallback_count: sum,
    /// Number of Unop/Binop evaluations that took the *fabricate-fresh-symbolic*
    /// BYPASS (the `any_sym` arm of `eval_unop`/`eval_binop`): the op returned an
    /// `OpError`, an input was symbolic, so a fresh unconstrained symbolic stood
    /// in for the real value. For those op arms it is a strict subset of
    /// `python_vex_op_fallback_count` (excludes the concrete-arg arm, which
    /// propagates a typed error). It ALSO counts the condition-flag arm of
    /// `eval_ccall` (angr-9ke6b.88), which fabricates under the same opt-in
    /// gate but does not bump `python_vex_op_fallback_count` — so the subset
    /// relation holds per-arm, not in aggregate. Likewise the
    /// no-handler-anywhere arm of `VEXInterpreter::handle_dirty_call`
    /// (angr-c7xno.44): same gate, same counter, no
    /// `python_vex_op_fallback_count` bump. Since
    /// angr-oyzvj this BYPASS is OPT-IN — it fires only when
    /// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` is set; by default the symbolic arm
    /// routes to Python (`NeedPythonFallback`), so this counter stays 0 unless
    /// the escape hatch is enabled. The three dispatch-fabricate families
    /// (VPerm/Pclmul*/Crc32C, angr-s6miz) are routed to Python fallback *before*
    /// this point, so they do NOT increment this counter. See the
    /// "Parse-succeeds / dispatch-fabricates (silent BYPASS)" section of
    /// `docs/extending-angr/rust_vex_ops.rst`.
    vex_bypass_fabricate_count: sum,
    /// Number of cold-block lifts served natively via the `libvex-ffi`
    /// `NativeLibVEXLifter` (gated on the `libvex-ffi` feature, which is not in
    /// cargo's `default` set but is appended by `setup.py`, so a stock
    /// `pip install -e .` build has it on). Each hit is a
    /// Python `lift_block` callback + JSON round-trip that did NOT happen.
    native_lift_count: sum,
    /// Number of native-lift *attempts* that fell back to the Python callback
    /// (feature off / disabled paths do not count — only a live attempt that
    /// could not complete: missing-or-symbolic block bytes, unsupported arch,
    /// or a libVEX lift error). A high fallback:hit ratio means the native
    /// path is not paying off for that workload.
    native_lift_fallback_count: sum,
    /// Subset of `native_lift_fallback_count`: misses at an address that lies
    /// outside every loaded binary region, so no lifter — native or pyvex — can
    /// produce a block. The Python callback returns the `"{}"` sentinel and the
    /// state deadends; nothing was lost by falling back. Subtract this from
    /// `native_lift_fallback_count` to get the misses that are genuinely lost
    /// native-lift wins. Measured on `cow_fork_scaling` (angr-op0dn.2.3), all
    /// 256 apparent fallbacks were return-to-0x0 deadend probes of exactly this
    /// kind — the native path was in fact serving 21 of 21 real blocks.
    native_lift_deadend_probe_count: sum,
}
