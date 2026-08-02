//! `#[pymethods]` surface for [`RustExplorationManager`].
//!
//! Split out of `mod.rs` (angr-nbim4.1) to keep the parent module thin. PyO3
//! (no `multiple-pymethods` feature) requires exactly one `#[pymethods]` block
//! per pyclass, but that single block may live in a child module: the
//! `#[pyclass]` struct stays in the parent and `use super::*` pulls in the
//! parent's imports, `pub(crate)` fields, and type aliases so the method
//! bodies compile unchanged.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[pymethods]
impl RustExplorationManager {
    /// Create a new exploration manager.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let arch_info = arch_from_name(arch)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {arch}")))?;

        let vex_arch = arch_info.vex_arch();

        Ok(RustExplorationManager {
            environment: ExecutionEnvironment::new(
                arch.to_string(),
                vex_arch,
                default_cc_for_arch(arch),
                little_endian,
            ),
            sm: StashManager::new(),
            find_addrs: HashSet::new(),
            avoid_addrs: HashSet::new(),
            stop_addrs: HashSet::new(),
            block_granular: false,
            materialize_unconstrained_forks: false,
            find_needs_python: false,
            avoid_needs_python: false,
            exec_config: ExecutionConfig::default(),
            callbacks: None,
            hooks: HashSet::new(),
            simprocedures: HashMap::new(),
            pending_callbacks: FxHashMap::default(),
            current_stepping_state_id: None,
            steps: 0,
            errors: Vec::new(),
            num_find: 1,
            max_steps_per_run: 5000,
            native_procedures: Arc::new(NativeProcedureRegistry::new()),
            native_syscalls: Arc::new(NativeSyscallRegistry::new()),
            vex_fallback_count: 0,
            vex_fallback_addrs: HashMap::new(),
            dcas_unsupported_count: 0,
            vecret_gsptr_fallback_count: 0,
            simprocedure_python_fallback_count: 0,
            simprocedure_fallback_by_name: HashMap::new(),
            syscall_python_fallback_count: 0,
            syscall_python_fallback_by_num: HashMap::new(),
            syscall_native_count: 0,
            syscall_native_by_num: HashMap::new(),
            dcas_warned_states: HashSet::new(),
            skip_hook_stack: Vec::new(),
            policy: Arc::new(selection_policy::Fifo), // Default to BFS (FIFO)
            constraint_solver: ConstraintSolver::new(),
            memory_config: MemoryConfiguration::default(),
            max_active_states: None,
            max_active_warned: false,
            deferred_forks_dropped: 0,
            native_techniques: Vec::new(),
            constraint_tracker: ConstraintTracker::default(),
            profiling: ProfilingCollector::default(),
            reconvergence_collision_states: 0,
            reconvergence_active_observed: 0,
            reconvergence_samples: 0,
            reconvergence_max_group: 0,
            states_merged_native: 0,
            parallel_num_workers: std::env::var("ANGR_PARALLEL_WORKERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&w| w >= 1)
                .unwrap_or(4),
            parallel_real_workers: std::env::var("RUST_PARALLEL_WORKERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&w| w >= 1)
                .unwrap_or(1),
            parallel_migrations: 0,
            parallel_tasks: 0,
            parallel_reattaches: 0,
            parallel_bounce_roundtrips: 0,
            parallel_resume_reinjects: 0,
            parallel_max_active_width: 0,
            parallel_width_hist: [0; 5],
            parallel_worker_dispatch: Vec::new(),
            parallel_worker_dispatch_folded: 0,
            parallel_worker_of: HashMap::new(),
            shadow_probe: std::env::var("RUST_PARALLEL_SHADOW_PROBE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            parallel_shadow_migration_ns: 0,
            parallel_shadow_migration_states: 0,
            parallel_shadow_migration_bytes: 0,
            shadow_probe_chan: None,
            pending_parallel_bounces: Vec::new(),
            #[cfg(feature = "vex-engine-z3")]
            parallel_pool: None,
            #[cfg(feature = "vex-engine-z3")]
            parallel_session: None,
            parallel_frontier_residency: false,
            parallel_steady_env: std::env::var("RUST_PARALLEL_STEADY")
                .map(|v| v == "1")
                .unwrap_or(false),
            parallel_residual_drains: 0,
            parallel_steady_budget_yields: 0,
            parallel_post_cancel_steps: 0,
        })
    }

    // =========================================================================
    // PyAPI methods (thin getters/accessors kept inline in this single
    // #[pymethods] block — larger bodies live in sibling modules per
    // `invariant-pyo3-single-pymethods-impl`)
    // =========================================================================

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.environment.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.sm
            .get(STASH_ACTIVE)
            .map(std::collections::VecDeque::len)
            .unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.sm
            .stashes()
            .get(STASH_FOUND)
            .map(std::collections::VecDeque::len)
            .unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in self.sm.stashes() {
            dict.set_item(name, stash.len())?;
        }
        Ok(dict)
    }

    /// Set find addresses.
    pub fn set_find_addrs(&mut self, addrs: Vec<u64>) {
        self.steady_config_guard();
        self.find_addrs = addrs.into_iter().collect();
        self.find_needs_python = false;
        self.rebuild_stop_addrs();
    }

    /// Set avoid addresses.
    pub fn set_avoid_addrs(&mut self, addrs: Vec<u64>) {
        self.steady_config_guard();
        self.avoid_addrs = addrs.into_iter().collect();
        self.avoid_needs_python = false;
        self.rebuild_stop_addrs();
    }

    /// Recompute the union of find + avoid addresses (the interpreter's
    /// block-chain stop set). Called whenever either set changes.
    fn rebuild_stop_addrs(&mut self) {
        self.stop_addrs = self
            .find_addrs
            .iter()
            .chain(self.avoid_addrs.iter())
            .copied()
            .collect();
    }

    /// Mark that find condition has callable predicates (needs Python).
    pub fn set_find_needs_python(&mut self, needs: bool) {
        self.steady_config_guard();
        self.find_needs_python = needs;
    }

    /// Mark that avoid condition has callable predicates (needs Python).
    pub fn set_avoid_needs_python(&mut self, needs: bool) {
        self.steady_config_guard();
        self.avoid_needs_python = needs;
    }

    /// angr-nkoct steady-state opt-in: the Python driver's promise that nothing
    /// reads or mutates the ACTIVE stash between `run()` calls, which is what
    /// makes it safe for worker-local frontiers to stay resident across a
    /// `need_callback` return. rust_manager.py sets this per explore loop
    /// (address-based explore without `until`/techniques); every other driver
    /// path must leave it false. Turning it off finalizes any live session.
    pub fn set_parallel_frontier_residency(&mut self, enabled: bool) {
        if !enabled {
            self.steady_config_guard();
        }
        self.parallel_frontier_residency = enabled;
    }

    /// Finalize a live steady-state session, if any (angr-nkoct): cancel it,
    /// drain every worker's residual frontier and the injector surplus back
    /// into the active stash, and fold its counters. rust_manager.py calls
    /// this at explore-loop exits so a timeout/error break never leaves states
    /// parked inside worker Z3 contexts. No-op without a live session.
    ///
    /// Also flushes any wave-parked bounce queue back to `STASH_ACTIVE`
    /// (angr-05kiw): those states live in NO stash, so an explore() that ends
    /// with bounces parked would otherwise under-report the resumable frontier
    /// in `stash_counts()` relative to the serial loop. Unlike the steady
    /// drain, this half is not session-gated.
    #[cfg(feature = "vex-engine-z3")]
    pub fn finalize_parallel_session(&mut self, py: Python<'_>) -> PyResult<()> {
        self.finalize_steady_session(py)?;
        self.flush_parked_bounces_to_active();
        Ok(())
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn finalize_parallel_session(&mut self) -> PyResult<()> {
        self.flush_parked_bounces_to_active();
        Ok(())
    }

    /// Whether a steady-state session is currently live (angr-nkoct). True only
    /// while worker frontiers are resident across a `need_callback` return, so
    /// some live states are inside worker Z3 contexts rather than any stash.
    /// The Python driver reads this to skip state-cache cleanup that would
    /// mis-classify a resident state as dead. Always false on non-Z3 builds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn parallel_session_active(&self) -> bool {
        self.parallel_session.is_some()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn parallel_session_active(&self) -> bool {
        false
    }

    /// P9 fix: Set state selection to LIFO (DFS - depth-first search).
    pub fn set_state_selection_lifo(&mut self) {
        self.steady_config_guard();
        self.policy = Arc::new(selection_policy::Lifo);
        log::debug!("State selection set to LIFO (DFS)");
    }

    /// P9 fix: Set state selection to FIFO (BFS - breadth-first search).
    pub fn set_state_selection_fifo(&mut self) {
        self.steady_config_guard();
        self.policy = Arc::new(selection_policy::Fifo);
        log::debug!("State selection set to FIFO (BFS)");
    }

    /// angr-a32jl.2 prototype: set state selection to random-state — step a
    /// uniformly-random active state, seeded for reproducibility. Opt-in only;
    /// never a default. `seed` fixes the SplitMix64 stream so a run is
    /// byte-reproducible.
    pub fn set_state_selection_random(&mut self, seed: u64) {
        self.steady_config_guard();
        self.policy = Arc::new(selection_policy::RandomState::new(seed));
        log::debug!("State selection set to RANDOM (seed={seed})");
    }

    /// angr-m9fpp prototype: set state selection to coverage-guided
    /// new-block-first — step the oldest active state parked on a block never
    /// dispatched before, degrading to FIFO once all active blocks are seen.
    /// Opt-in only; never a default.
    pub fn set_state_selection_coverage(&mut self) {
        self.steady_config_guard();
        self.policy = Arc::new(selection_policy::CoverageGuided::new());
        log::debug!("State selection set to COVERAGE (new-block-first)");
    }

    /// angr-caplg prototype: set state selection to loop-head round-robin —
    /// rotate dispatch across (loop-head, callstack-class) buckets so a looping
    /// state cannot starve sibling paths. Opt-in only; never a default.
    pub fn set_state_selection_loop_head(&mut self) {
        self.steady_config_guard();
        self.policy = Arc::new(selection_policy::LoopHeadRoundRobin::new());
        log::debug!("State selection set to LOOP_HEAD (round-robin fairness)");
    }

    /// angr-a32jl.4: set state selection to CFG-distance directed beam search.
    /// `distances` is a one-time `addr -> distance-to-target` snapshot computed
    /// Python-side from the angr CFG and shipped in as plain metadata (zero
    /// runtime bounces); `beam_width` (default 2 from the Python setter) keeps
    /// best-first out of the greedy trap on data-dependent targets. Opt-in
    /// only; never a default.
    pub fn set_state_selection_directed(
        &mut self,
        distances: std::collections::HashMap<u64, u64>,
        beam_width: usize,
    ) {
        self.steady_config_guard();
        let n = distances.len();
        self.policy = Arc::new(selection_policy::DirectedCfgDistance::new(
            distances, beam_width,
        ));
        log::debug!("State selection set to DIRECTED (beam={beam_width}, {n} mapped blocks)");
    }

    /// angr-lnzcu: set state selection to find-directed novelty/CFG-distance —
    /// under `num_find == 1`, dispatch the active state most likely to reach a
    /// find target first, ranking novel (never-dispatched) blocks ahead of seen
    /// ones and steering the novel frontier by `distances` (an
    /// `addr -> distance-to-find` snapshot computed Python-side from the angr
    /// CFG, zero runtime bounces). Novelty is the anti-greedy-trap mechanism, so
    /// no beam width is needed. Opt-in only; never a default.
    pub fn set_state_selection_find_directed(
        &mut self,
        distances: std::collections::HashMap<u64, u64>,
    ) {
        self.steady_config_guard();
        let n = distances.len();
        self.policy = Arc::new(selection_policy::FindDirected::new(distances));
        log::debug!("State selection set to FIND_DIRECTED ({n} mapped blocks)");
    }

    /// Set the number of solutions to find before stopping.
    pub fn set_num_find(&mut self, n: usize) {
        self.steady_config_guard();
        self.num_find = n;
    }

    /// angr-op0dn.13.5 (M5-B16): programmatic parallel-workers setter, the
    /// non-env twin of the `RUST_PARALLEL_WORKERS` gate read once in `new()`.
    /// `rust_manager.py` calls this from the `parallel_workers=` constructor
    /// kwarg so a caller can request real parallel workers without setting an
    /// env var. `n < 1` clamps to 1 (single-threaded). The env var, when set,
    /// wins: `_engage_parallel_workers` on the Python side skips this call so
    /// benches keep their env override. Honors `steady_config_guard` like every
    /// other exploration-config mutation.
    pub fn set_parallel_workers(&mut self, n: usize) {
        self.steady_config_guard();
        self.parallel_real_workers = n.max(1);
    }

    /// Read-back of the effective real-worker count (env override or the
    /// programmatic `set_parallel_workers` value). Lets the Python driver
    /// observe the engaged worker count without duplicating the env-precedence
    /// logic.
    pub fn parallel_workers(&self) -> usize {
        self.parallel_real_workers
    }

    /// Set maximum steps per run iteration.
    pub fn set_max_steps_per_run(&mut self, n: u32) {
        self.steady_config_guard();
        self.max_steps_per_run = n;
    }

    /// Enable lazy solves mode (skip satisfiability checks on forks).
    pub fn set_lazy_solves(&mut self, enabled: bool) {
        self.steady_config_guard();
        self.constraint_solver.lazy_solves = enabled;
    }

    /// Enable zero-fill for unconstrained memory reads.
    /// When true, unmapped memory returns zero instead of fresh symbolic values.
    pub fn set_zero_fill_unconstrained(&mut self, enabled: bool) {
        self.steady_config_guard();
        self.memory_config.zero_fill_unconstrained = enabled;
    }

    /// Toggle deferred-fork mode for the whole manager (angr-027h two-phase
    /// explore). `true` (the default) is the fast deferred path: forward-branch
    /// loop exits are deferred and the loop-continuation is taken as the main
    /// chain. `false` makes every fork materialize eagerly (BFS), which lets a
    /// find-guided search reach a target behind a symbolic loop exit at the cost
    /// of a wider active stash. Python's `_explore_with_addresses` runs phase 1
    /// deferred and, only if it exhausts to `active_empty` without finding,
    /// re-seeds the initial states with this set to `false`.
    pub fn set_use_deferred_forks(&mut self, enabled: bool) {
        // Steady-state note: today's only caller mid-explore is
        // `_maybe_phase2_eager_retry`, which fires on `active_empty` — where
        // the session is already finalized — so this guard is a no-op there;
        // it exists for any future caller that flips the mode mid-session.
        self.steady_config_guard();
        self.exec_config.use_deferred_forks = enabled;
    }

    /// Toggle block-granular stepping (angr-bmyx). When `true`, the VEX
    /// interpreter stops chaining basic blocks and returns to the step boundary
    /// after every block, so each `step(n=1)` advances exactly one block and
    /// every interior pc is observable — matching Python angr's block-granular
    /// `step()`. This is what lets a bare step-loop (e.g. CADET solve.py phase
    /// 3) detect a mid-path target address that the chained interpreter would
    /// run straight through; address-based `explore(find=...)` does not need it
    /// because `set_find_addrs` already breaks the chain at those specific
    /// addresses. Default `false` keeps chaining on for `explore()` and
    /// benchmark throughput. Returns the previous value so callers (e.g. a
    /// scoped step-loop) can restore it.
    pub fn set_block_granular(&mut self, enabled: bool) -> bool {
        self.steady_config_guard();
        let prev = self.block_granular;
        self.block_granular = enabled;
        prev
    }

    /// Whether block-granular stepping is currently enabled (angr-bmyx).
    pub fn block_granular(&self) -> bool {
        self.block_granular
    }

    /// Toggle materialization of loop-exit deferred forks at an
    /// `UnconstrainedJump` (angr-ckdy). When enabled, the forks that
    /// deferred-fork mode would otherwise DROP are instead materialized
    /// eagerly and routed to the active stash, so a bare step-loop that
    /// bypasses `explore()` (CADET solve.py phase 3) keeps progressing toward
    /// a target behind a symbolic loop exit instead of collapsing to
    /// `active_empty` and spinning. Pair with `set_block_granular(true)` so
    /// the target block is observable before the materialized subtree
    /// explodes. Default `false`: dropping is what `explore()`'s two-phase
    /// eager retry relies on, so this stays opt-in. Returns the previous value.
    pub fn set_materialize_unconstrained_forks(&mut self, enabled: bool) -> bool {
        self.steady_config_guard();
        let prev = self.materialize_unconstrained_forks;
        self.materialize_unconstrained_forks = enabled;
        prev
    }

    /// Whether unconstrained-fork materialization is enabled (angr-ckdy).
    pub fn materialize_unconstrained_forks(&self) -> bool {
        self.materialize_unconstrained_forks
    }

    /// Cumulative number of loop-exit deferred forks dropped at an
    /// `UnconstrainedJump` while deferred-fork mode was active (angr-ckdy).
    /// Non-resetting: a step-driven `explore()`-bypassing loop (CADET solve.py
    /// phase 3) polls this after `active_empty` to decide whether the stash
    /// collapsed because egg-reaching forks were discarded — if so the Python
    /// wrapper re-seeds in eager mode. Returns 0 when no forks were ever
    /// dropped (genuine exhaustion), so a spurious eager re-run is avoided.
    pub fn deferred_forks_dropped(&self) -> u64 {
        self.deferred_forks_dropped
    }

    /// Set the Z3 solver timeout in milliseconds (default:
    /// [`DEFAULT_SOLVER_TIMEOUT_MS`](crate::symbolic::DEFAULT_SOLVER_TIMEOUT_MS)).
    pub fn set_solver_timeout(&mut self, timeout_ms: u32) {
        // Workers snapshot the solver config into their StepContext, so a
        // mid-session change would never reach a live steady session; finalize
        // it first like the other guarded config mutators (angr-1yge9.3).
        self.steady_config_guard();
        self.constraint_solver.solver_timeout_ms = timeout_ms;
    }

    /// Turn strict-deterministic witness selection on or off (angr-op0dn.10.3).
    ///
    /// With it on, every state entering a stash — and, by inheritance, every
    /// fork of one — evaluates with `SymContext`'s canonical-witness path:
    /// `eval` returns the unsigned minimum of the feasible set and `eval_upto`
    /// its ascending prefix, so a truncated result is reproducible instead of
    /// being whatever model Z3 happened to build. States already in a stash
    /// are updated in place, so the order of this call relative to
    /// `add_state` does not matter.
    ///
    /// Costs Z3 checks (an `O(log width)` binary search per witness), hence
    /// opt-in. Note this makes *witness choice* deterministic; with more than
    /// one real scheduler worker the steal order still varies, so the found
    /// set is stable but the order states are reported in is not.
    pub fn set_deterministic(&mut self, v: bool) {
        // Finalize a live steady session first: workers snapshot the solver
        // config and resident/parked-pending states never appear in the
        // stashes we iterate below, so without the guard a mid-session flip
        // would leave those states minting non-canonical witnesses.
        self.steady_config_guard();
        self.constraint_solver.deterministic = v;
        for states in self.sm.stashes_mut().values_mut() {
            for state in states.iter() {
                constraints::apply_state_deterministic(state, v);
            }
        }
        log::debug!("Strict-deterministic witness selection set to {v}");
    }

    /// Whether strict-deterministic witness selection is on (angr-op0dn.10.3).
    pub fn is_deterministic(&self) -> bool {
        self.constraint_solver.deterministic
    }

    /// Whether one state's solver is in strict-deterministic mode. The
    /// round-trip probe for `set_deterministic`: the manager-level flag is
    /// only meaningful if it actually reached the per-state `SymContext`.
    pub fn state_is_deterministic(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| {
            Ok(constraints::state_is_deterministic(state))
        })
    }

    /// Set the maximum number of states in the active stash.
    /// When the limit is reached, new forked states are pruned to avoid OOM.
    /// None (default) means no limit.
    ///
    /// Guarded like every sibling exploration-config setter (angr-9ke6b.53):
    /// `ensure_steady_session` hands the cap to `RunSession::new_with_policy`
    /// once, for the session's whole life, so a mid-steady mutation would
    /// otherwise be silently ignored by the resident frontier while the
    /// manager reported the new value.
    #[pyo3(signature = (limit=None))]
    pub fn set_max_active_states(&mut self, limit: Option<usize>) {
        self.steady_config_guard();
        self.max_active_states = limit;
    }

    /// Get the current max_active_states limit.
    pub fn get_max_active_states(&self) -> Option<usize> {
        self.max_active_states
    }

    /// Set the global VEX optimization level (0-3).
    /// None = use pyvex default (typically 1).
    /// Level 0: no optimization. Level 1: standard. Level 2-3: aggressive.
    #[pyo3(signature = (level=None))]
    pub fn set_vex_opt_level(&mut self, level: Option<i32>) {
        self.steady_config_guard();
        self.memory_config.vex_opt_level = level;
        // Invalidate block cache since opt_level affects IR output
        self.environment.block_cache.clear();
    }

    /// Get the current VEX optimization level.
    pub fn get_vex_opt_level(&self) -> Option<i32> {
        self.memory_config.vex_opt_level
    }

    /// Enable/disable native (in-process) libVEX cold-block lifting.
    ///
    /// Propagated to every per-step `VEXInterpreter` (z087y Stage-2). Only
    /// effective on a `libvex-ffi` build; on the default build the
    /// interpreter setter is a no-op stub. Python gates the `True` call on
    /// `libvex_ffi_enabled()` + AMD64, so a mistaken enable on an
    /// unsupported build is inert either way.
    pub fn set_native_lift_enabled(&mut self, enabled: bool) {
        self.steady_config_guard();
        self.memory_config.native_lift_enabled = enabled;
    }

    /// Set a per-address VEX optimization level override.
    /// Blocks at this address will be lifted with the specified opt_level.
    pub fn set_vex_opt_level_override(&mut self, addr: u64, level: i32) {
        self.steady_config_guard();
        self.memory_config
            .vex_opt_level_overrides
            .insert(addr, level);
        // Remove this address from block cache since opt_level changed
        self.environment.block_cache.pop(&addr);
    }

    /// Remove a per-address VEX optimization level override.
    pub fn remove_vex_opt_level_override(&mut self, addr: u64) {
        self.memory_config.vex_opt_level_overrides.remove(&addr);
        self.environment.block_cache.pop(&addr);
    }

    /// Clear all per-address VEX optimization level overrides.
    pub fn clear_vex_opt_level_overrides(&mut self) {
        let addrs: Vec<u64> = self
            .memory_config
            .vex_opt_level_overrides
            .keys()
            .copied()
            .collect();
        self.memory_config.vex_opt_level_overrides.clear();
        for addr in addrs {
            self.environment.block_cache.pop(&addr);
        }
    }

    /// Resolve the VEX optimization level for a given address.
    /// Per-address overrides take precedence over the global level.
    pub fn resolve_vex_opt_level(&self, addr: u64) -> Option<i32> {
        self.memory_config
            .vex_opt_level_overrides
            .get(&addr)
            .copied()
            .or(self.memory_config.vex_opt_level)
    }

    /// Set whether to drop terminal states (avoid/pruned/deadended) immediately.
    /// When true (default), terminal states are dropped to save memory.
    /// Set to false when states need to be recovered (e.g., factory.callable()).
    pub fn set_drop_terminal_states(&mut self, enabled: bool) {
        self.sm.set_drop_terminal_states(enabled);
    }

    /// Configure address concretization strategies to match Python's configuration.
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `read_range_limit` - Range limit for read strategies (default: 1024)
    /// * `write_range_limit` - Range limit for write strategies (default: 128)
    /// * `symbolic_write_addresses` - Whether SYMBOLIC_WRITE_ADDRESSES is enabled
    /// * `avoid_multivalued_reads` - Whether AVOID_MULTIVALUED_READS is enabled
    /// * `avoid_multivalued_writes` - Whether AVOID_MULTIVALUED_WRITES is enabled
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false, avoid_multivalued_reads=false, avoid_multivalued_writes=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
        avoid_multivalued_reads: bool,
        avoid_multivalued_writes: bool,
    ) {
        self.memory_config.concretizer_config.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
            avoid_multivalued_reads,
            avoid_multivalued_writes,
        );
    }

    /// Read back the current address-concretization configuration as a dict.
    ///
    /// Mirrors the fields set by `configure_concretization_strategies`, with
    /// boolean flags encoded as `0`/`1`. Lets Python tests assert that
    /// SimOption propagation (APPROXIMATE_MEMORY_INDICES / SYMBOLIC_WRITE_ADDRESSES /
    /// AVOID_MULTIVALUED_*) and the read/write `_limit` sniffing in
    /// `rust_manager._add_rust_state` reached the Rust-side concretizer
    /// rather than silently no-op'ing on a positional-arg swap.
    pub fn get_concretization_config(&self) -> HashMap<String, u64> {
        let c = &self.memory_config.concretizer_config;
        let mut m = HashMap::new();
        m.insert("use_approximate".to_string(), c.use_approximate as u64);
        m.insert(
            "symbolic_write_addresses".to_string(),
            c.symbolic_write_addresses as u64,
        );
        m.insert(
            "avoid_multivalued_reads".to_string(),
            c.avoid_multivalued_reads as u64,
        );
        m.insert(
            "avoid_multivalued_writes".to_string(),
            c.avoid_multivalued_writes as u64,
        );
        m.insert("read_range_limit".to_string(), c.read_range_limit);
        m.insert("write_range_limit".to_string(), c.write_range_limit);
        m
    }

    /// Enable or disable Rust-side profiling.
    /// When enabled, per-step timing and counters are accumulated.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling.profiling_enabled = enabled;
        // angr-1ilq.7: the GIL/wall accumulators are deliberately NOT reset
        // here. A bench runs in its own process and may build several managers
        // (one per `simulation_manager()` call / exploration phase); we want the
        // process-cumulative GIL vs run-loop-wall totals across ALL of them, so
        // the last manager's `stats()` reports the whole-bench fraction. The
        // thread-locals start at zero per process, so there is no cross-bench
        // contamination. `gil_profile::reset()` is `#[cfg(test)]` precisely so
        // this stays true — it exists to isolate unit tests, not to be wired in
        // here (angr-9ke6b.218 item 3).
    }

    /// Set the maximum length of each state's `history` / `detailed_history`
    /// ring buffers. 0 means unlimited (legacy behavior — can OOM on long
    /// explorations). Default is 1000. The new value is applied to every
    /// state already in any stash, plus any future state created via this
    /// manager.
    pub fn set_max_history(&mut self, max: usize) {
        self.environment.max_history = max;
        for stash in self.sm.stashes_mut().values_mut() {
            for state in stash.iter_mut() {
                state.set_max_history(max);
            }
        }
    }

    /// Get the current per-state max_history value. 0 = unlimited.
    pub fn get_max_history(&self) -> usize {
        self.environment.max_history
    }

    /// Set the OS / SimOS name. Defaults to `"linux"`; pass `"cgc"` for
    /// DECREE binaries so the syscall dispatcher routes to the CGC ABI
    /// table instead of the per-arch Linux tables. Case-insensitive; the
    /// value is lowercased before storage so callers can pass `"CGC"` or
    /// `"Linux"` interchangeably.
    pub fn set_os_name(&mut self, name: String) {
        // os_name feeds the syscall-dispatch ABI selection baked into a
        // worker's snapshotted StepContext, so finalize a live steady session
        // before changing it like the other guarded config mutators
        // (angr-1yge9.3).
        self.steady_config_guard();
        self.environment.os_name = name.to_lowercase();
    }

    /// Get the current OS / SimOS name (lowercase).
    pub fn get_os_name(&self) -> &str {
        &self.environment.os_name
    }

    /// Get accumulated execution statistics as a dict.
    pub fn get_execution_stats(&self) -> HashMap<String, u64> {
        self.profiling.accumulated_stats.to_hashmap()
    }

    /// Reset accumulated execution statistics.
    pub fn reset_execution_stats(&mut self) {
        self.profiling.accumulated_stats.reset();
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
    }

    /// GC traversal: visit Python callback refs held inside the cloned
    /// PythonCallbacks struct. The Python wrapper `mgr` owns
    /// `mgr._rust_mgr` (this object), and via `set_callbacks` this object
    /// holds a cloned PythonCallbacks whose Py<PyAny> bound methods point
    /// back at `mgr` — a non-trivial cycle that cycle-GC can break only if
    /// __traverse__/__clear__ are exposed.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(cbs) = &self.callbacks {
            cbs.traverse_fields(&visit)?;
        }
        Ok(())
    }

    /// GC clear: drop the bound-method refs held inside the cloned
    /// PythonCallbacks struct. After this returns, the engine can no
    /// longer call back into Python — but cycle-GC only invokes __clear__
    /// when the object is being collected, so further callbacks would not
    /// be issued.
    fn __clear__(&mut self) {
        if let Some(cbs) = &mut self.callbacks {
            cbs.clear_fields();
        }
    }

    /// Add a hook address.
    ///
    /// Guarded like `register_simprocedures`: a live steady session snapshots
    /// `hooks` into each worker's `StepContext`, so mutating the set mid-run
    /// must finalize first or the new hook silently never fires.
    pub fn add_hook(&mut self, addr: u64) {
        self.steady_config_guard();
        self.hooks.insert(addr);
    }

    /// Add multiple hook addresses.
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        self.steady_config_guard();
        for addr in addrs {
            self.hooks.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    /// Register a SimProcedure.
    #[pyo3(signature = (addr, name, num_args=0, no_return=false))]
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        self.steady_config_guard();
        self.hooks.insert(addr);
        self.simprocedures.insert(addr, (name, num_args, no_return));
    }

    /// Register multiple SimProcedures.
    ///
    /// Steady-state note: a live session's workers snapshot `hooks` /
    /// `simprocedures` into their `StepContext` at session creation, so a
    /// mid-session change here MUST finalize first (the guard) or workers
    /// would keep stepping against the stale hook set. Continuation
    /// SimProcedures re-hook mid-run by design, so continuation-heavy
    /// workloads finalize repeatedly — steady mode degrades gracefully there.
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        self.steady_config_guard();
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
    }

    /// Unregister multiple SimProcedures (e.g. after `proj.unhook(addr)` on a
    /// live manager). Removes each address from both the hook set and the
    /// SimProcedure table so a stale hook no longer fires (angr-969g).
    pub fn unregister_simprocedures(&mut self, addrs: Vec<u64>) {
        self.steady_config_guard();
        for addr in addrs {
            self.hooks.remove(&addr);
            self.simprocedures.remove(&addr);
        }
    }

    /// Load binary code regions.
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) {
        self.environment.binary_regions = regions
            .into_iter()
            .map(|(base, data)| (base, Arc::new(data)))
            .collect();
    }

    /// Record the main object's `[start, end)` code span. Hooks inside it always
    /// dispatch to Python (user `proj.hook()` overrides); see
    /// `execution_env::prefer_native_dispatch`.
    pub fn set_main_object_range(&mut self, start: u64, end: u64) {
        self.environment.main_object_range = Some((start, end));
    }

    /// Opt-in: prefer native procedures for `use_sim_procedures` library hooks
    /// that land inside a non-main loaded object (angr-a8epx / angr-gorvf.3.2).
    pub fn set_prefer_native_library_hooks(&mut self, enabled: bool) {
        self.environment.prefer_native_library_hooks = enabled;
    }

    /// Create a new RustSimState and add it to a stash.
    /// See `state_lifecycle::_create_state` for the body.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        self._create_state(stash)
    }

    /// Add an existing RustSimState to a stash.
    /// See `state_lifecycle::_add_state` for the body.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        self.steady_config_guard();
        self._add_state(stash, state)
    }

    /// Merge multiple states into one using symbolic merge conditions.
    ///
    /// Each state's constraints are guarded by a fresh 1-bit merge flag.
    /// Registers and memory that differ between states become ITE expressions.
    /// The merged state is placed into `dest_stash`.
    ///
    /// Returns the merged state's ID.
    /// See `state_lifecycle::_merge_states` for the body.
    #[pyo3(signature = (state_ids, dest_stash="active"))]
    pub fn merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        self._merge_states(state_ids, dest_stash)
    }

    /// Fork an existing state (including the pending callback state) and add
    /// the fork to `stash`. Returns the new state's ID. Inherits the parent's
    /// lineage root.
    ///
    /// Write-through SimProc fork API (angr-t3mr). See
    /// `state_lifecycle::_fork_state_to_stash` for the body.
    #[pyo3(signature = (parent_id, stash="active"))]
    pub fn fork_state_to_stash(&mut self, parent_id: u64, stash: &str) -> PyResult<u64> {
        self._fork_state_to_stash(parent_id, stash)
    }

    /// Get the PC of a state in a stash by index.
    ///
    /// Applies the same angr-4rq7 pc==0 IP fallback as `get_state_pc_by_id` so
    /// the two accessors agree for a stale-pc forked state (angr-ph300.23).
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.sm
            .get(stash)
            .and_then(|s| s.get(index))
            .map(super::native_proc_dispatch::effective_pc)
    }

    /// Get the PC of a state by its ID (O(1) via state index, no full export).
    pub fn get_state_pc_by_id(&self, state_id: u64) -> Option<u64> {
        // find_state already checks pending_callback first.
        // angr-4rq7 (root cause #2): see `native_proc_dispatch::effective_pc` for why a
        // pc==0 state falls back to the IP register.
        self.find_state(state_id)
            .map(super::native_proc_dispatch::effective_pc)
    }

    /// Get the tail of a state's bbl history (last `n` addresses).
    /// Avoids cloning the full Vec on long benches where max_history=0 means
    /// `history` may grow to 100k+ entries.
    /// `n == 0` is treated as "all entries". Returns None if state not found.
    #[pyo3(signature = (state_id, n=256))]
    pub fn get_state_bbl_history_tail(&self, state_id: u64, n: usize) -> Option<Vec<u64>> {
        let hist = self.find_state(state_id)?.history();
        let start = if n == 0 {
            0
        } else {
            hist.len().saturating_sub(n)
        };
        Some(hist.range(start..).copied().collect())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.sm
            .get(stash)
            .map(|s| s.iter().map(crate::state::RustSimState::state_id).collect())
            .unwrap_or_default()
    }

    /// State ids of the callbacks currently parked in `pending_callbacks`
    /// (angr-op0dn.13.14). A parked callback's state lives in NO stash, so it
    /// is invisible to `get_state_ids` / `stash_counts` — this is the only way
    /// for Python (and the snapshot tests) to observe it.
    pub fn pending_callback_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.pending_callbacks.keys().map(|k| k.raw()).collect();
        ids.sort_unstable();
        ids
    }

    /// Get (state_id, addr, stdout_len) tuples for states in a stash.
    /// Used by Python predicate caching to skip re-evaluation when
    /// a state's address and stdout haven't changed.
    ///
    /// `addr` goes through `native_proc_dispatch::effective_pc`: a stale-pc forked successor
    /// would otherwise be cached under key `(sid, 0)` and its find predicate
    /// evaluated at address 0, missing the genuine find (angr-ph300.23).
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_predicate_info(&self, stash: &str) -> Vec<(u64, u64, usize)> {
        self.sm
            .get(stash)
            .map(|s| {
                s.iter()
                    .map(|state| {
                        (
                            state.state_id(),
                            super::native_proc_dispatch::effective_pc(state),
                            state.stdout_buffer().len(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if there are any active states (O(1), no allocation).
    pub fn has_active_states(&self) -> bool {
        self.sm.get(STASH_ACTIVE).is_some_and(|s| !s.is_empty())
    }

    /// Get the number of states in a stash (O(1), no allocation).
    #[pyo3(signature = (stash="active"))]
    pub fn stash_count(&self, stash: &str) -> usize {
        self.sm
            .get(stash)
            .map_or(0, std::collections::VecDeque::len)
    }

    /// Get the stash name a state currently belongs to (O(1) via state index).
    /// Returns None if the state isn't found in any stash. Used by
    /// RustStateProxy.__repr__ for cheap REPL debugging output.
    pub fn state_stash(&self, state_id: u64) -> Option<String> {
        self.sm
            .stash_of(state_id)
            .map(std::string::ToString::to_string)
    }

    /// Get the number of solver constraints for a state (O(1), reads
    /// the SymContext's atomic counter — no Z3 traversal). Returns None
    /// if the state isn't found. Used by RustStateProxy.__repr__.
    pub fn state_constraint_count(&self, state_id: u64) -> Option<usize> {
        let state = self.find_state(state_id)?;
        Some(state.solver().borrow().num_constraints())
    }

    /// Rebuild the state index after run() modifies stashes internally.
    /// Call from Python after run() returns to keep index up to date.
    pub fn sync_state_index(&mut self) {
        self.rebuild_state_index();
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.sm.roots().get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    /// See `pending_api::_set_pending_state_pc` for the body.
    pub fn set_pending_state_pc(&mut self, state_id: u64, pc: u64) -> PyResult<()> {
        self._set_pending_state_pc(state_id, pc)
    }

    /// Map memory in active states.
    /// See `pending_api::_active_states_map_memory` for the body.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        // Steady-state: this iterates STASH_ACTIVE, which misses a resident
        // frontier — finalize first so every live state is back in the stash.
        self.steady_config_guard();
        self._active_states_map_memory(addr, data, permissions)
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    /// See `pending_api::_get_pending_branch_condition` for the body.
    pub fn get_pending_branch_condition(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Py<PyAny>> {
        self._get_pending_branch_condition(py, state_id)
    }

    /// Get register value from pending state (concrete only).
    /// See `pending_api::_get_pending_register` for the body.
    pub fn get_pending_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self._get_pending_register(state_id, name)
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    /// See `pending_api::_get_pending_register_ast` for the body.
    pub fn get_pending_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        self._get_pending_register_ast(py, state_id, name)
    }

    /// Get history (BBL addresses) and jumpkind from pending callback state
    /// in a single FFI call. Fetching both in one crossing avoids the GIL +
    /// boundary-crossing cost of two separate calls from the callback
    /// dispatcher hot path. Used by Python to initialize history + callstack
    /// on callback states, preventing IndexError when hooks access
    /// `state.history.recent_bbl_addrs[-1]`.
    /// See `pending_api::_get_pending_history_and_jumpkind` for the body.
    pub fn get_pending_history_and_jumpkind(&self, state_id: u64) -> PyResult<(Vec<u64>, String)> {
        self._get_pending_history_and_jumpkind(state_id)
    }

    /// Set register value in pending state.
    /// See `pending_api::_set_pending_register` for the body.
    pub fn set_pending_register(&mut self, state_id: u64, name: &str, value: u128) -> PyResult<()> {
        self._set_pending_register(state_id, name, value)
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    /// See `pending_api::_set_pending_register_symbolic` for the body.
    pub fn set_pending_register_symbolic(
        &mut self,
        state_id: u64,
        name: &str,
        handle_id: u64,
    ) -> PyResult<()> {
        self._set_pending_register_symbolic(state_id, name, handle_id)
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    /// See `pending_api::_set_pending_register_symbolic_ast` for the body.
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_pending_register_symbolic_ast(py, state_id, reg_name, ast)
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    /// See `pending_api::_import_symbolic_to_state` for the body.
    #[pyo3(signature = (state_id, addr, ast))]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_to_state(py, state_id, addr, ast)
    }

    /// See `pending_api::_import_symbolic_memory` for the body.
    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_memory(py, state_id, addr, ast)
    }

    /// Replace a state's filesystem current working directory (angr-0xyq2
    /// Phase 3). Python `state.fs._files` keys are cwd-normalized (default
    /// `/home/user`) while the Rust `FileSystem` cwd defaults to `/`, so the
    /// init-time export must push the Python cwd BEFORE registering file
    /// content — otherwise a relative guest `open()` never matches the
    /// registry keys. Non-UTF-8 cwds are gated out by the Python caller
    /// (the Rust path model is UTF-8-lossy).
    /// See `pending_api::_set_fs_cwd` for the body.
    pub fn set_fs_cwd(&mut self, state_id: u64, cwd: &str) -> PyResult<()> {
        self._set_fs_cwd(state_id, cwd)
    }

    /// Register bounded symbolic file content for `path` on a state's
    /// filesystem (angr-0xyq2 Phase 3): one 8-bit claripy AST per byte,
    /// converted to `RustBV` via the claripy bridge (which preserves BVS
    /// identity by hash and name+width, so constraints added natively on
    /// these bytes evaluate correctly against the original Python ASTs on
    /// the found state — no sync-back injection needed). A subsequent
    /// native `open()` of the (cwd-normalized) path attaches the content
    /// and reads are served natively. Errors (not panics) on unknown
    /// `state_id`, a non-convertible AST, or a non-8-bit entry.
    /// See `pending_api::_register_file_content` for the body.
    pub fn register_file_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        path: &str,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._register_file_content(py, state_id, path, byte_asts)
    }

    /// Seed fd 0 with the harness's own symbolic stdin bytes (angr-mb09c):
    /// one 8-bit claripy AST per byte, in file order, taken from a
    /// Python-filled `state.posix.stdin.content`. Native `read(0, ...)`
    /// serves these instead of minting fresh `stdin_*` symbols, so the path
    /// condition on an exported found state references the harness's BVS and
    /// `solver.eval(bvs)` / `posix.dumps(0)` return the real solution. Reads
    /// past the seeded content fall back to fresh symbols. Errors (not
    /// panics) on unknown `state_id`, a non-convertible AST, or a non-8-bit
    /// entry. See `pending_api::_seed_stdin_content` for the body.
    pub fn seed_stdin_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._seed_stdin_content(py, state_id, byte_asts)
    }

    /// The cwd-normalized paths a state's lineage has demoted to Python
    /// ownership via native writes (angr-qluof). The Python re-add path
    /// (merge / legacy-fork push) queries these before re-registering
    /// exported content so a demoted path is not re-armed on the new
    /// state. Errors on unknown `state_id`.
    /// See `pending_api::_get_demoted_paths` for the body.
    pub fn get_demoted_paths(&self, state_id: u64) -> PyResult<Vec<String>> {
        self._get_demoted_paths(state_id)
    }

    /// Re-apply a symbolic-content demotion for `path` on `state_id`
    /// (angr-qluof): drops any registered content the export just re-armed,
    /// WITHOUT bumping the native write-demotion counter. Returns `true`
    /// when registered content or fd state was cleared. Errors on unknown
    /// `state_id`. See `pending_api::_demote_file_path` for the body.
    pub fn demote_file_path(&mut self, state_id: u64, path: &str) -> PyResult<bool> {
        self._demote_file_path(state_id, path)
    }

    /// Get memory from pending state.
    /// See `pending_api::_get_pending_memory` for the body.
    pub fn get_pending_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._get_pending_memory(state_id, addr, size)
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    /// See `pending_api::_get_pending_dirty_pages` for the body.
    pub fn get_pending_dirty_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_dirty_pages(state_id)
    }

    /// Clear dirty page tracking in pending state.
    /// See `pending_api::_clear_pending_dirty_tracking` for the body.
    pub fn clear_pending_dirty_tracking(&mut self, state_id: u64) -> PyResult<()> {
        self._clear_pending_dirty_tracking(state_id)
    }

    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    /// See `pending_api::_export_pending_constraints` for the body.
    pub fn export_pending_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._export_pending_constraints(py, state_id)
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    /// See `pending_api::_get_active_handle_ids` for the body.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        self._get_active_handle_ids()
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    /// See `pending_api::_export_pending_state` for the body.
    pub fn export_pending_state(
        &self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_pending_state(state_id)
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    /// See `pending_api::_get_pending_root_state_id` for the body.
    pub fn get_pending_root_state_id(&self, state_id: u64) -> PyResult<Option<u64>> {
        self._get_pending_root_state_id(state_id)
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    ///
    /// The walk is best-effort: it follows parent links only through states the
    /// manager still holds (pending callbacks + stashes) and stops at the first
    /// ancestor that has been consumed or dropped. The lineage root from
    /// `sm.roots()` is always appended when not already present, so the list is
    /// never shorter than the previous `[state, parent, root]` behaviour.
    /// See `pending_api::_get_pending_ancestry` for the body.
    pub fn get_pending_ancestry(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_ancestry(state_id)
    }

    /// Fork the pending state's solver context for Python callbacks.
    ///
    /// This creates a new RustSolverContext that inherits all constraints
    /// accumulated during Rust exploration. The forked context can be
    /// attached to the Python callback state, ensuring SimProcedures
    /// see the full constraint context.
    ///
    /// This is critical for proper constraint propagation: without it,
    /// callbacks would create fresh solver contexts without parent
    /// constraints, leading to incorrect symbolic evaluation.
    /// Export a callback bundle: registers, solver context, history, jumpkind
    /// in a single FFI call. Reduces ~20 individual calls to 1.
    ///
    /// Returns a Python dict with:
    /// - "registers": dict of register_name -> concrete u128 value (None if symbolic)
    /// - "solver": forked RustSolverContext
    /// - "history": list of u64 BBL addresses
    /// - "jumpkind": string
    /// - "constraint_count": u64
    /// - "stdout": bytes (accumulated stdout buffer)
    /// See `pending_api::_export_callback_bundle` for the body.
    #[pyo3(signature = (state_id, register_names, shared_solver=true))]
    pub fn export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._export_callback_bundle(py, state_id, register_names, shared_solver)
    }

    /// See `pending_api::_fork_pending_solver` for the body.
    pub fn fork_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._fork_pending_solver(state_id)
    }

    /// Borrow the pending state's solver context without forking.
    ///
    /// Returns a RustSolverContext that shares the same Z3 solver as the
    /// pending state via Rc reference counting. This is O(1) instead of
    /// the ~3ms Z3 solver clone in fork_pending_solver().
    ///
    /// Constraints added through this solver go directly to the pending state,
    /// so post-callback constraint sync via add_constraints_to_pending() should
    /// be skipped to avoid double-adding.
    /// See `pending_api::_borrow_pending_solver` for the body.
    pub fn borrow_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._borrow_pending_solver(state_id)
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    /// See `pending_api::_add_constraints_to_pending` for the body.
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._add_constraints_to_pending(py, state_id, constraints)
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        self._add_constraints_to_state(py, state_id, constraints)
    }

    /// Export raw Z3 assertion pointers from a state's solver.
    /// Lossless — captures ALL Z3 assertions, not just those tracked
    /// in assumed_constraints (which drops constraints where claripy_to_rustbv fails).
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        self._export_z3_constraint_ptrs(state_id)
    }

    /// Import raw Z3 assertion pointers to a state's solver.
    #[cfg(feature = "vex-engine-z3")]
    pub fn import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        self._import_z3_constraint_ptrs(state_id, ptrs)
    }

    /// Debug: dump solver state for a given state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        self._debug_solver_info(state_id)
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._export_state_constraints(py, state_id)
    }

    /// Unsat core for a state, as the subset of `export_state_constraints`
    /// that Z3 blames for the contradiction. Empty when the state is SAT.
    pub fn state_unsat_core(
        &self,
        py: Python<'_>,
        state_id: u64,
        extra_constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._state_unsat_core(py, state_id, extra_constraints)
    }

    /// Set the Z3 solver timeout (ms) on a specific state's solver context.
    ///
    /// Future forks of this state inherit the new timeout.  Used by
    /// RustSolverProxy to honor `state.solver.timeout = N` assignments.
    pub fn set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        self._set_state_solver_timeout(state_id, timeout_ms)
    }

    /// Get the Z3 solver timeout (ms) on a specific state's solver context.
    pub fn get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        self._get_state_solver_timeout(state_id)
    }

    /// Get the per-state mmap base pointer (mirrors Python's
    /// `state.heap.mmap_base`). The native mmap syscall handler bumps this
    /// on `addr=0` calls; Python imports it on stash export to keep the two
    /// engines from handing out overlapping mmap regions.
    pub fn get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_mmap_base(state_id)
    }

    /// Set the per-state mmap base pointer. Used by tests and by Python-side
    /// fallbacks that allocate from `state.heap.mmap_base` and need to push
    /// the advance back into Rust so subsequent native mmaps don't collide.
    pub fn set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_mmap_base(state_id, addr)
    }

    /// Get the per-state posix brk pointer (mirrors Python's
    /// `state.posix.brk`). The native brk syscall handler bumps this on
    /// concrete `brk(addr)` calls; Python imports it on stash export to keep
    /// a Python-side `set_brk` fallback from handing out heap addresses that
    /// overlap a Rust-allocated region.
    pub fn get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_posix_brk(state_id)
    }

    /// Set the per-state posix brk pointer. Used by tests and by Python-side
    /// fallbacks (symbolic `brk` argument, collision retry) that bump
    /// `state.posix.brk` and need to push the advance back into Rust.
    pub fn set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_posix_brk(state_id, addr)
    }

    /// Get the per-state heap brk pointer (mirrors Python's
    /// `state.heap.heap_location`, the malloc bump allocator). Native
    /// heap-allocating procedures (malloc/calloc/realloc/strdup/fopen) bump
    /// this via `heap_alloc`; Python imports it on stash export so a Python
    /// fallback SimProcedure doesn't hand out an address Rust already
    /// allocated. See bead angr-um39j.
    pub fn get_state_heap_brk(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_heap_brk(state_id)
    }

    /// Set the per-state heap brk pointer. Used by tests and on import to
    /// push a Python-side `state.heap.heap_location` advance back into Rust
    /// so subsequent native allocations don't collide.
    pub fn set_state_heap_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_heap_brk(state_id, addr)
    }

    // ----- Per-state Python-AST metadata (symbolic_pages /
    //       hook_symbolic_memory / addr_to_ast). Storage now lives in
    //       RustSimState; these methods are the FFI surface that replaces the
    //       old Python `_state_metadata: Dict[int, StateMetadata]` map.

    /// Replace the whole `symbolic_pages` map for a state. Mirrors the
    /// previous Python-side `_state_md(sid).symbolic_pages = pages` write.
    pub fn set_state_symbolic_pages<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        pages: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self._set_state_symbolic_pages(py, state_id, pages)
    }

    /// Snapshot the `symbolic_pages` map for a state as a Python dict.
    /// Returns an empty dict if the state is unknown or has no entries — the
    /// truthiness check at call sites already handles both cases.
    pub fn get_state_symbolic_pages<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_symbolic_pages(py, state_id)
    }

    /// Insert/replace an entry in `hook_symbolic_memory` for a state.
    pub fn set_state_hook_symbolic_memory(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_hook_symbolic_memory(state_id, addr, ast, size)
    }

    /// Snapshot the `hook_symbolic_memory` map as a `dict[int, (ast, size)]`.
    pub fn get_state_hook_symbolic_memory<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_hook_symbolic_memory(py, state_id)
    }

    /// Insert/replace an entry in `addr_to_ast` for a state.
    pub fn set_state_addr_to_ast(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_addr_to_ast(state_id, addr, ast, size)
    }

    /// Snapshot the `addr_to_ast` map as a `dict[int, (ast, size)]`.
    pub fn get_state_addr_to_ast<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_addr_to_ast(py, state_id)
    }

    /// Drop all per-state metadata (`symbolic_pages`, `hook_symbolic_memory`,
    /// `addr_to_ast`) for a state. No-op if the state is unknown — matches the
    /// `_state_metadata.pop(state_id, None)` semantics it replaces.
    pub fn clear_state_metadata(&mut self, state_id: u64) -> PyResult<()> {
        self._clear_state_metadata(state_id)
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._fork_state_solver(state_id)
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    /// See `pending_api::_get_pending_mapped_pages` for the body.
    pub fn get_pending_mapped_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_mapped_pages(state_id)
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    /// See `pending_api::_pending_memory_load_page` for the body.
    pub fn pending_memory_load_page(&self, state_id: u64, page_addr: u64) -> PyResult<Vec<u8>> {
        self._pending_memory_load_page(state_id, page_addr)
    }

    /// Symbolic counterpart of `pending_memory_load_page`: returns the
    /// (addr, claripy AST) pairs for every multi-byte symbolic object whose
    /// base address falls on the given page.
    /// See `pending_api::_pending_memory_load_symbolic_page` for the body.
    pub fn pending_memory_load_symbolic_page<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
        page_addr: u64,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        self._pending_memory_load_symbolic_page(py, state_id, page_addr)
    }

    /// See `pending_api::_pending_memory_load` for the body.
    pub fn pending_memory_load(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._pending_memory_load(state_id, addr, size)
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    /// See `pending_api::_set_skip_hook_addr` for the body.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        self._set_skip_hook_addr(addr)
    }

    /// Clear all pending skip_hook entries.
    /// See `pending_api::_clear_skip_hook_addr` for the body.
    pub fn clear_skip_hook_addr(&mut self) {
        self._clear_skip_hook_addr()
    }

    /// Clear skip entry for a specific address.
    /// See `pending_api::_clear_skip_hook_for_addr` for the body.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self._clear_skip_hook_for_addr(addr)
    }

    /// Get errors encountered during exploration.
    pub fn get_errors(&self) -> Vec<(u64, String, u64)> {
        self.errors.clone()
    }

    /// Clear error log.
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Move states between stashes.
    /// See `state_lifecycle::_move_states` for the body.
    pub fn move_states(
        &mut self,
        from_stash: &str,
        to_stash: &str,
        filter_fn: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        self._move_states(from_stash, to_stash, filter_fn)
    }

    /// P8 fix: Move a single state by ID between stashes.
    /// See `state_lifecycle::_move_state` for the body.
    pub fn move_state(
        &mut self,
        state_id: u64,
        from_stash: &str,
        to_stash: &str,
    ) -> PyResult<bool> {
        self._move_state(state_id, from_stash, to_stash)
    }

    /// P8 fix: Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        self.sm.clear(stash);
    }

    /// Drop a single state from a specific stash by ID (angr-yhe0).
    ///
    /// Removes the state from the stash's `VecDeque`, drops the lineage-root
    /// and state-index entries, and lets the `RustSimState` destructor free
    /// the Z3 solver clone. No-op (returns `false`) when the state is not in
    /// the named stash — the caller is responsible for picking the right
    /// stash (today this is only ever `"_copies"`, the holding area for
    /// `RustStateProxy.copy()` clones).
    ///
    /// Backs `RustStateProxy.__del__` — when a copy-proxy is GC'd by Python,
    /// the Rust-side state can be reclaimed without waiting for the whole
    /// manager to drop. Returns `true` when a state was actually dropped.
    pub fn drop_state_from_stash(&mut self, state_id: u64, stash: &str) -> bool {
        // Drop clears the root too (the state is gone for good); take_state_from
        // handles the stash removal + unindex (angr-ph300.27).
        if self.sm.take_state_from(state_id, stash).is_some() {
            // Dropping from STASH_ACTIVE outside `policy.select` — notify so a
            // memoizing policy doesn't leak a memo entry (angr-myzjx.25). Today
            // the caller only passes "_copies", so this is a no-op guard now,
            // but it keeps the invariant robust if the API gains callers.
            if stash == STASH_ACTIVE {
                self.policy.on_state_removed(state_id);
            }
            self.sm.remove_root(state_id);
            true
        } else {
            false
        }
    }

    /// Prepare for a new exploration stage: move a specific found state
    /// to active and clear all other stashes. Returns the state ID of the
    /// moved state. This avoids constraint transfer between managers.
    /// See `state_lifecycle::_reset_for_stage` for the body.
    pub fn reset_for_stage(&mut self, found_state_id: u64) -> PyResult<u64> {
        self._reset_for_stage(found_state_id)
    }

    /// Get statistics.
    /// See `stats_api::_stats` for the body.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._stats(py)
    }

    /// Get VEX fallback statistics: total count and per-address reasons.
    ///
    /// Returns a dict with:
    ///   "count": total number of VEX fallbacks
    ///   "addresses": dict mapping hex address string -> reason string
    ///   "dcas_unsupported_count": subset of fallbacks driven by double-CAS
    ///   "simprocedure_python_fallback_count": SimProcedures dispatched to Python
    ///   "syscall_python_fallback_count": syscalls dispatched to Python
    ///   "syscall_native_count": syscalls handled natively (no Python round-trip)
    /// See `stats_api::_get_fallback_stats` for the body.
    pub fn get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._get_fallback_stats(py)
    }

    // =========================================================================
    // Native Procedure Management
    // =========================================================================

    /// Disable all native procedures (always use Python).
    pub fn disable_native_procedures(&mut self) {
        Arc::make_mut(&mut self.native_procedures).disable_all();
    }

    /// Enable all native procedures.
    pub fn enable_native_procedures(&mut self) {
        Arc::make_mut(&mut self.native_procedures).enable_all();
    }

    /// Check if native procedures are enabled.
    pub fn native_procedures_enabled(&self) -> bool {
        self.native_procedures.is_enabled()
    }

    /// Disable a specific native procedure (fall back to Python).
    pub fn disable_native_procedure(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).disable(name);
    }

    /// Enable a specific native procedure.
    pub fn enable_native_procedure(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).enable(name);
    }

    /// Set a Python override for a procedure.
    ///
    /// When set, the native implementation is never called.
    pub fn set_python_override(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).set_python_override(name);
    }

    /// Remove a Python override.
    pub fn remove_python_override(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).remove_python_override(name);
    }

    /// Get list of available native procedures.
    pub fn list_native_procedures(&self) -> Vec<String> {
        self.native_procedures
            .procedure_names()
            .iter()
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    /// See `stats_api::_native_procedure_stats` for the body.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._native_procedure_stats(py)
    }

    /// Register a Python callable as a native procedure.
    ///
    /// The callable receives `list[int]` of concrete arg values and must
    /// return `Optional[int]` for the return value (None = no value).
    /// Symbolic arguments cause an automatic fallback to the regular
    /// Python SimProcedure path; the registered callable is only invoked
    /// when all args are concrete.
    #[pyo3(signature = (name, num_args, no_return, callable))]
    pub fn register_python_procedure(
        &mut self,
        name: String,
        num_args: usize,
        no_return: bool,
        callable: Py<PyAny>,
    ) {
        let proc = std::sync::Arc::new(crate::procedures::python_proc::PythonNativeProcedure::new(
            name, num_args, no_return, callable,
        ));
        Arc::make_mut(&mut self.native_procedures).register(proc);
    }

    /// Register native `ReturnUnconstrained` stubs: `(display_name, ret_bits)`.
    ///
    /// A `SimLibrary` hands out `ReturnUnconstrained` for every symbol it has
    /// no model for, so those hooks are keyed on the *binary's* symbol name
    /// rather than a libc one and cannot live in the static registry. Python
    /// (`RustExplorationManager._register_simprocedures`) filters
    /// `project._sim_procedures` down to the plain stubs — no `return_val=`
    /// kwarg, prototype return size known — and passes them here.
    ///
    /// Names that already have a real native procedure, or that carry a zero
    /// return width, are skipped: a genuine implementation always outranks a
    /// "return a fresh symbol" stub.
    pub fn register_unconstrained_stubs(&mut self, stubs: Vec<(String, u32)>) {
        self.steady_config_guard();
        let registry = Arc::make_mut(&mut self.native_procedures);
        for (name, ret_bits) in stubs {
            if ret_bits == 0 || registry.has_native(&name) {
                continue;
            }
            registry.register(Arc::new(
                crate::procedures::stub::NativeReturnUnconstrained::new(&name, ret_bits),
            ));
        }
    }

    // =========================================================================
    // Native Uniqueness Filter
    // =========================================================================

    /// Enable native uniqueness filter with given register names.
    ///
    /// After each step in run(), states with duplicate register tuples
    /// are moved to 'not_unique' stash. This replaces the Python
    /// CheckUniqueness technique with zero FFI overhead.
    pub fn register_uniqueness_filter(&mut self, register_names: Vec<String>) {
        self.steady_config_guard();
        self.constraint_tracker.uniqueness_registers = register_names;
        self.constraint_tracker.uniqueness_set.clear();
        // Ensure not_unique stash exists
        self.sm
            .stashes_mut()
            .entry("not_unique".to_string())
            .or_default();
    }

    /// Disable the native uniqueness filter.
    pub fn disable_uniqueness_filter(&mut self) {
        self.steady_config_guard();
        self.constraint_tracker.uniqueness_registers.clear();
        self.constraint_tracker.uniqueness_set.clear();
    }

    /// Check if native uniqueness filter is enabled.
    pub fn uniqueness_filter_enabled(&self) -> bool {
        !self.constraint_tracker.uniqueness_registers.is_empty()
    }

    /// Get the number of unique register tuples seen.
    pub fn uniqueness_set_size(&self) -> usize {
        self.constraint_tracker.uniqueness_set.len()
    }

    // =========================================================================
    // Native Exploration Techniques
    // =========================================================================

    /// Register a native LengthLimiter technique.
    ///
    /// States whose history exceeds `max_length` blocks are moved to "cut"
    /// (or "_DROP" if `drop` is true). Runs entirely in Rust with zero FFI overhead.
    pub fn register_length_limiter(&mut self, max_length: usize, drop: bool) {
        self.native_techniques
            .push(NativeTechnique::LengthLimiter { max_length, drop });
        if !drop {
            self.sm.stashes_mut().entry("cut".to_string()).or_default();
        }
    }

    /// Register a native Timeout technique.
    ///
    /// Exploration stops after `timeout_secs` seconds. All active states are
    /// moved to "timeout" stash. Timer starts on first call to apply_native_techniques().
    pub fn register_timeout(&mut self, timeout_secs: f64) {
        self.native_techniques.push(NativeTechnique::Timeout {
            timeout_secs,
            start_time: None,
        });
        self.sm
            .stashes_mut()
            .entry("timeout".to_string())
            .or_default();
    }

    /// Register a native LoopBound technique.
    ///
    /// States where any single address appears more than `bound` times in
    /// their history are moved to `discard_stash`. This is a simplified
    /// version of LoopSeer that doesn't require CFG analysis.
    #[pyo3(signature = (bound, discard_stash="spinning"))]
    pub fn register_loop_bound(&mut self, bound: usize, discard_stash: &str) {
        self.native_techniques.push(NativeTechnique::LoopBound {
            bound,
            discard_stash: discard_stash.to_string(),
        });
        self.sm
            .stashes_mut()
            .entry(discard_stash.to_string())
            .or_default();
    }

    /// Register a native MergePoint technique (ManualMergepoint parity,
    /// angr-op0dn.11.5).
    ///
    /// States reaching `address` are parked in a per-address wait stash; once
    /// the active stash drains (or `wait_counter` post-step rounds elapse
    /// without a fresh arrival) the waiters are grouped by callstack and each
    /// ≥2 group is merged in-Rust via `_merge_states`. A lone waiter is
    /// released back to active unmerged (count preserved, no stall).
    #[pyo3(signature = (address, wait_counter=10))]
    pub fn register_merge_point(&mut self, address: u64, wait_counter: usize) {
        let wait_stash = format!("merge_waiting_{address:#x}");
        self.native_techniques.push(NativeTechnique::MergePoint {
            address,
            wait_counter_limit: wait_counter,
            counter: 0,
            wait_stash: wait_stash.clone(),
        });
        self.sm.stashes_mut().entry(wait_stash).or_default();
    }

    /// Get the number of registered native techniques.
    pub fn native_technique_count(&self) -> usize {
        self.native_techniques.len()
    }

    /// Clear all native techniques.
    pub fn clear_native_techniques(&mut self) {
        self.native_techniques.clear();
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    /// Deferred writes (pending writes and Multi cells) are flushed first, so
    /// this is now an alias for `export_state_flushed` (angr-9ke6b.101).
    pub fn export_state(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state(state_id)
    }

    /// Step a single state out-of-band and return its successors bucketed by
    /// category, WITHOUT stashing them (E1.a).
    ///
    /// `extra_stop_points` are stop addresses honored for this call only, on
    /// top of the manager-level stop addresses. Every returned state is parked
    /// in the `_step_out` stash; the caller places it with `move_state()`.
    /// See `stepping::_step_state` for the body.
    #[pyo3(signature = (state_id, extra_stop_points=None))]
    pub fn step_state(
        &mut self,
        state_id: u64,
        extra_stop_points: Option<Vec<u64>>,
    ) -> PyResult<HashMap<String, Vec<crate::state::ExplorationStateSnapshot>>> {
        self._step_state(state_id, extra_stop_points)
    }

    /// Export a state by ID, flushing pending writes first.
    pub fn export_state_flushed(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state_flushed(state_id)
    }

    /// Export all found states as snapshots.
    ///
    /// Flushes each state's deferred writes first (angr-9ke6b.101) — this is
    /// the primary `explore(find=...)` result API, so an unflushed export here
    /// silently drops Multi-cell bytes.
    pub fn export_found_states(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_found_states()
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        self._state_symbolic_info(state_id, addr)
    }

    /// Get Z3 AST pointers for all symbolic objects in a state's memory.
    ///
    /// Returns Vec<(addr, z3_ast_ptr_as_usize, width_bits)> for each symbolic
    /// object. The Z3 ASTs are built in the shared Z3 context, so Python can
    /// directly wrap them as z3.BitVecRef and convert to claripy ASTs.
    ///
    /// This is used to export Rust-computed symbolic expressions (e.g., flag
    /// computations in asisctf) to Python state memory during state export.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_state_symbolic_z3_asts(&self, state_id: u64) -> PyResult<Vec<(u64, usize, u32)>> {
        self._get_state_symbolic_z3_asts(state_id)
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        self._state_satisfiable(state_id)
    }

    /// Whether strict memory permission enforcement is enabled on a state.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn state_enforce_permissions(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_permissions(state_id)
    }

    /// Whether non-executable page enforcement is enabled on a state.
    /// Mirrors angr's ENABLE_NX option.
    pub fn state_enforce_nx(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_nx(state_id)
    }

    /// Whether NO_IP_CONCRETIZATION is active on a state.
    /// When set, symbolic jump targets short-circuit to the unconstrained
    /// stash without enumeration.
    pub fn state_no_ip_concretization(&self, state_id: u64) -> PyResult<bool> {
        self._state_no_ip_concretization(state_id)
    }

    /// Whether NO_SYMBOLIC_JUMP_RESOLUTION is active on a state.
    /// Same Rust effect as `state_no_ip_concretization` — symbolic jump
    /// targets route to the unconstrained stash without enumeration.
    pub fn state_no_symbolic_jump_resolution(&self, state_id: u64) -> PyResult<bool> {
        self._state_no_symbolic_jump_resolution(state_id)
    }

    /// Whether KEEP_IP_SYMBOLIC is active on a state.
    /// When set, the IP register on each post-concretization successor stays
    /// holding the original symbolic next-pc expression (no `target == addr`
    /// narrowing constraint is added). The next block lift still drives from
    /// the concretized `state.pc` value.
    pub fn state_keep_ip_symbolic(&self, state_id: u64) -> PyResult<bool> {
        self._state_keep_ip_symbolic(state_id)
    }

    /// Whether the named symex-relevant SimOption (e.g. `"SHORT_READS"`) is
    /// active on a state (angr-kzjv6). Mirrors the option subset that
    /// `_add_rust_state` threads onto the Rust state for native SimProcedures.
    pub fn state_has_option(&self, state_id: u64, name: &str) -> PyResult<bool> {
        self._state_has_option(state_id, name)
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self._get_state_register(state_id, name)
    }

    /// Get multiple register values from a state in one FFI call.
    /// Returns a list of `Option<u128>` in the same order as the input names.
    pub fn get_state_registers_batch(
        &self,
        state_id: u64,
        names: Vec<String>,
    ) -> PyResult<Vec<Option<u128>>> {
        self._get_state_registers_batch(state_id, names)
    }

    /// Get the claripy AST for a register on a state (angr-4pm1).
    ///
    /// Mirrors `get_pending_register_ast` for an arbitrary `state_id`.
    /// Returns the claripy AST built from Rust's stored `RustBV`, preserving
    /// identity for symbolic values so constraints added by the proxy land on
    /// the same Z3 symbol Rust is tracking. Returns `None` if the register
    /// name is unknown or the state holds no value for it.
    /// See `state_api::_get_state_register_ast` for the body.
    pub fn get_state_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self._get_state_register_ast(py, state_id, name)
    }

    /// Set a state's register to a symbolic value from a claripy AST
    /// (angr-4pm1). Mirrors `set_pending_register_symbolic_ast` for an
    /// arbitrary `state_id`, routing through `claripy_to_rustbv` so the
    /// symbol is registered in the shared cache and the inverse
    /// `get_state_register_ast` round-trip preserves identity.
    /// See `state_api::_set_state_register_symbolic_ast` for the body.
    pub fn set_state_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_register_symbolic_ast(py, state_id, reg_name, ast)
    }

    /// Get memory from a state.
    pub fn get_state_memory(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        self._get_state_memory(state_id, addr, size)
    }

    /// Get memory from a state as a claripy AST (angr-8dop.1). Returns the
    /// symbolic AST verbatim — never concretizes via the solver. Used by
    /// `RustMemoryProxy.load` when the gate is on so symbolic libc
    /// SimProcedures (strlen/strchr/memchr/...) see real symbolic bytes
    /// instead of an arbitrary solver witness.
    /// See `state_api::_get_state_memory_ast` for the body.
    pub fn get_state_memory_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Py<PyAny>>> {
        self._get_state_memory_ast(py, state_id, addr, size)
    }

    /// Set memory on a state from concrete bytes (angr-j28e write-through).
    /// See `state_api::_set_state_memory_concrete` for the body.
    pub fn set_state_memory_concrete(
        &mut self,
        state_id: u64,
        addr: u64,
        data: &[u8],
    ) -> PyResult<()> {
        self._set_state_memory_concrete(state_id, addr, data)
    }

    /// Like `set_state_memory_concrete`, but widens the state's lazy region to
    /// cover the target page so a store to an otherwise-unmapped address
    /// auto-maps instead of erroring `Unmapped` (angr-ijwp0). Used by
    /// `RustMemoryProxy.store` to mirror angr's map-on-write memory when
    /// STRICT_PAGE_ACCESS is off.
    /// See `state_api::_set_state_memory_concrete_automap` for the body.
    pub fn set_state_memory_concrete_automap(
        &mut self,
        state_id: u64,
        addr: u64,
        data: &[u8],
    ) -> PyResult<()> {
        self._set_state_memory_concrete_automap(state_id, addr, data)
    }

    /// Set memory on a state from a claripy AST (angr-j28e write-through).
    /// Used when the value is symbolic (e.g., a BVS or expression). The
    /// address is concrete; symbolic addresses are not supported on the
    /// proxy write path — callers fall back to the Python engine.
    /// See `state_api::_set_state_memory_ast` for the body.
    pub fn set_state_memory_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_memory_ast(py, state_id, addr, ast)
    }

    /// Like `set_state_memory_ast`, but widens the state's lazy region to cover
    /// the target page so a store to an address outside any existing lazy
    /// region auto-maps instead of erroring `Unmapped` (angr-5rjbq). Used by
    /// the callback-memory-proxy symbolic-address store fallback.
    pub fn set_state_memory_ast_automap(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_memory_ast_automap(py, state_id, addr, ast)
    }

    /// Phase 1.4 (angr-5zw8): route a symbolic-address store through the
    /// Multi-cell lazy path on the given state. Used by
    /// `_cb_memory_store_symbolic_full` when the address AST carries a
    /// `MultiwriteAnnotation` — the SimProcedures in `libc/strchr.py`,
    /// `libc/gets.py`, `libc/fgets.py` tag returned addresses with this
    /// annotation so Range concretization picks up >1 candidate.
    ///
    /// Returns `true` on success. Returns `false` if conversion or store
    /// fails (caller should fall back to the existing Python state path
    /// to keep progress).
    pub fn state_memory_store_symbolic_multi<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        addr_ast: &Bound<'py, PyAny>,
        data_ast: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        self._state_memory_store_symbolic_multi(py, state_id, addr_ast, data_ast)
    }

    /// Get the stdout buffer for a state by ID.
    ///
    /// Returns the accumulated output from native puts/printf calls.
    pub fn get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self._get_state_stdout(state_id)
    }

    /// Get the output buffer for a specific file descriptor.
    ///
    /// Returns the accumulated output from native write/puts/printf calls.
    pub fn get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_output(state_id, fd)
    }

    /// Check whether a file descriptor has any output (no allocation).
    pub fn has_state_fd_output(&self, state_id: u64, fd: u32) -> bool {
        self._has_state_fd_output(state_id, fd)
    }

    /// Append bytes to a file descriptor's output buffer.
    ///
    /// The write-back half of the SimProcedure-callback posix channel
    /// (angr-op0dn.14.1.4): a bounced fwrite/fputc/fprintf writes the Python
    /// callback state's posix stream, and this pushes the new suffix into the
    /// Rust state so `posix.dumps(fd)` on a later export sees it.
    ///
    /// Returns `false` when the write was refused because the fd carried
    /// bounded symbolic content (see `RustSimState::write_fd`); the caller
    /// should treat that as a lossy fallback, not an error.
    pub fn append_state_fd_output(
        &mut self,
        state_id: u64,
        fd: u32,
        data: Vec<u8>,
    ) -> PyResult<bool> {
        self._append_state_fd_output(state_id, fd, data)
    }

    /// Check if a state has recorded stdin symbols from native fgets/fgetc/getchar.
    pub fn has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self._has_state_stdin_symbols(state_id)
    }

    /// Get the stdin symbols for a state by ID.
    ///
    /// Returns list of (name, bit_width) tuples for symbolic variables
    /// created by native fgets/fgetc/getchar. Used to reconstruct stdin
    /// data in Python's posix plugin for posix.dumps(0).
    pub fn get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self._get_state_stdin_symbols(state_id)
    }

    /// Get the call stack for a state by ID.
    ///
    /// Returns list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_state_call_stack(&self, state_id: u64) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self._get_state_call_stack(state_id)
    }

    /// Get the call stack depth for a state by ID.
    pub fn get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self._get_state_call_stack_depth(state_id)
    }

    /// Get the detailed execution history for a state by ID.
    ///
    /// Returns list of (addr, jumpkind, jump_target) tuples.
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_state_detailed_history(&self, state_id: u64) -> PyResult<Vec<(u64, u8, u64)>> {
        self._get_state_detailed_history(state_id)
    }

    /// Get heap metadata for a state by ID.
    ///
    /// Returns dict with:
    /// - allocated: list of (addr, size) tuples for active allocations
    /// - freed: list of distinct freed addresses (a set — see `HeapMetadata::freed`)
    /// - alloc_count: number of active allocations
    /// - free_count: number of distinct freed addresses
    pub fn get_state_heap_metadata(&self, state_id: u64) -> PyResult<HeapMetadataReturn> {
        self._get_state_heap_metadata(state_id)
    }

    /// Get the list of open file descriptors for a state.
    ///
    /// Returns list of (fd, name, position, flags, content_len, is_open) tuples.
    pub fn get_state_open_fds(&self, state_id: u64) -> PyResult<Vec<OpenFdInfo>> {
        self._get_state_open_fds(state_id)
    }

    /// True when the state has an open fd above stderr.
    ///
    /// Fast-path gate for the inbound callback fd sync (angr-op0dn.14.1.6) so
    /// the common bounce (nothing opened natively) never pays for the
    /// `get_state_open_fds` tuple list.
    pub fn has_state_extra_fds(&self, state_id: u64) -> bool {
        self._has_state_extra_fds(state_id)
    }

    /// Get the content of a file descriptor for a state.
    pub fn get_state_fd_content(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_content(state_id, fd)
    }

    /// Adopt an fd that a bounced Python SimProcedure opened, at the fd number
    /// Python chose. `flags` is a POSIX open(2) bitfield; `content` the
    /// concrete bytes of the backing SimFile; `position` its seek offset.
    ///
    /// Returns False when the fd is already known natively (nothing changes),
    /// which makes the caller's fd-table diff idempotent across the repeated
    /// callbacks that share one cached state.
    pub fn register_state_fd(
        &mut self,
        state_id: u64,
        fd: u32,
        name: String,
        flags: u32,
        content: Vec<u8>,
        position: u64,
    ) -> PyResult<bool> {
        self._register_state_fd(state_id, fd, name, flags, content, position)
    }

    /// Evaluate a stdin symbol by name and width using the state's solver.
    ///
    /// `width` is the symbol's recorded bit-width (8 for byte reads, 32/64 for
    /// scanf numeric conversions). Passing the wrong width mints a distinct,
    /// unconstrained Z3 const and yields a garbage model value (angr-ph300.18).
    /// Returns the concrete value as `Option<u64>`, or None if the symbol
    /// cannot be found or evaluated.
    pub fn eval_stdin_symbol(&self, state_id: u64, name: &str, width: u32) -> Option<u64> {
        self._eval_stdin_symbol(state_id, name, width)
    }

    // =========================================================================
    // Run loop (body in run_loop.rs + its run_loop_{single,wave,steady} siblings)
    // =========================================================================

    /// Run the exploration loop.
    ///
    /// Returns an ExplorationEvent when:
    /// - Found enough solutions
    /// - Active stash is empty
    /// - Need Python callback (SimProcedure, syscall)
    /// - Max steps reached
    #[pyo3(signature = (n=None))]
    pub fn run(&mut self, py: Python<'_>, n: Option<u32>) -> PyResult<ExplorationEvent> {
        // Python re-entered the loop without resuming a parked callback: that
        // gap is driver overhead, not a bounce excursion. Drop the clock.
        crate::gil_profile::park_cancel();
        let event = self.run_loop(py, n)?;
        // Handing control back to Python with a callback outstanding starts a
        // park-and-bounce excursion; the matching `resume_after_*` banks it as
        // `GilClass::Bounce` (bd angr-gorvf.8).
        crate::gil_profile::park_start(
            self.profiling.profiling_enabled && !self.pending_callbacks.is_empty(),
        );
        Ok(event)
    }

    // =========================================================================
    // Resume methods (from resume.rs)
    // =========================================================================

    /// Resume after a SimProcedure callback. See `resume::_resume_after_simprocedure` for the body.
    #[pyo3(signature = (state_id, new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_simprocedure(
            py,
            state_id,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Resume after a syscall callback.
    #[pyo3(signature = (state_id, new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_syscall(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self._resume_after_simprocedure(
            py,
            state_id,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Fast-path: deadend the pending callback state without full apply_changes.
    /// Used for SimProcedure continuations known to just call exit().
    /// See `resume::_deadend_pending_callback` for the body.
    pub fn deadend_pending_callback(&mut self, state_id: u64) -> PyResult<()> {
        self._deadend_pending_callback(state_id)
    }

    /// Resume after an error occurred during callback execution (P17).
    /// See `resume::_resume_after_error` for the body.
    pub fn resume_after_error(&mut self, state_id: u64, error_msg: &str) -> PyResult<()> {
        self._resume_after_error(state_id, error_msg)
    }

    /// Resume after Python handles a symbolic branch.
    /// See `resume::_resume_after_symbolic_branch` for the body.
    #[pyo3(signature = (state_id, true_pc, false_pc, true_constraints=None, false_constraints=None))]
    pub fn resume_after_symbolic_branch(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_symbolic_branch(
            py,
            state_id,
            true_pc,
            false_pc,
            true_constraints,
            false_constraints,
        )
    }

    /// Resume after Python evaluates a find predicate (P2).
    /// See `resume::_resume_find_predicate` for the body.
    pub fn resume_find_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        self._resume_find_predicate(state_id, matched)
    }

    /// Resume after Python evaluates an avoid predicate (P7).
    /// See `resume::_resume_avoid_predicate` for the body.
    pub fn resume_avoid_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        self._resume_avoid_predicate(state_id, matched)
    }

    // =========================================================================
    // Solver Profiling Stats
    // =========================================================================

    /// Get global Z3 solver profiling stats as a dict.
    #[staticmethod]
    pub fn get_solver_stats() -> std::collections::HashMap<String, u64> {
        crate::symbolic::get_solver_stats()
    }

    /// Reset global Z3 solver profiling stats to zero.
    #[staticmethod]
    pub fn reset_solver_stats() {
        crate::symbolic::reset_solver_stats()
    }

    /// Walk every state's assumed-constraint RustBV graph and report
    /// pointer-identity vs structural-identity sharing (angr-zdho).
    ///
    /// Returns a dict with:
    ///   - `total_visits`     — DAG descents, counting Arc re-visits.
    ///   - `unique_pointers`  — distinct RustBV Arc allocations seen. The
    ///     current per-call `to_z3_ast_cached` cache collapses repeat
    ///     visits of the same Arc pointer down to this.
    ///   - `unique_shapes`    — distinct structural shapes. A
    ///     construction-time hash-cons (angr-behq) would dedupe to this.
    ///   - `structural_duplicates` — `unique_pointers - unique_shapes`.
    ///   - `states_analyzed`  — how many states contributed constraints.
    ///   - `constraints_analyzed` — total `(RustBV, bool)` pairs folded in.
    ///
    /// Walks ALL stashes (so it's deterministic across exploration
    /// outcomes — no `find`/`avoid` bias).
    pub fn analyze_constraint_sharing(&self) -> std::collections::HashMap<String, u64> {
        let mut walk = crate::symbolic::ConstraintSharingWalk::new();
        let mut states_analyzed: u64 = 0;
        let mut constraints_analyzed: u64 = 0;
        for stash in self.sm.stashes().values() {
            for state in stash.iter() {
                let ctx = state.solver().borrow();
                let n_constraints = ctx.assumed_constraint_count();
                if n_constraints == 0 {
                    continue;
                }
                ctx.fold_sharing_walk(&mut walk);
                states_analyzed += 1;
                constraints_analyzed = constraints_analyzed.saturating_add(n_constraints as u64);
            }
        }
        let stats = walk.into_stats();
        let mut out = std::collections::HashMap::new();
        out.insert("total_visits".into(), stats.total_visits);
        out.insert("unique_pointers".into(), stats.unique_pointers);
        out.insert("unique_shapes".into(), stats.unique_shapes);
        out.insert(
            "structural_duplicates".into(),
            stats.unique_pointers.saturating_sub(stats.unique_shapes),
        );
        out.insert("states_analyzed".into(), states_analyzed);
        out.insert("constraints_analyzed".into(), constraints_analyzed);
        out
    }

    /// Serialize the full stash manager (all stashes, lineage, counters) to
    /// a versioned byte envelope. Wraps [`StashManager::dump_snapshot`]
    /// (see `stash.rs::STASH_SNAPSHOT_VERSION`). Bucket-D `Py<PyAny>`
    /// overlays (symbolic_pages / hook_symbolic_memory / addr_to_ast) are
    /// NOT captured — the Python wrapper handles those via
    /// `claripy.dumps`/`loads`.
    ///
    /// angr-op0dn.13.6: takes `&mut self` and finalizes a live steady session
    /// first. Under steady mode (angr-nkoct) the frontier lives inside the
    /// worker Z3 contexts and is in NO stash, so dumping `self.sm` mid-session
    /// would silently truncate the search — the snapshot would look valid and
    /// resume with a smaller frontier. Finalize-then-capture is the contract.
    pub fn dump_snapshot_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.steady_config_guard();
        self.flush_parked_bounces_to_active();
        let bytes = self.sm.dump_snapshot();
        PyBytes::new(py, &bytes)
    }

    /// Restore the stash manager from a [`Self::dump_snapshot_bytes`]
    /// envelope. Replaces `self.sm` wholesale; manager-level configuration
    /// (find/avoid addrs, hooks, simprocedures, solver/memory config) is
    /// preserved. An empty envelope or stale version byte raises
    /// `ValueError`.
    ///
    /// angr-ph300.21: mirrors the dump-side finalize contract before the
    /// swap. A manager parked on `need_callback`, or with a live steady
    /// session (`RUST_PARALLEL_STEADY=1`), carries worker-resident states and
    /// pending callback/bounce entries that belong to the PRE-restore world.
    /// Without a teardown, the next `run()` would finalize the OLD steady
    /// session — draining those states into the freshly restored stashes (two
    /// explorations merged) — and a surviving pending callback could resume a
    /// state from the discarded world. We therefore `steady_config_guard()`
    /// first (drains any live session into the soon-to-be-discarded `self.sm`)
    /// and clear all pending single-step / callback / parked-bounce state, so
    /// the restored frontier starts from a clean manager.
    pub fn load_snapshot_bytes(&mut self, bytes: &[u8]) -> PyResult<()> {
        let restored = StashManager::load_snapshot(bytes)
            .map_err(|e| PyValueError::new_err(format!("snapshot load failed: {e}")))?;
        self.steady_config_guard();
        self.pending_callbacks.clear();
        self.pending_parallel_bounces.clear();
        self.current_stepping_state_id = None;
        self.sm = restored;
        Ok(())
    }
}

#[cfg(test)]
#[path = "manager_methods_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
