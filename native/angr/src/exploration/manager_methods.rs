//! `#[pymethods]` for [`RustExplorationManager`]: construction, configuration
//! and callback wiring.
//!
//! Split out of `mod.rs` (angr-nbim4.1) to keep the parent module thin: the
//! `#[pyclass]` struct stays in the parent and `use super::*` pulls in the
//! parent's imports, `pub(crate)` fields, and type aliases so the method
//! bodies compile unchanged.
//!
//! This module used to hold the *entire* ~228-method surface in one block,
//! because PyO3 without the `multiple-pymethods` feature allows exactly one
//! `#[pymethods]` block per pyclass. That feature is now enabled
//! (`native/angr/Cargo.toml`, angr-9ke6b.50), so the surface is split along
//! its former section banners into sibling `manager_methods_*.rs` modules,
//! each with its own `#[pymethods] impl RustExplorationManager` block:
//! `_hooks`, `_state`, `_constraints`, `_procedures`, `_techniques`,
//! `_export`, `_run`, `_stats`. Adding a method means picking the module that
//! matches its section — no need to grow any single file.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
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
            parallel_shadow_migration_failures: 0,
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
    // Configuration PyAPI (thin getters/setters kept inline here — larger
    // bodies live in sibling modules, and the rest of the surface lives in the
    // `manager_methods_*` blocks listed in this module's doc)
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
        Arc::make_mut(&mut self.memory_config.vex_opt_level_overrides).insert(addr, level);
        // Remove this address from block cache since opt_level changed
        self.environment.block_cache.pop(&addr);
    }

    /// Remove a per-address VEX optimization level override.
    pub fn remove_vex_opt_level_override(&mut self, addr: u64) {
        Arc::make_mut(&mut self.memory_config.vex_opt_level_overrides).remove(&addr);
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
        Arc::make_mut(&mut self.memory_config.vex_opt_level_overrides).clear();
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
}

#[cfg(test)]
#[path = "manager_methods_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
