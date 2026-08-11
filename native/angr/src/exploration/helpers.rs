//! Bookkeeping helpers hung off `RustExplorationManager` (extension-impl
//! pattern, mirroring `stepping.rs` / `run_loop.rs` / `state_api.rs`). Nothing
//! here is pyclass-exposed: these are the internal seams the stepping and
//! run-loop modules call into, grouped by concern:
//!
//! - **Scheduler accounting folds** — [`RustExplorationManager::fold_scheduler_dispatch_stats`]
//!   absorbs a real parallel run's per-worker dispatch counts / frontier-width
//!   histogram / dead-path terminals; [`RustExplorationManager::record_reconvergence_sample`]
//!   and [`RustExplorationManager::record_migration_sample`] maintain the
//!   *modelled* (single-threaded) equivalents so the same counters stay
//!   populated when no scheduler ran.
//! - **State lookup and borrow adapters** — `with_state{,_mut}` /
//!   `with_pending{,_mut}` close over a `StateId` and hand out a checked
//!   borrow; [`RustExplorationManager::find_state`],
//!   [`RustExplorationManager::index_state`] and
//!   [`RustExplorationManager::rebuild_state_index`] back them with the
//!   id → stash index.
//! - **Stash routing and caps** — [`RustExplorationManager::route_successor`]
//!   is the single decision point for where a successor lands;
//!   `push_to_active_or_drop` / `push_found_capped` / `push_or_drop_terminal`
//!   enforce the per-stash limits it depends on.
//! - **Uniqueness filtering** — [`RustExplorationManager::compute_register_tuple_hash`]
//!   plus [`RustExplorationManager::apply_uniqueness_filter`], the active-stash
//!   dedup pass.
//! - **Callback argument extraction** — `extract_procedure_args` /
//!   `extract_syscall_args` / `get_return_addr` marshal a state's ABI registers
//!   into the values a Python callback receives.
//! - **Callback solver setup** — the free fn [`prepare_shared_callback_solver`],
//!   the shared pre-callback-snapshot + shared-solver-context preamble.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]` and has no
//! production `#[allow]` opt-out — the only one is on the `mod tests` include
//! at the bottom of the file, which re-permits them for test code because the
//! module-level `deny` overrides `lib.rs`'s crate-wide `cfg_attr(test, allow(..))`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

impl RustExplorationManager {
    /// Fold a scheduler run's real (not modelled) accounting into the manager:
    /// per-worker dispatch counts, the frontier-width histogram / peak
    /// (angr-op0dn.13.9), and the dead-path terminal counts the workers only
    /// summarized (angr-op0dn.13.15). Shared by BOTH parallel loops — the wave loop
    /// and the steady-state coordinator — so frontier residency is no longer a
    /// blind spot for the width audit and the S7 find-all gate.
    ///
    /// The width buckets come from the same sampler as the serial model's
    /// `record_migration_sample`, so `parallel_width_hist` stays a single
    /// comparable series regardless of which loop produced it.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn fold_scheduler_dispatch_stats(&mut self, stats: &scheduler::SchedulerStats) {
        for (i, count) in stats.width_hist.iter().enumerate() {
            self.parallel_width_hist[i] += *count as u64;
        }
        self.parallel_max_active_width = self.parallel_max_active_width.max(stats.max_width as u64);

        // Trim to the REAL pool size, not `parallel_num_workers` (the migration
        // model's modelled worker count, which is independent and defaults to 4):
        // the counter array is fixed-size (`MAX_TRACKED_WORKERS`), and slots past
        // the real pool can never be non-zero. Sizing by the modelled count would
        // pad the vector with phantom zero-dispatch workers and make the gate's
        // max/min balance ratio a spurious `inf`. Both parallel loops clamp the
        // pool to `max(2)`, matching the `max(2)` here.
        let workers = self
            .parallel_real_workers
            .clamp(2, scheduler::MAX_TRACKED_WORKERS);
        if self.parallel_worker_dispatch.len() < workers {
            self.parallel_worker_dispatch.resize(workers, 0);
        }
        for (i, count) in stats.worker_dispatches.iter().take(workers).enumerate() {
            self.parallel_worker_dispatch[i] += *count as u64;
        }
        // angr-9ke6b.67: carry the degraded-data marker through. Non-zero means
        // `RUST_PARALLEL_WORKERS > MAX_TRACKED_WORKERS`, so the tail workers all
        // merged into the last slot above and the vector's max/min balance ratio
        // is not trustworthy.
        self.parallel_worker_dispatch_folded += stats.folded_worker_dispatches as u64;

        // Dead-path terminal accounting (angr-op0dn.13.15). Workers drop the
        // summarized states in their own context — their full symbolic content is
        // not recoverable through the parallel path — but the COUNTS must still
        // land, or `stats()["deadended_count"]` reads 0 on a parallel run where
        // the serial loop reports N terminal paths. Materialized terminals
        // (found / unconstrained / bounce) are counted by the coordinator's
        // `push_or_drop_terminal` in `route_materialized_terminal` and never
        // appear as summaries, so there is no double count.
        self.sm.deadended_count += stats.summarized_deadended as u64;
        self.sm.errored_count += stats.summarized_errored as u64;
        self.sm.pruned_count += stats.summarized_pruned as u64;
        // `avoided` is summarized in-worker too (angr-pwu71): `parallel_process_state`
        // matches a successor's pc against `avoid_addrs` before the state ever
        // reaches the coordinator's avoid-routing, so without this fold a
        // fork-successor landing on an avoid address is invisible to
        // `stats()["avoided_count"]` on a parallel run.
        self.sm.avoided_count += stats.summarized_avoided as u64;
    }

    /// DS-instr (angr-11djq.16): sample the active stash once per step and
    /// accumulate state-reconvergence counters. A "collision" is two or more
    /// active states sharing a `(pc, callstack-return-addr-chain)` key at the
    /// same step — i.e. paths that have reconverged on the same program point.
    /// Counters feed directed-search pruning (.14.2) and the merging-go/no-go
    /// signal (.10). Cheap: `<2` active states is the common case and short-
    /// circuits to two counter increments; only `>=2` builds the per-key map.
    /// No behaviour change — counters only.
    pub(crate) fn record_reconvergence_sample(&mut self) {
        let active = match self.sm.get(STASH_ACTIVE) {
            Some(a) if !a.is_empty() => a,
            _ => return,
        };
        let observed = active.len() as u64;
        let (colliding, max_group) = if active.len() < 2 {
            (0u64, 1u64)
        } else {
            use std::hash::{Hash, Hasher};
            let mut counts: HashMap<u64, u32> = HashMap::new();
            for state in active.iter() {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                state.pc().hash(&mut h);
                for entry in state.call_stack() {
                    entry.return_addr.hash(&mut h);
                }
                *counts.entry(h.finish()).or_insert(0) += 1;
            }
            let mut colliding = 0u64;
            let mut max_group = 1u64;
            for &c in counts.values() {
                if c >= 2 {
                    colliding += c as u64;
                }
                max_group = max_group.max(c as u64);
            }
            (colliding, max_group)
        };
        self.reconvergence_collision_states += colliding;
        self.reconvergence_active_observed += observed;
        self.reconvergence_samples += 1;
        if max_group > self.reconvergence_max_group {
            self.reconvergence_max_group = max_group;
        }
    }

    /// angr-panhl.1 (Phase 0 kill-gate): model a work-stealing scheduler over
    /// the active frontier *without any real threading*, and count how often a
    /// state WOULD be migrated across worker boundaries (a "steal"). Sampled
    /// once per step at dispatch time, where `stepped` is the just-popped
    /// state id (the active stash no longer contains it). Each dispatch is one
    /// "task".
    ///
    /// Model: `parallel_num_workers` workers, each owning a deque. Homes are
    /// sticky across steps (locality); a new state is placed on the
    /// least-loaded worker. A steal is counted when the dispatched state's home
    /// worker still has a backlog (≥2 queued tasks, incl. this one) while some
    /// other worker is idle (0 queued) — that idle worker would steal this task
    /// across a boundary. Two `O(width)` passes; short-circuits the common
    /// `<2`-state frontier. Counters only; no behaviour change. See bd memory
    /// `parallel-migration-model`.
    pub(crate) fn record_migration_sample(&mut self, stepped: u64) {
        self.parallel_tasks += 1;
        let m = self.parallel_num_workers;

        // Schedulable frontier = dispatched state + whatever remains active.
        let mut active_ids = self.sm.state_ids(STASH_ACTIVE);
        active_ids.push(stepped);

        let width = active_ids.len() as u64;
        if width > self.parallel_max_active_width {
            self.parallel_max_active_width = width;
        }
        // angr-panhl.3: step-weighted width histogram. Bucket BEFORE the
        // <2-width / single-worker early return so width-1 steps (the common
        // narrow-path case the audit must distinguish) are counted.
        self.parallel_width_hist[width_bucket(width)] += 1;
        // <2 schedulable states (or single worker): no cross-boundary steal.
        if active_ids.len() < 2 || m < 2 {
            return;
        }

        // Rebuild the home map for the surviving frontier: preserves sticky
        // homes for states still active, drops the rest (bounds memory to the
        // active width).
        let mut new_of: HashMap<u64, usize> = HashMap::with_capacity(active_ids.len());
        let mut load = vec![0i64; m];
        for &sid in &active_ids {
            if let Some(&w) = self.parallel_worker_of.get(&sid)
                && w < m
            {
                new_of.insert(sid, w);
                load[w] += 1;
            }
        }
        for &sid in &active_ids {
            if let std::collections::hash_map::Entry::Vacant(e) = new_of.entry(sid) {
                let w = load
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &l)| l)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                e.insert(w);
                load[w] += 1;
            }
        }
        if let Some(&h) = new_of.get(&stepped) {
            let idle = load.contains(&0);
            if idle && load[h] >= 2 {
                self.parallel_migrations += 1;
            }
        }
        self.parallel_worker_of = new_of;
    }

    /// Run a closure with an immutable borrow of the pending callback state
    /// identified by `state_id`. Returns Err(PyRuntimeError) when no callback
    /// is pending for that id.
    #[inline]
    pub(crate) fn with_pending<T, F>(&self, state_id: impl Into<StateId>, f: F) -> PyResult<T>
    where
        F: FnOnce(&PendingCallback) -> PyResult<T>,
    {
        let pending = self
            .pending_callbacks
            .get(&state_id.into())
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        f(pending)
    }

    /// Run a closure with a mutable borrow of the pending callback state
    /// identified by `state_id`. Returns Err(PyRuntimeError) when no callback
    /// is pending for that id.
    #[inline]
    pub(crate) fn with_pending_mut<T, F>(
        &mut self,
        state_id: impl Into<StateId>,
        f: F,
    ) -> PyResult<T>
    where
        F: FnOnce(&mut PendingCallback) -> PyResult<T>,
    {
        let pending = self
            .pending_callbacks
            .get_mut(&state_id.into())
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        f(pending)
    }

    /// Run a closure with an immutable borrow of a state by ID.
    /// Looks up the pending callback first (matching `find_state` semantics),
    /// then the stashes. Returns Err(PyValueError) when not found.
    #[inline]
    pub(crate) fn with_state<T, F>(&self, state_id: impl Into<StateId>, f: F) -> PyResult<T>
    where
        F: FnOnce(&RustSimState) -> PyResult<T>,
    {
        let sid = state_id.into();
        let state = self
            .find_state(sid)
            .ok_or_else(|| PyValueError::new_err(format!("state {sid} not found")))?;
        f(state)
    }

    /// Run a closure with a mutable borrow of a state by ID.
    /// Matches `with_state`'s lookup order: pending callback first, then
    /// stashes. Returns Err(PyValueError) when not found.
    #[inline]
    pub(crate) fn with_state_mut<T, F>(&mut self, state_id: impl Into<StateId>, f: F) -> PyResult<T>
    where
        F: FnOnce(&mut RustSimState) -> PyResult<T>,
    {
        let sid = state_id.into();
        let state = self
            .find_state_mut(sid)
            .ok_or_else(|| PyValueError::new_err(format!("state {sid} not found")))?;
        f(state)
    }

    /// Push a new state to the active stash, respecting `max_active_states`.
    /// If the limit is reached, the state is sent to the pruned stash (or
    /// dropped, if `drop_terminal_states` is enabled). Returns true if the
    /// state was added to the active stash, false if pruned.
    #[inline]
    pub(crate) fn push_to_active_or_drop(&mut self, state: RustSimState) -> bool {
        if let Some(limit) = self.max_active_states
            && self.sm.active_count() >= limit
        {
            if !self.max_active_warned {
                // First hit: warn loudly so a runaway explosion is visible.
                // Subsequent hits drop to debug to avoid log spam on tight
                // fork loops. Pruned states are also surfaced via the
                // `pruned` stash counter.
                log::warn!(
                    "max_active_states limit ({limit}) reached; pruning excess forks \
                     (likely path explosion). Raise/disable max_active_states if \
                     this is a legitimately wide exploration."
                );
                self.max_active_warned = true;
            } else {
                log::debug!(
                    "max_active_states limit ({}) reached, pruning state {}",
                    limit,
                    state.state_id()
                );
            }
            self.push_or_drop_terminal(STASH_PRUNED, state);
            return false;
        }
        self.sm.push_active(&*self.policy, state);
        true
    }

    /// Route a successor state to the found/avoid/active stash by its PC.
    ///
    /// Centralizes the find_addrs/avoid_addrs/active triage that the
    /// successor and deferred-fork loops in run_loop_single.rs and resume.rs would
    /// otherwise open-code identically. The `find_addrs` -> STASH_FOUND and
    /// `avoid_addrs` -> push_or_drop_terminal(STASH_AVOID) legs are byte
    /// identical at every call site; only the FOUND-push gating varies.
    ///
    /// When `gate_found_on_sat` is true the FOUND push is skipped for states
    /// the solver *proved* unsatisfiable (`survives_sat_prune`, which keeps a
    /// state whose satisfiability query timed out — the run_loop
    /// successor/loop-exit sites that have not yet filtered satisfiability);
    /// such a state is routed to `STASH_PRUNED` — NOT silently dropped —
    /// exactly as the popped-state path in `check_terminal_conditions` does
    /// for the same condition, so `pruned_count` is arrival-path invariant
    /// (angr-ph300.13). The resume.rs sites pass false because satisfiability
    /// was already established upstream (the whole block is gated on it), so
    /// the FOUND push is unconditional there and the prune arm is unreachable.
    #[inline]
    pub(crate) fn route_successor(&mut self, state: RustSimState, gate_found_on_sat: bool) {
        let spc = state.pc();
        if self.find_addrs.contains(&spc) {
            if !gate_found_on_sat || state.survives_sat_prune(self.constraint_solver.lazy_solves) {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(state);
            } else {
                log::debug!("State at find address 0x{spc:x} is UNSAT, pruning");
                self.push_or_drop_terminal(STASH_PRUNED, state);
            }
        } else if self.avoid_addrs.contains(&spc) {
            self.push_or_drop_terminal(STASH_AVOID, state);
        } else {
            self.push_to_active_or_drop(state);
        }
    }

    /// Push a parallel-path found terminal into `STASH_FOUND`, but respect
    /// `num_find`: once the found stash already holds `num_find` states, the
    /// surplus is routed to `STASH_ACTIVE` instead of collected.
    ///
    /// The parallel loops (wave + steady) can over-produce found terminals
    /// relative to the serial baseline: several workers reach a find address
    /// (or drain a resident frontier state that sits at one) before the
    /// `num_find` cancel propagates, and the steady finalize drain re-routes
    /// residual frontier states through the find gate. Capping here at the
    /// single point where a parallel found terminal lands makes the found-set
    /// *count* worker-invariant and equal to the serial loop, which stops at
    /// `num_find` (angr-op0dn.13.17). The surplus stays in `STASH_ACTIVE` — an
    /// un-collected frontier state at the find pc — so a later resume explore
    /// with a larger `num_find` re-finds it exactly as the serial loop would.
    ///
    /// Retained without Z3: its only callers are the `vex-engine-z3`-gated
    /// parallel wave/steady loops (angr-sqfj8.139).
    #[cfg_attr(not(feature = "vex-engine-z3"), allow(dead_code))]
    pub(crate) fn push_found_capped(&mut self, state: RustSimState) {
        if self.found_count() >= self.num_find {
            self.push_to_active_or_drop(state);
        } else {
            self.sm
                .stashes_mut()
                .entry(STASH_FOUND.to_string())
                .or_default()
                .push_back(state);
        }
    }

    /// Track a state in the state_index.
    #[inline]
    pub(crate) fn index_state(&mut self, state_id: impl Into<StateId>, stash: &str) {
        self.sm.index(state_id.into().raw(), stash);
    }

    /// Rebuild the state_index from scratch by scanning all stashes.
    /// Called after run() to ensure index is up to date for Python API calls.
    pub(crate) fn rebuild_state_index(&mut self) {
        self.sm.rebuild_index();
    }

    /// Find an immutable reference to a state by ID using the index.
    /// Falls back to linear scan if the index is stale.
    pub(crate) fn find_state(&self, state_id: impl Into<StateId>) -> Option<&RustSimState> {
        let sid = state_id.into();
        // Check pending callback states first (during find_predicate evaluation)
        if let Some(pending) = self.pending_callbacks.get(&sid) {
            return Some(&pending.state);
        }
        self.sm.find_state(sid.raw())
    }

    /// Find a mutable reference to a state by ID using the index.
    /// Falls back to linear scan if the index is stale.
    ///
    /// Mirrors `find_state`'s pending-first lookup: while a SimProcedure
    /// callback (or find/avoid predicate) is in flight the state lives in
    /// `pending_callback` and is NOT present in any stash. Without this
    /// pending-aware fallback, write-through FFI shims (the proxy plugins
    /// at angr-4scu memory / angr-qj30 registers) would fail with
    /// "state N not found" any time a Python SimProc wrote to
    /// `state.regs.<name>` or `state.memory.store(...)` during a callback.
    pub(crate) fn find_state_mut(
        &mut self,
        state_id: impl Into<StateId>,
    ) -> Option<&mut RustSimState> {
        let sid = state_id.into();
        if let Some(pending) = self.pending_callbacks.get_mut(&sid) {
            return Some(&mut pending.state);
        }
        self.sm.find_state_mut(sid.raw())
    }

    /// Compute a hash for a state's register tuple for uniqueness checking.
    ///
    /// For each register in uniqueness_registers, gets the concrete value
    /// (or uses u64::MAX as sentinel for symbolic). Hashes the tuple.
    pub(crate) fn compute_register_tuple_hash(&self, state: &RustSimState) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for reg_name in &self.constraint_tracker.uniqueness_registers {
            match state.get_register(reg_name) {
                Some(bv) => {
                    if let Some(val) = bv.as_u64() {
                        val.hash(&mut hasher);
                    } else {
                        // Symbolic register — use sentinel
                        u64::MAX.hash(&mut hasher);
                        // Also hash 1 to distinguish from concrete u64::MAX
                        1u8.hash(&mut hasher);
                    }
                }
                None => {
                    // Register not found — hash 0
                    0u64.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Apply the native uniqueness filter to the active stash.
    ///
    /// Moves states with duplicate register tuples to 'not_unique'.
    /// Called after each step in the run() loop.
    pub(crate) fn apply_uniqueness_filter(&mut self) {
        if self.constraint_tracker.uniqueness_registers.is_empty() {
            return;
        }

        // First pass: compute hashes and find duplicates (immutable borrow of active)
        let to_remove = {
            let active = match self.sm.get(STASH_ACTIVE) {
                Some(s) => s,
                None => return,
            };

            let mut remove_indices = Vec::new();
            for (i, state) in active.iter().enumerate() {
                let hash = self.compute_register_tuple_hash(state);
                if !self.constraint_tracker.uniqueness_set.insert(hash) {
                    remove_indices.push(i);
                }
            }
            remove_indices
        };

        if to_remove.is_empty() {
            return;
        }

        // Second pass: remove duplicates (mutable borrow of stashes)
        let active = match self.sm.get_mut(STASH_ACTIVE) {
            Some(s) => s,
            None => return,
        };
        let mut removed_states = Vec::new();
        for &idx in to_remove.iter().rev() {
            if let Some(state) = active.remove(idx) {
                removed_states.push(state);
            }
        }
        // These states leave STASH_ACTIVE outside `policy.select` — notify so a
        // memoizing policy (LoopHeadRoundRobin's key_cache) doesn't leak a memo
        // entry for a state it will never select again (angr-myzjx.25).
        for state in &removed_states {
            self.policy.on_state_removed(state.state_id());
        }

        if !self.sm.drop_terminal_states() {
            let not_unique = self
                .sm
                .stashes_mut()
                .entry("not_unique".to_string())
                .or_default();
            for state in removed_states {
                not_unique.push_back(state);
            }
        }
        // else dropped states are simply discarded
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended/unconstrained),
    /// or drop it if `drop_terminal_states` is enabled. Increments the
    /// appropriate counter.
    ///
    /// Every terminal routing site should go through this wrapper rather than
    /// calling `self.sm.push_or_drop_terminal` directly (angr-9ke6b.56), so
    /// manager-level bookkeeping added here cannot be silently bypassed. The
    /// one exception is `STASH_ERRORED`, which never drops and therefore has
    /// its own chokepoint, `push_errored` below — see the module docs on
    /// `apply_terminal` in `run_loop.rs`.
    pub(crate) fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        self.sm.push_or_drop_terminal(stash_name, state);
    }

    /// Push a state onto the errored stash, incrementing `errored_count`.
    ///
    /// The errored counterpart of `push_or_drop_terminal` (angr-9ke6b.231):
    /// errored states are never dropped, but they must still be counted, or a
    /// serial run reports `errored_count == 0` where the parallel loop folds
    /// `stats.summarized_errored` (see `fold_parallel_stats` above). Callers
    /// still own recording the `(pc, message, state_id)` triple in
    /// `self.errors` — that is per-site data this wrapper cannot reconstruct.
    pub(crate) fn push_errored(&mut self, state: RustSimState) {
        self.sm.push_errored(state);
    }

    /// Extract procedure arguments from state registers (and stack, when
    /// `num_args` exceeds the register portion of the calling convention).
    ///
    /// Thin adapter over [`extract_args_with_abi`], which holds the extraction
    /// logic and documents the error semantics; the parallel post-step path
    /// adapts the same helper from the scalar `CcSnapshot`.
    pub(crate) fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let cc = &self.environment.calling_convention;
        let abi = ArgExtractAbi {
            arg_registers: cc.arg_registers(),
            pointer_size: cc.pointer_size(),
            stack_arg_offset: Some(cc.stack_arg_offset()),
        };
        extract_args_with_abi(&abi, state, num_args)
    }

    /// Extract syscall arguments from state registers.
    ///
    /// Uses the calling convention's `syscall_arg_registers()` rather than
    /// `arg_registers()`. On Linux amd64 these differ at the 4th argument
    /// (R10 vs RCX). Only MIPS O32 spills syscall args (5+ at `sp+16`); every
    /// other syscall ABI passes `None` for the stack window, so an over-wide
    /// request reports [`ExtractionError::RegisterOverflow`].
    ///
    /// Thin adapter over [`extract_args_with_abi`], which holds the extraction
    /// logic and documents the error semantics; the parallel post-step path
    /// adapts the same helper from the scalar `CcSnapshot`.
    ///
    /// Retained as a direct-call test harness (`helpers_tests.rs`); the
    /// production syscall path now extracts via `CcSnapshot::extract_syscall_args`
    /// inside the post-step core (angr-vh834). Gated to test builds — no
    /// production caller, so it compiles only when the test harness needs it.
    #[cfg(test)]
    pub(crate) fn extract_syscall_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let cc = &self.environment.calling_convention;
        let abi = ArgExtractAbi {
            arg_registers: cc.syscall_arg_registers(),
            pointer_size: cc.pointer_size(),
            stack_arg_offset: cc.syscall_stack_arg_offset(),
        };
        extract_args_with_abi(&abi, state, num_args)
    }

    /// Get the caller's return address at a call boundary, resolved through the
    /// calling convention.
    ///
    /// On link-register ABIs (ARM/ARM64/MIPS) `CallingConvention::get_return_addr`
    /// reads LR/X30/`$ra` out of the register file and needs no memory view. On
    /// stack-return ABIs (x86/AMD64) it declines — it is handed `None` for
    /// memory, mirroring the `Returned` arm of `RustExplorationManager::step_one`
    /// — and we read `[sp]` here, where the state's memory view is available.
    ///
    /// Routing through the convention instead of reading `[sp]` unconditionally
    /// is a precondition for `CallingConvention::link_register` being wired up:
    /// the serial `NativeProcDisposition::SubCall` arm feeds this value into
    /// `NativeSubcall::caller_return_addr`, which the resume path later installs
    /// as the PC. On a link-register ABI `[sp]` is not the return address, so
    /// that would jump to whatever happened to be on the stack (bead
    /// angr-9ke6b.3).
    pub(crate) fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let cc = &self.environment.calling_convention;
        {
            let ctx = state.solver().borrow();
            if let Some(addr) = cc.get_return_addr(state.registers(), None, &ctx) {
                return Some(addr);
            }
        }
        if !cc.pops_return_addr() {
            // Link-register ABI whose LR/$ra is symbolic or unset — there is no
            // stack slot to fall back to.
            return None;
        }
        let sp = state.get_sp().as_u64()?;
        state.memory_load(sp, cc.pointer_size()).ok()?.as_u64()
    }

    /// [`Self::get_return_addr`], substituting `0` and logging when the
    /// answer is symbolic/unavailable instead of silently carrying a
    /// plausible-looking-but-wrong address forward. Mirrors
    /// `VEXInterpreter::get_return_addr_or_log` (same contract, same
    /// rationale — `angr-sqfj8.62`).
    // SILENT(cat-c): a symbolic/unavailable return address collapsing to the
    // literal 0 is a wrong-answer risk; route through this single logged
    // fallback rather than a bare `.unwrap_or(0)`.
    pub(crate) fn get_return_addr_or_log(&self, state: &RustSimState, context: &str) -> u64 {
        silent_default!(
            cat_c,
            self.get_return_addr(state),
            0,
            "get_return_addr() returned None ({context}); using return_addr=0 \
             — likely a symbolic or unavailable return address collapsed to a wrong value"
        )
    }
}

/// The ABI facts [`extract_args_with_abi`] needs, decoupled from where they
/// came from.
///
/// The single-threaded path reads them off the manager's
/// `Box<dyn CallingConvention>`; the parallel post-step path reads them off the
/// scalar `CcSnapshot` (the trait object is not `Clone`). Both funnel through
/// this borrow so the extraction logic exists once (angr-sqfj8.145) — the same
/// adapter shape [`SubcallAbi`](crate::exploration::stepping::SubcallAbi) uses
/// for `setup_native_subcall_with_abi`.
///
/// `arg_registers` is the *procedure* or *syscall* register window depending on
/// which caller built it; on Linux amd64 those differ at the 4th argument
/// (R10 vs RCX).
pub(crate) struct ArgExtractAbi<'a> {
    pub(crate) arg_registers: &'a [u32],
    pub(crate) pointer_size: u32,
    /// Byte offset from SP of the first stack-spilled argument, or `None` for
    /// an ABI that has no stack window (every syscall ABI except MIPS O32).
    pub(crate) stack_arg_offset: Option<u64>,
}

/// Extract `num_args` ABI arguments from a state's registers, spilling to the
/// stack once the register window is exhausted.
///
/// Returns an [`ExtractionError`] instead of silently zero-padding when the ABI
/// has no stack window ([`ExtractionError::RegisterOverflow`]), the stack
/// pointer is symbolic ([`ExtractionError::SpSymbolic`]), or a stack slot
/// cannot be read ([`ExtractionError::StackUnmapped`]). Callers — the native
/// procedure/syscall dispatchers in `core_outcome_handlers.rs` and
/// `run_loop_single.rs` — treat any error as a signal to skip the native fast
/// path and fall through to the Python callback rather than handing the handler
/// a fabricated `RustBV::zero` that would mask a real stack-setup bug (the
/// angr-ydli fix to the trait method).
///
/// Every procedure ABI supplies a stack window, so `RegisterOverflow` is
/// reachable only from the syscall adapters. MIPS O32 is the one syscall ABI
/// that spills (args 5+ at `sp+16`).
pub(crate) fn extract_args_with_abi(
    abi: &ArgExtractAbi<'_>,
    state: &RustSimState,
    num_args: usize,
) -> Result<Vec<RustBV>, ExtractionError> {
    let ptr_size = abi.pointer_size;
    let mut args = Vec::with_capacity(num_args);
    for &offset in abi.arg_registers.iter().take(num_args) {
        args.push(state.get_register_by_offset(offset, ptr_size));
    }

    if args.len() < num_args {
        let stack_offset = abi
            .stack_arg_offset
            .ok_or(ExtractionError::RegisterOverflow {
                requested: num_args,
                available: abi.arg_registers.len(),
            })?;
        let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
        let stack_start = sp + stack_offset;
        let already = args.len();
        for i in 0..(num_args - already) {
            let addr = stack_start + (i as u64 * ptr_size as u64);
            let value =
                state
                    .memory_load(addr, ptr_size)
                    .map_err(|_| ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    })?;
            args.push(value);
        }
    }

    Ok(args)
}

/// Prepare the per-callback solver context for a Python round-trip.
///
/// Two things every "return to Python" exit needs: (1) a pre-callback
/// snapshot of `state`, but *only* when deferred forks need it — `state.fork()`
/// clones the Z3 solver (~3-40ms), so we skip it otherwise; and (2) a
/// `RustSolverContext` wrapping the state's shared solver (an O(1) `Rc` clone,
/// not a fork) so the callback evaluates against the live constraints.
///
/// Returns `(pre_callback_snapshot, shared_ctx)` ready to hand to
/// `PendingCallback::with_context`. The Hook arm in `stepping.rs` keeps its own
/// inline copy because it interleaves fork-timing profiling around the snapshot;
/// the symbolic-branch / VEX-fallback exits pass `(None, None)` by design and
/// do not use this.
pub(crate) fn prepare_shared_callback_solver(
    state: &RustSimState,
    deferred_forks: &[DeferredFork],
) -> (Option<RustSimState>, RustSolverContext) {
    let pre_callback_snapshot = if !deferred_forks.is_empty() {
        Some(state.fork())
    } else {
        None
    };
    let solver_ref = state.solver();
    let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
    (pre_callback_snapshot, shared_ctx)
}

/// Pop the return-address slot off the stack after a native procedure returns.
///
/// Only stack-return ABIs (x86/AMD64) push the return address, so only they
/// advance SP here; on link-register ABIs (ARM/ARM64 LR/X30, MIPS `$ra`) the
/// caller's return address never went on the stack and bumping SP would
/// discard a live stack slot (angr-sqfj8.37), so this is a no-op there.
///
/// Both native-return sites — `core_outcome_handlers.rs`'s
/// `handle_simprocedure_core` and the inline native path in
/// `run_loop_single.rs`'s `step_one` — must call this rather than inlining
/// the bump: the two used to be kept in sync by a comment alone, and both
/// drifted into the same `get_sp().as_u64().unwrap_or(0)` bug (angr-c7xno.29).
///
/// A symbolic SP is bumped **symbolically** (`sp + ptr_size`) rather than
/// collapsed to a concrete value. `extract_args_with_abi` only reads SP when
/// the argument count overflows the register window, so an all-register-args
/// native proc can run to completion with SP still symbolic; the old
/// `unwrap_or(0)` rewrote that SP to a bogus concrete `ptr_size`.
pub(crate) fn advance_sp_past_return_addr(state: &mut RustSimState, pops_return_addr: bool) {
    if !pops_return_addr {
        return;
    }
    let ptr_size = state.arch().bytes() as u64;
    let bits = state.arch().bits();
    let sp = state.get_sp();
    let bumped = match sp.as_u64() {
        Some(concrete_sp) => RustBV::concrete((concrete_sp.wrapping_add(ptr_size)) as u128, bits),
        None => {
            log::debug!(
                "native-proc return: stack pointer is symbolic; advancing it \
                 symbolically by {ptr_size} bytes instead of concretizing"
            );
            let delta = RustBV::concrete(ptr_size as u128, sp.width());
            let ctx = state.solver().borrow();
            sp.add_into(delta, &ctx)
        }
    };
    state.set_sp(bumped);
}

test_submod!("helpers_tests.rs" => tests);
