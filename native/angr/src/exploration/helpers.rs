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
        // narrow-path case the audit must distinguish) are counted. Buckets:
        // [==1, ==2, 3–4, 5–8, ≥9].
        let bucket = match width {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        };
        self.parallel_width_hist[bucket] += 1;
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
    /// successor and deferred-fork loops in run_loop.rs and resume.rs would
    /// otherwise open-code identically. The `find_addrs` -> STASH_FOUND and
    /// `avoid_addrs` -> push_or_drop_terminal(STASH_AVOID) legs are byte
    /// identical at every call site; only the FOUND-push gating varies.
    ///
    /// When `gate_found_on_sat` is true the FOUND push is skipped for states
    /// that are neither `lazy_solves` nor `satisfiable()` (the run_loop
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
            if !gate_found_on_sat || self.constraint_solver.lazy_solves || state.satisfiable() {
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

    /// Apply native exploration techniques to the active stash.
    ///
    /// Runs after each step in the exploration loop. Checks each technique
    /// and moves/stops states as needed, entirely in Rust.
    pub(crate) fn apply_native_techniques(&mut self) -> bool {
        if self.native_techniques.is_empty() {
            return false;
        }

        let mut complete = false;

        for tech_idx in 0..self.native_techniques.len() {
            // MergePoint is handled out-of-band because merging calls
            // `_merge_states` (needs `&mut self`), which conflicts with the
            // `&mut self.native_techniques[tech_idx]` the match would hold.
            if let NativeTechnique::MergePoint {
                address,
                wait_counter_limit,
                wait_stash,
                ..
            } = &self.native_techniques[tech_idx]
            {
                let address = *address;
                let limit = *wait_counter_limit;
                let wait_stash = wait_stash.clone();
                self.apply_merge_point(tech_idx, address, limit, &wait_stash);
                continue;
            }
            match &mut self.native_techniques[tech_idx] {
                NativeTechnique::Timeout {
                    timeout_secs,
                    start_time,
                } => {
                    let start = start_time.get_or_insert_with(std::time::Instant::now);
                    if start.elapsed().as_secs_f64() > *timeout_secs {
                        log::info!(
                            "Native Timeout: exploration timed out after {timeout_secs:.1}s"
                        );
                        // Move all active states to "timeout" stash
                        if let Some(active) = self.sm.get_mut(STASH_ACTIVE) {
                            let states: Vec<_> = active.drain(..).collect();
                            let timeout_stash = self
                                .sm
                                .stashes_mut()
                                .entry("timeout".to_string())
                                .or_default();
                            for s in states {
                                timeout_stash.push_back(s);
                            }
                        }
                        complete = true;
                    }
                }
                NativeTechnique::LengthLimiter { max_length, drop } => {
                    let max_len = *max_length;
                    let do_drop = *drop;

                    // Find states exceeding the length limit
                    let to_remove = {
                        let active = match self.sm.get(STASH_ACTIVE) {
                            Some(s) => s,
                            None => continue,
                        };
                        let mut indices = Vec::new();
                        for (i, state) in active.iter().enumerate() {
                            if state.history().len() > max_len {
                                indices.push(i);
                            }
                        }
                        indices
                    };

                    if to_remove.is_empty() {
                        continue;
                    }

                    let active = match self.sm.get_mut(STASH_ACTIVE) {
                        Some(s) => s,
                        None => continue,
                    };
                    let mut removed_states = Vec::new();
                    for &idx in to_remove.iter().rev() {
                        if let Some(state) = active.remove(idx) {
                            removed_states.push(state);
                        }
                    }

                    if do_drop {
                        // States are simply discarded
                    } else {
                        let cut_stash = self.sm.stashes_mut().entry("cut".to_string()).or_default();
                        for state in removed_states {
                            cut_stash.push_back(state);
                        }
                    }
                }
                NativeTechnique::LoopBound {
                    bound,
                    discard_stash,
                } => {
                    let max_bound = *bound;
                    let stash_name = discard_stash.clone();

                    // Find states where any address appears more than `bound` times
                    let to_remove = {
                        let active = match self.sm.get(STASH_ACTIVE) {
                            Some(s) => s,
                            None => continue,
                        };
                        let mut indices = Vec::new();
                        for (i, state) in active.iter().enumerate() {
                            let history = state.history();
                            if Self::exceeds_loop_bound(history, max_bound) {
                                indices.push(i);
                            }
                        }
                        indices
                    };

                    if to_remove.is_empty() {
                        continue;
                    }

                    let active = match self.sm.get_mut(STASH_ACTIVE) {
                        Some(s) => s,
                        None => continue,
                    };
                    let mut removed_states = Vec::new();
                    for &idx in to_remove.iter().rev() {
                        if let Some(state) = active.remove(idx) {
                            removed_states.push(state);
                        }
                    }

                    if !self.sm.drop_terminal_states() {
                        let target = self.sm.stashes_mut().entry(stash_name).or_default();
                        for state in removed_states {
                            target.push_back(state);
                        }
                    }
                }
                // Handled out-of-band above the match.
                NativeTechnique::MergePoint { .. } => {}
            }
        }

        complete
    }

    /// One post-step round of the native MergePoint technique
    /// (angr-op0dn.11.5). Mirrors `ManualMergepoint.step`: park active states
    /// sitting at `address` into `wait_stash`, then — once the active frontier
    /// drains or `limit` rounds elapse since the last arrival — group the
    /// waiters by callstack and merge each ≥2 group via `_merge_states`.
    ///
    /// Kept as a dedicated `&mut self` method (not a match arm) because
    /// `_merge_states` reborrows all of `self`; the caller copies the immutable
    /// technique fields out first and this method writes `counter` back through
    /// short-lived reborrows keyed by `tech_idx`.
    fn apply_merge_point(&mut self, tech_idx: usize, address: u64, limit: usize, wait_stash: &str) {
        // 1. Park active states at the merge address; a fresh arrival resets
        //    the wait counter (ManualMergepoint: `self.wait_counter = 0`).
        let parked = self.park_states_at_address(STASH_ACTIVE, wait_stash, address);
        if parked > 0
            && let NativeTechnique::MergePoint { counter, .. } =
                &mut self.native_techniques[tech_idx]
        {
            *counter = 0;
        }

        let wait_len = self.sm.get(wait_stash).map_or(0, VecDeque::len);
        if wait_len == 0 {
            return;
        }

        // 2. Tick the round counter (once per post-step apply).
        let counter_now = match &mut self.native_techniques[tech_idx] {
            NativeTechnique::MergePoint { counter, .. } => {
                *counter += 1;
                *counter
            }
            _ => return,
        };

        // 3. Keep waiting while the frontier is non-empty and we are under the
        //    round limit — more paths may still reconverge here.
        let active_empty = self.sm.get(STASH_ACTIVE).is_none_or(VecDeque::is_empty);
        if !active_empty && counter_now < limit {
            return;
        }

        // 4. A single waiter has nothing to merge with: release it unmerged so
        //    the path count is preserved and exploration does not stall.
        if wait_len == 1 {
            let _ = self._move_states(wait_stash, STASH_ACTIVE, None);
            return;
        }

        // 5. Group waiters by callstack (return-address chain — the same signal
        //    the reconvergence sampler reads) and merge each ≥2 group.
        self.merge_waiters_by_callstack(wait_stash);
    }

    /// Move every state in `from` whose pc equals `address` into `to`,
    /// updating the state index. Returns the number of states moved.
    fn park_states_at_address(&mut self, from: &str, to: &str, address: u64) -> usize {
        let moved: Vec<RustSimState> = {
            let stash = match self.sm.get_mut(from) {
                Some(s) => s,
                None => return 0,
            };
            let mut kept = VecDeque::with_capacity(stash.len());
            let mut moved = Vec::new();
            for state in stash.drain(..) {
                if state.pc() == address {
                    moved.push(state);
                } else {
                    kept.push_back(state);
                }
            }
            *stash = kept;
            moved
        };
        let count = moved.len();
        if count == 0 {
            return 0;
        }
        for state in moved {
            let sid = state.state_id();
            self.sm.ensure_stash(to).push_back(state);
            self.sm.index(sid, to);
        }
        count
    }

    /// Group the states currently in `wait_stash` by callstack return-address
    /// chain (first-appearance order for determinism) and merge each group of
    /// ≥2 into the active stash via `_merge_states`, dropping the consumed
    /// sources. Lone-callstack waiters are released back to active unmerged.
    fn merge_waiters_by_callstack(&mut self, wait_stash: &str) {
        // Build callstack-keyed groups in first-seen order (HashMap iteration
        // is unordered; ManualMergepoint parity requires deterministic merges).
        let groups: Vec<Vec<u64>> = {
            let stash = match self.sm.get(wait_stash) {
                Some(s) if !s.is_empty() => s,
                _ => return,
            };
            let mut order: Vec<Vec<u64>> = Vec::new();
            let mut by_key: HashMap<Vec<u64>, Vec<u64>> = HashMap::new();
            for state in stash.iter() {
                let key: Vec<u64> = state.call_stack().iter().map(|e| e.return_addr).collect();
                if !by_key.contains_key(&key) {
                    order.push(key.clone());
                }
                by_key.entry(key).or_default().push(state.state_id());
            }
            order
                .into_iter()
                .map(|k| by_key.remove(&k).expect("key inserted above"))
                .collect()
        };

        for ids in groups {
            if ids.len() < 2 {
                // Lone callstack: release the single waiter back to active.
                self.move_state_by_id(wait_stash, STASH_ACTIVE, ids[0]);
                continue;
            }
            // `_merge_states` forks the sources and pushes the merged state to
            // active (bumping `states_merged_native`); it does NOT consume the
            // originals, so drop them from the wait stash afterwards.
            match self._merge_states(ids.clone(), STASH_ACTIVE) {
                Ok(_) => self.drop_states_by_id(wait_stash, &ids),
                Err(e) => {
                    log::warn!("MergePoint merge failed, leaving waiters parked: {e}");
                }
            }
        }
    }

    /// Move a single state (by id) from `from` to `to`, updating the index.
    fn move_state_by_id(&mut self, from: &str, to: &str, sid: u64) {
        let state = {
            let stash = match self.sm.get_mut(from) {
                Some(s) => s,
                None => return,
            };
            match stash.iter().position(|s| s.state_id() == sid) {
                Some(pos) => stash.remove(pos),
                None => None,
            }
        };
        if let Some(state) = state {
            self.sm.ensure_stash(to).push_back(state);
            self.sm.index(sid, to);
        }
    }

    /// Drop the given state ids from `stash`, removing their index entries.
    fn drop_states_by_id(&mut self, stash: &str, ids: &[u64]) {
        let drop: std::collections::HashSet<u64> = ids.iter().copied().collect();
        if let Some(s) = self.sm.get_mut(stash) {
            s.retain(|state| !drop.contains(&state.state_id()));
        }
        for &sid in ids {
            self.sm.unindex(sid);
        }
    }

    /// Check if any address in the history exceeds the loop bound.
    fn exceeds_loop_bound(history: &std::collections::VecDeque<u64>, bound: usize) -> bool {
        // Use a small HashMap to count address frequencies.
        // For typical histories this is fast since most addresses appear once.
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &addr in history {
            let count = counts.entry(addr).or_insert(0);
            *count += 1;
            if *count > bound {
                return true;
            }
        }
        false
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    pub(crate) fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        self.sm.push_or_drop_terminal(stash_name, state);
    }

    /// Sync constraints from Python callbacks back to the Rust state's solver.
    ///
    /// Python SimProcedures may add new constraints (e.g., strcmp conditions);
    /// this method syncs those new constraints BACK to Rust after the callback.
    /// The reverse direction (Rust→Python) is handled by attaching a
    /// `RustSolverContext` to the Python state in
    /// `rust_callback_dispatch._install_rust_solver_on_callback_state`, so
    /// constraints never need to be replayed across the FFI.
    ///
    /// Without this, constraints added by SimProcedures would be lost when
    /// Rust resumes execution, leading to incorrect symbolic evaluation.
    ///
    /// Conversion strategy: claripy_to_rustbv first (preserves assumed_constraints
    /// tracking that Python re-import depends on), then fall back to lossless Z3
    /// pointer extraction when claripy_to_rustbv hits an unsupported op (FP, etc).
    /// Both paths share the same Z3 context with claripy via the shared backend.
    ///
    /// Returns:
    ///   - Ok(true): Constraints synced and state is SAT (satisfiable)
    ///   - Ok(false): State became UNSAT after syncing - should be pruned (P12)
    ///   - Err: Python error during sync
    pub(crate) fn sync_constraints_from_python(
        &self,
        py: Python<'_>,
        state: &RustSimState,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        use pyo3::types::PyListMethods;

        let solver_ref = state.solver();
        let sym_ctx = solver_ref.borrow();
        let ctx_ref: &SymContext = &sym_ctx;

        let mut success_count = 0usize;
        let mut z3_ptr_fallback_count = 0usize;
        let mut failed_count = 0usize;

        // Lazily resolve claripy.backends.z3 once for the Z3 ptr fallback path.
        // None on first failure so we don't keep paying the import cost.
        #[cfg(feature = "vex-engine-z3")]
        let mut z3_backend: Option<Bound<'_, PyAny>> = None;

        // Extract list items - we need to convert each to RustBV
        let len = constraints.len();
        for i in 0..len {
            // Use get_item with usize index
            if let Ok(constraint) = constraints.get_item(i) {
                // Convert claripy AST to RustBV
                match claripy_to_rustbv(py, &constraint, ctx_ref) {
                    Ok(bv) => {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                                success_count += 1;
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                                success_count += 1;
                            }
                        }
                        #[cfg(not(feature = "vex-engine-z3"))]
                        {
                            // Without Z3, constraints are tracked but not solved
                            success_count += 1;
                        }
                    }
                    Err(e) => {
                        // Fallback: extract the constraint's underlying Z3 AST
                        // pointer via claripy.backends.z3 and assert it on the
                        // solver directly. claripy and the Rust solver share a
                        // Z3 context, so the pointer is valid here. This rescues
                        // constraints that use ops claripy_to_rustbv doesn't
                        // model (e.g. FP), keeping the round-trip lossless.
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            let backend = match z3_backend.as_ref() {
                                Some(b) => Some(b.clone()),
                                None => match Self::resolve_claripy_z3_backend(py) {
                                    Ok(b) => {
                                        z3_backend = Some(b.clone());
                                        Some(b)
                                    }
                                    Err(import_err) => {
                                        log::debug!(
                                            "claripy.backends.z3 unavailable for fallback: {import_err}"
                                        );
                                        None
                                    }
                                },
                            };

                            let z3_ast = backend.as_ref().and_then(|b| {
                                Self::extract_z3_ptr_from_claripy(b, &constraint)
                                    .ok()
                                    .flatten()
                            });

                            if let Some(ast) = z3_ast {
                                sym_ctx.add_constraint_raw(ast);
                                z3_ptr_fallback_count += 1;
                                success_count += 1;
                                log::debug!(
                                    "Constraint {i} fell back to Z3 ptr (claripy_to_rustbv: {e})"
                                );
                                continue;
                            }
                        }

                        failed_count += 1;
                        log::warn!(
                            "Constraint {i} conversion failed: {e}. Solver state may diverge."
                        );
                    }
                }
            }
        }

        if z3_ptr_fallback_count > 0 {
            log::debug!(
                "sync_constraints_from_python: {z3_ptr_fallback_count} constraints rescued via Z3 ptr fallback"
            );
        }

        if failed_count > 0 {
            log::warn!(
                "sync_constraints_from_python: {}/{} constraints failed to convert",
                failed_count,
                failed_count + success_count
            );
        }

        if success_count > 0 {
            log::debug!("Synced {success_count} constraints from Python to Rust");
        }

        // P12: Check satisfiability and return status so callers can prune UNSAT states
        #[cfg(feature = "vex-engine-z3")]
        {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P12: Constraints are UNSAT after syncing {success_count} from Python (failed={failed_count}). \
                     Returning false to trigger pruning."
                );
                return Ok(false);
            }
        }

        // P14: If many constraints failed to convert, do explicit SAT check
        // Failed conversions can leave state in divergent state
        #[cfg(feature = "vex-engine-z3")]
        if failed_count > 0 && success_count > 0 {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P14: State became UNSAT with partial constraint sync ({}/{} failed). Pruning.",
                    failed_count,
                    failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Resolve `claripy.backends.z3` once per call to `sync_constraints_from_python`.
    /// Returned as a `Bound<PyAny>` because that's what `convert(...)` needs.
    #[cfg(feature = "vex-engine-z3")]
    fn resolve_claripy_z3_backend(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let claripy = py.import("claripy")?;
        let backends = claripy.getattr("backends")?;
        backends.getattr("z3")
    }

    /// Extract a typed [`Z3AstPtr`] from a claripy AST by going through
    /// `claripy.backends.z3.convert(ast).as_ast().value`. Returns `Ok(None)`
    /// if the conversion succeeded but the extracted pointer is null;
    /// returns `Err` if the Python conversion path itself failed.
    ///
    /// The returned handle has its own `Z3_inc_ref` ref; callers can drop
    /// it without affecting claripy's cached AST.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_z3_ptr_from_claripy(
        z3_backend: &Bound<'_, PyAny>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Z3AstPtr>> {
        let z3_obj = z3_backend.call_method1("convert", (ast,))?;
        let raw_ast = z3_obj.call_method0("as_ast")?;
        let ptr = raw_ast.getattr("value")?.extract::<usize>()?;
        let ctx = z3::Context::thread_local();
        // SAFETY: claripy's z3 backend returned this pointer for a live
        // AST it holds in its own cache; the AST is in the process-global
        // Z3 context, which matches our thread-local context.
        Ok(unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) })
    }

    /// Extract procedure arguments from state registers (and stack, when
    /// `num_args` exceeds the register portion of the calling convention).
    ///
    /// Returns an [`ExtractionError`] instead of silently zero-padding when
    /// the stack pointer is symbolic or a stack slot cannot be read. Callers
    /// (the native-procedure dispatchers in `stepping.rs` / `run_loop.rs`)
    /// treat any error as a signal to skip the native fast path and fall
    /// through to the Python SimProcedure callback rather than handing the
    /// handler a fabricated `RustBV::zero` that would silently mask a real
    /// stack-setup bug (mirror of the `angr-ydli` fix to the trait method).
    pub(crate) fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let arg_regs = self.environment.calling_convention.arg_registers();
        let ptr_size = self.environment.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        for &offset in arg_regs.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + self.environment.calling_convention.stack_arg_offset();
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        drop(ctx);
        Ok(args)
    }

    /// Extract syscall arguments from state registers.
    ///
    /// Uses the calling convention's `syscall_arg_registers()` rather than
    /// `arg_registers()`. On Linux amd64 these differ at the 4th argument
    /// (R10 vs RCX). Most syscall ABIs expose every argument in registers, so
    /// a request for more arguments than the register window holds returns
    /// [`ExtractionError::RegisterOverflow`] and the caller falls through to
    /// the Python syscall callback instead of running a native handler with
    /// fabricated zero arguments.
    ///
    /// The exception is MIPS O32, whose syscall ABI spills args 5+ onto the
    /// stack at `sp+16` (`syscall_stack_arg_offset()`). For those ABIs the
    /// remaining args are read from the stack, but only when SP is concrete
    /// and the slots are mapped — a symbolic SP ([`ExtractionError::SpSymbolic`])
    /// or an unmapped slot ([`ExtractionError::StackUnmapped`]) still falls
    /// through to Python rather than fabricating values.
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
        let arg_regs = cc.syscall_arg_registers();
        let ptr_size = cc.pointer_size();
        let mut args = Vec::with_capacity(num_args);
        for &offset in arg_regs.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            // Need stack args. Only ABIs that spill syscall args to the stack
            // (currently MIPS O32) provide a syscall stack offset; otherwise
            // report RegisterOverflow so the caller falls through to Python.
            let stack_offset =
                cc.syscall_stack_arg_offset()
                    .ok_or(ExtractionError::RegisterOverflow {
                        requested: num_args,
                        available: arg_regs.len(),
                    })?;
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + stack_offset;
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        Ok(args)
    }

    /// Get return address from stack.
    pub(crate) fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let sp = state.get_sp().as_u64()?;
        let ptr_size = self.environment.calling_convention.pointer_size();

        // On x86/AMD64, return address is at [rsp] after call
        state.memory_load(sp, ptr_size).ok()?.as_u64()
    }
}

/// Pack a concrete byte slice into 16-byte `RustBV::concrete` chunks and store
/// each via the caller-supplied sink.
///
/// angr-5aj8: `RustBV::Concrete` is backed by a `u128`, so packing more than
/// 16 bytes into a single value shift-overflows for byte indices >= 16 and the
/// downstream `store_concrete` page-fill loop emits a 16-byte-cycle pattern
/// across the entire `data.len()` range, corrupting memory wholesale. This is
/// the canonical safe loop (was hand-copied into `_set_state_memory_concrete`
/// and `_pending_memory_store`); the sink closure abstracts the only
/// divergence between call sites (which memory API to write through, and how
/// to map its error into a `PyErr`).
pub(crate) fn store_concrete_bytes_chunked<F>(addr: u64, data: &[u8], mut store: F) -> PyResult<()>
where
    F: FnMut(u64, RustBV) -> PyResult<()>,
{
    let mut offset = 0usize;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let chunk_size = remaining.min(16);
        let chunk = &data[offset..offset + chunk_size];
        let width = (chunk_size * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        store(addr + offset as u64, bv)?;
        offset += chunk_size;
    }
    Ok(())
}

/// Unpack the low `size` bytes of a `u128` into a little-endian byte vector.
///
/// `size` must be `<= 16`; for `i >= 16` the `val >> (i * 8)` shift wraps mod
/// 128 and would repeat earlier bytes (callers that need wider reads chunk
/// first — see `_get_state_memory`). This is the read-side counterpart of the
/// pack loop in `store_concrete_bytes_chunked`, hand-copied across the eval /
/// memory-get paths in `state_api`/`pending_api` before consolidation.
pub(crate) fn u128_to_le_bytes(val: u128, size: usize) -> Vec<u8> {
    (0..size).map(|i| (val >> (i * 8)) as u8).collect()
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

/// Install a deferred fork's branch guard on the continuing state, firing the
/// `state.inspect.constraints` BP around the add (angr-op0dn.14.4.1).
///
/// `is_true` selects the polarity: `assume_true(guard)` for the taken path,
/// `assume_false(guard)` for the fallthrough. Mirrors Python's
/// `state_plugins/solver.py::add`, which fires `constraints` BP_BEFORE with
/// `added_constraints` and BP_AFTER once the solver has them — here the
/// successor receiving the guard is the base state itself (Rust mutates the
/// stepped state in place and mints a *separate* state for the unexplored
/// side, whose inverted guard is installed inside `build_unexplored_fork`).
///
/// Bit 19 = `_INSPECT_EVENT_SPECS["constraints"]`; the no-BP case costs a
/// single relaxed atomic load. Mutation of `added_constraints` by the BP is
/// NOT honored on this path (the guard is already lowered to a `RustBV`);
/// the `RustSolverProxyPlugin.add` path does honor it.
#[inline]
pub(crate) fn add_fork_guard_constraint(
    callbacks: Option<&PythonCallbacks>,
    state: &RustSimState,
    guard: &RustBV,
    is_true: bool,
) {
    let cb = callbacks.filter(|c| c.inspect_event_enabled(19));
    let state_id = state.state_id() as i64;
    if let Some(c) = cb
        && let Err(e) = c.call_inspect_constraints(state_id, "before", guard, is_true)
    {
        log::debug!("constraints inspect dispatch raised (state {state_id}, before): {e}");
    }
    if is_true {
        state.solver().borrow().assume_true(guard);
    } else {
        state.solver().borrow().assume_false(guard);
    }
    if let Some(c) = cb
        && let Err(e) = c.call_inspect_constraints(state_id, "after", guard, is_true)
    {
        log::debug!("constraints inspect dispatch raised (state {state_id}, after): {e}");
    }
}

/// Reconstruct a deferred fork's branch condition from its stored claripy AST
/// — the "P11" fallback — when the condition is absent from
/// `stored_conditions`.
///
/// Returns `None` when `stored_condition` is already `Some` (nothing to
/// reconstruct), when the fork carries no `condition_ast`, or when the
/// claripy→RustBV conversion fails. The returned owned `RustBV` is evaluated
/// against `fork_base`'s solver context so it shares the base's z3
/// declarations.
///
/// This is the single source of truth for the four verbatim copies that used
/// to live inline in `_resume_after_simprocedure`, `_deadend_pending_callback`,
/// `_resume_after_symbolic_branch` (resume.rs) and the run-loop callback path
/// (run_loop.rs). angr-ph300.7 fixed a drop-bug by porting this arm into the
/// deadend copy verbatim; consolidating removes that copy-paste hazard (memory
/// invariant-deferred-fork-p11-p15-fallback).
pub(crate) fn reconstruct_deferred_fork_condition(
    stored_condition: Option<&RustBV>,
    fork: &DeferredFork,
    fork_base: &RustSimState,
) -> Option<RustBV> {
    if stored_condition.is_some() {
        return None;
    }
    let py_ast = fork.condition_ast.as_ref()?;
    Python::attach(|py| {
        let ast = py_ast.bind(py);
        let solver_ref = fork_base.solver();
        let ctx: &SymContext = &solver_ref.borrow();
        crate::claripy_bridge::claripy_to_rustbv(py, ast, ctx).ok()
    })
}

/// Build the unexplored-path fork for one deferred branch.
///
/// Every deferred-fork materialization site (the two `stepping.rs` helpers,
/// [`materialize_deferred_forks`], and the parallel mirror in
/// `core_outcome_handlers.rs`) constructs the opposite-path state with the same
/// byte-identical 3-way branch: prefer a pre-branch solver `snapshot` (and
/// re-assume the *opposite* constraint onto it) when one was captured,
/// otherwise `fork_false` / `fork_true` off `base` depending on which side the
/// main path took. The forked state's PC is then set to
/// `fork.unexplored_target`.
pub(crate) fn build_unexplored_fork(
    base: &RustSimState,
    fork: &DeferredFork,
    condition: &RustBV,
    snapshots: &mut FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    priors: &PriorGuards,
) -> RustSimState {
    let mut forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
        // Snapshot predates the branch constraint, so re-assume the opposite
        // side to keep the unexplored path's constraints consistent.
        let f = base.fork_from_snapshot(snapshot);
        // The snapshot is a clone of the *interpreter's* block solver, which
        // deliberately never assumes taken-path guards permanently (see the
        // NOTE in `interpreter/statements.rs`) — so it carries NONE of the
        // guards from earlier deferred branches in the same step, no matter
        // how well-guarded `base` is. Replay them here (angr-62ar5).
        priors.replay_onto(&f);
        if fork.path_taken {
            f.solver().borrow().assume_false(condition);
        } else {
            f.solver().borrow().assume_true(condition);
        }
        f
    } else {
        let f = if fork.path_taken {
            base.fork_false(condition)
        } else {
            base.fork_true(condition)
        };
        if !priors.base_carries {
            priors.replay_onto(&f);
        }
        f
    };
    forked.set_pc(fork.unexplored_target);
    forked
}

/// The taken-path guards of every deferred fork already materialized in this
/// step, in dispatch order, plus whether the caller's fork base already carries
/// them.
///
/// A fork built for branch *i* must inherit the decisions of branches
/// `0..i` — they are on its path by construction. Callers that apply each
/// guard to the continuing state as they iterate (`stepping.rs`,
/// `core_outcome_handlers.rs`) fork off a base that already has them and set
/// `base_carries: true`; [`materialize_deferred_forks`] forks off a fixed
/// guard-free base and sets `false`. The snapshot path replays them
/// unconditionally — see [`build_unexplored_fork`].
pub(crate) struct PriorGuards {
    guards: Vec<(RustBV, bool)>,
    base_carries: bool,
}

impl PriorGuards {
    /// Fresh accumulator. `base_carries` records whether the caller's fork base
    /// already holds the taken-path guards (see the struct doc); it gates the
    /// non-snapshot replay in [`build_unexplored_fork`].
    pub(crate) fn new(base_carries: bool) -> Self {
        Self {
            guards: Vec::new(),
            base_carries,
        }
    }

    /// Append the taken-path guard of a just-materialized fork so later forks in
    /// the same step inherit it. Callers must `record` in dispatch order.
    pub(crate) fn record(&mut self, cond: RustBV, path_taken: bool) {
        self.guards.push((cond, path_taken));
    }

    fn replay_onto(&self, state: &RustSimState) {
        for (cond, taken) in &self.guards {
            let solver = state.solver();
            let solver = solver.borrow();
            if *taken {
                solver.assume_true(cond);
            } else {
                solver.assume_false(cond);
            }
        }
    }
}

/// SAT / UNSAT split produced by [`materialize_deferred_forks`], in fork
/// dispatch order. Neither vector has had `set_root` applied — lineage needs
/// `&mut StateManager`, which the callers own; they register both vectors
/// (UNSAT forks were registered by every legacy copy too, before the SAT check).
pub(crate) struct MaterializedForks {
    pub(crate) sat: Vec<RustSimState>,
    pub(crate) unsat: Vec<RustSimState>,
}

/// Everything [`materialize_deferred_forks`] needs beyond the fork list itself.
pub(crate) struct MaterializeForkCtx<'a> {
    /// The guard-free base every unexplored side forks from. Callers pick this
    /// (pre-callback snapshot, bounce state, …); it must NOT carry the taken
    /// path's branch guard or the unexplored side inherits it.
    pub(crate) fork_base: &'a RustSimState,
    pub(crate) stored_conditions: &'a FxHashMap<u64, RustBV>,
    /// Pre-branch solver snapshots, consumed (`remove`d) per fork by
    /// [`build_unexplored_fork`].
    pub(crate) snapshots: &'a mut FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    /// When true, skip the per-fork SAT check and treat every fork as SAT.
    pub(crate) lazy_solves: bool,
    /// Continuing state that should receive the *taken*-path guard as each fork
    /// is materialized (`assume_true` when `path_taken`, `assume_false`
    /// otherwise). `None` for callers whose continuing state already carries the
    /// guard (`_resume_after_simprocedure`) or that mint both directions
    /// themselves (`_resume_after_symbolic_branch`).
    pub(crate) guard_sink: Option<&'a RustSimState>,
    /// `Some` iff profiling is enabled; receives the deferred-fork / fork-op /
    /// SAT timers and counts.
    pub(crate) stats: Option<&'a mut ExecutionStats>,
}

/// Materialize a step's deferred forks into SAT / UNSAT state lists.
///
/// Single source of truth for the condition-lookup → P11 `condition_ast`
/// reconstruction → P15 conservative-fork → [`build_unexplored_fork`] → SAT
/// check pipeline that used to be open-coded in three near-identical copies:
/// `step_one`'s find/avoid callback arm (`run_loop.rs`), and both
/// `_resume_after_simprocedure` / `_deadend_pending_callback` /
/// `_resume_after_symbolic_branch` loops (`resume.rs`). Only the parallel
/// mirror (`core_outcome_handlers.rs::materialize_deferred_forks_core`) was
/// unit-tested, so the serial copies were free to drift — angr-ph300.7 was
/// exactly that drift (a missing P11 arm silently dropping AST-only forks).
///
/// The parallel mirror is deliberately NOT folded in here: it has no P11
/// fallback, fires the `constraints` inspect BP via
/// [`add_fork_guard_constraint`], and accumulates into atomics rather than
/// `ExecutionStats` (angr-ph300.73 tracks unifying it).
pub(crate) fn materialize_deferred_forks(
    forks: Vec<DeferredFork>,
    ctx: MaterializeForkCtx<'_>,
) -> MaterializedForks {
    let MaterializeForkCtx {
        fork_base,
        stored_conditions,
        snapshots,
        lazy_solves,
        guard_sink,
        mut stats,
    } = ctx;
    let mut out = MaterializedForks {
        sat: Vec::new(),
        unsat: Vec::new(),
    };
    // NOTE: no early return on an empty fork list. The batch timer below is
    // charged unconditionally when profiling is on, matching the legacy
    // `step_one` site — `deferred_fork_time_ns` is the liveness signal the
    // profiling-gate regression test reads, and on binaries whose find/avoid
    // callbacks carry no deferred forks the empty batches are its only source.
    let total = forks.len() as u64;
    let batch_start = stats.is_some().then(std::time::Instant::now);
    // Taken-path guards of the forks already materialized, replayed onto each
    // later fork (`fork_base` is fixed and guard-free here — angr-62ar5).
    let mut prior_guards = PriorGuards::new(false);

    for fork in forks {
        let condition = stored_conditions.get(&fork.condition_id);
        // P11: reconstruct from the stored claripy AST when the condition is
        // absent from `stored_conditions`.
        let reconstructed = reconstruct_deferred_fork_condition(condition, &fork, fork_base);

        if let Some(cond) = condition.or(reconstructed.as_ref()) {
            if let Some(sink) = guard_sink {
                if fork.path_taken {
                    sink.solver().borrow().assume_true(cond);
                } else {
                    sink.solver().borrow().assume_false(cond);
                }
            }
            let fork_start = stats.is_some().then(std::time::Instant::now);
            let forked = build_unexplored_fork(fork_base, &fork, cond, snapshots, &prior_guards);
            prior_guards.record(cond.clone(), fork.path_taken);
            if let (Some(s), Some(start)) = (stats.as_deref_mut(), fork_start) {
                s.solver_fork_time_ns += start.elapsed().as_nanos() as u64;
                s.solver_fork_count += 1;
            }
            if reconstructed.is_some() {
                log::debug!(
                    "P11: Reconstructed condition from condition_ast for fork at 0x{:x}",
                    fork.branch_addr
                );
            }
            let sat_start = stats.is_some().then(std::time::Instant::now);
            let sat = lazy_solves || forked.satisfiable();
            if let (Some(s), Some(start)) = (stats.as_deref_mut(), sat_start) {
                s.solver_sat_time_ns += start.elapsed().as_nanos() as u64;
                s.solver_sat_count += 1;
            }
            if sat {
                out.sat.push(forked);
            } else {
                log::debug!(
                    "P13: Forked state at 0x{:x} is UNSAT, adding to pruned",
                    fork.unexplored_target
                );
                out.unsat.push(forked);
            }
        } else {
            // P15: no condition from either source — build a conservative
            // unconstrained fork so the unexplored branch is still routed
            // rather than dropped.
            log::warn!(
                "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                 Creating conservative fork to explore the path.",
                fork.branch_addr,
                fork.condition_id
            );
            let mut forked = fork_base.fork();
            prior_guards.replay_onto(&forked);
            forked.set_pc(fork.unexplored_target);
            if lazy_solves || forked.satisfiable() {
                out.sat.push(forked);
            } else {
                log::debug!(
                    "P13: Unconstrained fork at 0x{:x} is UNSAT, adding to pruned",
                    fork.unexplored_target
                );
                out.unsat.push(forked);
            }
        }
    }

    if let (Some(s), Some(start)) = (stats, batch_start) {
        s.deferred_fork_time_ns += start.elapsed().as_nanos() as u64;
        s.deferred_fork_count += total;
    }
    out
}

/// The address a state should report to Python, with the angr-4rq7 pc==0
/// fallback applied.
///
/// `self.pc` is stale at 0 for states produced by register-file-replace paths
/// that never round-trip through `set_pc` — notably forked successors under the
/// register-proxy write-through gate, whose proxy is bound to the parent
/// state_id — while the IP register already holds the real branch target. The
/// full-export path (`_snapshot_to_angr`) derives `state.addr` from the IP
/// register, so every accessor Python treats as "the state's address" must
/// agree with it or the find/avoid predicate cache keys on addr 0 and misses
/// the genuine find until a later step refreshes `pc` (angr-ph300.23).
///
/// A nonzero `self.pc` is always authoritative (the gate-off path keeps the two
/// in sync); only the `pc == 0` case falls back, and a genuinely-zero IP
/// register still reports 0.
pub(crate) fn effective_pc(state: &RustSimState) -> u64 {
    let pc = state.pc();
    if pc != 0 {
        pc
    } else {
        state.get_ip().as_u64().unwrap_or(0)
    }
}

/// What the native fast path decided, so the follow-up (register/PC/SP
/// landing, sub-call setup, Python bounce) runs after the borrow of the
/// procedure registry is released.
///
/// Both proc dispatch sites consume this: `step_one`'s serial arm
/// (`run_loop.rs`) and `handle_simprocedure_core`'s parallel arm
/// (`core_outcome_handlers.rs`). The landing itself is deliberately NOT shared
/// — the serial path resolves the return address through the calling
/// convention (`get_return_addr` + `pops_return_addr`, so LR/X30/$ra arches
/// work) while the core path is handed `return_addr` by the interpreter — but
/// the *decision* is, so the two cannot drift (angr-ph300.73).
pub(crate) enum NativeProcDisposition {
    Returned {
        no_return: bool,
        ret_val: Option<RustBV>,
    },
    SubCall {
        proc_name: String,
        saved_args: Vec<RustBV>,
        target: u64,
        sub_args: Vec<RustBV>,
        resume_tag: u32,
    },
    Fallback,
    /// The native proc touched an unmapped page while STRICT_PAGE_ACCESS is on:
    /// Python's `PrivilegedPagingMixin._initialize_page` would raise
    /// `SimSegfaultException`, so the state is terminal-errored natively instead
    /// of bouncing to Python just to raise. Carries the message Python formats
    /// (`"{page_addr:#x} (unmapped)"`). Only produced when the caller passes
    /// `mirror_segfault`.
    Segfault(String),
}

/// Map a native procedure error onto the `SimSegfaultException` Python would
/// raise for the same access, or `None` when Python would service it (and we
/// must therefore fall back).
///
/// Only the unmapped-page case is mirrored: with STRICT_PAGE_ACCESS off Python
/// lazily initializes the page and keeps going, so the state must still bounce.
pub(crate) fn segfault_message(state: &RustSimState, err: &ProcedureError) -> Option<String> {
    if !state.enforce_permissions() {
        return None;
    }
    match err {
        ProcedureError::Memory(crate::memory::MemoryError::Unmapped { addr, .. }) => {
            let page_addr = addr & !(crate::memory::PAGE_SIZE - 1);
            Some(format!("{page_addr:#x} (unmapped)"))
        }
        _ => None,
    }
}

/// Mutable views onto whichever counter home the caller owns — `CoreCounters`
/// for the parallel path, `profiling.native_proc_stats` for the serial one.
/// The two structs carry the same six fields under two different names
/// (`native_python_fallbacks` vs `python_fallbacks`), so a bundle of `&mut`
/// borrows is the cheapest way to let one dispatcher bump either.
pub(crate) struct NativeProcCounters<'a> {
    pub(crate) native_calls: &'a mut u64,
    pub(crate) python_fallbacks: &'a mut u64,
    pub(crate) call_counts: &'a mut HashMap<String, u64>,
    pub(crate) symbolic_fallbacks_by_name: &'a mut HashMap<String, u64>,
    pub(crate) not_implemented_fallbacks_by_name: &'a mut HashMap<String, u64>,
    pub(crate) other_fallbacks_by_name: &'a mut HashMap<String, u64>,
}

impl NativeProcCounters<'_> {
    fn fallback(&mut self, name: &str, bucket: FallbackBucket) {
        *self.python_fallbacks += 1;
        let map = match bucket {
            FallbackBucket::Symbolic => &mut *self.symbolic_fallbacks_by_name,
            FallbackBucket::NotImplemented => &mut *self.not_implemented_fallbacks_by_name,
            FallbackBucket::Other => &mut *self.other_fallbacks_by_name,
        };
        *map.entry(name.to_string()).or_insert(0) += 1;
    }

    fn native_call(&mut self, name: &str) {
        *self.native_calls += 1;
        *self.call_counts.entry(name.to_string()).or_insert(0) += 1;
    }
}

enum FallbackBucket {
    Symbolic,
    NotImplemented,
    Other,
}

/// Run one native SimProcedure and classify the result, bumping the per-proc
/// counters the same way on both dispatch paths.
///
/// Callers own the parts that genuinely differ:
/// * argument extraction (`args`) — the serial path reads the manager's
///   calling convention, the core path a `CcSnapshot`;
/// * `no_return` — the serial path trusts the Python registration tuple, the
///   core path `native_proc.no_return()`;
/// * the [`NativeProcDisposition::SubCall`] counter bump, which is deliberately
///   NOT done here: the core path counts the call as soon as `call_ex` returns,
///   while `step_one` counts it only once `setup_native_subcall` succeeds and
///   otherwise books a Python fallback. Both are preserved by leaving the bump
///   to the caller;
/// * `mirror_segfault` — only the core path terminal-errors natively today;
///   `step_one` passes `false` so a strict-page-access fault still bounces to
///   Python via the ordinary fallback counters.
pub(crate) fn dispatch_native_proc(
    native_proc: &dyn crate::procedures::NativeSimProcedure,
    state: &mut RustSimState,
    name: &str,
    no_return: bool,
    args: Result<Vec<RustBV>, crate::arch::ExtractionError>,
    mirror_segfault: bool,
    counters: &mut NativeProcCounters<'_>,
) -> NativeProcDisposition {
    let args = match args {
        Err(e) => {
            log::debug!("Skipping native procedure {name} (arg extraction failed: {e:?})");
            counters.fallback(name, FallbackBucket::Other);
            return NativeProcDisposition::Fallback;
        }
        Ok(args) => args,
    };

    match native_proc.call_ex(state, &args) {
        Ok(ProcOutcome::Return(ret_val)) => {
            counters.native_call(name);
            NativeProcDisposition::Returned { no_return, ret_val }
        }
        Ok(ProcOutcome::CallAndResume {
            target,
            args: sub_args,
            resume_tag,
        }) => NativeProcDisposition::SubCall {
            proc_name: name.to_string(),
            saved_args: args,
            target,
            sub_args,
            resume_tag,
        },
        Err(e) => {
            if mirror_segfault && let Some(msg) = segfault_message(state, &e) {
                log::debug!("Native procedure {name} segfaulted: {msg}");
                counters.native_call(name);
                return NativeProcDisposition::Segfault(msg);
            }
            log::debug!("Native procedure {name} returned error, falling back to Python: {e:?}");
            counters.fallback(
                name,
                match e {
                    ProcedureError::SymbolicArgument(_) => FallbackBucket::Symbolic,
                    ProcedureError::NotImplemented => FallbackBucket::NotImplemented,
                    _ => FallbackBucket::Other,
                },
            );
            NativeProcDisposition::Fallback
        }
    }
}

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
