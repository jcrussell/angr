//! Bookkeeping helpers hung off `RustExplorationManager` (extension-impl
//! pattern, mirroring `stepping.rs` / `run_loop.rs`): scheduler-accounting
//! folds, stash pushes, and the veritesting waiter-merge grouping.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. The one
//! surviving `expect` in [`RustExplorationManager::merge_waiters_by_callstack`]
//! is a same-function map invariant, not an input check — see its `#[allow]`
//! reason.
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
    /// successor and deferred-fork loops in run_loop_single.rs and resume.rs would
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

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    pub(crate) fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        self.sm.push_or_drop_terminal(stash_name, state);
    }

    /// Extract procedure arguments from state registers (and stack, when
    /// `num_args` exceeds the register portion of the calling convention).
    ///
    /// Returns an [`ExtractionError`] instead of silently zero-padding when
    /// the stack pointer is symbolic or a stack slot cannot be read. Callers
    /// (the native-procedure dispatchers in `stepping.rs` / `run_loop_single.rs`)
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
}

/// Pack a concrete byte slice into 16-byte `RustBV::concrete` chunks and store
/// each via the caller-supplied sink.
///
/// angr-5aj8: `RustBV::Concrete` is backed by a `u128`, so packing more than
/// 16 bytes into a single value shift-overflows for byte indices >= 16 and the
/// downstream `store_concrete` page-fill loop emits a 16-byte-cycle pattern
/// across the entire `data.len()` range, corrupting memory wholesale. This is
/// the canonical safe loop (it was hand-copied into every concrete-store entry
/// point before angr-5aj8); the sink closure abstracts the only divergence
/// between call sites (which memory API to write through, and how to map its
/// error into a `PyErr`).
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
/// first — see `load_concrete_bytes_chunked`). This is the read-side
/// counterpart of the pack loop in `store_concrete_bytes_chunked`, hand-copied
/// across the eval / memory-get paths in `state_api`/`pending_api` before
/// consolidation.
pub(crate) fn u128_to_le_bytes(val: u128, size: usize) -> Vec<u8> {
    (0..size).map(|i| (val >> (i * 8)) as u8).collect()
}

/// Read `size` bytes at `addr` in `<= 16`-byte chunks via the caller-supplied
/// source, concatenating the results.
///
/// Read-side mirror of `store_concrete_bytes_chunked`, and the reason both
/// exist is the same: `RustBV::Concrete` is `u128`-backed, so `as_u128()` /
/// `u128_to_le_bytes` only round-trip 16 bytes. A single wider load either
/// truncates the tail or wraps mod 128 into a repeating 16-byte pattern
/// (angr-ph300.19). The source closure abstracts the only divergence between
/// call sites: which memory API to read through and how to map its failure
/// into a `PyErr`. It is called once per chunk with `(chunk_addr, chunk_size)`
/// and must return exactly `chunk_size` bytes; the first error propagates and
/// the partial prefix is dropped.
///
/// Callers do their state / pending lookup **once** around this call and read
/// inside that single borrow. The hand-copies this replaced
/// (`_get_state_memory`, `_get_pending_memory`, `_pending_memory_load`) each
/// recursed into themselves per chunk, redoing the `with_state` / `with_pending`
/// lookup — and, for the pending pair, re-borrowing the solver — every 16 bytes
/// (angr-9ke6b.82).
///
/// `size == 0` yields an empty vector without invoking the source at all,
/// matching `store_concrete_bytes_chunked`'s empty-slice behavior.
pub(crate) fn load_concrete_bytes_chunked<F>(addr: u64, size: u32, mut load: F) -> PyResult<Vec<u8>>
where
    F: FnMut(u64, u32) -> PyResult<Vec<u8>>,
{
    let mut out = Vec::with_capacity(size as usize);
    let mut offset = 0u32;
    while offset < size {
        let chunk_size = (size - offset).min(16);
        out.extend_from_slice(&load(addr + offset as u64, chunk_size)?);
        offset += chunk_size;
    }
    Ok(out)
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

#[cfg(test)]
#[path = "helpers_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
