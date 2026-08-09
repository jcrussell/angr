//! `NativeTechnique` — exploration techniques that run entirely in Rust.
//!
//! Split out of `exploration::mod` (angr-zel8z.3). These variants avoid Python
//! callback overhead for common technique patterns; they are applied in the
//! Rust exploration loop — the application logic lives in this file's
//! `impl RustExplorationManager` block (moved here from `exploration::helpers`
//! by angr-9ke6b.76).
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. The one
//! surviving `expect` in [`RustExplorationManager::merge_waiters_by_callstack`]
//! is a same-function map invariant, not an input check — see its `#[allow]`
//! reason.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// Stash the [`NativeTechnique::Timeout`] arm parks the active frontier in.
/// Declared up front by `RustExplorationManager::register_timeout` so the
/// stash exists (empty) even when the deadline never fires.
pub(crate) const TIMEOUT_STASH: &str = "timeout";

/// Native exploration technique variants.
///
/// These techniques run entirely in Rust during the exploration loop,
/// avoiding Python callback overhead for common technique patterns.
#[derive(Debug, Clone)]
pub(crate) enum NativeTechnique {
    /// Limits path length by block count. States exceeding `max_length` blocks
    /// are moved to the "cut" stash, or — when `drop` is true — discarded
    /// in place (there is no stash for them).
    LengthLimiter { max_length: usize, drop: bool },
    /// Wall-clock timeout. Exploration stops after `timeout_secs` seconds.
    Timeout {
        timeout_secs: f64,
        start_time: Option<std::time::Instant>,
    },
    /// Basic loop bounding: limits how many times a single address can appear
    /// in a state's history. States exceeding the bound are moved to
    /// `discard_stash` — but only when `StashManager::drop_terminal_states()`
    /// is false; with it set, they are discarded in place and `discard_stash`
    /// stays empty (same conditionality `LengthLimiter` spells out via its own
    /// `drop` field).
    LoopBound { bound: usize, discard_stash: String },
    /// Native `ManualMergepoint` parity (angr-op0dn.11.5). States whose pc
    /// reaches `address` are parked in `wait_stash`; once the active stash
    /// drains (or `wait_counter_limit` post-step rounds elapse since the last
    /// arrival), the waiters are grouped by callstack and every group with ≥2
    /// members is merged in-Rust via `_merge_states`. `counter` is the live
    /// post-step round counter — reset to 0 whenever a fresh waiter arrives.
    MergePoint {
        address: u64,
        wait_counter_limit: usize,
        counter: usize,
        wait_stash: String,
    },
}

impl RustExplorationManager {
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
                        let states: Vec<_> = self
                            .sm
                            .get_mut(STASH_ACTIVE)
                            .map(|active| active.drain(..).collect::<Vec<_>>())
                            .unwrap_or_default();
                        // Draining STASH_ACTIVE bypasses `policy.select`;
                        // notify so a memoizing policy doesn't leak its
                        // per-state memo entries (angr-myzjx.25).
                        for s in &states {
                            self.policy.on_state_removed(s.state_id());
                        }
                        // Re-home through `index`, not a raw `stashes_mut()`
                        // push: a raw move leaves `state_index` pointing at
                        // STASH_ACTIVE until the next `sync_state_index`, and
                        // within one `run()` that stale entry is what
                        // `find_state_mut` / `take_state` resolve against
                        // (angr-c7xno.97).
                        for s in states {
                            let sid = s.state_id();
                            self.sm.ensure_stash(TIMEOUT_STASH).push_back(s);
                            self.sm.index(sid, TIMEOUT_STASH);
                        }
                        complete = true;
                    }
                }
                NativeTechnique::LengthLimiter { max_length, drop } => {
                    let max_len = *max_length;
                    // `drop` states have no stash of their own; the rest are cut.
                    let dest = if *drop { None } else { Some("cut") };
                    self.evict_active_states(|state| state.history().len() > max_len, dest);
                }
                NativeTechnique::LoopBound {
                    bound,
                    discard_stash,
                } => {
                    let max_bound = *bound;
                    let stash_name = discard_stash.clone();
                    // Unlike `LengthLimiter`'s per-technique `drop` flag, the
                    // discard decision here is the manager-wide one.
                    let dest = (!self.sm.drop_terminal_states()).then_some(stash_name);
                    self.evict_active_states(
                        |state| Self::exceeds_loop_bound(state.history(), max_bound),
                        dest.as_deref(),
                    );
                }
                // Handled out-of-band above the match.
                NativeTechnique::MergePoint { .. } => {}
            }
        }

        complete
    }

    /// Evict every state in the active stash matching `predicate`, then either
    /// park the evictees in `dest` or discard them (`dest == None`).
    ///
    /// Shared body of the `LengthLimiter` and `LoopBound` arms of
    /// [`RustExplorationManager::apply_native_techniques`]: both scan
    /// `STASH_ACTIVE` for states past a limit and differ only in the predicate
    /// and in how the destination stash is chosen. States leave `STASH_ACTIVE`
    /// here outside `policy.select`, so the policy is notified explicitly — a
    /// memoizing policy would otherwise leak its per-state memo entries
    /// (angr-myzjx.25).
    fn evict_active_states(
        &mut self,
        predicate: impl Fn(&RustSimState) -> bool,
        dest: Option<&str>,
    ) {
        let to_remove: Vec<usize> = {
            let Some(active) = self.sm.get(STASH_ACTIVE) else {
                return;
            };
            active
                .iter()
                .enumerate()
                .filter(|(_, state)| predicate(state))
                .map(|(i, _)| i)
                .collect()
        };
        if to_remove.is_empty() {
            return;
        }

        let removed_states: Vec<RustSimState> = {
            let Some(active) = self.sm.get_mut(STASH_ACTIVE) else {
                return;
            };
            // Descending index order so each removal leaves the earlier
            // indices valid.
            to_remove
                .iter()
                .rev()
                .filter_map(|&idx| active.remove(idx))
                .collect()
        };
        for state in &removed_states {
            self.policy.on_state_removed(state.state_id());
        }

        // Both arms keep `state_index` (and, on the discard arm, `state_roots`)
        // moving with the state. A raw `stashes_mut()` push left the evictees
        // indexed under STASH_ACTIVE until Python's post-`run()`
        // `sync_state_index`, and mid-run that stale entry is what
        // `find_state_mut` / `take_state` resolve against (angr-c7xno.97).
        if let Some(dest) = dest {
            for state in removed_states {
                let sid = state.state_id();
                self.sm.ensure_stash(dest).push_back(state);
                self.sm.index(sid, dest);
            }
        } else {
            // Discarded in place: module invariant I6 — a state that leaves
            // the stashes takes both of its map entries with it.
            for state in &removed_states {
                let sid = state.state_id();
                self.sm.unindex(sid);
                self.sm.remove_root(sid);
            }
        }
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
    #[allow(
        clippy::expect_used,
        reason = "`order` is built in the loop directly above by pushing exactly the keys that are also inserted into `by_key`, and each key is pushed only on first sight, so the drain removes each key exactly once and never misses"
    )]
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
                    // Merge failed: do NOT strand the waiters in `wait_stash`
                    // (nothing else ever un-parks them — they would leak past
                    // exploration end). Release them back to active so they
                    // continue to be scheduled independently, unmerged. This
                    // mirrors the lone-callstack release above; `_merge_states`
                    // forks rather than consumes, so the originals are intact.
                    log::warn!(
                        "MergePoint merge failed, releasing waiters to active unmerged: {e}"
                    );
                    for &sid in &ids {
                        self.move_state_by_id(wait_stash, STASH_ACTIVE, sid);
                    }
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
}

test_submod!("native_technique_tests.rs" => tests);
