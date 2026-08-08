//! `#[pymethods]` for [`RustExplorationManager`]: hook, SimProcedure and binary-region registration.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
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
    /// Add a hook address.
    ///
    /// Guarded like `register_simprocedures`: a live steady session snapshots
    /// `hooks` into each worker's `StepContext`, so mutating the set mid-run
    /// must finalize first or the new hook silently never fires.
    #[angr_macros::steady_guarded]
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Add multiple hook addresses.
    #[angr_macros::steady_guarded]
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        for addr in addrs {
            self.hooks.insert(addr);
        }
    }

    /// Clear all hooks, including every registered SimProcedure.
    ///
    /// Guarded like its sibling mutators (`add_hook`/`add_hooks`) — `hooks` is
    /// snapshotted into each worker's `StepContext`, so clearing it mid-run
    /// must finalize first or a worker keeps dispatching to an already-cleared
    /// hook.
    ///
    /// Clears `simprocedures` too, upholding the same both-maps invariant
    /// `unregister_simprocedures` does (angr-sqfj8.30): the run loop reads
    /// `simprocedures.get(&pc)` *inside* the `hooks.contains(&pc)` gate, so a
    /// stale entry surviving a clear would resurrect as a SimProcedure call
    /// the moment the same address was re-hooked via a plain `add_hook`.
    #[angr_macros::steady_guarded]
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
        self.simprocedures.clear();
    }

    /// Register a SimProcedure.
    #[pyo3(signature = (addr, name, num_args=0, no_return=false))]
    #[angr_macros::steady_guarded]
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
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
    #[angr_macros::steady_guarded]
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
    }

    /// Unregister multiple SimProcedures (e.g. after `proj.unhook(addr)` on a
    /// live manager). Removes each address from both the hook set and the
    /// SimProcedure table so a stale hook no longer fires (angr-969g).
    #[angr_macros::steady_guarded]
    pub fn unregister_simprocedures(&mut self, addrs: Vec<u64>) {
        for addr in addrs {
            self.hooks.remove(&addr);
            self.simprocedures.remove(&addr);
        }
    }

    /// Load binary code regions.
    ///
    /// `binary_regions` is snapshotted into each worker's `StepContext`
    /// (`step_core.rs`), so mutating it mid-run must finalize a live steady
    /// session first or the change silently never reaches a resident worker.
    #[angr_macros::steady_guarded]
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) {
        self.environment.binary_regions = regions
            .into_iter()
            .map(|(base, data)| (base, Arc::new(data)))
            .collect();
    }

    /// Record the main object's `[start, end)` code span. Hooks inside it always
    /// dispatch to Python (user `proj.hook()` overrides); see
    /// `execution_env::prefer_native_dispatch`.
    ///
    /// `main_object_range` is snapshotted into each worker's `StepContext`
    /// like `binary_regions` above.
    #[angr_macros::steady_guarded]
    pub fn set_main_object_range(&mut self, start: u64, end: u64) {
        self.environment.main_object_range = Some((start, end));
    }

    /// Opt-in: prefer native procedures for `use_sim_procedures` library hooks
    /// that land inside a non-main loaded object (angr-a8epx / angr-gorvf.3.2).
    ///
    /// `prefer_native_library_hooks` is snapshotted into each worker's
    /// `StepContext` like `binary_regions`/`main_object_range` above.
    #[angr_macros::steady_guarded]
    pub fn set_prefer_native_library_hooks(&mut self, enabled: bool) {
        self.environment.prefer_native_library_hooks = enabled;
    }
}

test_submod!("manager_methods_hooks_tests.rs" => tests);
