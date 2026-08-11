//! `#[pymethods]` for [`RustExplorationManager`]: native uniqueness filter and native exploration techniques.
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
    /// Enable native uniqueness filter with given register names.
    ///
    /// After each step in run(), states with duplicate register tuples
    /// are moved to 'not_unique' stash. This replaces the Python
    /// CheckUniqueness technique with zero FFI overhead.
    #[angr_macros::steady_guarded]
    pub fn register_uniqueness_filter(&mut self, register_names: Vec<String>) {
        self.constraint_tracker.uniqueness_registers = register_names;
        self.constraint_tracker.uniqueness_set.clear();
        // Ensure not_unique stash exists
        self.sm.declare_stash("not_unique");
    }

    /// Disable the native uniqueness filter.
    #[angr_macros::steady_guarded]
    pub fn disable_uniqueness_filter(&mut self) {
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
    #[angr_macros::steady_guarded]
    pub fn register_length_limiter(&mut self, max_length: usize, drop: bool) {
        self.native_techniques
            .push(NativeTechnique::LengthLimiter { max_length, drop });
        if !drop {
            self.sm.declare_stash("cut");
        }
    }

    /// Register a native Timeout technique.
    ///
    /// Exploration stops after `timeout_secs` seconds. All active states are
    /// moved to "timeout" stash. Timer starts on first call to apply_native_techniques().
    #[angr_macros::steady_guarded]
    pub fn register_timeout(&mut self, timeout_secs: f64) {
        self.native_techniques.push(NativeTechnique::Timeout {
            timeout_secs,
            start_time: None,
        });
        self.sm.declare_stash(TIMEOUT_STASH);
    }

    /// Register a native LoopBound technique.
    ///
    /// States where any single address appears more than `bound` times in
    /// their history are moved to `discard_stash`. This is a simplified
    /// version of LoopSeer that doesn't require CFG analysis.
    #[pyo3(signature = (bound, discard_stash="spinning"))]
    #[angr_macros::steady_guarded]
    pub fn register_loop_bound(&mut self, bound: usize, discard_stash: &str) {
        self.native_techniques.push(NativeTechnique::LoopBound {
            bound,
            discard_stash: discard_stash.to_string(),
        });
        self.sm.declare_stash(discard_stash);
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
    #[angr_macros::steady_guarded]
    pub fn register_merge_point(&mut self, address: u64, wait_counter: usize) {
        let wait_stash = format!("merge_waiting_{address:#x}");
        self.sm.declare_stash(&wait_stash);
        self.native_techniques.push(NativeTechnique::MergePoint {
            address,
            wait_counter_limit: wait_counter,
            counter: 0,
            wait_stash,
        });
    }

    /// Get the number of registered native techniques.
    pub fn native_technique_count(&self) -> usize {
        self.native_techniques.len()
    }

    /// Clear all native techniques.
    pub fn clear_native_techniques(&mut self) {
        self.native_techniques.clear();
    }
}

test_submod!("manager_methods_techniques_tests.rs" => tests);
