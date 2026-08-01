//! Native implementations of VEX dirty helpers.
//!
//! This module provides Rust implementations of common dirty helper functions
//! used by VEX IR. When a dirty call can be handled natively, we avoid the
//! Python callback overhead.
//!
//! Supported helpers (only those actually reachable from the interpreter):
//! - RDTSC (timestamp counter; no args, returns ULong)
//! - IN/OUT port access (concrete port arg, safe defaults)
//!
//! NOTE: CPUID and RDTSCP are deliberately NOT handled here. VEX passes them
//! a GSPTR (guest-state pointer) argument, and the interpreter aborts dirty
//! statements with `NeedPythonFallback` while evaluating GSPTR/VECRET args
//! (see `interpreter/expressions.rs` + `statements.rs`) *before* native
//! dispatch runs. Any handler registered for them would be dead code, so the
//! Python engine handles CPUID/RDTSCP. (angr-2iow measured zero GSPTR hits
//! corpus-wide; angr-t1ok removed the dead CPUID/RDTSCP handlers.)

use std::collections::HashMap;

/// Result of a native dirty helper call.
#[derive(Debug, Clone)]
pub struct DirtyHelperResult {
    /// Return value (if any).
    pub return_value: Option<u64>,
    /// Register writes: (offset, value).
    pub reg_writes: Vec<(u32, u64)>,
}

impl DirtyHelperResult {
    /// Create a result with just a return value.
    pub fn with_return(value: u64) -> Self {
        DirtyHelperResult {
            return_value: Some(value),
            reg_writes: Vec::new(),
        }
    }

    /// Create an empty result (no return value).
    pub fn empty() -> Self {
        DirtyHelperResult {
            return_value: None,
            reg_writes: Vec::new(),
        }
    }
}

/// Dirty helper dispatch table.
pub struct DirtyHelperDispatch {
    handlers: HashMap<&'static str, DirtyHandlerFn>,
}

type DirtyHandlerFn = fn(&mut DirtyHelperState, &[u64]) -> Option<DirtyHelperResult>;

/// Initial value of a state's simulated timestamp counter.
pub const TSC_INITIAL: u64 = 0x1000000000;

/// Amount the simulated timestamp counter advances per `RDTSC`.
pub const TSC_STEP: u64 = 1000;

/// Mutable per-state scratch space threaded into the dirty helpers.
///
/// Helpers that model stateful hardware (currently only `RDTSC`) read and
/// advance this instead of a process-wide static, so two states executing
/// independently — including concurrently under the parallel scheduler — see
/// values that depend only on their own execution history. That makes RDTSC
/// reproducible across runs and across worker counts, which matters for the
/// timing / anti-analysis checks that actually read the TSC.
///
/// Lives on `VEXInterpreter` for the duration of a step and is transferred
/// to/from `RustSimState::tsc_counter` by `run_interpreter_step_core` and
/// `apply_interpreter_step_result`, so it forks and merges with the state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtyHelperState {
    /// Next value `RDTSC` will return.
    pub tsc: u64,
}

impl Default for DirtyHelperState {
    fn default() -> Self {
        DirtyHelperState { tsc: TSC_INITIAL }
    }
}

impl Default for DirtyHelperDispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl DirtyHelperDispatch {
    /// Create a new dispatch table with all supported helpers.
    pub fn new() -> Self {
        let mut handlers: HashMap<&'static str, DirtyHandlerFn> = HashMap::new();

        // RDTSC helper (no args, returns the timestamp counter)
        handlers.insert("amd64g_dirtyhelper_RDTSC", handle_rdtsc);
        handlers.insert("x86g_dirtyhelper_RDTSC", handle_rdtsc);

        // IN/OUT port helpers (return safe defaults)
        handlers.insert("amd64g_dirtyhelper_IN", handle_in_port);
        handlers.insert("x86g_dirtyhelper_IN", handle_in_port);
        handlers.insert("amd64g_dirtyhelper_OUT", handle_out_port);
        handlers.insert("x86g_dirtyhelper_OUT", handle_out_port);

        DirtyHelperDispatch { handlers }
    }

    /// Try to handle a dirty call natively.
    ///
    /// # Arguments
    /// * `helper_state` - Per-state scratch space (e.g. the simulated TSC)
    /// * `name` - The helper function name
    /// * `args` - Concrete argument values
    ///
    /// # Returns
    /// Some(result) if handled natively, None if Python callback needed.
    pub fn try_call(
        &self,
        helper_state: &mut DirtyHelperState,
        name: &str,
        args: &[u64],
    ) -> Option<DirtyHelperResult> {
        if let Some(handler) = self.handlers.get(name) {
            handler(helper_state, args)
        } else {
            None
        }
    }
}

// ============================================================================
// RDTSC Handlers
// ============================================================================

/// Simulated TSC value: return the state's current counter and advance it.
///
/// The counter is per-state (see [`DirtyHelperState`]), not process-wide, so
/// the Nth `RDTSC` along a given execution path always yields the same value
/// regardless of what other states did first. Saturates rather than wrapping
/// so a pathological RDTSC loop cannot make time appear to run backwards.
fn handle_rdtsc(state: &mut DirtyHelperState, _args: &[u64]) -> Option<DirtyHelperResult> {
    let tsc = state.tsc;
    state.tsc = state.tsc.saturating_add(TSC_STEP);
    Some(DirtyHelperResult::with_return(tsc))
}

// ============================================================================
// I/O Port Handlers
// ============================================================================

fn handle_in_port(_state: &mut DirtyHelperState, _args: &[u64]) -> Option<DirtyHelperResult> {
    // Return 0xFF for all IN port reads (safe default)
    Some(DirtyHelperResult::with_return(0xFF))
}

fn handle_out_port(_state: &mut DirtyHelperState, _args: &[u64]) -> Option<DirtyHelperResult> {
    // OUT has no return value - just ignore it
    Some(DirtyHelperResult::empty())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "dirty_tests.rs"]
mod tests;
