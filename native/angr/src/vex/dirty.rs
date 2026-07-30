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

type DirtyHandlerFn = fn(&[u64]) -> Option<DirtyHelperResult>;

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
    /// * `name` - The helper function name
    /// * `args` - Concrete argument values
    ///
    /// # Returns
    /// Some(result) if handled natively, None if Python callback needed.
    pub fn try_call(&self, name: &str, args: &[u64]) -> Option<DirtyHelperResult> {
        if let Some(handler) = self.handlers.get(name) {
            handler(args)
        } else {
            None
        }
    }
}

// ============================================================================
// RDTSC Handlers
// ============================================================================

/// Simulated TSC value.
/// We use a static counter that increments on each call.
use std::sync::atomic::{AtomicU64, Ordering};

static TSC_COUNTER: AtomicU64 = AtomicU64::new(0x1000000000);

fn handle_rdtsc(_args: &[u64]) -> Option<DirtyHelperResult> {
    // Return an incrementing timestamp value
    let tsc = TSC_COUNTER.fetch_add(1000, Ordering::Relaxed);
    Some(DirtyHelperResult::with_return(tsc))
}

// ============================================================================
// I/O Port Handlers
// ============================================================================

fn handle_in_port(_args: &[u64]) -> Option<DirtyHelperResult> {
    // Return 0xFF for all IN port reads (safe default)
    Some(DirtyHelperResult::with_return(0xFF))
}

fn handle_out_port(_args: &[u64]) -> Option<DirtyHelperResult> {
    // OUT has no return value - just ignore it
    Some(DirtyHelperResult::empty())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "dirty_tests.rs"]
mod tests;
