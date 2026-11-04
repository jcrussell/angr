//! Native implementations of VEX dirty helpers.
//!
//! This module provides Rust implementations of common dirty helper functions
//! used by VEX IR. When a dirty call can be handled natively, we avoid the
//! Python callback overhead.
//!
//! Supported helpers (only those actually reachable from the interpreter):
//! - OUT port access (concrete port arg, no effect)
//!
//! NOTE: RDTSC and IN are deliberately NOT handled here. The Python engine
//! returns a fresh symbol for each (`RDTSC`, keyed `("hardware", "rdtsc")`;
//! `IN`, keyed `("hardware", "in")`), so a branch on a timestamp delta or a
//! port value keeps both sides. A concrete stand-in would silently drop one,
//! so both go through the Python `dirty_call` callback (angr-vmi2l.69).
//!
//! NOTE: CPUID and RDTSCP are deliberately NOT handled here. VEX passes them
//! a GSPTR (guest-state pointer) argument, and the interpreter aborts dirty
//! statements with `NeedPythonFallback` while evaluating GSPTR/VECRET args
//! (see `interpreter/expressions.rs` + `statements.rs`) *before* native
//! dispatch runs. Any handler registered for them would be dead code, so the
//! Python engine handles CPUID/RDTSCP and no native handler exists for them
//! (angr-2iow, angr-t1ok).

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

        // OUT port helpers (no return value, no modelled effect)
        handlers.insert("amd64g_dirtyhelper_OUT", handle_out_port);
        handlers.insert("x86g_dirtyhelper_OUT", handle_out_port);

        DirtyHelperDispatch { handlers }
    }

    /// Whether a native handler exists for `name`.
    pub fn handles(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
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
        self.handlers.get(name).and_then(|handler| handler(args))
    }
}

// ============================================================================
// I/O Port Handlers
// ============================================================================

fn handle_out_port(_args: &[u64]) -> Option<DirtyHelperResult> {
    // OUT has no return value - just ignore it
    Some(DirtyHelperResult::empty())
}

// ============================================================================
// Tests
// ============================================================================

test_submod!("dirty_tests.rs" => tests);
