//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::{Mutex, RwLock};

use super::RustBV;

/// Error type for constraint sync operations.
#[derive(Debug, Clone)]
pub enum ConstraintSyncError {
    /// Conversion failed for a constraint.
    ConversionFailed(String),
    /// Constraints became unsatisfiable after sync.
    Unsatisfiable,
    /// Invalid rollback (no transaction to rollback).
    NoTransaction,
}

impl std::fmt::Display for ConstraintSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintSyncError::ConversionFailed(msg) => {
                write!(f, "constraint conversion failed: {}", msg)
            }
            ConstraintSyncError::Unsatisfiable => {
                write!(f, "constraints became unsatisfiable after sync")
            }
            ConstraintSyncError::NoTransaction => {
                write!(f, "no transaction to rollback")
            }
        }
    }
}

impl std::error::Error for ConstraintSyncError {}

/// Solver context for symbolic execution.
///
/// Manages symbolic variable creation and, when Z3 is available,
/// constraint solving and satisfiability checking.
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// to store a reference to it. All Z3 operations on a thread share
/// the same context automatically.
///
/// ## Transactional Constraint Sync
///
/// The context supports transactional constraint sync with push/pop semantics:
/// - `transaction_begin()`: Start a new transaction
/// - `transaction_commit()`: Commit constraints (validate and keep)
/// - `transaction_rollback()`: Rollback on failure
pub struct SymContext {
    /// Counter for generating unique symbol IDs.
    next_id: AtomicU64,
    /// Number of constraints added (for tracking).
    constraint_count: AtomicUsize,
    /// Named symbolic variables for debugging.
    symbol_table: RwLock<HashMap<String, u64>>,
    /// Current push level for transaction tracking.
    push_level: AtomicUsize,
    /// Constraint count at each push level (for rollback).
    push_constraint_counts: Mutex<Vec<usize>>,
    /// Phase 2 Fix: Track assumed RustBV constraints for export to Python.
    /// Each entry is (constraint, is_assumed_true) - constraint == 1 if true, == 0 if false.
    assumed_constraints: Mutex<Vec<(RustBV, bool)>>,

    // Z3-specific fields (when feature is enabled)
    #[cfg(feature = "vex-engine-z3")]
    solver: Mutex<z3::Solver>,
    /// Cached SAT result, invalidated on constraint addition.
    #[cfg(feature = "vex-engine-z3")]
    sat_cache: Cell<Option<bool>>,
    /// Cached Z3 model, invalidated on constraint addition.
    #[cfg(feature = "vex-engine-z3")]
    model_cache: RefCell<Option<z3::Model>>,
    /// List of Z3 tracking boolean constants for unsat core mapping.
    /// Each entry is a (track_bool, constraint_ast) pair.
    #[cfg(feature = "vex-engine-z3")]
    constraint_trackers: Mutex<Vec<z3::ast::Bool>>,
}

impl SymContext {
    /// Create a new mock solver context (without Z3).
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new_mock() -> Self {
        SymContext {
            next_id: AtomicU64::new(0),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(HashMap::new()),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(Vec::new()),
            assumed_constraints: Mutex::new(Vec::new()),
        }
    }

    /// Alias for new_mock when Z3 is not available.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new() -> Self {
        Self::new_mock()
    }

    /// Create a new solver context with Z3.
    ///
    /// With z3-rs 0.19+, the Z3 context is thread-local.
    /// All Z3 operations on this thread will use the same context.
    #[cfg(feature = "vex-engine-z3")]
    pub fn new() -> Self {
        // Create solver with unsat_core support enabled and timeout
        let solver = z3::Solver::new();
        let mut params = z3::Params::new();
        params.set_bool("unsat_core", true);
        // Phase 2 Fix: Add 30 second timeout to prevent indefinite hangs
        params.set_u32("timeout", 30000);
        solver.set_params(&params);

        SymContext {
            next_id: AtomicU64::new(0),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(Vec::new()),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(HashMap::new()),
            assumed_constraints: Mutex::new(Vec::new()),
            solver: Mutex::new(solver),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
        }
    }

    /// Create a mock context for testing (when Z3 is enabled but not needed).
    #[cfg(feature = "vex-engine-z3")]
    pub fn new_mock() -> Self {
        Self::new()
    }

    /// Get the next unique ID for a symbolic variable.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraint_count.load(Ordering::SeqCst)
    }

    // =========================================================================
    // Symbol Management
    // =========================================================================

    /// Create a new symbolic bitvector with a unique name.
    pub fn new_bv(&self, name: &str, width: u32) -> RustBV {
        let unique_name = self.unique_name(name);
        RustBV::symbolic(self, &unique_name, width)
    }

    /// Create a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        let id = self.next_id();
        format!("{}_{}", base, id)
    }

    // =========================================================================
    // Constraint Management (Z3-backed)
    // =========================================================================

    /// Add a constraint.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint(&self, constraint: z3::ast::Bool) {
        // Create a tracking boolean for unsat core extraction
        let idx = self.constraint_count.load(Ordering::SeqCst);
        let track_name = format!("__track_{}", idx);
        let track_bool = z3::ast::Bool::new_const(track_name.as_str());

        // Track the boolean for unsat core mapping
        {
            let mut trackers = self.constraint_trackers.lock();
            trackers.push(track_bool.clone());
        }

        // Use assert_and_track to enable unsat core extraction
        self.solver.lock().assert_and_track(&constraint, &track_bool);
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        // Invalidate caches - constraint set has changed
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    /// Add a constraint that the bitvector equals a specific value.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_bv_constraint(&self, bv: &RustBV, value: u128) {
        use z3::ast::Ast;
        let ast = bv.to_z3_ast();
        let val_ast = if bv.width() <= 64 {
            z3::ast::BV::from_u64(value as u64, bv.width())
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, bv.width() - 64);
            hi.concat(&lo)
        };
        let constraint = ast._eq(&val_ast);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_true(&self, cond: &RustBV) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python
        self.assumed_constraints.lock().push((cond.clone(), true));
        let ast = cond.to_z3_ast();
        let one = z3::ast::BV::from_u64(1, 1);
        let constraint = ast._eq(&one);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&self, cond: &RustBV) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python
        self.assumed_constraints.lock().push((cond.clone(), false));
        let ast = cond.to_z3_ast();
        let zero = z3::ast::BV::from_u64(0, 1);
        let constraint = ast._eq(&zero);
        self.add_constraint(constraint);
    }

    // =========================================================================
    // Satisfiability & Evaluation (Z3-backed)
    // =========================================================================

    /// Check if the current constraints are satisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat(&self) -> bool {
        // Check cache first
        if let Some(cached) = self.sat_cache.get() {
            return cached;
        }
        // Perform actual SAT check
        let result = matches!(self.solver.lock().check(), z3::SatResult::Sat);
        self.sat_cache.set(Some(result));
        result
    }

    /// Prime the SAT cache with a known value.
    /// Used after symbolic branch forking where the interpreter already
    /// proved feasibility — avoids redundant Z3 check() calls.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_sat_cache(&self, value: bool) {
        self.sat_cache.set(Some(value));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn set_sat_cache(&self, _value: bool) {}

    /// Check if a bitvector condition can be true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        self.check_branch_feasibility(cond).0
    }

    /// Check if a bitvector condition can be false.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        self.check_branch_feasibility(cond).1
    }

    /// Check both branch directions in a single solver session.
    /// Returns (can_be_true, can_be_false). Holds the solver lock once
    /// for both checks, reducing lock acquisitions from 8 to 2.
    /// When only one direction is feasible, skips the second Z3 check.
    #[cfg(feature = "vex-engine-z3")]
    pub fn check_branch_feasibility(&self, cond: &RustBV) -> (bool, bool) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return (v != 0, v == 0);
        }
        let ast = cond.to_z3_ast();
        let one = z3::ast::BV::from_u64(1, 1);
        let zero = z3::ast::BV::from_u64(0, 1);

        let solver = self.solver.lock();

        // Check true branch
        solver.push();
        solver.assert(&ast._eq(&one));
        let can_true = matches!(solver.check(), z3::SatResult::Sat);
        solver.pop(1);

        if !can_true {
            return (false, true); // Must be false-only
        }

        // Check false branch
        solver.push();
        solver.assert(&ast._eq(&zero));
        let can_false = matches!(solver.check(), z3::SatResult::Sat);
        solver.pop(1);

        (can_true, can_false)
    }

    /// Evaluate a bitvector to a concrete value if possible.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        // Try to use cached model first (avoids re-checking SAT)
        {
            let cache = self.model_cache.borrow();
            if let Some(ref model) = *cache {
                let ast = bv.to_z3_ast();
                if let Some(result) = model.eval(&ast, true) {
                    return Self::extract_bv_value(&result);
                }
            }
        }

        // Need to get a fresh model - must call check() first for Z3
        let solver = self.solver.lock();
        match solver.check() {
            z3::SatResult::Sat => {
                self.sat_cache.set(Some(true));
            }
            _ => {
                self.sat_cache.set(Some(false));
                return None;
            }
        }

        // Get model from the check we just did
        let model = solver.get_model()?;
        let ast = bv.to_z3_ast();
        let result = model.eval(&ast, true)?;
        let value = Self::extract_bv_value(&result);
        drop(solver); // Release lock before borrowing model_cache
        *self.model_cache.borrow_mut() = Some(model);
        value
    }

    /// Evaluate a bitvector to bytes (for values > 128 bits).
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            let byte_len = ((width + 7) / 8) as usize;
            let mut result = vec![0u8; byte_len];
            let bytes = v.to_be_bytes();
            let offset = byte_len.saturating_sub(16);
            for (i, &b) in bytes.iter().enumerate() {
                let src_idx = 16 - byte_len.min(16) + i;
                if src_idx < 16 && offset + i < byte_len {
                    result[offset + i] = bytes[src_idx];
                }
            }
            // Handle case where byte_len <= 16
            if byte_len <= 16 {
                result = bytes[16 - byte_len..].to_vec();
            }
            return Some(result);
        }

        // Need to get a model from Z3
        let solver = self.solver.lock();
        match solver.check() {
            z3::SatResult::Sat => {}
            _ => return None,
        }

        let model = solver.get_model()?;
        let ast = bv.to_z3_ast();
        let result = model.eval(&ast, true)?;
        Self::extract_bv_value_wide(&result, width)
    }

    /// Extract a u128 value from a Z3 BV result.
    /// For values > 128 bits, returns the low 128 bits (caller should use
    /// extract_bv_value_wide for arbitrarily large values).
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value(bv: &z3::ast::BV) -> Option<u128> {
        // Try as u64 first (fast path for <= 64-bit)
        if let Some(v) = bv.as_u64() {
            return Some(v as u128);
        }
        // For larger values, parse the string representation
        Self::extract_bv_value_from_string(bv)
    }

    /// Extract a BV value by parsing its string representation.
    /// Handles arbitrarily large values, returns low 128 bits.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value_from_string(bv: &z3::ast::BV) -> Option<u128> {
        let s = format!("{}", bv);
        // Z3 uses formats: #xHEXDIGITS, #bBINARY, or decimal
        if let Some(hex_str) = s.strip_prefix("#x") {
            // Parse as hex, taking low 128 bits
            parse_wide_hex_low128(hex_str)
        } else if let Some(bin_str) = s.strip_prefix("#b") {
            // Parse as binary, taking low 128 bits
            parse_wide_binary_low128(bin_str)
        } else {
            // Try decimal
            s.parse::<u128>().ok()
        }
    }

    /// Extract an arbitrarily large BV value as a Vec<u8> (big-endian).
    /// Used for values > 128 bits where we need the full value.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value_wide(bv: &z3::ast::BV, width: u32) -> Option<Vec<u8>> {
        let s = format!("{}", bv);
        if let Some(hex_str) = s.strip_prefix("#x") {
            // Parse full hex value to bytes
            parse_hex_to_bytes(hex_str, width)
        } else if let Some(bin_str) = s.strip_prefix("#b") {
            // Parse full binary value to bytes
            parse_binary_to_bytes(bin_str, width)
        } else {
            // Decimal - parse and convert
            parse_decimal_to_bytes(&s, width)
        }
    }

    /// Evaluate a bitvector and return up to n solutions.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return vec![v];
        }

        if n == 0 {
            return vec![];
        }

        let mut results = Vec::with_capacity(n);
        let ast = bv.to_z3_ast();

        // Hold lock for entire operation to avoid lifetime issues
        let solver = self.solver.lock();
        solver.push();

        for _ in 0..n {
            match solver.check() {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(result) = model.eval(&ast, true) {
                            if let Some(value) = Self::extract_bv_value(&result) {
                                results.push(value);
                                // Add constraint to exclude this value
                                let val_ast = if bv.width() <= 64 {
                                    z3::ast::BV::from_u64(value as u64, bv.width())
                                } else {
                                    let lo = z3::ast::BV::from_u64(value as u64, 64);
                                    let hi = z3::ast::BV::from_u64(
                                        (value >> 64) as u64,
                                        bv.width() - 64,
                                    );
                                    hi.concat(&lo)
                                };
                                solver.assert(&ast._eq(&val_ast).not());
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        results
    }

    /// Create a Z3 BV constant from a u128 value.
    #[cfg(feature = "vex-engine-z3")]
    fn make_bv_const(value: u128, width: u32) -> z3::ast::BV {
        if width <= 64 {
            z3::ast::BV::from_u64(value as u64, width)
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
            hi.concat(&lo)
        }
    }

    /// Get the minimum value of a bitvector using binary search (O(log N)).
    ///
    /// This implementation uses pure SAT checks without model value extraction,
    /// which allows it to work with bitvectors of any width (including >64 bits).
    /// Based on claripy's _extrema algorithm.
    #[cfg(feature = "vex-engine-z3")]
    pub fn min(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        // Set initial bounds based on signedness
        // For unsigned: [0, 2^width - 1]
        // For signed: [-(2^(width-1)), 2^(width-1) - 1] represented in two's complement
        let (mut lo, mut hi): (u128, u128) = if signed {
            // Signed: lo is most negative (0x8000...), hi is most positive (0x7FFF...)
            let sign_bit = 1u128 << (width - 1);
            let max_positive = sign_bit - 1;
            // In two's complement ordering for binary search, we search [0, max_positive] then [sign_bit, max_val]
            // But for signed comparison, Z3 handles this correctly with bvsle/bvsge
            // Start with the full signed range in two's complement representation
            (sign_bit, max_positive)
        } else {
            // Unsigned: [0, 2^width - 1]
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            (0, max_val)
        };

        let solver = self.solver.lock();
        solver.push();

        if signed {
            // For signed values, we need to handle the two's complement ordering
            // First check if a negative value (sign bit set) is possible
            solver.push();
            let zero = Self::make_bv_const(0, width);
            solver.assert(&ast.bvslt(&zero)); // bv < 0 (signed)
            let has_negative = matches!(solver.check(), z3::SatResult::Sat);
            solver.pop(1);

            if has_negative {
                // Minimum is negative, search in [sign_bit, all_ones] range
                let sign_bit = 1u128 << (width - 1);
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                lo = sign_bit;
                hi = max_val;

                // Binary search for minimum negative value (smallest = most negative)
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be <= mid (signed comparison)
                    solver.assert(&ast.bvsle(&mid_ast));
                    let can_be_le_mid = matches!(solver.check(), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_le_mid {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            } else {
                // Minimum is non-negative, search in [0, max_positive] range
                let max_positive = (1u128 << (width - 1)) - 1;
                lo = 0;
                hi = max_positive;

                // Binary search for minimum non-negative value
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be <= mid (signed comparison)
                    solver.assert(&ast.bvsle(&mid_ast));
                    let can_be_le_mid = matches!(solver.check(), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_le_mid {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            }
        } else {
            // Unsigned binary search
            while lo < hi {
                let mid = lo + (hi - lo) / 2;

                solver.push();
                let mid_ast = Self::make_bv_const(mid, width);
                // Check if bv can be <= mid (unsigned comparison)
                solver.assert(&ast.bvule(&mid_ast));
                let can_be_le_mid = matches!(solver.check(), z3::SatResult::Sat);
                solver.pop(1);

                if can_be_le_mid {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
        }

        solver.pop(1);
        Some(lo)
    }

    /// Get the maximum value of a bitvector using binary search (O(log N)).
    ///
    /// This implementation uses pure SAT checks without model value extraction,
    /// which allows it to work with bitvectors of any width (including >64 bits).
    /// Based on claripy's _extrema algorithm.
    #[cfg(feature = "vex-engine-z3")]
    pub fn max(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        let solver = self.solver.lock();
        solver.push();

        let (mut lo, mut hi): (u128, u128);

        if signed {
            // For signed values, we need to handle the two's complement ordering
            // First check if a non-negative value (sign bit clear) is possible
            let zero = Self::make_bv_const(0, width);

            solver.push();
            solver.assert(&ast.bvsge(&zero)); // bv >= 0 (signed)
            let has_non_negative = matches!(solver.check(), z3::SatResult::Sat);
            solver.pop(1);

            if has_non_negative {
                // Maximum is non-negative, search in [0, max_positive] range
                let max_positive = (1u128 << (width - 1)) - 1;
                lo = 0;
                hi = max_positive;

                // Binary search for maximum non-negative value
                while lo < hi {
                    // Use ceiling division to avoid infinite loop when lo + 1 == hi
                    let mid = lo + (hi - lo + 1) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be >= mid (signed comparison)
                    solver.assert(&ast.bvsge(&mid_ast));
                    let can_be_ge_mid = matches!(solver.check(), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_ge_mid {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
            } else {
                // Maximum is negative, search in [sign_bit, all_ones] range
                let sign_bit = 1u128 << (width - 1);
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                lo = sign_bit;
                hi = max_val;

                // Binary search for maximum negative value (largest = least negative = closest to 0)
                while lo < hi {
                    // Use ceiling division to avoid infinite loop when lo + 1 == hi
                    let mid = lo + (hi - lo + 1) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be >= mid (signed comparison)
                    solver.assert(&ast.bvsge(&mid_ast));
                    let can_be_ge_mid = matches!(solver.check(), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_ge_mid {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
            }
        } else {
            // Unsigned binary search
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            lo = 0;
            hi = max_val;

            while lo < hi {
                // Use ceiling division to avoid infinite loop when lo + 1 == hi
                let mid = lo + (hi - lo + 1) / 2;

                solver.push();
                let mid_ast = Self::make_bv_const(mid, width);
                // Check if bv can be >= mid (unsigned comparison)
                solver.assert(&ast.bvuge(&mid_ast));
                let can_be_ge_mid = matches!(solver.check(), z3::SatResult::Sat);
                solver.pop(1);

                if can_be_ge_mid {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
        }

        solver.pop(1);
        Some(lo)
    }

    /// Get the range [min, max] of possible values for a bitvector.
    ///
    /// Returns None if the constraints are unsatisfiable or evaluation fails.
    #[cfg(feature = "vex-engine-z3")]
    pub fn range(&self, bv: &RustBV) -> Option<(u128, u128)> {
        let min = self.min(bv, false)?;
        let max = self.max(bv, false)?;
        Some((min, max))
    }

    /// Get up to n concrete solutions for a bitvector.
    ///
    /// This is a convenience wrapper around eval_upto.
    #[cfg(feature = "vex-engine-z3")]
    pub fn solutions(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        self.eval_upto(bv, n)
    }

    /// Check if a specific value is a valid solution for a bitvector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return v == value;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        let val_ast = if width <= 64 {
            z3::ast::BV::from_u64(value as u64, width)
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
            hi.concat(&lo)
        };

        let constraint = ast._eq(&val_ast);

        self.solver.lock().push();
        self.solver.lock().assert(&constraint);
        let result = matches!(self.solver.lock().check(), z3::SatResult::Sat);
        self.solver.lock().pop(1);

        result
    }

    /// Save solver state for temporary constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn push(&self) {
        self.solver.lock().push();
        // Invalidate caches since constraint set may change
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    /// Restore solver state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn pop(&self) {
        self.solver.lock().pop(1);
        // Invalidate caches since constraint set has changed
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    // =========================================================================
    // Transactional Constraint Sync
    // =========================================================================

    /// Begin a new transaction.
    ///
    /// This pushes a new solver frame and records the constraint count,
    /// allowing rollback on failure via `transaction_rollback()`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_begin(&self) {
        self.push();
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    /// Commit the current transaction.
    ///
    /// This validates that constraints are satisfiable before committing.
    /// Returns an error if constraints became unsatisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Validate constraints are satisfiable before committing
        if !self.is_sat() {
            // Rollback on failure
            self.transaction_rollback()?;
            return Err(ConstraintSyncError::Unsatisfiable);
        }

        // Pop the solver frame but keep the constraints
        // Note: We don't actually pop here since we want to keep constraints
        // The push was just for protection during sync
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Rollback the current transaction.
    ///
    /// This restores the solver state to before `transaction_begin()` was called.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Pop the solver frame (discards constraints added since begin)
        self.pop();

        // Restore constraint count
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }

        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Get the current transaction level.
    ///
    /// Returns 0 if no transaction is active.
    pub fn current_push_level(&self) -> usize {
        self.push_level.load(Ordering::SeqCst)
    }

    /// Check if currently in a transaction.
    pub fn in_transaction(&self) -> bool {
        self.current_push_level() > 0
    }

    // Non-Z3 versions of transaction methods
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_begin(&self) {
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    #[cfg(feature = "vex-engine-z3")]
    pub fn unsat_core(&self) -> Vec<usize> {
        let solver = self.solver.lock();
        let core = solver.get_unsat_core();

        let trackers = self.constraint_trackers.lock();
        let mut result = Vec::new();

        // Match core tracking booleans to stored tracker indices by string representation
        for core_ast in core.iter() {
            let core_str = format!("{}", core_ast);
            for (i, tracker) in trackers.iter().enumerate() {
                if format!("{}", tracker) == core_str {
                    result.push(i);
                    break;
                }
            }
        }

        result
    }

    /// Get all solver assertions as strings.
    ///
    /// Returns string representations of all Z3 constraints. Useful for debugging
    /// and for syncing constraint state to Python. While not a full AST export,
    /// this allows Python to understand what constraints are active.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        let solver = self.solver.lock();
        solver.get_assertions().iter().map(|a| format!("{}", a)).collect()
    }

    /// Check the total number of assertions in the Z3 solver.
    ///
    /// This can be used to verify constraint sync between Rust and Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn z3_assertion_count(&self) -> usize {
        let solver = self.solver.lock();
        solver.get_assertions().len()
    }

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn is_sat(&self) -> bool {
        // Without Z3, we assume everything is satisfiable
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        // Without Z3, assume symbolic can be true
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        // Without Z3, assume symbolic can be false
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Without Z3, can only evaluate concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        // Without Z3, can only evaluate concrete values
        let width = bv.width();
        bv.as_u128().map(|v| {
            let byte_len = ((width + 7) / 8) as usize;
            let bytes = v.to_be_bytes();
            if byte_len <= 16 {
                bytes[16 - byte_len..].to_vec()
            } else {
                let mut result = vec![0u8; byte_len];
                result[byte_len - 16..].copy_from_slice(&bytes);
                result
            }
        })
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        if n == 0 {
            return vec![];
        }
        // Without Z3, can only return concrete values
        bv.as_u128().map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn min(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn max(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        // Without Z3, can only check concrete values
        bv.as_u128().map(|v| v == value).unwrap_or(true)
    }

    /// Get the range [min, max] of possible values for a bitvector.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn range(&self, bv: &RustBV) -> Option<(u128, u128)> {
        // Without Z3, can only return range for concrete values
        bv.as_u128().map(|v| (v, v))
    }

    /// Get up to n concrete solutions for a bitvector.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solutions(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        if n == 0 {
            return vec![];
        }
        // Without Z3, can only return concrete values
        bv.as_u128().map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn push(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn pop(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn unsat_core(&self) -> Vec<usize> {
        // Without Z3, no unsat core available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        // Without Z3, no constraints available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn z3_assertion_count(&self) -> usize {
        // Without Z3, no assertions
        0
    }

    // =========================================================================
    // Constraint Export (Phase 2 Fix)
    // =========================================================================

    /// Get the assumed constraints as (RustBV, is_assumed_true) pairs.
    ///
    /// This exports the tracked path constraints that can be converted to claripy
    /// ASTs for Python constraint sync. Each constraint is a 1-bit RustBV that was
    /// either assumed true or false during symbolic execution.
    pub fn get_assumed_constraints(&self) -> Vec<(RustBV, bool)> {
        self.assumed_constraints.lock().clone()
    }

    /// Get the number of assumed constraints.
    pub fn assumed_constraint_count(&self) -> usize {
        self.assumed_constraints.lock().len()
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the context, creating a new context with all constraints preserved.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        // Create a fresh Z3 solver and replay constraints instead of using
        // Z3_solver_translate (Solver::clone). Replaying is much faster:
        // assert_and_track is O(1) per constraint vs Z3_solver_translate
        // which copies internal solver state.
        let new_solver = z3::Solver::new();
        let mut params = z3::Params::new();
        params.set_bool("unsat_core", true);
        params.set_u32("timeout", 30000);
        new_solver.set_params(&params);

        // Clone assumed constraints for the fork
        let cloned_assumed = self.assumed_constraints.lock().clone();

        // Replay all constraints on the fresh solver
        let mut new_trackers = Vec::with_capacity(cloned_assumed.len());
        {
            use z3::ast::Ast;
            for (idx, (cond, is_true)) in cloned_assumed.iter().enumerate() {
                let ast = cond.to_z3_ast();
                let constraint = if *is_true {
                    let one = z3::ast::BV::from_u64(1, 1);
                    ast._eq(&one)
                } else {
                    let zero = z3::ast::BV::from_u64(0, 1);
                    ast._eq(&zero)
                };
                let track_name = format!("__track_{}", idx);
                let track_bool = z3::ast::Bool::new_const(track_name.as_str());
                new_solver.assert_and_track(&constraint, &track_bool);
                new_trackers.push(track_bool);
            }
        }

        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(cloned_assumed.len()),
            symbol_table: RwLock::new(self.symbol_table.read().clone()),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(Vec::new()),
            assumed_constraints: Mutex::new(cloned_assumed),
            solver: Mutex::new(new_solver),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(new_trackers),
        }
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork(&self) -> Self {
        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(self.symbol_table.read().clone()),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(Vec::new()),
            assumed_constraints: Mutex::new(self.assumed_constraints.lock().clone()),
        }
    }

    /// Fork with an additional constraint on the true branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_true(&self, cond: &RustBV) -> Self {
        let forked = self.fork();
        forked.assume_true(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_true(&self, _cond: &RustBV) -> Self {
        self.fork()
    }

    /// Fork with an additional constraint on the false branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_false(&self, cond: &RustBV) -> Self {
        let forked = self.fork();
        forked.assume_false(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_false(&self, _cond: &RustBV) -> Self {
        self.fork()
    }
}

impl Clone for SymContext {
    fn clone(&self) -> Self {
        self.fork()
    }
}

// =============================================================================
// Helper functions for parsing Z3 BV string representations
// =============================================================================

/// Parse a hex string to u128, taking low 128 bits if larger.
#[cfg(feature = "vex-engine-z3")]
fn parse_wide_hex_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 32 hex chars), take low 32 chars
    let low_hex = if s.len() > 32 {
        &s[s.len() - 32..]
    } else {
        s
    };
    u128::from_str_radix(low_hex, 16).ok()
}

/// Parse a binary string to u128, taking low 128 bits if larger.
#[cfg(feature = "vex-engine-z3")]
fn parse_wide_binary_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 128 bin chars), take low 128 chars
    let low_bin = if s.len() > 128 {
        &s[s.len() - 128..]
    } else {
        s
    };
    u128::from_str_radix(low_bin, 2).ok()
}

/// Parse a hex string to full bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_hex_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = ((width + 7) / 8) as usize;
    let mut result = vec![0u8; byte_len];

    // Pad hex string to even length
    let padded = if s.len() % 2 == 1 {
        format!("0{}", s)
    } else {
        s.to_string()
    };

    // Parse hex pairs from right to left (big-endian output)
    let hex_bytes: Vec<u8> = (0..padded.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&padded[i..i + 2], 16).ok())
        .collect();

    // Copy to result (right-aligned, big-endian)
    let offset = byte_len.saturating_sub(hex_bytes.len());
    for (i, &b) in hex_bytes.iter().enumerate() {
        if offset + i < byte_len {
            result[offset + i] = b;
        }
    }

    Some(result)
}

/// Parse a binary string to full bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_binary_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = ((width + 7) / 8) as usize;
    let mut result = vec![0u8; byte_len];

    // Parse bits from right to left
    let bits: Vec<u8> = s.chars().filter_map(|c| match c {
        '0' => Some(0),
        '1' => Some(1),
        _ => None,
    }).collect();

    // Build bytes from bits (big-endian)
    let bit_offset = byte_len * 8 - bits.len();
    for (i, &bit) in bits.iter().enumerate() {
        let bit_pos = bit_offset + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = 7 - (bit_pos % 8);
        if byte_idx < byte_len {
            result[byte_idx] |= bit << bit_idx;
        }
    }

    Some(result)
}

/// Parse a decimal string to bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_decimal_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    // For small values, parse and convert
    if let Ok(v) = s.parse::<u128>() {
        let byte_len = ((width + 7) / 8) as usize;
        let mut result = vec![0u8; byte_len];
        let bytes = v.to_be_bytes();
        let offset = byte_len.saturating_sub(16);
        for (i, &b) in bytes.iter().enumerate() {
            if offset + i < byte_len {
                result[offset + i] = b;
            }
        }
        return Some(result);
    }

    // For very large decimals, we'd need big integer parsing
    // This is rare in practice as Z3 typically uses hex format
    None
}

impl Default for SymContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generation() {
        let ctx = SymContext::new_mock();
        assert_eq!(ctx.next_id(), 0);
        assert_eq!(ctx.next_id(), 1);
        assert_eq!(ctx.next_id(), 2);
    }

    #[test]
    fn test_unique_names() {
        let ctx = SymContext::new_mock();
        let name1 = ctx.unique_name("x");
        let name2 = ctx.unique_name("x");
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_concrete_eval() {
        let ctx = SymContext::new_mock();
        let bv = RustBV::concrete(42, 32);
        assert_eq!(ctx.eval(&bv), Some(42));
    }

    #[test]
    fn test_fork() {
        let ctx = SymContext::new_mock();
        let id1 = ctx.next_id();

        let forked = ctx.fork();
        let id2 = forked.next_id();

        // Forked context should continue from same ID
        assert_eq!(id2, id1 + 1);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_z3_variable_identity() {
        // Test that Z3 variables with the same name are treated as the same variable
        let ctx = SymContext::new();

        // Create a symbolic variable
        let x = RustBV::symbolic(&ctx, "x", 32);

        // Add constraint: x > 10
        let ten = RustBV::concrete(10, 32);
        let gt_ten = x.ugt(&ten, &ctx);
        ctx.assume_true(&gt_ten);

        // Verify constraint is enforced
        assert!(ctx.solution(&x, 15)); // 15 > 10, should be true
        assert!(!ctx.solution(&x, 5));  // 5 > 10 is false, should be unsat

        // Now add constraint: x < 20
        let twenty = RustBV::concrete(20, 32);
        let lt_twenty = x.ult(&twenty, &ctx);
        ctx.assume_true(&lt_twenty);

        // Verify both constraints are enforced
        assert!(ctx.solution(&x, 15));  // 10 < 15 < 20
        assert!(!ctx.solution(&x, 5));   // 5 < 10
        assert!(!ctx.solution(&x, 25));  // 25 > 20
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_constrained() {
        // Test min/max with constrained variable
        let ctx = SymContext::new();

        // Create a symbolic variable
        let x = RustBV::symbolic(&ctx, "x", 32);

        // Add constraints: 10 < x < 20
        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&ten, &ctx));
        ctx.assume_true(&x.ult(&twenty, &ctx));

        // min should be 11, max should be 19
        let min_val = ctx.min(&x, false);
        let max_val = ctx.max(&x, false);

        assert_eq!(min_val, Some(11), "min should be 11");
        assert_eq!(max_val, Some(19), "max should be 19");
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_z3_same_name_different_create() {
        // Test that creating variables with the same name but different calls
        // still references the same Z3 variable
        use z3::ast::Ast;

        let ctx = SymContext::new();

        // Create two RustBV::symbolic with the same name
        let x1 = RustBV::symbolic(&ctx, "x_test", 32);
        let x2 = RustBV::symbolic(&ctx, "x_test", 32);

        // Add constraint using x1: x1 > 10
        let ten = RustBV::concrete(10, 32);
        let gt_ten = x1.ugt(&ten, &ctx);
        ctx.assume_true(&gt_ten);

        // Check using x2 - should have the same constraint if same variable
        // If they're different variables, x2 wouldn't have the constraint
        let ast2 = x2.to_z3_ast();

        // Try to find if x2 can be 5 (should be UNSAT if same as x1)
        ctx.push();
        let five = z3::ast::BV::from_u64(5, 32);
        let eq_five = ast2._eq(&five);
        ctx.add_constraint(eq_five);
        let can_be_five = ctx.is_sat();
        ctx.pop();

        assert!(!can_be_five, "x2 should have same constraints as x1 since same name");
    }
}
