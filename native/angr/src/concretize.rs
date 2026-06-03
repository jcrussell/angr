//! Address concretization for symbolic memory access.
//!
//! This module provides the ability to concretize symbolic addresses into
//! concrete address sets, enabling memory operations on symbolic addresses
//! with bounded ranges.
//!
//! Matches Python's pluggable strategy pattern with separate read/write configs:
//! - Read: Range(limit) → Any (single arbitrary solution as fallback)
//! - Write: Range(limit) → Max (maximum solution as fallback)
//!
//! The concretization algorithm:
//! 1. Fast path: If address is concrete, return Single
//! 2. Try small enumeration (≤16 solutions)
//! 3. Get range [min, max] via solver
//! 4. If range ≤ limit, enumerate all solutions
//! 5. If range > limit, apply fallback: Any (reads) or Max (writes)

use crate::symbolic::{RustBV, SymContext, record_concretize_read, record_concretize_write};

/// Minimum range limit when `APPROXIMATE_MEMORY_INDICES` is enabled.
///
/// When the SimOption is on, both `read_range_limit` and `write_range_limit`
/// are bumped to at least 4 KiB so a single page-sized buffer can be
/// enumerated without falling back to the Any/Max strategies.
const APPROXIMATE_MIN_RANGE: u64 = 4096;
/// This determines which strategy chain to use (different limits and fallbacks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcretizationMode {
    /// Read concretization: Range(read_limit) → Any fallback
    Read,
    /// Write concretization: Range(write_limit) → Max fallback
    Write,
}

/// Result of address concretization.
#[derive(Debug, Clone)]
pub enum ConcretizationResult {
    /// Single concrete address (fast path or uniquely determined).
    Single(u64),
    /// Multiple concrete addresses (2-256 addresses for ITE chains).
    Multiple(Vec<u64>),
    /// Strided access pattern detected (e.g., arr[i*4] for stride=4).
    /// This enables efficient balanced ITE tree construction.
    Strided {
        /// Base address (minimum).
        base: u64,
        /// Stride between consecutive addresses.
        stride: u64,
        /// Number of elements (addresses = base, base+stride, ..., base+(count-1)*stride).
        count: u64,
    },
    /// Address range too large to enumerate.
    TooLarge {
        /// Minimum possible address.
        min: u64,
        /// Maximum possible address.
        max: u64,
        /// The configured limit.
        limit: u64,
    },
    /// Concretization failed (unsatisfiable, unsupported, etc.).
    Failed(String),
}

impl ConcretizationResult {
    /// Check if this is a successful concretization (Single, Multiple, or Strided).
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            ConcretizationResult::Single(_)
                | ConcretizationResult::Multiple(_)
                | ConcretizationResult::Strided { .. }
        )
    }

    /// Get the concrete addresses if successful.
    /// For Strided results, generates all addresses in the pattern.
    pub fn addresses(&self) -> Option<Vec<u64>> {
        match self {
            ConcretizationResult::Single(addr) => Some(vec![*addr]),
            ConcretizationResult::Multiple(addrs) => Some(addrs.clone()),
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => Some((0..*count).map(|i| base + i * stride).collect()),
            _ => None,
        }
    }

    /// Get a single address if there's exactly one.
    pub fn single(&self) -> Option<u64> {
        match self {
            ConcretizationResult::Single(addr) => Some(*addr),
            _ => None,
        }
    }

    /// Check if this is a strided access pattern.
    pub fn is_strided(&self) -> bool {
        matches!(self, ConcretizationResult::Strided { .. })
    }

    /// Get strided pattern parameters if this is a Strided result.
    pub fn strided_params(&self) -> Option<(u64, u64, u64)> {
        match self {
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => Some((*base, *stride, *count)),
            _ => None,
        }
    }
}

/// Candidate count `K` for a concretization result, used by the
/// `concretize_*_count` / `concretize_*_candidates` instrumentation
/// (angr-2j5v). `TooLarge` and `Failed` count as 0 — they're failure modes,
/// not enumerable address sets.
fn candidate_count(result: &ConcretizationResult) -> u32 {
    match result {
        ConcretizationResult::Single(_) => 1,
        ConcretizationResult::Multiple(addrs) => addrs.len().min(u32::MAX as usize) as u32,
        ConcretizationResult::Strided { count, .. } => (*count).min(u32::MAX as u64) as u32,
        ConcretizationResult::TooLarge { .. } | ConcretizationResult::Failed(_) => 0,
    }
}

/// Configuration for address concretization.
///
/// Mirrors Python's strategy chain with separate read/write configurations:
/// - Read: SimConcretizationStrategyRange(read_range_limit) → SimConcretizationStrategyAny()
/// - Write: SimConcretizationStrategyRange(write_range_limit) → SimConcretizationStrategyMax()
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AddressConcretizer {
    /// Maximum address range for read concretization (default: 1024, matching Python).
    pub read_range_limit: u64,
    /// Maximum address range for write concretization (default: 128, matching Python).
    pub write_range_limit: u64,
    /// Maximum number of solutions to enumerate (default: 256).
    pub max_solutions: usize,
    /// Maximum number of elements for strided access (default: 16384).
    /// If stride count exceeds this, falls back to TooLarge.
    pub max_stride_count: u64,
    /// Whether to enable stride detection (default: true).
    pub enable_stride_detection: bool,
    /// Minimum number of samples for stride detection (default: 4).
    pub stride_sample_count: usize,
    /// Whether to use approximate memory indices (from APPROXIMATE_MEMORY_INDICES option).
    /// When true, allows more aggressive concretization with potential approximation.
    pub use_approximate: bool,
    /// Whether SYMBOLIC_WRITE_ADDRESSES is enabled.
    /// When true, writes to symbolic addresses use Range strategy.
    /// When false, writes only concretize to a single value (or max as fallback).
    pub symbolic_write_addresses: bool,
    /// Read fallback strategy: return any single solution when range is too large.
    /// Matches Python's SimConcretizationStrategyAny.
    pub read_fallback_any: bool,
    /// Write fallback strategy: return maximum solution when range is too large.
    /// Matches Python's SimConcretizationStrategyMax.
    pub write_fallback_max: bool,
    /// AVOID_MULTIVALUED_READS: when true, symbolic-address loads skip
    /// concretization and return an unconstrained value. Mirrors
    /// `angr.storage.memory_mixins.address_concretization_mixin._load_one`
    /// branch at line 272 (`return self._default_value(...)`).
    pub avoid_multivalued_reads: bool,
    /// AVOID_MULTIVALUED_WRITES: when true, symbolic-address stores are
    /// silently dropped (no-op). Mirrors the early `return` at
    /// `address_concretization_mixin.py:327-329`.
    pub avoid_multivalued_writes: bool,

    // Legacy field for backward compatibility with callers using .max_range
    // This is kept in sync with read_range_limit.
    pub max_range: u64,
}

impl Default for AddressConcretizer {
    fn default() -> Self {
        AddressConcretizer {
            read_range_limit: 1024, // Match Python's default read range
            write_range_limit: 128, // Match Python's default write range
            max_solutions: 256,
            max_stride_count: 16384, // Max 16K elements in strided access
            enable_stride_detection: true,
            stride_sample_count: 4, // Sample 4 solutions for stride detection
            use_approximate: false, // Default to precise concretization
            symbolic_write_addresses: false, // Python default
            read_fallback_any: true, // Match Python: Any() fallback for reads
            write_fallback_max: true, // Match Python: Max() fallback for writes
            avoid_multivalued_reads: false, // Default: enumerate within strategy limits
            avoid_multivalued_writes: false,
            max_range: 1024,        // Legacy, kept in sync with read_range_limit
        }
    }
}

impl AddressConcretizer {
    /// Create a new concretizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a concretizer with custom settings.
    pub fn with_limits(max_range: u64, max_solutions: usize) -> Self {
        AddressConcretizer {
            read_range_limit: max_range,
            write_range_limit: max_range.min(128),
            max_range,
            max_solutions,
            ..Default::default()
        }
    }

    /// Create a concretizer with full custom settings.
    pub fn with_full_config(
        max_range: u64,
        max_solutions: usize,
        max_stride_count: u64,
        enable_stride_detection: bool,
    ) -> Self {
        AddressConcretizer {
            read_range_limit: max_range,
            write_range_limit: max_range.min(128),
            max_range,
            max_solutions,
            max_stride_count,
            enable_stride_detection,
            stride_sample_count: 4,
            use_approximate: false,
            symbolic_write_addresses: false,
            read_fallback_any: true,
            write_fallback_max: true,
            avoid_multivalued_reads: false,
            avoid_multivalued_writes: false,
        }
    }

    /// Configure from Python sim_options (legacy interface).
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `range_limit` - Optional custom range limit (default: 1024)
    pub fn configure(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.use_approximate = use_approximate;
        if let Some(limit) = range_limit {
            self.read_range_limit = limit;
            self.max_range = limit;
        }
        // When approximate is enabled, we can be more aggressive with range
        if use_approximate && self.read_range_limit < APPROXIMATE_MIN_RANGE {
            self.read_range_limit = APPROXIMATE_MIN_RANGE;
            self.max_range = APPROXIMATE_MIN_RANGE;
        }
    }

    /// Configure from Python's full strategy configuration.
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `read_range_limit` - Range limit for read strategies (default: 1024)
    /// * `write_range_limit` - Range limit for write strategies (default: 128)
    /// * `symbolic_write_addresses` - Whether SYMBOLIC_WRITE_ADDRESSES is enabled
    pub fn configure_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
        avoid_multivalued_reads: bool,
        avoid_multivalued_writes: bool,
    ) {
        self.use_approximate = use_approximate;
        self.symbolic_write_addresses = symbolic_write_addresses;
        self.avoid_multivalued_reads = avoid_multivalued_reads;
        self.avoid_multivalued_writes = avoid_multivalued_writes;

        if let Some(limit) = read_range_limit {
            self.read_range_limit = limit;
            self.max_range = limit;
        }
        if let Some(limit) = write_range_limit {
            self.write_range_limit = limit;
        }

        // When approximate is enabled, be more aggressive with range limits
        if use_approximate {
            if self.read_range_limit < APPROXIMATE_MIN_RANGE {
                self.read_range_limit = APPROXIMATE_MIN_RANGE;
                self.max_range = APPROXIMATE_MIN_RANGE;
            }
            if self.write_range_limit < APPROXIMATE_MIN_RANGE {
                self.write_range_limit = APPROXIMATE_MIN_RANGE;
            }
        }
    }

    /// Returns true when a load from `addr` should bypass concretization
    /// and produce an unconstrained value. Mirrors Python's
    /// `AVOID_MULTIVALUED_READS` short-circuit: only fires when the address
    /// is symbolic and the option is on. Concrete addresses always read
    /// through, even with the option set.
    #[inline]
    pub fn should_avoid_multivalued_read(&self, addr: &RustBV) -> bool {
        self.avoid_multivalued_reads && addr.as_u64().is_none()
    }

    /// Returns true when a store to `addr` should be dropped entirely.
    /// Mirrors Python's `AVOID_MULTIVALUED_WRITES` early-return: only fires
    /// when the address is symbolic and the option is on.
    #[inline]
    pub fn should_avoid_multivalued_write(&self, addr: &RustBV) -> bool {
        self.avoid_multivalued_writes && addr.as_u64().is_none()
    }

    /// Get the range limit for a given mode.
    fn range_limit_for_mode(&self, mode: ConcretizationMode) -> u64 {
        match mode {
            ConcretizationMode::Read => self.read_range_limit,
            ConcretizationMode::Write => {
                if self.symbolic_write_addresses {
                    self.write_range_limit
                } else {
                    // Without SYMBOLIC_WRITE_ADDRESSES, writes should concretize to single value
                    // But we still allow the range limit for annotated writes
                    self.write_range_limit
                }
            }
        }
    }

    /// Concretize for a read operation.
    /// Uses read_range_limit and falls back to Any (single solution) if range is too large.
    pub fn concretize_read(&self, addr: &RustBV, ctx: &SymContext) -> ConcretizationResult {
        let result = self.concretize_with_mode(addr, ctx, ConcretizationMode::Read);
        let result = match result {
            ConcretizationResult::TooLarge { .. } if self.read_fallback_any => {
                // Fallback: SimConcretizationStrategyAny - return single arbitrary solution
                if let Some(val) = ctx.eval(addr) {
                    ConcretizationResult::Single(val as u64)
                } else {
                    result
                }
            }
            _ => result,
        };
        record_concretize_read(candidate_count(&result));
        result
    }

    /// Concretize for a write operation.
    /// Uses write_range_limit and falls back to Max (maximum solution) if range is too large.
    pub fn concretize_write(&self, addr: &RustBV, ctx: &SymContext) -> ConcretizationResult {
        let result = self.concretize_with_mode(addr, ctx, ConcretizationMode::Write);
        let result = match result {
            ConcretizationResult::TooLarge { .. } if self.write_fallback_max => {
                // Fallback: SimConcretizationStrategyMax - return maximum solution
                if let Some((_min, max)) = ctx.range(addr) {
                    ConcretizationResult::Single(max as u64)
                } else if let Some(val) = ctx.eval(addr) {
                    ConcretizationResult::Single(val as u64)
                } else {
                    result
                }
            }
            _ => result,
        };
        record_concretize_write(candidate_count(&result));
        result
    }

    /// Concretize with a specific mode (determines range limit).
    fn concretize_with_mode(
        &self,
        addr: &RustBV,
        ctx: &SymContext,
        mode: ConcretizationMode,
    ) -> ConcretizationResult {
        let range_limit = self.range_limit_for_mode(mode);
        self.concretize_internal(addr, ctx, range_limit)
    }

    /// Concretize a symbolic address to a set of concrete addresses.
    /// Uses the read_range_limit (legacy behavior, equivalent to concretize_read without fallback).
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to concretize
    /// * `ctx` - The solver context with current constraints
    ///
    /// # Returns
    /// * `Single` if the address has exactly one possible value
    /// * `Multiple` if the address has 2-256 possible values
    /// * `Strided` if a regular stride pattern is detected (e.g., arr[i*4])
    /// * `TooLarge` if the address range exceeds the limit
    /// * `Failed` if concretization is not possible
    pub fn concretize(&self, addr: &RustBV, ctx: &SymContext) -> ConcretizationResult {
        self.concretize_internal(addr, ctx, self.read_range_limit)
    }

    /// Internal concretization with explicit range limit.
    fn concretize_internal(
        &self,
        addr: &RustBV,
        ctx: &SymContext,
        range_limit: u64,
    ) -> ConcretizationResult {
        // Fast path: if address is already concrete, return immediately
        if let Some(concrete_addr) = addr.as_u64() {
            return ConcretizationResult::Single(concrete_addr);
        }

        // Optimization: try direct solution enumeration first with a small limit.
        // For addresses with few solutions (common case: boolean index → 2 values),
        // this is dramatically faster than binary-search min/max which does ~64 Z3
        // checks regardless of solution count. eval_upto for 2 solutions: ~3 checks.
        const FAST_ENUM_LIMIT: usize = 16;
        let fast_solutions = ctx.solutions(addr, FAST_ENUM_LIMIT + 1);

        if fast_solutions.is_empty() {
            // No solutions — try single eval as fallback
            if let Some(single) = ctx.eval(addr) {
                return ConcretizationResult::Single(single as u64);
            }
            return ConcretizationResult::Failed("no solutions found".to_string());
        }

        if fast_solutions.len() == 1 {
            return ConcretizationResult::Single(fast_solutions[0] as u64);
        }

        if fast_solutions.len() <= FAST_ENUM_LIMIT {
            // Small solution set — return directly without expensive range() call
            let mut addrs: Vec<u64> = fast_solutions.iter().map(|&v| v as u64).collect();
            addrs.sort_unstable();

            if self.enable_stride_detection
                && addrs.len() >= 2
                && let Some(strided) = self.detect_stride_from_solutions(&addrs)
            {
                return strided;
            }

            return ConcretizationResult::Multiple(addrs);
        }

        // More than FAST_ENUM_LIMIT solutions — fall back to range-based approach.
        // Use range_seeded() with the solutions we already enumerated to tighten
        // the binary-search bounds (saves ~half the SAT calls for `min` when the
        // smallest known solution is well below 2^width).
        let mut smallest_known = u128::MAX;
        let mut largest_known: u128 = 0;
        for &s in &fast_solutions {
            if s < smallest_known {
                smallest_known = s;
            }
            if s > largest_known {
                largest_known = s;
            }
        }
        let (min, max) = match ctx.range_seeded(addr, smallest_known, largest_known) {
            Some((min, max)) => (min as u64, max as u64),
            None => {
                // Range failed but we have solutions from fast enum — use them
                let mut addrs: Vec<u64> = fast_solutions
                    .iter()
                    .take(FAST_ENUM_LIMIT)
                    .map(|&v| v as u64)
                    .collect();
                addrs.sort_unstable();
                return ConcretizationResult::Multiple(addrs);
            }
        };

        if min == max {
            return ConcretizationResult::Single(min);
        }

        let range_size = max.saturating_sub(min);
        if range_size > range_limit {
            if self.enable_stride_detection
                && let Some(strided) = self.try_detect_stride(addr, ctx, min, max)
            {
                return strided;
            }
            return ConcretizationResult::TooLarge {
                min,
                max,
                limit: range_limit,
            };
        }

        // Range is manageable — enumerate all solutions
        let solutions = ctx.solutions(addr, self.max_solutions);

        match solutions.len() {
            0 => {
                if let Some(single) = ctx.eval(addr) {
                    return ConcretizationResult::Single(single as u64);
                }
                ConcretizationResult::Failed("no solutions found".to_string())
            }
            1 => ConcretizationResult::Single(solutions[0] as u64),
            _ => {
                let mut addrs: Vec<u64> = solutions.iter().map(|&v| v as u64).collect();
                addrs.sort_unstable();

                if self.enable_stride_detection
                    && addrs.len() >= 2
                    && let Some(strided) = self.detect_stride_from_solutions(&addrs)
                {
                    return strided;
                }

                ConcretizationResult::Multiple(addrs)
            }
        }
    }

    /// Try to detect a stride pattern by sampling solutions.
    ///
    /// This is used when the address range is too large to enumerate,
    /// but might have a regular stride pattern (e.g., arr[i*4]).
    fn try_detect_stride(
        &self,
        addr: &RustBV,
        ctx: &SymContext,
        min: u64,
        max: u64,
    ) -> Option<ConcretizationResult> {
        // Sample a few solutions to detect stride
        let samples = ctx.solutions(addr, self.stride_sample_count.max(4));
        if samples.len() < 2 {
            return None;
        }

        let mut addrs: Vec<u64> = samples.iter().map(|&v| v as u64).collect();
        addrs.sort_unstable();

        // Compute GCD of all differences
        let stride = self.compute_gcd_of_differences(&addrs)?;

        // Verify the stride covers the range properly
        if stride > 1 {
            let count = (max - min) / stride + 1;
            if count <= self.max_stride_count {
                return Some(ConcretizationResult::Strided {
                    base: min,
                    stride,
                    count,
                });
            }
        }

        None
    }

    /// Detect stride pattern from a list of sorted addresses.
    fn detect_stride_from_solutions(&self, addrs: &[u64]) -> Option<ConcretizationResult> {
        if addrs.len() < 2 {
            return None;
        }

        let stride = self.compute_gcd_of_differences(addrs)?;

        // Only use Strided if stride > 1 and it's a complete pattern
        if stride > 1 {
            let base = addrs[0];
            let max = addrs[addrs.len() - 1];
            let expected_count = (max - base) / stride + 1;

            // Check if all expected addresses are present
            if expected_count == addrs.len() as u64 {
                return Some(ConcretizationResult::Strided {
                    base,
                    stride,
                    count: expected_count,
                });
            }
        }

        None
    }

    /// Compute GCD of differences between consecutive sorted addresses.
    fn compute_gcd_of_differences(&self, addrs: &[u64]) -> Option<u64> {
        if addrs.len() < 2 {
            return None;
        }

        let mut gcd: Option<u64> = None;

        for i in 1..addrs.len() {
            let diff = addrs[i].saturating_sub(addrs[i - 1]);
            if diff == 0 {
                continue; // Skip duplicates
            }

            gcd = Some(match gcd {
                None => diff,
                Some(g) => Self::gcd(g, diff),
            });
        }

        gcd.filter(|&g| g > 0)
    }

    /// Compute GCD using Euclidean algorithm.
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    /// Concretize with base+offset optimization.
    ///
    /// For patterns like `base + constant_offset` where `base` is symbolic
    /// but `offset` is constant, we can concretize the base once and adjust.
    ///
    /// # Arguments
    /// * `base` - The symbolic base address
    /// * `offset` - The constant offset (can be negative via i64)
    /// * `ctx` - The solver context
    ///
    /// # Returns
    /// Concretization result with addresses adjusted by offset.
    pub fn concretize_with_offset(
        &self,
        base: &RustBV,
        offset: i64,
        ctx: &SymContext,
    ) -> ConcretizationResult {
        match self.concretize(base, ctx) {
            ConcretizationResult::Single(addr) => {
                ConcretizationResult::Single((addr as i64).wrapping_add(offset) as u64)
            }
            ConcretizationResult::Multiple(addrs) => {
                let adjusted: Vec<u64> = addrs
                    .iter()
                    .map(|&a| (a as i64).wrapping_add(offset) as u64)
                    .collect();
                ConcretizationResult::Multiple(adjusted)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_address() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();

        let addr = RustBV::concrete(0x1000, 64);
        let result = concretizer.concretize(&addr, &ctx);

        match result {
            ConcretizationResult::Single(a) => assert_eq!(a, 0x1000),
            _ => panic!("expected Single result"),
        }
    }

    #[test]
    fn test_result_helpers() {
        let single = ConcretizationResult::Single(0x1000);
        assert!(single.is_success());
        assert_eq!(single.single(), Some(0x1000));
        assert_eq!(single.addresses(), Some(vec![0x1000]));
        assert!(!single.is_strided());

        let multi = ConcretizationResult::Multiple(vec![0x1000, 0x1004, 0x1008]);
        assert!(multi.is_success());
        assert_eq!(multi.single(), None);
        assert_eq!(multi.addresses(), Some(vec![0x1000, 0x1004, 0x1008]));
        assert!(!multi.is_strided());

        let strided = ConcretizationResult::Strided {
            base: 0x1000,
            stride: 4,
            count: 10,
        };
        assert!(strided.is_success());
        assert!(strided.is_strided());
        assert_eq!(strided.strided_params(), Some((0x1000, 4, 10)));
        assert_eq!(
            strided.addresses(),
            Some(vec![
                0x1000, 0x1004, 0x1008, 0x100c, 0x1010, 0x1014, 0x1018, 0x101c, 0x1020, 0x1024
            ])
        );

        let failed = ConcretizationResult::Failed("test".to_string());
        assert!(!failed.is_success());
        assert_eq!(failed.addresses(), None);
    }

    #[test]
    fn test_offset_adjustment() {
        let ctx = SymContext::new_mock();
        let concretizer = AddressConcretizer::new();

        let base = RustBV::concrete(0x8000, 64);
        let result = concretizer.concretize_with_offset(&base, -0x100, &ctx);

        match result {
            ConcretizationResult::Single(a) => assert_eq!(a, 0x7F00),
            _ => panic!("expected Single result"),
        }
    }

    #[test]
    fn test_gcd() {
        assert_eq!(AddressConcretizer::gcd(12, 8), 4);
        assert_eq!(AddressConcretizer::gcd(17, 13), 1);
        assert_eq!(AddressConcretizer::gcd(100, 25), 25);
        assert_eq!(AddressConcretizer::gcd(0, 5), 5);
        assert_eq!(AddressConcretizer::gcd(5, 0), 5);
    }

    #[test]
    fn test_stride_detection_from_solutions() {
        let concretizer = AddressConcretizer::new();

        // Perfect stride pattern
        let addrs = vec![0x1000, 0x1004, 0x1008, 0x100c, 0x1010];
        let result = concretizer.detect_stride_from_solutions(&addrs);
        assert!(result.is_some());
        if let Some(ConcretizationResult::Strided {
            base,
            stride,
            count,
        }) = result
        {
            assert_eq!(base, 0x1000);
            assert_eq!(stride, 4);
            assert_eq!(count, 5);
        }

        // Irregular pattern - no stride
        let irregular = vec![0x1000, 0x1004, 0x1010, 0x1020];
        let result = concretizer.detect_stride_from_solutions(&irregular);
        assert!(result.is_none());
    }

    #[test]
    fn test_default_config() {
        let concretizer = AddressConcretizer::default();
        assert_eq!(concretizer.read_range_limit, 1024); // Match Python default
        assert_eq!(concretizer.write_range_limit, 128); // Match Python default
        assert_eq!(concretizer.max_range, 1024); // Legacy compatibility
        assert_eq!(concretizer.max_solutions, 256);
        assert_eq!(concretizer.max_stride_count, 16384);
        assert!(concretizer.enable_stride_detection);
        assert!(!concretizer.use_approximate);
        assert!(!concretizer.symbolic_write_addresses);
        assert!(concretizer.read_fallback_any);
        assert!(concretizer.write_fallback_max);
    }

    #[test]
    fn test_configure() {
        let mut concretizer = AddressConcretizer::default();

        // Configure without approximate
        concretizer.configure(false, Some(2048));
        assert_eq!(concretizer.read_range_limit, 2048);
        assert!(!concretizer.use_approximate);

        // Configure with approximate - should increase range to at least 4096
        concretizer.configure(true, None);
        assert!(concretizer.use_approximate);
        assert!(concretizer.read_range_limit >= 4096);
    }

    #[test]
    fn test_configure_strategies() {
        let mut concretizer = AddressConcretizer::default();

        concretizer.configure_strategies(false, Some(2048), Some(256), true, false, false);
        assert_eq!(concretizer.read_range_limit, 2048);
        assert_eq!(concretizer.write_range_limit, 256);
        assert!(concretizer.symbolic_write_addresses);
        assert!(!concretizer.use_approximate);
        assert!(!concretizer.avoid_multivalued_reads);
        assert!(!concretizer.avoid_multivalued_writes);

        // With approximate, limits should increase to at least 4096
        concretizer.configure_strategies(true, Some(512), Some(128), false, false, false);
        assert!(concretizer.read_range_limit >= 4096);
        assert!(concretizer.write_range_limit >= 4096);

        // Avoid-multivalued flags pass through.
        concretizer.configure_strategies(false, Some(1024), Some(128), false, true, true);
        assert!(concretizer.avoid_multivalued_reads);
        assert!(concretizer.avoid_multivalued_writes);
    }
}
