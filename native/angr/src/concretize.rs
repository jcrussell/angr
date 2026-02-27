//! Address concretization for symbolic memory access.
//!
//! This module provides the ability to concretize symbolic addresses into
//! concrete address sets, enabling memory operations on symbolic addresses
//! with bounded ranges.
//!
//! The concretization strategy:
//! 1. Fast path: If address is concrete, return Single
//! 2. Get range [min, max] via solver
//! 3. If range > max_range, return TooLarge
//! 4. Get actual solutions via solver
//! 5. Return Single or Multiple based on solution count

use crate::symbolic::{RustBV, SymContext};

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
            ConcretizationResult::Strided { base, stride, count } => {
                Some((0..*count).map(|i| base + i * stride).collect())
            }
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
            ConcretizationResult::Strided { base, stride, count } => Some((*base, *stride, *count)),
            _ => None,
        }
    }
}

/// Configuration for address concretization.
#[derive(Debug, Clone)]
pub struct AddressConcretizer {
    /// Maximum address range to consider (default: 1024, matching Python).
    /// If max - min > max_range, concretization fails.
    pub max_range: u64,
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
}

impl Default for AddressConcretizer {
    fn default() -> Self {
        AddressConcretizer {
            max_range: 1024,            // Match Python's default range limit
            max_solutions: 256,
            max_stride_count: 16384,    // Max 16K elements in strided access
            enable_stride_detection: true,
            stride_sample_count: 4,     // Sample 4 solutions for stride detection
            use_approximate: false,     // Default to precise concretization
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
            max_range,
            max_solutions,
            max_stride_count,
            enable_stride_detection,
            stride_sample_count: 4,
            use_approximate: false,
        }
    }

    /// Configure from Python sim_options.
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `range_limit` - Optional custom range limit (default: 1024)
    pub fn configure(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.use_approximate = use_approximate;
        if let Some(limit) = range_limit {
            self.max_range = limit;
        }
        // When approximate is enabled, we can be more aggressive with range
        if use_approximate && self.max_range < 4096 {
            self.max_range = 4096;
        }
    }

    /// Concretize a symbolic address to a set of concrete addresses.
    ///
    /// This method tries to determine the concrete addresses that a symbolic
    /// address can take, given the current constraint context.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to concretize
    /// * `ctx` - The solver context with current constraints
    ///
    /// # Returns
    /// * `Single` if the address has exactly one possible value
    /// * `Multiple` if the address has 2-256 possible values
    /// * `Strided` if a regular stride pattern is detected (e.g., arr[i*4])
    /// * `TooLarge` if the address range exceeds max_range
    /// * `Failed` if concretization is not possible
    pub fn concretize(&self, addr: &RustBV, ctx: &SymContext) -> ConcretizationResult {
        // Fast path: if address is already concrete, return immediately
        if let Some(concrete_addr) = addr.as_u64() {
            return ConcretizationResult::Single(concrete_addr);
        }

        // Get the range of possible values
        let (min, max) = match ctx.range(addr) {
            Some((min, max)) => (min as u64, max as u64),
            None => {
                // Fallback: try to get any single solution when range fails
                if let Some(single) = ctx.eval(addr) {
                    return ConcretizationResult::Single(single as u64);
                }
                return ConcretizationResult::Failed(
                    "could not determine address range".to_string()
                );
            }
        };

        // If min == max, there's only one solution
        if min == max {
            return ConcretizationResult::Single(min);
        }

        // Check if the range is too large
        let range_size = max.saturating_sub(min);
        if range_size > self.max_range {
            // Try stride detection for large ranges
            if self.enable_stride_detection {
                if let Some(strided) = self.try_detect_stride(addr, ctx, min, max) {
                    return strided;
                }
            }
            return ConcretizationResult::TooLarge {
                min,
                max,
                limit: self.max_range,
            };
        }

        // Get actual solutions
        let solutions = ctx.solutions(addr, self.max_solutions);

        match solutions.len() {
            0 => {
                // Fallback: try single eval when enumeration fails
                if let Some(single) = ctx.eval(addr) {
                    return ConcretizationResult::Single(single as u64);
                }
                ConcretizationResult::Failed("no solutions found".to_string())
            }
            1 => ConcretizationResult::Single(solutions[0] as u64),
            _ => {
                // Convert to u64 and sort for deterministic ITE chain ordering
                let mut addrs: Vec<u64> = solutions.iter().map(|&v| v as u64).collect();
                addrs.sort_unstable();

                // Check for stride pattern in the solutions
                if self.enable_stride_detection && addrs.len() >= 2 {
                    if let Some(strided) = self.detect_stride_from_solutions(&addrs) {
                        return strided;
                    }
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
        if let Some(ConcretizationResult::Strided { base, stride, count }) = result {
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
        assert_eq!(concretizer.max_range, 1024);  // Match Python default
        assert_eq!(concretizer.max_solutions, 256);
        assert_eq!(concretizer.max_stride_count, 16384);
        assert!(concretizer.enable_stride_detection);
        assert!(!concretizer.use_approximate);
    }

    #[test]
    fn test_configure() {
        let mut concretizer = AddressConcretizer::default();

        // Configure without approximate
        concretizer.configure(false, Some(2048));
        assert_eq!(concretizer.max_range, 2048);
        assert!(!concretizer.use_approximate);

        // Configure with approximate - should increase range to at least 4096
        concretizer.configure(true, None);
        assert!(concretizer.use_approximate);
        assert!(concretizer.max_range >= 4096);
    }
}
