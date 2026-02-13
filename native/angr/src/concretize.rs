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
    /// Check if this is a successful concretization (Single or Multiple).
    pub fn is_success(&self) -> bool {
        matches!(self, ConcretizationResult::Single(_) | ConcretizationResult::Multiple(_))
    }

    /// Get the concrete addresses if successful.
    pub fn addresses(&self) -> Option<Vec<u64>> {
        match self {
            ConcretizationResult::Single(addr) => Some(vec![*addr]),
            ConcretizationResult::Multiple(addrs) => Some(addrs.clone()),
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
}

/// Configuration for address concretization.
#[derive(Debug, Clone)]
pub struct AddressConcretizer {
    /// Maximum address range to consider (default: 1024).
    /// If max - min > max_range, concretization fails.
    pub max_range: u64,
    /// Maximum number of solutions to enumerate (default: 256).
    pub max_solutions: usize,
}

impl Default for AddressConcretizer {
    fn default() -> Self {
        AddressConcretizer {
            max_range: 1024,
            max_solutions: 256,
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

        // Check if the range is too large
        let range_size = max.saturating_sub(min);
        if range_size > self.max_range {
            return ConcretizationResult::TooLarge {
                min,
                max,
                limit: self.max_range,
            };
        }

        // If min == max, there's only one solution
        if min == max {
            return ConcretizationResult::Single(min);
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
                ConcretizationResult::Multiple(addrs)
            }
        }
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

        let multi = ConcretizationResult::Multiple(vec![0x1000, 0x1004, 0x1008]);
        assert!(multi.is_success());
        assert_eq!(multi.single(), None);
        assert_eq!(multi.addresses(), Some(vec![0x1000, 0x1004, 0x1008]));

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
}
