//! Per-block address concretization cache + unconstrained-read minting.
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Get the address concretizer.
    pub fn concretizer(&self) -> &AddressConcretizer {
        &self.concretizer
    }

    /// Set custom concretizer settings.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Mint an unconstrained read value for the AVOID_MULTIVALUED_READS
    /// short-circuit. Delegates to `SymbolicMemory::unconstrained_read_value`
    /// when Rust-side memory is attached so the result honors
    /// `zero_fill_unconstrained`; otherwise falls back to a fresh
    /// process-unique BVS named `symbolic_read_unconstrained_N`.
    pub(super) fn fresh_unconstrained_read(&self, size: usize) -> RustBV {
        if let Some(rust_mem) = self.rust_memory.as_ref() {
            return rust_mem.unconstrained_read_value(size as u32, self.ctx);
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static UNC_READ_ID: AtomicU64 = AtomicU64::new(0);
        let id = UNC_READ_ID.fetch_add(1, Ordering::Relaxed);
        RustBV::symbolic(
            self.ctx,
            format!("symbolic_read_unconstrained_{}", id),
            (size * 8) as u32,
        )
    }

    /// Concretize for read with per-block caching.
    /// Uses read_range_limit and falls back to Any (single solution) if range is too large.
    ///
    /// Returns an `Arc<ConcretizationResult>` so cache hits / inserts only pay an
    /// atomic refcount bump rather than cloning a `Vec<u64>` for the Multiple variant.
    pub(super) fn concretize_cached_read(&mut self, addr: &RustBV) -> Arc<ConcretizationResult> {
        if let Some(concrete_addr) = addr.as_u64() {
            return Arc::new(ConcretizationResult::Single(concrete_addr));
        }

        let cache_key = Self::bv_cache_key(addr);
        // Note: read and write may produce different results for same address,
        // but within a block they're typically used consistently for a given address.
        // Cache the raw result and apply fallback after cache lookup.
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply read fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.read_fallback_any => {
                    if let Some(val) = self.ctx.eval(addr) {
                        Arc::new(ConcretizationResult::Single(val as u64))
                    } else {
                        result
                    }
                }
                _ => result,
            };
        }

        let conc_start = std::time::Instant::now();
        let result = Arc::new(self.concretizer.concretize_read(addr, self.ctx));
        let conc_elapsed = conc_start.elapsed();
        if self.profiling_enabled {
            self.stats.concretize_count += 1;
            self.stats.concretize_time_ns += conc_elapsed.as_nanos() as u64;
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Concretize for write with per-block caching.
    /// Uses write_range_limit and falls back to Max solution if range is too large.
    ///
    /// Returns an `Arc<ConcretizationResult>` (see `concretize_cached_read`).
    pub(super) fn concretize_cached_write(&mut self, addr: &RustBV) -> Arc<ConcretizationResult> {
        if let Some(concrete_addr) = addr.as_u64() {
            return Arc::new(ConcretizationResult::Single(concrete_addr));
        }

        let cache_key = Self::bv_cache_key(addr);
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply write fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.write_fallback_max => {
                    if let Some((_min, max)) = self.ctx.range(addr) {
                        Arc::new(ConcretizationResult::Single(max as u64))
                    } else if let Some(val) = self.ctx.eval(addr) {
                        Arc::new(ConcretizationResult::Single(val as u64))
                    } else {
                        result
                    }
                }
                _ => result,
            };
        }

        let conc_start = std::time::Instant::now();
        let result = Arc::new(self.concretizer.concretize_write(addr, self.ctx));
        let conc_elapsed = conc_start.elapsed();
        if self.profiling_enabled {
            self.stats.concretize_count += 1;
            self.stats.concretize_time_ns += conc_elapsed.as_nanos() as u64;
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Compute a cache key for a RustBV value.
    /// Uses the symbolic id for Symbolic/Constrained, and a hash of op+operand structure for Expression.
    fn bv_cache_key(bv: &RustBV) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        match bv {
            RustBV::Concrete { value, .. } => *value as u64,
            RustBV::Symbolic { id, .. } => *id,
            RustBV::Constrained { id, .. } => *id,
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                let mut hasher = DefaultHasher::new();
                // Hash op discriminant + width + operand keys recursively (1 level deep)
                std::mem::discriminant(op).hash(&mut hasher);
                width.hash(&mut hasher);
                for operand in operands.iter() {
                    match operand {
                        RustBV::Concrete { value, .. } => {
                            value.hash(&mut hasher);
                        }
                        RustBV::Symbolic { id, .. } => {
                            id.hash(&mut hasher);
                        }
                        RustBV::Constrained { id, .. } => {
                            id.hash(&mut hasher);
                        }
                        RustBV::Expression {
                            op: sub_op,
                            width: sub_w,
                            ..
                        } => {
                            std::mem::discriminant(sub_op).hash(&mut hasher);
                            sub_w.hash(&mut hasher);
                        }
                    }
                }
                hasher.finish()
            }
        }
    }
}

#[cfg(test)]
#[path = "concretize_cache_tests.rs"]
mod concretize_cache_tests;
