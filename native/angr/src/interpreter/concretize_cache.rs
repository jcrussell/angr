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
            format!("symbolic_read_unconstrained_{id}"),
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
    /// Uses the symbolic id for Symbolic/Constrained, and a hash of the FULL op
    /// tree for Expression.
    ///
    /// The Expression case recurses over the entire operand tree (angr-owr37):
    /// a nested `Expression` operand carries the `RustBV::EXPRESSION_ID`
    /// sentinel for its `id`, so hashing only `(discriminant(op), width)` at each
    /// nested node — as the original 1-level implementation did — dropped all
    /// leaf identity for symbols sitting >= 2 levels deep. Two distinct
    /// addresses like `Add(And(x,0xf),c)` vs `Add(And(y,0xf),c)` then collided,
    /// and `concretize_cached_write` returned the first store's target for the
    /// second store — silent wrong-address memory corruption.
    fn bv_cache_key(bv: &RustBV) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        match bv {
            RustBV::Concrete { value, .. } => *value as u64,
            RustBV::Symbolic { id, .. } => *id,
            RustBV::Constrained { id, .. } => *id,
            RustBV::Expression { .. } => {
                let mut hasher = DefaultHasher::new();
                Self::hash_bv(bv, &mut hasher);
                hasher.finish()
            }
        }
    }

    /// Feed the full structure of `bv` into `hasher`, recursing into nested
    /// `Expression` operands so leaf identity at every depth participates in the
    /// key. A per-variant tag byte plus the operand count prevent structurally
    /// distinct trees from hashing alike.
    fn hash_bv(bv: &RustBV, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        match bv {
            RustBV::Concrete { value, width } => {
                0u8.hash(hasher);
                value.hash(hasher);
                width.hash(hasher);
            }
            RustBV::Symbolic { id, width, .. } => {
                1u8.hash(hasher);
                id.hash(hasher);
                width.hash(hasher);
            }
            RustBV::Constrained { id, width, .. } => {
                2u8.hash(hasher);
                id.hash(hasher);
                width.hash(hasher);
            }
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                3u8.hash(hasher);
                // Hash the whole op, not just its discriminant: BVOp payloads
                // (ZeroExt(n), Extract(hi,lo), Float{..}) distinguish otherwise
                // structurally-identical trees — e.g. Extract(7,0,x) vs
                // Extract(15,8,x) must not share a key (angr-owr37).
                op.hash(hasher);
                width.hash(hasher);
                operands.len().hash(hasher);
                for operand in operands.iter() {
                    Self::hash_bv(operand, hasher);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "concretize_cache_tests.rs"]
mod concretize_cache_tests;
