//! Per-block address concretization cache + unconstrained-read minting.
use super::*;

/// Which strategy chain a `concretize_cache` entry was produced under.
///
/// The cache stores the *raw* shape a chain returned, and `Multiple` /
/// `Strided` / `TooLarge` all encode the range limit that produced them. So an
/// entry is only reusable by a consumer that runs the **same** limit — sharing
/// across limits lets a wide `Multiple` computed under `read_range_limit`
/// (default 1024) be handed to a store whose own `write_range_limit` (default
/// 128) would have collapsed it to a single Max-pinned address, making the
/// store fan out to hundreds of cells purely as a function of statement order
/// within the block (angr-360gt).
///
/// Only `TooLarge` is re-validated on a hit (each consumer re-applies its own
/// fallback), which is why the pre-fix code looked safe: the *other* shapes
/// were passed through verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConcretizeNs {
    /// `concretize_cached_read` and `concretize_cached_jump`. Both drive
    /// `Range(read_range_limit)`, so their raw shapes are interchangeable; they
    /// differ only in the post-hoc fallback each applies to a `TooLarge`, and
    /// read's Any fallback pins `addr == chosen` before caching a `Single`, so
    /// a fresh jump concretization of the same expression yields that `Single`
    /// too.
    ReadJump,
    /// `concretize_cached_write`. Runs `Range(write_range_limit)` → Max (or a
    /// Max-only chain when `SYMBOLIC_WRITE_ADDRESSES` is off), a different
    /// limit *and* a different fallback from the read side.
    Write,
}

impl<'a> VEXInterpreter<'a> {
    /// Set custom concretizer settings.
    pub(crate) fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
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

        // Keyed in the ReadJump namespace: a write's raw shape was computed
        // under `write_range_limit` and must not be reused here (see
        // `ConcretizeNs`).
        let cache_key = Self::bv_cache_key(ConcretizeNs::ReadJump, addr);
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply read fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.read_fallback_any => {
                    if let Some(val) = self.ctx.eval(addr) {
                        crate::concretize::pin_fallback_addr(self.ctx, addr, val as u64);
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
            self.stats.concretize_time_ns += crate::duration_ns(conc_elapsed);
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

        // Private namespace: `write_range_limit` (default 128) is narrower than
        // the read side's 1024, so a `Multiple`/`Strided` cached by a load or a
        // jump target would bypass this site's limit and Max fallback entirely
        // (angr-360gt).
        let cache_key = Self::bv_cache_key(ConcretizeNs::Write, addr);
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply write fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.write_fallback_max => {
                    if let Some((_min, max)) = self.ctx.range(addr) {
                        crate::concretize::pin_fallback_addr(self.ctx, addr, max as u64);
                        Arc::new(ConcretizationResult::Single(max as u64))
                    } else if let Some(val) = self.ctx.eval(addr) {
                        crate::concretize::pin_fallback_addr(self.ctx, addr, val as u64);
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
            self.stats.concretize_time_ns += crate::duration_ns(conc_elapsed);
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Concretize a **jump target** with per-block caching.
    ///
    /// Backs `exits.rs::eval_next_addr`, which needs the plain
    /// `AddressConcretizer::concretize` chain — `Range(read_range_limit)` with
    /// *no* fallback, since a jump target that does not resolve to exactly one
    /// address is surfaced as `Unsupported` and deferred to Python rather than
    /// pinned to an arbitrary solution.
    ///
    /// Shares the `ConcretizeNs::ReadJump` half of `concretize_cache` with
    /// `concretize_cached_read` — both run `Range(read_range_limit)`, so their
    /// raw shapes are interchangeable. It deliberately does **not** see
    /// `concretize_cached_write`'s entries: those were computed under
    /// `write_range_limit` (angr-360gt).
    ///
    /// Reusing read's entries is sound because every `Single` read can store is
    /// either a genuinely unique solution or a fallback value that
    /// `concretize::pin_fallback_addr` has already asserted into the solver — so
    /// a fresh `concretize` on the same expression would return that same
    /// `Single` anyway. Every non-`Single` shape (`Multiple` / `Strided` /
    /// `TooLarge`) maps to the same `Unsupported` deferral here.
    ///
    /// Returns an `Arc<ConcretizationResult>` (see `concretize_cached_read`).
    pub(super) fn concretize_cached_jump(&mut self, addr: &RustBV) -> Arc<ConcretizationResult> {
        if let Some(concrete_addr) = addr.as_u64() {
            return Arc::new(ConcretizationResult::Single(concrete_addr));
        }

        let cache_key = Self::bv_cache_key(ConcretizeNs::ReadJump, addr);
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            // No jump-side fallback to re-apply, unlike the read/write variants.
            return Arc::clone(cached);
        }

        let conc_start = std::time::Instant::now();
        let result = Arc::new(self.concretizer.concretize(addr, self.ctx));
        let conc_elapsed = conc_start.elapsed();
        if self.profiling_enabled {
            self.stats.concretize_count += 1;
            self.stats.concretize_time_ns += crate::duration_ns(conc_elapsed);
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Compute a cache key for a RustBV value: the consumer's `ConcretizeNs`
    /// tag plus a hash of the FULL op tree, variant-tagged at every node by
    /// `hash_bv`.
    ///
    /// The namespace tag partitions the one map so entries never cross between
    /// strategy chains with different range limits — see `ConcretizeNs`.
    ///
    /// Every variant goes through `hash_bv` rather than shortcutting leaves to
    /// their raw `id` / `value`. An untagged leaf key would let a
    /// `Concrete { value: 5 }` and a `Symbolic { id: 5 }` share a cache slot —
    /// harmless today, since all three call sites early-return on
    /// `addr.as_u64()` (`Some` for both `Concrete` and `Constrained`) and so
    /// only ever reach here with a `Symbolic` or `Expression`, but a foot-gun
    /// for any future caller that skips that filter.
    ///
    /// The Expression case recurses over the entire operand tree (angr-owr37):
    /// a nested `Expression` operand carries the `RustBV::EXPRESSION_ID`
    /// sentinel for its `id`, so hashing only `(discriminant(op), width)` at each
    /// nested node — as the original 1-level implementation did — dropped all
    /// leaf identity for symbols sitting >= 2 levels deep. Two distinct
    /// addresses like `Add(And(x,0xf),c)` vs `Add(And(y,0xf),c)` then collided,
    /// and `concretize_cached_write` returned the first store's target for the
    /// second store — silent wrong-address memory corruption.
    fn bv_cache_key(ns: ConcretizeNs, bv: &RustBV) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        match ns {
            ConcretizeNs::ReadJump => 0u8.hash(&mut hasher),
            ConcretizeNs::Write => 1u8.hash(&mut hasher),
        }
        Self::hash_bv(bv, &mut hasher);
        hasher.finish()
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

test_submod!("concretize_cache_tests.rs" => concretize_cache_tests);
