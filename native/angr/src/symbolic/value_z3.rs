//! Z3 AST construction for [`RustBV`] (integer + floating-point).
//!
//! Slice .2 of the `symbolic/value.rs` split (angr-7hwz). Carries the Z3
//! conversion `impl RustBV` block: `to_z3_ast` / `to_z3_ast_cached` /
//! `to_z3_bool` / `to_z3_bool_cached`, the cached integer builder
//! `build_z3_ast_cached` (+ `emit_extract_z3_cached`), and the FP builders
//! `build_fp_z3_ast_cached` and its `build_fp_*` round/arith/convert helpers.
//!
//! Whole module is gated on `vex-engine-z3` (every method was individually so
//! in `value.rs`). Z3 types are pulled in via function-local `use` inside each
//! method; stats counters are reached through the qualified `super::stats::`
//! path, so the only crate-level import needed is the value enums.

use super::value::{BVOp, FloatOpKind, FloatPrec, RustBV};

impl RustBV {
    // =========================================================================
    // Z3 Integration (when feature is enabled)
    // =========================================================================

    /// Convert this RustBV to a Z3 AST.
    ///
    /// For Expression nodes, the Z3 AST is computed lazily by recursively
    /// building from the operation tree. This avoids creating Z3 ASTs for
    /// intermediate results that never become constraints (matching claripy's
    /// lazy evaluation approach).
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_ast(&self) -> z3::ast::BV {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_ast_cached(&mut cache)
    }

    /// Build Z3 AST with caching to avoid exponential blowup on DAG expressions.
    ///
    /// Expression trees built from symbolic memory stores (ITE chains) often share
    /// sub-expressions via Arc. Without caching, to_z3_ast() traverses the DAG as
    /// a tree, rebuilding shared subtrees exponentially. This version caches by
    /// Arc pointer identity, ensuring each unique sub-expression is built once.
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_ast_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        // For Expression nodes, use pointer identity as cache key
        // (Arc-shared sub-expressions will have the same pointer)
        let cache_key = self as *const RustBV as usize;
        if let Some(cached) = cache.get(&cache_key) {
            super::stats::record_z3_ast_cache_hit();
            return cached.clone();
        }
        super::stats::record_z3_ast_cache_miss();
        let result = match self {
            RustBV::Concrete { value, width } => super::bv_codec::make_bv_const(*value, *width),
            RustBV::Symbolic { ast, .. } => ast.clone(),
            RustBV::Constrained { value, width, .. } => {
                super::bv_codec::make_bv_const(*value, *width)
            }
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => Self::build_z3_ast_cached(op, operands, *width, cache),
        };
        cache.insert(cache_key, result.clone());
        result
    }

    /// Convert a 1-bit RustBV to a native Z3 Bool, avoiding ITE wrapping.
    ///
    /// For comparison ops (Eq, Ne, Ult, etc.), produces the native Z3 Bool
    /// directly instead of going through ITE(cmp, BV(1,1), BV(0,1)) then
    /// `._eq(BV(1,1))`. Saves 3 Z3 AST nodes per comparison constraint.
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_bool(&self) -> z3::ast::Bool {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_bool_cached(&mut cache)
    }

    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_bool_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::Bool {
        match self {
            RustBV::Concrete { value, .. } => z3::ast::Bool::from_bool(*value != 0),
            RustBV::Constrained { value, .. } => z3::ast::Bool::from_bool(*value != 0),
            RustBV::Expression { op, operands, .. } => {
                match op {
                    BVOp::Eq => operands[0]
                        .to_z3_ast_cached(cache)
                        .eq(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ne => operands[0]
                        .to_z3_ast_cached(cache)
                        .eq(operands[1].to_z3_ast_cached(cache))
                        .not(),
                    BVOp::Ult => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvult(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ule => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvule(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ugt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvugt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Uge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvuge(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Slt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvslt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sle => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsle(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sgt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsgt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsge(operands[1].to_z3_ast_cached(cache)),
                    // Not() on a 1-bit comparison: negate the inner bool
                    BVOp::Not if operands.len() == 1 => operands[0].to_z3_bool_cached(cache).not(),
                    // Fallback: convert BV to Bool via _eq(1)
                    _ => {
                        let ast = self.to_z3_ast_cached(cache);
                        let one = z3::ast::BV::from_u64(1, 1);
                        ast.eq(&one)
                    }
                }
            }
            // Symbolic BV — use _eq(1)
            _ => {
                let ast = self.to_z3_ast_cached(cache);
                let one = z3::ast::BV::from_u64(1, 1);
                ast.eq(&one)
            }
        }
    }

    /// Emit a Z3 AST for `Extract(high, low, inner)` while re-applying the
    /// canonicalization rules from `extract_into` at Z3-emission time.
    ///
    /// `extract_into` only fires at construction time. Extract nodes built via
    /// `truncate_into` or `extract_no_ctx` bypass those rules, and so do Extract
    /// nodes whose inner shape was rewritten *after* the Extract was created.
    /// This walks the inner operand and distributes the Extract through
    /// Concat/Reverse/ZeroExt/SignExt/Extract patterns before handing anything
    /// to Z3, which avoids emitting intermediate Z3 ASTs that Z3's bv_rewriter
    /// would have to simplify (and, in the Reverse case, often can't — Z3 has
    /// no native Reverse, so the Concat-of-Extracts encoding survives).
    #[cfg(feature = "vex-engine-z3")]
    fn emit_extract_z3_cached(
        inner: &RustBV,
        high: u32,
        low: u32,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        debug_assert!(high >= low);
        debug_assert!(high < inner.width());
        let result_width = high - low + 1;

        // Identity: Extract(width-1, 0, x) → x
        if high == inner.width() - 1 && low == 0 {
            return inner.to_z3_ast_cached(cache);
        }

        // Concrete fast path — fold the extraction at the Rust level so Z3
        // never sees Extract over a literal.
        if let Some(v) = inner.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            return super::bv_codec::make_bv_const(extracted, result_width);
        }

        if let RustBV::Expression { op, operands, .. } = inner {
            match op {
                // Rule 1: Extract(h2, l2, Extract(h1, l1, x)) → Extract(l1+h2, l1+l2, x)
                BVOp::Extract(_, inner_low) => {
                    return Self::emit_extract_z3_cached(
                        &operands[0],
                        inner_low + high,
                        inner_low + low,
                        cache,
                    );
                }

                // Rule 2: Extract(Concat(a, b)) → distribute to the relevant part(s)
                BVOp::Concat if operands.len() == 2 => {
                    let b_width = operands[1].width();
                    if high < b_width {
                        return Self::emit_extract_z3_cached(&operands[1], high, low, cache);
                    } else if low >= b_width {
                        return Self::emit_extract_z3_cached(
                            &operands[0],
                            high - b_width,
                            low - b_width,
                            cache,
                        );
                    }
                    // Crosses boundary — extract from each part and concat.
                    let lo_part =
                        Self::emit_extract_z3_cached(&operands[1], b_width - 1, low, cache);
                    let hi_part =
                        Self::emit_extract_z3_cached(&operands[0], high - b_width, 0, cache);
                    return hi_part.concat(&lo_part);
                }

                // Rule 3: Extract(Reverse(x)) with byte-aligned bounds.
                // Single-byte: Reverse is a no-op, just extract the flipped byte from x.
                // Multi-byte: extract the matching byte range from x, then byte-reverse
                // (emitted as the canonical Concat-of-Extracts Z3 shape, matching
                // build_z3_ast_cached's BVOp::Reverse arm).
                BVOp::Reverse
                    if operands[0].width() % 8 == 0 && high % 8 == 7 && low.is_multiple_of(8) =>
                {
                    let w = operands[0].width();
                    let inner_ast = Self::emit_extract_z3_cached(
                        &operands[0],
                        w - 1 - low,
                        w - 1 - high,
                        cache,
                    );
                    if high - low + 1 == 8 {
                        return inner_ast;
                    }
                    let inner_w = high - low + 1;
                    let byte_count = inner_w / 8;
                    let parts: Vec<z3::ast::BV> = (0..byte_count)
                        .map(|i| inner_ast.extract(i * 8 + 7, i * 8))
                        .collect();
                    let mut result = parts[0].clone();
                    for part in &parts[1..] {
                        result = result.concat(part);
                    }
                    return result;
                }

                // Rule 4: Extract(ZeroExt(x)) — collapse to original or zero
                BVOp::ZeroExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return Self::emit_extract_z3_cached(&operands[0], high, low, cache);
                    } else if low >= inner_width {
                        return z3::ast::BV::from_u64(0, result_width);
                    }
                    // Straddles the extension boundary — fall through to the
                    // generic path. Don't try to split here: the existing
                    // build_z3_ast_cached emits ZeroExt as a concat of zero
                    // bits, so Z3's bv_rewriter already collapses
                    // Extract(zero_ext) cleanly.
                }

                // Rule 5: Extract(SignExt(x)) — collapse if entirely within original width
                BVOp::SignExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return Self::emit_extract_z3_cached(&operands[0], high, low, cache);
                    }
                    // Otherwise fall through — the SignExt encoding handles
                    // sign-bit propagation; bv_rewriter folds the Extract.
                }

                _ => {}
            }
        }

        // Default: build the inner Z3 AST and apply Extract.
        inner.to_z3_ast_cached(cache).extract(high, low)
    }

    /// Build Z3 AST with caching to avoid exponential blowup on DAG expressions.
    /// See `to_z3_ast_cached()` for rationale.
    #[cfg(feature = "vex-engine-z3")]
    fn build_z3_ast_cached(
        op: &BVOp,
        operands: &[RustBV],
        _width: u32,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        match op {
            // Arithmetic
            BVOp::Add => operands[0]
                .to_z3_ast_cached(cache)
                .bvadd(operands[1].to_z3_ast_cached(cache)),
            BVOp::Sub => operands[0]
                .to_z3_ast_cached(cache)
                .bvsub(operands[1].to_z3_ast_cached(cache)),
            BVOp::Mul => operands[0]
                .to_z3_ast_cached(cache)
                .bvmul(operands[1].to_z3_ast_cached(cache)),
            BVOp::UDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvudiv(operands[1].to_z3_ast_cached(cache)),
            BVOp::SDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvsdiv(operands[1].to_z3_ast_cached(cache)),
            BVOp::URem => operands[0]
                .to_z3_ast_cached(cache)
                .bvurem(operands[1].to_z3_ast_cached(cache)),
            BVOp::SRem => operands[0]
                .to_z3_ast_cached(cache)
                .bvsrem(operands[1].to_z3_ast_cached(cache)),
            BVOp::Neg => operands[0].to_z3_ast_cached(cache).bvneg(),

            // Bitwise
            BVOp::And => operands[0]
                .to_z3_ast_cached(cache)
                .bvand(operands[1].to_z3_ast_cached(cache)),
            BVOp::Or => operands[0]
                .to_z3_ast_cached(cache)
                .bvor(operands[1].to_z3_ast_cached(cache)),
            BVOp::Xor => operands[0]
                .to_z3_ast_cached(cache)
                .bvxor(operands[1].to_z3_ast_cached(cache)),
            BVOp::Not => operands[0].to_z3_ast_cached(cache).bvnot(),

            // Shifts
            BVOp::Shl => operands[0]
                .to_z3_ast_cached(cache)
                .bvshl(operands[1].to_z3_ast_cached(cache)),
            BVOp::Lshr => operands[0]
                .to_z3_ast_cached(cache)
                .bvlshr(operands[1].to_z3_ast_cached(cache)),
            BVOp::Ashr => operands[0]
                .to_z3_ast_cached(cache)
                .bvashr(operands[1].to_z3_ast_cached(cache)),
            BVOp::RotL => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotl(operands[1].to_z3_ast_cached(cache)),
            BVOp::RotR => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotr(operands[1].to_z3_ast_cached(cache)),

            // Comparisons (return 1-bit BV: If(cmp, BV(1,1), BV(0,1)))
            BVOp::Eq => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ne => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(operands[1].to_z3_ast_cached(cache))
                    .not();
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ult => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvult(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ule => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvule(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ugt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvugt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Uge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvuge(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Slt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvslt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sle => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsle(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sgt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsgt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsge(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }

            // Conversions
            BVOp::ZeroExt(bits) => operands[0].to_z3_ast_cached(cache).zero_ext(*bits),
            BVOp::SignExt(bits) => operands[0].to_z3_ast_cached(cache).sign_ext(*bits),
            BVOp::Extract(high, low) => {
                Self::emit_extract_z3_cached(&operands[0], *high, *low, cache)
            }
            BVOp::Concat => {
                fn collect_concat_leaves_cached(
                    bv: &RustBV,
                    leaves: &mut Vec<z3::ast::BV>,
                    cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
                ) {
                    if let RustBV::Expression {
                        op: BVOp::Concat,
                        operands,
                        ..
                    } = bv
                    {
                        collect_concat_leaves_cached(&operands[0], leaves, cache);
                        collect_concat_leaves_cached(&operands[1], leaves, cache);
                    } else {
                        leaves.push(bv.to_z3_ast_cached(cache));
                    }
                }
                let mut leaves = Vec::new();
                collect_concat_leaves_cached(&operands[0], &mut leaves, cache);
                collect_concat_leaves_cached(&operands[1], &mut leaves, cache);
                let mut result = leaves
                    .pop()
                    .expect("Concat operands always produce at least one leaf");
                while let Some(part) = leaves.pop() {
                    result = part.concat(&result);
                }
                result
            }

            // Conditional
            BVOp::Ite => {
                let cond = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(z3::ast::BV::from_u64(0, operands[0].width()))
                    .not();
                cond.ite(
                    &operands[1].to_z3_ast_cached(cache),
                    &operands[2].to_z3_ast_cached(cache),
                )
            }

            // Byte reverse
            BVOp::Reverse => {
                if let RustBV::Expression {
                    op: BVOp::Concat, ..
                } = &operands[0]
                {
                    fn collect_concat_parts_cached(bv: &RustBV, parts: &mut Vec<RustBV>) {
                        if let RustBV::Expression {
                            op: BVOp::Concat,
                            operands,
                            ..
                        } = bv
                        {
                            collect_concat_parts_cached(&operands[0], parts);
                            collect_concat_parts_cached(&operands[1], parts);
                        } else {
                            parts.push(bv.clone());
                        }
                    }
                    let mut parts = Vec::new();
                    collect_concat_parts_cached(&operands[0], &mut parts);
                    parts.reverse();
                    let reversed_asts: Vec<z3::ast::BV> = parts
                        .iter()
                        .map(|p| {
                            let w = p.width();
                            if w == 8 {
                                p.to_z3_ast_cached(cache)
                            } else {
                                Self::build_z3_ast_cached(
                                    &BVOp::Reverse,
                                    std::slice::from_ref(p),
                                    w,
                                    cache,
                                )
                            }
                        })
                        .collect();
                    let mut result = reversed_asts[0].clone();
                    for ast in &reversed_asts[1..] {
                        result = result.concat(ast);
                    }
                    return result;
                }

                let ast = operands[0].to_z3_ast_cached(cache);
                let w = operands[0].width();
                if w.is_multiple_of(8) && w >= 16 {
                    // Same canonical shape as the non-cached path above.
                    let bytes = w / 8;
                    let parts: Vec<z3::ast::BV> =
                        (0..bytes).map(|i| ast.extract(i * 8 + 7, i * 8)).collect();
                    let mut result = parts[0].clone();
                    for part in &parts[1..] {
                        result = result.concat(part);
                    }
                    result
                } else {
                    ast
                }
            }

            // Bit counting ops
            BVOp::Clz => z3::ast::BV::new_const("clz", _width),
            BVOp::Ctz => z3::ast::BV::new_const("ctz", _width),
            BVOp::Popcount => z3::ast::BV::new_const("popcount", _width),

            // Floating-point operations via Z3 FP theory.
            BVOp::Float { kind, prec } => {
                Self::build_fp_z3_ast_cached(*kind, *prec, operands, cache)
            }
        }
    }

    /// Build a Z3 AST for a symbolic float operation, returning the result
    /// encoded as an IEEE-754 bit-vector (or 1-bit BV for compares).
    ///
    /// Operands are RustBVs holding IEEE-754 bit patterns; we reinterpret
    /// them as Z3 Float values (`Z3_mk_fpa_to_fp_bv`), apply the FP op,
    /// and convert results back to IEEE bits via `to_ieee_bv`.
    ///
    /// Each intermediate Z3 ast is wrapped via `Ast::wrap` so its refcount
    /// is properly tracked — passing raw `Z3_ast` pointers to multiple FFI
    /// calls is unsafe because Z3's ref-counted contexts may GC the
    /// intermediate ASTs between calls.
    ///
    /// # SAFETY invariants for all `unsafe { z3_sys::… }` calls in this and
    /// the sibling `build_fp_*_cached` helpers
    ///
    /// 1. **Context validity**: `raw_ctx` is `z3::Context::thread_local()
    ///    .get_z3_context()`. The thread-local context lives for the
    ///    duration of the thread, so the handle is valid for the call.
    /// 2. **Pointer validity**: All `Z3_ast` operands are obtained from
    ///    `.get_z3_ast()` on live wrappers (`RustBV`, `Float`, `Bool`,
    ///    `RoundingMode`, `Sort`) bound in the same scope, so Z3 holds at
    ///    least one refcount on each for the duration of the FFI call.
    /// 3. **Null handling**: Every `Z3_mk_fpa_*` is followed by
    ///    `.expect(…)`, converting the only failure mode (NULL) into a
    ///    panic — so any pointer that escapes the `unsafe` block is
    ///    non-null and points at a fresh Z3 AST with one refcount.
    /// 4. **Refcount discipline**: Each fresh raw `Z3_ast` is immediately
    ///    handed to `Float::wrap` / `BV::wrap` / `Bool::wrap`, which takes
    ///    over the refcount Z3 added at construction. The wrapper is then
    ///    held in a local for the rest of its use. The wrap functions are
    ///    `unsafe` only because they require this caller-supplied refcount
    ///    discipline; no other invariant is needed.
    /// 5. **Thread-safety**: All ASTs in this helper live in the
    ///    thread-local context; nothing escapes the calling thread, so
    ///    Z3's per-context single-thread requirement is upheld.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_z3_ast_cached(
        kind: FloatOpKind,
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Bool, Float, RoundingMode};
        use z3_sys::{
            Z3_mk_fpa_abs, Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_eq, Z3_mk_fpa_fma,
            Z3_mk_fpa_is_nan, Z3_mk_fpa_leq, Z3_mk_fpa_lt, Z3_mk_fpa_mul, Z3_mk_fpa_neg,
            Z3_mk_fpa_sqrt, Z3_mk_fpa_sub, Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv,
        };

        // RoundToInt has a non-Float operand (the rm BV) and needs its own path.
        if let FloatOpKind::RoundToInt = kind {
            return Self::build_fp_round_to_int_cached(prec, operands, cache);
        }
        // FP conversions (FtoI, ItoF, FtoF) have non-uniform operand types
        // (BV vs FP, varying widths/sorts). Each routes to a dedicated helper
        // per the split-helper invariant for FloatOpKind metadata operands.
        match kind {
            FloatOpKind::ConvertItoF { src_bits, signed } => {
                return Self::build_fp_i_to_f_cached(prec, src_bits, signed, operands, cache);
            }
            FloatOpKind::ConvertFtoI { dst_bits, signed } => {
                return Self::build_fp_f_to_i_cached(
                    prec, dst_bits, signed, /*rm*/ None, operands, cache,
                );
            }
            FloatOpKind::ConvertFtoIRm { dst_bits, signed } => {
                return Self::build_fp_f_to_i_cached(
                    prec,
                    dst_bits,
                    signed,
                    Some(()),
                    operands,
                    cache,
                );
            }
            FloatOpKind::ConvertFtoF { src_prec } => {
                return Self::build_fp_f_to_f_cached(
                    prec, src_prec, /*has_rm*/ false, operands, cache,
                );
            }
            FloatOpKind::ConvertFtoFRm { src_prec } => {
                return Self::build_fp_f_to_f_cached(
                    prec, src_prec, /*has_rm*/ true, operands, cache,
                );
            }
            FloatOpKind::AddRm
            | FloatOpKind::SubRm
            | FloatOpKind::MulRm
            | FloatOpKind::DivRm
            | FloatOpKind::SqrtRm => {
                return Self::build_fp_arith_rm_cached(kind, prec, operands, cache);
            }
            _ => {}
        }

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert each BV operand to a Z3 Float wrapper. Holding the wrapper
        // (not just the raw Z3_ast) ensures the intermediate AST has its
        // refcount incremented and is not freed before we use it.
        let fp_args: Vec<Float> = operands
            .iter()
            .map(|bv| {
                let bv_ast = bv.to_z3_ast_cached(cache);
                // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
                // `bv_ast` is a live `BV` wrapper; `raw_sort` is a live
                // `Sort` handle. `Z3_mk_fpa_to_fp_bv` returns a fresh AST
                // (or NULL → panic via `.expect`).
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, bv_ast.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
                // SAFETY: `raw` is a fresh non-null Z3_ast in `z3_ctx`
                // with one refcount held by Z3; `Float::wrap` takes it.
                unsafe { Float::wrap(&z3_ctx, raw) }
            })
            .collect();

        // Round-to-nearest-ties-to-even (IEEE-754 default).
        let rm = RoundingMode::round_nearest_ties_to_even();
        let rm_raw = rm.get_z3_ast();

        // Apply the operation. Wrap the result as Float (or Bool) so its
        // refcount is held until we convert it.
        let raw_a = fp_args[0].get_z3_ast();
        // Helper to safely get raw_b / raw_c only when needed (some ops are unary).
        let raw_b = || fp_args[1].get_z3_ast();
        let raw_c = || fp_args[2].get_z3_ast();

        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `rm_raw` is held by the `rm: RoundingMode` wrapper; `raw_a`,
        // `raw_b()`, `raw_c()` come from the `fp_args` Float wrappers and
        // remain live for the duration of this match. Every `Z3_mk_fpa_*`
        // returns a fresh AST (or NULL → panic via `.expect`).
        let result_raw = unsafe {
            match kind {
                FloatOpKind::Add => Z3_mk_fpa_add(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Sub => Z3_mk_fpa_sub(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Mul => Z3_mk_fpa_mul(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Div => Z3_mk_fpa_div(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Sqrt => Z3_mk_fpa_sqrt(raw_ctx, rm_raw, raw_a),
                FloatOpKind::Neg => Z3_mk_fpa_neg(raw_ctx, raw_a),
                FloatOpKind::Abs => Z3_mk_fpa_abs(raw_ctx, raw_a),
                FloatOpKind::Fma => Z3_mk_fpa_fma(raw_ctx, rm_raw, raw_a, raw_b(), raw_c()),
                FloatOpKind::Fms => {
                    // a*b - c == a*b + (-c). Wrap neg_c in a Float so the
                    // intermediate AST is held while we build the FMA.
                    let neg_c_raw =
                        Z3_mk_fpa_neg(raw_ctx, raw_c()).expect("Z3_mk_fpa_neg returned NULL");
                    let neg_c = Float::wrap(&z3_ctx, neg_c_raw);
                    Z3_mk_fpa_fma(raw_ctx, rm_raw, raw_a, raw_b(), neg_c.get_z3_ast())
                }
                FloatOpKind::CmpEq => Z3_mk_fpa_eq(raw_ctx, raw_a, raw_b()),
                FloatOpKind::CmpLt => Z3_mk_fpa_lt(raw_ctx, raw_a, raw_b()),
                FloatOpKind::CmpLe => Z3_mk_fpa_leq(raw_ctx, raw_a, raw_b()),
                FloatOpKind::IsNaN => Z3_mk_fpa_is_nan(raw_ctx, raw_a),
                FloatOpKind::RoundToInt
                | FloatOpKind::ConvertItoF { .. }
                | FloatOpKind::ConvertFtoI { .. }
                | FloatOpKind::ConvertFtoIRm { .. }
                | FloatOpKind::ConvertFtoF { .. }
                | FloatOpKind::ConvertFtoFRm { .. }
                | FloatOpKind::AddRm
                | FloatOpKind::SubRm
                | FloatOpKind::MulRm
                | FloatOpKind::DivRm
                | FloatOpKind::SqrtRm => unreachable!("handled above"),
            }
            .expect("Z3 FPA op returned NULL")
        };

        if kind.is_compare() {
            // SAFETY: `result_raw` is the fresh Z3 Bool produced by the
            // comparison op above (one refcount held by Z3); `Bool::wrap`
            // takes that refcount. Same context as the wrapper.
            let cmp_bool = unsafe { Bool::wrap(&z3_ctx, result_raw) };
            cmp_bool.ite(&BV::from_u64(1, 1), &BV::from_u64(0, 1))
        } else {
            // SAFETY: `result_raw` is the fresh Z3 Float produced above;
            // `Float::wrap` takes its refcount.
            let result_fp = unsafe { Float::wrap(&z3_ctx, result_raw) };
            // SAFETY: `result_fp` is a live Float in `z3_ctx`;
            // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL →
            // panic via `.expect`).
            let ieee_bv_raw = unsafe {
                Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                    .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
            };
            // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call
            // above; `BV::wrap` takes its refcount.
            unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
        }
    }

    /// Build the Z3 AST for a `FloatOpKind::RoundToInt` operation.
    ///
    /// VEX rounding modes (low 2 bits of operand\[0\]):
    ///   0 = nearest (ties to even), 1 = -inf, 2 = +inf, 3 = zero (truncate).
    ///
    /// Concrete rm: pick the matching Z3 RoundingMode and call
    /// `Z3_mk_fpa_round_to_integral` once. Symbolic rm: build all four
    /// variants and ITE on the rm low-bits — Z3 simplifies away dead arms
    /// at solve time.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_round_to_int_cached(
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_round_to_integral, Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv};

        debug_assert_eq!(operands.len(), 2);
        let rm_bv = &operands[0];
        let value_bv = &operands[1];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert the value BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort
        // handle; `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose refcount
        // `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        // Helper: round value with one concrete VEX rounding mode (0..3).
        let round_with = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by the `value_fp` wrapper above.
            // `Z3_mk_fpa_round_to_integral` returns a fresh AST (or NULL →
            // panic via `.expect`).
            let raw = unsafe {
                Z3_mk_fpa_round_to_integral(raw_ctx, rm.get_z3_ast(), value_raw)
                    .expect("Z3_mk_fpa_round_to_integral returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            round_with((m & 0x3) as u8)
        } else {
            // Symbolic rm: build all 4 results and ITE on rm[1:0].
            let r0 = round_with(0);
            let r1 = round_with(1);
            let r2 = round_with(2);
            let r3 = round_with(3);
            let rm_z3 = rm_bv.to_z3_ast_cached(cache);
            let rm_low2 = rm_z3.extract(1, 0);
            let zero = BV::from_u64(0, 2);
            let one = BV::from_u64(1, 2);
            let two = BV::from_u64(2, 2);
            // Chain: rm==0 ? r0 : rm==1 ? r1 : rm==2 ? r2 : r3
            let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
            let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
            rm_low2.eq(&zero).ite(&r0, &pick123)
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for an FP arithmetic op with explicit rounding mode
    /// (`AddRm`/`SubRm`/`MulRm`/`DivRm`/`SqrtRm`). operand\[0\] is the rm BV;
    /// remaining operands are the FP operands. Concrete rm picks one Z3
    /// `RoundingMode`; symbolic rm builds all four variants and ITEs on the
    /// rm low-2-bits — Z3 simplifies away dead arms at solve time. Mirrors
    /// the pattern in `build_fp_round_to_int_cached`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_arith_rm_cached(
        kind: FloatOpKind,
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{
            Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_mul, Z3_mk_fpa_sqrt, Z3_mk_fpa_sub,
            Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv,
        };

        let is_unary = matches!(kind, FloatOpKind::SqrtRm);
        debug_assert_eq!(operands.len(), if is_unary { 2 } else { 3 });
        let rm_bv = &operands[0];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert FP operand BVs to Z3 Float wrappers; keep them alive across
        // the helper closure since intermediate ASTs may be GC'd otherwise.
        let to_fp =
            |bv: &RustBV, cache: &mut std::collections::HashMap<usize, z3::ast::BV>| -> Float {
                let z3 = bv.to_z3_ast_cached(cache);
                // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
                // `z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort.
                // `Z3_mk_fpa_to_fp_bv` returns a fresh AST (or NULL →
                // panic via `.expect`).
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, z3.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
                // SAFETY: `raw` is the fresh Float AST from above;
                // `Float::wrap` takes its refcount.
                unsafe { Float::wrap(&z3_ctx, raw) }
            };
        let a_fp = to_fp(&operands[1], cache);
        let b_fp = if is_unary {
            None
        } else {
            Some(to_fp(&operands[2], cache))
        };

        let apply = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            let rm_raw = rm.get_z3_ast();
            let raw_a = a_fp.get_z3_ast();
            // SAFETY: `rm` (RoundingMode) and `a_fp` / `b_fp` (Float) are
            // live wrappers in `z3_ctx` for the duration of this closure
            // body, so `rm_raw`, `raw_a` and the `get_z3_ast()` calls on
            // `b_fp` all yield valid Z3_ast pointers. Each `Z3_mk_fpa_*`
            // returns a fresh AST (or NULL → panic via `.expect`).
            let raw = unsafe {
                match kind {
                    FloatOpKind::AddRm => {
                        Z3_mk_fpa_add(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::SubRm => {
                        Z3_mk_fpa_sub(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::MulRm => {
                        Z3_mk_fpa_mul(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::DivRm => {
                        Z3_mk_fpa_div(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::SqrtRm => Z3_mk_fpa_sqrt(raw_ctx, rm_raw, raw_a),
                    _ => unreachable!("non-Rm FP arith kind in build_fp_arith_rm_cached"),
                }
                .expect("Z3 FPA arith op returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            apply((m & 0x3) as u8)
        } else {
            // Symbolic rm: build all 4 results and ITE on rm[1:0]. Z3 folds
            // dead arms during simplification.
            let r0 = apply(0);
            let r1 = apply(1);
            let r2 = apply(2);
            let r3 = apply(3);
            let rm_z3 = rm_bv.to_z3_ast_cached(cache);
            let rm_low2 = rm_z3.extract(1, 0);
            let zero = BV::from_u64(0, 2);
            let one = BV::from_u64(1, 2);
            let two = BV::from_u64(2, 2);
            let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
            let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
            rm_low2.eq(&zero).ite(&r0, &pick123)
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertItoF`. operand\[0\] is a BV
    /// of width `src_bits` interpreted as signed/unsigned per `signed`.
    /// Result is the IEEE bits of the FP at `prec`. RNE rounding.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_i_to_f_cached(
        prec: FloatPrec,
        src_bits: u8,
        signed: bool,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_signed, Z3_mk_fpa_to_fp_unsigned, Z3_mk_fpa_to_ieee_bv};

        debug_assert_eq!(operands.len(), 1);
        let src_bv = &operands[0];
        debug_assert_eq!(src_bv.width(), src_bits as u32);

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        let src_z3 = src_bv.to_z3_ast_cached(cache);
        let rm = RoundingMode::round_nearest_ties_to_even();
        let rm_raw = rm.get_z3_ast();

        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `rm_raw` is held by `rm: RoundingMode`; `src_z3` is a live BV
        // wrapper; `raw_sort` is a live Sort. The signed/unsigned
        // `Z3_mk_fpa_to_fp_*` calls return a fresh Float AST (or NULL →
        // panic via `.expect`).
        let fp_raw = unsafe {
            if signed {
                Z3_mk_fpa_to_fp_signed(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_signed returned NULL")
            } else {
                Z3_mk_fpa_to_fp_unsigned(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_unsigned returned NULL")
            }
        };
        // SAFETY: `fp_raw` is the fresh Float AST from the call above;
        // `Float::wrap` takes its refcount.
        let fp_wrap = unsafe { Float::wrap(&z3_ctx, fp_raw) };
        // SAFETY: `fp_wrap` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, fp_wrap.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertFtoI` (no rm operand,
    /// implicit RNE) or `FloatOpKind::ConvertFtoIRm` (operand\[0\] = rm BV,
    /// operand\[1\] = FP value). The result is a BV of width `dst_bits`,
    /// signed or unsigned 2's complement per `signed`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_f_to_i_cached(
        prec: FloatPrec,
        dst_bits: u8,
        signed: bool,
        rm_marker: Option<()>,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_sbv, Z3_mk_fpa_to_ubv};

        let (rm_bv_opt, value_bv) = if rm_marker.is_some() {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert the operand BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort.
        // `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose refcount
        // `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        // Helper: convert the value to BV using one concrete VEX rounding mode (0..3).
        let convert_with = |vex_rm: u8| -> BV {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above. The signed/unsigned
            // `Z3_mk_fpa_to_*bv` calls return a fresh BV AST (or NULL →
            // panic via `.expect`).
            let raw = unsafe {
                if signed {
                    Z3_mk_fpa_to_sbv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_sbv returned NULL")
                } else {
                    Z3_mk_fpa_to_ubv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_ubv returned NULL")
                }
            };
            // SAFETY: `raw` is the fresh BV AST from the call above;
            // `BV::wrap` takes its refcount.
            unsafe { BV::wrap(&z3_ctx, raw) }
        };

        match rm_bv_opt {
            // Implicit RNE.
            None => convert_with(0),
            // Explicit rm; if symbolic, ITE over the four cases.
            Some(rm_bv) => {
                if let Some(m) = rm_bv.as_u128() {
                    convert_with((m & 0x3) as u8)
                } else {
                    let r0 = convert_with(0);
                    let r1 = convert_with(1);
                    let r2 = convert_with(2);
                    let r3 = convert_with(3);
                    let rm_z3 = rm_bv.to_z3_ast_cached(cache);
                    let rm_low2 = rm_z3.extract(1, 0);
                    let zero = BV::from_u64(0, 2);
                    let one = BV::from_u64(1, 2);
                    let two = BV::from_u64(2, 2);
                    let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
                    let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
                    rm_low2.eq(&zero).ite(&r0, &pick123)
                }
            }
        }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertFtoF` (no rm) or
    /// `FloatOpKind::ConvertFtoFRm` (operand\[0\] = rm BV, operand\[1\] = FP).
    /// Source FP is at `src_prec`, destination FP is at `prec`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_f_to_f_cached(
        prec: FloatPrec,
        src_prec: FloatPrec,
        has_rm: bool,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_fp_float, Z3_mk_fpa_to_ieee_bv};

        let (rm_bv_opt, value_bv) = if has_rm {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let src_sort = match src_prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let dst_sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let src_raw_sort = src_sort.get_z3_sort();
        let dst_raw_sort = dst_sort.get_z3_sort();

        // Convert the source operand BV to a Z3 Float at src_prec.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `src_raw_sort` is a live
        // Sort handle. `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose
        // refcount `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), src_raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        let convert_with = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above; `dst_raw_sort` is a live
            // Sort handle. `Z3_mk_fpa_to_fp_float` returns a fresh Float
            // AST (or NULL → panic via `.expect`).
            let raw = unsafe {
                Z3_mk_fpa_to_fp_float(raw_ctx, rm.get_z3_ast(), value_raw, dst_raw_sort)
                    .expect("Z3_mk_fpa_to_fp_float returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = match rm_bv_opt {
            None => convert_with(0),
            Some(rm_bv) => {
                if let Some(m) = rm_bv.as_u128() {
                    convert_with((m & 0x3) as u8)
                } else {
                    let r0 = convert_with(0);
                    let r1 = convert_with(1);
                    let r2 = convert_with(2);
                    let r3 = convert_with(3);
                    let rm_z3 = rm_bv.to_z3_ast_cached(cache);
                    let rm_low2 = rm_z3.extract(1, 0);
                    let zero = BV::from_u64(0, 2);
                    let one = BV::from_u64(1, 2);
                    let two = BV::from_u64(2, 2);
                    let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
                    let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
                    rm_low2.eq(&zero).ite(&r0, &pick123)
                }
            }
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }
}
