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
            RustBV::Concrete { value, width } | RustBV::Constrained { value, width, .. } => {
                super::bv_codec::make_bv_const(*value, *width)
            }
            RustBV::Symbolic { ast, .. } => ast.clone(),
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

    /// Map a comparison `BVOp` to its native Z3 `Bool`, building the two operand
    /// ASTs from `operands`. Returns `None` for any non-comparison op so callers
    /// can fall through to their own handling.
    ///
    /// Shared by `to_z3_bool_cached` (native Bool path) and `build_z3_ast_cached`
    /// (which wraps the Bool in `If(cmp, BV(1,1), BV(0,1))`), so the op→bv*-method
    /// mapping lives in exactly one place.
    #[cfg(feature = "vex-engine-z3")]
    fn cmp_bool_for_cached(
        op: &BVOp,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> Option<z3::ast::Bool> {
        match op {
            BVOp::Eq
            | BVOp::Ne
            | BVOp::Ult
            | BVOp::Ule
            | BVOp::Ugt
            | BVOp::Uge
            | BVOp::Slt
            | BVOp::Sle
            | BVOp::Sgt
            | BVOp::Sge => {}
            _ => return None,
        }
        let a = operands[0].to_z3_ast_cached(cache);
        let b = operands[1].to_z3_ast_cached(cache);
        Some(match op {
            BVOp::Eq => a.eq(b),
            BVOp::Ne => a.eq(b).not(),
            BVOp::Ult => a.bvult(b),
            BVOp::Ule => a.bvule(b),
            BVOp::Ugt => a.bvugt(b),
            BVOp::Uge => a.bvuge(b),
            BVOp::Slt => a.bvslt(b),
            BVOp::Sle => a.bvsle(b),
            BVOp::Sgt => a.bvsgt(b),
            BVOp::Sge => a.bvsge(b),
            _ => unreachable!("guarded by the comparison-op match above"),
        })
    }

    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_bool_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::Bool {
        match self {
            RustBV::Concrete { value, .. } | RustBV::Constrained { value, .. } => {
                z3::ast::Bool::from_bool(*value != 0)
            }
            RustBV::Expression { op, operands, .. } => {
                if let Some(b) = Self::cmp_bool_for_cached(op, operands, cache) {
                    return b;
                }
                match op {
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
        // Plain binary BVOps differ only in the Z3 bv* method name; the
        // signed/unsigned distinction is fully encoded in the method itself.
        macro_rules! z3_binop {
            ($m:ident) => {
                operands[0]
                    .to_z3_ast_cached(cache)
                    .$m(operands[1].to_z3_ast_cached(cache))
            };
        }
        match op {
            // Arithmetic
            BVOp::Add => z3_binop!(bvadd),
            BVOp::Sub => z3_binop!(bvsub),
            BVOp::Mul => z3_binop!(bvmul),
            BVOp::UDiv => z3_binop!(bvudiv),
            BVOp::SDiv => z3_binop!(bvsdiv),
            BVOp::URem => z3_binop!(bvurem),
            BVOp::SRem => z3_binop!(bvsrem),
            BVOp::Neg => operands[0].to_z3_ast_cached(cache).bvneg(),

            // Bitwise
            BVOp::And => z3_binop!(bvand),
            BVOp::Or => z3_binop!(bvor),
            BVOp::Xor => z3_binop!(bvxor),
            BVOp::Not => operands[0].to_z3_ast_cached(cache).bvnot(),

            // Shifts
            BVOp::Shl => z3_binop!(bvshl),
            BVOp::Lshr => z3_binop!(bvlshr),
            BVOp::Ashr => z3_binop!(bvashr),
            BVOp::RotL => z3_binop!(bvrotl),
            BVOp::RotR => z3_binop!(bvrotr),

            // Comparisons (return 1-bit BV: If(cmp, BV(1,1), BV(0,1)))
            BVOp::Eq
            | BVOp::Ne
            | BVOp::Ult
            | BVOp::Ule
            | BVOp::Ugt
            | BVOp::Uge
            | BVOp::Slt
            | BVOp::Sle
            | BVOp::Sgt
            | BVOp::Sge => {
                let cmp = Self::cmp_bool_for_cached(op, operands, cache)
                    .expect("comparison op handled by cmp_bool_for_cached");
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
    ///    `.unwrap_or_else(|| fresh_unconstrained_raw(…))`, replacing the
    ///    only failure mode (NULL — reachable only on a type-checker-precluded
    ///    sort mismatch or OOM) with a fresh unconstrained AST of the matching
    ///    sort — so any pointer that escapes the `unsafe` block is non-null
    ///    and points at a fresh Z3 AST with one refcount.
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
            Z3_mk_bool_sort, Z3_mk_fpa_abs, Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_eq,
            Z3_mk_fpa_fma, Z3_mk_fpa_is_nan, Z3_mk_fpa_leq, Z3_mk_fpa_lt, Z3_mk_fpa_mul,
            Z3_mk_fpa_neg, Z3_mk_fpa_sqrt, Z3_mk_fpa_sub,
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
        let sort = prec.z3_sort();
        let raw_sort = sort.get_z3_sort();

        // Convert each BV operand to a Z3 Float wrapper. Holding the wrapper
        // (not just the raw Z3_ast) ensures the intermediate AST has its
        // refcount incremented and is not freed before we use it.
        let fp_args: Vec<Float> = operands
            .iter()
            .map(|bv| {
                let bv_ast = bv.to_z3_ast_cached(cache);
                bv_to_z3_float(&z3_ctx, raw_ctx, &bv_ast, raw_sort)
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
        // returns a fresh AST (or NULL → fresh unconstrained AST via `unwrap_or_else`).
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
                    let neg_c_raw = Z3_mk_fpa_neg(raw_ctx, raw_c())
                        .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort));
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
            .unwrap_or_else(|| {
                // Compares yield a Bool; every other op yields a Float at
                // `raw_sort`. Fabricate the matching sort so the downstream
                // wrap + tail produce a width-correct unconstrained result.
                let fallback_sort = if kind.is_compare() {
                    Z3_mk_bool_sort(raw_ctx).expect("Z3_mk_bool_sort returned NULL")
                } else {
                    raw_sort
                };
                fresh_unconstrained_raw(raw_ctx, fallback_sort)
            })
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
            float_to_ieee_bv(&z3_ctx, raw_ctx, &result_fp)
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
        use z3::ast::{Ast, Float};
        use z3_sys::Z3_mk_fpa_round_to_integral;

        debug_assert_eq!(operands.len(), 2);
        let rm_bv = &operands[0];
        let value_bv = &operands[1];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = prec.z3_sort();
        let raw_sort = sort.get_z3_sort();

        // Convert the value BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        let value_fp = bv_to_z3_float(&z3_ctx, raw_ctx, &value_z3, raw_sort);
        let value_raw = value_fp.get_z3_ast();

        // Helper: round value with one concrete VEX rounding mode (0..3).
        let round_with = |vex_rm: u8| -> Float {
            let rm = vex_rm_to_z3(vex_rm);
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by the `value_fp` wrapper above.
            // `Z3_mk_fpa_round_to_integral` returns a fresh AST (or NULL →
            // panic via `.expect`).
            let raw = unsafe {
                Z3_mk_fpa_round_to_integral(raw_ctx, rm.get_z3_ast(), value_raw)
                    .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort))
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            round_with((m & 0x3) as u8)
        } else {
            dispatch_symbolic_rm(rm_bv, cache, round_with)
        };

        float_to_ieee_bv(&z3_ctx, raw_ctx, &result_fp)
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
        use z3::ast::{Ast, Float};
        use z3_sys::{Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_mul, Z3_mk_fpa_sqrt, Z3_mk_fpa_sub};

        let is_unary = matches!(kind, FloatOpKind::SqrtRm);
        debug_assert_eq!(operands.len(), if is_unary { 2 } else { 3 });
        let rm_bv = &operands[0];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = prec.z3_sort();
        let raw_sort = sort.get_z3_sort();

        // Convert FP operand BVs to Z3 Float wrappers; keep them alive across
        // the helper closure since intermediate ASTs may be GC'd otherwise.
        let to_fp =
            |bv: &RustBV, cache: &mut std::collections::HashMap<usize, z3::ast::BV>| -> Float {
                let z3 = bv.to_z3_ast_cached(cache);
                bv_to_z3_float(&z3_ctx, raw_ctx, &z3, raw_sort)
            };
        let a_fp = to_fp(&operands[1], cache);
        let b_fp = if is_unary {
            None
        } else {
            Some(to_fp(&operands[2], cache))
        };

        let apply = |vex_rm: u8| -> Float {
            let rm = vex_rm_to_z3(vex_rm);
            let rm_raw = rm.get_z3_ast();
            let raw_a = a_fp.get_z3_ast();
            // SAFETY: `rm` (RoundingMode) and `a_fp` / `b_fp` (Float) are
            // live wrappers in `z3_ctx` for the duration of this closure
            // body, so `rm_raw`, `raw_a` and the `get_z3_ast()` calls on
            // `b_fp` all yield valid Z3_ast pointers. Each `Z3_mk_fpa_*`
            // returns a fresh AST (or NULL → fresh unconstrained AST via `unwrap_or_else`).
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
                .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort))
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            apply((m & 0x3) as u8)
        } else {
            dispatch_symbolic_rm(rm_bv, cache, apply)
        };

        float_to_ieee_bv(&z3_ctx, raw_ctx, &result_fp)
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
        use z3::ast::{Ast, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_signed, Z3_mk_fpa_to_fp_unsigned};

        debug_assert_eq!(operands.len(), 1);
        let src_bv = &operands[0];
        debug_assert_eq!(src_bv.width(), src_bits as u32);

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = prec.z3_sort();
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
                    .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort))
            } else {
                Z3_mk_fpa_to_fp_unsigned(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort))
            }
        };
        // SAFETY: `fp_raw` is the fresh Float AST from the call above;
        // `Float::wrap` takes its refcount.
        let fp_wrap = unsafe { Float::wrap(&z3_ctx, fp_raw) };
        float_to_ieee_bv(&z3_ctx, raw_ctx, &fp_wrap)
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
        use z3::ast::{Ast, BV};
        use z3_sys::{Z3_mk_bv_sort, Z3_mk_fpa_to_sbv, Z3_mk_fpa_to_ubv};

        let (rm_bv_opt, value_bv) = if rm_marker.is_some() {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = prec.z3_sort();
        let raw_sort = sort.get_z3_sort();

        // Convert the operand BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        let value_fp = bv_to_z3_float(&z3_ctx, raw_ctx, &value_z3, raw_sort);
        let value_raw = value_fp.get_z3_ast();

        // Helper: convert the value to BV using one concrete VEX rounding mode (0..3).
        let convert_with = |vex_rm: u8| -> BV {
            let rm = vex_rm_to_z3(vex_rm);
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above. The signed/unsigned
            // `Z3_mk_fpa_to_*bv` calls return a fresh BV AST (or NULL →
            // panic via `.expect`).
            // Result is a BV of `dst_bits`; fabricate that sort on the NULL
            // fallback path so the wrapped value keeps the correct width.
            let bv_fallback = || unsafe {
                let bv_sort =
                    Z3_mk_bv_sort(raw_ctx, dst_bits as u32).expect("Z3_mk_bv_sort returned NULL");
                fresh_unconstrained_raw(raw_ctx, bv_sort)
            };
            let raw = unsafe {
                if signed {
                    Z3_mk_fpa_to_sbv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .unwrap_or_else(bv_fallback)
                } else {
                    Z3_mk_fpa_to_ubv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .unwrap_or_else(bv_fallback)
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
                    dispatch_symbolic_rm(rm_bv, cache, convert_with)
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
        use z3::ast::{Ast, Float};
        use z3_sys::Z3_mk_fpa_to_fp_float;

        let (rm_bv_opt, value_bv) = if has_rm {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let src_sort = src_prec.z3_sort();
        let dst_sort = prec.z3_sort();
        let src_raw_sort = src_sort.get_z3_sort();
        let dst_raw_sort = dst_sort.get_z3_sort();

        // Convert the source operand BV to a Z3 Float at src_prec.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        let value_fp = bv_to_z3_float(&z3_ctx, raw_ctx, &value_z3, src_raw_sort);
        let value_raw = value_fp.get_z3_ast();

        let convert_with = |vex_rm: u8| -> Float {
            let rm = vex_rm_to_z3(vex_rm);
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above; `dst_raw_sort` is a live
            // Sort handle. `Z3_mk_fpa_to_fp_float` returns a fresh Float
            // AST (or NULL → fresh unconstrained AST via `unwrap_or_else`).
            let raw = unsafe {
                Z3_mk_fpa_to_fp_float(raw_ctx, rm.get_z3_ast(), value_raw, dst_raw_sort)
                    .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, dst_raw_sort))
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
                    dispatch_symbolic_rm(rm_bv, cache, convert_with)
                }
            }
        };

        float_to_ieee_bv(&z3_ctx, raw_ctx, &result_fp)
    }
}

/// Map a 2-bit VEX rounding-mode selector (0..3) to the corresponding Z3
/// `RoundingMode`. Centralizes the 4-way match repeated by every
/// rm-aware FP builder (round-to-int / arith-rm / f-to-i / f-to-f).
#[cfg(feature = "vex-engine-z3")]
fn vex_rm_to_z3(vex_rm: u8) -> z3::ast::RoundingMode {
    use z3::ast::RoundingMode;
    match vex_rm & 0x3 {
        0 => RoundingMode::round_nearest_ties_to_even(),
        1 => RoundingMode::round_towards_negative(),
        2 => RoundingMode::round_towards_positive(),
        3 => RoundingMode::round_towards_zero(),
        _ => unreachable!(),
    }
}

/// Build the symbolic-rounding-mode ITE fan-out shared by the rm-aware FP
/// builders: evaluate `build` for all four concrete VEX rounding modes,
/// then select on `rm_bv`'s low 2 bits via a nested `ite` chain
/// (`rm==0 ? r0 : rm==1 ? r1 : rm==2 ? r2 : r3`). Z3 folds the dead arms
/// away at solve time. Generic over the AST kind so the same fan-out
/// serves both the `Float`-producing builders and the `BV`-producing
/// f-to-i builder.
#[cfg(feature = "vex-engine-z3")]
fn dispatch_symbolic_rm<T, F>(
    rm_bv: &RustBV,
    cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    build: F,
) -> T
where
    T: z3::ast::Ast,
    F: Fn(u8) -> T,
{
    use z3::ast::BV;
    let r0 = build(0);
    let r1 = build(1);
    let r2 = build(2);
    let r3 = build(3);
    let rm_z3 = rm_bv.to_z3_ast_cached(cache);
    let rm_low2 = rm_z3.extract(1, 0);
    let zero = BV::from_u64(0, 2);
    let one = BV::from_u64(1, 2);
    let two = BV::from_u64(2, 2);
    let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
    let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
    rm_low2.eq(&zero).ite(&r0, &pick123)
}

/// Shared tail of the FP-op builders that return IEEE-754 bits: convert a
/// live Z3 `Float` to its bit pattern via `Z3_mk_fpa_to_ieee_bv` and wrap
/// the fresh BV AST, taking its refcount. `z3_ctx` / `raw_ctx` are the
/// thread-local context and its raw handle; `fp` must be a live Float in
/// `z3_ctx`.
/// Wrap a BV operand as a Z3 `Float` of sort `raw_sort` via
/// `Z3_mk_fpa_to_fp_bv`. Centralizes the unsafe make-fp-from-bits +
/// `Float::wrap` tail shared by every `build_fp_*` builder.
#[cfg(feature = "vex-engine-z3")]
fn bv_to_z3_float(
    z3_ctx: &z3::Context,
    raw_ctx: z3_sys::Z3_context,
    bv: &z3::ast::BV,
    raw_sort: z3_sys::Z3_sort,
) -> z3::ast::Float {
    use z3::ast::{Ast, Float};
    use z3_sys::Z3_mk_fpa_to_fp_bv;
    // SAFETY: `bv` is a live BV in `z3_ctx`; `raw_sort` is a live Sort
    // handle. `Z3_mk_fpa_to_fp_bv` returns a fresh AST (or NULL → panic
    // via `.expect`).
    let raw = unsafe {
        Z3_mk_fpa_to_fp_bv(raw_ctx, bv.get_z3_ast(), raw_sort)
            .unwrap_or_else(|| fresh_unconstrained_raw(raw_ctx, raw_sort))
    };
    // SAFETY: `raw` is the fresh non-null Float AST from above;
    // `Float::wrap` takes its refcount.
    unsafe { Float::wrap(z3_ctx, raw) }
}

#[cfg(feature = "vex-engine-z3")]
fn float_to_ieee_bv(
    z3_ctx: &z3::Context,
    raw_ctx: z3_sys::Z3_context,
    fp: &z3::ast::Float,
) -> z3::ast::BV {
    use z3::ast::{Ast, BV};
    use z3_sys::{
        Z3_fpa_get_ebits, Z3_fpa_get_sbits, Z3_get_sort, Z3_mk_bv_sort, Z3_mk_fpa_to_ieee_bv,
    };
    // SAFETY: `fp` is a live Float in `z3_ctx`. `Z3_mk_fpa_to_ieee_bv`
    // returns a fresh BV AST, or NULL on the (type-checker-precluded)
    // sort-mismatch / OOM path, where we fall back to a fresh unconstrained
    // BV of the matching IEEE width (sign + ebits + stored mantissa =
    // ebits + sbits) so the result keeps the correct width.
    let ieee_bv_raw = unsafe {
        Z3_mk_fpa_to_ieee_bv(raw_ctx, fp.get_z3_ast()).unwrap_or_else(|| {
            let fp_sort = Z3_get_sort(raw_ctx, fp.get_z3_ast()).expect("Z3_get_sort returned NULL");
            let width = Z3_fpa_get_ebits(raw_ctx, fp_sort) + Z3_fpa_get_sbits(raw_ctx, fp_sort);
            let bv_sort = Z3_mk_bv_sort(raw_ctx, width).expect("Z3_mk_bv_sort returned NULL");
            fresh_unconstrained_raw(raw_ctx, bv_sort)
        })
    };
    // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
    // `BV::wrap` takes its refcount.
    unsafe { BV::wrap(z3_ctx, ieee_bv_raw) }
}

/// Fabricate a fresh unconstrained Z3 AST of `raw_sort` as a graceful
/// fallback when a `Z3_mk_fpa_*` builder returns NULL.
///
/// The Z3 FPA builders return NULL only on a sort/width mismatch or on true
/// allocation failure. Operand widths flow from VEX `IRType`, which pyvex's
/// type-checker is expected to keep consistent, so the mismatch case should
/// be unreachable — this is defense-in-depth. Because the crate is built with
/// `panic = "abort"` (see `Cargo.toml`), an `.expect` here would tear down the
/// whole process and could not be caught by any downstream `catch_unwind`.
/// Degrading to a fresh unconstrained value of the *correct sort* instead
/// keeps the public `to_z3_ast` / `to_z3_ast_cached` chain panic-free while
/// letting the normal IEEE-bits tail (`float_to_ieee_bv`) produce a
/// width-correct result.
///
/// # Safety
/// `raw_ctx` and `raw_sort` must be live handles in the thread-local Z3
/// context. The returned raw `Z3_ast` carries one refcount and must be
/// handed to a `*::wrap` constructor by the caller, exactly like the
/// non-fallback path.
#[cfg(feature = "vex-engine-z3")]
unsafe fn fresh_unconstrained_raw(
    raw_ctx: z3_sys::Z3_context,
    raw_sort: z3_sys::Z3_sort,
) -> z3_sys::Z3_ast {
    use z3_sys::Z3_mk_fresh_const;
    let prefix: z3_sys::Z3_string = c"fpfb".as_ptr().cast();
    // SAFETY: caller upholds the live-handle contract above.
    // `Z3_mk_fresh_const` returns NULL only on genuine allocation failure,
    // where aborting is the only sane outcome.
    unsafe {
        Z3_mk_fresh_const(raw_ctx, prefix, raw_sort)
            .expect("Z3_mk_fresh_const returned NULL (out of memory)")
    }
}
