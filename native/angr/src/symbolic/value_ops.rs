//! Bitvector arithmetic / bitwise / comparison / structural ops for [`RustBV`].
//!
//! Slice .2 of the `symbolic/value.rs` split (angr-7hwz). Carries the
//! second `impl RustBV` block: the canonicalize helpers, integer arithmetic
//! (`add`/`sub`/`mul`/`udiv`/`sdiv`/`urem`/`srem`/`neg`), bitwise
//! (`and`/`or`/`xor`/`not`/`reverse`/shifts/rotates), comparisons (`eq`/`ne`
//! plus the macro-generated unsigned/signed ordering pairs), structural ops
//! (`zero_extend`/`sign_extend`/`truncate`/`extract`/`concat`/`ite`), and
//! the bit-counting ops (`clz`/`ctz`/`popcount`). Inherent impls may live in
//! any module of the same crate; this lives as a child of `symbolic`.
//!
//! The two `define_{unsigned,signed}_cmp_pair!` macros and the free helpers
//! `try_zext_const_cmp_fold` / `sign_extend` / `sign_extend_to` move here
//! with the ops they serve (used nowhere else). `RustBV::all_ones_mask` stays
//! in `value.rs` (also used by the `ones` constructor) promoted to
//! `pub(super)`.

use std::sync::Arc;

use super::SymContext;
use super::stats::{
    record_bvop_concat, record_bvop_extract, record_bvop_reverse, record_commutative_canonicalize,
    record_zext_cmp_collapse, record_zext_cmp_trivial_decide,
};
use super::value::{BVOp, RustBV};

/// Generate a borrow + consuming comparison pair for an unsigned ordering
/// operator (Ult/Ule/Ugt/Uge).
///
/// Entry shape:
///   `name: into_name, BVOp::Variant, concrete_op, doc;
///    zext_fwd: ZExtCmp::Variant, zext_swap: ZExtCmp::Variant`
///
/// The concrete branch evaluates `if a $cmp b { 1 } else { 0 }` on the raw
/// `u128` representations. The symbolic branch tries the bidirectional
/// `try_zext_const_cmp_fold` shortcut (angr-g7nq pattern (b)) before falling
/// through to a generic `Expression` node.
macro_rules! define_unsigned_cmp_pair {
    (
        $name:ident, $into_name:ident, $op:expr, $cmp:tt,
        $doc_borrow:literal, $doc_consume:literal,
        $zext_fwd:expr, $zext_swap:expr
    ) => {
        #[doc = $doc_borrow]
        #[inline]
        pub fn $name(&self, other: &Self, ctx: &SymContext) -> Self {
            self.clone().$into_name(other.clone(), ctx)
        }

        #[doc = $doc_consume]
        #[inline]
        pub fn $into_name(self, other: Self, ctx: &SymContext) -> Self {
            debug_assert_eq!(self.width(), other.width());
            match (self.as_u128(), other.as_u128()) {
                (Some(a), Some(b)) => Self::concrete(if a $cmp b { 1 } else { 0 }, 1),
                _ => {
                    if let Some(folded) =
                        try_zext_const_cmp_fold(&self, &other, $zext_fwd, ctx)
                    {
                        return folded;
                    }
                    if let Some(folded) =
                        try_zext_const_cmp_fold(&other, &self, $zext_swap, ctx)
                    {
                        return folded;
                    }
                    RustBV::Expression {
                        id: Self::EXPRESSION_ID,
                        width: 1,
                        op: $op,
                        operands: Arc::<[RustBV]>::from([self, other]),
                    }
                }
            }
        }
    };
}

/// Generate a borrow + consuming comparison pair for a signed ordering
/// operator (Slt/Sle/Sgt/Sge).
///
/// Entry shape: `name: into_name, BVOp::Variant, signed_op, doc`.
///
/// The concrete branch sign-extends both operands to `i128` (via the module-
/// level `sign_extend` helper) before applying the operator. No zext-fold
/// shortcut — that path is unsigned-only.
macro_rules! define_signed_cmp_pair {
    (
        $name:ident, $into_name:ident, $op:expr, $cmp:tt,
        $doc_borrow:literal, $doc_consume:literal
    ) => {
        #[doc = $doc_borrow]
        #[inline]
        pub fn $name(&self, other: &Self, ctx: &SymContext) -> Self {
            self.clone().$into_name(other.clone(), ctx)
        }

        #[doc = $doc_consume]
        #[inline]
        pub fn $into_name(self, other: Self, _ctx: &SymContext) -> Self {
            debug_assert_eq!(self.width(), other.width());
            match (self.as_u128(), other.as_u128()) {
                (Some(a), Some(b)) => {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete(if a_signed $cmp b_signed { 1 } else { 0 }, 1)
                }
                _ => RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: $op,
                    operands: Arc::<[RustBV]>::from([self, other]),
                },
            }
        }
    };
}

impl RustBV {
    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Canonical sort key for commutative-operand ordering (angr-kkpr).
    ///
    /// Z3's mk_bv* hash-cons by AST identity, but `mk_bvadd(x, y)` and
    /// `mk_bvadd(y, x)` produce DIFFERENT Z3 AST nodes — Z3's preprocessing
    /// tactics canonicalize commutative arg order only on the formulas given
    /// to the solver, not on intermediate ASTs used as branch conditions,
    /// register values, etc. Sorting operands by this key at construction
    /// time so two callers that build the same commutative op in opposite
    /// orders produce the same RustBV (and therefore the same Z3 AST).
    ///
    /// Ordering (low → high):
    ///   0: Symbolic    (variables — most "primary")
    ///   1: Constrained
    ///   2: Expression  (subtrees with sub-key from operand-Arc pointer)
    ///   3: Concrete    (constants always end up on the right)
    ///
    /// The Expression sub-key uses the operand-Arc pointer for cheap
    /// within-run determinism. Two structurally-equal Expressions with
    /// distinct Arc allocations sort differently, but Z3 still merges
    /// their ASTs via its own hash-cons at to_z3_ast time, so this does
    /// not weaken Z3-side dedup. The key is not stable across process
    /// runs — that is intentional: RustBVs don't survive across runs.
    #[inline]
    fn canonical_sort_key(&self) -> (u8, u128) {
        match self {
            RustBV::Symbolic { id, .. } => (0, *id as u128),
            RustBV::Constrained { id, .. } => (1, *id as u128),
            RustBV::Expression { operands, .. } => {
                (2, Arc::as_ptr(operands) as *const () as usize as u128)
            }
            RustBV::Concrete { value, .. } => (3, *value),
        }
    }

    /// Sort a commutative operand pair into canonical order.
    ///
    /// Call only after constant-folding short-circuits (e.g. `x + 0 → x`,
    /// `x & all_ones → x`); those rely on the original `(self, other)`
    /// argument order to fire.
    #[inline]
    fn canonicalize_commutative(self, other: Self) -> (Self, Self) {
        if self.canonical_sort_key() <= other.canonical_sort_key() {
            record_commutative_canonicalize(false);
            (self, other)
        } else {
            record_commutative_canonicalize(true);
            (other, self)
        }
    }

    /// Add two bitvectors.
    #[inline]
    pub fn add(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().add_into(other.clone(), ctx)
    }

    /// Add two bitvectors, consuming both arguments.
    ///
    /// Avoids `self.clone()`/`other.clone()` for the Expression branch and
    /// identity simplifications. Hot-path callers (e.g. `VEXOps::binop`) that
    /// already own the operands should prefer this.
    #[inline]
    pub fn add_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_add(b), self.width()),
            // x + 0 → x
            (None, Some(0)) => self,
            // 0 + x → x
            (Some(0), None) => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Add,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Subtract two bitvectors.
    #[inline]
    pub fn sub(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sub_into(other.clone(), ctx)
    }

    /// Subtract two bitvectors, consuming both arguments.
    #[inline]
    pub fn sub_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_sub(b), self.width()),
            // x - 0 → x
            (None, Some(0)) => self,
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Sub,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Multiply two bitvectors.
    #[inline]
    pub fn mul(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().mul_into(other.clone(), ctx)
    }

    /// Multiply two bitvectors, consuming both arguments.
    #[inline]
    pub fn mul_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let width = self.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_mul(b), width),
            // x * 0 → 0
            (_, Some(0)) | (Some(0), _) => Self::zero(width),
            // x * 1 → x
            (None, Some(1)) => self,
            // 1 * x → x
            (Some(1), None) => other,
            // sym * 2^k → sym << k (avoids Z3's O(N^2) Dadda bit-blast)
            (None, Some(b)) if b.is_power_of_two() => {
                let k = b.trailing_zeros();
                let amt = Self::concrete(k as u128, width);
                self.shl_into(amt, _ctx)
            }
            // 2^k * sym → sym << k
            (Some(a), None) if a.is_power_of_two() => {
                let k = a.trailing_zeros();
                let amt = Self::concrete(k as u128, width);
                other.shl_into(amt, _ctx)
            }
            _ => {
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Mul,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Unsigned division.
    #[inline]
    pub fn udiv(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().udiv_into(other.clone(), ctx)
    }

    /// Unsigned division, consuming both arguments.
    #[inline]
    pub fn udiv_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    Self::concrete(a / b, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::UDiv,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Signed division.
    #[inline]
    pub fn sdiv(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sdiv_into(other.clone(), ctx)
    }

    /// Signed division, consuming both arguments.
    #[inline]
    pub fn sdiv_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete((a_signed / b_signed) as u128, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::SDiv,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Unsigned remainder.
    #[inline]
    pub fn urem(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().urem_into(other.clone(), ctx)
    }

    /// Unsigned remainder, consuming both arguments.
    #[inline]
    pub fn urem_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self
                } else {
                    Self::concrete(a % b, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::URem,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Signed remainder.
    #[inline]
    pub fn srem(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().srem_into(other.clone(), ctx)
    }

    /// Signed remainder, consuming both arguments.
    #[inline]
    pub fn srem_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self
                } else {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete((a_signed % b_signed) as u128, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::SRem,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Negate (two's complement).
    #[inline]
    pub fn neg(&self, ctx: &SymContext) -> Self {
        self.clone().neg_into(ctx)
    }

    /// Negate (two's complement), consuming the argument.
    #[inline]
    pub fn neg_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete((!v).wrapping_add(1), self.width()),
            None => {
                // neg(neg(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Neg,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Neg,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    /// Bitwise AND.
    #[inline]
    pub fn and(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().and_into(other.clone(), ctx)
    }

    /// Bitwise AND, consuming both arguments.
    #[inline]
    pub fn and_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let all_ones = Self::all_ones_mask(self.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a & b, self.width()),
            // x & 0 → 0
            (_, Some(0)) | (Some(0), _) => Self::zero(self.width()),
            // x & all_ones → x
            (None, Some(v)) if v == all_ones => self,
            (Some(v), None) if v == all_ones => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::And,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise OR.
    #[inline]
    pub fn or(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().or_into(other.clone(), ctx)
    }

    /// Bitwise OR, consuming both arguments.
    #[inline]
    pub fn or_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let all_ones = Self::all_ones_mask(self.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a | b, self.width()),
            // x | 0 → x
            (None, Some(0)) => self,
            (Some(0), None) => other,
            // x | all_ones → all_ones
            (_, Some(v)) if v == all_ones => Self::ones(self.width()),
            (Some(v), _) if v == all_ones => Self::ones(self.width()),
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Or,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise XOR.
    #[inline]
    pub fn xor(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().xor_into(other.clone(), ctx)
    }

    /// Bitwise XOR, consuming both arguments.
    #[inline]
    pub fn xor_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a ^ b, self.width()),
            // x ^ 0 → x
            (None, Some(0)) => self,
            (Some(0), None) => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Xor,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise NOT.
    #[inline]
    pub fn not(&self, ctx: &SymContext) -> Self {
        self.clone().not_into(ctx)
    }

    /// Bitwise NOT, consuming the argument.
    #[inline]
    pub fn not_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(!v, self.width()),
            None => {
                // not(not(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Not,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Not,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Byte-reverse a bitvector (endianness swap).
    #[inline]
    pub fn reverse(&self, ctx: &SymContext) -> Self {
        self.clone().reverse_into(ctx)
    }

    /// Byte-reverse a bitvector, consuming the argument.
    #[inline]
    pub fn reverse_into(self, _ctx: &SymContext) -> Self {
        let w = self.width();
        debug_assert!(w.is_multiple_of(8), "reverse requires byte-aligned width");
        if w <= 8 {
            return self; // Single byte, no-op
        }
        match self.as_u128() {
            Some(v) => {
                let num_bytes = (w / 8) as usize;
                let mut result: u128 = 0;
                for i in 0..num_bytes {
                    let byte = (v >> (i * 8)) & 0xff;
                    result |= byte << ((num_bytes - 1 - i) * 8);
                }
                Self::concrete(result, w)
            }
            None => {
                // reverse(reverse(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Reverse,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                record_bvop_reverse();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: w,
                    op: BVOp::Reverse,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    // =========================================================================
    // Shift Operations
    // =========================================================================

    /// Logical shift left.
    #[inline]
    pub fn shl(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().shl_into(amount.clone(), ctx)
    }

    /// Logical shift left, consuming both arguments.
    #[inline]
    pub fn shl_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                Self::concrete(v.wrapping_shl(amt), width)
            }
            // x << 0 → x
            (None, Some(0)) => self,
            // 0 << x → 0
            (Some(0), None) => Self::zero(width),
            // sym << c for 0 < c < w → Concat(Extract(w-c-1, 0, sym), 0^c).
            // Avoids Z3 bit-blasting an O(N^2) mux tree for the symbolic shift.
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let lo = self.extract_into(width - c - 1, 0, _ctx);
                lo.concat_into(Self::zero(c), _ctx)
            }
            // sym << c with c >= w → 0
            (None, Some(_)) => Self::zero(width),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Shl,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Logical shift right.
    #[inline]
    pub fn lshr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().lshr_into(amount.clone(), ctx)
    }

    /// Logical shift right, consuming both arguments.
    #[inline]
    pub fn lshr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                Self::concrete(v.wrapping_shr(amt), width)
            }
            // x >> 0 → x
            (None, Some(0)) => self,
            // 0 >> x → 0
            (Some(0), None) => Self::zero(width),
            // sym >> c for 0 < c < w → Concat(0^c, Extract(w-1, c, sym)).
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let hi = self.extract_into(width - 1, c, _ctx);
                Self::zero(c).concat_into(hi, _ctx)
            }
            // sym >> c with c >= w → 0
            (None, Some(_)) => Self::zero(width),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Lshr,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Arithmetic shift right (sign-extending).
    #[inline]
    pub fn ashr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().ashr_into(amount.clone(), ctx)
    }

    /// Arithmetic shift right, consuming both arguments.
    #[inline]
    pub fn ashr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                let signed = sign_extend(v, width);
                Self::concrete((signed >> amt) as u128, width)
            }
            // x >>> 0 → x
            (None, Some(0)) => self,
            // sym >>> c for 0 < c < w → SignExt(Extract(w-1, c, sym), w).
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let hi = self.extract_into(width - 1, c, _ctx);
                hi.sign_extend_into(width, _ctx)
            }
            // sym >>> c with c >= w → SignExt(MSB, w) (saturates to all sign bits).
            (None, Some(_)) => {
                let msb = self.extract_into(width - 1, width - 1, _ctx);
                msb.sign_extend_into(width, _ctx)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Ashr,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Rotate left.
    #[inline]
    pub fn rotl(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().rotl_into(amount.clone(), ctx)
    }

    /// Rotate left, consuming both arguments.
    #[inline]
    pub fn rotl_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v << amt) | (v >> (w - amt));
                Self::concrete(rotated, w)
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::RotL,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
        }
    }

    /// Rotate right.
    #[inline]
    pub fn rotr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().rotr_into(amount.clone(), ctx)
    }

    /// Rotate right, consuming both arguments.
    #[inline]
    pub fn rotr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v >> amt) | (v << (w - amt));
                Self::concrete(rotated, w)
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::RotR,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
        }
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit result).
    #[inline]
    pub fn eq(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().eq_into(other.clone(), ctx)
    }

    /// Equality comparison, consuming both arguments.
    #[inline]
    pub fn eq_into(self, other: Self, ctx: &SymContext) -> Self {
        // Width mismatch guard — return concrete 0 instead of panicking
        if self.width() != other.width() {
            return Self::concrete(0, 1);
        }
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a == b { 1 } else { 0 }, 1),
            _ => {
                // angr-g7nq pattern (b): `Eq(ZeroExt(k, x), BVV(c, W))`.
                // ZeroExt is zero in the top k bits, so:
                //   * if `c >> (W-k) != 0` the equation is unsatisfiable → false
                //   * if `c >> (W-k) == 0` rewrite to `Eq(x, BVV(c, W-k))` so a
                //     smaller Z3 AST is built (and downstream rules — e.g.
                //     `eq_into`'s concrete branch — can fold further).
                // Handles either operand order; signed ext is intentionally not
                // touched (the sign-bit handling is more delicate, see
                // `try_zext_const_cmp_fold`).
                if let Some(folded) = try_zext_const_cmp_fold(&self, &other, ZExtCmp::Eq, ctx) {
                    return folded;
                }
                if let Some(folded) = try_zext_const_cmp_fold(&other, &self, ZExtCmp::Eq, ctx) {
                    return folded;
                }
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: BVOp::Eq,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Inequality comparison (returns 1-bit result).
    #[inline]
    pub fn ne(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ne_into(other.clone(), ctx)
    }

    /// Inequality comparison, consuming both arguments.
    #[inline]
    pub fn ne_into(self, other: Self, ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a != b { 1 } else { 0 }, 1),
            _ => {
                // angr-g7nq pattern (b), see `eq_into`. For Ne the trivial-decide
                // direction inverts (high-bit-nonzero const → always not-equal).
                if let Some(folded) = try_zext_const_cmp_fold(&self, &other, ZExtCmp::Ne, ctx) {
                    return folded;
                }
                if let Some(folded) = try_zext_const_cmp_fold(&other, &self, ZExtCmp::Ne, ctx) {
                    return folded;
                }
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: BVOp::Ne,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    // Unsigned ordering comparisons. `Ult` direction matters for the
    // `try_zext_const_cmp_fold` shortcut: high-bit-nonzero const → different
    // trivial answers for `Ult(zext, const)` vs `Ult(const, zext)`. `Ugt(a,b)
    // == Ult(b,a)`, so the forward zext arg is the swapped variant. Symmetric
    // story for Ule/Uge.
    define_unsigned_cmp_pair!(
        ult, ult_into, BVOp::Ult, <,
        "Unsigned less-than comparison.",
        "Unsigned less-than, consuming both arguments.",
        ZExtCmp::Ult, ZExtCmp::UltSwapped
    );
    define_unsigned_cmp_pair!(
        ule, ule_into, BVOp::Ule, <=,
        "Unsigned less-than-or-equal comparison.",
        "Unsigned less-than-or-equal, consuming both arguments.",
        ZExtCmp::Ule, ZExtCmp::UleSwapped
    );
    define_unsigned_cmp_pair!(
        ugt, ugt_into, BVOp::Ugt, >,
        "Unsigned greater-than comparison.",
        "Unsigned greater-than, consuming both arguments.",
        ZExtCmp::UltSwapped, ZExtCmp::Ult
    );
    define_unsigned_cmp_pair!(
        uge, uge_into, BVOp::Uge, >=,
        "Unsigned greater-than-or-equal comparison.",
        "Unsigned greater-than-or-equal, consuming both arguments.",
        ZExtCmp::UleSwapped, ZExtCmp::Ule
    );

    // Signed ordering comparisons. Sign-extend both operands to i128 before
    // applying the operator. No zext-const-cmp-fold (unsigned-only shortcut).
    define_signed_cmp_pair!(
        slt, slt_into, BVOp::Slt, <,
        "Signed less-than comparison.",
        "Signed less-than, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sle, sle_into, BVOp::Sle, <=,
        "Signed less-than-or-equal comparison.",
        "Signed less-than-or-equal, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sgt, sgt_into, BVOp::Sgt, >,
        "Signed greater-than comparison.",
        "Signed greater-than, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sge, sge_into, BVOp::Sge, >=,
        "Signed greater-than-or-equal comparison.",
        "Signed greater-than-or-equal, consuming both arguments."
    );

    // =========================================================================
    // Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    #[inline]
    pub fn zero_extend(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().zero_extend_into(to_width, ctx)
    }

    /// Zero-extend to a wider width, consuming the argument.
    #[inline]
    pub fn zero_extend_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        if to_width <= self.width() {
            // No extension needed (or truncation — just return self)
            return self;
        }
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                op: BVOp::ZeroExt(extend_bits),
                operands: Arc::<[RustBV]>::from([self]),
            },
        }
    }

    /// Sign-extend to a wider width.
    #[inline]
    pub fn sign_extend(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().sign_extend_into(to_width, ctx)
    }

    /// Sign-extend to a wider width, consuming the argument.
    #[inline]
    pub fn sign_extend_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width >= self.width());
        // No extension needed
        if to_width == self.width() {
            return self;
        }
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => {
                let extended = sign_extend_to(v, self.width(), to_width);
                Self::concrete(extended, to_width)
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                op: BVOp::SignExt(extend_bits),
                operands: Arc::<[RustBV]>::from([self]),
            },
        }
    }

    /// Truncate to a narrower width.
    #[inline]
    pub fn truncate(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().truncate_into(to_width, ctx)
    }

    /// Truncate to a narrower width, consuming the argument.
    #[inline]
    pub fn truncate_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width <= self.width());
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => {
                record_bvop_extract();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: to_width,
                    op: BVOp::Extract(to_width - 1, 0),
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Extract bits \[high:low\] (inclusive).
    #[inline]
    pub fn extract(&self, high: u32, low: u32, ctx: &SymContext) -> Self {
        self.clone().extract_into(high, low, ctx)
    }

    /// Extract bits \[high:low\] (inclusive), consuming the argument.
    #[inline]
    pub fn extract_into(self, high: u32, low: u32, _ctx: &SymContext) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;

        // Fast path for concrete values
        if let Some(v) = self.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            return Self::concrete(extracted, result_width);
        }

        // Identity extraction: Extract(width-1, 0, x) → x
        if high == self.width() - 1 && low == 0 {
            return self;
        }

        // Canonicalization rules for Expression nodes — borrow self to inspect,
        // then either delegate (returning early) or fall through to construct.
        if let RustBV::Expression { op, operands, .. } = &self {
            match op {
                // Rule 1: Extract(Extract(x)) → fused single Extract
                // Extract(h2, l2, Extract(h1, l1, x)) → Extract(l1+h2, l1+l2, x)
                BVOp::Extract(_, inner_low) => {
                    return operands[0].extract(inner_low + high, inner_low + low, _ctx);
                }

                // Rule 2: Extract(Concat(a, b)) → distribute to relevant part(s)
                BVOp::Concat if operands.len() == 2 => {
                    let b_width = operands[1].width();
                    if high < b_width {
                        // Entirely within the low part (b)
                        return operands[1].extract(high, low, _ctx);
                    } else if low >= b_width {
                        // Entirely within the high part (a)
                        return operands[0].extract(high - b_width, low - b_width, _ctx);
                    }
                    // Crosses boundary — extract from each part and concat
                    let lo_part = operands[1].extract(b_width - 1, low, _ctx);
                    let hi_part = operands[0].extract(high - b_width, 0, _ctx);
                    return hi_part.concat_into(lo_part, _ctx);
                }

                // Rule 3: Extract(Reverse(x)) with byte-aligned bounds.
                //
                // The byte at position k of Reverse(x) (with B = x.width/8 bytes)
                // is x's byte at position B-1-k. Extracting bytes [l/8..h/8] from
                // Reverse(x) is therefore the byte sequence of x at indices
                // [B-1-h/8 .. B-1-l/8], delivered in reversed order.
                //
                // → Single-byte (h == l + 7): plain Extract(w-1-low, w-1-high, x).
                // → Multi-byte:                Reverse(Extract(w-1-low, w-1-high, x)).
                //
                // The previous form dropped the byte-shuffle for the multi-byte
                // case, which is silently wrong on a Z3 round-trip.
                BVOp::Reverse
                    if operands[0].width() % 8 == 0 && high % 8 == 7 && low.is_multiple_of(8) =>
                {
                    let w = operands[0].width();
                    let inner = operands[0].extract(w - 1 - low, w - 1 - high, _ctx);
                    if high - low + 1 == 8 {
                        return inner;
                    }
                    return inner.reverse(_ctx);
                }

                // Rule 4: Extract(ZeroExt(x)) — if entirely within original width,
                // extract from x directly; if entirely in extended bits, result is 0
                BVOp::ZeroExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return operands[0].extract(high, low, _ctx);
                    } else if low >= inner_width {
                        return Self::zero(result_width);
                    }
                }

                // Rule 5: Extract(SignExt(x)) — if entirely within original width,
                // extract from x directly
                BVOp::SignExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return operands[0].extract(high, low, _ctx);
                    }
                }

                _ => {}
            }
        }

        record_bvop_extract();
        RustBV::Expression {
            id: Self::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: Arc::<[RustBV]>::from([self]),
        }
    }

    /// Concatenate two bitvectors (self becomes high bits).
    #[inline]
    pub fn concat(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().concat_into(other.clone(), ctx)
    }

    /// Concatenate two bitvectors, consuming both arguments.
    #[inline]
    pub fn concat_into(self, other: Self, _ctx: &SymContext) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => {
                record_bvop_concat();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: result_width,
                    op: BVOp::Concat,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Extract bits without requiring a SymContext (same logic, ctx unused).
    #[inline]
    pub fn extract_no_ctx(&self, high: u32, low: u32) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;

        if let Some(v) = self.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            return Self::concrete(extracted, result_width);
        }

        if high == self.width() - 1 && low == 0 {
            return self.clone();
        }

        record_bvop_extract();
        RustBV::Expression {
            id: Self::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: Arc::<[RustBV]>::from([self.clone()]),
        }
    }

    /// Build a balanced Concat tree from `parts`, ordered HIGH bits first
    /// and LOW bits last (i.e. `parts\[0\]` becomes the high bits of the
    /// result, `parts[parts.len()-1]` becomes the low bits).
    ///
    /// A linear left-fold (`acc = concat(acc, next)`) produces a skewed
    /// AST of depth N-1; pre-Z3 analysis passes walking this DAG do extra
    /// work. This helper recursively splits the slice in half, producing
    /// a tree of depth `ceil(log2(N))`. Z3's flat=true rewriter still
    /// flattens at solve time, but starting from a balanced shape exposes
    /// structural sharing to max-bv-sharing (which matches by AST node
    /// identity) and saves rewrite cycles.
    ///
    /// Panics if `parts` is empty.
    #[inline]
    pub fn concat_balanced(parts: &[RustBV], ctx: &SymContext) -> Self {
        assert!(
            !parts.is_empty(),
            "concat_balanced requires at least one element"
        );
        if parts.len() == 1 {
            return parts[0].clone();
        }
        let mid = parts.len() / 2;
        let high = Self::concat_balanced(&parts[..mid], ctx);
        let low = Self::concat_balanced(&parts[mid..], ctx);
        high.concat_into(low, ctx)
    }

    /// Concatenate without requiring a SymContext (same logic, ctx unused).
    #[inline]
    pub fn concat_no_ctx(&self, other: &Self) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => {
                record_bvop_concat();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: result_width,
                    op: BVOp::Concat,
                    operands: Arc::<[RustBV]>::from([self.clone(), other.clone()]),
                }
            }
        }
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    /// If-then-else: returns `then_val` if `self` is non-zero, else `else_val`.
    #[inline]
    pub fn ite(&self, then_val: &Self, else_val: &Self, ctx: &SymContext) -> Self {
        self.clone()
            .ite_into(then_val.clone(), else_val.clone(), ctx)
    }

    /// If-then-else, consuming all three arguments.
    #[inline]
    pub fn ite_into(self, then_val: Self, else_val: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(then_val.width(), else_val.width());
        match self.as_u128() {
            Some(v) => {
                if v != 0 {
                    then_val
                } else {
                    else_val
                }
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: then_val.width(),
                op: BVOp::Ite,
                operands: Arc::<[RustBV]>::from([self, then_val, else_val]),
            },
        }
    }

    /// Count leading zeros.
    #[inline]
    pub fn clz(&self, ctx: &SymContext) -> Self {
        self.clone().clz_into(ctx)
    }

    /// Count leading zeros, consuming the argument.
    #[inline]
    pub fn clz_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let leading = if v == 0 {
                    self.width()
                } else {
                    self.width() - (128 - v.leading_zeros())
                };
                Self::concrete(leading as u128, self.width())
            }
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Clz,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Count trailing zeros.
    #[inline]
    pub fn ctz(&self, ctx: &SymContext) -> Self {
        self.clone().ctz_into(ctx)
    }

    /// Count trailing zeros, consuming the argument.
    #[inline]
    pub fn ctz_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let trailing = if v == 0 {
                    self.width()
                } else {
                    v.trailing_zeros().min(self.width())
                };
                Self::concrete(trailing as u128, self.width())
            }
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Ctz,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Population count (number of set bits).
    #[inline]
    pub fn popcount(&self, ctx: &SymContext) -> Self {
        self.clone().popcount_into(ctx)
    }

    /// Population count, consuming the argument.
    #[inline]
    pub fn popcount_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(v.count_ones() as u128, self.width()),
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Popcount,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// angr-g7nq: comparison-op tag for `try_zext_const_cmp_fold`. The `Swapped`
/// variants describe the case where the ZeroExt operand sits on the right of
/// the original operator (e.g. `Ult(const, ZeroExt(x))` is `UltSwapped` from
/// the helper's perspective, since the helper always takes `(zext_side,
/// const_side)`).
#[derive(Copy, Clone)]
enum ZExtCmp {
    Eq,
    Ne,
    Ult,
    UltSwapped,
    Ule,
    UleSwapped,
}

/// Fold `Cmp(ZeroExt(k, x), BVV(c, W))` (and commuted variants) using the
/// fact that `ZeroExt(k, x)` is always in the range `[0, 2^(W-k))`.
///
/// Two outcomes:
/// * Trivial decide — the comparison is structurally `true` or `false`
///   regardless of `x`. Returns `Self::concrete(0|1, 1)`. Bumps
///   `zext_cmp_trivial_decide_count`.
/// * Collapse — the high `k` bits of `c` are zero (or the comparison is
///   insensitive to them), so the comparison is rewritten on the `W-k`-bit
///   operands by recursing into the same op on the narrowed sides. Bumps
///   `zext_cmp_collapse_count`.
///
/// Returns `None` if `zext_side` is not a `BVOp::ZeroExt(_)` expression, or
/// `const_side` is not concrete, or the widths don't match the expected shape.
fn try_zext_const_cmp_fold(
    zext_side: &RustBV,
    const_side: &RustBV,
    op: ZExtCmp,
    ctx: &SymContext,
) -> Option<RustBV> {
    let (extend_bits, inner) = match zext_side {
        RustBV::Expression {
            op: BVOp::ZeroExt(k),
            operands,
            ..
        } => (*k, &operands[0]),
        _ => return None,
    };
    // ZeroExt(0, x) → x; the wrapping caller hands us a normal width
    // comparison and we shouldn't bother. `extend_bits == 0` is unusual but
    // safe to skip — falling through builds the same Cmp expression as before.
    if extend_bits == 0 {
        return None;
    }
    let c = const_side.as_u128()?;
    let total_width = zext_side.width();
    let inner_width = inner.width();
    debug_assert_eq!(inner_width + extend_bits, total_width);
    // Compute `c_high = c >> inner_width` (the bits that ZeroExt forces to
    // zero) and `c_low` (the low W-k bits) without overflowing the shift on
    // total_width == 128.
    let (c_high, c_low) = if inner_width >= 128 {
        // Defensive; in practice all BVVs we touch fit in <= 128 bits and the
        // ZeroExt source can't be wider than 128 either.
        (0u128, c)
    } else {
        let low_mask = (1u128 << inner_width) - 1;
        (c >> inner_width, c & low_mask)
    };

    match op {
        ZExtCmp::Eq | ZExtCmp::Ne => {
            if c_high != 0 {
                // ZeroExt produces high bits zero; const_side disagrees → eq
                // is unsat, ne is tautological.
                record_zext_cmp_trivial_decide();
                let val = match op {
                    ZExtCmp::Eq => 0,
                    ZExtCmp::Ne => 1,
                    _ => unreachable!(),
                };
                return Some(RustBV::concrete(val, 1));
            }
            // High bits agree (both zero) → narrow.
            record_zext_cmp_collapse();
            let narrowed_const = RustBV::concrete(c_low, inner_width);
            let narrowed_inner = inner.clone();
            let result = match op {
                ZExtCmp::Eq => narrowed_inner.eq_into(narrowed_const, ctx),
                ZExtCmp::Ne => narrowed_inner.ne_into(narrowed_const, ctx),
                _ => unreachable!(),
            };
            Some(result)
        }
        ZExtCmp::Ult => {
            // `ZeroExt(k, x) < c`. ZeroExt ∈ [0, 2^(W-k)).
            if c_high != 0 {
                // c >= 2^(W-k) ⇒ ZeroExt(x) < c is always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            if c_low == 0 {
                // ZeroExt(x) < 0 is unsat.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            record_zext_cmp_collapse();
            Some(
                inner
                    .clone()
                    .ult_into(RustBV::concrete(c_low, inner_width), ctx),
            )
        }
        ZExtCmp::UltSwapped => {
            // `c < ZeroExt(k, x)`. ZeroExt ∈ [0, 2^(W-k)).
            if c_high != 0 {
                // c >= 2^(W-k) > ZeroExt(x); never c < ZeroExt(x).
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            record_zext_cmp_collapse();
            Some(RustBV::concrete(c_low, inner_width).ult_into(inner.clone(), ctx))
        }
        ZExtCmp::Ule => {
            // `ZeroExt(k, x) <= c`.
            if c_high != 0 {
                // c >= 2^(W-k) > all ZeroExt values; always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            record_zext_cmp_collapse();
            Some(
                inner
                    .clone()
                    .ule_into(RustBV::concrete(c_low, inner_width), ctx),
            )
        }
        ZExtCmp::UleSwapped => {
            // `c <= ZeroExt(k, x)`.
            if c_high != 0 {
                // c >= 2^(W-k) > ZeroExt(x); never c <= ZeroExt(x).
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            if c_low == 0 {
                // 0 <= ZeroExt(x) is always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            record_zext_cmp_collapse();
            Some(RustBV::concrete(c_low, inner_width).ule_into(inner.clone(), ctx))
        }
    }
}

/// Sign-extend a value from `width` bits to i128.
fn sign_extend(value: u128, width: u32) -> i128 {
    if width >= 128 {
        value as i128
    } else {
        let sign_bit = 1u128 << (width - 1);
        if value & sign_bit != 0 {
            (value | !((1u128 << width) - 1)) as i128
        } else {
            value as i128
        }
    }
}

/// Sign-extend a value from one width to another (staying in u128).
fn sign_extend_to(value: u128, from_width: u32, to_width: u32) -> u128 {
    if from_width >= to_width {
        value
    } else {
        let sign_bit = 1u128 << (from_width - 1);
        if value & sign_bit != 0 {
            let extension_mask = ((1u128 << to_width) - 1) & !((1u128 << from_width) - 1);
            value | extension_mask
        } else {
            value
        }
    }
}
