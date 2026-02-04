//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that can be either:
//! - Concrete: A known fixed value
//! - Symbolic: Represents an unknown value (with optional Z3 backing)

use std::fmt;

use super::SymContext;

/// Bit width for parameterized operations.
/// This reduces ~200 VEX ops to ~30 parameterized variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitWidth {
    W1 = 1,
    W8 = 8,
    W16 = 16,
    W32 = 32,
    W64 = 64,
    W128 = 128,
}

impl BitWidth {
    pub fn bits(&self) -> u32 {
        *self as u32
    }

    pub fn bytes(&self) -> u32 {
        self.bits() / 8
    }

    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            1 => Some(BitWidth::W1),
            8 => Some(BitWidth::W8),
            16 => Some(BitWidth::W16),
            32 => Some(BitWidth::W32),
            64 => Some(BitWidth::W64),
            128 => Some(BitWidth::W128),
            _ => None,
        }
    }

    pub fn mask(&self) -> u128 {
        match self {
            BitWidth::W1 => 0x1,
            BitWidth::W8 => 0xFF,
            BitWidth::W16 => 0xFFFF,
            BitWidth::W32 => 0xFFFF_FFFF,
            BitWidth::W64 => 0xFFFF_FFFF_FFFF_FFFF,
            BitWidth::W128 => u128::MAX,
        }
    }
}

/// Signedness for comparison and extension operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// A bitvector value that can be concrete or symbolic.
///
/// This is the core value type for the VEX execution engine. All register
/// and memory values are represented as `RustBV`.
#[derive(Clone)]
pub enum RustBV<'ctx> {
    /// A known concrete value.
    Concrete {
        /// The value, masked to `width` bits.
        value: u128,
        /// Width in bits (1, 8, 16, 32, 64, or 128).
        width: u32,
    },
    /// A symbolic value (backed by Z3 when available, otherwise just a name).
    Symbolic {
        /// Unique identifier for this symbolic value.
        id: u64,
        /// Width in bits.
        width: u32,
        /// Name (for debugging).
        name: String,
        /// Phantom data for lifetime.
        _ctx: std::marker::PhantomData<&'ctx ()>,
        /// Z3 AST when z3 feature is enabled.
        #[cfg(feature = "vex-engine-z3")]
        ast: z3::ast::BV<'ctx>,
    },
    /// Optimization: a symbolic value with a known concrete value.
    Constrained {
        /// The symbolic identifier.
        id: u64,
        /// The known concrete value.
        value: u128,
        /// Width in bits.
        width: u32,
        /// Phantom data for lifetime.
        _ctx: std::marker::PhantomData<&'ctx ()>,
    },
}

impl<'ctx> RustBV<'ctx> {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a concrete bitvector from a value.
    #[inline]
    pub fn concrete(value: u128, width: u32) -> Self {
        let mask = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        RustBV::Concrete {
            value: value & mask,
            width,
        }
    }

    /// Create a concrete bitvector from a u64.
    #[inline]
    pub fn from_u64(value: u64, width: u32) -> Self {
        Self::concrete(value as u128, width)
    }

    /// Create a concrete bitvector from a u32.
    pub fn from_u32(value: u32, width: u32) -> Self {
        Self::concrete(value as u128, width)
    }

    /// Create a zero bitvector.
    #[inline]
    pub fn zero(width: u32) -> Self {
        RustBV::Concrete { value: 0, width }
    }

    /// Create a bitvector with all bits set to 1.
    pub fn ones(width: u32) -> Self {
        let mask = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        RustBV::Concrete { value: mask, width }
    }

    /// Create a symbolic bitvector variable.
    pub fn symbolic(ctx: &'ctx SymContext<'ctx>, name: &str, width: u32) -> Self {
        let id = ctx.next_id();
        #[cfg(feature = "vex-engine-z3")]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
                _ctx: std::marker::PhantomData,
                ast: z3::ast::BV::new_const(ctx.z3_ctx(), name, width),
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
                _ctx: std::marker::PhantomData,
            }
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the width in bits.
    #[inline]
    pub fn width(&self) -> u32 {
        match self {
            RustBV::Concrete { width, .. } => *width,
            RustBV::Symbolic { width, .. } => *width,
            RustBV::Constrained { width, .. } => *width,
        }
    }

    /// Check if this value is concrete.
    #[inline]
    pub fn is_concrete(&self) -> bool {
        matches!(self, RustBV::Concrete { .. })
    }

    /// Check if this value is symbolic.
    #[inline]
    pub fn is_symbolic(&self) -> bool {
        matches!(self, RustBV::Symbolic { .. } | RustBV::Constrained { .. })
    }

    /// Try to get the concrete value.
    #[inline]
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            RustBV::Concrete { value, .. } => Some(*value),
            RustBV::Constrained { value, .. } => Some(*value),
            RustBV::Symbolic { .. } => None,
        }
    }

    /// Get the concrete value, panicking if symbolic.
    #[inline]
    pub fn to_u128(&self) -> u128 {
        self.as_u128().expect("value is symbolic")
    }

    /// Try to get the concrete value as u64.
    #[inline]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_u128().map(|v| v as u64)
    }

    /// Get the concrete value as u64, panicking if symbolic or too wide.
    #[inline]
    pub fn to_u64(&self) -> u64 {
        self.as_u64().expect("value is symbolic or too wide")
    }

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Add two bitvectors.
    pub fn add(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_add(b), self.width()),
            _ => {
                // For symbolic, return a placeholder
                RustBV::Symbolic {
                    id: _ctx.next_id(),
                    width: self.width(),
                    name: "add_result".to_string(),
                    _ctx: std::marker::PhantomData,
                    #[cfg(feature = "vex-engine-z3")]
                    ast: self.to_z3_ast(_ctx).bvadd(&other.to_z3_ast(_ctx)),
                }
            }
        }
    }

    /// Subtract two bitvectors.
    pub fn sub(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_sub(b), self.width()),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "sub_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvsub(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Multiply two bitvectors.
    pub fn mul(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_mul(b), self.width()),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "mul_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvmul(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Unsigned division.
    pub fn udiv(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    Self::concrete(a / b, self.width())
                }
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "udiv_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvudiv(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Signed division.
    pub fn sdiv(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
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
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "sdiv_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvsdiv(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Unsigned remainder.
    pub fn urem(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self.clone()
                } else {
                    Self::concrete(a % b, self.width())
                }
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "urem_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvurem(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Signed remainder.
    pub fn srem(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self.clone()
                } else {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete((a_signed % b_signed) as u128, self.width())
                }
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "srem_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvsrem(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Negate (two's complement).
    pub fn neg(&self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete((!v).wrapping_add(1), self.width()),
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "neg_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvneg(),
            },
        }
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    /// Bitwise AND.
    pub fn and(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a & b, self.width()),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "and_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvand(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Bitwise OR.
    pub fn or(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a | b, self.width()),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "or_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvor(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Bitwise XOR.
    pub fn xor(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a ^ b, self.width()),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "xor_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvxor(&other.to_z3_ast(_ctx)),
            },
        }
    }

    /// Bitwise NOT.
    pub fn not(&self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(!v, self.width()),
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "not_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvnot(),
            },
        }
    }

    // =========================================================================
    // Shift Operations
    // =========================================================================

    /// Logical shift left.
    pub fn shl(&self, amount: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shl(amt), self.width())
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "shl_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvshl(&amount.to_z3_ast(_ctx)),
            },
        }
    }

    /// Logical shift right.
    pub fn lshr(&self, amount: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shr(amt), self.width())
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "lshr_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvlshr(&amount.to_z3_ast(_ctx)),
            },
        }
    }

    /// Arithmetic shift right (sign-extending).
    pub fn ashr(&self, amount: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                let signed = sign_extend(v, self.width());
                Self::concrete((signed >> amt) as u128, self.width())
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "ashr_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvashr(&amount.to_z3_ast(_ctx)),
            },
        }
    }

    /// Rotate left.
    pub fn rotl(&self, amount: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v << amt) | (v >> (w - amt));
                Self::concrete(rotated, w)
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "rotl_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvrotl(&amount.to_z3_ast(_ctx)),
            },
        }
    }

    /// Rotate right.
    pub fn rotr(&self, amount: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v >> amt) | (v << (w - amt));
                Self::concrete(rotated, w)
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "rotr_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).bvrotr(&amount.to_z3_ast(_ctx)),
            },
        }
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit result).
    pub fn eq(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a == b { 1 } else { 0 }, 1),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: 1,
                name: "eq_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let eq = self.to_z3_ast(_ctx)._eq(&other.to_z3_ast(_ctx));
                    eq.ite(
                        &z3::ast::BV::from_u64(ctx, 1, 1),
                        &z3::ast::BV::from_u64(ctx, 0, 1),
                    )
                },
            },
        }
    }

    /// Inequality comparison (returns 1-bit result).
    pub fn ne(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        let eq_result = self.eq(other, _ctx);
        eq_result.not(_ctx)
    }

    /// Unsigned less-than comparison.
    pub fn ult(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a < b { 1 } else { 0 }, 1),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: 1,
                name: "ult_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let lt = self.to_z3_ast(_ctx).bvult(&other.to_z3_ast(_ctx));
                    lt.ite(
                        &z3::ast::BV::from_u64(ctx, 1, 1),
                        &z3::ast::BV::from_u64(ctx, 0, 1),
                    )
                },
            },
        }
    }

    /// Unsigned less-than-or-equal comparison.
    pub fn ule(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a <= b { 1 } else { 0 }, 1),
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: 1,
                name: "ule_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let le = self.to_z3_ast(_ctx).bvule(&other.to_z3_ast(_ctx));
                    le.ite(
                        &z3::ast::BV::from_u64(ctx, 1, 1),
                        &z3::ast::BV::from_u64(ctx, 0, 1),
                    )
                },
            },
        }
    }

    /// Unsigned greater-than comparison.
    pub fn ugt(&self, other: &Self, ctx: &'ctx SymContext<'ctx>) -> Self {
        other.ult(self, ctx)
    }

    /// Unsigned greater-than-or-equal comparison.
    pub fn uge(&self, other: &Self, ctx: &'ctx SymContext<'ctx>) -> Self {
        other.ule(self, ctx)
    }

    /// Signed less-than comparison.
    pub fn slt(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed < b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: 1,
                name: "slt_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let lt = self.to_z3_ast(_ctx).bvslt(&other.to_z3_ast(_ctx));
                    lt.ite(
                        &z3::ast::BV::from_u64(ctx, 1, 1),
                        &z3::ast::BV::from_u64(ctx, 0, 1),
                    )
                },
            },
        }
    }

    /// Signed less-than-or-equal comparison.
    pub fn sle(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed <= b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: 1,
                name: "sle_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let le = self.to_z3_ast(_ctx).bvsle(&other.to_z3_ast(_ctx));
                    le.ite(
                        &z3::ast::BV::from_u64(ctx, 1, 1),
                        &z3::ast::BV::from_u64(ctx, 0, 1),
                    )
                },
            },
        }
    }

    /// Signed greater-than comparison.
    pub fn sgt(&self, other: &Self, ctx: &'ctx SymContext<'ctx>) -> Self {
        other.slt(self, ctx)
    }

    /// Signed greater-than-or-equal comparison.
    pub fn sge(&self, other: &Self, ctx: &'ctx SymContext<'ctx>) -> Self {
        other.sle(self, ctx)
    }

    // =========================================================================
    // Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn zero_extend(&self, to_width: u32, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert!(to_width >= self.width());
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: to_width,
                name: "zext_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).zero_ext(to_width - self.width()),
            },
        }
    }

    /// Sign-extend to a wider width.
    pub fn sign_extend(&self, to_width: u32, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert!(to_width >= self.width());
        match self.as_u128() {
            Some(v) => {
                let extended = sign_extend_to(v, self.width(), to_width);
                Self::concrete(extended, to_width)
            }
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: to_width,
                name: "sext_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).sign_ext(to_width - self.width()),
            },
        }
    }

    /// Truncate to a narrower width.
    pub fn truncate(&self, to_width: u32, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert!(to_width <= self.width());
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: to_width,
                name: "trunc_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).extract(to_width - 1, 0),
            },
        }
    }

    /// Extract bits [high:low] (inclusive).
    pub fn extract(&self, high: u32, low: u32, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;
        match self.as_u128() {
            Some(v) => {
                let extracted = (v >> low) & ((1u128 << result_width) - 1);
                Self::concrete(extracted, result_width)
            }
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: result_width,
                name: "extract_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).extract(high, low),
            },
        }
    }

    /// Concatenate two bitvectors (self becomes high bits).
    pub fn concat(&self, other: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: result_width,
                name: "concat_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast(_ctx).concat(&other.to_z3_ast(_ctx)),
            },
        }
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    /// If-then-else: returns `then_val` if `self` is non-zero, else `else_val`.
    pub fn ite(&self, then_val: &Self, else_val: &Self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        debug_assert_eq!(then_val.width(), else_val.width());
        match self.as_u128() {
            Some(v) => {
                if v != 0 {
                    then_val.clone()
                } else {
                    else_val.clone()
                }
            }
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: then_val.width(),
                name: "ite_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ctx = _ctx.z3_ctx();
                    let cond = self
                        .to_z3_ast(_ctx)
                        ._eq(&z3::ast::BV::from_u64(ctx, 0, self.width()))
                        .not();
                    cond.ite(&then_val.to_z3_ast(_ctx), &else_val.to_z3_ast(_ctx))
                },
            },
        }
    }

    /// Count leading zeros.
    pub fn clz(&self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        match self.as_u128() {
            Some(v) => {
                let leading = if v == 0 {
                    self.width()
                } else {
                    (self.width() - (128 - v.leading_zeros())).max(0)
                };
                Self::concrete(leading as u128, self.width())
            }
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "clz_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    // Build CLZ symbolically - this is complex
                    // For now, just return a symbolic value
                    z3::ast::BV::new_const(_ctx.z3_ctx(), "clz", self.width())
                },
            },
        }
    }

    /// Count trailing zeros.
    pub fn ctz(&self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        match self.as_u128() {
            Some(v) => {
                let trailing = if v == 0 {
                    self.width()
                } else {
                    v.trailing_zeros().min(self.width())
                };
                Self::concrete(trailing as u128, self.width())
            }
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "ctz_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: z3::ast::BV::new_const(_ctx.z3_ctx(), "ctz", self.width()),
            },
        }
    }

    /// Population count (number of set bits).
    pub fn popcount(&self, _ctx: &'ctx SymContext<'ctx>) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(v.count_ones() as u128, self.width()),
            None => RustBV::Symbolic {
                id: _ctx.next_id(),
                width: self.width(),
                name: "popcount_result".to_string(),
                _ctx: std::marker::PhantomData,
                #[cfg(feature = "vex-engine-z3")]
                ast: z3::ast::BV::new_const(_ctx.z3_ctx(), "popcount", self.width()),
            },
        }
    }

    // =========================================================================
    // Z3 Integration (when feature is enabled)
    // =========================================================================

    #[cfg(feature = "vex-engine-z3")]
    fn to_z3_ast(&self, ctx: &'ctx SymContext<'ctx>) -> z3::ast::BV<'ctx> {
        use z3::ast::Ast;
        match self {
            RustBV::Concrete { value, width } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(ctx.z3_ctx(), *value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(ctx.z3_ctx(), *value as u64, 64);
                    let hi = z3::ast::BV::from_u64(ctx.z3_ctx(), (*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
            RustBV::Symbolic { ast, .. } => ast.clone(),
            RustBV::Constrained { value, width, .. } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(ctx.z3_ctx(), *value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(ctx.z3_ctx(), *value as u64, 64);
                    let hi = z3::ast::BV::from_u64(ctx.z3_ctx(), (*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

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

// =============================================================================
// Trait Implementations
// =============================================================================

impl fmt::Debug for RustBV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustBV::Concrete { value, width } => {
                write!(f, "Concrete(0x{:x}, {})", value, width)
            }
            RustBV::Symbolic { width, name, .. } => {
                write!(f, "Symbolic({}, {})", name, width)
            }
            RustBV::Constrained { value, width, .. } => {
                write!(f, "Constrained(0x{:x}, {})", value, width)
            }
        }
    }
}

impl fmt::Display for RustBV<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustBV::Concrete { value, width } => {
                write!(f, "<BV{} 0x{:x}>", width, value)
            }
            RustBV::Symbolic { width, name, .. } => {
                write!(f, "<BV{} {}>", width, name)
            }
            RustBV::Constrained { value, width, .. } => {
                write!(f, "<BV{} 0x{:x} (constrained)>", width, value)
            }
        }
    }
}

impl PartialEq for RustBV<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => a == b && self.width() == other.width(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_add() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(3, 32);
        let result = a.add(&b, &ctx);
        assert_eq!(result.as_u64(), Some(8));
    }

    #[test]
    fn test_concrete_sub() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(10, 32);
        let b = RustBV::concrete(3, 32);
        let result = a.sub(&b, &ctx);
        assert_eq!(result.as_u64(), Some(7));
    }

    #[test]
    fn test_concrete_overflow() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8);
        let b = RustBV::concrete(1, 8);
        let result = a.add(&b, &ctx);
        assert_eq!(result.as_u64(), Some(0)); // 8-bit overflow wraps to 0
    }

    #[test]
    fn test_sign_extend() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits
        let result = a.sign_extend(32, &ctx);
        assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
    }

    #[test]
    fn test_zero_extend() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8);
        let result = a.zero_extend(32, &ctx);
        assert_eq!(result.as_u64(), Some(0xFF));
    }

    #[test]
    fn test_truncate() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0x12345678, 32);
        let result = a.truncate(8, &ctx);
        assert_eq!(result.as_u64(), Some(0x78));
    }

    #[test]
    fn test_extract() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xABCD, 16);
        let result = a.extract(11, 4, &ctx);
        assert_eq!(result.width(), 8);
        assert_eq!(result.as_u64(), Some(0xBC));
    }

    #[test]
    fn test_concat() {
        let ctx = SymContext::new_mock();
        let hi = RustBV::concrete(0xAB, 8);
        let lo = RustBV::concrete(0xCD, 8);
        let result = hi.concat(&lo, &ctx);
        assert_eq!(result.width(), 16);
        assert_eq!(result.as_u64(), Some(0xABCD));
    }

    #[test]
    fn test_comparison_unsigned() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(10, 32);

        assert_eq!(a.ult(&b, &ctx).as_u64(), Some(1));
        assert_eq!(a.ule(&b, &ctx).as_u64(), Some(1));
        assert_eq!(a.ugt(&b, &ctx).as_u64(), Some(0));
        assert_eq!(a.uge(&b, &ctx).as_u64(), Some(0));
    }

    #[test]
    fn test_comparison_signed() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8); // -1 signed
        let b = RustBV::concrete(1, 8);

        assert_eq!(a.slt(&b, &ctx).as_u64(), Some(1)); // -1 < 1
        assert_eq!(a.ult(&b, &ctx).as_u64(), Some(0)); // 255 > 1 unsigned
    }
}
