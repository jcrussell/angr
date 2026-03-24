//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that can be either:
//! - Concrete: A known fixed value
//! - Symbolic: Represents an unknown value (with optional Z3 backing)
//! - Expression: A compound expression with operation tree for reconstruction

use std::fmt;
use std::sync::Arc;

use super::SymContext;

/// Bitvector operation type for expression tree reconstruction.
///
/// This enum represents all operations that can be performed on bitvectors,
/// enabling reconstruction of claripy ASTs from Rust expression trees.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BVOp {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Neg,

    // Bitwise operations
    And,
    Or,
    Xor,
    Not,

    // Shift operations
    Shl,
    Lshr,
    Ashr,
    RotL,
    RotR,

    // Conversion operations
    ZeroExt(u32),   // Number of bits to extend
    SignExt(u32),   // Number of bits to extend
    Extract(u32, u32), // (high, low) bit indices
    Concat,

    // Comparison operations (return 1-bit result)
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,

    // Conditional
    Ite,

    // Utility
    Reverse,
    Clz,
    Ctz,
    Popcount,
}

impl BVOp {
    /// Get the claripy method name for this operation.
    pub fn claripy_method(&self) -> &'static str {
        match self {
            BVOp::Add => "__add__",
            BVOp::Sub => "__sub__",
            BVOp::Mul => "__mul__",
            BVOp::UDiv => "UDiv",
            BVOp::SDiv => "SDiv",
            BVOp::URem => "URem",
            BVOp::SRem => "SMod",
            BVOp::Neg => "__neg__",
            BVOp::And => "__and__",
            BVOp::Or => "__or__",
            BVOp::Xor => "__xor__",
            BVOp::Not => "__invert__",
            BVOp::Shl => "__lshift__",
            BVOp::Lshr => "LShR",
            BVOp::Ashr => "__rshift__",
            BVOp::RotL => "RotateLeft",
            BVOp::RotR => "RotateRight",
            BVOp::ZeroExt(_) => "ZeroExt",
            BVOp::SignExt(_) => "SignExt",
            BVOp::Extract(_, _) => "Extract",
            BVOp::Concat => "Concat",
            BVOp::Eq => "__eq__",
            BVOp::Ne => "__ne__",
            BVOp::Ult => "ULT",
            BVOp::Ule => "ULE",
            BVOp::Ugt => "UGT",
            BVOp::Uge => "UGE",
            BVOp::Slt => "SLT",
            BVOp::Sle => "SLE",
            BVOp::Sgt => "SGT",
            BVOp::Sge => "SGE",
            BVOp::Ite => "If",
            BVOp::Reverse => "Reverse",
            BVOp::Clz => "clz",
            BVOp::Ctz => "ctz",
            BVOp::Popcount => "popcount",
        }
    }

    /// Check if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            BVOp::Neg
                | BVOp::Not
                | BVOp::ZeroExt(_)
                | BVOp::SignExt(_)
                | BVOp::Extract(_, _)
                | BVOp::Reverse
                | BVOp::Clz
                | BVOp::Ctz
                | BVOp::Popcount
        )
    }

    /// Check if this is a binary operation.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            BVOp::Add
                | BVOp::Sub
                | BVOp::Mul
                | BVOp::UDiv
                | BVOp::SDiv
                | BVOp::URem
                | BVOp::SRem
                | BVOp::And
                | BVOp::Or
                | BVOp::Xor
                | BVOp::Shl
                | BVOp::Lshr
                | BVOp::Ashr
                | BVOp::RotL
                | BVOp::RotR
                | BVOp::Eq
                | BVOp::Ne
                | BVOp::Ult
                | BVOp::Ule
                | BVOp::Ugt
                | BVOp::Uge
                | BVOp::Slt
                | BVOp::Sle
                | BVOp::Sgt
                | BVOp::Sge
        )
    }

    /// Check if this is a ternary operation (ITE).
    pub fn is_ternary(&self) -> bool {
        matches!(self, BVOp::Ite)
    }
}

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
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// lifetime parameters on the Z3 AST.
#[derive(Clone)]
pub enum RustBV {
    /// A known concrete value.
    Concrete {
        /// The value, masked to `width` bits.
        value: u128,
        /// Width in bits (1, 8, 16, 32, 64, or 128).
        width: u32,
    },
    /// A symbolic value (backed by Z3 when available, otherwise just a name).
    /// This represents a leaf symbolic variable (e.g., BVS("x", 32)).
    Symbolic {
        /// Unique identifier for this symbolic value.
        id: u64,
        /// Width in bits.
        width: u32,
        /// Name (for debugging).
        name: String,
        /// Z3 AST when z3 feature is enabled.
        #[cfg(feature = "vex-engine-z3")]
        ast: z3::ast::BV,
    },
    /// Optimization: a symbolic value with a known concrete value.
    Constrained {
        /// The symbolic identifier.
        id: u64,
        /// The known concrete value.
        value: u128,
        /// Width in bits.
        width: u32,
    },
    /// A compound expression with operation tree for claripy reconstruction.
    ///
    /// This variant stores the operation and operands that created this value,
    /// enabling accurate conversion back to claripy ASTs without creating
    /// fresh symbolic variables.
    Expression {
        /// Unique identifier (typically EXPRESSION_ID sentinel).
        id: u64,
        /// Width in bits.
        width: u32,
        /// Z3 AST when z3 feature is enabled.
        #[cfg(feature = "vex-engine-z3")]
        ast: z3::ast::BV,
        /// The operation that created this expression.
        op: BVOp,
        /// The operands to this operation. Uses Arc for subtree sharing.
        operands: Vec<Arc<RustBV>>,
    },
}

impl RustBV {
    /// Sentinel ID for expression results (not real symbolic variables).
    /// Using u64::MAX avoids allocating new IDs for every intermediate operation.
    /// This improves solver compatibility since only true symbolic variables
    /// (inputs like `x`, `y`) get unique IDs that need tracking.
    pub const EXPRESSION_ID: u64 = u64::MAX;

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
    pub fn symbolic(ctx: &SymContext, name: &str, width: u32) -> Self {
        let id = ctx.next_id();
        #[cfg(feature = "vex-engine-z3")]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
                ast: z3::ast::BV::new_const(name, width),
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
            }
        }
    }

    /// Create a symbolic bitvector variable with a specific ID.
    ///
    /// This is used for identity preservation when the same symbol
    /// was previously imported from Python. By reusing the same ID,
    /// we ensure that constraints on the original symbol apply correctly.
    pub fn symbolic_with_id(id: u64, name: &str, width: u32) -> Self {
        #[cfg(feature = "vex-engine-z3")]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
                ast: z3::ast::BV::new_const(name, width),
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic {
                id,
                width,
                name: name.to_string(),
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
            RustBV::Expression { width, .. } => *width,
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
        matches!(self, RustBV::Symbolic { .. } | RustBV::Constrained { .. } | RustBV::Expression { .. })
    }

    /// Check if this value is an expression (compound symbolic).
    #[inline]
    pub fn is_expression(&self) -> bool {
        matches!(self, RustBV::Expression { .. })
    }

    /// Try to get the concrete value.
    #[inline]
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            RustBV::Concrete { value, .. } => Some(*value),
            RustBV::Constrained { value, .. } => Some(*value),
            RustBV::Symbolic { .. } => None,
            RustBV::Expression { .. } => None,
        }
    }

    /// Get the operation if this is an Expression variant.
    #[inline]
    pub fn op(&self) -> Option<&BVOp> {
        match self {
            RustBV::Expression { op, .. } => Some(op),
            _ => None,
        }
    }

    /// Get the operands if this is an Expression variant.
    #[inline]
    pub fn operands(&self) -> Option<&[Arc<RustBV>]> {
        match self {
            RustBV::Expression { operands, .. } => Some(operands),
            _ => None,
        }
    }

    /// Wrap this RustBV in an Arc for use as an operand.
    #[inline]
    pub fn into_arc(self) -> Arc<RustBV> {
        Arc::new(self)
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
    pub fn add(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_add(b), self.width()),
            _ => {
                // Build expression tree for claripy reconstruction
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: self.width(),
                    #[cfg(feature = "vex-engine-z3")]
                    ast: self.to_z3_ast().bvadd(&other.to_z3_ast()),
                    op: BVOp::Add,
                    operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
                }
            }
        }
    }

    /// Subtract two bitvectors.
    pub fn sub(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_sub(b), self.width()),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvsub(&other.to_z3_ast()),
                op: BVOp::Sub,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Multiply two bitvectors.
    pub fn mul(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_mul(b), self.width()),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvmul(&other.to_z3_ast()),
                op: BVOp::Mul,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned division.
    pub fn udiv(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    Self::concrete(a / b, self.width())
                }
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvudiv(&other.to_z3_ast()),
                op: BVOp::UDiv,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed division.
    pub fn sdiv(&self, other: &Self, _ctx: &SymContext) -> Self {
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
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvsdiv(&other.to_z3_ast()),
                op: BVOp::SDiv,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned remainder.
    pub fn urem(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self.clone()
                } else {
                    Self::concrete(a % b, self.width())
                }
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvurem(&other.to_z3_ast()),
                op: BVOp::URem,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed remainder.
    pub fn srem(&self, other: &Self, _ctx: &SymContext) -> Self {
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
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvsrem(&other.to_z3_ast()),
                op: BVOp::SRem,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Negate (two's complement).
    pub fn neg(&self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete((!v).wrapping_add(1), self.width()),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvneg(),
                op: BVOp::Neg,
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    /// Bitwise AND.
    pub fn and(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a & b, self.width()),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvand(&other.to_z3_ast()),
                op: BVOp::And,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Bitwise OR.
    pub fn or(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a | b, self.width()),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvor(&other.to_z3_ast()),
                op: BVOp::Or,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Bitwise XOR.
    pub fn xor(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a ^ b, self.width()),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvxor(&other.to_z3_ast()),
                op: BVOp::Xor,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Bitwise NOT.
    pub fn not(&self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(!v, self.width()),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvnot(),
                op: BVOp::Not,
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    // =========================================================================
    // Shift Operations
    // =========================================================================

    /// Logical shift left.
    pub fn shl(&self, amount: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shl(amt), self.width())
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvshl(&amount.to_z3_ast()),
                op: BVOp::Shl,
                operands: vec![Arc::new(self.clone()), Arc::new(amount.clone())],
            },
        }
    }

    /// Logical shift right.
    pub fn lshr(&self, amount: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shr(amt), self.width())
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvlshr(&amount.to_z3_ast()),
                op: BVOp::Lshr,
                operands: vec![Arc::new(self.clone()), Arc::new(amount.clone())],
            },
        }
    }

    /// Arithmetic shift right (sign-extending).
    pub fn ashr(&self, amount: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                let signed = sign_extend(v, self.width());
                Self::concrete((signed >> amt) as u128, self.width())
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvashr(&amount.to_z3_ast()),
                op: BVOp::Ashr,
                operands: vec![Arc::new(self.clone()), Arc::new(amount.clone())],
            },
        }
    }

    /// Rotate left.
    pub fn rotl(&self, amount: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v << amt) | (v >> (w - amt));
                Self::concrete(rotated, w)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvrotl(&amount.to_z3_ast()),
                op: BVOp::RotL,
                operands: vec![Arc::new(self.clone()), Arc::new(amount.clone())],
            },
        }
    }

    /// Rotate right.
    pub fn rotr(&self, amount: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v >> amt) | (v << (w - amt));
                Self::concrete(rotated, w)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().bvrotr(&amount.to_z3_ast()),
                op: BVOp::RotR,
                operands: vec![Arc::new(self.clone()), Arc::new(amount.clone())],
            },
        }
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit result).
    pub fn eq(&self, other: &Self, _ctx: &SymContext) -> Self {
        // Width mismatch guard — return concrete 0 instead of panicking
        if self.width() != other.width() {
            return Self::concrete(0, 1);
        }
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a == b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    use z3::ast::Ast;
                    let eq = self.to_z3_ast()._eq(&other.to_z3_ast());
                    eq.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Eq,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Inequality comparison (returns 1-bit result).
    pub fn ne(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a != b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    use z3::ast::Ast;
                    let eq = self.to_z3_ast()._eq(&other.to_z3_ast()).not();
                    eq.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Ne,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned less-than comparison.
    pub fn ult(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a < b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let lt = self.to_z3_ast().bvult(&other.to_z3_ast());
                    lt.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Ult,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned less-than-or-equal comparison.
    pub fn ule(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a <= b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let le = self.to_z3_ast().bvule(&other.to_z3_ast());
                    le.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Ule,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned greater-than comparison.
    pub fn ugt(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a > b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let gt = self.to_z3_ast().bvugt(&other.to_z3_ast());
                    gt.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Ugt,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Unsigned greater-than-or-equal comparison.
    pub fn uge(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a >= b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ge = self.to_z3_ast().bvuge(&other.to_z3_ast());
                    ge.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Uge,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed less-than comparison.
    pub fn slt(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed < b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let lt = self.to_z3_ast().bvslt(&other.to_z3_ast());
                    lt.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Slt,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed less-than-or-equal comparison.
    pub fn sle(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed <= b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let le = self.to_z3_ast().bvsle(&other.to_z3_ast());
                    le.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Sle,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed greater-than comparison.
    pub fn sgt(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed > b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let gt = self.to_z3_ast().bvsgt(&other.to_z3_ast());
                    gt.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Sgt,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    /// Signed greater-than-or-equal comparison.
    pub fn sge(&self, other: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                let a_signed = sign_extend(a, self.width());
                let b_signed = sign_extend(b, self.width());
                Self::concrete(if a_signed >= b_signed { 1 } else { 0 }, 1)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    let ge = self.to_z3_ast().bvsge(&other.to_z3_ast());
                    ge.ite(
                        &z3::ast::BV::from_u64(1, 1),
                        &z3::ast::BV::from_u64(0, 1),
                    )
                },
                op: BVOp::Sge,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    // =========================================================================
    // Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn zero_extend(&self, to_width: u32, _ctx: &SymContext) -> Self {
        if to_width <= self.width() {
            // No extension needed (or truncation — just return self)
            return self.clone();
        }
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().zero_ext(extend_bits),
                op: BVOp::ZeroExt(extend_bits),
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Sign-extend to a wider width.
    pub fn sign_extend(&self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width >= self.width());
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => {
                let extended = sign_extend_to(v, self.width(), to_width);
                Self::concrete(extended, to_width)
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().sign_ext(extend_bits),
                op: BVOp::SignExt(extend_bits),
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Truncate to a narrower width.
    pub fn truncate(&self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width <= self.width());
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().extract(to_width - 1, 0),
                op: BVOp::Extract(to_width - 1, 0),
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Extract bits [high:low] (inclusive).
    pub fn extract(&self, high: u32, low: u32, _ctx: &SymContext) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;
        match self.as_u128() {
            Some(v) => {
                let extracted = (v >> low) & ((1u128 << result_width) - 1);
                Self::concrete(extracted, result_width)
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: result_width,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().extract(high, low),
                op: BVOp::Extract(high, low),
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Concatenate two bitvectors (self becomes high bits).
    pub fn concat(&self, other: &Self, _ctx: &SymContext) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: result_width,
                #[cfg(feature = "vex-engine-z3")]
                ast: self.to_z3_ast().concat(&other.to_z3_ast()),
                op: BVOp::Concat,
                operands: vec![Arc::new(self.clone()), Arc::new(other.clone())],
            },
        }
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    /// If-then-else: returns `then_val` if `self` is non-zero, else `else_val`.
    pub fn ite(&self, then_val: &Self, else_val: &Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(then_val.width(), else_val.width());
        match self.as_u128() {
            Some(v) => {
                if v != 0 {
                    then_val.clone()
                } else {
                    else_val.clone()
                }
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: then_val.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    use z3::ast::Ast;
                    let cond = self
                        .to_z3_ast()
                        ._eq(&z3::ast::BV::from_u64(0, self.width()))
                        .not();
                    cond.ite(&then_val.to_z3_ast(), &else_val.to_z3_ast())
                },
                op: BVOp::Ite,
                operands: vec![
                    Arc::new(self.clone()),
                    Arc::new(then_val.clone()),
                    Arc::new(else_val.clone()),
                ],
            },
        }
    }

    /// Count leading zeros.
    pub fn clz(&self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let leading = if v == 0 {
                    self.width()
                } else {
                    (self.width() - (128 - v.leading_zeros())).max(0)
                };
                Self::concrete(leading as u128, self.width())
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: {
                    // Build CLZ symbolically - this is complex
                    // For now, just return a symbolic value
                    z3::ast::BV::new_const("clz", self.width())
                },
                op: BVOp::Clz,
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Count trailing zeros.
    pub fn ctz(&self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let trailing = if v == 0 {
                    self.width()
                } else {
                    v.trailing_zeros().min(self.width())
                };
                Self::concrete(trailing as u128, self.width())
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: z3::ast::BV::new_const("ctz", self.width()),
                op: BVOp::Ctz,
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    /// Population count (number of set bits).
    pub fn popcount(&self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(v.count_ones() as u128, self.width()),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: self.width(),
                #[cfg(feature = "vex-engine-z3")]
                ast: z3::ast::BV::new_const("popcount", self.width()),
                op: BVOp::Popcount,
                operands: vec![Arc::new(self.clone())],
            },
        }
    }

    // =========================================================================
    // Z3 Integration (when feature is enabled)
    // =========================================================================

    /// Convert this RustBV to a Z3 AST.
    ///
    /// With z3-rs 0.19+, the context is thread-local so we don't need
    /// to pass it explicitly.
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_ast(&self) -> z3::ast::BV {
        use z3::ast::Ast;
        match self {
            RustBV::Concrete { value, width } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(*value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(*value as u64, 64);
                    let hi = z3::ast::BV::from_u64((*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
            RustBV::Symbolic { ast, .. } => ast.clone(),
            RustBV::Constrained { value, width, .. } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(*value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(*value as u64, 64);
                    let hi = z3::ast::BV::from_u64((*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
            RustBV::Expression { ast, .. } => ast.clone(),
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

impl fmt::Debug for RustBV {
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
            RustBV::Expression { width, op, operands, .. } => {
                write!(f, "Expression({:?}, {}, {} operands)", op, width, operands.len())
            }
        }
    }
}

impl fmt::Display for RustBV {
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
            RustBV::Expression { width, op, .. } => {
                write!(f, "<BV{} {:?}>", width, op)
            }
        }
    }
}

impl PartialEq for RustBV {
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
