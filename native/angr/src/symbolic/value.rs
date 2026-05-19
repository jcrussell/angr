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
    ZeroExt(u32),      // Number of bits to extend
    SignExt(u32),      // Number of bits to extend
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

    // Floating-point operations (Z3 FP theory).
    // Operands are RustBVs holding the IEEE-754 bit pattern at the given
    // precision; the op is applied via Z3 FP and the result is converted
    // back to the IEEE bit-vector. Round-to-nearest-even is used for
    // arith; conversions and comparison results follow IEEE-754.
    Float { kind: FloatOpKind, prec: FloatPrec },
}

/// IEEE-754 precision for symbolic float operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatPrec {
    /// 32-bit single precision (8 ebits, 24 sbits).
    F32,
    /// 64-bit double precision (11 ebits, 53 sbits).
    F64,
}

impl FloatPrec {
    /// Width of the IEEE bit-vector encoding for this precision.
    #[inline]
    pub fn bits(&self) -> u32 {
        match self {
            FloatPrec::F32 => 32,
            FloatPrec::F64 => 64,
        }
    }
}

/// Kinds of symbolic float operations expressible via Z3 FP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatOpKind {
    // Binary arithmetic (round to nearest, ties to even).
    Add,
    Sub,
    Mul,
    Div,
    // Unary.
    Sqrt,
    Neg,
    Abs,
    // Ternary fused multiply-add / multiply-sub: a*b + c, a*b - c.
    Fma,
    Fms,
    // Comparisons: produce 1-bit BV (1 = true).
    CmpEq,
    CmpLt,
    CmpLe,
    /// Round to integer with rounding mode.
    /// operand[0] = rm BV (32-bit, VEX rounding mode 0..3),
    /// operand[1] = value BV (prec.bits()).
    RoundToInt,
    /// Convert int (signed/unsigned 2's complement) BV to FP at `prec`.
    /// operand[0] = src BV at src_bits. RNE rounding implicit.
    ConvertItoF {
        src_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement).
    /// operand[0] = src FP BV at prec.bits(). RNE rounding implicit.
    ConvertFtoI {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement)
    /// with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = src FP BV at prec.bits().
    ConvertFtoIRm {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `src_prec` to FP at `prec`. RNE rounding implicit.
    /// operand[0] = src FP BV at src_prec.bits().
    ConvertFtoF {
        src_prec: FloatPrec,
    },
    /// Convert FP at `src_prec` to FP at `prec` with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = src FP BV at src_prec.bits().
    ConvertFtoFRm {
        src_prec: FloatPrec,
    },
    /// Binary FP arithmetic with explicit rounding mode.
    /// operand[0] = rm BV (32-bit, VEX rm low-2-bits 0..3),
    /// operand[1] = a BV at prec.bits(), operand[2] = b BV at prec.bits().
    AddRm,
    SubRm,
    MulRm,
    DivRm,
    /// Unary FP sqrt with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = a BV at prec.bits().
    SqrtRm,
}

impl FloatOpKind {
    /// Number of operands this op consumes.
    #[inline]
    pub fn arity(&self) -> usize {
        match self {
            FloatOpKind::Sqrt
            | FloatOpKind::Neg
            | FloatOpKind::Abs
            | FloatOpKind::ConvertItoF { .. }
            | FloatOpKind::ConvertFtoI { .. }
            | FloatOpKind::ConvertFtoF { .. } => 1,
            FloatOpKind::Add
            | FloatOpKind::Sub
            | FloatOpKind::Mul
            | FloatOpKind::Div
            | FloatOpKind::CmpEq
            | FloatOpKind::CmpLt
            | FloatOpKind::CmpLe
            | FloatOpKind::RoundToInt
            | FloatOpKind::ConvertFtoIRm { .. }
            | FloatOpKind::ConvertFtoFRm { .. }
            | FloatOpKind::SqrtRm => 2,
            FloatOpKind::Fma
            | FloatOpKind::Fms
            | FloatOpKind::AddRm
            | FloatOpKind::SubRm
            | FloatOpKind::MulRm
            | FloatOpKind::DivRm => 3,
        }
    }

    /// Whether this op produces a 1-bit (boolean) result instead of a BV.
    #[inline]
    pub fn is_compare(&self) -> bool {
        matches!(
            self,
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe
        )
    }

    /// Width (in bits) of the BV result for this op, given the `prec` field
    /// of the enclosing `BVOp::Float`. Compares are 1-bit; FtoI conversions
    /// take their result width from `dst_bits`; everything else returns the
    /// IEEE encoding width of `prec`.
    #[inline]
    pub fn result_bits(&self, prec: FloatPrec) -> u32 {
        match self {
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe => 1,
            FloatOpKind::ConvertFtoI { dst_bits, .. }
            | FloatOpKind::ConvertFtoIRm { dst_bits, .. } => *dst_bits as u32,
            _ => prec.bits(),
        }
    }
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
            // No clean claripy mapping — float ops returning to Python fall
            // back to fresh symbolic in rustbv_to_claripy (constraint info
            // stays in Z3 within the Rust engine).
            BVOp::Float { .. } => "fpOp",
        }
    }

    /// Check if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 1;
        }
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
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 2;
        }
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

    /// Check if this is a ternary operation (ITE or FP fused MAdd/MSub).
    pub fn is_ternary(&self) -> bool {
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 3;
        }
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
        /// Name (for debugging). `Arc<str>` so cloning a Symbolic is alloc-free —
        /// hot on the RdTmp path where a temp slot is read multiple times per block.
        name: Arc<str>,
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
    /// fresh symbolic variables. Z3 ASTs are computed lazily on demand
    /// via `to_z3_ast()` rather than eagerly at construction time.
    Expression {
        /// Unique identifier (typically EXPRESSION_ID sentinel).
        id: u64,
        /// Width in bits.
        width: u32,
        /// The operation that created this expression.
        op: BVOp,
        /// The operands to this operation. `Arc<[T]>` lets cloning the outer
        /// Expression bump a single refcount (alloc-free; matters on the RdTmp
        /// path). Operands are stored inline rather than wrapped in `Arc<RustBV>`
        /// so each Expression construction is one allocation instead of N+1.
        operands: Arc<[RustBV]>,
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
        RustBV::Concrete {
            value: Self::all_ones_mask(width),
            width,
        }
    }

    /// Return the all-ones mask for a given bit width.
    #[inline]
    fn all_ones_mask(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }

    /// Create a symbolic bitvector variable.
    ///
    /// Accepts anything String-like via `AsRef<str>`. The internal name is
    /// stored as `Arc<str>` so cloning a Symbolic is alloc-free, which matters
    /// on the RdTmp path where a temp slot is read multiple times per block.
    pub fn symbolic(ctx: &SymContext, name: impl AsRef<str>, width: u32) -> Self {
        let id = ctx.next_id();
        let name: Arc<str> = Arc::from(name.as_ref());
        #[cfg(feature = "vex-engine-z3")]
        {
            let ast = z3::ast::BV::new_const(&*name, width);
            RustBV::Symbolic {
                id,
                width,
                name,
                ast,
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic { id, width, name }
        }
    }

    /// Create a symbolic bitvector variable with a specific ID.
    ///
    /// This is used for identity preservation when the same symbol
    /// was previously imported from Python. By reusing the same ID,
    /// we ensure that constraints on the original symbol apply correctly.
    pub fn symbolic_with_id(id: u64, name: impl AsRef<str>, width: u32) -> Self {
        let name: Arc<str> = Arc::from(name.as_ref());
        #[cfg(feature = "vex-engine-z3")]
        {
            let ast = z3::ast::BV::new_const(&*name, width);
            RustBV::Symbolic {
                id,
                width,
                name,
                ast,
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic { id, width, name }
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
        matches!(
            self,
            RustBV::Symbolic { .. } | RustBV::Constrained { .. } | RustBV::Expression { .. }
        )
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
    pub fn operands(&self) -> Option<&[RustBV]> {
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
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Add,
                    operands: Arc::<[RustBV]>::from([self, other]),
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
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_mul(b), self.width()),
            // x * 0 → 0
            (_, Some(0)) | (Some(0), _) => Self::zero(self.width()),
            // x * 1 → x
            (None, Some(1)) => self,
            // 1 * x → x
            (Some(1), None) => other,
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Mul,
                    operands: Arc::<[RustBV]>::from([self, other]),
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
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::And,
                    operands: Arc::<[RustBV]>::from([self, other]),
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
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Or,
                    operands: Arc::<[RustBV]>::from([self, other]),
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
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Xor,
                    operands: Arc::<[RustBV]>::from([self, other]),
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
        debug_assert!(w % 8 == 0, "reverse requires byte-aligned width");
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
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shl(amt), self.width())
            }
            // x << 0 → x
            (None, Some(0)) => self,
            // 0 << x → 0
            (Some(0), None) => Self::zero(self.width()),
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Shl,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
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
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                Self::concrete(v.wrapping_shr(amt), self.width())
            }
            // x >> 0 → x
            (None, Some(0)) => self,
            // 0 >> x → 0
            (Some(0), None) => Self::zero(self.width()),
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Lshr,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
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
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(self.width());
                let signed = sign_extend(v, self.width());
                Self::concrete((signed >> amt) as u128, self.width())
            }
            // x >>> 0 → x
            (None, Some(0)) => self,
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Ashr,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
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
    pub fn eq_into(self, other: Self, _ctx: &SymContext) -> Self {
        // Width mismatch guard — return concrete 0 instead of panicking
        if self.width() != other.width() {
            return Self::concrete(0, 1);
        }
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a == b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Eq,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Inequality comparison (returns 1-bit result).
    #[inline]
    pub fn ne(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ne_into(other.clone(), ctx)
    }

    /// Inequality comparison, consuming both arguments.
    #[inline]
    pub fn ne_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a != b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Ne,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Unsigned less-than comparison.
    #[inline]
    pub fn ult(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ult_into(other.clone(), ctx)
    }

    /// Unsigned less-than, consuming both arguments.
    #[inline]
    pub fn ult_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a < b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Ult,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Unsigned less-than-or-equal comparison.
    #[inline]
    pub fn ule(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ule_into(other.clone(), ctx)
    }

    /// Unsigned less-than-or-equal, consuming both arguments.
    #[inline]
    pub fn ule_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a <= b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Ule,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Unsigned greater-than comparison.
    #[inline]
    pub fn ugt(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ugt_into(other.clone(), ctx)
    }

    /// Unsigned greater-than, consuming both arguments.
    #[inline]
    pub fn ugt_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a > b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Ugt,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Unsigned greater-than-or-equal comparison.
    #[inline]
    pub fn uge(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().uge_into(other.clone(), ctx)
    }

    /// Unsigned greater-than-or-equal, consuming both arguments.
    #[inline]
    pub fn uge_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a >= b { 1 } else { 0 }, 1),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: 1,
                op: BVOp::Uge,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Signed less-than comparison.
    #[inline]
    pub fn slt(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().slt_into(other.clone(), ctx)
    }

    /// Signed less-than, consuming both arguments.
    #[inline]
    pub fn slt_into(self, other: Self, _ctx: &SymContext) -> Self {
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
                op: BVOp::Slt,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Signed less-than-or-equal comparison.
    #[inline]
    pub fn sle(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sle_into(other.clone(), ctx)
    }

    /// Signed less-than-or-equal, consuming both arguments.
    #[inline]
    pub fn sle_into(self, other: Self, _ctx: &SymContext) -> Self {
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
                op: BVOp::Sle,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Signed greater-than comparison.
    #[inline]
    pub fn sgt(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sgt_into(other.clone(), ctx)
    }

    /// Signed greater-than, consuming both arguments.
    #[inline]
    pub fn sgt_into(self, other: Self, _ctx: &SymContext) -> Self {
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
                op: BVOp::Sgt,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

    /// Signed greater-than-or-equal comparison.
    #[inline]
    pub fn sge(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sge_into(other.clone(), ctx)
    }

    /// Signed greater-than-or-equal, consuming both arguments.
    #[inline]
    pub fn sge_into(self, other: Self, _ctx: &SymContext) -> Self {
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
                op: BVOp::Sge,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
        }
    }

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
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                op: BVOp::Extract(to_width - 1, 0),
                operands: Arc::<[RustBV]>::from([self]),
            },
        }
    }

    /// Extract bits [high:low] (inclusive).
    #[inline]
    pub fn extract(&self, high: u32, low: u32, ctx: &SymContext) -> Self {
        self.clone().extract_into(high, low, ctx)
    }

    /// Extract bits [high:low] (inclusive), consuming the argument.
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

                // Rule 3: Extract(Reverse(x)) with byte-aligned bounds
                // → Extract(width-1-lo, width-1-hi, x) — eliminates the Reverse
                BVOp::Reverse if operands[0].width() % 8 == 0 && high % 8 == 7 && low % 8 == 0 => {
                    let w = operands[0].width();
                    return operands[0].extract(w - 1 - low, w - 1 - high, _ctx);
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
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: result_width,
                op: BVOp::Concat,
                operands: Arc::<[RustBV]>::from([self, other]),
            },
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

        RustBV::Expression {
            id: Self::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: Arc::<[RustBV]>::from([self.clone()]),
        }
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
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: result_width,
                op: BVOp::Concat,
                operands: Arc::<[RustBV]>::from([self.clone(), other.clone()]),
            },
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
                    (self.width() - (128 - v.leading_zeros())).max(0)
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
        super::context::record_z3_ast_build();
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
            return cached.clone();
        }
        let result = match self {
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
        super::context::record_z3_ast_build();
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
                        .eq(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ne => operands[0]
                        .to_z3_ast_cached(cache)
                        .eq(&operands[1].to_z3_ast_cached(cache))
                        .not(),
                    BVOp::Ult => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvult(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ule => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvule(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ugt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvugt(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Uge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvuge(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Slt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvslt(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sle => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsle(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sgt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsgt(&operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsge(&operands[1].to_z3_ast_cached(cache)),
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

    /// Build a Z3 AST from an operation and its operands.
    /// Called lazily when a Z3 AST is actually needed (e.g., for constraint solving).
    #[cfg(feature = "vex-engine-z3")]
    #[allow(dead_code)]
    fn build_z3_ast(op: &BVOp, operands: &[RustBV], _width: u32) -> z3::ast::BV {
        match op {
            // Arithmetic
            BVOp::Add => operands[0].to_z3_ast().bvadd(&operands[1].to_z3_ast()),
            BVOp::Sub => operands[0].to_z3_ast().bvsub(&operands[1].to_z3_ast()),
            BVOp::Mul => operands[0].to_z3_ast().bvmul(&operands[1].to_z3_ast()),
            BVOp::UDiv => operands[0].to_z3_ast().bvudiv(&operands[1].to_z3_ast()),
            BVOp::SDiv => operands[0].to_z3_ast().bvsdiv(&operands[1].to_z3_ast()),
            BVOp::URem => operands[0].to_z3_ast().bvurem(&operands[1].to_z3_ast()),
            BVOp::SRem => operands[0].to_z3_ast().bvsrem(&operands[1].to_z3_ast()),
            BVOp::Neg => operands[0].to_z3_ast().bvneg(),

            // Bitwise
            BVOp::And => operands[0].to_z3_ast().bvand(&operands[1].to_z3_ast()),
            BVOp::Or => operands[0].to_z3_ast().bvor(&operands[1].to_z3_ast()),
            BVOp::Xor => operands[0].to_z3_ast().bvxor(&operands[1].to_z3_ast()),
            BVOp::Not => operands[0].to_z3_ast().bvnot(),

            // Shifts
            BVOp::Shl => operands[0].to_z3_ast().bvshl(&operands[1].to_z3_ast()),
            BVOp::Lshr => operands[0].to_z3_ast().bvlshr(&operands[1].to_z3_ast()),
            BVOp::Ashr => operands[0].to_z3_ast().bvashr(&operands[1].to_z3_ast()),
            BVOp::RotL => operands[0].to_z3_ast().bvrotl(&operands[1].to_z3_ast()),
            BVOp::RotR => operands[0].to_z3_ast().bvrotr(&operands[1].to_z3_ast()),

            // Comparisons (return 1-bit BV: If(cmp, BV(1,1), BV(0,1)))
            BVOp::Eq => {
                let cmp = operands[0].to_z3_ast().eq(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ne => {
                let cmp = operands[0].to_z3_ast().eq(&operands[1].to_z3_ast()).not();
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ult => {
                let cmp = operands[0].to_z3_ast().bvult(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ule => {
                let cmp = operands[0].to_z3_ast().bvule(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ugt => {
                let cmp = operands[0].to_z3_ast().bvugt(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Uge => {
                let cmp = operands[0].to_z3_ast().bvuge(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Slt => {
                let cmp = operands[0].to_z3_ast().bvslt(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sle => {
                let cmp = operands[0].to_z3_ast().bvsle(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sgt => {
                let cmp = operands[0].to_z3_ast().bvsgt(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sge => {
                let cmp = operands[0].to_z3_ast().bvsge(&operands[1].to_z3_ast());
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }

            // Conversions
            BVOp::ZeroExt(bits) => operands[0].to_z3_ast().zero_ext(*bits),
            BVOp::SignExt(bits) => {
                // Decompose SignExt into Concat of sign-bit extracts to match
                // claripy's canonical form. SignExt(n, x) becomes:
                //   Concat(Extract(msb,msb,x), ..., Extract(msb,msb,x), x)
                // where msb is the sign bit position. This flat representation
                // allows Z3's max_bv_sharing tactic to share sign-bit extracts
                // across multiple constraints (e.g., hackcon's 20 linear equations).
                let inner = operands[0].to_z3_ast();
                let inner_width = operands[0].width();
                let sign_bit = inner.extract(inner_width - 1, inner_width - 1);
                let mut result = inner;
                for _ in 0..*bits {
                    result = sign_bit.clone().concat(&result);
                }
                result
            }
            BVOp::Extract(high, low) => operands[0].to_z3_ast().extract(*high, *low),
            BVOp::Concat => {
                // Flatten nested left-associative Concat trees into a single
                // right-associative chain. Left-associative trees from memory loads
                // (Concat(Concat(Concat(b0,b1),b2),b3)) produce deeper Z3 ASTs.
                // Collect all leaves, then build right-to-left for better sharing.
                fn collect_concat_leaves(bv: &RustBV, leaves: &mut Vec<z3::ast::BV>) {
                    if let RustBV::Expression {
                        op: BVOp::Concat,
                        operands,
                        ..
                    } = bv
                    {
                        collect_concat_leaves(&operands[0], leaves);
                        collect_concat_leaves(&operands[1], leaves);
                    } else {
                        leaves.push(bv.to_z3_ast());
                    }
                }
                let mut leaves = Vec::new();
                collect_concat_leaves(&operands[0], &mut leaves);
                collect_concat_leaves(&operands[1], &mut leaves);
                // Build right-associative: leaves[0].concat(leaves[1].concat(...))
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
                    .to_z3_ast()
                    .eq(&z3::ast::BV::from_u64(0, operands[0].width()))
                    .not();
                cond.ite(&operands[1].to_z3_ast(), &operands[2].to_z3_ast())
            }

            // Byte reverse
            BVOp::Reverse => {
                // Rule 5: Reverse(Concat(a,b)) → Concat(Reverse(b), Reverse(a))
                // Distribute reverse across concat parts before building Z3 AST
                if let RustBV::Expression {
                    op: BVOp::Concat,
                    operands: _inner_ops,
                    ..
                } = &operands[0]
                {
                    fn collect_concat_parts(bv: &RustBV, parts: &mut Vec<RustBV>) {
                        if let RustBV::Expression {
                            op: BVOp::Concat,
                            operands,
                            ..
                        } = bv
                        {
                            collect_concat_parts(&operands[0], parts);
                            collect_concat_parts(&operands[1], parts);
                        } else {
                            parts.push(bv.clone());
                        }
                    }
                    let mut parts = Vec::new();
                    collect_concat_parts(&operands[0], &mut parts);
                    // Reverse the order, then reverse each part individually
                    parts.reverse();
                    let reversed_asts: Vec<z3::ast::BV> = parts
                        .iter()
                        .map(|p| {
                            // Build reverse of each part
                            let w = p.width();
                            if w == 8 {
                                p.to_z3_ast() // Single byte — no reverse needed
                            } else {
                                Self::build_z3_ast(&BVOp::Reverse, &[p.clone()], w)
                            }
                        })
                        .collect();
                    let mut result = reversed_asts[0].clone();
                    for ast in &reversed_asts[1..] {
                        result = result.concat(ast);
                    }
                    return result;
                }

                let ast = operands[0].to_z3_ast();
                let w = operands[0].width();
                if w % 8 == 0 && w >= 16 {
                    // Emit Concat(extract[7:0,x], extract[15:8,x], ..., extract[N-1:N-8,x]),
                    // matching claripy's canonical Reverse shape. The HIGH byte of the
                    // reversed value is the LOW byte of x, so the first part (extract[7:0])
                    // accumulates as the highest bits via left-associative `result.concat(part)`.
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

            // Bit counting ops - create fresh symbolic (can't express in Z3 BV theory)
            BVOp::Clz => z3::ast::BV::new_const("clz", _width),
            BVOp::Ctz => z3::ast::BV::new_const("ctz", _width),
            BVOp::Popcount => z3::ast::BV::new_const("popcount", _width),

            // Floating-point — defer to the cached path for the actual work.
            BVOp::Float { kind, prec } => {
                let mut cache = std::collections::HashMap::new();
                Self::build_fp_z3_ast_cached(*kind, *prec, operands, &mut cache)
            }
        }
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
                .bvadd(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Sub => operands[0]
                .to_z3_ast_cached(cache)
                .bvsub(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Mul => operands[0]
                .to_z3_ast_cached(cache)
                .bvmul(&operands[1].to_z3_ast_cached(cache)),
            BVOp::UDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvudiv(&operands[1].to_z3_ast_cached(cache)),
            BVOp::SDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvsdiv(&operands[1].to_z3_ast_cached(cache)),
            BVOp::URem => operands[0]
                .to_z3_ast_cached(cache)
                .bvurem(&operands[1].to_z3_ast_cached(cache)),
            BVOp::SRem => operands[0]
                .to_z3_ast_cached(cache)
                .bvsrem(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Neg => operands[0].to_z3_ast_cached(cache).bvneg(),

            // Bitwise
            BVOp::And => operands[0]
                .to_z3_ast_cached(cache)
                .bvand(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Or => operands[0]
                .to_z3_ast_cached(cache)
                .bvor(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Xor => operands[0]
                .to_z3_ast_cached(cache)
                .bvxor(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Not => operands[0].to_z3_ast_cached(cache).bvnot(),

            // Shifts
            BVOp::Shl => operands[0]
                .to_z3_ast_cached(cache)
                .bvshl(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Lshr => operands[0]
                .to_z3_ast_cached(cache)
                .bvlshr(&operands[1].to_z3_ast_cached(cache)),
            BVOp::Ashr => operands[0]
                .to_z3_ast_cached(cache)
                .bvashr(&operands[1].to_z3_ast_cached(cache)),
            BVOp::RotL => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotl(&operands[1].to_z3_ast_cached(cache)),
            BVOp::RotR => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotr(&operands[1].to_z3_ast_cached(cache)),

            // Comparisons (return 1-bit BV: If(cmp, BV(1,1), BV(0,1)))
            BVOp::Eq => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ne => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(&operands[1].to_z3_ast_cached(cache))
                    .not();
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ult => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvult(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ule => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvule(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ugt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvugt(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Uge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvuge(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Slt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvslt(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sle => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsle(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sgt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsgt(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsge(&operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }

            // Conversions
            BVOp::ZeroExt(bits) => operands[0].to_z3_ast_cached(cache).zero_ext(*bits),
            BVOp::SignExt(bits) => {
                let inner = operands[0].to_z3_ast_cached(cache);
                let inner_width = operands[0].width();
                let sign_bit = inner.extract(inner_width - 1, inner_width - 1);
                let mut result = inner;
                for _ in 0..*bits {
                    result = sign_bit.clone().concat(&result);
                }
                result
            }
            BVOp::Extract(high, low) => operands[0].to_z3_ast_cached(cache).extract(*high, *low),
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
                    .eq(&z3::ast::BV::from_u64(0, operands[0].width()))
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
                                Self::build_z3_ast_cached(&BVOp::Reverse, &[p.clone()], w, cache)
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
                if w % 8 == 0 && w >= 16 {
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
            Z3_mk_fpa_leq, Z3_mk_fpa_lt, Z3_mk_fpa_mul, Z3_mk_fpa_neg, Z3_mk_fpa_sqrt,
            Z3_mk_fpa_sub, Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv,
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
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, bv_ast.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
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
            let cmp_bool = unsafe { Bool::wrap(&z3_ctx, result_raw) };
            cmp_bool.ite(&BV::from_u64(1, 1), &BV::from_u64(0, 1))
        } else {
            let result_fp = unsafe { Float::wrap(&z3_ctx, result_raw) };
            let ieee_bv_raw = unsafe {
                Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                    .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
            };
            unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
        }
    }

    /// Build the Z3 AST for a `FloatOpKind::RoundToInt` operation.
    ///
    /// VEX rounding modes (low 2 bits of operand[0]):
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
            let raw = unsafe {
                Z3_mk_fpa_round_to_integral(raw_ctx, rm.get_z3_ast(), value_raw)
                    .expect("Z3_mk_fpa_round_to_integral returned NULL")
            };
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

        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for an FP arithmetic op with explicit rounding mode
    /// (`AddRm`/`SubRm`/`MulRm`/`DivRm`/`SqrtRm`). operand[0] is the rm BV;
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
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, z3.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
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

        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertItoF`. operand[0] is a BV
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

        let fp_raw = unsafe {
            if signed {
                Z3_mk_fpa_to_fp_signed(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_signed returned NULL")
            } else {
                Z3_mk_fpa_to_fp_unsigned(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_unsigned returned NULL")
            }
        };
        let fp_wrap = unsafe { Float::wrap(&z3_ctx, fp_raw) };
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, fp_wrap.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertFtoI` (no rm operand,
    /// implicit RNE) or `FloatOpKind::ConvertFtoIRm` (operand[0] = rm BV,
    /// operand[1] = FP value). The result is a BV of width `dst_bits`,
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
            let raw = unsafe {
                if signed {
                    Z3_mk_fpa_to_sbv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_sbv returned NULL")
                } else {
                    Z3_mk_fpa_to_ubv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_ubv returned NULL")
                }
            };
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
    /// `FloatOpKind::ConvertFtoFRm` (operand[0] = rm BV, operand[1] = FP).
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
            let raw = unsafe {
                Z3_mk_fpa_to_fp_float(raw_ctx, rm.get_z3_ast(), value_raw, dst_raw_sort)
                    .expect("Z3_mk_fpa_to_fp_float returned NULL")
            };
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

        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
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
            RustBV::Expression {
                width,
                op,
                operands,
                ..
            } => {
                write!(
                    f,
                    "Expression({:?}, {}, {} operands)",
                    op,
                    width,
                    operands.len()
                )
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

    // =========================================================================
    // Expression simplification tests
    // =========================================================================

    #[test]
    fn test_add_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x + 0 → x (should return symbolic, not expression)
        let r1 = x.add(&zero, &ctx);
        assert!(matches!(r1, RustBV::Symbolic { .. }));
        // 0 + x → x
        let r2 = zero.add(&x, &ctx);
        assert!(matches!(r2, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_sub_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x - 0 → x
        let r = x.sub(&zero, &ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_mul_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        let one = RustBV::concrete(1, 32);
        // x * 0 → 0
        assert_eq!(x.mul(&zero, &ctx).as_u128(), Some(0));
        // 0 * x → 0
        assert_eq!(zero.mul(&x, &ctx).as_u128(), Some(0));
        // x * 1 → x
        assert!(matches!(x.mul(&one, &ctx), RustBV::Symbolic { .. }));
        // 1 * x → x
        assert!(matches!(one.mul(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_and_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zero = RustBV::zero(8);
        let ones = RustBV::ones(8);
        // x & 0 → 0
        assert_eq!(x.and(&zero, &ctx).as_u128(), Some(0));
        // x & 0xFF → x
        assert!(matches!(x.and(&ones, &ctx), RustBV::Symbolic { .. }));
        // 0xFF & x → x
        assert!(matches!(ones.and(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_or_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zero = RustBV::zero(8);
        let ones = RustBV::ones(8);
        // x | 0 → x
        assert!(matches!(x.or(&zero, &ctx), RustBV::Symbolic { .. }));
        // 0 | x → x
        assert!(matches!(zero.or(&x, &ctx), RustBV::Symbolic { .. }));
        // x | 0xFF → 0xFF
        assert_eq!(x.or(&ones, &ctx).as_u128(), Some(0xFF));
    }

    #[test]
    fn test_xor_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x ^ 0 → x
        assert!(matches!(x.xor(&zero, &ctx), RustBV::Symbolic { .. }));
        // 0 ^ x → x
        assert!(matches!(zero.xor(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_not_double_negation() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // not(not(x)) → x
        let r = x.not(&ctx).not(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_neg_double_negation() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // neg(neg(x)) → x
        let r = x.neg(&ctx).neg(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_reverse_double() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // reverse(reverse(x)) → x
        let r = x.reverse(&ctx).reverse(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_shift_by_zero() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x << 0 → x
        assert!(matches!(x.shl(&zero, &ctx), RustBV::Symbolic { .. }));
        // x >> 0 → x
        assert!(matches!(x.lshr(&zero, &ctx), RustBV::Symbolic { .. }));
        // x >>> 0 → x
        assert!(matches!(x.ashr(&zero, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_shift_zero_value() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // 0 << x → 0
        assert_eq!(zero.shl(&x, &ctx).as_u128(), Some(0));
        // 0 >> x → 0
        assert_eq!(zero.lshr(&x, &ctx).as_u128(), Some(0));
    }

    #[test]
    fn test_sign_extend_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // sign_extend to same width → x
        let r = x.sign_extend(32, &ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
        assert_eq!(r.width(), 32);
    }

    #[test]
    fn test_extract_zero_ext() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let ext = x.zero_extend(32, &ctx); // 8-bit → 32-bit
        // Extract low 8 bits → original x
        let lo = ext.extract(7, 0, &ctx);
        assert!(matches!(lo, RustBV::Symbolic { .. }));
        assert_eq!(lo.width(), 8);
        // Extract high 8 bits → 0
        let hi = ext.extract(31, 24, &ctx);
        assert_eq!(hi.as_u128(), Some(0));
    }

    #[test]
    fn test_extract_sign_ext_low_bits() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let ext = x.sign_extend(32, &ctx);
        // Extract low 8 bits → original x
        let lo = ext.extract(7, 0, &ctx);
        assert!(matches!(lo, RustBV::Symbolic { .. }));
        assert_eq!(lo.width(), 8);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_32bit_leaf() {
        // Reverse(x) over a symbolic leaf should produce byte-reversed value
        // through the Z3 emission path.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx); // Expression { Reverse, [x] }
        // Pin x = 0x11223344; expect Reverse(x) = 0x44332211
        let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(&z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0x44332211));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_64bit_leaf() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 64);
        let rev = x.reverse(&ctx);
        let pinned = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(&z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0xEFCDAB8967452301));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_16bit_leaf() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 16);
        let rev = x.reverse(&ctx);
        let pinned = x.eq(&RustBV::concrete(0xAABB, 16), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(&z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0xBBAA));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_concat_of_bytes() {
        // Reverse(Concat(b0, b1, ..., b7)) with independent byte BVSes.
        // Memory loads in angr typically produce this shape; the result should
        // be Concat(b7, b6, ..., b0) — the byte-reversed value.
        let ctx = SymContext::new_mock();
        let bytes: Vec<RustBV> = (0..8u32)
            .map(|i| RustBV::symbolic(&ctx, &format!("b{}", i), 8))
            .collect();
        // Build claripy-style Concat(b0, b1, ..., b7) with b0 as high.
        let mut concat = bytes[0].clone();
        for b in &bytes[1..] {
            concat = concat.concat(b, &ctx);
        }
        let rev = concat.reverse(&ctx);
        // Pin each byte to a distinct value and verify byte-reversed result.
        let vals: [u64; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for (b, v) in bytes.iter().zip(vals.iter()) {
            let pin = b.eq(&RustBV::concrete(*v as u128, 8), &ctx);
            ctx.add_constraint(pin.to_z3_ast().eq(&z3::ast::BV::from_u64(1, 1)));
        }
        // concat = 0x1122334455667788; reverse → 0x8877665544332211
        assert_eq!(ctx.eval(&rev), Some(0x8877665544332211));
    }
}
