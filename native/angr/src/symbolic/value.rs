//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that can be either:
//! - Concrete: A known fixed value
//! - Symbolic: Represents an unknown value (with optional Z3 backing)
//! - Expression: A compound expression with operation tree for reconstruction

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::SymContext;

/// Bitvector operation type for expression tree reconstruction.
///
/// This enum represents all operations that can be performed on bitvectors,
/// enabling reconstruction of claripy ASTs from Rust expression trees.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Unary IEEE-754 isNaN predicate; 1-bit result. Lets NaN detection
    /// avoid building `CmpEq(v, v)` with the operand cloned twice.
    IsNaN,
    /// Round to integer with rounding mode.
    /// operand\[0\] = rm BV (32-bit, VEX rounding mode 0..3),
    /// operand\[1\] = value BV (prec.bits()).
    RoundToInt,
    /// Convert int (signed/unsigned 2's complement) BV to FP at `prec`.
    /// operand\[0\] = src BV at src_bits. RNE rounding implicit.
    ConvertItoF {
        src_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement).
    /// operand\[0\] = src FP BV at prec.bits(). RNE rounding implicit.
    ConvertFtoI {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement)
    /// with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit), operand\[1\] = src FP BV at prec.bits().
    ConvertFtoIRm {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `src_prec` to FP at `prec`. RNE rounding implicit.
    /// operand\[0\] = src FP BV at src_prec.bits().
    ConvertFtoF {
        src_prec: FloatPrec,
    },
    /// Convert FP at `src_prec` to FP at `prec` with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit), operand\[1\] = src FP BV at src_prec.bits().
    ConvertFtoFRm {
        src_prec: FloatPrec,
    },
    /// Binary FP arithmetic with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit, VEX rm low-2-bits 0..3),
    /// operand\[1\] = a BV at prec.bits(), operand\[2\] = b BV at prec.bits().
    AddRm,
    SubRm,
    MulRm,
    DivRm,
    /// Unary FP sqrt with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit), operand\[1\] = a BV at prec.bits().
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
            | FloatOpKind::IsNaN
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
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe | FloatOpKind::IsNaN
        )
    }

    /// Width (in bits) of the BV result for this op, given the `prec` field
    /// of the enclosing `BVOp::Float`. Compares are 1-bit; FtoI conversions
    /// take their result width from `dst_bits`; everything else returns the
    /// IEEE encoding width of `prec`.
    #[inline]
    pub fn result_bits(&self, prec: FloatPrec) -> u32 {
        match self {
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe | FloatOpKind::IsNaN => 1,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitWidth {
    W1 = 1,
    W8 = 8,
    W16 = 16,
    W32 = 32,
    W64 = 64,
    W128 = 128,
}

impl BitWidth {
    #[inline]
    pub fn bits(&self) -> u32 {
        *self as u32
    }

    #[inline]
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

    #[inline]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
///
/// ## Serialization
///
/// `RustBV` implements `Serialize`/`Deserialize` via the `RustBVData`
/// shadow type — the Z3 AST cache on `Symbolic` is skipped at serialize
/// time and rebuilt on first `to_z3_ast()` call after load
/// (a fresh `BV::new_const(name, width)` in the active Z3 thread-local
/// context). Deserialization must happen with a Z3 context active when
/// the `vex-engine-z3` feature is enabled.
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "RustBVData", into = "RustBVData")]
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
        /// Identity discriminant — **NOT** a content hash or interner id.
        ///
        /// In practice this field is always set to [`RustBV::EXPRESSION_ID`]
        /// (`u64::MAX`) by every `Expression` constructor in this module
        /// (see the `define_*_op` macro families and direct builders such as
        /// `add_into`, `mul_into`, `and_into`, etc.). The sentinel
        /// distinguishes compound expressions from leaf symbols (`Symbolic`
        /// / `Constrained`), which carry real allocated ids from
        /// [`crate::symbolic::SymbolicIdentityRegistry`].
        ///
        /// **Why a sentinel, not a per-node hash:** allocating a unique id
        /// for every intermediate operation was rejected for solver
        /// compatibility (only real leaf symbols need tracking by the
        /// claripy-side registry) and to avoid per-construction overhead.
        ///
        /// **Identity for caching purposes** is keyed by *other* fields,
        /// not this one: the structural pointer `Arc::as_ptr(operands)`
        /// is the cache key for `EXPRESSION_BY_OPERANDS_PTR` in
        /// `claripy_bridge` (stable across `RustBV::clone`, which just
        /// bumps the Arc).
        ///
        /// **Collision potential:** none — there is no content hash here
        /// to collide on. Two structurally distinct `Expression` values
        /// share the same `id` sentinel.
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

/// Serde shadow form for [`RustBV`].
///
/// The wire format mirrors `RustBV` field-for-field except:
///
/// - The `Symbolic` variant's `ast: z3::ast::BV` cache is dropped (Z3
///   ASTs cannot cross process boundaries and are reconstructable from
///   `(name, width)`). On deserialize, the AST is recreated lazily via
///   `BV::new_const(&name, width)` inside the active thread-local Z3
///   context — callers must run within `with_z3_context` (or equivalent)
///   when the `vex-engine-z3` feature is enabled.
/// - `Arc<str>` / `Arc<[RustBV]>` collapse to owned `String` / `Vec`
///   on the wire and rebuild Arc handles on load.
///
/// This is the snapshot/serialization format from the angr-x04s spike
/// (see `snapshot-serialization-design` bd memory). Intermediate cache
/// fields (`EXPRESSION_BY_OPERANDS_PTR`, content hashes) are not part of
/// the wire format; they rewarm naturally as the loaded ops are touched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RustBVData {
    Concrete {
        value: u128,
        width: u32,
    },
    Symbolic {
        id: u64,
        width: u32,
        name: String,
    },
    Constrained {
        id: u64,
        value: u128,
        width: u32,
    },
    Expression {
        id: u64,
        width: u32,
        op: BVOp,
        operands: Vec<RustBVData>,
    },
}

impl From<RustBV> for RustBVData {
    fn from(bv: RustBV) -> Self {
        match bv {
            RustBV::Concrete { value, width } => RustBVData::Concrete { value, width },
            RustBV::Symbolic {
                id, width, name, ..
            } => RustBVData::Symbolic {
                id,
                width,
                name: name.to_string(),
            },
            RustBV::Constrained { id, value, width } => {
                RustBVData::Constrained { id, value, width }
            }
            RustBV::Expression {
                id,
                width,
                op,
                operands,
            } => RustBVData::Expression {
                id,
                width,
                op,
                operands: operands.iter().cloned().map(RustBVData::from).collect(),
            },
        }
    }
}

impl From<RustBVData> for RustBV {
    fn from(data: RustBVData) -> Self {
        match data {
            RustBVData::Concrete { value, width } => RustBV::Concrete { value, width },
            RustBVData::Symbolic { id, width, name } => {
                #[cfg(feature = "vex-engine-z3")]
                {
                    // Rebuild the Z3 AST in the active thread-local context.
                    // Callers must be inside with_z3_context when this runs.
                    let ast = z3::ast::BV::new_const(name.as_str(), width);
                    RustBV::Symbolic {
                        id,
                        width,
                        name: Arc::<str>::from(name),
                        ast,
                    }
                }
                #[cfg(not(feature = "vex-engine-z3"))]
                {
                    RustBV::Symbolic {
                        id,
                        width,
                        name: Arc::<str>::from(name),
                    }
                }
            }
            RustBVData::Constrained { id, value, width } => {
                RustBV::Constrained { id, value, width }
            }
            RustBVData::Expression {
                id,
                width,
                op,
                operands,
            } => RustBV::Expression {
                id,
                width,
                op,
                operands: Arc::<[RustBV]>::from(
                    operands.into_iter().map(RustBV::from).collect::<Vec<_>>(),
                ),
            },
        }
    }
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

    /// Create a zero bitvector.
    #[inline]
    pub fn zero(width: u32) -> Self {
        RustBV::Concrete { value: 0, width }
    }

    /// Create a bitvector with all bits set to 1.
    #[inline]
    pub fn ones(width: u32) -> Self {
        RustBV::Concrete {
            value: Self::all_ones_mask(width),
            width,
        }
    }

    /// Return the all-ones mask for a given bit width.
    #[inline]
    pub(super) fn all_ones_mask(width: u32) -> u128 {
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

// angr-7hwz.1: RustBV unit tests, extracted out of the former in-file
// `mod tests` (1219 lines) into a sibling file to shrink value.rs below the
// god-object threshold. Declared as a direct child of `value` so
// `use super::*` reaches `value`'s private items.
#[cfg(test)]
#[path = "value_tests.rs"]
mod value_tests;
