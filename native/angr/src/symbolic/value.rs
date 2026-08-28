//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that is one of four variants:
//! - Concrete: A known fixed value
//! - Symbolic: Represents an unknown value (with optional Z3 backing)
//! - Constrained: A symbolic leaf whose value is already pinned to a concrete
//!   one. Not interchangeable with `Concrete` — it keeps the symbol's `id`, and
//!   the identity-preservation invariant that a *no-op* fold (a width-identity
//!   extend/truncate, a full-width extract) must not rebuild it as `Concrete`
//!   and drop that id (angr-9ke6b.128) is cited throughout this file and
//!   `value_ops.rs`.
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

    /// Owned Z3 floating-point `Sort` for this precision.
    ///
    /// Callers must bind the returned wrapper to a local before calling
    /// `get_z3_sort()` on it: the raw `Z3_sort` handle is only valid while the
    /// owned wrapper stays alive across subsequent unsafe Z3 calls.
    #[cfg(feature = "vex-engine-z3")]
    #[inline]
    pub fn z3_sort(&self) -> z3::Sort {
        match self {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
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
    /// IEEE-754-2008 `maxNum` / `minNum` (SMT-LIB `fp.max` / `fp.min`): when
    /// exactly one operand is NaN the *other* operand is returned, unlike the
    /// plain compare-and-select `FMax`/`FMin` lane ops in
    /// `vex::ops::lane_traits`, which propagate whichever operand the compare
    /// happens to fall through to. Backs `IROp::FMaxNum`/`FMinNum` (ARM32
    /// VMAXNM/VMINNM).
    MaxNum,
    MinNum,
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
    /// Ternary fused multiply-add / multiply-sub with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit, VEX rm low-2-bits 0..3), operands
    /// \[1\]/\[2\]/\[3\] = a/b/c BVs at prec.bits(). VEX delivers FMA as a Qop
    /// `(rm, a, b, c)`; the rm-less `Fma`/`Fms` above are the RNE forms.
    FmaRm,
    FmsRm,
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
            | FloatOpKind::MaxNum
            | FloatOpKind::MinNum
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
            FloatOpKind::FmaRm | FloatOpKind::FmsRm => 4,
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
        /// The value, masked to `width` bits (or to its low 128 bits when
        /// `width > 128`).
        value: u128,
        /// Width in bits. The common widths are 1/8/16/32/64/128, but this is
        /// not an exhaustive list: wider values (AVX/YMM-scale, up to 256) are
        /// deliberately supported and exercised by `WIDE_WIDTHS` in
        /// `value_ops_property_tests.rs`. Because the payload is a `u128`,
        /// such a value only stores its low 128 bits — see the `width.min(128)`
        /// branches in `value_ops.rs`'s `shl_into` / `lshr_into` / `ashr_into`
        /// and `solving_ops.rs`'s `max_val_for_width`.
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
        /// Persistent Z3 AST memo (angr-ovqja.3).
        ///
        /// Caches the built `z3::ast::BV` for this exact `Expression` value so
        /// repeat top-level conversions of the SAME node (eval→min→max,
        /// `range()` which calls min then max, address-concretize) become a
        /// refcount bump instead of a full compound-tree rebuild. Mirrors the
        /// `Symbolic { ast }` leaf-cache pattern, but lazy: populated on first
        /// [`to_z3_ast_cached`](RustBV::to_z3_ast_cached) and guarded against
        /// thread-local Z3 context swaps (the cached BV carries its own
        /// `Context`, compared to the active one on read).
        ///
        /// Excluded from the serde shadow (`RustBVData` has no such field),
        /// from `PartialEq`/`Debug`, and from any identity key — it is a pure
        /// deterministic-function cache. With the `vex-engine-z3` feature off
        /// it collapses to `()` so construction sites stay cfg-free.
        memo: ExprMemo,
    },
}

/// Lazy Z3 AST memo carried by [`RustBV::Expression`] (angr-ovqja.3).
///
/// `RefCell<Option<BV>>` under Z3 (single-threaded engine — the state graph is
/// `Rc<RefCell<SymContext>>`, so interior mutability is sound), `()` otherwise
/// so every `Expression` constructor can write `memo: Default::default()`
/// without a `#[cfg]`.
#[cfg(feature = "vex-engine-z3")]
type ExprMemo = std::cell::RefCell<Option<z3::ast::BV>>;
/// See the Z3-enabled variant above.
#[cfg(not(feature = "vex-engine-z3"))]
type ExprMemo = ();

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
/// This is the snapshot/serialization format from the angr-x04s spike.
/// Intermediate cache
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
                ..
            } => RustBVData::Expression {
                id,
                width,
                op,
                operands: operands.iter().cloned().map(RustBVData::from).collect(),
            },
        }
    }
}

/// Shift a deserialized symbol id into the active rebase range (angr-euw28).
///
/// A no-op (offset 0) unless a [`SymbolIdRebase`](crate::symbolic::SymbolIdRebase)
/// guard is live on this thread — i.e. unless we are restoring a snapshot minted
/// by a foreign process. The [`RustBV::EXPRESSION_ID`] sentinel is not an
/// allocated id and must stay `u64::MAX`.
#[inline]
fn rebase_id(id: u64) -> u64 {
    if id == RustBV::EXPRESSION_ID {
        return id;
    }
    id.saturating_add(crate::symbolic::symbol_id_rebase_offset())
}

impl From<RustBVData> for RustBV {
    fn from(data: RustBVData) -> Self {
        match data {
            RustBVData::Concrete { value, width } => RustBV::Concrete { value, width },
            RustBVData::Symbolic { id, width, name } => {
                // Rebuilds the Z3 AST in the active thread-local context (z3
                // feature); callers must be inside with_z3_context when this runs.
                RustBV::from_parts(rebase_id(id), Arc::<str>::from(name), width)
            }
            RustBVData::Constrained { id, value, width } => RustBV::Constrained {
                id: rebase_id(id),
                value,
                width,
            },
            RustBVData::Expression {
                id,
                width,
                op,
                operands,
            } => RustBV::Expression {
                id: rebase_id(id),
                width,
                op,
                operands: Arc::<[RustBV]>::from(
                    operands.into_iter().map(RustBV::from).collect::<Vec<_>>(),
                ),
                memo: Default::default(),
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
        RustBV::Concrete {
            value: value & Self::all_ones_mask(width),
            width,
        }
    }

    /// Create a zero bitvector.
    #[inline]
    pub fn zero(width: u32) -> Self {
        RustBV::Concrete { value: 0, width }
    }

    /// Create a bitvector with all bits set to 1.
    ///
    /// **Payload-limited above 128 bits.** A `Concrete` stores a `u128`, so at
    /// `width > 128` this yields `2^128 - 1` — an ordinary number roughly
    /// `2^(width-128)` times too small, *not* the width's all-ones value, which
    /// simply has no `Concrete` representation. Callers on a path a `width > 128`
    /// vector can reach must decline instead of materialising this (see
    /// `value_ops::is_all_ones` and the `x / 0` arms of `udiv`/`sdiv`);
    /// `value_ops::bits_beyond_storage` is the predicate to gate on.
    #[inline]
    pub fn ones(width: u32) -> Self {
        RustBV::Concrete {
            value: Self::all_ones_mask(width),
            width,
        }
    }

    /// Return the all-ones mask for a given bit width.
    ///
    /// Single source of truth for the saturating `(1u128 << width) - 1` shift:
    /// `solving_ops::max_val_for_width` (and through it `query_class`'s
    /// unsigned-bound arms) delegate here, because a width's all-ones mask and
    /// its largest unsigned value are the same number and had drifted into
    /// three independent copies (angr-0jh0j.55). Widths `>= 128` saturate at
    /// `u128::MAX` — shifting a `u128` by `>= 128` panics in debug and wraps
    /// the shift amount in release, and the `Concrete` payload cannot represent
    /// anything wider anyway.
    #[inline]
    pub(super) fn all_ones_mask(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }

    /// Build a `Symbolic` node from its parts, rebuilding the cfg-gated Z3 AST.
    ///
    /// Single source of truth for the `#[cfg(vex-engine-z3)]` ast-rebuild +
    /// `Symbolic` construction shared by `symbolic`, `symbolic_with_id`, and the
    /// `From<RustBVData>` deserialize arm — keeps the three in lockstep so the
    /// deserialize path can't drift from the constructors. Callers must be inside
    /// `with_z3_context` when the z3 feature is enabled (BV::new_const reads the
    /// active thread-local context).
    #[inline]
    pub(super) fn from_parts(id: u64, name: Arc<str>, width: u32) -> Self {
        #[cfg(feature = "vex-engine-z3")]
        {
            // A claripy `Bool` leaf has no `RustBV` sort of its own — it is
            // modelled as a width-1 `Symbolic` whose NAME carries the sort tag
            // (`SymbolKind::rust_symbol_name`). Decode it here so the Z3 term
            // is a genuine Bool constant lowered to 1 bit, which is exactly how
            // claripy's own z3 backend encodes `BoolS` — see
            // `strip_bool_symbol_name` for why both halves of that matter.
            let ast = match super::registry::strip_bool_symbol_name(&name) {
                Some(claripy_name) => z3::ast::Bool::new_const(claripy_name)
                    .ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1)),
                None => z3::ast::BV::new_const(&*name, width),
            };
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

    /// Create a symbolic bitvector variable.
    ///
    /// Accepts anything String-like via `AsRef<str>`. The internal name is
    /// stored as `Arc<str>` so cloning a Symbolic is alloc-free, which matters
    /// on the RdTmp path where a temp slot is read multiple times per block.
    pub fn symbolic(ctx: &SymContext, name: impl AsRef<str>, width: u32) -> Self {
        RustBV::from_parts(ctx.next_id(), Arc::from(name.as_ref()), width)
    }

    /// Create a symbolic bitvector variable with a specific ID.
    ///
    /// This is used for identity preservation when the same symbol
    /// was previously imported from Python. By reusing the same ID,
    /// we ensure that constraints on the original symbol apply correctly.
    pub fn symbolic_with_id(id: u64, name: impl AsRef<str>, width: u32) -> Self {
        RustBV::from_parts(id, Arc::from(name.as_ref()), width)
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

    /// Try to get the concrete value as u64.
    #[inline]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_u128().map(|v| v as u64)
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl fmt::Debug for RustBV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustBV::Concrete { value, width } => {
                write!(f, "Concrete(0x{value:x}, {width})")
            }
            // `id` is the actual identity key for these two leaf variants —
            // `query_class` matches on it, memory ITE-building and the claripy
            // registry key off it — so Debug must print it or two values that
            // differ only by id read as identical in logs (angr-sqfj8.99).
            RustBV::Symbolic {
                id, width, name, ..
            } => {
                write!(f, "Symbolic(#{id}, {name}, {width})")
            }
            RustBV::Constrained { id, value, width } => {
                write!(f, "Constrained(#{id}, 0x{value:x}, {width})")
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
                write!(f, "<BV{width} 0x{value:x}>")
            }
            RustBV::Symbolic { width, name, .. } => {
                write!(f, "<BV{width} {name}>")
            }
            RustBV::Constrained { value, width, .. } => {
                write!(f, "<BV{width} 0x{value:x} (constrained)>")
            }
            RustBV::Expression { width, op, .. } => {
                write!(f, "<BV{width} {op:?}>")
            }
        }
    }
}

/// Equality is **value-based where a value is known, identity-based otherwise**
/// (angr-sqfj8.96).
///
/// The relation partitions `RustBV` into three classes that never compare
/// equal across the class boundary:
///
/// 1. **Known-value** (`Concrete`, and `Constrained` — which carries a pinned
///    concrete `value`): equal iff same `as_u128()` and same width. A
///    `Constrained` therefore equals the `Concrete` holding its value, which
///    is the point — the two are interchangeable results of the same
///    computation.
/// 2. **Leaf symbols** (`Symbolic`): equal iff same allocated `id` (from
///    [`crate::symbolic::SymbolicIdentityRegistry`]) and width. `name` is
///    debug-only and ids are unique, so id equality is the identity test.
/// 3. **Compound expressions** (`Expression`): equal iff same `op`, same
///    width, and *pointer-identical* operand slices. The `id` field is the
///    `EXPRESSION_ID` sentinel for every `Expression` (see its doc), so it
///    carries no identity; `Arc::ptr_eq` is what survives a `clone` (which
///    only bumps the refcount). Deliberately **not** a deep structural walk:
///    `==` sits on hot paths and expression trees are unbounded in depth, so
///    two independently-built but structurally identical trees compare
///    unequal. Treat `Expression` equality as "same node", not "same value".
///    The `memo` Z3 cache is excluded (pure deterministic-function cache).
///
/// Each class uses a genuine equivalence relation and the classes are
/// disjoint, so the whole relation is reflexive, symmetric and transitive —
/// `x == x` holds for every variant. It used to short-circuit to `false`
/// whenever either side lacked an `as_u128()`, which broke reflexivity for
/// `Symbolic`/`Expression` and made `==` a trap for dedup/`contains` callers.
///
/// There is intentionally no `Eq`/`Hash`: two `RustBV`s that compare equal
/// (a `Concrete` and the matching `Constrained`) have no shared cheap hash,
/// and `Expression`'s pointer identity would hash differently after a
/// serde round-trip.
impl PartialEq for RustBV {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                RustBV::Symbolic {
                    id: a, width: wa, ..
                },
                RustBV::Symbolic {
                    id: b, width: wb, ..
                },
            ) => a == b && wa == wb,
            (
                RustBV::Expression {
                    width: wa,
                    op: opa,
                    operands: oa,
                    ..
                },
                RustBV::Expression {
                    width: wb,
                    op: opb,
                    operands: ob,
                    ..
                },
            ) => wa == wb && opa == opb && Arc::ptr_eq(oa, ob),
            _ => match (self.as_u128(), other.as_u128()) {
                (Some(a), Some(b)) => a == b && self.width() == other.width(),
                _ => false,
            },
        }
    }
}

// angr-7hwz.1: RustBV unit tests, extracted out of the former in-file
// `mod tests` (1219 lines) into a sibling file to shrink value.rs below the
// god-object threshold. Declared as a direct child of `value` so
// `use super::*` reaches `value`'s private items.
// Gated on vex-engine-z3: these tests call `RustBV::to_z3_ast` /
// `SymContext::add_constraint` and reference the `z3` crate, none of which
// exist in a no-z3 build. The no-default-features / vex-engine (no-z3) nightly
// combos build the lib test harness, so an ungated decl breaks `cargo test`
// there (bd angr-cagbn). Default (z3-on) build still compiles and runs them.
// angr-5mnx3.49: split again at 2062 lines, back over the <2000-line
// threshold, into one topical module per former section banner plus a shared
// `value_tests_support` for the two helpers more than one of them uses.
test_submod!(z3 "value_tests_support.rs" => value_tests_support);
test_submod!(z3 "value_tests.rs" => value_tests);
test_submod!(z3 "value_simplify_tests.rs" => value_simplify_tests);
test_submod!(z3 "value_zext_cmp_tests.rs" => value_zext_cmp_tests);
test_submod!(z3 "value_serde_tests.rs" => value_serde_tests);
test_submod!(z3 "value_concrete_arm_tests.rs" => value_concrete_arm_tests);
test_submod!(z3 "value_eq_debug_tests.rs" => value_eq_debug_tests);
