//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that is one of these variants:
//! - Concrete: A known fixed value
//! - Ast: A clarirs term — every symbolic value, named leaves included

use std::fmt;
use std::sync::Arc;

use clarirs_core::ast::wire::WireDag;
use clarirs_core::prelude::{AstFactory, AstRef, AstType, BitVec, ClarirsError};
use serde::{Deserialize, Serialize};

use super::SymContext;
use crate::claripy::ast::GLOBAL_CONTEXT;

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
    /// operand\[0\] = rm BV (32-bit, a [`VexRm`] encoding),
    /// operand\[1\] = value BV (prec.bits()).
    RoundToInt,
    /// Convert int (signed/unsigned 2's complement) BV to FP at `prec`.
    /// operand\[0\] = src BV at src_bits. RNE rounding implicit.
    ConvertItoF {
        src_bits: u8,
        signed: bool,
    },
    /// Convert int BV to FP at `prec` with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit), operand\[1\] = src BV at src_bits.
    ConvertItoFRm {
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
    /// operand\[0\] = rm BV (32-bit, a [`VexRm`] encoding),
    /// operand\[1\] = a BV at prec.bits(), operand\[2\] = b BV at prec.bits().
    AddRm,
    SubRm,
    MulRm,
    DivRm,
    /// Unary FP sqrt with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit), operand\[1\] = a BV at prec.bits().
    SqrtRm,
    /// Ternary fused multiply-add / multiply-sub with explicit rounding mode.
    /// operand\[0\] = rm BV (32-bit, a [`VexRm`] encoding), operands
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
            | FloatOpKind::ConvertItoFRm { .. }
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

/// A VEX `IRRoundingMode` (`libvex_ir.h`), decoded once for every consumer:
/// the concrete folds in `vex::ops`, the `*_with_rm` round-to-nearest fast
/// paths and the symbolic lowering in `value_clarirs::with_vex_rm`.
///
/// The mapping mirrors the Python engine's `irop.py::translate_rm` exactly:
/// the whole operand value selects the mode, 0..=4 as below, and every other
/// value is an [`UnsupportedRoundingMode`] error, the analogue of Python's
/// `UnsupportedIROpError`. That covers 5..=7 (libVEX's PREPARE_SHORTER /
/// AWAY_FROM_ZERO / NEAREST_TIE_TOWARD_0, which no supported front end
/// emits) and the AVX-512 embedded-rounding encodings 8..=11, which Python
/// rejects too. There is no silent fallback to a default mode: a wrong guess
/// would be a silently wrong value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VexRm {
    /// `Irrm_NEAREST` (0): round to nearest, ties to even.
    NearestEven,
    /// `Irrm_NegINF` (1): toward negative infinity.
    TowardNegative,
    /// `Irrm_PosINF` (2): toward positive infinity.
    TowardPositive,
    /// `Irrm_ZERO` (3): toward zero.
    TowardZero,
    /// `Irrm_NEAREST_TIE_AWAY_0` (4): round to nearest, ties away from zero
    /// (AArch64 FCVTAS/FCVTAU, FRINTA's rounding).
    NearestAway,
}

/// A concrete VEX rounding-mode operand outside the modes [`VexRm`] decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported VEX rounding mode {0}")]
pub struct UnsupportedRoundingMode(pub u128);

impl VexRm {
    /// Every decodable mode, in encoding order.
    pub const ALL: [VexRm; 5] = [
        VexRm::NearestEven,
        VexRm::TowardNegative,
        VexRm::TowardPositive,
        VexRm::TowardZero,
        VexRm::NearestAway,
    ];

    /// Decode a concrete VEX rounding-mode operand (see the type doc).
    #[inline]
    pub fn decode(mode: u128) -> Result<Self, UnsupportedRoundingMode> {
        match mode {
            0 => Ok(VexRm::NearestEven),
            1 => Ok(VexRm::TowardNegative),
            2 => Ok(VexRm::TowardPositive),
            3 => Ok(VexRm::TowardZero),
            4 => Ok(VexRm::NearestAway),
            other => Err(UnsupportedRoundingMode(other)),
        }
    }

    /// The `IRRoundingMode` value that decodes to this mode.
    #[inline]
    pub const fn encoding(self) -> u32 {
        match self {
            VexRm::NearestEven => 0,
            VexRm::TowardNegative => 1,
            VexRm::TowardPositive => 2,
            VexRm::TowardZero => 3,
            VexRm::NearestAway => 4,
        }
    }
}

/// A bitvector value that can be concrete or symbolic.
///
/// This is the core value type for the VEX execution engine. All register
/// and memory values are represented as `RustBV`.
///
/// A value holds no Z3 term: `to_z3_ast` lowers an `Ast` through
/// `z3_lower`, whose per-thread cache keyed by clarirs node is the only
/// lowering cache (angr-mx00p.116.4).
///
/// ## Serialization
///
/// `RustBV` implements `Serialize`/`Deserialize` via the `RustBVData`
/// shadow type.
#[derive(Clone, Serialize, Deserialize)]
#[serde(try_from = "RustBVData", into = "RustBVData")]
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
    /// Every symbolic value: a clarirs term (angr-mx00p.11, .12). Always
    /// bitvector-sorted: [`RustBV::from_clarirs`] encodes a Bool as
    /// `ite(b, 1, 0)` and a float as its IEEE bits. Built by `from_clarirs`
    /// (deserializing included) and by `from_clarirs_lowerable` for the
    /// constructors, the leaves and the operands `RustBV::node` presents;
    /// both fold a literal that fits a `u128` into [`RustBV::Concrete`].
    ///
    /// Equality is node identity: clarirs interns, so equal terms are one node.
    Ast {
        /// Width in bits, `ast.size()`.
        width: u32,
        ast: AstRef<'static>,
    },
}

// Registers and memory pages hold these, so the value carries no Z3 term
// (angr-mx00p.116.4): a `Concrete`'s `u128` sets the size.
const _: () = assert!(std::mem::size_of::<RustBV>() <= 32);

/// Serde shadow form for [`RustBV`].
///
/// `Concrete` mirrors [`RustBV::Concrete`]; an `Ast` travels as a [`WireDag`]
/// of its clarirs term.
///
/// This is the snapshot/serialization format (angr-x04s).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RustBVData {
    Concrete {
        value: u128,
        width: u32,
    },
    /// The clarirs node with every subterm shared, decoded into
    /// `GLOBAL_CONTEXT` on load.
    Ast {
        dag: WireDag,
    },
}

impl From<RustBV> for RustBVData {
    fn from(bv: RustBV) -> Self {
        match bv {
            RustBV::Concrete { value, width } => RustBVData::Concrete { value, width },
            RustBV::Ast { ast, .. } => RustBVData::Ast {
                dag: WireDag::encode(std::slice::from_ref(&ast)),
            },
        }
    }
}

/// Fallible because an `Ast` payload is a node list that must decode to one
/// bitvector-sorted root; every other variant always converts.
impl TryFrom<RustBVData> for RustBV {
    type Error = String;

    fn try_from(data: RustBVData) -> Result<Self, String> {
        Ok(match data {
            RustBVData::Concrete { value, width } => RustBV::Concrete { value, width },
            RustBVData::Ast { dag } => {
                let roots = dag
                    .decode(&GLOBAL_CONTEXT)
                    .map_err(|e| format!("RustBV::Ast: {e}"))?;
                let [root] = <[_; 1]>::try_from(roots)
                    .map_err(|r| format!("RustBV::Ast: {} roots, expected 1", r.len()))?;
                // Encoding only ever writes the bitvector a `RustBV::Ast` holds.
                if !root.ast_type().is_bitvec() {
                    return Err(format!(
                        "RustBV::Ast: root is {:?}, not a bitvector",
                        root.ast_type()
                    ));
                }
                // Through the constructor, so a decoded term is checked to
                // lower exactly as a freshly built one is.
                RustBV::from_clarirs(root).map_err(|e| format!("RustBV::Ast: {e}"))?
            }
        })
    }
}

impl RustBV {
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

    /// Build a leaf from its parts: the clarirs variable named `name`.
    ///
    /// Single source of truth for leaf construction shared by `symbolic` and
    /// `symbolic_with_id`, so the two cannot drift apart.
    ///
    /// **Width 0 degenerates to `Concrete`.** Z3 has no 0-width bitvector sort,
    /// so lowering a width-0 leaf panics inside the sort constructor — and the
    /// extension builds with `panic="abort"`, making that a hard SIGABRT of the
    /// whole process rather than a catchable Python exception. Both symbolic
    /// leaf constructors reach this function and take a caller-chosen width
    /// that `check_bv_width` deliberately permits to be 0 (see `width_guards`,
    /// where 0 is documented as a supported *concrete* degenerate value) — so
    /// `claripy.BVS(name, 0)` imported through `claripy_bridge::import` was a
    /// live process abort (angr-fs8kb.28).
    ///
    /// The empty bitvector has exactly one inhabitant, so a width-0 *symbolic*
    /// leaf denotes the same single value a width-0 concrete does: collapsing it
    /// loses no information, and no constraint can distinguish two of them. That
    /// makes the degenerate answer correct here rather than merely safe, which
    /// is why this returns a value instead of plumbing a `Result` through
    /// infallible signatures (same reasoning as the `w == 0` rotate arm in
    /// `value_ops`).
    #[inline]
    #[allow(
        clippy::expect_used,
        reason = "clarirs' sort check accepts every BVS and BoolS leaf (`AstOp::validate`), and `bit_of` over a BoolS builds a well-sorted ite, so the build cannot fail"
    )]
    pub(super) fn from_parts(id: u64, name: Arc<str>, width: u32) -> Self {
        if width == 0 {
            // SILENT(cat-a): expected control flow — the empty bitvector's only
            // value, not a fallback for a failure.
            return RustBV::Concrete { value: 0, width: 0 };
        }
        // A leaf is the clarirs variable of its name (angr-mx00p.12): identity
        // is the name, as it already is in Z3
        // (`invariant-symbol-identity-is-by-name`). A Bool-tagged name is
        // `ite(BoolS, 1, 0)`, the encoding `from_clarirs` and `leaf_var` read
        // back. An empty name is given one from `id`, so that leaf stays its
        // own variable.
        let name: Arc<str> = if name.is_empty() {
            Arc::from(format!("unnamed_{id}"))
        } else {
            name
        };
        let gc = &*crate::claripy::ast::GLOBAL_CONTEXT;
        let term = match super::symbol_kind::strip_bool_symbol_name(&name) {
            Some(bool_name) => gc.bools(bool_name).and_then(super::value_clarirs::bit_of),
            None => gc.bvs(&*name, width),
        };
        RustBV::from_clarirs_lowerable(term.expect("clarirs accepts every BVS / BoolS leaf"))
    }

    /// Create a symbolic bitvector variable: the clarirs variable `name`
    /// (see `Self::from_parts`), so every call with the same name and width
    /// returns the same leaf.
    pub fn symbolic(ctx: &SymContext, name: impl AsRef<str>, width: u32) -> Self {
        RustBV::from_parts(ctx.next_id(), Arc::from(name.as_ref()), width)
    }

    /// Create a fresh symbolic bitvector variable named `{hint}_{n}`, with `n`
    /// from `names::next(hint)`, so every call is its own variable,
    /// as each `claripy.BVS(hint, width)` is. The counter is keyed by `hint`
    /// and rides in snapshots (`names::watermarks`), so a restored state's
    /// names are not reminted.
    pub fn fresh(hint: &str, width: u32) -> Self {
        let n = super::names::next(hint);
        // The name is never empty, so `from_parts` never reads the id.
        RustBV::from_parts(0, Arc::from(format!("{hint}_{n}")), width)
    }

    /// Create a symbolic bitvector variable named `name`. A leaf is
    /// identified by its name, not by `id` (see `Self::from_parts`).
    pub fn symbolic_with_id(id: u64, name: impl AsRef<str>, width: u32) -> Self {
        RustBV::from_parts(id, Arc::from(name.as_ref()), width)
    }

    /// Wrap a clarirs AST as a [`RustBV::Ast`], in the bitvector encoding the
    /// other variants use: a Bool becomes `ite(b, 1, 0)` (width 1), a float its
    /// IEEE bits (`fpToIEEEBV`). A literal that fits a `u128` folds to
    /// [`RustBV::Concrete`]. Strings have no `RustBV` form and are refused.
    ///
    /// The term is lowered here, which both warms `z3_lower`'s cache and
    /// refuses what has no Z3 form (strings nested in a bitvector, the VSA ops, a term
    /// deeper than `z3_lower::MAX_LOWER_DEPTH`) at the boundary. Terms the ops
    /// build later over the result can still outgrow that depth, which is why
    /// `to_z3_ast` returns a `Result` (angr-mx00p.116.2).
    pub fn from_clarirs(ast: AstRef<'static>) -> Result<Self, ClarirsError> {
        let ctx = &*GLOBAL_CONTEXT;
        let ast = match ast.ast_type() {
            AstType::BitVec(_) => ast,
            AstType::Bool => ctx.ite(
                &ast,
                ctx.bvv(BitVec::from((1, 1)))?,
                ctx.bvv(BitVec::from((0, 1)))?,
            )?,
            AstType::Float(_) => ctx.fp_to_ieeebv(&ast)?,
            AstType::String => {
                return Err(ClarirsError::TypeError(
                    "a string AST has no RustBV form".into(),
                ));
            }
        };
        if let clarirs_core::prelude::AstOp::BVV(value) = ast.op()
            && let Ok(v) = u128::try_from(&value.to_biguint())
        {
            return Ok(RustBV::concrete(v, value.len()));
        }
        super::z3_lower::lower_bv(&ast)
            .map_err(|e| ClarirsError::UnsupportedOperation(e.to_string()))?;
        Ok(RustBV::Ast {
            width: ast.size(),
            ast,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the width in bits.
    #[inline]
    pub fn width(&self) -> u32 {
        match self {
            RustBV::Concrete { width, .. } => *width,
            RustBV::Ast { width, .. } => *width,
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
        matches!(self, RustBV::Ast { .. })
    }

    /// Try to get the concrete value.
    #[inline]
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            RustBV::Concrete { value, .. } => Some(*value),
            // `from_clarirs` folds every literal that fits into `Concrete`.
            RustBV::Ast { .. } => None,
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
            RustBV::Ast { width, ast, .. } => {
                write!(f, "Ast({}, {width})", ast.to_smtlib_shallow(2))
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
            RustBV::Ast { width, ast, .. } => {
                write!(f, "<BV{width} {}>", ast.to_smtlib_shallow(2))
            }
        }
    }
}

/// Equality is **value-based where a value is known, identity-based otherwise**
/// (angr-sqfj8.96).
///
/// The relation partitions `RustBV` into two classes, which never compare
/// equal across the class boundary:
///
/// 1. **Known values** (`Concrete`): equal iff same value and same width.
/// 2. **clarirs terms** (`Ast`): equal iff the same node. clarirs interns, so
///    this is structural equality, and it is cheap. A named leaf is the
///    clarirs variable of its name, so two leaves are equal iff they share
///    name and width — the Z3 constant (`invariant-symbol-identity-is-by-name`).
///
/// Each class uses a genuine equivalence relation and the classes are
/// disjoint, so the whole relation is reflexive, symmetric and transitive —
/// `x == x` holds for every variant. It used to short-circuit to `false`
/// whenever either side lacked an `as_u128()`, which broke reflexivity for
/// symbolic values and made `==` a trap for dedup/`contains` callers.
///
/// There is intentionally no `Eq`/`Hash`: node identity is pointer identity,
/// which a serde round-trip does not preserve outside one process.
impl PartialEq for RustBV {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RustBV::Ast { ast: a, .. }, RustBV::Ast { ast: b, .. }) => Arc::ptr_eq(a, b),
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
// angr-5mnx3.49: split again at 2062 lines, back over the <2000-line
// threshold, into one topical module per former section banner plus a shared
// `value_tests_support` for the two helpers more than one of them uses.
test_submod!("value_tests_support.rs" => value_tests_support);
test_submod!("value_tests.rs" => value_tests);
test_submod!("value_simplify_tests.rs" => value_simplify_tests);
test_submod!("value_zext_cmp_tests.rs" => value_zext_cmp_tests);
test_submod!("value_serde_tests.rs" => value_serde_tests);
test_submod!("value_ast_tests.rs" => value_ast_tests);
test_submod!("value_concrete_arm_tests.rs" => value_concrete_arm_tests);
test_submod!("value_eq_debug_tests.rs" => value_eq_debug_tests);
