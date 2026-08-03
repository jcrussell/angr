//! `IROp` -> `VexOpFamily` instrumentation classifier.
//!
//! Extracted from `ops/mod.rs` (angr-9ke6b.170). Re-exported by `ops/mod.rs`
//! so the public path stays `vex::ops::iropclass`.

use crate::symbolic::VexOpFamily;
use crate::vex::ir::IROp;

/// Classify an `IROp` into a coarse-grained family for instrumentation
/// (angr-2j5v). Counts roll up into `vex_op_<family>` counters via the
/// `record_vex_*` recorder fns at the entry of the IRExpr dispatch in
/// `interpreter/expressions.rs`.
///
/// `Vec` captures everything prefixed with `V*` (SIMD/NEON). `Fp` captures
/// scalar FP plus the unprefixed FP conversions. `Other` is the catch-all
/// for `Raw` opcode escapes, the `NeonUnimplemented` typed-error sentinel,
/// and the `Unmapped` typed-error sentinel (angr-tkbr.2).
#[inline]
pub fn iropclass(op: &IROp) -> VexOpFamily {
    match op {
        // Integer arithmetic (incl. widening / divmod / mul-hi / neg)
        IROp::Add(_)
        | IROp::Sub(_)
        | IROp::Mul(_)
        | IROp::MullS(_)
        | IROp::MullU(_)
        | IROp::DivS(_)
        | IROp::DivU(_)
        | IROp::ModS(_)
        | IROp::ModU(_)
        | IROp::DivModU64to32
        | IROp::DivModS64to32
        | IROp::DivModU128to64
        | IROp::DivModS128to64 => VexOpFamily::Arith,

        // Bitwise logic
        IROp::And(_) | IROp::Or(_) | IROp::Xor(_) | IROp::Not(_) => VexOpFamily::Logic,

        // Shifts
        IROp::Shl(_) | IROp::Shr(_) | IROp::Sar(_) => VexOpFamily::Shift,

        // Integer comparison
        IROp::CmpEQ(_)
        | IROp::CmpNE(_)
        | IROp::CmpLT(_)
        | IROp::CmpLE(_)
        | IROp::CmpLTU(_)
        | IROp::CmpLEU(_) => VexOpFamily::Cmp,

        // Width adjustment, bit-count, reinterpret, concat/extract.
        //
        // `Concat` is classified `Ext` here (it is a width adjustment) even
        // though the `binop()` router *dispatches* it to `binop_vec_int`
        // (`ops/mod.rs`), where its two-narrow-inputs/one-wide-output shape
        // matches the vector-int helpers. The mismatch is deliberate
        // (angr-9ke6b.168): dispatch grouping tracks code shape, family
        // grouping tracks semantics. `test_vec_int_binops_iropclass_is_vec`
        // in `ops/tests_vec_dispatch.rs` pins `Concat` to `Ext` so the two
        // groupings can drift apart only on purpose.
        IROp::SignExtend { .. }
        | IROp::ZeroExtend { .. }
        | IROp::Truncate { .. }
        | IROp::Clz(_)
        | IROp::Ctz(_)
        | IROp::PopCount(_)
        | IROp::Reinterpret { .. }
        | IROp::Concat { .. }
        | IROp::Extract { .. } => VexOpFamily::Ext,

        // Scalar FP (arith + cmp + conversions + rounding)
        IROp::FAdd(_)
        | IROp::FSub(_)
        | IROp::FMul(_)
        | IROp::FDiv(_)
        | IROp::FNeg(_)
        | IROp::FAbs(_)
        | IROp::FSqrt(_)
        | IROp::FMAdd(_)
        | IROp::FMSub(_)
        | IROp::FCmpEQ(_)
        | IROp::FCmpLT(_)
        | IROp::FCmpLE(_)
        | IROp::FCmpScalarLane { .. }
        | IROp::FCmpVecPacked { .. }
        | IROp::FComCC(_)
        | IROp::F32toF64
        | IROp::F64toF32
        | IROp::I32StoF32
        | IROp::I32StoF64
        | IROp::I64StoF32
        | IROp::I64StoF64
        | IROp::I32UtoF32
        | IROp::I32UtoF64
        | IROp::I64UtoF32
        | IROp::I64UtoF64
        | IROp::F32toI32S
        | IROp::F64toI32S
        | IROp::F32toI64S
        | IROp::F64toI64S
        | IROp::F32toI32U
        | IROp::F64toI32U
        | IROp::F32toI64U
        | IROp::F64toI64U
        | IROp::RoundF32toInt
        | IROp::RoundF64toInt => VexOpFamily::Fp,

        // SIMD/NEON — every V*-prefixed variant plus the V128 setters.
        IROp::VFAddS { .. }
        | IROp::VFSubS { .. }
        | IROp::VFMulS { .. }
        | IROp::VFDivS { .. }
        | IROp::VFSqrtS { .. }
        | IROp::VFMaxS { .. }
        | IROp::VFMinS { .. }
        | IROp::SetV128lo32
        | IROp::SetV128lo64
        | IROp::VAdd { .. }
        | IROp::VSub { .. }
        | IROp::VMul { .. }
        | IROp::VMull { .. }
        | IROp::VQDMull { .. }
        | IROp::VAnd(_)
        | IROp::VOr(_)
        | IROp::VXor(_)
        | IROp::VNot(_)
        | IROp::VShlN { .. }
        | IROp::VShrN { .. }
        | IROp::VSarN { .. }
        | IROp::VShl { .. }
        | IROp::VShr { .. }
        | IROp::VSar { .. }
        | IROp::VCmpEQ { .. }
        | IROp::VCmpGT { .. }
        | IROp::VInterleaveLO { .. }
        | IROp::VInterleaveHI { .. }
        | IROp::VPerm { .. }
        | IROp::VGetElem { .. }
        | IROp::VSetElem { .. }
        | IROp::VDup { .. }
        | IROp::VWiden { .. }
        | IROp::VNarrowUn { .. }
        | IROp::VNarrowBin { .. }
        | IROp::VQNarrowUn { .. }
        | IROp::VQNarrowBin { .. }
        | IROp::VReverse { .. }
        | IROp::VQAdd { .. }
        | IROp::VQSub { .. }
        | IROp::VQShlSat { .. }
        | IROp::VPwAdd { .. }
        | IROp::VPwAddL { .. }
        | IROp::VPwMin { .. }
        | IROp::VPwMax { .. }
        | IROp::VAvg { .. }
        | IROp::VCnt { .. }
        | IROp::VGetMSBs { .. }
        | IROp::VClz { .. }
        | IROp::VCls { .. }
        | IROp::VPolynomialMul { .. }
        | IROp::VMin { .. }
        | IROp::VMax { .. }
        | IROp::VAbs { .. }
        | IROp::VFAdd { .. }
        | IROp::VFSub { .. }
        | IROp::VFMul { .. }
        | IROp::VFDiv { .. }
        | IROp::VFSqrt { .. }
        | IROp::VFAbs { .. }
        | IROp::VFMin { .. }
        | IROp::VFMax { .. }
        | IROp::VFPwAdd { .. }
        | IROp::VFRecipEst { .. }
        | IROp::VFRecipStep { .. }
        | IROp::VFRSqrtEst { .. }
        | IROp::VFRSqrtStep { .. }
        | IROp::VFRecipEstS { .. }
        | IROp::VFRSqrtEstS { .. }
        | IROp::VIRecipEst { .. }
        | IROp::VIRSqrtEst { .. } => VexOpFamily::Vec,

        // x86-specific carry-less multiply / CRC32 — classified as Arith
        // (they're integer ops in the polynomial / checksum sense).
        IROp::PclmulLQLQ
        | IROp::PclmulHQHQ
        | IROp::PclmulLQHQ
        | IROp::PclmulHQLQ
        | IROp::Crc32C => VexOpFamily::Arith,

        // Raw opcode escape + NEON panic sentinel + unmapped-opcode
        // typed-error sentinel (angr-tkbr.2): not pre-classified.
        IROp::NeonUnimplemented(_) | IROp::Unmapped(_) | IROp::Raw(_) => VexOpFamily::Other,
    }
}
