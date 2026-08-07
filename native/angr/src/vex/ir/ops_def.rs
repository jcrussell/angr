use super::*;

/// Kind of FP compare used by SSE scalar-lane and packed-vector compares.
/// `Un` detects NaN (unordered). `Gt` / `Ge` are emitted by ARM NEON
/// (Iop_CmpGT/GE32Fx2) and SSE packed (Iop_CmpGT/GE32Fx4); the scalar-lane
/// SSE compares only emit Eq/Lt/Le/Un.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FCmpKind {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Un,
}

/// IR operations.
///
/// Unlike libVEX which has ~200 separate opcodes (e.g., Iop_Add8, Iop_Add16, ...),
/// we use parameterized operations to reduce code duplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IROp {
    // =========================================================================
    // Arithmetic (parameterized by width)
    // =========================================================================
    Add(IRType),
    Sub(IRType),
    Mul(IRType),
    MullS(IRType), // Signed widening multiply
    MullU(IRType), // Unsigned widening multiply
    DivS(IRType),  // Signed division
    DivU(IRType),  // Unsigned division
    ModS(IRType),  // Signed modulo
    ModU(IRType),  // Unsigned modulo

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit (low=quotient, high=remainder)
    DivModU64to32, // Unsigned
    DivModS64to32, // Signed

    /// DivMod: 128-bit dividend / 64-bit divisor -> 128-bit (low=quotient, high=remainder).
    /// libVEX types these `:: V128,I64 -> V128` (vector tag, not `Ity_I128`), and so do we —
    /// see `result_type`. The surrounding chain agrees: guests build the dividend with
    /// `Iop_64HLtoV128` and split the result with `Iop_V128to64` / `Iop_V128HIto64`.
    DivModU128to64, // Unsigned
    DivModS128to64, // Signed

    // =========================================================================
    // Bitwise (parameterized by width)
    // =========================================================================
    And(IRType),
    Or(IRType),
    Xor(IRType),
    Not(IRType),

    // =========================================================================
    // Shifts (parameterized by width)
    // =========================================================================
    Shl(IRType), // Logical left shift
    Shr(IRType), // Logical right shift
    Sar(IRType), // Arithmetic right shift

    // =========================================================================
    // Comparisons (return I1)
    // =========================================================================
    CmpEQ(IRType),  // Equal
    CmpNE(IRType),  // Not equal
    CmpLT(IRType),  // Less than (signed)
    CmpLE(IRType),  // Less or equal (signed)
    CmpLTU(IRType), // Less than (unsigned)
    CmpLEU(IRType), // Less or equal (unsigned)

    // =========================================================================
    // Conversions
    // =========================================================================
    /// Widen with sign extension.
    SignExtend {
        from: IRType,
        to: IRType,
    },
    /// Widen with zero extension.
    ZeroExtend {
        from: IRType,
        to: IRType,
    },
    /// Narrow (truncate).
    Truncate {
        from: IRType,
        to: IRType,
    },

    // =========================================================================
    // Bit manipulation
    // =========================================================================
    Clz(IRType),      // Count leading zeros
    Ctz(IRType),      // Count trailing zeros
    PopCount(IRType), // Population count

    // =========================================================================
    // Floating point operations
    // =========================================================================
    FAdd(IRType),
    FSub(IRType),
    FMul(IRType),
    FDiv(IRType),
    FNeg(IRType),
    FAbs(IRType),
    FSqrt(IRType),
    /// Fused multiply-add: a*b + c (with rounding mode)
    FMAdd(IRType),
    /// Fused multiply-sub: a*b - c (with rounding mode)
    FMSub(IRType),

    // Float comparisons
    FCmpEQ(IRType),
    FCmpLT(IRType),
    FCmpLE(IRType),

    /// SSE scalar-lane FP compare (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}).
    /// Operates on lane 0 only; result is V128 with lane 0 set to all-1s
    /// (e.g. 0xFFFFFFFF for F32, 0xFFFFFFFFFFFFFFFF for F64) on true and 0
    /// on false. Upper lanes are passed through from the left operand.
    /// `Un` is the unordered (NaN-detect) compare.
    FCmpScalarLane {
        kind: FCmpKind,
        ty: IRType,
    },

    /// Packed FP compare (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}).
    /// Each lane independently produces 0 (false) or all-ones (true).
    /// Result width = elem.bits() * count: 32Fx2 -> I64, 32Fx4 / 64Fx2 -> V128.
    /// `Un` is the unordered (NaN-detect) compare.
    FCmpVecPacked {
        kind: FCmpKind,
        elem: IRType,
        count: u8,
    },

    /// x87 FCOM-style compare (Iop_CmpF32/F64/F128).
    /// Returns I32 encoded as 0x00 = GT, 0x01 = LT, 0x40 = EQ, 0x45 = UN.
    FComCC(IRType),

    // Float conversions
    F32toF64,
    F64toF32,
    I32StoF32,
    I32StoF64,
    I64StoF32,
    I64StoF64,
    I32UtoF32,
    I32UtoF64,
    I64UtoF32,
    I64UtoF64,
    F32toI32S,
    F64toI32S,
    F32toI64S,
    F64toI64S,
    F32toI32U,
    F64toI32U,
    F32toI64U,
    F64toI64U,

    // Rounding mode operations
    RoundF32toInt,
    RoundF64toInt,

    // =========================================================================
    // Scalar-in-vector float operations (SSE scalar ops)
    // These operate on element 0 only, passing through other elements.
    // =========================================================================
    /// Scalar float add in vector (e.g., Add32F0x4 for ADDSS)
    VFAddS {
        elem: IRType,
    },
    /// Scalar float sub in vector (e.g., Sub32F0x4 for SUBSS)
    VFSubS {
        elem: IRType,
    },
    /// Scalar float mul in vector (e.g., Mul32F0x4 for MULSS)
    VFMulS {
        elem: IRType,
    },
    /// Scalar float div in vector (e.g., Div32F0x4 for DIVSS)
    VFDivS {
        elem: IRType,
    },
    /// Scalar float sqrt in vector (e.g., Sqrt32F0x4 for SQRTSS)
    VFSqrtS {
        elem: IRType,
    },
    /// Scalar float max in vector (e.g., Max32F0x4 for MAXSS)
    VFMaxS {
        elem: IRType,
    },
    /// Scalar float min in vector (e.g., Min32F0x4 for MINSS)
    VFMinS {
        elem: IRType,
    },
    /// Set low 32 bits of V128 (used by SSE scalar ops)
    SetV128lo32,
    /// Set low 64 bits of V128
    SetV128lo64,

    // =========================================================================
    // SIMD / Vector operations (parameterized)
    // =========================================================================
    /// Vector add: (element_type, num_elements)
    VAdd {
        elem: IRType,
        count: u8,
    },
    /// Vector sub
    VSub {
        elem: IRType,
        count: u8,
    },
    /// Vector mul
    VMul {
        elem: IRType,
        count: u8,
    },
    /// Widening vector multiply — each contributing `elem`-wide input lane is
    /// sign/zero-extended to `2*elem` bits, multiplied with the matching lane
    /// of the other operand, and truncated to `2*elem` bits to form one output
    /// lane. `even=false` widens all `count` lanes (Iop_Mull{N}{S,U}x{M},
    /// (I64,I64)->V128, NEON VMULL); `even=true` widens only the even-indexed
    /// lanes (Iop_MullEven{N}{S,U}x{M}, (V128,V128)->V128, SSE PMULDQ/PMULUDQ).
    VMull {
        elem: IRType,
        count: u8,
        signed: bool,
        even: bool,
    },
    /// Signed doubling saturating widening multiply — Iop_QDMull{N}Sx{M}
    /// ((I64,I64)->V128, NEON VQDMULL). Each `elem`-wide signed input lane is
    /// widened, multiplied with the matching lane of the other operand, the
    /// product doubled, then saturated into the signed `2*elem`-bit output
    /// lane. Saturation only triggers on the `MIN * MIN` corner (both inputs
    /// `-2^(elem-1)`), whose doubled product `2^(2*elem-1)` overflows the
    /// signed `2*elem`-bit max and clamps to `2^(2*elem-1) - 1`. Always signed,
    /// always full-lane (no even-lane variant exists in libVEX).
    VQDMull {
        elem: IRType,
        count: u8,
    },
    /// Vector and
    VAnd(IRType), // V128 or V256
    /// Vector or
    VOr(IRType),
    /// Vector xor
    VXor(IRType),
    /// Vector not
    VNot(IRType),
    /// Vector shift left (by immediate)
    VShlN {
        elem: IRType,
        count: u8,
    },
    /// Vector shift right logical
    VShrN {
        elem: IRType,
        count: u8,
    },
    /// Vector shift right arithmetic
    VSarN {
        elem: IRType,
        count: u8,
    },

    /// NEON vector shift left by vector — `Iop_Shl{N}x{M}` (and `Iop_Sal{N}x{M}`,
    /// which has identical bit-level semantics: left shift on two's complement
    /// is the same operation whether labelled "logical" or "arithmetic").
    /// Both operands are the full vector width; lane `i` of the result is
    /// `lane_a\[i\] << lane_b\[i\]`, with the shift amount treated as unsigned
    /// (Z3 `bvshl` semantics — counts ≥ lane width produce zero). Maps to
    /// ARM USHL (DDI 0487 C7.2.310) when the count vector is non-negative;
    /// the negative-count branch of NEON USHL/SSHL is decomposed by libVEX
    /// into a separate `Iop_Shr`/`Iop_Sar`, so this op only sees the
    /// unsigned-count case.
    VShl {
        elem: IRType,
        count: u8,
    },
    /// NEON vector shift right logical by vector — `Iop_Shr{N}x{M}`.
    /// Same shape as `VShl`; uses Z3 `bvlshr`. Maps to ARM USHL with negative
    /// (right) count after libVEX decomposition.
    VShr {
        elem: IRType,
        count: u8,
    },
    /// NEON vector shift right arithmetic by vector — `Iop_Sar{N}x{M}`.
    /// Same shape as `VShl`; uses Z3 `bvashr` (sign-replicating). Maps to
    /// ARM SSHL with negative (right) count after libVEX decomposition.
    VSar {
        elem: IRType,
        count: u8,
    },
    /// Vector compare equal
    VCmpEQ {
        elem: IRType,
        count: u8,
    },
    /// Vector compare greater than. `signed` selects the S-suffixed
    /// (`Iop_CmpGT{N}Sx{M}`) vs U-suffixed (`Iop_CmpGT{N}Ux{M}`) family —
    /// ARM NEON `VCGT.U*` and SSE/AVX unsigned compares need the latter
    /// (angr-9ke6b.160).
    VCmpGT {
        elem: IRType,
        count: u8,
        signed: bool,
    },
    /// Interleave high
    VInterleaveLO {
        elem: IRType,
    },
    VInterleaveHI {
        elem: IRType,
    },
    /// Permute/shuffle (`Iop_Perm{N}x{M}`): (table, control) -> vec.
    /// `count` is the lane count, so the result width is `elem * count`
    /// (64 for `Perm8x8`, 128 for `Perm8x16`/`Perm32x4`, 256 for `Perm32x8`).
    /// No native dispatch arm — see `is_dispatch_fabricate_family`
    /// (`interpreter/expressions.rs`); the whole family routes to Python.
    VPerm {
        elem: IRType,
        count: u8,
    },
    /// NEON lane extract (Iop_GetElem{N}x{M}): (vec, idx) -> scalar lane.
    /// Binop; idx is Ity_I8. Result width = elem.bits().
    VGetElem {
        elem: IRType,
        count: u8,
    },
    /// NEON lane insert (Iop_SetElem{N}x{M}): (vec, idx, val) -> vec.
    /// Triop in VEX, but not rm-bearing — dispatched through
    /// binop_with_rm by reinterpreting (rm, left, right) as (vec, idx, val).
    VSetElem {
        elem: IRType,
        count: u8,
    },

    /// NEON broadcast scalar to vector (Iop_Dup{N}x{M}): (scalar) -> vec.
    /// Unop. Input width = elem.bits(); result width = elem.bits() * count.
    VDup {
        elem: IRType,
        count: u8,
    },

    /// NEON widen each lane (Iop_Widen{N}{S/U}to{2N}x{M}): (vec) -> vec.
    /// Unop. Input has `count` lanes of width `from`; result has `count` lanes
    /// of width `from.bits()*2`. `signed` selects sign- vs zero-extension.
    VWiden {
        from: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON unary narrow (Iop_NarrowUn{N}to{N/2}x{M}): (vec) -> vec.
    /// Unop. Input has `count` lanes of width `from`; result has `count` lanes
    /// of width `from.bits()/2` (low bits truncated).
    VNarrowUn {
        from: IRType,
        count: u8,
    },

    /// NEON binary narrow (Iop_NarrowBin{N}to{N/2}x{M}): (lo, hi) -> vec.
    /// Binop. `count` is the result lane count; each input has `count/2` lanes
    /// of width `from`. Result has `count` lanes of width `from.bits()/2`.
    VNarrowBin {
        from: IRType,
        count: u8,
    },

    /// NEON unary saturating narrow (Iop_QNarrowUn{N}{S/U}to{N/2}{S/U}x{M}).
    /// Unop. Same shape as VNarrowUn but saturates instead of truncating.
    /// `src_signed` reflects the source interpretation; `dst_signed` the
    /// saturation range (signed -> [-2^(w-1), 2^(w-1)-1], unsigned -> [0, 2^w-1]).
    VQNarrowUn {
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
    },

    /// NEON binary saturating narrow (Iop_QNarrowBin{N}{S/U}to{N/2}{S/U}x{M}).
    /// Binop variant of VQNarrowUn; `count` is total result lanes (each input
    /// contributes `count/2` lanes).
    VQNarrowBin {
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
    },

    /// NEON byte/halfword/word/bit reversal within each lane —
    /// Iop_Reverse{sub_width}sIn{elem.bits()}_x{count}. Reverse the order of
    /// `sub_width`-bit sub-units inside each `elem`-wide lane; `count` lanes
    /// total. Maps to ARM REV16/REV32/REV64 (sub_width = 8/16/32) and RBIT
    /// (sub_width = 1). Result width = elem.bits() * count.
    VReverse {
        sub_width: u8,
        elem: IRType,
        count: u8,
    },

    /// NEON saturating integer add — Iop_QAdd{N}{S/U}x{M}.
    /// Per-lane addition where positive/negative overflow clamps to the lane
    /// type's max/min. Signed lanes clamp to `[-2^(N-1), 2^(N-1)-1]`; unsigned
    /// lanes clamp to `[0, 2^N - 1]`. Maps to ARM VQADD (DDI 0487 C7.2.379)
    /// and SSE PADDS{B,W}/PADDUS{B,W}.
    VQAdd {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON saturating integer sub — Iop_QSub{N}{S/U}x{M}.
    /// Per-lane subtraction with the same clamp semantics as VQAdd. Maps to
    /// ARM VQSUB (DDI 0487 C7.2.395) and SSE PSUBS{B,W}/PSUBUS{B,W}.
    VQSub {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned,
    /// `signed=false`) / `Iop_QSal{N}x{M}` (signed, `signed=true`). Unlike
    /// `VQAdd`/`VQSub` the signedness is encoded in the *prefix* (Shl vs Sal),
    /// not a `S`/`U` infix in the suffix. Per-lane semantics:
    ///   * Cast the shift-amount lane as a signed `elem`-bit integer.
    ///   * If `amt >= 0`: shift left by `amt`; saturate the result to
    ///     `[0, 2^N-1]` (unsigned) or `[-2^(N-1), 2^(N-1)-1]` (signed).
    ///     Shifts ≥ lane width force the saturation boundary based on the
    ///     sign of the operand.
    ///   * If `amt < 0`: shift right (logical for unsigned, arithmetic for
    ///     signed) by `-amt`; OOR right shifts collapse to 0 (unsigned) or
    ///     sign-fill (signed).
    ///
    /// Maps to ARM UQSHL / SQSHL (DDI 0487 C7.2.327 / C7.2.298). Derived from
    /// libVEX `host_generic_simd64/simd128.c` h_generic_calc_QShl* helpers;
    /// no `_op_generic_QShl` exists in claripy.
    VQShlSat {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer add — `Iop_PwAdd{N}x{M}`. Binary; output has the
    /// same lane width and count as the inputs. Per-lane semantics:
    ///   * result\[i\]            = a\[2i\]   + a\[2i+1\]            for i in 0..count/2
    ///   * result[count/2 + i]  = b\[2i\]   + b\[2i+1\]            for i in 0..count/2
    ///
    /// Maps to ARM VPADD (DDI 0487 C7.2.270) — `Iop_PwAdd32Fx2` (FP variant)
    /// is NOT routed here and remains unimplemented.
    VPwAdd {
        elem: IRType,
        count: u8,
    },

    /// NEON pairwise widening integer add — `Iop_PwAddL{N}{S/U}x{M}`. Unary;
    /// output lane width is `2 * elem`, lane count is `count / 2`, total
    /// width preserved. Per-lane semantics:
    ///   * result\[i\] = sext_or_zext(a\[2i\]) + sext_or_zext(a\[2i+1\])
    ///
    /// Maps to ARM SADDLP / UADDLP (DDI 0487 C7.2.348 / C7.2.418).
    VPwAddL {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer min — `Iop_PwMin{N}{S/U}x{M}`. Binary; same shape
    /// rules as `VPwAdd` (interleave a-half then b-half). Maps to ARM VPMIN
    /// (DDI 0487 C7.2.273).
    VPwMin {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer max — `Iop_PwMax{N}{S/U}x{M}`. Binary; same shape
    /// rules as `VPwAdd`. Maps to ARM VPMAX (DDI 0487 C7.2.272).
    VPwMax {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise FP add — `Iop_PwAdd32Fx2` (ARM VPADD.F32, D-reg). Binary;
    /// the FP analogue of `VPwAdd`, with the same interleave shape (first half
    /// from `a`, second half from `b`) but per-lane FP add instead of integer
    /// add. VEX only emits the `32Fx2` form (two F32 lanes per 64-bit reg), so
    /// the single output pair is `[a0+a1, b0+b1]`. Returns `Ity_I64`.
    VFPwAdd {
        elem: IRType,
        count: u8,
    },

    /// NEON rounding halving add (a.k.a. rounding-average) —
    /// `Iop_Avg{N}{S/U}x{M}`. Binary; output has the same lane shape as the
    /// inputs. Per-lane semantics (widening to `elem+1` bits avoids overflow):
    ///   result\[i\] = ((a\[i\] + b\[i\] + 1) >> 1) truncated to `elem` bits.
    /// Unsigned variants map to ARM URHADD (DDI 0487 C7.2.420) and SSE
    /// PAVGB/PAVGW (which are unsigned-only). Signed variants map to ARM
    /// SRHADD (DDI 0487 C7.2.353). Distinct from the truncating halving add
    /// `(a+b) >> 1` exposed in claripy as `_op_generic_HAdd`.
    VAvg {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON per-byte population count — `Iop_Cnt8x{8,16}`. Unary; each 8-bit
    /// lane is replaced by the number of set bits in that lane (0..=8). Result
    /// width preserved (8 * count bits). Maps to ARM CNT (DDI 0487 C7.2.62).
    /// VEX only defines this op for 8-bit lanes — wider element popcounts are
    /// not part of the NEON ISA.
    VCnt {
        count: u8,
    },

    /// NEON per-lane count leading zeros — `Iop_Clz{N}x{M}`. Unary; each
    /// `elem`-wide lane is replaced by its leading-zero count (0..=N). Result
    /// width preserved. Maps to ARM CLZ (DDI 0487 C7.2.57); lanes are 8/16/32
    /// bits across D-reg (total=64) and Q-reg (total=128) shapes.
    VClz {
        elem: IRType,
        count: u8,
    },

    /// NEON per-lane count leading sign bits — `Iop_Cls{N}x{M}`. Unary; each
    /// `elem`-wide lane is replaced by the number of consecutive bits below
    /// the most significant bit that equal the MSB (range 0..=N-1). All-sign
    /// lanes yield `N-1`. Maps to ARM CLS (DDI 0487 C7.2.56); lanes are 8/16/32
    /// bits across D-reg (total=64) and Q-reg (total=128) shapes.
    VCls {
        elem: IRType,
        count: u8,
    },

    /// SSE byte-mask extract — `Iop_GetMSBs8x{8,16}` (x86 PMOVMSKB). Unary;
    /// reduces a vector of `count` bytes to a `count`-bit integer whose bit
    /// `i` is the most-significant bit (bit 7) of byte `i`. Result is I8 for
    /// the V64 (8-byte) form and I16 for the V128 (16-byte) form. Hit early in
    /// real glibc SSE string routines (strlen/memchr); without it vanilla
    /// symbolic execution of libc-linked binaries errors on startup (angr-75mc).
    VGetMSBs {
        count: u8,
    },

    /// NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    /// widening, `widen=false`) and `Iop_PolynomialMull8x8` (widening,
    /// `widen=true`). Per-lane carry-less multiply over GF(2): for `a*b` with
    /// `a,b` being 8-bit polynomials, the result is XOR of shifted copies of
    /// `b` selected by the bits of `a`. Non-widening returns the low 8 bits
    /// per lane (width preserved). Widening returns 16 bits per lane (output
    /// total = 2 * input total). Maps to ARM PMUL / PMULL (DDI 0487 C7.2.281).
    /// Claripy has no generic equivalent — universality tests use the spec-
    /// replay template against the same XOR-shift primitive.
    VPolynomialMul {
        count: u8,
        widen: bool,
    },

    /// NEON integer reciprocal estimate — `Iop_RecipEst32Ux{2,4}` (ARM URECPE,
    /// DDI 0487 C7.2.336). Per-32-bit-lane unsigned reciprocal seed for the
    /// software Newton-Raphson refinement loop. Returns a fresh symbolic per
    /// lane (matches the policy used for the FP variants `VFRecipEst`):
    /// claripy raises `UnsupportedIROpError` on these integer ops, so any
    /// precision result computed here would be more faithful than angr Python
    /// and could cause cross-engine divergence. Binaries that use URECPE
    /// typically follow with one or more refinement steps which converge to
    /// the exact reciprocal regardless of the seed.
    VIRecipEst {
        count: u8,
    },

    /// NEON integer reciprocal-sqrt estimate — `Iop_RSqrtEst32Ux{2,4}` (ARM
    /// URSQRTE, DDI 0487 C7.2.351). Fresh-symbolic per 32-bit lane; see
    /// `VIRecipEst` for the policy rationale.
    VIRSqrtEst {
        count: u8,
    },

    // =========================================================================
    // Packed integer min/max/abs
    // =========================================================================
    /// Packed integer min (PMINSB/PMINSW/PMINSD/PMINUB/PMINUW/PMINUD)
    VMin {
        elem: IRType,
        count: u8,
        signed: bool,
    },
    /// Packed integer max (PMAXSB/PMAXSW/PMAXSD/PMAXUB/PMAXUW/PMAXUD)
    VMax {
        elem: IRType,
        count: u8,
        signed: bool,
    },
    /// Packed integer absolute value (PABSB/PABSW/PABSD/PABSQ)
    VAbs {
        elem: IRType,
        count: u8,
    },

    // =========================================================================
    // Packed FP arithmetic (whole-vector — *not* the scalar-lane VF*S variants)
    // =========================================================================
    /// Packed float add (ADDPS/ADDPD)
    VFAdd {
        elem: IRType,
        count: u8,
    },
    /// Packed float sub (SUBPS/SUBPD)
    VFSub {
        elem: IRType,
        count: u8,
    },
    /// Packed float mul (MULPS/MULPD)
    VFMul {
        elem: IRType,
        count: u8,
    },
    /// Packed float div (DIVPS/DIVPD)
    VFDiv {
        elem: IRType,
        count: u8,
    },
    /// Packed float sqrt (SQRTPS/SQRTPD)
    VFSqrt {
        elem: IRType,
        count: u8,
    },
    /// Packed float abs (Iop_Abs32Fx4/Iop_Abs64Fx2)
    VFAbs {
        elem: IRType,
        count: u8,
    },
    /// Packed float min (MINPS/MINPD)
    VFMin {
        elem: IRType,
        count: u8,
    },
    /// Packed float max (MAXPS/MAXPD)
    VFMax {
        elem: IRType,
        count: u8,
    },
    /// Packed FP reciprocal estimate (1/x approximation) — RCPPS / NEON FRECPE.
    /// Returns a fresh symbolic per lane: VEX leaves precision implementation-
    /// defined, so binaries that use this typically follow with one or two
    /// Newton-Raphson refinement steps (RecipStep) which converge to the exact
    /// 1/x irrespective of the seed.
    VFRecipEst {
        elem: IRType,
        count: u8,
    },
    /// Packed FP Newton-Raphson reciprocal step — NEON FRECPS.
    /// Mathematically 2.0 - x*y per lane; treated as fresh-symbolic per lane to
    /// match angr Python's conservative handling (no `_op_fgeneric_RecipStep`).
    VFRecipStep {
        elem: IRType,
        count: u8,
    },
    /// Packed FP reciprocal-sqrt estimate (1/sqrt(x) approximation) —
    /// RSQRTPS / NEON FRSQRTE. Fresh-symbolic per lane, see VFRecipEst.
    VFRSqrtEst {
        elem: IRType,
        count: u8,
    },
    /// Packed FP Newton-Raphson reciprocal-sqrt step — NEON FRSQRTS.
    /// Mathematically (3.0 - x*y*y) / 2.0 per lane; treated as fresh-symbolic
    /// per lane (mirrors angr Python, which has no `_op_fgeneric_RSqrtStep`).
    VFRSqrtStep {
        elem: IRType,
        count: u8,
    },
    /// SSE scalar-in-vector reciprocal estimate (RCPSS, Iop_RecipEst32F0x4).
    /// Lane 0 fresh-symbolic, upper lanes pass through from arg.
    VFRecipEstS {
        elem: IRType,
    },
    /// SSE scalar-in-vector reciprocal-sqrt estimate (RSQRTSS,
    /// Iop_RSqrtEst32F0x4). Lane 0 fresh-symbolic, upper lanes pass through.
    VFRSqrtEstS {
        elem: IRType,
    },

    // =========================================================================
    // Special operations
    // =========================================================================
    /// Reinterpret bits as different type.
    Reinterpret {
        from: IRType,
        to: IRType,
    },

    /// Concatenate two values.
    Concat {
        ty: IRType,
    },

    /// Extract bits.
    Extract {
        from: IRType,
        to: IRType,
        low_bit: u8,
    },

    // =========================================================================
    // x86-specific operations
    // =========================================================================
    /// x86 PCLMUL (carry-less multiply)
    PclmulLQLQ,
    PclmulHQHQ,
    PclmulLQHQ,
    PclmulHQLQ,

    /// x86 CRC32
    Crc32C,

    // =========================================================================
    // ARM/AArch64 NEON SIMD ops (scaffolded — panic on dispatch)
    // =========================================================================
    /// Placeholder for a NEON SIMD opcode that has been mapped from pyvex
    /// (so it does NOT silently fall back to a fresh-symbolic result), but
    /// whose semantics have not been implemented yet. Dispatch (`VEXOps::unop`,
    /// `VEXOps::binop`, etc.) panics with the captured opcode name so missing
    /// NEON coverage is visible immediately. Implementations land one-by-one
    /// in angr-bkcs.2 by replacing the matching `opcode_map` entry with a
    /// real `IROp::V*` variant.
    NeonUnimplemented(&'static str),

    // =========================================================================
    // Unmapped opcode (no entry in `parse_opcode`)
    // =========================================================================
    /// Opcode string that `parse_opcode` could not match to any known
    /// `IROp` variant. Holds an interned `&'static str` of the original
    /// pyvex opcode name (e.g. `"Iop_FakeNotARealOp"`). Dispatch in
    /// `VEXOps::unop`/`binop`/`ternop`/`qop` surfaces this as
    /// `OpError::UnsupportedVexOp { op_name }`, which the engine maps to
    /// `RustUnsupportedVexOpError(op_name, arch)` (angr-tkbr.2). Replaces
    /// the previous silent `IROp::Raw(0)` fallback that lost the name
    /// and produced fresh-symbolic results.
    Unmapped(&'static str),

    // =========================================================================
    // Raw VEX opcode (for unhandled operations)
    // =========================================================================
    /// Fallback for operations not yet implemented.
    Raw(u32),
}

impl IROp {
    /// Map a packed-vector's total bit-width to its `IRType`:
    /// 64 → I64 (NEON D-reg), 128 → V128 (Q-reg / SSE), 256 → V256 (AVX);
    /// anything else → None. Arms whose decode never yields a 256-bit total
    /// (D/Q-reg-only families) can share this — 256 is simply unreachable
    /// for them, so the extra branch is harmless.
    fn width_total_to_type(total: u32) -> Option<IRType> {
        match total {
            64 => Some(IRType::I64),
            128 => Some(IRType::V128),
            256 => Some(IRType::V256),
            _ => None,
        }
    }

    /// Get the result type of this operation.
    pub fn result_type(&self) -> Option<IRType> {
        match self {
            // Arithmetic ops return same type as input
            IROp::Add(t)
            | IROp::Sub(t)
            | IROp::Mul(t)
            | IROp::DivS(t)
            | IROp::DivU(t)
            | IROp::ModS(t)
            | IROp::ModU(t) => Some(*t),

            // Widening multiply
            IROp::MullS(t) | IROp::MullU(t) => match t {
                IRType::I8 => Some(IRType::I16),
                IRType::I16 => Some(IRType::I32),
                IRType::I32 => Some(IRType::I64),
                IRType::I64 => Some(IRType::I128),
                _ => None,
            },

            // DivMod: 64-bit / 32-bit -> 64-bit
            IROp::DivModU64to32 | IROp::DivModS64to32 => Some(IRType::I64),

            // DivMod: 128-bit / 64-bit -> 128-bit. Vector-tagged (`V128`) to match libVEX's
            // `:: V128,I64 -> V128` signature; only the width (128) is load-bearing here,
            // since `divmod_128_to_64` dispatches on `.width()` alone.
            IROp::DivModU128to64 | IROp::DivModS128to64 => Some(IRType::V128),

            // Bitwise ops return same type
            IROp::And(t)
            | IROp::Or(t)
            | IROp::Xor(t)
            | IROp::Not(t)
            | IROp::Shl(t)
            | IROp::Shr(t)
            | IROp::Sar(t) => Some(*t),

            // Comparisons return I1
            IROp::CmpEQ(_)
            | IROp::CmpNE(_)
            | IROp::CmpLT(_)
            | IROp::CmpLE(_)
            | IROp::CmpLTU(_)
            | IROp::CmpLEU(_) => Some(IRType::I1),

            // Conversions
            IROp::SignExtend { to, .. }
            | IROp::ZeroExtend { to, .. }
            | IROp::Truncate { to, .. } => Some(*to),

            // Bit manipulation returns same type
            IROp::Clz(t) | IROp::Ctz(t) | IROp::PopCount(t) => Some(*t),

            // Float ops
            IROp::FAdd(t)
            | IROp::FSub(t)
            | IROp::FMul(t)
            | IROp::FDiv(t)
            | IROp::FNeg(t)
            | IROp::FAbs(t)
            | IROp::FSqrt(t)
            | IROp::FMAdd(t)
            | IROp::FMSub(t) => Some(*t),

            IROp::FCmpEQ(_) | IROp::FCmpLT(_) | IROp::FCmpLE(_) => Some(IRType::I1),

            // Scalar-lane SSE compares write into a V128 register (lane 0 mask
            // + upper lanes from `left`).
            IROp::FCmpScalarLane { .. } => Some(IRType::V128),

            // Packed FP compare: total = elem.bits() * count.
            // 32Fx2 -> I64, 32Fx4 / 64Fx2 -> V128.
            IROp::FCmpVecPacked { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // x87 FCOM-style compare encodes the result as a 32-bit value.
            IROp::FComCC(_) => Some(IRType::I32),

            // Scalar-in-vector float ops return V128
            IROp::VFAddS { elem: _elem }
            | IROp::VFSubS { elem: _elem }
            | IROp::VFMulS { elem: _elem }
            | IROp::VFDivS { elem: _elem }
            | IROp::VFSqrtS { elem: _elem }
            | IROp::VFMaxS { elem: _elem }
            | IROp::VFMinS { elem: _elem }
            | IROp::VFRecipEstS { elem: _elem }
            | IROp::VFRSqrtEstS { elem: _elem } => Some(IRType::V128),

            // SetV128lo ops return V128
            IROp::SetV128lo32 | IROp::SetV128lo64 => Some(IRType::V128),

            // Float conversions
            IROp::F32toF64 => Some(IRType::F64),
            IROp::F64toF32 => Some(IRType::F32),
            IROp::I32StoF32 | IROp::I32UtoF32 | IROp::I64StoF32 | IROp::I64UtoF32 => {
                Some(IRType::F32)
            }
            IROp::I32StoF64 | IROp::I32UtoF64 | IROp::I64StoF64 | IROp::I64UtoF64 => {
                Some(IRType::F64)
            }
            IROp::F32toI32S | IROp::F64toI32S | IROp::F32toI32U | IROp::F64toI32U => {
                Some(IRType::I32)
            }
            IROp::F32toI64S | IROp::F64toI64S | IROp::F32toI64U | IROp::F64toI64U => {
                Some(IRType::I64)
            }
            IROp::RoundF32toInt => Some(IRType::F32),
            IROp::RoundF64toInt => Some(IRType::F64),

            // Vector ops
            IROp::VAnd(t) | IROp::VOr(t) | IROp::VXor(t) | IROp::VNot(t) => Some(*t),
            // Width-preserving packed lane ops: total = elem * count. The
            // mapped shapes span D-reg (64), Q-reg/SSE (128) and — for
            // VAdd/VSub — AVX2 (256, e.g. Iop_Add8x32), so this must be
            // computed rather than hardcoded to V128 (angr-9ke6b.163).
            // Packed compares are included: they yield a full-width lane mask.
            IROp::VAdd { elem, count }
            | IROp::VSub { elem, count }
            | IROp::VMul { elem, count }
            | IROp::VShlN { elem, count }
            | IROp::VShrN { elem, count }
            | IROp::VSarN { elem, count }
            | IROp::VCmpEQ { elem, count }
            | IROp::VCmpGT { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Widening multiplies. `VMull` with `even=false` (Iop_Mull{N}{S,U}x{M})
            // doubles each lane width: total = 2 * elem * count. `even=true`
            // (Iop_MullEven*) halves the lane count while doubling the width, so
            // total = elem * count. `VQDMull` is always full-lane widening.
            IROp::VMull {
                elem, count, even, ..
            } => {
                let lanes = if *even {
                    *count as u32 / 2
                } else {
                    *count as u32
                };
                Self::width_total_to_type(elem.bits() * 2 * lanes)
            }
            IROp::VQDMull { elem, count } => {
                Self::width_total_to_type(elem.bits() * 2 * (*count as u32))
            }

            // Vector shift by vector (Iop_Shl/Shr/Sar/Sal{N}x{M}): width
            // preserved — total = elem * count, either 64 or 128 bits.
            IROp::VShl { elem, count }
            | IROp::VShr { elem, count }
            | IROp::VSar { elem, count } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }
            IROp::VInterleaveLO { .. } | IROp::VInterleaveHI { .. } => Some(IRType::V128),

            // Perm preserves the vector width: total = elem * count, which is
            // 64 for Iop_Perm8x8 and 256 for Iop_Perm32x8 — neither of which
            // the old hardcoded V128 covered (angr-sqfj8.117).
            IROp::VPerm { elem, count } => Self::width_total_to_type(elem.bits() * (*count as u32)),

            // GetElem returns one lane.
            IROp::VGetElem { elem, .. } => Some(*elem),

            // SetElem returns the full vector — width = elem * count.
            IROp::VSetElem { elem, count } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Dup: total width = elem * count.
            IROp::VDup { elem, count } => Self::width_total_to_type(elem.bits() * (*count as u32)),

            // Widen: each lane doubles in width; total = (from.bits()*2) * count.
            IROp::VWiden { from, count, .. } => {
                Self::width_total_to_type(from.bits() * 2 * (*count as u32))
            }

            // Narrow (unary or binary, saturating or not): each lane halves;
            // total = (from.bits()/2) * count.
            IROp::VNarrowUn { from, count }
            | IROp::VNarrowBin { from, count }
            | IROp::VQNarrowUn { from, count, .. }
            | IROp::VQNarrowBin { from, count, .. } => {
                Self::width_total_to_type((from.bits() / 2) * (*count as u32))
            }

            // Reverse: width preserved (sub-units permuted within each lane).
            IROp::VReverse { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Saturating add/sub: width preserved.
            IROp::VQAdd { elem, count, .. } | IROp::VQSub { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Saturating shift by vector: width preserved.
            IROp::VQShlSat { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Pairwise add/min/max (non-widening): output width = elem * count.
            // The FP pairwise add `VFPwAdd` shares the same width rule (32Fx2 →
            // 64-bit → Ity_I64).
            IROp::VPwAdd { elem, count }
            | IROp::VFPwAdd { elem, count }
            | IROp::VPwMin { elem, count, .. }
            | IROp::VPwMax { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Pairwise widening add: output total = input total = elem * count
            // (lane width doubles, lane count halves).
            IROp::VPwAddL { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Rounding halving add (Iop_Avg*): width preserved.
            IROp::VAvg { elem, count, .. } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Per-byte popcount (Iop_Cnt8x{8,16}): width preserved.
            IROp::VCnt { count } => match *count {
                8 => Some(IRType::I64),
                16 => Some(IRType::V128),
                _ => None,
            },

            // PMOVMSKB (Iop_GetMSBs8x{8,16}): reduces N bytes to an N-bit int.
            IROp::VGetMSBs { count } => match *count {
                8 => Some(IRType::I8),
                16 => Some(IRType::I16),
                _ => None,
            },

            // Per-lane Clz/Cls: width preserved (lane width = elem bits).
            IROp::VClz { elem, count } | IROp::VCls { elem, count } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            // Polynomial multiply (Iop_PolynomialMul / Mull): non-widening
            // preserves width; widening doubles per-lane width (8 → 16) so
            // 8x8 → V128 and the (theoretical) 8x16 widening shape (not
            // emitted by libVEX) would be V256.
            IROp::VPolynomialMul { count, widen } => {
                let elem_out = if *widen { 16 } else { 8 };
                Self::width_total_to_type(elem_out * (*count as u32))
            }

            // Packed integer min/max/abs and packed FP arith all return V128 (or V256
            // for AVX variants — we pick V128 to match the rest of the family for now)
            IROp::VMin { .. }
            | IROp::VMax { .. }
            | IROp::VAbs { .. }
            | IROp::VFAdd { .. }
            | IROp::VFSub { .. }
            | IROp::VFMul { .. }
            | IROp::VFDiv { .. }
            | IROp::VFSqrt { .. }
            | IROp::VFAbs { .. }
            | IROp::VFMin { .. }
            | IROp::VFMax { .. } => Some(IRType::V128),

            // Integer reciprocal/rsqrt estimates (ARM URECPE / URSQRTE): 32-bit
            // lanes, count=2 → I64 (D-reg) or count=4 → V128 (Q-reg).
            IROp::VIRecipEst { count } | IROp::VIRSqrtEst { count } => match *count {
                2 => Some(IRType::I64),
                4 => Some(IRType::V128),
                _ => None,
            },

            // Newton-Raphson reciprocal/rsqrt families: result width = elem * count
            // (NEON D-reg variants are I64, Q-reg variants are V128).
            IROp::VFRecipEst { elem, count }
            | IROp::VFRecipStep { elem, count }
            | IROp::VFRSqrtEst { elem, count }
            | IROp::VFRSqrtStep { elem, count } => {
                Self::width_total_to_type(elem.bits() * (*count as u32))
            }

            IROp::Reinterpret { to, .. } => Some(*to),
            IROp::Concat { ty } => Some(*ty),
            IROp::Extract { to, .. } => Some(*to),

            IROp::PclmulLQLQ
            | IROp::PclmulHQHQ
            | IROp::PclmulLQHQ
            | IROp::PclmulHQLQ
            | IROp::Crc32C => Some(IRType::I64),

            // NEON ops are scaffolded — dispatch panics before result_type
            // is consulted in a hot path. Returning None here means callers
            // that *do* peek at the result type (e.g. fallback width guess
            // in expressions.rs) won't crash, but in practice the dispatch
            // panic fires first.
            IROp::NeonUnimplemented(_) => None,

            // Unmapped opcode — dispatch surfaces UnsupportedVexOp before
            // result_type is consulted. None matches the Raw(_) convention.
            IROp::Unmapped(_) => None,

            IROp::Raw(_) => None,
        }
    }
}
