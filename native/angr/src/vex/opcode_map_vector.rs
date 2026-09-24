//! Vector / SIMD opcode parsing for [`super::opcode_map`].
//!
//! Holds the three `parse_*` families that dominate the NEON and AVX opcode
//! surface: `parse_vector` (the integer SIMD families — add/sub/mul, min/max,
//! saturating and widening/narrowing forms, lane extract/insert, shifts),
//! `parse_vreverse` (`Iop_Reverse{N}sIn{M}_x{K}`) and
//! `parse_neon_unimplemented` (the claimed-but-unhandled scaffold).
//!
//! Siblings: `super::opcode_map` (dispatcher, macros, integer families,
//! `parse_special`, type/endness/jumpkind parsing) and
//! `super::opcode_map_float` (scalar/packed FP and transcendentals).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::ir::{IROp, IRType};
use super::opcode_map::{
    cast_arms, vec_arms, vec_mull_arms, vec_narrow_arms, vec_qnarrow_arms, vec_signed_arms,
    vec_widen_arms,
};

/// Parse vector/SIMD operations
pub(super) fn parse_vector(op_str: &str) -> Option<IROp> {
    // Vector add/sub: 8/16/32/64-bit elements across NEON-D / NEON-Q / AVX widths.
    vec_arms!(op_str; "Iop_Add" => VAdd {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    vec_arms!(op_str; "Iop_Sub" => VSub {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });

    // NEON saturating add/sub — Iop_QAdd{N}{S/U}x{M} / Iop_QSub{N}{S/U}x{M}.
    // Per-lane saturating arithmetic; S/U selects clamp range. D-reg
    // (total=64), Q-reg (total=128) and AVX2 (total=256) variants. The 256-bit
    // tier is `VPADDS{B,W}`/`VPADDUS{B,W}` and their `VPSUB*` twins, hence
    // 8/16-bit lanes only — `vendor/pyvex_ffi.h` declares no `Iop_QAdd32*x8`
    // or `Iop_QAdd64*x4` (angr-li4ox.1).
    //
    // Widening this table to V256 needs no evaluator change:
    // `VEXOps::vec_int_saturating` gates its u128 concrete fast path on
    // `total_width <= 128` and otherwise runs the symbolic per-lane path,
    // whose `concat_le_elements` keeps a symbolic `Concat` above 128 bits
    // (bd `invariant-concrete-bv-u128-16-byte-limit`, hazard family 3).
    vec_signed_arms!(op_str; "Iop_QAdd" => VQAdd {
        "8Sx8" => (I8, 8, true), "16Sx4" => (I16, 4, true),
        "32Sx2" => (I32, 2, true), "64Sx1" => (I64, 1, true),
        "8Ux8" => (I8, 8, false), "16Ux4" => (I16, 4, false),
        "32Ux2" => (I32, 2, false), "64Ux1" => (I64, 1, false),
        "8Sx16" => (I8, 16, true), "16Sx8" => (I16, 8, true),
        "32Sx4" => (I32, 4, true), "64Sx2" => (I64, 2, true),
        "8Ux16" => (I8, 16, false), "16Ux8" => (I16, 8, false),
        "32Ux4" => (I32, 4, false), "64Ux2" => (I64, 2, false),
        "8Sx32" => (I8, 32, true), "16Sx16" => (I16, 16, true),
        "8Ux32" => (I8, 32, false), "16Ux16" => (I16, 16, false),
    });
    vec_signed_arms!(op_str; "Iop_QSub" => VQSub {
        "8Sx8" => (I8, 8, true), "16Sx4" => (I16, 4, true),
        "32Sx2" => (I32, 2, true), "64Sx1" => (I64, 1, true),
        "8Ux8" => (I8, 8, false), "16Ux4" => (I16, 4, false),
        "32Ux2" => (I32, 2, false), "64Ux1" => (I64, 1, false),
        "8Sx16" => (I8, 16, true), "16Sx8" => (I16, 8, true),
        "32Sx4" => (I32, 4, true), "64Sx2" => (I64, 2, true),
        "8Ux16" => (I8, 16, false), "16Ux8" => (I16, 8, false),
        "32Ux4" => (I32, 4, false), "64Ux2" => (I64, 2, false),
        "8Sx32" => (I8, 32, true), "16Sx16" => (I16, 16, true),
        "8Ux32" => (I8, 32, false), "16Ux16" => (I16, 16, false),
    });

    // NEON vector shift by vector — `Iop_Shl{N}x{M}` / `Iop_Shr{N}x{M}` /
    // `Iop_Sar{N}x{M}` / `Iop_Sal{N}x{M}`. Both operands are full-vector;
    // each lane shifts by the corresponding count lane (Z3 bvshl/bvlshr/bvashr
    // semantics — counts ≥ lane width produce zero or sign-fill). `Sal` shares
    // semantics with `Shl` on two's complement; libVEX emits both names from
    // ARM SSHL/USHL decomposition (positive-count branches).
    //
    // Only `Sal` has a D-reg 64-bit-lane form: `vendor/pyvex_ffi.h` declares
    // `Iop_Sal64x1` but no `Iop_Shl64x1` / `Iop_Shr64x1` / `Iop_Sar64x1`, so the
    // other three tables stop at `32x2` (angr-0jh0j.62 — the `64x1` arms were
    // copy-pasted from `Sal` and could never match).
    vec_arms!(op_str; "Iop_Shl" => VShl {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Sal" => VShl {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2), "64x1" => (I64, 1),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Shr" => VShr {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Sar" => VSar {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });

    // NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned) /
    // `Iop_QSal{N}x{M}` (signed). Signedness is encoded in the prefix, not an
    // S/U infix, so it can't vary within one arm table — but `vec_signed_arms!`
    // still fits: route each prefix through its own invocation with the sign
    // baked as a constant into every arm. Maps to ARM UQSHL / SQSHL.
    vec_signed_arms!(op_str; "Iop_QShl" => VQShlSat {
        "8x8" => (I8, 8, false), "16x4" => (I16, 4, false),
        "32x2" => (I32, 2, false), "64x1" => (I64, 1, false),
        "8x16" => (I8, 16, false), "16x8" => (I16, 8, false),
        "32x4" => (I32, 4, false), "64x2" => (I64, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_QSal" => VQShlSat {
        "8x8" => (I8, 8, true), "16x4" => (I16, 4, true),
        "32x2" => (I32, 2, true), "64x1" => (I64, 1, true),
        "8x16" => (I8, 16, true), "16x8" => (I16, 8, true),
        "32x4" => (I32, 4, true), "64x2" => (I64, 2, true),
    });
    // NEON pairwise add — `Iop_PwAdd{N}x{M}` (no signedness, binary). Output
    // has the same lane shape as the inputs; first half from a, second half
    // from b. Iop_PwAdd32Fx2 is the float variant routed via parse_float to
    // IROp::VFPwAdd, NOT here.
    vec_arms!(op_str; "Iop_PwAdd" => VPwAdd {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON pairwise widening add — `Iop_PwAddL{N}{S/U}x{M}` (unary). Lane
    // width doubles and count halves; total preserved.
    vec_signed_arms!(op_str; "Iop_PwAddL" => VPwAddL {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
        "8Sx16" => (I8, 16, true), "8Ux16" => (I8, 16, false),
        "16Sx8" => (I16, 8, true), "16Ux8" => (I16, 8, false),
        "32Sx4" => (I32, 4, true), "32Ux4" => (I32, 4, false),
        // 64-bit lanes exist only unsigned (libVEX emits Iop_PwAddL64Ux2 with
        // no signed twin); the pair widens to one 128-bit output lane.
        "64Ux2" => (I64, 2, false),
    });

    // NEON pairwise integer min/max — `Iop_PwMin{N}{S/U}x{M}` /
    // `Iop_PwMax{N}{S/U}x{M}`. Same shape as VPwAdd; D-reg-only (no x16/x8/x4
    // emitted by libVEX for these).
    vec_signed_arms!(op_str; "Iop_PwMin" => VPwMin {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_PwMax" => VPwMax {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
    });

    // NEON rounding halving add (a.k.a. rounding-average) —
    // `Iop_Avg{N}{S/U}x{M}`. Binary; output shape matches inputs. Both D-reg
    // (total=64) and Q-reg (total=128) variants exist for 8/16/32-bit lanes.
    // Maps to ARM URHADD/SRHADD; SSE PAVGB/PAVGW (unsigned-only) lifts here too.
    //
    // The D-reg (total=64) half is **unsigned-only** and 8/16-bit-lane only:
    // `vendor/pyvex_ffi.h` declares just `Iop_Avg8Ux8` / `Iop_Avg16Ux4` there —
    // no signed D-reg forms and no `Iop_Avg32{S,U}x2` at all (angr-0jh0j.62;
    // the signed arms contradicted this family's own "unsigned-only" note
    // above). Q-reg (total=128) has both signednesses for 8/16/32-bit lanes,
    // plus 64-bit lanes (`Iop_Avg64{S,U}x2`, no D-reg twin). The 256-bit tier
    // is AVX2 `VPAVGB`/`VPAVGW`, hence unsigned-only and 8/16-bit-lane only.
    //
    // Widening this table to V256 is safe because `VEXOps::vec_rounding_avg`
    // has no `total_width <= 128` concrete fast path to outgrow: it is written
    // lane-at-a-time over `extract`/`extend_into`/`add_into`, and the only
    // >128-bit value it builds comes from `concat_le_elements`, whose
    // `concat_into` already keeps a symbolic `Concat` above 128 bits
    // (bd `invariant-concrete-bv-u128-16-byte-limit`, hazard family 3).
    vec_signed_arms!(op_str; "Iop_Avg" => VAvg {
        "8Ux8"  => (I8, 8, false), "16Ux4" => (I16, 4, false),
        "8Sx16" => (I8, 16, true), "8Ux16" => (I8, 16, false),
        "16Sx8" => (I16, 8, true), "16Ux8" => (I16, 8, false),
        "32Sx4" => (I32, 4, true), "32Ux4" => (I32, 4, false),
        "64Sx2" => (I64, 2, true), "64Ux2" => (I64, 2, false),
        "8Ux32" => (I8, 32, false), "16Ux16" => (I16, 16, false),
    });

    // PPC bit-matrix transpose — `Iop_PwBitMtxXpose64x2` (unary, V128 only).
    // Backs PowerPC `vgbbd`; not a NEON op despite the `Pw` prefix it shares
    // with the pairwise family above. libVEX declares exactly one shape, so
    // this is a plain string match rather than a `vec_arms!` table.
    if op_str == "Iop_PwBitMtxXpose64x2" {
        return Some(IROp::VPwBitMtxXpose);
    }

    // NEON per-byte popcount — `Iop_Cnt8x{8,16}` (unary, 8-bit lanes only).
    // ARM CNT (DDI 0487 C7.2.62).
    match op_str {
        "Iop_Cnt8x8" => return Some(IROp::VCnt { count: 8 }),
        "Iop_Cnt8x16" => return Some(IROp::VCnt { count: 16 }),
        _ => {}
    }

    // SSE byte-mask extract — `Iop_GetMSBs8x{8,16}` (x86 PMOVMSKB, unary).
    // Reduces a vector of N bytes to an N-bit integer of their MSBs.
    match op_str {
        "Iop_GetMSBs8x8" => return Some(IROp::VGetMSBs { count: 8 }),
        "Iop_GetMSBs8x16" => return Some(IROp::VGetMSBs { count: 16 }),
        _ => {}
    }

    // NEON per-lane count leading zeros — `Iop_Clz{N}x{M}` (unary). ARM CLZ
    // (DDI 0487 C7.2.57). D-reg (total=64) and Q-reg (total=128) shapes for
    // 8/16/32-bit lanes, plus the Q-reg-only 64-bit lane shape (libVEX emits
    // Iop_Clz64x2 but no Cls64x2 twin, hence the asymmetry with the Cls table
    // below).
    vec_arms!(op_str; "Iop_Clz" => VClz {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });

    // NEON per-lane count leading sign bits — `Iop_Cls{N}x{M}` (unary). ARM
    // CLS (DDI 0487 C7.2.56). Same shapes as Clz.
    vec_arms!(op_str; "Iop_Cls" => VCls {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    // widening) and `Iop_PolynomialMull8x8` (widening). ARM PMUL / PMULL
    // (DDI 0487 C7.2.281). Only 8-bit lanes are emitted by libVEX.
    match op_str {
        "Iop_PolynomialMul8x8" => {
            return Some(IROp::VPolynomialMul {
                count: 8,
                widen: false,
            });
        }
        "Iop_PolynomialMul8x16" => {
            return Some(IROp::VPolynomialMul {
                count: 16,
                widen: false,
            });
        }
        "Iop_PolynomialMull8x8" => {
            return Some(IROp::VPolynomialMul {
                count: 8,
                widen: true,
            });
        }
        _ => {}
    }

    // Vector multiply: 8-bit is NEON-only (VMUL.I8); 16/32-bit are SSE+NEON,
    // and their AVX2 256-bit tier is `VPMULLW`/`VPMULLD` (no 8-bit or 64-bit
    // packed-multiply opcode exists at that width). Like the VQAdd/VQSub table
    // above, the V256 shapes need no evaluator change:
    // `VEXOps::vec_int_lane_op` gates its u128 concrete fast path on
    // `total_width <= 128` and falls through to the symbolic per-lane path
    // (angr-li4ox.1).
    vec_arms!(op_str; "Iop_Mul" => VMul {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
    });
    // High half of the widening vector multiply — Iop_MulHi{N}{U,S}x{M}
    // (SSE PMULHW/PMULHUW, NEON VMULH, AVX2 256-bit forms). Width-preserving:
    // total = elem * count, so the D-reg (64), Q-reg/SSE (128) and AVX2 (256)
    // tiers are all reachable. libVEX puts S/U AFTER the lane size, so the
    // leftover after stripping "Iop_Mul" ("Hi16Ux4") matches no VMul suffix
    // above and the two invocations are order-independent. The ARM doubling
    // variants (Iop_QDMulHi*, Iop_QRDMulHi*) are a different op and stay
    // unmapped — see the NEON_MULHI entry in opcode_map_tests.rs.
    vec_signed_arms!(op_str; "Iop_MulHi" => VMulHi {
        "8Ux16" => (I8, 16, false), "8Sx16" => (I8, 16, true),
        "16Ux4" => (I16, 4, false), "16Sx4" => (I16, 4, true),
        "16Ux8" => (I16, 8, false), "16Sx8" => (I16, 8, true),
        "16Ux16" => (I16, 16, false), "16Sx16" => (I16, 16, true),
        "32Ux4" => (I32, 4, false), "32Sx4" => (I32, 4, true),
    });
    // Widening vector multiply (angr-ph300.78). Two families, both -> V128:
    //   Iop_Mull{N}{S,U}x{M}      full-lane, (I64,I64)->V128, NEON VMULL
    //   Iop_MullEven{N}{S,U}x{M}  even-lane, (V128,V128)->V128, SSE PMULDQ/PMULUDQ
    // `even` selects which input lanes contribute; `signed` picks sign vs zero
    // extension. libVEX puts S/U AFTER the lane size (e.g. Iop_Mull32Sx2), so
    // these do not collide with the "Iop_Mul" VMul arm above. The "Iop_Mull"
    // prefix cannot false-match "Iop_MullEven*" — the leftover "Even8Ux16" hits
    // no suffix — so the two invocations are order-independent.
    vec_mull_arms!(op_str; "Iop_Mull" => VMull {
        "8Ux8" => (I8, 8, false, false), "8Sx8" => (I8, 8, true, false),
        "16Ux4" => (I16, 4, false, false), "16Sx4" => (I16, 4, true, false),
        "32Ux2" => (I32, 2, false, false), "32Sx2" => (I32, 2, true, false),
    });
    vec_mull_arms!(op_str; "Iop_MullEven" => VMull {
        "8Ux16" => (I8, 16, false, true), "8Sx16" => (I8, 16, true, true),
        "16Ux8" => (I16, 8, false, true), "16Sx8" => (I16, 8, true, true),
        "32Ux4" => (I32, 4, false, true), "32Sx4" => (I32, 4, true, true),
    });
    // Signed doubling saturating widening multiply — Iop_QDMull{N}Sx{M}
    // ((I64,I64)->V128, NEON VQDMULL). Only 16Sx4 / 32Sx2 exist in libVEX;
    // always signed, always full-lane. A dedicated match (not a macro) since
    // there are just two opcodes and no U/even axes to enumerate.
    match op_str {
        "Iop_QDMull16Sx4" => {
            return Some(IROp::VQDMull {
                elem: IRType::I16,
                count: 4,
            });
        }
        "Iop_QDMull32Sx2" => {
            return Some(IROp::VQDMull {
                elem: IRType::I32,
                count: 2,
            });
        }
        _ => {}
    }

    // NEON lane extract / insert — Iop_{Get,Set}Elem{N}x{M}: (vec, idx[, val]).
    vec_arms!(op_str; "Iop_GetElem" => VGetElem {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_SetElem" => VSetElem {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    // NEON broadcast scalar to vector — Iop_Dup{N}x{M}.
    vec_arms!(op_str; "Iop_Dup" => VDup {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON widen lane element width — Iop_Widen{N}{S/U}to{2N}x{M}.
    vec_widen_arms!(op_str; "Iop_Widen" => VWiden {
        "8Sto16x8"  => (I8, 8, true),  "8Uto16x8"  => (I8, 8, false),
        "16Sto32x4" => (I16, 4, true), "16Uto32x4" => (I16, 4, false),
        "32Sto64x2" => (I32, 2, true), "32Uto64x2" => (I32, 2, false),
    });

    // NEON narrow (truncating) — Iop_Narrow{Un,Bin}{N}to{N/2}x{M}.
    vec_narrow_arms!(op_str; "Iop_NarrowUn" => VNarrowUn {
        "16to8x8" => (I16, 8), "32to16x4" => (I32, 4), "64to32x2" => (I64, 2),
    });
    vec_narrow_arms!(op_str; "Iop_NarrowBin" => VNarrowBin {
        "16to8x8" => (I16, 8), "32to16x4" => (I32, 4),
        "16to8x16" => (I16, 16), "32to16x8" => (I32, 8), "64to32x4" => (I64, 4),
    });

    // NEON saturating narrow — Iop_QNarrow{Un,Bin}{N}{S/U}to{N/2}{S/U}x{M}.
    vec_qnarrow_arms!(op_str; "Iop_QNarrowUn" => VQNarrowUn {
        "16Sto8Sx8"  => (I16, 8, true,  true),
        "16Sto8Ux8"  => (I16, 8, true,  false),
        "16Uto8Ux8"  => (I16, 8, false, false),
        "32Sto16Sx4" => (I32, 4, true,  true),
        "32Sto16Ux4" => (I32, 4, true,  false),
        "32Uto16Ux4" => (I32, 4, false, false),
        "64Sto32Sx2" => (I64, 2, true,  true),
        "64Sto32Ux2" => (I64, 2, true,  false),
        "64Uto32Ux2" => (I64, 2, false, false),
    });
    // The binary family's D-reg (x8/x4) block has no `32Sto16Ux4`: libVEX only
    // declares the signed-source/unsigned-dest 32-bit narrow at Q-reg width, as
    // `Iop_QNarrowBin32Sto16Ux8` (mapped below). angr-0jh0j.62 removed the D-reg
    // arm, which was the Q-reg suffix miscopied into this block.
    vec_qnarrow_arms!(op_str; "Iop_QNarrowBin" => VQNarrowBin {
        "16Sto8Sx8"  => (I16, 8,  true,  true),
        "16Sto8Ux8"  => (I16, 8,  true,  false),
        "32Sto16Sx4" => (I32, 4,  true,  true),
        "16Sto8Sx16" => (I16, 16, true,  true),
        "16Sto8Ux16" => (I16, 16, true,  false),
        "16Uto8Ux16" => (I16, 16, false, false),
        "32Sto16Sx8" => (I32, 8,  true,  true),
        "32Sto16Ux8" => (I32, 8,  true,  false),
        "32Uto16Ux8" => (I32, 8,  false, false),
        "64Sto32Sx4" => (I64, 4,  true,  true),
        "64Uto32Ux4" => (I64, 4,  false, false),
    });

    // Vector compare equal / greater-than (signed). The AVX2 256-bit shapes
    // are safe to map because `vec_int_lane_op` (ops/vec_int_lane.rs) gates its
    // u128 concrete fast path on `total_width <= 128` and falls back to the
    // per-lane symbolic path above that — see
    // `invariant-concrete-bv-u128-16-byte-limit` (angr-sqfj8.112).
    vec_arms!(op_str; "Iop_CmpEQ" => VCmpEQ {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    // Signed (S) and unsigned (U) suffixes both exist in libVEX; the U family
    // backs ARM NEON VCGT.U8/U16/U32 and the SSE/AVX unsigned compares
    // (angr-9ke6b.160). libVEX defines no Iop_CmpGT64Ux1 (D-reg), so that
    // suffix is absent by design rather than omission. The AVX2 256-bit forms
    // are **signed-only** — vendor/pyvex_ffi.h declares Iop_CmpGT8Sx32 and
    // friends but no `*Ux32`/`*Ux16`/`*Ux8`/`*Ux4` counterparts, matching the
    // ISA (VPCMPGT* is signed).
    vec_signed_arms!(op_str; "Iop_CmpGT" => VCmpGT {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true), "8Sx32" => (I8, 32, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true), "16Sx16" => (I16, 16, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true), "32Sx8" => (I32, 8, true),
        "64Sx2" => (I64, 2, true), "64Sx4" => (I64, 4, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false),
        "64Ux2" => (I64, 2, false),
    });

    // Vector interleave. Both the 64-bit D-reg NEON shapes (8x8/16x4/32x2) and
    // the 128-bit Q-reg/SSE ones are mapped, so `count` must be carried
    // explicitly — `elem` alone cannot distinguish 8x8 from 8x16, and
    // `result_type()` needs the total width (angr-sqfj8.142).
    // The Iop_Interleave{Even,Odd}Lanes* families are a separate, unmapped op.
    vec_arms!(op_str; "Iop_InterleaveLO" => VInterleaveLO {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_InterleaveHI" => VInterleaveHI {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });

    // Packed integer min/max — signed (S suffix) and unsigned (U suffix).
    // The AVX2 256-bit block covers 8/16/32-bit lanes only: the sole 64-bit-lane
    // 256-bit min/max libVEX declares are the *float* `Iop_Min64Fx4` /
    // `Iop_Max64Fx4` (routed via parse_float), so `64{S,U}x4` integer arms were
    // dead and are gone (angr-0jh0j.62). 64-bit lanes stop at Q-reg width.
    vec_signed_arms!(op_str; "Iop_Min" => VMin {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true), "8Sx32" => (I8, 32, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true), "16Sx16" => (I16, 16, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true), "32Sx8" => (I32, 8, true),
        "64Sx2" => (I64, 2, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false), "8Ux32" => (I8, 32, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false), "16Ux16" => (I16, 16, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false), "32Ux8" => (I32, 8, false),
        "64Ux2" => (I64, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_Max" => VMax {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true), "8Sx32" => (I8, 32, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true), "16Sx16" => (I16, 16, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true), "32Sx8" => (I32, 8, true),
        "64Sx2" => (I64, 2, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false), "8Ux32" => (I8, 32, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false), "16Ux16" => (I16, 16, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false), "32Ux8" => (I32, 8, false),
        "64Ux2" => (I64, 2, false),
    });

    // Packed integer absolute value — Iop_Abs{N}x{M}. D-reg (8x8/16x4/32x2) and
    // Q-reg (8x16/16x8/32x4/64x2) only: this VEX pin declares no 256-bit `Abs`
    // family at all, integer or float, so the `8x32`/`16x16`/`32x8`/`64x4` arms
    // were dead (angr-0jh0j.62).
    vec_arms!(op_str; "Iop_Abs" => VAbs {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });

    // NEON integer reciprocal estimate — Iop_RecipEst32Ux{2,4} (URECPE) and
    // Iop_RSqrtEst32Ux{2,4} (URSQRTE). Fresh-symbolic per lane via VIRecipEst /
    // VIRSqrtEst. Implemented in angr-tukg.5.
    if let Some(rest) = op_str.strip_prefix("Iop_RecipEst") {
        match rest {
            "32Ux2" => return Some(IROp::VIRecipEst { count: 2 }),
            "32Ux4" => return Some(IROp::VIRecipEst { count: 4 }),
            _ => {}
        }
    }
    if let Some(rest) = op_str.strip_prefix("Iop_RSqrtEst") {
        match rest {
            "32Ux2" => return Some(IROp::VIRSqrtEst { count: 2 }),
            "32Ux4" => return Some(IROp::VIRSqrtEst { count: 4 }),
            _ => {}
        }
    }

    // V128/V256 to/from conversions.
    cast_arms!(op_str; "Iop_" => Truncate    { "V128to64"   => (V128, I64) });
    cast_arms!(op_str; "Iop_" => ZeroExtend  { "64UtoV128"  => (I64, V128), "32UtoV128" => (I32, V128) });
    // NOTE: no "SetV128lo64" arm here — parse_float claims it first (returns
    // IROp::SetV128lo64, a binop preserving the upper 64 bits), so a Reinterpret
    // arm would be both dead and semantically wrong. See angr-ph300.62.

    match op_str {
        "Iop_V128HIto64" => Some(IROp::Extract {
            from: IRType::V128,
            to: IRType::I64,
            low_bit: 64,
        }),
        _ => None,
    }
}

/// Parse NEON byte/halfword/word/bit reversal opcodes —
/// `Iop_Reverse{sub_width}sIn{elem_width}_x{count}`. Returns
/// `IROp::VReverse { sub_width, elem, count }` which dispatches through
/// `VEXOps::unop` to `vec_reverse`. Implemented in angr-tukg.4.
///
/// ARM ISA mapping (DDI 0487 C7.2.288 / C7.2.297-300):
///   - `Reverse1sIn8_*`   → RBIT (bit reverse within each byte)
///   - `Reverse8sIn16_*`  → REV16 (byte swap within halfwords)
///   - `Reverse8sIn32_*`  → REV32 (byte swap within words)
///   - `Reverse8sIn64_*`  → REV64 (byte swap within doublewords)
///   - `Reverse16sIn32_*` → REV32 (halfword swap within words)
///   - `Reverse16sIn64_*` → REV64 (halfword swap within doublewords)
///   - `Reverse32sIn64_*` → REV64 (word swap within doublewords)
pub(super) fn parse_vreverse(op_str: &str) -> Option<IROp> {
    let (sub_width, elem, count) = match op_str {
        // Byte reversal within 16-bit halfwords (REV16).
        "Iop_Reverse8sIn16_x4" => (8u8, IRType::I16, 4u8),
        "Iop_Reverse8sIn16_x8" => (8, IRType::I16, 8),
        // Byte reversal within 32-bit words (REV32 / x86 BSWAP-like).
        "Iop_Reverse8sIn32_x2" => (8, IRType::I32, 2),
        "Iop_Reverse8sIn32_x4" => (8, IRType::I32, 4),
        // Byte reversal within 64-bit doublewords (REV64).
        "Iop_Reverse8sIn64_x1" => (8, IRType::I64, 1),
        "Iop_Reverse8sIn64_x2" => (8, IRType::I64, 2),
        // Halfword reversal within 32-bit words.
        "Iop_Reverse16sIn32_x2" => (16, IRType::I32, 2),
        "Iop_Reverse16sIn32_x4" => (16, IRType::I32, 4),
        // Halfword reversal within 64-bit doublewords.
        "Iop_Reverse16sIn64_x1" => (16, IRType::I64, 1),
        "Iop_Reverse16sIn64_x2" => (16, IRType::I64, 2),
        // Word swap within 64-bit doublewords.
        "Iop_Reverse32sIn64_x1" => (32, IRType::I64, 1),
        "Iop_Reverse32sIn64_x2" => (32, IRType::I64, 2),
        // Bit reversal within each byte (RBIT).
        "Iop_Reverse1sIn8_x8" => (1, IRType::I8, 8),
        "Iop_Reverse1sIn8_x16" => (1, IRType::I8, 16),
        _ => return None,
    };
    Some(IROp::VReverse {
        sub_width,
        elem,
        count,
    })
}

/// Parse ARM/AArch64 NEON SIMD opcodes that have been claimed but not yet
/// implemented. Hits here route through `IROp::NeonUnimplemented(name)` so
/// dispatch in `VEXOps::unop` / `binop` / `qop` returns
/// `Err(OpError::UnsupportedNeon { name })` carrying the original opcode name
/// instead of silently returning a fresh-symbolic value.
///
/// Implementations are added one-at-a-time in angr-bkcs.2 by:
///   1. Removing the opcode's entry from this function.
///   2. Adding it to `parse_vector` (or `parse_float`) with a real IROp variant.
///   3. Wiring that variant into `VEXOps::unop` / `binop`.
///
/// Scope: D-register (Ity_I64) and Q-register (Ity_V128) opcodes that pyvex
/// emits for ARM/AArch64 NEON and that this engine currently has no handler
/// for. Opcodes already handled by `parse_vector` (e.g. `Iop_Add8x8` ->
/// `VAdd`) are deliberately excluded so we do not regress existing coverage.
///
/// # Historical coverage
///
/// This function currently claims nothing — every NEON family that once
/// routed here has graduated to a real handler, the last being
/// `Iop_PwAdd32Fx2` -> `IROp::VFPwAdd` in angr-cudgw.6. The record below maps
/// each family to the bead that implemented it and the parse fn it now routes
/// through, so a future NEON gap can be checked against what was already
/// covered:
///
/// - `Iop_GetElem*` / `Iop_SetElem*` (lane extract/insert) — angr-bkcs.2, via
///   `parse_vector` to `IROp::VGetElem` / `VSetElem`.
/// - `Iop_Dup*` / `Iop_Widen*` / `Iop_Narrow{Bin,Un}*` / `Iop_QNarrow{Bin,Un}*`
///   — angr-hzs0, via `parse_vector` to `IROp::VDup` / `VWiden` /
///   `VNarrow{Un,Bin}` / `VQNarrow{Un,Bin}`.
/// - FP `RecipEst` / `RecipStep` / `RSqrtEst` / `RSqrtStep` (`{32,64}{F0,Fx}*`)
///   — angr-iyon, via `parse_float` to `IROp::VFRecipEst{,S}` / `VFRecipStep` /
///   `VFRSqrtEst{,S}` / `VFRSqrtStep`.
/// - `Iop_RecipEst32Ux{2,4}` (URECPE) and `Iop_RSqrtEst32Ux{2,4}` (URSQRTE) —
///   angr-tukg.5, via `parse_vector` to `IROp::VIRecipEst` / `VIRSqrtEst`
///   (fresh-symbolic per lane).
/// - `Iop_QAdd{N}{S/U}x{M}` / `Iop_QSub{N}{S/U}x{M}` (saturating integer
///   add/sub) — angr-tukg.1, via `parse_vector` to `IROp::VQAdd` / `VQSub`.
/// - `Iop_Avg{N}{S/U}x{M}` (rounding halving add) — angr-tukg.3, via
///   `parse_vector` to `IROp::VAvg`.
/// - `Iop_Reverse{N}sIn{M}_x{K}` (byte/halfword/word/bit reversal within lane)
///   — angr-tukg.4, via `parse_vreverse` to `IROp::VReverse`.
/// - Integer `Iop_PwAdd{N}x{M}`, `Iop_PwAddL{N}{S/U}x{M}`,
///   `Iop_PwMin{N}{S/U}x{M}`, `Iop_PwMax{N}{S/U}x{M}` — angr-tukg.2, via
///   `parse_vector` to `IROp::VPwAdd` / `VPwAddL` / `VPwMin` / `VPwMax`; the FP
///   pairwise `Iop_PwAdd32Fx2` — angr-cudgw.6, via `parse_float` to
///   `IROp::VFPwAdd`; the FP pairwise `Iop_PwMax32Fx{2,4}` /
///   `Iop_PwMin32Fx{2,4}` — angr-sqfj8.116, via `parse_float` to
///   `IROp::VFPwMax` / `VFPwMin`. The one non-NEON member of the family,
///   `Iop_PwBitMtxXpose64x2` (PPC vgbbd bit-matrix transpose) — angr-sqfj8.143,
///   via `parse_vector` to `IROp::VPwBitMtxXpose`. No `Pw*` op is unmapped.
/// - `Iop_PolynomialMul8x{8,16}` / `Iop_PolynomialMull8x8` (GF(2) carry-less
///   multiply) — angr-tukg.6, via `parse_vector` to `IROp::VPolynomialMul`.
/// - `Iop_Cnt8x{8,16}` (per-byte popcount), `Iop_Clz{N}x{M}` and
///   `Iop_Cls{N}x{M}` (per-lane count-leading-zeros / count-leading-sign-bits)
///   — angr-tukg.6, via `parse_vector` to `IROp::VCnt` / `VClz` / `VCls`.
/// - Vector shift by *vector* (`Shl`/`Shr`/`Sar`/`Sal{N}x{M}`) — angr-tukg.7,
///   via `parse_vector` to `IROp::VShl` / `VShr` / `VSar` (`Sal` -> `VShl`).
/// - `Iop_QShl{N}x{M}` / `Iop_QSal{N}x{M}` (saturating shift-left by vector) —
///   angr-tukg.8, via `parse_vector` to `IROp::VQShlSat`. `QShlN`
///   (shift-by-immediate) is still unimplemented.
pub(super) fn parse_neon_unimplemented(op_str: &str) -> Option<IROp> {
    // The `NeonUnimplemented` sentinel and this fn are kept as the scaffold
    // point for the next NEON op — add a match on `op_str` here returning
    // `Some(IROp::NeonUnimplemented("Iop_Foo"))`, and move it to the historical
    // coverage list above once it graduates to a real handler.
    let _ = op_str;
    None
}
