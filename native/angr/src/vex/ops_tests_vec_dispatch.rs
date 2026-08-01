// angr-ph300.63: guard the vector-IROp triple-list drift.
//
// The SIMD integer ops are enumerated in three hand-maintained match lists
// with no compile-time coupling between them:
//   1. `iropclass()`            (ops.rs) — telemetry family classification
//   2. the `binop()` router     (ops.rs) — routes to `binop_vec_int`
//   3. `binop_vec_int()`        (ops.rs) — the actual per-op bindings
//
// A variant added to the router but not to `iropclass`'s `Vec` arm skews the
// `vex_op_vec` counter; a variant dropped from either the router or
// `binop_vec_int` silently falls through to `binop_misc`'s / `binop_vec_int`'s
// `_ => Err(NotBinary)` arm and defers to Python at runtime — caught only when
// that specific op is exercised (this is the same class that produced the
// phantom VMulLo arm). These two tests pin the coupling:
//
//   * `test_vec_int_binops_iropclass_is_vec` — every canonical vector-integer
//     binop classifies as `VexOpFamily::Vec` (Concat is the one router-dispatched
//     exception, classified `Ext`; asserted explicitly).
//   * `test_vec_int_binops_dispatch_reachable` — every op whose operands are
//     two V128 vectors (or V128 + a small scalar) reaches a real handler rather
//     than the `NotBinary` fallthrough. The width-heterogeneous ops (VMull,
//     VNarrowBin, VQNarrowBin) have dedicated dispatch coverage in
//     ops_tests_vec_permute_mul.rs / ops_tests_vec_saturate.rs / ops_tests_vec_lane.rs.

use super::*;

/// Canonical list of the vector-integer binops the `binop()` router forwards to
/// `binop_vec_int` (ops.rs), excluding `Concat` (classified `Ext`, asserted
/// separately). Adding a new `V*` integer binop should add it here so both
/// tests below cover it.
fn vec_int_binops() -> Vec<IROp> {
    use IRType::*;
    vec![
        IROp::VAnd(I128),
        IROp::VOr(I128),
        IROp::VXor(I128),
        IROp::VAdd {
            elem: I32,
            count: 4,
        },
        IROp::VSub {
            elem: I32,
            count: 4,
        },
        IROp::VMul {
            elem: I32,
            count: 4,
        },
        IROp::VMull {
            elem: I32,
            count: 4,
            signed: false,
            even: true,
        },
        IROp::VQAdd {
            elem: I16,
            count: 8,
            signed: true,
        },
        IROp::VQSub {
            elem: I16,
            count: 8,
            signed: true,
        },
        IROp::VQShlSat {
            elem: I16,
            count: 8,
            signed: true,
        },
        IROp::VPwAdd {
            elem: I16,
            count: 8,
        },
        IROp::VPwMin {
            elem: I16,
            count: 8,
            signed: true,
        },
        IROp::VPwMax {
            elem: I16,
            count: 8,
            signed: true,
        },
        IROp::VAvg {
            elem: I8,
            count: 16,
            signed: false,
        },
        IROp::VPolynomialMul {
            count: 16,
            widen: false,
        },
        IROp::VCmpEQ {
            elem: I32,
            count: 4,
        },
        IROp::VCmpGT {
            elem: I32,
            count: 4,
            signed: true,
        },
        IROp::VGetElem {
            elem: I32,
            count: 4,
        },
        IROp::VNarrowBin {
            from: I32,
            count: 8,
        },
        IROp::VQNarrowBin {
            from: I32,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        IROp::VInterleaveLO { elem: I8 },
        IROp::VInterleaveHI { elem: I8 },
        IROp::VShlN {
            elem: I32,
            count: 4,
        },
        IROp::VShrN {
            elem: I32,
            count: 4,
        },
        IROp::VSarN {
            elem: I32,
            count: 4,
        },
        IROp::VShl {
            elem: I32,
            count: 4,
        },
        IROp::VShr {
            elem: I32,
            count: 4,
        },
        IROp::VSar {
            elem: I32,
            count: 4,
        },
        IROp::VMin {
            elem: I32,
            count: 4,
            signed: false,
        },
        IROp::VMax {
            elem: I32,
            count: 4,
            signed: false,
        },
    ]
}

#[test]
fn test_vec_int_binops_iropclass_is_vec() {
    for op in vec_int_binops() {
        assert!(
            iropclass(&op) == VexOpFamily::Vec,
            "vector-integer op {op:?} is not classified as VexOpFamily::Vec — \
             iropclass() and the binop() router have drifted",
        );
    }
    // Concat is dispatched to binop_vec_int by the router but is a scalar
    // widen/concat, deliberately classified Ext (not Vec).
    assert!(
        iropclass(&IROp::Concat { ty: IRType::I128 }) == VexOpFamily::Ext,
        "Concat should classify as Ext",
    );
}

#[test]
fn test_vec_int_binops_dispatch_reachable() {
    let ctx = SymContext::new_mock();

    // (op, left_width, right_width) for the ops whose correct operands are two
    // V128 vectors or V128 + a small scalar. Zero operands suffice: we only
    // assert the op does NOT hit the NotBinary fallthrough, i.e. the router and
    // binop_vec_int agree it is a real vector binop.
    let cases: &[(IROp, u32, u32)] = &[
        (IROp::VAnd(IRType::I128), 128, 128),
        (IROp::VOr(IRType::I128), 128, 128),
        (IROp::VXor(IRType::I128), 128, 128),
        (
            IROp::VAdd {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VSub {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VMul {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VQAdd {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            128,
            128,
        ),
        (
            IROp::VQSub {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            128,
            128,
        ),
        (
            IROp::VQShlSat {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            128,
            128,
        ),
        (
            IROp::VPwAdd {
                elem: IRType::I16,
                count: 8,
            },
            128,
            128,
        ),
        (
            IROp::VPwMin {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            128,
            128,
        ),
        (
            IROp::VPwMax {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            128,
            128,
        ),
        (
            IROp::VAvg {
                elem: IRType::I8,
                count: 16,
                signed: false,
            },
            128,
            128,
        ),
        (
            IROp::VPolynomialMul {
                count: 16,
                widen: false,
            },
            128,
            128,
        ),
        (
            IROp::VCmpEQ {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VCmpGT {
                elem: IRType::I32,
                count: 4,
                signed: true,
            },
            128,
            128,
        ),
        (IROp::VInterleaveLO { elem: IRType::I8 }, 128, 128),
        (IROp::VInterleaveHI { elem: IRType::I8 }, 128, 128),
        (
            IROp::VShl {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VShr {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VSar {
                elem: IRType::I32,
                count: 4,
            },
            128,
            128,
        ),
        (
            IROp::VMin {
                elem: IRType::I32,
                count: 4,
                signed: false,
            },
            128,
            128,
        ),
        (
            IROp::VMax {
                elem: IRType::I32,
                count: 4,
                signed: false,
            },
            128,
            128,
        ),
        // vec + immediate/index: second operand is an 8-bit scalar.
        (
            IROp::VShlN {
                elem: IRType::I32,
                count: 4,
            },
            128,
            8,
        ),
        (
            IROp::VShrN {
                elem: IRType::I32,
                count: 4,
            },
            128,
            8,
        ),
        (
            IROp::VSarN {
                elem: IRType::I32,
                count: 4,
            },
            128,
            8,
        ),
        (
            IROp::VGetElem {
                elem: IRType::I32,
                count: 4,
            },
            128,
            8,
        ),
        // Concat (scalar widen, router-dispatched to binop_vec_int).
        (IROp::Concat { ty: IRType::I128 }, 64, 64),
    ];

    for (op, lw, rw) in cases {
        let left = RustBV::concrete(0, *lw);
        let right = RustBV::concrete(0, *rw);
        let result = VEXOps::binop(*op, left, right, &ctx);
        assert!(
            !matches!(result, Err(OpError::NotBinary(_))),
            "vector op {op:?} fell through to the NotBinary arm — the binop() \
             router and binop_vec_int have drifted (got {result:?})",
        );
    }
}
