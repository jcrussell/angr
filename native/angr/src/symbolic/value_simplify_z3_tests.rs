//! The solver-facing half of `bv/value_simplify_tests.rs`: these tests lower or solve, so
//! they stay in `symbolic` (angr-mx00p.116.7).

use angr_bv::*;
use crate::symbolic::RustBV;
// Expression-simplification tests: algebraic identities, the shift/mul →
// Concat rewrites, and the pre-Z3 `Extract` rewrite pass (angr-p8cz).

use crate::symbolic::ToZ3;
use crate::symbolic::SymContext;
#[test]
fn test_reverse_z3_emission_32bit_leaf() {
    // Reverse(x) over a symbolic leaf should produce byte-reversed value
    // through the Z3 emission path.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let rev = x.reverse(); // Expression { Reverse, [x] }
    // Pin x = 0x11223344; expect Reverse(x) = 0x44332211
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0x44332211));
}
#[test]
fn test_reverse_z3_emission_64bit_leaf() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 64);
    let rev = x.reverse();
    let pinned = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0xEFCDAB8967452301));
}
#[test]
fn test_reverse_z3_emission_16bit_leaf() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 16);
    let rev = x.reverse();
    let pinned = x.eq(&RustBV::concrete(0xAABB, 16));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0xBBAA));
}
#[test]
fn test_reverse_z3_emission_concat_of_bytes() {
    // Reverse(Concat(b0, b1, ..., b7)) with independent byte BVSes.
    // Memory loads in angr typically produce this shape; the result should
    // be Concat(b7, b6, ..., b0) — the byte-reversed value.
    let ctx = SymContext::new_mock();
    let bytes: Vec<RustBV> = (0..8u32)
        .map(|i| RustBV::symbol(format!("b{i}"), 8))
        .collect();
    // Build claripy-style Concat(b0, b1, ..., b7) with b0 as high.
    let mut concat = bytes[0].clone();
    for b in &bytes[1..] {
        concat = concat.concat(b);
    }
    let rev = concat.reverse();
    // Pin each byte to a distinct value and verify byte-reversed result.
    let vals: [u64; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    for (b, v) in bytes.iter().zip(vals.iter()) {
        let pin = b.eq(&RustBV::concrete(*v as u128, 8));
        ctx.add_constraint(pin.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    }
    // concat = 0x1122334455667788; reverse → 0x8877665544332211
    assert_eq!(ctx.eval(&rev), Some(0x8877665544332211));
}
// --- Pre-Z3 Extract rewrite pass (angr-p8cz) ---


#[test]
fn test_pre_z3_extract_over_reverse_byte_aligned() {
    // Extract a single byte from Reverse(x). With the rewrite pass the
    // Reverse should be eliminated entirely from the Z3 AST.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let rev = x.reverse();
    // Bypass extract_into via the raw-node helper: pretend this Extract was
    // built by truncate_into after the Reverse existed.
    // Reverse(x) byte 0 ([7:0]) is byte 3 ([31:24]) of x.
    let lo_byte = raw_extract_node(rev, 7, 0);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&lo_byte), Some(0x11));
}
#[test]
fn test_pre_z3_extract_over_reverse_crossing_byte() {
    // Two-byte extract across a byte boundary on Reverse — still
    // byte-aligned (high % 8 == 7, low % 8 == 0), should be rewritten.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let rev = x.reverse();
    // Reverse(x)[15:0] = bytes 0,1 of reverse = bytes 3,2 of x = top half reversed.
    let lower16 = raw_extract_node(rev, 15, 0);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    // Reverse(0x11223344) = 0x44332211; low 16 bits = 0x2211
    assert_eq!(ctx.eval(&lower16), Some(0x2211));
}
#[test]
fn test_pre_z3_extract_over_concat_within_low() {
    // Extract entirely within the low (right) part of a Concat — should
    // delegate to that operand alone.
    let ctx = SymContext::new_mock();
    let a = RustBV::symbol("a", 16);
    let b = RustBV::symbol("b", 16);
    let cat = a.concat(&b); // a:high, b:low
    // Extract [15:0] of cat = entirely within b.
    let lo = raw_extract_node(cat, 15, 0);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0xAAAA, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0xBBBB, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0xBBBB));
}
#[test]
fn test_pre_z3_extract_over_concat_within_high() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbol("a", 16);
    let b = RustBV::symbol("b", 16);
    let cat = a.concat(&b);
    // Extract [31:16] of cat = entirely within a.
    let hi = raw_extract_node(cat, 31, 16);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0xAAAA, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0xBBBB, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&hi), Some(0xAAAA));
}
#[test]
fn test_pre_z3_extract_over_concat_crossing() {
    // Crosses the a/b boundary in the middle — distribute Extract to both.
    let ctx = SymContext::new_mock();
    let a = RustBV::symbol("a", 16);
    let b = RustBV::symbol("b", 16);
    let cat = a.concat(&b);
    // Extract [23:8] = high 8 bits of b ([15:8]) concat with low 8 bits of a ([7:0]).
    let mid = raw_extract_node(cat, 23, 8);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0x1234, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0x5678, 16))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    // cat = 0x12345678; extract [23:8] = 0x3456
    assert_eq!(ctx.eval(&mid), Some(0x3456));
}
#[test]
fn test_pre_z3_extract_over_extract_fused() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 64);
    let mid = x.extract(47, 16); // width 32, = bits 16..=47 of x
    // Build outer Extract WITHOUT going through extract_into.
    let outer = raw_extract_node(mid, 23, 8); // mid[23:8] = bits [39:24] of x
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x0011_2233_4455_6677, 64))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    // x bytes (LSB→MSB): 0x77 0x66 0x55 0x44 0x33 0x22 0x11 0x00.
    // bits [39:24] of x = byte indices 3..=4 = 0x33:0x44 (high:low) = 0x3344.
    assert_eq!(ctx.eval(&outer), Some(0x3344));
}
#[test]
fn test_pre_z3_extract_over_zero_ext_low_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 8);
    let zx = x.zero_extend(32); // width 32
    // Extract [7:0] entirely within original x.
    let lo = raw_extract_node(zx, 7, 0);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0xAB, 8))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0xAB));
}
#[test]
fn test_pre_z3_extract_over_zero_ext_extended_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 8);
    let zx = x.zero_extend(32);
    // Extract [31:24] is entirely in the zero-extended region.
    let top = raw_extract_node(zx, 31, 24);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0xFF, 8))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&top), Some(0));
}
/// Regression for the latent bug in extract_into's Rule 3: multi-byte
/// byte-aligned Extract of Reverse(x) used to drop the byte shuffle
/// (returned plain Extract from x), giving the wrong byte order in the
/// constraint tree. Now it returns Reverse(Extract(...)), preserving
/// semantics through every consumer (Z3 round-trip, downstream rewrites).
#[test]
fn test_extract_over_reverse_multibyte_construction() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let rev = x.reverse();
    // Use the normal extract (which now goes through the fixed Rule 3).
    let lower16 = rev.extract(15, 0);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    // Reverse(0x11223344) = 0x44332211; low 16 = 0x2211.
    assert_eq!(ctx.eval(&lower16), Some(0x2211));
}
#[test]
fn test_extract_over_reverse_single_byte_construction() {
    // Single-byte case: rule reduces to plain Extract from x.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let rev = x.reverse();
    // Byte 0 of reverse = byte 3 of x.
    let b0 = rev.extract(7, 0);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x11223344, 32))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&b0), Some(0x11));
}
#[test]
fn test_pre_z3_extract_over_sign_ext_low_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 8);
    let sx = x.sign_extend(32);
    let lo = raw_extract_node(sx, 7, 0);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x80, 8))
            .to_z3_ast().unwrap()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0x80));
}

/// A bare `Extract` node, with none of `extract`'s rewrites (the same helper
/// as `bv/value_tests_support.rs`, which this side cannot import).
fn raw_extract_node(inner: RustBV, high: u32, low: u32) -> RustBV {
    RustBV::from_op(high - low + 1, BVOp::Extract(high, low), vec![inner])
}
