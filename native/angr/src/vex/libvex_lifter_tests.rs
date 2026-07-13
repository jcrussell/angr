//! Tests for the native libVEX lifter (feature `libvex-ffi`, AMD64 only).
//!
//! These exercise the real `libpyvex.so` `vex_lift` shim, so they only build
//! and run under `cargo test --features libvex-ffi`. The exhaustive
//! native-vs-pyvex structural parity gate lives in the .4 corpus harness; here
//! we just assert the marshaller produces a well-formed `IRSB` for a handful of
//! representative AMD64 blocks.

use super::*;
use crate::vex::{IRExpr, IRSB, IRStmt, JumpKind, LiftError, VEXLifter, VexArch};

fn lift(bytes: &[u8], addr: u64) -> IRSB {
    let lifter = NativeLibVEXLifter::new();
    lifter
        .lift(bytes, addr, VexArch::AMD64)
        .expect("native libVEX lift should succeed")
}

#[test]
fn test_lift_add_rax_rbx() {
    // 48 01 d8  =  add rax, rbx
    let irsb = lift(&[0x48, 0x01, 0xd8], 0x400000);

    assert_eq!(irsb.arch, VexArch::AMD64);
    assert_eq!(irsb.addr, 0x400000);
    assert!(!irsb.statements.is_empty(), "block has statements");

    // First statement of any decoded block is an IMark at the block address.
    match &irsb.statements[0] {
        IRStmt::IMark { addr, len, .. } => {
            assert_eq!(*addr, 0x400000);
            assert_eq!(*len, 3, "add rax,rbx is 3 bytes");
        }
        other => panic!("expected leading IMark, got {other:?}"),
    }

    // Temporaries are typed, so tyenv must be populated.
    assert!(!irsb.tyenv.types.is_empty(), "tyenv populated");

    // The block flows through (Boring) to a constant fallthrough address.
    assert_eq!(irsb.jumpkind, JumpKind::Boring);
    assert!(matches!(irsb.next, IRExpr::Const(_)));
}

#[test]
fn test_lift_ret_jumpkind() {
    // c3  =  ret
    let irsb = lift(&[0xc3], 0x401000);
    assert_eq!(irsb.jumpkind, JumpKind::Ret);
}

#[test]
fn test_lift_nop() {
    // 90  =  nop  (decodes to a single-instruction Boring block)
    let irsb = lift(&[0x90], 0x402000);
    assert_eq!(irsb.jumpkind, JumpKind::Boring);
    assert!(matches!(irsb.statements[0], IRStmt::IMark { .. }));
}

#[test]
fn test_lift_conditional_jump_has_exit() {
    // 48 85 c0  test rax,rax ; 74 02  jz +2 ; 90 nop ; 90 nop
    let bytes = [0x48, 0x85, 0xc0, 0x74, 0x02, 0x90, 0x90];
    let irsb = lift(&bytes, 0x403000);
    let has_exit = irsb
        .statements
        .iter()
        .any(|s| matches!(s, IRStmt::Exit { .. }));
    assert!(has_exit, "conditional block should contain an Exit stmt");
}

#[test]
fn test_non_amd64_rejected() {
    let lifter = NativeLibVEXLifter::new();
    let err = lifter
        .lift(&[0x90], 0x400000, VexArch::X86)
        .expect_err("non-AMD64 must be rejected in Stage-1");
    assert!(matches!(err, LiftError::InvalidArch(_)));
}

/// libVEX stores restricted-vector consts as one bit per byte lane; pyvex
/// expands them, so the marshaller must too. The AMD64 corpus only ever
/// produces the all-zero and all-ones patterns, so cover a mixed one here.
#[test]
fn test_expand_v128_pattern() {
    assert_eq!(expand_v128(0x0000), 0);
    assert_eq!(expand_v128(0xffff), u128::MAX);
    assert_eq!(expand_v128(0x0001), 0xff);
    assert_eq!(expand_v128(0x8000), 0xff << 120);
    assert_eq!(expand_v128(0x0003), 0xffff);
}

#[test]
fn test_expand_v256_pattern() {
    assert_eq!(expand_v256(0x0000_0000), [0; 4]);
    assert_eq!(expand_v256(0xffff_ffff), [u64::MAX; 4]);
    assert_eq!(expand_v256(0x0000_0001), [0xff, 0, 0, 0]);
    // Bit 8 is the first lane of the second limb.
    assert_eq!(expand_v256(0x0000_0100), [0, 0xff, 0, 0]);
    assert_eq!(expand_v256(0x8000_0000), [0, 0, 0, 0xff << 56]);
}
