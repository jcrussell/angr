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
