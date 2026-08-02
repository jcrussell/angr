//! Tests for the native libVEX lifter (feature `libvex-ffi`).
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

/// The four arches `ffi_vex_arch` still has no archinfo mapping for must be
/// refused up front so the caller falls back to the pyvex-callback path. This
/// is deliberately *not* "everything except AMD64" — see
/// `test_supported_arches_are_not_rejected`.
#[test]
fn test_unsupported_arches_rejected() {
    let lifter = NativeLibVEXLifter::new();
    for arch in [VexArch::X86, VexArch::PPC32, VexArch::PPC64, VexArch::S390X] {
        let err = lifter
            .lift(&[0x90], 0x400000, arch)
            .expect_err("arch without an ffi_vex_arch mapping must be rejected");
        assert!(
            matches!(err, LiftError::InvalidArch(_)),
            "{arch:?} should be InvalidArch, got {err:?}"
        );
    }
}

/// Companion to the above: since angr-qwyti.20 the native lifter accepts ARM,
/// ARM64, MIPS32 and MIPS64 alongside AMD64 (the corpus parity gate in
/// `libvex_corpus_tests.rs` replays real blocks for all five). Pin the arch
/// gate itself rather than a lift, so the assertion does not depend on
/// hand-assembling valid guest bytes per arch.
#[test]
fn test_supported_arches_are_not_rejected() {
    for arch in [
        VexArch::AMD64,
        VexArch::ARM,
        VexArch::ARM64,
        VexArch::MIPS32,
        VexArch::MIPS64,
    ] {
        assert!(
            ffi_vex_arch(arch).is_some(),
            "{arch:?} must map to an ffi::VexArch guest tag"
        );
    }
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

/// Every `ILGop_*` tag the vendored cffi cdef defines today.
///
/// `loadg_op` matches these by name; anything else falls through to
/// `IRLoadGOp::Unknown`, which makes the interpreter drop the widening and
/// silently load the wrong value rather than fail. Keep in sync with
/// `test_loadg_op_handles_every_vendored_variant` below.
const VENDORED_ILGOP_TAGS: &[&str] = &[
    "ILGop_INVALID",
    "ILGop_IdentV128",
    "ILGop_Ident64",
    "ILGop_Ident32",
    "ILGop_16Uto32",
    "ILGop_16Sto32",
    "ILGop_8Uto32",
    "ILGop_8Sto32",
];

/// Tripwire for a VEX pin bump that grows `IRLoadGOp`.
///
/// The two lifting paths diverge here on purpose: `pyvex_bridge::parse_loadg_op`
/// (JSON path) also accepts the `ILGop_{16,32}{U,S}to64` widening forms
/// defensively, but `loadg_op` (native FFI path) cannot — it matches bindgen
/// constants, and bindgen only emits what the vendored header declares, which
/// today is the eight tags above and no `*to64` form. So the native path has no
/// way to pre-handle a variant that does not exist yet; the next best thing is
/// to fail loudly the moment one appears. If this test breaks after
/// `tools/regen-pyvex-ffi-header.py`, add the new tag to `loadg_op`, to
/// `parse_loadg_op`, and to `VENDORED_ILGOP_TAGS`.
#[test]
fn test_vendored_header_ilgop_variant_set_is_unchanged() {
    let header = include_str!("../../vendor/pyvex_ffi.h");

    let mut found: Vec<&str> = Vec::new();
    let mut rest = header;
    while let Some(pos) = rest.find("ILGop_") {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(tail.len());
        found.push(&tail[..end]);
        rest = &tail[end..];
    }
    found.sort_unstable();
    found.dedup();

    let mut expected: Vec<&str> = VENDORED_ILGOP_TAGS.to_vec();
    expected.sort_unstable();

    assert_eq!(
        found, expected,
        "vendored pyvex_ffi.h ILGop_* set changed; teach loadg_op (and \
         pyvex_bridge::parse_loadg_op) the new tags before updating this list"
    );
}

/// Pins the native FFI path's `IRLoadGOp` mapping, so a reordered enum (the
/// tags are implicitly numbered from `ILGop_INVALID=0x1D00`) or a dropped arm
/// shows up as a failure rather than a silent `Unknown`.
#[test]
fn test_loadg_op_handles_every_vendored_variant() {
    use crate::vex::ir::IRLoadGOp;

    assert_eq!(
        loadg_op(ffi::IRLoadGOp::ILGop_IdentV128),
        IRLoadGOp::Identity
    );
    assert_eq!(loadg_op(ffi::IRLoadGOp::ILGop_Ident64), IRLoadGOp::Identity);
    assert_eq!(loadg_op(ffi::IRLoadGOp::ILGop_Ident32), IRLoadGOp::Identity);
    assert_eq!(
        loadg_op(ffi::IRLoadGOp::ILGop_8Uto32),
        IRLoadGOp::WidenZ { src_bits: 8 }
    );
    assert_eq!(
        loadg_op(ffi::IRLoadGOp::ILGop_8Sto32),
        IRLoadGOp::WidenS { src_bits: 8 }
    );
    assert_eq!(
        loadg_op(ffi::IRLoadGOp::ILGop_16Uto32),
        IRLoadGOp::WidenZ { src_bits: 16 }
    );
    assert_eq!(
        loadg_op(ffi::IRLoadGOp::ILGop_16Sto32),
        IRLoadGOp::WidenS { src_bits: 16 }
    );
    // INVALID is the one vendored tag that legitimately has no mapping.
    assert_eq!(loadg_op(ffi::IRLoadGOp::ILGop_INVALID), IRLoadGOp::Unknown);
}
