//! Unit tests for [`super`] (arch/mod.rs).
//! Split out of mod.rs per `rust-mod-tests-sibling-extraction` (cfg(test)-only reorg).

use super::*;
use crate::symbolic::SymContext;

#[test]
fn test_arch_from_vex_supported() {
    // The six supported arches resolve to a singleton with the matching VexArch.
    for vex in [
        VexArch::X86,
        VexArch::AMD64,
        VexArch::ARM,
        VexArch::ARM64,
        VexArch::MIPS32,
        VexArch::MIPS64,
    ] {
        assert_eq!(arch_from_vex(vex).vex_arch(), vex);
    }
}

#[test]
#[should_panic(expected = "Rust engine does not support")]
fn test_arch_from_vex_ppc32_panics() {
    let _ = arch_from_vex(VexArch::PPC32);
}

#[test]
#[should_panic(expected = "Rust engine does not support")]
fn test_arch_from_vex_ppc64_panics() {
    let _ = arch_from_vex(VexArch::PPC64);
}

#[test]
#[should_panic(expected = "Rust engine does not support")]
fn test_arch_from_vex_s390x_panics() {
    let _ = arch_from_vex(VexArch::S390X);
}

#[test]
fn test_clone_box_dyn_arch_roundtrips() {
    // Cloning a boxed supported arch preserves its identity (no AMD64 fallback).
    for vex in [
        VexArch::X86,
        VexArch::AMD64,
        VexArch::ARM,
        VexArch::ARM64,
        VexArch::MIPS32,
        VexArch::MIPS64,
    ] {
        let boxed = arch_from_vex(vex);
        assert_eq!(boxed.clone().vex_arch(), vex);
    }
}

#[test]
fn test_register_file_concrete() {
    let ctx = SymContext::new_mock();

    let mut regs = RegisterFile::new(Box::new(AMD64));

    // Write RAX
    regs.put_reg("rax", RustBV::concrete(0x1234567890ABCDEF, 64));

    // Read back
    let rax = regs.get_reg("rax", &ctx).unwrap();
    assert_eq!(rax.as_u64(), Some(0x1234567890ABCDEF));

    // Read sub-register (AL is the low byte of RAX)
    // RAX is at offset 16 in VEX AMD64 guest state
    let al = regs.get(16, 1, &ctx);
    assert_eq!(al.as_u64(), Some(0xEF));
}

#[test]
fn test_register_file_symbolic() {
    let ctx = SymContext::new_mock();

    let mut regs = RegisterFile::new(Box::new(AMD64));

    // Write a symbolic value
    let sym = RustBV::symbolic(&ctx, "rax", 64);
    regs.put_reg("rax", sym.clone());

    // Read back
    let rax = regs.get_reg("rax", &ctx).unwrap();
    assert!(rax.is_symbolic());
}

/// Concrete-only round-trip: write a few registers, serialize, restore,
/// and verify the same values come back. Uses mock SymContext so the
/// test does not require Z3.
#[test]
fn serde_roundtrip_register_file_concrete() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new(Box::new(AMD64));
    regs.put_reg("rax", RustBV::concrete(0x1234567890ABCDEF, 64));
    regs.put_reg("rsp", RustBV::concrete(0x7fff_ffff_0000_1000, 64));

    let s = serde_json::to_string(&regs).expect("serialize");
    let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");

    assert_eq!(restored.arch().name(), "AMD64");
    assert_eq!(
        restored.get_reg("rax", &ctx).unwrap().as_u64(),
        Some(0x1234567890ABCDEF)
    );
    assert_eq!(
        restored.get_reg("rsp", &ctx).unwrap().as_u64(),
        Some(0x7fff_ffff_0000_1000)
    );
}

/// Symbolic round-trip: a register holding a `RustBV::Symbolic` is
/// reconstructed with a fresh Z3 AST under the active thread-local
/// context, and the rebuilt value is still recognized as symbolic
/// with the same `id`, `width`, and `name`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_register_file_symbolic() {
    use z3::{Config, Context, with_z3_context};

    let ctx = SymContext::new();
    let mut regs = RegisterFile::new(Box::new(AMD64));
    let sym = RustBV::symbolic(&ctx, "rdi_sym", 64);
    let (orig_id, orig_width) = match &sym {
        RustBV::Symbolic { id, width, .. } => (*id, *width),
        _ => panic!("expected symbolic"),
    };
    regs.put_reg("rdi", sym);

    let s = serde_json::to_string(&regs).expect("serialize");

    // Deserialize inside a fresh Z3 context to prove the AST cache
    // rebuilds correctly across context boundaries (the production
    // snapshot/restore path).
    let cfg = Config::new();
    let new_ctx = Context::new(&cfg);
    let (sym_id, sym_width, sym_name, is_symbolic) =
        with_z3_context(&new_ctx, || -> (u64, u32, String, bool) {
            let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");
            let ctx2 = SymContext::new();
            let bv = restored.get_reg("rdi", &ctx2).unwrap();
            match &bv {
                RustBV::Symbolic {
                    id, width, name, ..
                } => (*id, *width, name.to_string(), true),
                _ => (0, 0, String::new(), false),
            }
        });

    assert!(is_symbolic, "restored register must still be Symbolic");
    assert_eq!(sym_id, orig_id);
    assert_eq!(sym_width, orig_width);
    assert_eq!(sym_name, "rdi_sym");
}

/// Unknown arch names fall back to AMD64 (matches `Box<dyn Arch>::clone`
/// at arch/mod.rs:638), so a corrupted snapshot still loads into a
/// usable register file. Tests the fallback path explicitly.
#[test]
fn serde_unknown_arch_name_falls_back_to_amd64() {
    let bad = RegisterFileData {
        data: vec![0u8; AMD64.state_size()],
        symbolic: BTreeMap::new(),
        arch_name: "not_a_real_arch".to_string(),
    };
    let s = serde_json::to_string(&bad).expect("serialize");
    let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(restored.arch().name(), "AMD64");
}
