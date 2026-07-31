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

/// Internal consistency of the [`ALL_ARCHES`] registry: each row's `name` and
/// `vex` must agree with the arch `make_arch` actually builds, and the `vex`
/// column must be unique so `arch_from_vex`'s linear scan is unambiguous.
/// Everything else (`arch_from_name`, `cc_for_arch`) reads this table, so a
/// mismatched row would mis-route silently.
#[test]
fn test_all_arches_rows_are_self_consistent() {
    for (i, desc) in ALL_ARCHES.iter().enumerate() {
        let arch = desc.name;
        let built = (desc.make_arch)();
        assert_eq!(built.name(), desc.name, "{arch}: name vs Arch::name()");
        assert_eq!(
            built.vex_arch(),
            desc.vex,
            "{arch}: vex column vs Arch::vex_arch()"
        );
        assert_eq!(
            arch_from_vex(desc.vex).name(),
            desc.name,
            "{arch}: arch_from_vex round-trip"
        );
        for other in &ALL_ARCHES[i + 1..] {
            assert_ne!(
                desc.vex, other.vex,
                "{arch}: duplicate VexArch shared with {}",
                other.name
            );
        }
    }
}

/// Compile-time trip-wire for the `arch_from_vex` rewrite: it used to be an
/// exhaustive `match` on `VexArch`, so a new variant was a build error. Now it
/// is a linear scan of [`ALL_ARCHES`] with a runtime panic on a miss. This
/// exhaustive match restores the build error — adding a `VexArch` variant
/// forces a deliberate supported/unsupported decision here — and the loop then
/// checks `arch_from_vex` actually agrees with that decision.
#[test]
fn test_every_vex_arch_is_classified() {
    fn supported(vex: VexArch) -> bool {
        match vex {
            VexArch::X86
            | VexArch::AMD64
            | VexArch::ARM
            | VexArch::ARM64
            | VexArch::MIPS32
            | VexArch::MIPS64 => true,
            VexArch::PPC32 | VexArch::PPC64 | VexArch::S390X => false,
        }
    }

    for vex in [
        VexArch::X86,
        VexArch::AMD64,
        VexArch::ARM,
        VexArch::ARM64,
        VexArch::MIPS32,
        VexArch::MIPS64,
        VexArch::PPC32,
        VexArch::PPC64,
        VexArch::S390X,
    ] {
        let in_table = ALL_ARCHES.iter().any(|d| d.vex == vex);
        assert_eq!(
            in_table,
            supported(vex),
            "{vex:?}: ALL_ARCHES membership disagrees with the supported-arch list",
        );
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
    regs.put_reg("rax", sym);

    // Read back
    let rax = regs.get_reg("rax", &ctx).unwrap();
    assert!(rax.is_symbolic());
}

/// Copy-on-write fork isolation for the `Arc`-wrapped register buffer:
/// after `fork()` the child shares the parent's `data` Arc, but the first
/// concrete write to either side must `Arc::make_mut`-clone so the other
/// side is unaffected. Mirrors `memory::tests` fork isolation; the
/// faithfulness gate for angr-6t8z3.1 (CoW must not leak writes across
/// the parent/child boundary).
#[test]
fn test_register_file_fork_cow_isolation() {
    let ctx = SymContext::new_mock();

    let mut parent = RegisterFile::new(Box::new(AMD64));
    parent.put_reg("rax", RustBV::concrete(0xAAAA_AAAA_AAAA_AAAA, 64));

    // Fork: child shares the parent's data buffer until a write occurs.
    let mut child = parent.fork();
    assert_eq!(
        child.get_reg("rax", &ctx).unwrap().as_u64(),
        Some(0xAAAA_AAAA_AAAA_AAAA)
    );

    // Mutate the child via the concrete `put` path — must not touch parent.
    child.put_reg("rax", RustBV::concrete(0xBBBB_BBBB_BBBB_BBBB, 64));
    assert_eq!(
        child.get_reg("rax", &ctx).unwrap().as_u64(),
        Some(0xBBBB_BBBB_BBBB_BBBB)
    );
    assert_eq!(
        parent.get_reg("rax", &ctx).unwrap().as_u64(),
        Some(0xAAAA_AAAA_AAAA_AAAA),
        "child write leaked into parent (CoW broken)"
    );

    // Mutate the parent (rbx, untouched offset) — child must not see it.
    parent.put_reg("rbx", RustBV::concrete(0x1111_1111_1111_1111, 64));
    assert_eq!(
        parent.get_reg("rbx", &ctx).unwrap().as_u64(),
        Some(0x1111_1111_1111_1111)
    );
    assert_eq!(
        child.get_reg("rbx", &ctx).unwrap().as_u64(),
        Some(0),
        "parent write leaked into child (CoW broken)"
    );

    // copy_from_bytes is a separate &mut write path — also isolate it.
    let mut p2 = RegisterFile::new(Box::new(AMD64));
    p2.put_reg("rax", RustBV::concrete(0xDEAD_BEEF, 64));
    let c2 = p2.fork();
    let mut buf = vec![0u8; AMD64.state_size()];
    buf[16] = 0x77; // overwrite RAX low byte region
    p2.copy_from_bytes(&buf);
    assert_eq!(
        c2.get_reg("rax", &ctx).unwrap().as_u64(),
        Some(0xDEAD_BEEF),
        "copy_from_bytes leaked into forked child (CoW broken)"
    );
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
