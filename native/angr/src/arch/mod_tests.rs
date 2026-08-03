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

/// Expected scalars for the [`ALL_ARCHES`] sweeps below, one row per
/// architecture, joined to `ALL_ARCHES` by `name`.
///
/// Replaces the per-arch `test_*_basics` / `test_special_registers` /
/// `test_register_lookup` copies that used to live in `amd64_tests.rs`,
/// `x86_tests.rs`, `arm_tests.rs`, `arm64_tests.rs` and `mips_tests.rs`
/// (angr-9ke6b.215). Coverage of a minority arch used to depend on someone
/// remembering to hand-author the same assertion a sixth time, and it showed:
/// the two MIPS `*_basics` copies had silently dropped the `is_little_endian`
/// check and MIPS had no `test_special_registers` at all. Adding an assertion
/// or an architecture now buys the whole matrix.
struct ArchExpect {
    name: &'static str,
    bits: u32,
    little_endian: bool,
    ip_offset: u32,
    sp_offset: u32,
    bp_offset: Option<u32>,
    /// `(name, offset, size)` triples pinning alias-table entries from the
    /// Rust side, so `cargo test` alone catches a regression rather than
    /// relying on the Python archinfo parity gate
    /// (`tests/engines/rust/test_arch_offset_parity.py`). Two groups:
    ///
    /// * The architecture-independent full-width `sp`/`bp` names, plus the
    ///   legacy 16-bit x86/amd64 sub-registers that must have kept their
    ///   narrow widths when `sp`/`bp` were widened (angr-6qzik).
    /// * The MIPS `$N` spellings, which the parity gate cannot see at all
    ///   because archinfo has no register under those names.
    aliases: &'static [(&'static str, u32, u32)],
    /// The arch's own spelling of the instruction pointer, i.e. the name that
    /// must resolve to `ip_offset`.
    ip_name: &'static str,
}

const ARCH_EXPECTATIONS: &[ArchExpect] = &[
    ArchExpect {
        name: "X86",
        bits: 32,
        little_endian: true,
        ip_offset: 68,
        sp_offset: 24,
        bp_offset: Some(28),
        aliases: &[
            ("sp", 24, 4),
            ("bp", 28, 4),
            ("pc", 68, 4),
            ("ax", 8, 2),
            ("di", 36, 2),
        ],
        ip_name: "eip",
    },
    ArchExpect {
        name: "AMD64",
        bits: 64,
        little_endian: true,
        ip_offset: 184,
        sp_offset: 48,
        bp_offset: Some(56),
        aliases: &[
            ("sp", 48, 8),
            ("bp", 56, 8),
            ("pc", 184, 8),
            ("ax", 16, 2),
            ("si", 64, 2),
        ],
        ip_name: "rip",
    },
    ArchExpect {
        name: "ARM",
        bits: 32,
        little_endian: true,
        ip_offset: 68,       // PC (R15T)
        sp_offset: 60,       // R13
        bp_offset: Some(52), // R11/FP
        aliases: &[("sp", 60, 4)],
        ip_name: "pc",
    },
    ArchExpect {
        name: "ARM64",
        bits: 64,
        little_endian: true,
        ip_offset: 272,       // PC
        sp_offset: 264,       // XSP
        bp_offset: Some(248), // X29/FP
        aliases: &[("sp", 264, 8)],
        ip_name: "pc",
    },
    ArchExpect {
        name: "MIPS32",
        bits: 32,
        // MIPS is bi-endian; the Rust singleton defaults to little-endian and
        // callers select BE per state. This assertion is new — the hand-written
        // MIPS `*_basics` tests omitted it.
        little_endian: true,
        ip_offset: 136,       // PC
        sp_offset: 124,       // R29
        bp_offset: Some(128), // R30 (fp/s8)
        aliases: &[
            ("sp", 124, 4),
            ("$2", 16, 4),
            ("$29", 124, 4),
            ("$30", 128, 4),
        ],
        ip_name: "pc",
    },
    ArchExpect {
        name: "MIPS64",
        bits: 64,
        little_endian: true,
        ip_offset: 272,       // PC
        sp_offset: 248,       // R29
        bp_offset: Some(256), // R30 (fp/s8)
        aliases: &[
            ("sp", 248, 8),
            ("$2", 32, 8),
            ("$29", 248, 8),
            ("$30", 256, 8),
        ],
        ip_name: "pc",
    },
];

fn arch_expect(name: &str) -> &'static ArchExpect {
    ARCH_EXPECTATIONS
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| {
            panic!("{name}: no ARCH_EXPECTATIONS row; add one alongside the ALL_ARCHES row")
        })
}

/// The two tables must describe the same set of architectures, so adding a row
/// to [`ALL_ARCHES`] without an [`ARCH_EXPECTATIONS`] row (or vice versa) fails
/// loudly instead of silently shrinking the sweeps below.
#[test]
fn test_arch_expectations_cover_all_arches() {
    for desc in ALL_ARCHES {
        let _ = arch_expect(desc.name);
    }
    for expect in ARCH_EXPECTATIONS {
        assert!(
            ALL_ARCHES.iter().any(|d| d.name == expect.name),
            "{}: ARCH_EXPECTATIONS row has no matching ALL_ARCHES row",
            expect.name,
        );
    }
}

#[test]
fn test_all_arches_report_expected_bits_name_and_endianness() {
    for desc in ALL_ARCHES {
        let arch = (desc.make_arch)();
        let expect = arch_expect(desc.name);
        let name = expect.name;
        assert_eq!(arch.name(), name, "{name}: Arch::name()");
        assert_eq!(arch.bits(), expect.bits, "{name}: bits");
        assert_eq!(arch.bytes(), expect.bits / 8, "{name}: bytes");
        assert_eq!(
            arch.is_little_endian(),
            expect.little_endian,
            "{name}: is_little_endian",
        );
    }
}

#[test]
fn test_all_arches_report_expected_special_register_offsets() {
    for desc in ALL_ARCHES {
        let arch = (desc.make_arch)();
        let expect = arch_expect(desc.name);
        let name = expect.name;
        assert_eq!(arch.ip_offset(), expect.ip_offset, "{name}: ip_offset");
        assert_eq!(arch.sp_offset(), expect.sp_offset, "{name}: sp_offset");
        assert_eq!(arch.bp_offset(), expect.bp_offset, "{name}: bp_offset");

        // The special-register offsets must agree with the name lookup, so a
        // table edit that moves one but not the other cannot pass.
        assert_eq!(
            arch.register_offset(expect.ip_name),
            Some(expect.ip_offset),
            "{name}: register_offset({:?}) disagrees with ip_offset",
            expect.ip_name,
        );
        // Every arch must also answer to the architecture-independent `"pc"`
        // spelling (angr-9ke6b.217 closed the X86/AMD64 gap); archinfo defines
        // it on all six, and a direct Rust-level `set_register("pc", ...)`
        // never passes through `RustStateProxy._canonical_name`.
        assert_eq!(
            arch.register_offset("pc"),
            Some(expect.ip_offset),
            "{name}: register_offset(\"pc\") disagrees with ip_offset",
        );
        assert_eq!(
            arch.register_offset("sp"),
            Some(expect.sp_offset),
            "{name}: register_offset(\"sp\") disagrees with sp_offset",
        );
    }
}

#[test]
fn test_all_arches_resolve_their_alias_spellings() {
    for desc in ALL_ARCHES {
        let arch = (desc.make_arch)();
        let expect = arch_expect(desc.name);
        let name = expect.name;
        for &(alias, offset, size) in expect.aliases {
            assert_eq!(
                arch.register_offset(alias),
                Some(offset),
                "{name}: register_offset({alias:?})",
            );
            assert_eq!(
                arch.register_size(alias),
                Some(size),
                "{name}: register_size({alias:?})",
            );
        }
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

/// `copy_to_bytes` must report a now-symbolic register as zero, not as the
/// concrete value it held before the symbolic write (angr-9ke6b.5). `put`'s
/// symbolic branch only touches the `symbolic` map, so the stale bytes are
/// still sitting in `data` — a consumer of the flat buffer (the export
/// snapshot's `registers_raw`) cannot tell them apart from a live concrete
/// value.
#[test]
fn copy_to_bytes_zeroes_symbolic_registers() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new(Box::new(AMD64));

    // rax concrete, then overwritten symbolically; rbx stays concrete.
    regs.put_reg("rax", RustBV::concrete(0x1234_5678_9ABC_DEF0, 64));
    regs.put_reg("rbx", RustBV::concrete(0x0BAD_C0DE_0BAD_C0DE, 64));
    regs.put_reg("rax", RustBV::symbolic(&ctx, "rax_sym", 64));

    let mut bytes = vec![0u8; AMD64.state_size()];
    regs.copy_to_bytes(&mut bytes);

    let rax_off = AMD64.register_offset("rax").unwrap() as usize;
    let rbx_off = AMD64.register_offset("rbx").unwrap() as usize;
    assert_eq!(
        &bytes[rax_off..rax_off + 8],
        &[0u8; 8],
        "symbolic rax leaked its stale concrete bytes into the flat buffer"
    );
    assert_eq!(
        u64::from_le_bytes(bytes[rbx_off..rbx_off + 8].try_into().unwrap()),
        0x0BAD_C0DE_0BAD_C0DE,
        "zeroing the symbolic span must not touch neighboring concrete registers"
    );

    // Writing rax back concretely clears the symbolic overlay: the buffer
    // reports the live value again.
    regs.put_reg("rax", RustBV::concrete(0x00FF_00FF_00FF_00FF, 64));
    regs.copy_to_bytes(&mut bytes);
    assert_eq!(
        u64::from_le_bytes(bytes[rax_off..rax_off + 8].try_into().unwrap()),
        0x00FF_00FF_00FF_00FF
    );
}

/// A concrete sub-register write into a wider symbolic register (`put`'s
/// compose branch updates `data` for the written bytes but keeps a symbolic
/// entry covering the whole register) must still read back as all-zero: the
/// register as a whole is not concretely representable.
#[test]
fn copy_to_bytes_zeroes_partially_concrete_symbolic_register() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new(Box::new(AMD64));

    regs.put_reg("rax", RustBV::symbolic(&ctx, "rax_sym", 64));
    // `al` is the low byte of rax — composes into the symbolic entry.
    let al_off = AMD64.register_offset("rax").unwrap();
    regs.put(al_off, RustBV::concrete(0xEF, 8));

    let mut bytes = vec![0u8; AMD64.state_size()];
    regs.copy_to_bytes(&mut bytes);
    let rax_off = al_off as usize;
    assert_eq!(
        &bytes[rax_off..rax_off + 8],
        &[0u8; 8],
        "partially-concrete symbolic rax must not report a half-real value"
    );
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

/// Unknown arch names fall back to AMD64 (matches `impl Clone for
/// Box<dyn Arch>` in `arch/mod.rs`), so a corrupted snapshot still loads into a
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

/// `register_names()` is the set that crosses the Python boundary
/// (`register_names_for_arch` -> `_supported_register_names`, and
/// `export_full`'s `named_registers`). Two invariants, for every arch:
///
///  1. Every name resolves through `register_offset` / `register_size`,
///     otherwise `set_registers_bulk` raises `unknown register: ...` on the
///     Python side and the whole bulk write is lost.
///  2. No entry exceeds 16 bytes. The named-register channel carries each
///     value as a `u128` (`ExplorationStateSnapshot::get_registers_named`),
///     and `RegisterFile::get`'s concrete read shifts bytes into a `u128`, so
///     a wider register would silently produce garbage. This is why `fpreg`
///     (64 B) stays out of the x86/AMD64 lists (angr-9ke6b.6).
#[test]
fn register_names_all_resolve_and_fit_in_u128() {
    for desc in ALL_ARCHES {
        let arch = (desc.make_arch)();
        for &name in arch.register_names() {
            let offset = arch
                .register_offset(name)
                .unwrap_or_else(|| panic!("{}: register_names has unresolvable {name}", desc.name));
            let size = arch
                .register_size(name)
                .unwrap_or_else(|| panic!("{}: register_names has unsized {name}", desc.name));
            assert!(
                size <= 16,
                "{}: {name} is {size} bytes, too wide for the u128 named-register channel",
                desc.name
            );
            assert!(
                offset as usize + size as usize <= arch.state_size(),
                "{}: {name} runs past the guest state",
                desc.name
            );
        }
    }
}

/// Expected `(offset, size)` for the five VEX bookkeeping fields, one row per
/// architecture, joined to [`ALL_ARCHES`] by `name`. Every value here was read
/// off `archinfo.Arch*.registers` (angr-9ke6b.10).
const VEX_BOOKKEEPING: &[(&str, [(&str, u32, u32); 5])] = &[
    (
        "X86",
        [
            ("emnote", 320, 4),
            ("cmstart", 324, 4),
            ("cmlen", 328, 4),
            ("nraddr", 332, 4),
            ("ip_at_syscall", 340, 4),
        ],
    ),
    (
        "AMD64",
        [
            ("emnote", 992, 4),
            ("cmstart", 1000, 8),
            ("cmlen", 1008, 8),
            ("nraddr", 1016, 8),
            ("ip_at_syscall", 1040, 8),
        ],
    ),
    (
        "ARM",
        [
            ("emnote", 108, 4),
            ("cmstart", 112, 4),
            ("cmlen", 116, 4),
            ("nraddr", 120, 4),
            ("ip_at_syscall", 124, 4),
        ],
    ),
    (
        "ARM64",
        [
            ("emnote", 848, 4),
            ("cmstart", 856, 8),
            ("cmlen", 864, 8),
            ("nraddr", 872, 8),
            ("ip_at_syscall", 880, 8),
        ],
    ),
    (
        "MIPS32",
        [
            ("emnote", 432, 4),
            ("cmstart", 436, 4),
            ("cmlen", 440, 4),
            ("nraddr", 444, 4),
            ("ip_at_syscall", 492, 4),
        ],
    ),
    (
        "MIPS64",
        [
            ("emnote", 584, 4),
            ("cmstart", 592, 8),
            ("cmlen", 600, 8),
            ("nraddr", 608, 8),
            ("ip_at_syscall", 616, 8),
        ],
    ),
];

/// `state.regs.ip_at_syscall` (and the emnote/cmstart/cmlen/nraddr siblings)
/// must resolve on *every* arch, not just the three that happened to name them
/// first: X86, ARM and ARM64 had consts while AMD64 and MIPS32/64 silently
/// returned `None`, so the same Python read worked or failed depending on the
/// target (angr-9ke6b.10). Also pins the guest state wide enough to hold them —
/// MIPS32/64's `GUEST_STATE_SIZE` used to stop right where this block begins.
#[test]
fn vex_bookkeeping_fields_resolve_on_every_arch() {
    for desc in ALL_ARCHES {
        let arch = (desc.make_arch)();
        let (_, fields) = VEX_BOOKKEEPING
            .iter()
            .find(|(name, _)| *name == desc.name)
            .unwrap_or_else(|| {
                panic!(
                    "{}: no VEX_BOOKKEEPING row; add one alongside the ALL_ARCHES row",
                    desc.name
                )
            });
        for &(field, offset, size) in fields {
            assert_eq!(
                arch.register_offset(field),
                Some(offset),
                "{}: {field} offset",
                desc.name
            );
            assert_eq!(
                arch.register_size(field),
                Some(size),
                "{}: {field} size",
                desc.name
            );
            assert!(
                offset as usize + size as usize <= arch.state_size(),
                "{}: {field} runs past the guest state ({} bytes)",
                desc.name,
                arch.state_size()
            );
        }
    }
}

/// The x87 control/status words must round-trip through the named-register
/// path on both x86 arches — they were absent from `REGISTER_NAMES` while
/// present in `CANONICAL`, so x87 state silently never reached
/// `state.regs.*` after Rust-engine exploration (angr-9ke6b.6).
#[test]
fn x87_control_registers_round_trip_through_register_names() {
    let ctx = SymContext::new();
    for arch in [
        arch_from_name("AMD64").unwrap(),
        arch_from_name("X86").unwrap(),
    ] {
        let arch_name = arch.name();
        let mut regs = RegisterFile::new(arch);
        let names = regs.arch().register_names();
        for fpu in ["fptag", "fpround", "fc3210", "ftop"] {
            assert!(
                names.contains(&fpu),
                "{arch_name}: {fpu} missing from register_names()"
            );
        }
        assert!(
            !names.contains(&"fpreg"),
            "{arch_name}: fpreg is 64 bytes and cannot use the u128 channel"
        );

        for fpu in ["fptag", "fpround", "fc3210", "ftop"] {
            let bits = regs.arch().register_size(fpu).unwrap() * 8;
            assert!(
                regs.put_reg(fpu, RustBV::concrete(0x25, bits)),
                "{arch_name}: put_reg({fpu}) rejected"
            );
            let got = regs.get_reg(fpu, &ctx).expect("register readable");
            assert_eq!(got.as_u128(), Some(0x25), "{arch_name}: {fpu} round-trip");
            assert_eq!(got.width(), bits, "{arch_name}: {fpu} width");
        }
    }
}

/// `RegisterFile::merge` with symbolic values of *different* widths at the
/// same offset — one path wrote `eax` (32-bit), the other `rax` (64-bit).
/// The merge used to keep `self`'s value and silently drop `other`'s, so the
/// merged state behaved as if only one path could reach the register
/// (angr-9ke6b.13). Both sides must be widened to the larger width and
/// ITE-merged instead.
#[test]
fn merge_width_mismatch_ite_merges_both_paths() {
    let ctx = SymContext::new_mock();
    let rax_off = AMD64.register_offset("rax").unwrap();

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, RustBV::symbolic(&ctx, "narrow", 32));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, RustBV::symbolic(&ctx, "wide", 64));

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(
        a.merge(&b, &cond, &ctx),
        "width mismatch must count as merged"
    );

    let got = a.get(rax_off, 8, &ctx);
    assert_eq!(got.width(), 64, "merged rax must be the wider of the two");
    assert!(got.is_symbolic());

    // Both branches survive: the merged expression tree mentions both symbols.
    let mut names: Vec<String> = Vec::new();
    let mut stack = vec![got.clone()];
    while let Some(node) = stack.pop() {
        if let RustBV::Symbolic { name, .. } = &node {
            names.push(name.to_string());
        }
        if let Some(ops) = node.operands() {
            stack.extend(ops.iter().cloned());
        }
    }
    assert!(
        names.iter().any(|n| n == "narrow"),
        "self's value was dropped by the merge: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "wide"),
        "other's value was dropped by the merge: {names:?}"
    );
}

/// Semantic twin of `merge_width_mismatch_ite_merges_both_paths`: prove with
/// the solver that the widened ITE selects each path's value under the
/// matching merge condition. `self`'s 32-bit value composes with its own
/// (zero) high bytes, so the `cond == 0` leg must equal `zero_extend(narrow)`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn merge_width_mismatch_selects_each_path_under_solver() {
    let ctx = SymContext::new();
    let rax_off = AMD64.register_offset("rax").unwrap();

    let narrow = RustBV::symbolic(&ctx, "narrow", 32);
    let wide = RustBV::symbolic(&ctx, "wide", 64);

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, narrow.clone());
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, wide.clone());

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));
    let got = a.get(rax_off, 8, &ctx).to_z3_ast();

    // cond == 1 selects other's 64-bit value.
    ctx.push();
    ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    ctx.add_constraint(got.eq(wide.to_z3_ast()).not());
    assert!(!ctx.is_sat(), "cond=1 must select other's `wide` value");
    ctx.pop();

    // cond == 0 selects self's 32-bit value zero-extended over its own
    // concrete (zero) high half.
    ctx.push();
    ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(0, 1)));
    ctx.add_constraint(got.eq(narrow.to_z3_ast().zero_ext(32)).not());
    assert!(!ctx.is_sat(), "cond=0 must select self's `narrow` value");
    ctx.pop();
}
