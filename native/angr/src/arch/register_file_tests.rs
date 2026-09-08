//! Unit tests for [`super`] (arch/register_file.rs) — [`RegisterFile`]
//! get/put/merge, COW fork isolation, `copy_to_bytes`, big-endian mirroring,
//! and the serde round-trips through [`RegisterFileData`].
//!
//! Sibling of `registry_tests.rs`, which holds the [`ALL_ARCHES`] registry and
//! per-architecture descriptor sweeps. Split out of `mod_tests.rs` in
//! angr-6cp06.76 along the same seam round 6 (angr-5mnx3.1) used to split
//! `arch/mod.rs` into `register_file.rs` + `registry.rs`: per that file's own
//! header, get/put/merge "carry the densest regression history in `arch/`" and
//! sat interleaved with the unrelated architecture registry.

use super::*;
use crate::arch::registry::ALL_ARCHES;
use crate::symbolic::SymContext;

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

/// An unrecognized `arch_name` is rejected instead of silently reinterpreting
/// the register-byte image under AMD64's layout (angr-c7xno.3). Loud failure
/// here matches `RustSimState::from_snapshot`'s `SnapshotError::UnknownArch`
/// and `arch_from_vex`'s panic; the error text must name the offending arch so
/// a mismatched snapshot is diagnosable.
#[test]
fn serde_unknown_arch_name_is_rejected() {
    let bad = RegisterFileData {
        data: vec![0u8; AMD64.state_size()],
        symbolic: BTreeMap::new(),
        arch_name: "not_a_real_arch".to_string(),
        register_be: false,
    };
    let s = serde_json::to_string(&bad).expect("serialize");
    // `RegisterFile` is not `Debug`, so unwrap the error by hand rather than
    // via `expect_err`.
    let err = match serde_json::from_str::<RegisterFile>(&s) {
        Ok(_) => panic!("must reject unknown arch"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("not_a_real_arch"),
        "error must name the unrecognized arch, got: {err}"
    );
}

/// The rejection above is specific to *unrecognized* names — a register file
/// serialized under one known arch still reloads under that arch, with `data`
/// resized to its `state_size()`.
#[test]
fn serde_known_arch_name_round_trips() {
    let bad = RegisterFileData {
        data: vec![0u8; 4],
        symbolic: BTreeMap::new(),
        arch_name: "X86".to_string(),
        register_be: false,
    };
    let s = serde_json::to_string(&bad).expect("serialize");
    let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(restored.arch().name(), "X86");
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

/// The name-keyed `get_reg`/`put_reg` path reaches registers the u128
/// `register_names()` channel deliberately excludes — x86/amd64 `fpreg` is 64
/// bytes — and used to compose them straight into a `u128`. Reading shifted
/// past bit 127 (panic under `overflow-checks`, 16-byte-cycle garbage
/// otherwise, angr-0jh0j.1) and writing did the same in reverse
/// (angr-0jh0j.2). Neither had a test: the sibling
/// `x87_control_registers_round_trip_through_register_names` only asserts
/// fpreg's *absence* from `register_names()`.
///
/// Swept over the whole [`ALL_ARCHES`] × `canonical_registers()` cross-product
/// rather than hardcoding `fpreg` on the two x86 arches, so a wide register
/// added to any arch's table inherits the coverage instead of re-opening the
/// gap. `every_narrow_canonical_register_is_exported` is the same sweep for
/// the *export* channel; this one is the accessor path it feeds.
#[test]
fn registers_wider_than_u128_read_and_write_without_shift_overflow() {
    let ctx = SymContext::new();
    let mut covered = 0usize;
    for desc in ALL_ARCHES {
        let arch_name = desc.name;
        let wide: Vec<_> = (desc.make_arch)()
            .canonical_registers()
            .iter()
            .filter(|&&(_, _, size)| (size as usize) > MAX_CONCRETE_CHUNK)
            .copied()
            .collect();
        for (name, offset, size) in wide {
            covered += 1;
            // Fresh file per register: canonical entries may overlap, and the
            // zero-above-the-payload assertion below must see this write only.
            let mut regs = RegisterFile::new((desc.make_arch)());

            // Write: a u128-representable value into a wider-than-u128
            // register. The bytes above the payload are the value's own
            // implicit zeros, not a wrapped copy of the low 16.
            assert!(
                regs.put_reg(name, RustBV::concrete(0xdead_beef, size * 8)),
                "{arch_name}: put_reg({name}) rejected"
            );
            let offset = offset as usize;
            let stored = &regs.data[offset..offset + size as usize];
            assert_eq!(
                &stored[..4],
                &[0xef, 0xbe, 0xad, 0xde],
                "{arch_name}: {name} low bytes"
            );
            assert!(
                stored[4..].iter().all(|&b| b == 0),
                "{arch_name}: {name} bytes above the u128 payload must be \
                 zero, not a wrapped repeat of the low 16"
            );

            // Read: exact, and not representable as a u128 — which is what
            // makes the Python `get_register` surface report a clean error
            // instead of a truncated number.
            let got = regs.get_reg(name, &ctx).expect("register readable");
            assert_eq!(got.width(), size * 8, "{arch_name}: {name} read width");
            assert_eq!(
                got.as_u128(),
                None,
                "{arch_name}: a >128-bit read of {name} has no concrete \
                 representation"
            );
            // Byte 0 of the composed value is still the byte that was written.
            assert_eq!(
                got.extract(7, 0, &ctx).as_u128(),
                Some(0xef),
                "{arch_name}: {name} byte 0 survives composition"
            );
            assert_eq!(
                got.extract(size * 8 - 1, size * 8 - 8, &ctx).as_u128(),
                Some(0),
                "{arch_name}: {name} top byte"
            );
        }
    }
    // x86 and amd64 `fpreg` (64 B each) are the two the sweep exists for; if
    // the tables ever stop carrying a wide register the assertions above all
    // pass vacuously, which would silently retire the angr-0jh0j.1/.2 guard.
    assert!(
        covered >= 2,
        "expected at least the two fpreg entries wider than \
         MAX_CONCRETE_CHUNK, found {covered} — the sweep passed vacuously"
    );
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

/// Regression for angr-49v03: the widening merge must not drop an *adjacent*
/// overlay of `self`'s. `self` holds two independent byte overlays — `al` at
/// `rax_off` and `ah` at `rax_off + 1` — while `other` holds a full 64-bit
/// `rax`. `rax_off` takes the width-mismatch arm and records a widened span of
/// 8 bytes; the cleanup that follows then deletes every overlay strictly inside
/// that span, including the `ah` entry the same merge call had just written.
/// That is only sound if the wide value genuinely subsumes `ah`, which it did
/// not: `get`'s composition path used to substitute a neighbouring overlay only
/// when it exactly filled the remaining width, so `ah` (1 of the remaining 7
/// bytes) was silently replaced by stale concrete on both paths.
#[test]
fn merge_widen_keeps_adjacent_symbolic_overlay() {
    let ctx = SymContext::new_mock();
    let rax_off = AMD64.register_offset("rax").unwrap();

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, RustBV::symbolic(&ctx, "al", 8));
    a.put(rax_off + 1, RustBV::symbolic(&ctx, "ah", 8));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, RustBV::symbolic(&ctx, "rax", 64));

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));

    let collect_names = |bv: &RustBV| {
        let mut names: Vec<String> = Vec::new();
        let mut stack = vec![bv.clone()];
        while let Some(node) = stack.pop() {
            if let RustBV::Symbolic { name, .. } = &node {
                names.push(name.to_string());
            }
            if let Some(ops) = node.operands() {
                stack.extend(ops.iter().cloned());
            }
        }
        names
    };

    let wide = a.get(rax_off, 8, &ctx);
    assert_eq!(wide.width(), 64);
    let names = collect_names(&wide);
    for expected in ["al", "ah", "rax"] {
        assert!(
            names.iter().any(|n| n == expected),
            "merged rax lost `{expected}`: {names:?}"
        );
    }

    // Reading `ah` back must still see the symbolic byte, whether it survives as
    // its own overlay or as a slice of the widened one.
    let ah_back = a.get(rax_off + 1, 1, &ctx);
    assert_eq!(ah_back.width(), 8);
    let ah_names = collect_names(&ah_back);
    assert!(
        ah_names.iter().any(|n| n == "ah"),
        "reading ah after the merge returned a stale value: {ah_names:?}"
    );
}

/// Solver twin of `merge_widen_keeps_adjacent_symbolic_overlay`: the `cond == 0`
/// leg of the widened ITE must reproduce *both* of `self`'s byte overlays in
/// their own bit positions, zero-extended over its concrete high bytes.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn merge_widen_adjacent_overlay_selects_both_bytes_under_solver() {
    let ctx = SymContext::new();
    let rax_off = AMD64.register_offset("rax").unwrap();

    let al = RustBV::symbolic(&ctx, "al", 8);
    let ah = RustBV::symbolic(&ctx, "ah", 8);
    let rax = RustBV::symbolic(&ctx, "rax", 64);

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, al.clone());
    a.put(rax_off + 1, ah.clone());
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, rax.clone());

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));
    let got = a.get(rax_off, 8, &ctx).to_z3_ast();

    // cond == 1 selects other's whole 64-bit rax.
    ctx.push();
    ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    ctx.add_constraint(got.eq(rax.to_z3_ast()).not());
    assert!(!ctx.is_sat(), "cond=1 must select other's `rax`");
    ctx.pop();

    // cond == 0 selects self's two bytes over its own (zero) high bytes.
    let expected = ah.to_z3_ast().concat(al.to_z3_ast()).zero_ext(48);
    ctx.push();
    ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(0, 1)));
    ctx.add_constraint(got.eq(expected).not());
    assert!(
        !ctx.is_sat(),
        "cond=0 must select both of self's byte overlays"
    );
    ctx.pop();
}

/// angr-c7xno.1: the high-byte aliases ah/ch/dh/bh live at `GPR_offset + 1`, so
/// a branch can carry a symbolic overlay inside a register chunk while the
/// chunk's *aligned* offset is absent from both files' overlay maps. The
/// concrete-diff loop used to skip only on the aligned offset, so it fell
/// through to comparing raw backing bytes and inserted a concrete-only ITE at
/// `rax_off` — which `get` then returned verbatim, shadowing the merged `ah`.
#[test]
fn merge_concrete_chunk_keeps_unaligned_subregister_overlay() {
    let ctx = SymContext::new_mock();
    let rax_off = AMD64.register_offset("rax").unwrap();

    // Neither side has an overlay at `rax_off` itself; only `other` writes
    // `ah`. The underlying concrete bytes differ for an unrelated reason.
    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, RustBV::concrete(0xAAAA_AAAA_AAAA_AAAA, 64));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, RustBV::concrete(0xBBBB_BBBB_BBBB_BBBB, 64));
    b.put(rax_off + 1, RustBV::symbolic(&ctx, "ah", 8));
    assert!(!a.symbolic.contains_key(&rax_off));
    assert!(!b.symbolic.contains_key(&rax_off));

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));

    let mentions_ah = |bv: &RustBV| {
        let mut stack = vec![bv.clone()];
        while let Some(node) = stack.pop() {
            if let RustBV::Symbolic { name, .. } = &node
                && &**name == "ah"
            {
                return true;
            }
            if let Some(ops) = node.operands() {
                stack.extend(ops.iter().cloned());
            }
        }
        false
    };

    let wide = a.get(rax_off, 8, &ctx);
    assert_eq!(wide.width(), 64);
    assert!(
        mentions_ah(&wide),
        "full-width read after merge dropped the merged `ah` overlay: {wide:?}"
    );
    let ah_back = a.get(rax_off + 1, 1, &ctx);
    assert_eq!(ah_back.width(), 8);
    assert!(
        mentions_ah(&ah_back),
        "reading `ah` after the merge returned a stale value: {ah_back:?}"
    );
}

/// Solver twin of `merge_concrete_chunk_keeps_unaligned_subregister_overlay`:
/// each leg of the merged full-width value must equal that path's own
/// composition, `other`'s carrying `ah` in byte 1.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn merge_concrete_chunk_unaligned_overlay_selects_each_path_under_solver() {
    let ctx = SymContext::new();
    let rax_off = AMD64.register_offset("rax").unwrap();

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(rax_off, RustBV::concrete(0xAAAA_AAAA_AAAA_AAAA, 64));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(rax_off, RustBV::concrete(0xBBBB_BBBB_BBBB_BBBB, 64));
    b.put(rax_off + 1, RustBV::symbolic(&ctx, "ah", 8));

    let self_expected = a.get(rax_off, 8, &ctx).to_z3_ast();
    let other_expected = b.get(rax_off, 8, &ctx).to_z3_ast();

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));
    let got = a.get(rax_off, 8, &ctx).to_z3_ast();

    for (bit, expected, what) in [
        (1u64, other_expected, "other's `ah`-composed value"),
        (0u64, self_expected, "self's concrete value"),
    ] {
        ctx.push();
        ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(bit, 1)));
        ctx.add_constraint(got.eq(expected).not());
        assert!(!ctx.is_sat(), "cond={bit} must select {what}");
        ctx.pop();
    }
}

/// amd64's guest state is 1060 bytes — not a multiple of the 8-byte chunk
/// `RegisterFile::merge`'s concrete-diff scan walks in — so the trailing 4
/// bytes used to fall off the end of the scan entirely and a concrete
/// divergence there was silently resolved in `self`'s favour (angr-91vj9.13).
#[test]
fn merge_covers_short_tail_chunk_of_guest_state() {
    let ctx = SymContext::new_mock();
    let reg_bytes = AMD64.bits() as usize / 8;
    let tail = AMD64.state_size() % reg_bytes;
    assert_eq!(tail, 4, "amd64's guest state is expected to have a short tail");
    let tail_off = (AMD64.state_size() - tail) as u32;

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(tail_off, RustBV::concrete(0x1111_1111, 32));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(tail_off, RustBV::concrete(0x2222_2222, 32));
    // Concrete puts mirror into the backing bytes, leaving no overlay — the
    // divergence is visible only to the concrete-diff scan.
    assert!(!a.symbolic.contains_key(&tail_off));
    assert!(!b.symbolic.contains_key(&tail_off));

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(
        a.merge(&b, &cond, &ctx),
        "a divergence in the short tail chunk must count as a merge"
    );
    let merged = a.get(tail_off, 4, &ctx);
    assert_eq!(merged.width(), 32);
    assert!(
        merged.as_u64().is_none(),
        "tail chunk collapsed to a single path's concrete value: {merged:?}"
    );
}

/// Solver twin of `merge_covers_short_tail_chunk_of_guest_state`: each leg of
/// the merged tail value must equal that path's own bytes.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn merge_short_tail_chunk_selects_each_path_under_solver() {
    let ctx = SymContext::new();
    let reg_bytes = AMD64.bits() as usize / 8;
    let tail_off = (AMD64.state_size() - AMD64.state_size() % reg_bytes) as u32;

    let mut a = RegisterFile::new(Box::new(AMD64));
    a.put(tail_off, RustBV::concrete(0x1111_1111, 32));
    let mut b = RegisterFile::new(Box::new(AMD64));
    b.put(tail_off, RustBV::concrete(0x2222_2222, 32));

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx));
    let got = a.get(tail_off, 4, &ctx).to_z3_ast();

    for (bit, expected, what) in [
        (1u64, 0x2222_2222u64, "other's tail bytes"),
        (0u64, 0x1111_1111u64, "self's tail bytes"),
    ] {
        ctx.push();
        ctx.add_constraint(cond.to_z3_ast().eq(z3::ast::BV::from_u64(bit, 1)));
        ctx.add_constraint(got.eq(z3::ast::BV::from_u64(expected, 32)).not());
        assert!(!ctx.is_sat(), "cond={bit} must select {what}");
        ctx.pop();
    }
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

/// `RegisterFile::get`'s same-offset arm is a short-circuit for the contained-
/// overlay loop further down, so it must produce that loop's answer even when
/// the read runs past the end of the concrete backing bytes. It used to guard
/// itself with `offset + size <= data.len()` and fall through to the loop in
/// that case; the loop calls the same `compose_range`, which zero-pads out-of-
/// range positions, so the guard only bought an extra pair of map scans
/// (angr-03vl4.3). Pin the padded result so the two paths cannot diverge.
#[test]
fn get_composes_past_end_of_concrete_backing() {
    let ctx = SymContext::new_mock();
    let state_size = AMD64.state_size() as u32;
    // Straddle the end: the low 2 bytes have an overlay, the high 6 are past
    // the end of `data` entirely.
    let offset = state_size - 2;

    let mut regs = RegisterFile::new(Box::new(AMD64));
    regs.put(offset, RustBV::symbolic(&ctx, "straddle", 16));

    let got = regs.get(offset, 8, &ctx);
    assert_eq!(got.width(), 64, "read keeps its requested width");
    // Upper 6 bytes are the zero padding; the low 2 are the overlay.
    assert_eq!(
        got.extract(63, 16, &ctx).as_u128(),
        Some(0),
        "positions past the end of `data` pad with zero"
    );
    assert!(
        got.extract(15, 0, &ctx).is_symbolic(),
        "the overlay survives into the low bytes"
    );
}

/// angr-fuhmm: on a big-endian target, a VEX access narrower than the
/// register containing it must select the same half angr's Python engine
/// does. Python stores the register file in `arch.register_endness`
/// (`Iend_BE` for MIPS), so the register's MSB sits at its lowest byte
/// offset; this file stores little-endian and mirrors the offset instead
/// (see `RegisterFile::mirror_offset`). MIPS `mov.d` under FR=0 is the case
/// that surfaced it: VEX lifts it to an F32 copy of `fN_lo` at the
/// register's *base* offset.
#[test]
fn be_register_file_mirrors_sub_register_access() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new_with_endian(Box::new(MIPS32), false);
    let f2 = regs.arch().register_offset("f2").unwrap();
    let f0 = regs.arch().register_offset("f0").unwrap();

    // Whole-register access is unaffected — that is the crossing the
    // Python<->Rust value-level sync already agreed on.
    regs.put_reg("f2", RustBV::concrete(0xAABB_CCDD_1122_3344, 64));
    assert_eq!(
        regs.get_reg("f2", &ctx).unwrap().as_u64(),
        Some(0xAABB_CCDD_1122_3344)
    );

    // GET:I32 at f2's base offset is `f2_lo`; big-endian storage puts the
    // HIGH word there.
    let lo_field = regs.get(f2, 4, &ctx);
    assert_eq!(lo_field.as_u64(), Some(0xAABB_CCDD), "f2_lo reads high word");
    // ...and the second half of the register holds the low word.
    assert_eq!(regs.get(f2 + 4, 4, &ctx).as_u64(), Some(0x1122_3344));

    // PUT:I32 at f0's base offset lands in f0's high word, exactly as the
    // Python engine reports after `mov.d $f0, $f2`.
    regs.put(f0, lo_field);
    assert_eq!(
        regs.get_reg("f0", &ctx).unwrap().as_u64(),
        Some(0xAABB_CCDD_0000_0000)
    );
}

/// angr-6cp06.74: `RegisterFile::merge` works entirely in *storage*
/// coordinates — its offset set comes from both files' `symbolic` keys, which
/// `put` wrote after mirroring. It used to feed those keys to `get`, which
/// mirrors again; on a big-endian file that lands on the *other* half of the
/// containing register, so the merge composed and ITE'd the wrong sub-field.
/// Invisible on every little-endian arch (`mirror_offset` is the identity) and
/// on every pre-existing merge test, which all use the LE-only
/// `RegisterFile::new`.
///
/// The repro is the documented MIPS FR=0 shape: `self` writes a symbolic
/// 32-bit `f2_lo` at f2's base offset (stored under the mirrored key `f2 + 4`),
/// `other` writes a concrete whole-register `f2`. Reading `f2_lo` back out of
/// the merge must give the register's HIGH word, exactly as
/// `be_register_file_mirrors_sub_register_access` pins for a plain read.
#[test]
fn be_merge_reads_the_other_side_in_storage_coordinates() {
    let ctx = SymContext::new_mock();
    let f2 = MIPS32.register_offset("f2").unwrap();

    let build = || {
        let mut a = RegisterFile::new_with_endian(Box::new(MIPS32), false);
        a.put(f2, RustBV::symbolic(&ctx, "f2_lo", 32));
        let mut b = RegisterFile::new_with_endian(Box::new(MIPS32), false);
        b.put_reg("f2", RustBV::concrete(0xAABB_CCDD_1122_3344, 64));
        (a, b)
    };

    // cond = 1 folds the ITE to `other`'s side.
    let (mut a, b) = build();
    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));
    assert_eq!(
        a.get(f2, 4, &ctx).as_u64(),
        Some(0xAABB_CCDD),
        "merged f2_lo must be other's HIGH word, not its low one"
    );

    // cond = 0 folds to `self`'s side, which is still the symbol it wrote.
    let (mut a, b) = build();
    assert!(a.merge(&b, &RustBV::concrete(0, 1), &ctx));
    let got = a.get(f2, 4, &ctx);
    assert_eq!(got.width(), 32);
    assert!(
        matches!(&got, RustBV::Symbolic { name, .. } if &**name == "f2_lo"),
        "merged f2_lo must be self's own overlay, got {got:?}"
    );
}

/// The little-endian path must be byte-for-byte what it was before
/// angr-fuhmm: no containment table, `mirror_offset` the identity.
#[test]
fn le_register_file_does_not_mirror() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new_with_endian(Box::new(MIPS32), true);
    let f2 = regs.arch().register_offset("f2").unwrap();
    regs.put_reg("f2", RustBV::concrete(0xAABB_CCDD_1122_3344, 64));
    assert_eq!(regs.get(f2, 4, &ctx).as_u64(), Some(0x1122_3344));
    assert!(regs.containment.is_none());
}

/// `mirror_offset` is an involution on the ranges it moves, and the identity
/// on whole registers, on the uncovered VEX bookkeeping tail, and on a range
/// that straddles two registers.
#[test]
fn mirror_offset_is_an_involution_and_identity_where_it_must_be() {
    let regs = RegisterFile::new_with_endian(Box::new(MIPS32), false);
    let f0 = regs.arch().register_offset("f0").unwrap();
    let f0_size = regs.arch().register_size("f0").unwrap();

    for size in [1u32, 2, 4] {
        for byte in 0..f0_size - size {
            let m = regs.mirror_offset(f0 + byte, size);
            assert_eq!(
                regs.mirror_offset(m, size),
                f0 + byte,
                "mirror is an involution at +{byte} size {size}"
            );
        }
    }
    // Whole register: identity.
    assert_eq!(regs.mirror_offset(f0, f0_size), f0);
    // Straddling f0/f1: identity (meaningless under either model).
    assert_eq!(regs.mirror_offset(f0 + 4, 8), f0 + 4);
    // Past the end of the guest state: identity, no panic.
    let tail = regs.arch().state_size() as u32;
    assert_eq!(regs.mirror_offset(tail + 16, 4), tail + 16);
    // Degenerate size: identity, no underflow.
    assert_eq!(regs.mirror_offset(f0, 0), f0);
}

/// A big-endian register file survives the snapshot round-trip with its
/// adapter intact — `register_be` is serialized, `containment` rebuilt.
#[test]
fn be_register_file_survives_serde_round_trip() {
    let ctx = SymContext::new_mock();
    let mut regs = RegisterFile::new_with_endian(Box::new(MIPS32), false);
    regs.put_reg("f2", RustBV::concrete(0xAABB_CCDD_1122_3344, 64));
    let bytes = serde_json::to_string(&regs).expect("serialize");
    let restored: RegisterFile = serde_json::from_str(&bytes).expect("deserialize");
    assert!(restored.containment.is_some());
    let f2 = restored.arch().register_offset("f2").unwrap();
    assert_eq!(restored.get(f2, 4, &ctx).as_u64(), Some(0xAABB_CCDD));
}
