//! Overlapping writes over an existing symbolic value.
//!
//! Two flavours: a later symbolic store partially covering an earlier one
//! (later store wins, on both the eager and the lazy load paths, with the
//! address/value constraints surviving in the solver), and a *concrete* or
//! narrower-symbolic store landing inside a wider symbolic entry, which must
//! truncate that entry and clear its now-stale `symbolic_spans`.
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-wyxb: when two symbolic stores partially overlap, the address
/// constraint on each store's address expression and any value
/// constraints must remain in the solver after the stores complete.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_store_partial_overlap_constraint_propagation() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Symbolic addresses, each pinned to a specific value via a
    // constraint added to the solver up-front.
    let addr1 = RustBV::symbolic(&ctx, "addr1", 64);
    let addr2 = RustBV::symbolic(&ctx, "addr2", 64);
    ctx.assume_true(&addr1.eq(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr2.eq(&RustBV::concrete(0x1004, 64), &ctx));

    // Two 64-bit symbolic values; sym1 carries an additional constraint
    // (a specific u128 value) so we can verify that this value-side
    // constraint also survives the partial overlap.
    let sym1 = RustBV::symbolic(&ctx, "sym1", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2", 64);
    let pinned_sym1: u128 = 0xDEAD_BEEF_F00D_BABE;
    ctx.assume_true(&sym1.eq(&RustBV::concrete(pinned_sym1, 64), &ctx));

    // Partial overlap: sym1 covers [0x1000, 0x1008); sym2 covers
    // [0x1004, 0x100C). Bytes [0x1004, 0x1008) are written by both.
    mem.store_symbolic(addr1.clone(), sym1.clone(), &ctx, &concretizer)
        .expect("store_symbolic addr1 must succeed");
    mem.store_symbolic(addr2, sym2, &ctx, &concretizer)
        .expect("store_symbolic addr2 must succeed");

    // The base context must remain satisfiable.
    assert!(
        ctx.is_sat(),
        "context must stay SAT after partial-overlap stores"
    );

    // addr1's solution constraint (== 0x1000) must survive: probing
    // an alternative value in a forked context must be UNSAT.
    let probe_addr = ctx.fork();
    probe_addr.assume_true(&addr1.eq(&RustBV::concrete(0x2000, 64), &probe_addr));
    assert!(
        !probe_addr.is_sat(),
        "addr1==0x1000 must survive partial-overlap stores; \
         probing addr1==0x2000 was unexpectedly SAT"
    );

    // sym1's value constraint must survive: probing sym1 == 0 must
    // be UNSAT (sym1 is pinned to 0xDEAD_BEEF_F00D_BABE).
    let probe_sym1 = ctx.fork();
    probe_sym1.assume_true(&sym1.eq(&RustBV::concrete(0, 64), &probe_sym1));
    assert!(
        !probe_sym1.is_sat(),
        "sym1's value constraint must survive partial-overlap stores; \
         probing sym1==0 was unexpectedly SAT"
    );

    // Loading from 0x1000 must still produce a satisfiable expression
    // and stay consistent with the surviving value-side constraints.
    let loaded = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("load[0x1000:4] must succeed after partial-overlap stores");
    assert!(
        ctx.is_sat(),
        "context must remain SAT after the post-store load"
    );
    // Eval should produce *some* concrete model — addr/value constraints
    // narrow the model space but do not make it UNSAT.
    assert!(
        ctx.eval(&loaded).is_some(),
        "loaded value must be evaluable under the preserved constraints"
    );
}

/// angr-3zhl: load_concrete must not return a stale wider symbolic
/// object when a later store has partially overwritten its trailing
/// bytes. Setup: store sym1 (64-bit) at 0x1000, then sym2 (64-bit)
/// at 0x1004 — sym2's lower 4 bytes overwrite sym1's upper 4 bytes.
/// A subsequent load(0x1000, 8) must produce concat(sym2[31:0],
/// sym1[31:0]) for LE memory, not the entire sym1 via the
/// exact-address fast path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_partial_overlap_later_store_wins() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Pin two symbolic 64-bit values to known constants so we can
    // predict every byte after the partial overwrite.
    let k1: u128 = 0x1122_3344_5566_7788;
    let k2: u128 = 0xAABB_CCDD_EEFF_0011;
    let sym1 = RustBV::symbolic(&ctx, "sym1_3zhl", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2_3zhl", 64);
    ctx.assume_true(&sym1.eq(&RustBV::concrete(k1, 64), &ctx));
    ctx.assume_true(&sym2.eq(&RustBV::concrete(k2, 64), &ctx));

    // Store sym1 at 0x1000 (covers 0x1000..0x1008), then sym2 at
    // 0x1004 (covers 0x1004..0x100C). Bytes 0x1004..0x1008 are now
    // sym2's lower half; bytes 0x1000..0x1004 remain sym1's lower half.
    mem.store_concrete(0x1000, sym1).expect("store sym1");
    mem.store_concrete(0x1004, sym2).expect("store sym2");
    assert!(ctx.is_sat(), "context must remain SAT after both stores");

    // 8-byte load at 0x1000 must reflect both writes:
    //   bytes 0x1000..0x1004 = sym1[31:0]
    //   bytes 0x1004..0x1008 = sym2[31:0]
    // LE: result low half = sym1[31:0], result high half = sym2[31:0].
    let loaded = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte load at 0x1000 must succeed");
    let expected: u128 = ((k2 & 0xFFFF_FFFF) << 32) | (k1 & 0xFFFF_FFFF);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load(0x1000, 8) must merge sym1's low half with sym2's low half; \
         expected 0x{expected:016x}, the bug would return sym1 entire (0x{k1:016x})",
    );

    // Sanity checks for the unaffected ranges:
    // load(0x1000, 4) is sym1's low 4 bytes.
    let lower = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&lower),
        Some(k1 & 0xFFFF_FFFF),
        "load(0x1000, 4) must equal sym1's low 4 bytes"
    );
    // load(0x1004, 8) is the full sym2.
    let upper = mem
        .load_concrete(0x1004, 8, &ctx)
        .expect("8-byte load at 0x1004 must succeed");
    assert_eq!(
        ctx.eval(&upper),
        Some(k2),
        "load(0x1004, 8) must equal sym2 entire"
    );
}

/// angr-9ke6b.97: the same angr-3zhl partial-overlap merge must hold on the
/// *lazy* load path. Before `load_concrete` and `load_concrete_lazy_inner`
/// were unified behind `load_concrete_common`, the inner-overlap scan lived
/// only in the eager copy, so `load_concrete_lazy` — the path behind every
/// ITE-tree leaf and `store_concrete`'s read-modify-write — returned the
/// stale wider object with no error. Same setup as
/// `test_load_concrete_partial_overlap_later_store_wins`, loaded through
/// the lazy entry point.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_lazy_partial_overlap_later_store_wins() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k1: u128 = 0x1122_3344_5566_7788;
    let k2: u128 = 0xAABB_CCDD_EEFF_0011;
    let sym1 = RustBV::symbolic(&ctx, "sym1_97_lazy", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2_97_lazy", 64);
    ctx.assume_true(&sym1.eq(&RustBV::concrete(k1, 64), &ctx));
    ctx.assume_true(&sym2.eq(&RustBV::concrete(k2, 64), &ctx));

    mem.store_concrete(0x1000, sym1).expect("store sym1");
    mem.store_concrete(0x1004, sym2).expect("store sym2");
    assert!(ctx.is_sat(), "context must remain SAT after both stores");

    let loaded = mem
        .load_concrete_lazy(0x1000, 8, &ctx)
        .expect("8-byte lazy load at 0x1000 must succeed");
    let expected: u128 = ((k2 & 0xFFFF_FFFF) << 32) | (k1 & 0xFFFF_FFFF);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load_concrete_lazy(0x1000, 8) must merge sym1's low half with sym2's \
         low half; expected 0x{expected:016x}, the pre-unification bug returned \
         sym1 entire (0x{k1:016x})",
    );

    // The eager and lazy paths must now agree byte-for-byte on this setup.
    let eager = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte eager load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&eager),
        ctx.eval(&loaded),
        "load_concrete and load_concrete_lazy must agree after unification"
    );
}

/// angr-9ke6b.97: partial read of a wider symbolic object based at the load
/// address must work on the lazy path too. `load_concrete_lazy_inner`'s
/// exact-address fast path only accepted `sym.width() == size * 8`; a
/// narrower read fell through to the page scan and only recovered via the
/// LE-only `containing_wider_sym` tail. Sharing `load_concrete`'s
/// endianness-aware fast path fixes the big-endian case, which previously
/// extracted the wrong lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_lazy_partial_read_of_wider_sym_big_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_97_be", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));
    mem.store_concrete(0x1000, sym).expect("store sym");

    // BE: byte 0 of the stored value is the MSB, so a 4-byte read at the
    // base address is the high half (0x11223344), not the low half.
    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("4-byte lazy load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&loaded),
        Some(0x1122_3344),
        "BE load_concrete_lazy(0x1000, 4) must be the wide value's high half"
    );
    let eager = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte eager load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&eager),
        ctx.eval(&loaded),
        "load_concrete and load_concrete_lazy must agree on BE partial reads"
    );
}

/// angr-jvjf (case 1): a concrete byte store into the middle of a
/// wider symbolic object based at the same load address must not be
/// shadowed by the original wider sym. Pre-fix the exact-address
/// `symbolic_objects` fast path in `load_concrete_common` returns
/// `symbolic_objects[addr]` entire because its
/// width still matches the requested size — the concrete byte we
/// wrote at addr+3 is silently lost.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_concrete_overwrite_inner_byte_of_wider_sym_at_base() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Pin sym to a known constant so we can predict the bytes.
    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_jvjf_a", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));

    // Store the 64-bit sym at 0x1000 (covers 0x1000..0x1008).
    mem.store_concrete(0x1000, sym).expect("store sym");

    // Concrete-overwrite byte 3 (LE: byte 3 of k = 0x44) with 0xFF.
    mem.store_concrete(0x1003, RustBV::concrete(0xFF, 8))
        .expect("store concrete byte");
    assert!(ctx.is_sat(), "context must remain SAT after the stores");

    // 8-byte load at 0x1000 must reflect the concrete overwrite at
    // byte 3 — i.e. byte 3 of the loaded value is 0xFF, not 0x44.
    let loaded = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte load must succeed");
    let expected: u128 = (k & !(0xFFu128 << 24)) | (0xFFu128 << 24);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load(0x1000, 8) must reflect the concrete byte at 0x1003; \
         expected 0x{expected:016x}, the bug returns the original sym (0x{k:016x})",
    );
}

/// angr-jvjf (case 2): a concrete byte store into the middle of a
/// wider symbolic object based at an earlier address must not leave
/// the `symbolic_spans` entry stale. Pre-fix a 1-byte load at the
/// overwritten offset hits the `symbolic_spans` fast path in
/// `load_concrete_common` and returns the now-stale extract of the
/// wider sym.
#[test]
fn test_concrete_overwrite_clears_stale_symbolic_spans() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_jvjf_b", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));

    // Wider sym at 0x1000 produces span entries at 0x1001..0x1008.
    mem.store_concrete(0x1000, sym).expect("store sym");

    // Concrete-overwrite byte 3 with 0xFF (the byte covered by the
    // span 0x1003 -> (0x1000, 64)).
    mem.store_concrete(0x1003, RustBV::concrete(0xFF, 8))
        .expect("store concrete byte");

    // 1-byte load at 0x1003 must be the concrete 0xFF, not the
    // extracted sym byte 0x44.
    let byte = mem
        .load_concrete(0x1003, 1, &ctx)
        .expect("1-byte load must succeed");
    assert_eq!(
        ctx.eval(&byte),
        Some(0xFF),
        "load(0x1003, 1) must reflect the concrete overwrite; \
         the bug returns the stale extract from the wider sym"
    );
}

/// angr-7qon (case 1): a concrete write at the base of a wider sym
/// that doesn't cover the full sym width must not leave orphaned
/// page-bitmap bits for the surviving trailing bytes. Pre-fix,
/// `store_concrete`'s `symbolic_objects.remove(&addr)` cleanup
/// removes the wider sym entirely (and the spans),
/// but page.store_concrete only cleared the bitmap bits within the
/// concrete write range — so byte 0x1001 (the survivor) still has
/// its symbolic bit set with no symbolic_objects entry covering it,
/// and a subsequent 1-byte load fails with
/// "symbolic bytes not fully tracked".
#[test]
fn test_concrete_overwrite_at_base_truncates_wider_sym_byte_load_succeeds() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_a", 16);
    mem.store_concrete(0x1000, sym).expect("store 16-bit sym");

    // Concrete 1-byte write at 0x1000 truncates the wider sym (covers
    // only byte 0x1000 of the 2-byte sym at 0x1000..0x1002).
    mem.store_concrete(0x1000, RustBV::concrete(0xAA, 8))
        .expect("store concrete byte");

    // Pre-fix: load(0x1001, 1) fails because the page bitmap still
    // marks 0x1001 symbolic but no symbolic_objects entry covers it.
    // Post-fix: bitmap bit cleared, byte reclassified as concrete
    // (returns the pre-sym page byte, which is 0 here).
    let byte = mem
        .load_concrete(0x1001, 1, &ctx)
        .expect("1-byte load of survivor must succeed");
    assert_eq!(
        ctx.eval(&byte),
        Some(0),
        "load(0x1001, 1) returns the underlying concrete page byte (0); \
         pre-fix it errors with 'symbolic bytes not fully tracked'"
    );
}

/// angr-7qon (case 2): a concrete write at the base of a wider sym
/// that covers some but not all bytes of the sym (e.g. 4 bytes of a
/// 64-bit sym) must clear the page bitmap bits for the survivors
/// across the whole tail, not just immediately after `addr+size`.
#[test]
fn test_concrete_overwrite_partial_truncates_wider_sym_multibyte_tail() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_b", 64);
    mem.store_concrete(0x1000, sym).expect("store 64-bit sym");

    // Concrete 4-byte write at 0x1000 truncates the 64-bit sym, leaving
    // bytes 0x1004..0x1008 as survivors.
    mem.store_concrete(0x1000, RustBV::concrete(0xDEADBEEF, 32))
        .expect("store concrete 4 bytes");

    // Each survivor byte's page bit must be cleared; load must
    // succeed and return the page byte (0).
    for survivor in 0x1004u64..0x1008 {
        let byte = mem
            .load_concrete(survivor, 1, &ctx)
            .unwrap_or_else(|e| panic!("load(0x{survivor:x}, 1) must succeed: {e:?}"));
        assert_eq!(
            ctx.eval(&byte),
            Some(0),
            "load(0x{survivor:x}, 1) survivor byte must be concrete 0"
        );
    }

    // And a single 4-byte load over the whole tail must also succeed.
    let tail = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte tail load must succeed");
    assert_eq!(ctx.eval(&tail), Some(0));
}

/// angr-7qon (case 3): the truncation cleanup must walk pages
/// correctly when the wider sym crosses a page boundary. A 16-bit
/// sym at 0x1FFF spans pages 0 and 1; a 1-byte concrete write at
/// 0x1FFF leaves byte 0x2000 on the next page as the survivor.
#[test]
fn test_concrete_overwrite_at_base_truncates_wider_sym_crosses_page() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x2000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_c", 16);
    mem.store_concrete(0x1FFF, sym)
        .expect("store cross-page sym");

    mem.store_concrete(0x1FFF, RustBV::concrete(0xAA, 8))
        .expect("store concrete byte at base");

    let byte = mem
        .load_concrete(0x2000, 1, &ctx)
        .expect("survivor on next page must load");
    assert_eq!(
        ctx.eval(&byte),
        Some(0),
        "cross-page survivor at 0x2000 must reclassify as concrete"
    );
}

/// Regression (angr-9ke6b.98): a narrower symbolic value overwriting a wider
/// one at the same base must retire the abandoned tail. Before the fix,
/// `store_concrete`'s symbolic branch refreshed `symbolic_spans` only inside
/// the new width, so the tail bytes kept spans naming `(base, old_width)` —
/// a live object that no longer reaches them — and the page bitmap still said
/// symbolic. A later load of a tail byte followed that span into an
/// out-of-range extract and hard-failed with `SymbolicAddress`, so a readable
/// byte became an error. Mirrors the concrete branch's angr-7qon semantics:
/// the tail reclassifies as concrete.
// Reads a symbolic span back as a concrete value, which needs a real solver
// (angr-9ke6b.236, bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_narrower_sym_overwrite_at_base_truncates_wider_sym() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let wide = RustBV::symbolic(&ctx, "sym_98_wide", 128);
    mem.store_concrete(0x1000, wide).expect("store 16-byte sym");

    let narrow = RustBV::symbolic(&ctx, "sym_98_narrow", 64);
    mem.store_concrete(0x1000, narrow.clone())
        .expect("store 8-byte sym at same base");

    // The abandoned tail [0x1008, 0x1010) must load cleanly, not raise
    // MemoryError::SymbolicAddress ("symbolic bytes not fully tracked").
    let tail = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("abandoned tail must load, not hard-fail");
    assert_eq!(
        ctx.eval(&tail),
        Some(0),
        "abandoned tail must reclassify as concrete (angr-7qon parity)"
    );

    // Per-byte too: the first tail byte is the one whose stale span pointed
    // back at the still-live base.
    let tail_byte = mem
        .load_concrete(0x1008, 1, &ctx)
        .expect("first abandoned tail byte must load");
    assert_eq!(ctx.eval(&tail_byte), Some(0));

    // The surviving head must still be the narrow symbolic value.
    let head = mem.load_concrete(0x1000, 8, &ctx).expect("head must load");
    assert!(head.is_symbolic(), "head must stay symbolic");
    ctx.assume_true(&narrow.eq(&RustBV::concrete(0x1122_3344_5566_7788, 64), &ctx));
    assert_eq!(
        ctx.eval(&head),
        Some(0x1122_3344_5566_7788),
        "head must read back the narrow value that overwrote the wide one"
    );
}
