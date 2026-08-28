//! Phase 4.2 (angr-mmdh.2): `flush_multi_cells` run coalescing — including
//! its stop at the address-space wrap, the run-length cap, and the
//! `cond_fingerprint` key that decides which bytes may join one run.

use super::*;

// ============================================================================
// Phase 4.2 (angr-mmdh.2): flush_multi_cells run coalescing
// ============================================================================

/// A LE 4-byte symbolic-address store at two candidates installs 8 Multi
/// bytes (4 per candidate). After flush, those should coalesce into TWO
/// wider symbolic_objects (one per candidate base), each width 32, with
/// symbolic_spans covering the interior bytes — instead of 8 per-byte
/// entries. This is the export-cost win called out in `phase41-bottleneck`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase42_flush_coalesces_le_multi_byte_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_le_addr", 64);
    let value = RustBV::concrete(0xDEAD_BEEF, 32);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 8);

    mem.flush_multi_cells(&ctx);
    assert_eq!(mem.multi_cell_count(), 0);

    // Exactly two wider symbolic_objects entries: 0x1000 and 0x2000 with
    // width 32. The interior bytes (0x1001..0x1003 / 0x2001..0x2003) must
    // NOT have their own symbolic_objects entry — they go through the
    // symbolic_spans reverse index.
    assert_eq!(
        mem.symbolic_object_count(),
        2,
        "coalesced flush must yield one wider entry per candidate"
    );
    let sym_a = mem.get_symbolic_object(0x1000).expect("entry at A start");
    let sym_b = mem.get_symbolic_object(0x2000).expect("entry at B start");
    assert_eq!(sym_a.width(), 32, "wider entry covers all 4 bytes");
    assert_eq!(sym_b.width(), 32, "wider entry covers all 4 bytes");
    for off in 1..4 {
        assert!(
            mem.get_symbolic_object(0x1000 + off).is_none(),
            "interior bytes must not get their own symbolic_objects entry"
        );
        assert!(
            mem.get_symbolic_object(0x2000 + off).is_none(),
            "interior bytes must not get their own symbolic_objects entry"
        );
    }

    // Behavior check: load 4 bytes at each candidate under each
    // concretization. The wider symbolic_object must hold the original
    // value under cond=true and the concrete default (0) under cond=false.
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    let loaded_a = mem.load_concrete(0x1000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_a), Some(0xDEAD_BEEF));
    let loaded_b_under_a = mem.load_concrete(0x2000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_b_under_a), Some(0));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    let loaded_b = mem.load_concrete(0x2000, 4, &probe_b).unwrap();
    assert_eq!(probe_b.eval(&loaded_b), Some(0xDEAD_BEEF));
}

/// Big-endian variant of the LE coalesce test. byte 0 (lowest addr) is
/// the MSB in BE; the wider value's right-fold must reconstruct the
/// original word.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase42_flush_coalesces_be_multi_byte_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_be_addr", 64);
    let value = RustBV::concrete(0xCAFE_BABE, 32);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    mem.flush_multi_cells(&ctx);

    assert_eq!(mem.symbolic_object_count(), 2);
    let sym_a = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(sym_a.width(), 32);

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    let loaded_a = mem.load_concrete(0x1000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_a), Some(0xCAFE_BABE));
}

/// Single-byte stores produce non-adjacent Multi cells. The flush path
/// must fall back to the per-byte case and remain byte-identical to the
/// pre-Phase-4.2 behaviour.
#[test]
fn test_phase42_flush_singleton_no_coalesce() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_single_addr", 64);
    let value = RustBV::concrete(0xAB, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 2);

    mem.flush_multi_cells(&ctx);

    // Two per-byte entries, width 8 each. No spans installed.
    assert_eq!(mem.symbolic_object_count(), 2);
    let sym_a = mem.get_symbolic_object(0x1000).unwrap();
    let sym_b = mem.get_symbolic_object(0x2000).unwrap();
    assert_eq!(sym_a.width(), 8);
    assert_eq!(sym_b.width(), 8);
}

/// Address-space wrap guard (angr-0jh0j.36): a Multi cell at the last byte
/// of the address space and one at byte 0 are logically unrelated, even when
/// they share a cond. Coalescing them would mint a single width-16 object
/// spanning both ends of the 64-bit space. Both payloads carry the *same*
/// `MultiAlternative` clone, so the cond fingerprints match and the adjacency
/// test is the only thing that can terminate the run.
///
/// Honest scope: this passes under the pre-fix wrapping adjacency spelling
/// too, because the ascending sort in `flush_multi_cells` puts `u64::MAX`
/// last and there is no `entries[j]` after it. It pins the *outcome* at the
/// boundary — so a future reordering of that scan (or a switch to an
/// unsorted iteration order) reds here instead of silently fusing the two
/// ends of the address space.
#[test]
fn test_phase42_flush_no_coalesce_across_address_space_wrap() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);

    let addr_var = RustBV::symbolic(&ctx, "p42_wrap_addr", 64);
    let alt = make_alt(&ctx, &addr_var, 0x1000, 0xAB);
    mem.set_multi_alternatives(u64::MAX, MultiPayload::from_alternatives(vec![alt.clone()]));
    mem.set_multi_alternatives(0u64, MultiPayload::from_alternatives(vec![alt]));
    assert_eq!(mem.multi_cell_count(), 2);

    mem.flush_multi_cells(&ctx);

    assert_eq!(
        mem.symbolic_object_count(),
        2,
        "wrapped neighbours must flush as two per-byte entries"
    );
    assert_eq!(
        mem.get_symbolic_object(u64::MAX)
            .expect("entry at the last byte")
            .width(),
        8,
        "top-of-space byte must stay width 8"
    );
    assert_eq!(
        mem.get_symbolic_object(0u64)
            .expect("entry at byte 0")
            .width(),
        8,
        "byte 0 must stay width 8"
    );
}

/// Bytes whose Multi payloads carry mismatched cond fingerprints (e.g.
/// one byte has an extra alternative from a later store) must terminate
/// the coalesce run. Adjacent bytes either side are still coalesced
/// pairwise within their matching subsets.
#[test]
fn test_phase42_flush_fingerprint_mismatch_breaks_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    // First store: 4-byte value at two candidates → 4 Multi bytes per
    // candidate.
    let addr1 = RustBV::symbolic(&ctx, "p42_addr1", 64);
    mem.store_concrete_multi(&addr1, &RustBV::concrete(0x1111_2222, 32), &[0x1000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 4);

    // Second store: 1-byte value at byte 2 of the previous range. This
    // appends a second alternative to ONLY byte 0x1002, breaking its
    // fingerprint relative to its neighbours.
    let addr2 = RustBV::symbolic(&ctx, "p42_addr2", 64);
    mem.store_concrete_multi(&addr2, &RustBV::concrete(0xFF, 8), &[0x1002], &ctx)
        .unwrap();
    assert_eq!(
        mem.get_multi_alternatives(0x1002).unwrap().len(),
        2,
        "the merged-store byte carries 2 alternatives"
    );
    assert_eq!(
        mem.get_multi_alternatives(0x1001).unwrap().len(),
        1,
        "neighbour byte still has 1 alternative"
    );

    mem.flush_multi_cells(&ctx);

    // Expected runs:
    //   [0x1000, 0x1001] — coalesced wider entry, width 16.
    //   [0x1002]         — singleton, width 8.
    //   [0x1003]         — singleton, width 8.
    assert_eq!(mem.symbolic_object_count(), 3);
    let coalesced = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(coalesced.width(), 16, "first run coalesces 2 bytes");
    assert!(
        mem.get_symbolic_object(0x1001).is_none(),
        "interior of coalesced run not in symbolic_objects"
    );
    let singleton_mid = mem.get_symbolic_object(0x1002).unwrap();
    let singleton_tail = mem.get_symbolic_object(0x1003).unwrap();
    assert_eq!(singleton_mid.width(), 8);
    assert_eq!(singleton_tail.width(), 8);
}

/// A run wider than `COALESCE_MAX_RUN` (16 bytes) must coalesce only the
/// first 16 bytes; the remaining bytes flush as a separate run.
#[test]
fn test_phase42_flush_run_length_cap() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    // 24-byte value at a single candidate → 24 adjacent Multi bytes
    // sharing one cond. The cap forces two coalesced runs: 16 bytes and
    // 8 bytes (instead of one wider run of 24).
    let addr_var = RustBV::symbolic(&ctx, "p42_cap_addr", 64);
    // RustBV::concrete masks the u128 to the declared width, so a
    // 24-byte (192-bit) value can be passed directly. But we only need
    // a sequence of 24 Multi bytes; the value content does not matter
    // for the coalescing logic. Use a width-192 concrete BV.
    let value = RustBV::concrete(0x1234_5678_DEAD_BEEF, 192);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 24);

    mem.flush_multi_cells(&ctx);
    // Two wider entries: one at 0x1000 (width 128 = 16 bytes), one at
    // 0x1010 (width 64 = 8 bytes).
    assert_eq!(mem.symbolic_object_count(), 2);
    let first = mem.get_symbolic_object(0x1000).unwrap();
    let second = mem.get_symbolic_object(0x1010).unwrap();
    assert_eq!(first.width(), 128);
    assert_eq!(second.width(), 64);
}

/// `cond_fingerprint`'s `Expression` key mixes the op in alongside the
/// operands `Arc` pointer (angr-03vl4.38). Under today's construction every
/// `Expression` allocates a fresh operands `Arc`, so a shared pointer already
/// implies a shared op — this pins the defensive half, so a future
/// construction path that reuses an operands `Arc` across two different ops
/// cannot silently coalesce bytes whose conds are not equivalent.
#[test]
fn test_cond_fingerprint_distinguishes_op_on_shared_operands() {
    use crate::memory::multi::cond_fingerprint;
    use crate::symbolic::BVOp;
    use std::sync::Arc;

    let operands: Arc<[RustBV]> =
        Arc::from(vec![RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)]);
    let make = |op: BVOp| RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: 1,
        op,
        operands: Arc::clone(&operands),
        memo: Default::default(),
    };

    // Same Arc, different op → distinct fingerprints.
    assert_ne!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&make(BVOp::Ne)),
        "op must participate in the Expression fingerprint"
    );
    // Payload-carrying variants must not collapse onto their discriminant.
    assert_ne!(
        cond_fingerprint(&make(BVOp::Extract(7, 0))),
        cond_fingerprint(&make(BVOp::Extract(15, 8))),
        "op payload must participate too"
    );
    // The coalescing case itself is unaffected: same op + same Arc matches,
    // which is what a run of bytes cloned from one cond looks like.
    assert_eq!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&make(BVOp::Eq)),
        "clones of one cond must still coalesce"
    );
    // A fresh Arc with the same op is a different store path → no coalesce.
    let other_operands: Arc<[RustBV]> =
        Arc::from(vec![RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)]);
    assert_ne!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&RustBV::Expression {
            id: RustBV::EXPRESSION_ID,
            width: 1,
            op: BVOp::Eq,
            operands: other_operands,
            memo: Default::default(),
        }),
        "distinct operands allocations stay distinct"
    );
}
