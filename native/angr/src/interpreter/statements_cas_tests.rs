//! End-to-end `IRStmt::CAS` dispatch tests (extracted from
//! `statements_tests.rs` by angr-fs8kb.96).
//!
//! Everything here drives `execute_stmt_with_callbacks` with a real
//! `IRStmt::CAS` rather than reaching into `execute_cas_stmt`'s helpers, so the
//! whole path is covered: operand validation, half-type derivation, the
//! `addr + sizeof(half)` high-half address synthesis, the combined comparison,
//! the writeback, and the oldLo/oldHi temp assignments. Memory is the
//! `all_flushed_stores` layer (a load with no Python memory callback reads it)
//! and the writeback lands in `pending_stores`, same as any concrete store.
//!
//! The *unit*-level CAS-store tests (`cas_store_symbolic_data`,
//! `cas_writeback`'s no-callback policies) stay in `statements_tests.rs`
//! alongside the other `handle_symbolic_store` cases they share setup with.
//! Shared helpers live in `statements_tests_support.rs`.

use super::*;
use crate::interpreter::statements_tests_support::{make_irsb_with_temps, new_interp, with_python};
use crate::vex::ir::{Endness, IRType};


/// Build a CAS statement over I32 halves at a concrete I64 address. `*_hi`
/// all-`Some` selects DCAS, all-`None` single CAS; a mixed pair is exactly the
/// malformed-IR case `execute_cas_stmt` validates.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors IRStmt::CAS's own operand list"
)]
fn cas_stmt(
    old_hi: Option<u32>,
    old_lo: u32,
    addr: u64,
    expd_hi: Option<u32>,
    expd_lo: u32,
    data_hi: Option<u32>,
    data_lo: u32,
    endness: Endness,
) -> IRStmt {
    IRStmt::CAS {
        old_hi,
        old_lo,
        addr: Box::new(IRExpr::Const(IRConst::U64(addr))),
        expdHi: expd_hi.map(|v| Box::new(IRExpr::Const(IRConst::U32(v)))),
        expdLo: Box::new(IRExpr::Const(IRConst::U32(expd_lo))),
        dataHi: data_hi.map(|v| Box::new(IRExpr::Const(IRConst::U32(v)))),
        dataLo: Box::new(IRExpr::Const(IRConst::U32(data_lo))),
        endness,
    }
}

/// An interpreter with two I32 temps and 4-byte words pre-seeded at each
/// `(addr, value)` in the flushed-store layer.
fn interp_with_words<'c>(ctx: &'c SymContext, words: &[(u64, u32)]) -> VEXInterpreter<'c> {
    let mut interp = new_interp(ctx);
    for (addr, value) in words {
        interp
            .all_flushed_stores
            .insert(*addr, value.to_le_bytes().to_vec());
    }
    interp.temps.resize(2, None);
    interp
}

fn temp_u64(interp: &VEXInterpreter<'_>, tmp: usize) -> Option<u64> {
    interp.temps[tmp].as_ref().and_then(|bv| bv.as_u64())
}

fn stored_word(interp: &VEXInterpreter<'_>, addr: u64) -> Option<u32> {
    interp
        .pending_stores
        .try_load_assembled(addr, 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4-byte word")))
}

#[test]
fn cas_single_match_swaps_and_writes_old_lo() {
    let ctx = SymContext::new_mock();
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        None,
        0,
        0x5000,
        None,
        0x1111_1111,
        None,
        0xdead_beef,
        Endness::Little,
    );
    with_python(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("single CAS");
        assert!(matches!(res, StmtResult::Continue));
    });
    assert_eq!(
        temp_u64(&interp, 0),
        Some(0x1111_1111),
        "oldLo must receive the value read from memory"
    );
    assert_eq!(
        stored_word(&interp, 0x5000),
        Some(0xdead_beef),
        "a matching comparison must store dataLo"
    );
}

#[test]
fn cas_single_mismatch_leaves_memory_untouched() {
    let ctx = SymContext::new_mock();
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x2222_2222)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        None,
        0,
        0x5000,
        None,
        0x1111_1111,
        None,
        0xdead_beef,
        Endness::Little,
    );
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("single CAS");
    });
    assert_eq!(
        temp_u64(&interp, 0),
        Some(0x2222_2222),
        "oldLo must still report the (unchanged) memory value"
    );
    assert!(
        interp.pending_stores.is_empty(),
        "a failed comparison must not store anything"
    );
}

#[test]
fn cas_dcas_match_swaps_both_halves() {
    let ctx = SymContext::new_mock();
    // High half lives at addr + sizeof(I32); the CAS handler synthesizes that
    // address itself from the low address and the expdLo type.
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111), (0x5004, 0x3333_3333)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        Some(1),
        0,
        0x5000,
        Some(0x3333_3333),
        0x1111_1111,
        Some(0xcafe_f00d),
        0xdead_beef,
        Endness::Little,
    );
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("DCAS");
    });
    assert_eq!(temp_u64(&interp, 0), Some(0x1111_1111), "oldLo");
    assert_eq!(temp_u64(&interp, 1), Some(0x3333_3333), "oldHi");
    assert_eq!(stored_word(&interp, 0x5000), Some(0xdead_beef), "dataLo");
    assert_eq!(stored_word(&interp, 0x5004), Some(0xcafe_f00d), "dataHi");
}

#[test]
fn cas_dcas_high_half_mismatch_stores_neither_half() {
    let ctx = SymContext::new_mock();
    // Low half matches, high half does not: the comparison is the *conjunction*
    // of both halves, so neither store may happen.
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111), (0x5004, 0x9999_9999)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        Some(1),
        0,
        0x5000,
        Some(0x3333_3333),
        0x1111_1111,
        Some(0xcafe_f00d),
        0xdead_beef,
        Endness::Little,
    );
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("DCAS");
    });
    assert_eq!(temp_u64(&interp, 0), Some(0x1111_1111), "oldLo");
    assert_eq!(temp_u64(&interp, 1), Some(0x9999_9999), "oldHi");
    assert!(
        interp.pending_stores.is_empty(),
        "one mismatching half must veto both stores"
    );
}

#[test]
fn cas_mixed_some_none_operands_are_invalid_ir() {
    let ctx = SymContext::new_mock();
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    // Every mixed shape the guest lifter could hand us: oldHi without the
    // expdHi/dataHi that make it meaningful, and each of the value operands
    // present without oldHi to receive the loaded high half.
    let mixed = [
        cas_stmt(Some(1), 0, 0x5000, None, 1, None, 2, Endness::Little),
        cas_stmt(None, 0, 0x5000, Some(3), 1, None, 2, Endness::Little),
        cas_stmt(None, 0, 0x5000, None, 1, Some(4), 2, Endness::Little),
        cas_stmt(Some(1), 0, 0x5000, Some(3), 1, None, 2, Endness::Little),
    ];
    with_python(|cb| {
        for stmt in &mixed {
            let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111)]);
            // `StmtResult` has no `Debug`, so unwrap the error by hand
            // rather than via `expect_err`.
            let Err(err) = interp.execute_stmt_with_callbacks(cb, stmt, &irsb) else {
                panic!("mixed Some/None CAS operands must be rejected");
            };
            assert!(
                matches!(err, CbExecutionError::InvalidIR(_)),
                "expected InvalidIR, got {err:?}"
            );
            assert!(
                interp.pending_stores.is_empty(),
                "a rejected CAS must not have touched memory"
            );
        }
    });
}

#[test]
fn cas_big_endian_dcas_is_unsupported() {
    let ctx = SymContext::new_mock();
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111), (0x5004, 0x3333_3333)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        Some(1),
        0,
        0x5000,
        Some(0x3333_3333),
        0x1111_1111,
        Some(0xcafe_f00d),
        0xdead_beef,
        Endness::Big,
    );
    with_python(|cb| {
        let Err(err) = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb) else {
            panic!("BE DCAS must defer to Python rather than guess a layout");
        };
        let CbExecutionError::Unsupported(msg) = &err else {
            panic!("expected Unsupported, got {err:?}");
        };
        // angr-fs8kb.59: the manager string-matches this marker to bump
        // `dcas_unsupported_count` and emit its warning. This is the only
        // producer of a DCAS fallback, so a message without the marker leaves
        // that counter permanently dead.
        assert!(
            msg.contains(DCAS_UNSUPPORTED_REASON),
            "the BE-DCAS message must carry the marker the manager counts on, got {msg:?}"
        );
    });
    assert!(
        interp.pending_stores.is_empty(),
        "the BE-DCAS rejection must happen before any store"
    );
}

/// Only *double* CAS is rejected for big-endian memory — a single CAS under
/// `Iend_BE` must still execute. Byte order is deliberately not asserted here:
/// the Rust-side buffered memory layers (`pending_stores` /
/// `all_flushed_stores`) assemble bytes without consulting `endness` at all
/// (see `eval_load`, which forwards it only to the mem-read inspect hook), so
/// pinning a byte order would pin an unrelated layer's behaviour rather than
/// the CAS handler's. The seeded word is a byte-palindrome so the comparison
/// is well-defined either way.
#[test]
fn cas_single_big_endian_still_swaps() {
    let ctx = SymContext::new_mock();
    let mut interp = interp_with_words(&ctx, &[(0x5000, 0x1111_1111)]);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32, IRType::I32]);
    let stmt = cas_stmt(
        None,
        0,
        0x5000,
        None,
        0x1111_1111,
        None,
        0xdead_beef,
        Endness::Big,
    );
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("single BE CAS must not hit the DCAS-only BE rejection");
    });
    assert_eq!(temp_u64(&interp, 0), Some(0x1111_1111), "oldLo");
    assert_eq!(
        interp
            .pending_stores
            .try_load_assembled(0x5000, 4)
            .map(|b| b.len()),
        Some(4),
        "a matching BE single CAS must still store its 4-byte dataLo"
    );
}
