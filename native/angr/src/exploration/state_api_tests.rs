// Tests for exploration/state_api.rs (see rust-mod-tests-sibling-extraction
// for why these live in a sibling file rather than an inline `mod tests`).
use super::*;
use crate::concretize::AddressConcretizer;
use crate::memory::Permission;
use crate::stash::STASH_FOUND;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Build a state whose bytes at `0x1000` live in Multi cells: a
/// symbolic-address store resolving to `Multiple` is routed through
/// `install_multi_for_candidates_safe`, which leaves the value in
/// `multi_objects` rather than in the page's `symbolic_bitmap`.
#[cfg(feature = "vex-engine-z3")]
fn state_with_multi_cell_store(value: u32) -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x1000, 0x4000, Permission::RWX);

    let (addr_var, val_bv) = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "export_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        (addr_var, RustBV::concrete(value as u128, 32))
    };

    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies: the default
    // Max-only chain concretizes an unannotated symbolic address to a single
    // value and installs no Multi cells (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let ctx = state.solver().clone();
    let ctx = ctx.borrow();
    state
        .memory_mut()
        .store_symbolic_unified(addr_var, val_bv, &ctx, &concretizer)
        .expect("Multi store must succeed");
    drop(ctx);

    assert_ne!(
        state.memory().multi_cell_count(),
        0,
        "precondition: the store must have installed Multi cells"
    );
    state
}

/// `symbolic_offsets` of the snapshot page starting at `page_addr`.
#[cfg(feature = "vex-engine-z3")]
fn symbolic_offsets_at(snap: &crate::state::ExplorationStateSnapshot, page_addr: u64) -> Vec<u16> {
    let index = snap
        .page_addresses()
        .iter()
        .position(|&a| a == page_addr)
        .expect("page present in snapshot");
    snap.get_page(index).expect("page by index").3
}

/// Regression (angr-9ke6b.101): `export_found_states` — the primary
/// `explore(find=...)` result API — must flush Multi cells, otherwise the
/// stored bytes vanish from the snapshot because `export_full` reads only
/// `symbolic_bitmap` via `MemoryPage::symbolic_offsets`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn export_found_states_flushes_multi_cells() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.sm
        .push(STASH_FOUND, state_with_multi_cell_store(0xCAFE_BABE));

    let snaps = mgr._export_found_states();
    assert_eq!(snaps.len(), 1);

    let symbolic_offsets = symbolic_offsets_at(&snaps[0], 0x1000);
    for off in 0..4u16 {
        assert!(
            symbolic_offsets.contains(&off),
            "byte 0x{:x} must export as symbolic after the Multi flush, \
             got symbolic_offsets={symbolic_offsets:?}",
            0x1000 + off,
        );
    }
    assert_eq!(
        mgr.sm
            .get(STASH_FOUND)
            .and_then(|s| s.front())
            .map(|s| s.memory().multi_cell_count()),
        Some(0),
        "the flush must have drained the stashed state's Multi cells"
    );
}

/// Regression (angr-sqfj8.52): `_get_state_symbolic_z3_asts` iterated
/// `symbolic_objects_iter()` directly without flushing first, so a
/// Multi-covered byte was silently absent from the exported Z3-AST list —
/// the same shape as the two Multi-flush regressions above.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn get_state_symbolic_z3_asts_flushes_multi_cells() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = state_with_multi_cell_store(0xCAFE_BABE);
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);

    let asts = mgr
        ._get_state_symbolic_z3_asts(state_id)
        .expect("state found");
    assert!(
        asts.iter().any(|&(addr, _, _)| addr == 0x1000),
        "unflushed export dropped the Multi-cell byte at 0x1000: {asts:?}"
    );
}

/// Same contract for the single-state path behind `mgr.get_state_by_id()`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn export_state_flushes_multi_cells() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = state_with_multi_cell_store(0x1234_5678);
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);

    let snap = mgr._export_state(state_id).expect("state found");
    let symbolic_offsets = symbolic_offsets_at(&snap, 0x1000);
    assert!(
        (0..4u16).all(|off| symbolic_offsets.contains(&off)),
        "unflushed export dropped the Multi-cell bytes: {symbolic_offsets:?}"
    );
}

/// `_get_state_register`'s `None` is deliberately two-valued — an unmodeled
/// register name and a symbolic value are indistinguishable — while the
/// mirrored `_get_pending_register` raises for the first case. Pin all three
/// outcomes (concrete / symbolic / unmodeled) plus the batch form, so the
/// asymmetry documented on `_get_state_register` stays a decision rather than
/// drifting into an accident (angr-sqfj8.56).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn get_state_register_folds_unmodeled_name_and_symbolic_into_none() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");

    assert!(state.set_register("rax", RustBV::concrete(0x2a, 64)));
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    assert!(state.set_register("rbx", sym));
    // `ymm0` is one of the names angr's `arch.registers` carries but Rust's
    // `RegisterFile` does not model (invariant I5 on `set_registers_bulk`);
    // it is what makes the fold load-bearing for `RustRegisterProxy`.
    assert!(
        state.arch().register_offset("ymm0").is_none(),
        "precondition: ymm0 must be unmodeled for this test to mean anything",
    );

    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);

    assert_eq!(
        mgr._get_state_register(state_id, "rax").expect("rax read"),
        Some(0x2a),
    );
    assert_eq!(
        mgr._get_state_register(state_id, "rbx").expect("rbx read"),
        None,
        "symbolic register reads as None",
    );
    assert_eq!(
        mgr._get_state_register(state_id, "ymm0")
            .expect("ymm0 read"),
        None,
        "unmodeled register name reads as None rather than raising",
    );

    // The batch form must agree element-wise, and an unmodeled name must not
    // discard its siblings' values.
    let names = vec!["rax".to_string(), "ymm0".to_string(), "rbx".to_string()];
    assert_eq!(
        mgr._get_state_registers_batch(state_id, names)
            .expect("batch read"),
        vec![Some(0x2a), None, None],
    );
}

/// The fold above is Python-facing only: inside Rust the two `None` causes are
/// distinct `RegisterU128` variants, so a future caller can tell an unmodeled
/// name from a symbolic value without re-deriving it (angr-91vj9.9).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn read_register_u128_names_both_causes_of_none() {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    assert!(state.set_register("rax", RustBV::concrete(0x2a, 64)));
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    assert!(state.set_register("rbx", sym));

    assert!(matches!(
        read_register_u128(&state, "rax"),
        RegisterU128::Concrete(0x2a)
    ));
    assert!(matches!(
        read_register_u128(&state, "rbx"),
        RegisterU128::Symbolic
    ));
    assert!(matches!(
        read_register_u128(&state, "ymm0"),
        RegisterU128::UnknownName
    ));
    // ... and all three still collapse to the documented Python contract.
    assert_eq!(read_register_u128(&state, "rax").into_option(), Some(0x2a));
    assert_eq!(read_register_u128(&state, "rbx").into_option(), None);
    assert_eq!(read_register_u128(&state, "ymm0").into_option(), None);
}

/// A state with one 64-bit symbolic object at `0x1234` — page 1, in-page
/// offset 0x234 — inside a mapped page, so `_state_symbolic_info` has all
/// three of its inputs (`symbolic_objects`, the page table, the page's
/// `symbolic_bitmap`) populated with distinguishable values.
#[cfg(feature = "vex-engine-z3")]
fn state_with_symbolic_object_at_0x1234() -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x1000, 0x1000, Permission::RW);
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "sym_info_probe", 64)
    };
    state
        .memory_mut()
        .import_symbolic_value(0x1234, sym, None)
        .expect("64-bit symbolic import must be accepted");
    state
}

/// `_state_symbolic_info` is debug-only output, but its bit math is the same
/// address→(page, offset) decomposition the memory model uses, so a page-size
/// or `is_symbolic(offset)` change must not silently turn it into a liar
/// (angr-03vl4.26). Pins the symbolic address, a concrete byte in the *same*
/// mapped page, and an unmapped page.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn state_symbolic_info_reports_symbolic_and_concrete_addresses() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = state_with_symbolic_object_at_0x1234();
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);

    assert_eq!(
        mgr._state_symbolic_info(state_id, 0x1234)
            .expect("state found"),
        "total_sym_objs=1 at_0x1234=Some(64) page=mapped sym_at_offset=true",
    );
    // 0x1240 is past the 8 bytes the import marked, so the page is still
    // mapped but the offset is concrete and no object starts there.
    assert_eq!(
        mgr._state_symbolic_info(state_id, 0x1240)
            .expect("state found"),
        "total_sym_objs=1 at_0x1240=None page=mapped sym_at_offset=false",
    );
    assert_eq!(
        mgr._state_symbolic_info(state_id, 0x9234)
            .expect("state found"),
        "total_sym_objs=1 at_0x9234=None page=unmapped",
    );
}

/// The page/offset split must use *both* halves of the address: `0x9234`
/// above shares 0x1234's in-page offset but lives in an unmapped page, and
/// `0x1235` shares 0x1234's page while landing on a different — here still
/// symbolic — offset. Dropping either half would make one of these agree with
/// the 0x1234 answer.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn state_symbolic_info_uses_both_page_and_offset() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = state_with_symbolic_object_at_0x1234();
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);

    // Same page, offset 0x235: covered by the 8-byte import, but no symbolic
    // object *starts* there — the two sources of "symbolic" are independent.
    assert_eq!(
        mgr._state_symbolic_info(state_id, 0x1235)
            .expect("state found"),
        "total_sym_objs=1 at_0x1235=None page=mapped sym_at_offset=true",
    );
    // A state with no symbolic memory at all reports the empty case rather
    // than erroring.
    let empty = RustSimState::new("amd64").expect("amd64 state");
    let empty_id = empty.state_id();
    mgr.sm.push(STASH_FOUND, empty);
    assert_eq!(
        mgr._state_symbolic_info(empty_id, 0x1234)
            .expect("state found"),
        "total_sym_objs=0 at_0x1234=None page=unmapped",
    );
}
