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
