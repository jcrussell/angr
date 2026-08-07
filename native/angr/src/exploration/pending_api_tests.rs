// Tests for exploration/pending_api.rs.
use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use rustc_hash::FxHashMap;

/// Wrap an already-built state in a minimal Error-reason PendingCallback
/// (no solver fork / deferred forks), so lineage walks can be exercised
/// purely at the Rust unit-test level.
fn pending_for(state: RustSimState) -> PendingCallback {
    PendingCallback {
        state,
        pre_callback_snapshot: None,
        reason: CallbackReason::Error {
            message: "pending".to_string(),
        },
        jumpkind: None,
        solver_ctx: None,
        deferred_forks: Vec::new(),
        stored_conditions: FxHashMap::default(),
        fork_snapshots: FxHashMap::default(),
    }
}

/// A >=3-level fork chain A->B->C->D must report its INTERMEDIATE ancestors,
/// not just [D, C, A] (angr-ph300.25). Python's cache walks in
/// rust_callback_dispatch.py key on exactly these ids; dropping B made a
/// callback on D fall back to root A's stale snapshot.
///
/// Also pins the two halves of the walk: an ancestor held in a stash (C, A)
/// and one held as another pending callback (B) are both followed.
#[test]
fn pending_ancestry_includes_intermediate_ancestors() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let c = b.fork();
    let d = c.fork();
    let (a_id, b_id, c_id, d_id) = (a.state_id(), b.state_id(), c.state_id(), d.state_id());

    mgr.sm.push(STASH_ACTIVE, a);
    mgr.pending_callbacks
        .insert(StateId::new(b_id), pending_for(b));
    mgr.sm.push(STASH_ACTIVE, c);
    mgr.pending_callbacks
        .insert(StateId::new(d_id), pending_for(d));
    mgr.sm.set_root(d_id, a_id);

    let ancestry = mgr._get_pending_ancestry(d_id).expect("ancestry");
    assert_eq!(
        ancestry,
        vec![d_id, c_id, b_id, a_id],
        "full chain expected, with no duplicate root append"
    );
}

/// The walk is best-effort: each parent id it learns is recorded, but it stops
/// once the ancestor it would have to read next is no longer held (forks
/// consume their parent). The lineage root is still appended, so the result is
/// never shorter than the old [state, parent, root] behaviour.
#[test]
fn pending_ancestry_stops_at_dropped_ancestor_but_keeps_root() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let c = b.fork();
    let d = c.fork();
    let (a_id, c_id, d_id) = (a.state_id(), c.state_id(), d.state_id());

    // C is gone, so C's id is still recorded (D names it) but B is unreachable.
    mgr.sm.push(STASH_ACTIVE, a);
    drop(b);
    drop(c);
    mgr.pending_callbacks
        .insert(StateId::new(d_id), pending_for(d));
    mgr.sm.set_root(d_id, a_id);

    let ancestry = mgr._get_pending_ancestry(d_id).expect("ancestry");
    assert_eq!(ancestry, vec![d_id, c_id, a_id]);
}

/// With no lineage root recorded and no ancestors held, the result degrades to
/// [state, parent] — the pre-existing single-hop shape.
#[test]
fn pending_ancestry_without_root_is_state_plus_parent() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let (a_id, b_id) = (a.state_id(), b.state_id());
    drop(a);

    mgr.pending_callbacks
        .insert(StateId::new(b_id), pending_for(b));

    let ancestry = mgr._get_pending_ancestry(b_id).expect("ancestry");
    assert_eq!(ancestry, vec![b_id, a_id]);
}

/// GAP-6 regression (angr-9ke6b.47): two nested zero-length hooks at the SAME
/// address push two skip tokens, and each occurrence must consume exactly one.
/// The old `retain(|&(addr, _)| addr != pc)` collapsed both in a single call,
/// so the second occurrence found no token, re-fired the hook, and re-armed
/// the very infinite loop GAP 6 exists to prevent.
#[test]
fn consume_skip_hook_pops_one_entry_per_nested_occurrence() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr._set_skip_hook_addr(0x400123);
    mgr._set_skip_hook_addr(0x400123);
    assert_eq!(mgr.skip_hook_stack.len(), 2, "no dedup on push");

    assert!(mgr.consume_skip_hook(0x400123), "first occurrence skips");
    assert_eq!(
        mgr.skip_hook_stack.len(),
        1,
        "exactly one token consumed, not all matches"
    );
    assert!(
        mgr.consume_skip_hook(0x400123),
        "second nested occurrence still has its own token"
    );
    assert!(mgr.skip_hook_stack.is_empty());
    assert!(
        !mgr.consume_skip_hook(0x400123),
        "stack drained — hook fires normally again"
    );
}

/// A token for a different address is untouched by consumption at `pc`, and an
/// unmatched consume is a no-op rather than a blanket clear.
#[test]
fn consume_skip_hook_leaves_other_addresses_alone() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr._set_skip_hook_addr(0x400123);
    mgr._set_skip_hook_addr(0x400456);

    assert!(!mgr.consume_skip_hook(0x400789), "no token for this pc");
    assert_eq!(mgr.skip_hook_stack.len(), 2);

    assert!(mgr.consume_skip_hook(0x400123));
    assert_eq!(
        mgr.skip_hook_stack,
        vec![(0x400456, mgr.steps + 2)],
        "only the matching entry is popped"
    );
}

/// Expired entries are dropped before the match, so a token that outlived its
/// two-step window never suppresses a later, legitimate hook hit.
#[test]
fn consume_skip_hook_drops_expired_entries() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr._set_skip_hook_addr(0x400123); // expiry = steps + 2
    mgr.steps += 2;
    assert!(
        !mgr.consume_skip_hook(0x400123),
        "expired token must not skip"
    );
    assert!(mgr.skip_hook_stack.is_empty(), "expired entry pruned");
}

/// Build a state whose bytes at `0x1000` live in Multi cells: a
/// symbolic-address store resolving to `Multiple` is routed through
/// `store_symbolic_unified`, which leaves the value in `multi_objects`
/// rather than the page's `symbolic_bitmap`. Mirrors
/// `state_api_tests::state_with_multi_cell_store`.
#[cfg(feature = "vex-engine-z3")]
fn pending_state_with_multi_cell_store(value: u32) -> RustSimState {
    use crate::concretize::AddressConcretizer;
    use crate::memory::Permission;
    use crate::symbolic::RustBV;

    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x1000, 0x4000, Permission::RWX);

    let (addr_var, val_bv) = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "pending_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        (addr_var, RustBV::concrete(value as u128, 32))
    };
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

/// Regression (angr-sqfj8.27): `_export_pending_state` called the unflushed
/// `export_full` instead of `flush_and_export_full`, unlike the fixed
/// `_export_state`/`_export_stash` sites (angr-9ke6b.101) — so a
/// pending-callback state's Multi-cell bytes silently vanished from the
/// export.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn export_pending_state_flushes_multi_cells() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = pending_state_with_multi_cell_store(0xCAFE_BABE);
    let state_id = state.state_id();
    mgr.pending_callbacks
        .insert(StateId::new(state_id), pending_for(state));

    let snap = mgr
        ._export_pending_state(state_id)
        .expect("pending state found");
    let index = snap
        .page_addresses()
        .iter()
        .position(|&a| a == 0x1000)
        .expect("page present in snapshot");
    let symbolic_offsets = snap.get_page(index).expect("page by index").3;
    assert!(
        (0..4u16).all(|off| symbolic_offsets.contains(&off)),
        "unflushed export dropped the Multi-cell bytes: {symbolic_offsets:?}"
    );
}

/// Regression (angr-sqfj8.26): `_pending_memory_load_symbolic_page` iterated
/// `symbolic_objects_iter()` directly without flushing first, so a
/// Multi-covered byte was silently dropped from every pending-callback sync
/// (`_create_state_for_callback`'s symbolic-store replay).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn pending_memory_load_symbolic_page_flushes_multi_cells() {
    Python::initialize();
    Python::attach(|py| {
        if py.import("claripy").is_err() {
            return; // claripy not importable in this env — skip
        }
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = pending_state_with_multi_cell_store(0xCAFE_BABE);
        let state_id = state.state_id();
        mgr.pending_callbacks
            .insert(StateId::new(state_id), pending_for(state));

        let pairs = mgr
            ._pending_memory_load_symbolic_page(py, state_id, 0x1000)
            .expect("pending state found");
        assert!(
            pairs.iter().any(|&(addr, _)| addr == 0x1000),
            "unflushed replay dropped the Multi-cell byte at 0x1000"
        );
    });
}

/// The other half of the asymmetry pinned in
/// `state_api_tests::get_state_register_folds_unmodeled_name_and_symbolic_into_none`:
/// on the pending API an unknown register name is a hard `PyValueError`, so
/// `Ok(None)` means "known register, symbolic value" and nothing else
/// (angr-sqfj8.56).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn get_pending_register_raises_for_unknown_name_but_not_for_symbolic() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut state = RustSimState::new("amd64").expect("state");
    assert!(state.set_register("rax", crate::symbolic::RustBV::concrete(0x2a, 64)));
    let sym = {
        let ctx = state.solver().borrow();
        crate::symbolic::RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    assert!(state.set_register("rbx", sym));
    let state_id = state.state_id();
    mgr.pending_callbacks
        .insert(StateId::new(state_id), pending_for(state));

    assert_eq!(
        mgr._get_pending_register(state_id, "rax")
            .expect("rax read"),
        Some(0x2a),
    );
    assert_eq!(
        mgr._get_pending_register(state_id, "rbx")
            .expect("rbx read"),
        None,
        "a known-but-symbolic register is the ONLY source of Ok(None) here",
    );
    let err = mgr
        ._get_pending_register(state_id, "ymm0")
        .expect_err("unmodeled register name must be a loud error");
    assert!(
        err.to_string().contains("unknown register: ymm0"),
        "error names the offending register: {err}",
    );
}
