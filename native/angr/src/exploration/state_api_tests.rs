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

// ---------------------------------------------------------------------------
// Constraint import: undecided vs proven-unsat (angr-zoav9)
// ---------------------------------------------------------------------------

/// Park a state in `found` whose satisfiability query aborts to Z3 Unknown:
/// the easy `10 < x < 20` set under `rlimit=1`, the same rig
/// `state/tests/solver_gate.rs::test_survives_sat_prune_keeps_an_undecided_state`
/// uses (an rlimit beats a wall-clock timeout because it is deterministic).
#[cfg(feature = "vex-engine-z3")]
fn undecided_state_in(mgr: &mut RustExplorationManager, name: &str) -> u64 {
    let state = RustSimState::new("amd64").expect("amd64 state");
    {
        let ctx = state.solver().borrow();
        let x = RustBV::symbolic(&ctx, name, 32);
        ctx.assume_true(&x.ugt(&RustBV::concrete(10, 32), &ctx));
        ctx.assume_true(&x.ult(&RustBV::concrete(20, 32), &ctx));
        ctx.pin_rlimit_for_test(1);
    }
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);
    state_id
}

/// Lift the pinned rlimit so the same state decides instantly — proves the
/// preceding `None` came from the budget, not from a poisoned solver.
#[cfg(feature = "vex-engine-z3")]
fn unpin_rlimit(mgr: &RustExplorationManager, state_id: u64) {
    mgr.find_state(state_id)
        .expect("state found")
        .solver()
        .borrow()
        .pin_rlimit_for_test(0);
}

/// `add_constraints_to_state` reports an undecided post-import satisfiability
/// query as `False` (claripy-shaped, lenient), while the `_checked` sibling
/// keeps it distinguishable as `None`. Without the split, a Python caller that
/// drops the state on `False` deletes a state Z3 never proved contradictory
/// (`invariant-z3-unknown-not-unsat`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn add_constraints_to_state_separates_undecided_from_unsat() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state_id = undecided_state_in(&mut mgr, "add_constraints_undecided_x");
        // Empty list: the import itself is a no-op, so the only thing under
        // test is how the satisfiability verdict is reported. (A non-empty
        // list would need claripy, which the cargo-test binary has no
        // guaranteed venv for.)
        let empty = pyo3::types::PyList::empty(py);

        assert_eq!(
            mgr.add_constraints_to_state_checked(py, state_id, &empty)
                .expect("state found"),
            None,
            "an aborted (Unknown) satisfiability query must stay distinguishable",
        );
        assert!(
            !mgr.add_constraints_to_state(py, state_id, &empty)
                .expect("state found"),
            "the lenient form is what makes the checked one necessary: it \
             collapses the same Unknown into False",
        );

        unpin_rlimit(&mgr, state_id);
        assert_eq!(
            mgr.add_constraints_to_state_checked(py, state_id, &empty)
                .expect("state found"),
            Some(true),
        );
    });
}

/// The checked form is not a blanket "keep": a *proven* contradiction still
/// comes back as `Some(false)`, matching the lenient form exactly.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn add_constraints_to_state_checked_still_reports_a_proven_unsat() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let state = RustSimState::new("amd64").expect("amd64 state");
        {
            let ctx = state.solver().borrow();
            let x = RustBV::symbolic(&ctx, "add_constraints_unsat_x", 32);
            let five = RustBV::concrete(5, 32);
            ctx.assume_true(&x.eq(&five, &ctx));
            ctx.assume_true(&x.eq(&five, &ctx).not(&ctx));
        }
        let state_id = state.state_id();
        mgr.sm.push(STASH_FOUND, state);
        let empty = pyo3::types::PyList::empty(py);

        assert_eq!(
            mgr.add_constraints_to_state_checked(py, state_id, &empty)
                .expect("state found"),
            Some(false),
        );
        assert!(
            !mgr.add_constraints_to_state(py, state_id, &empty)
                .expect("state found"),
        );
    });
}

/// Same split on the raw-Z3-pointer import path, which shares the verdict
/// contract with `add_constraints_to_state`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn import_z3_constraint_ptrs_separates_undecided_from_unsat() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

        // Donor state supplies a live Bool-sorted assertion pointer; it stays
        // in the stash so the AST outlives the import.
        let donor = RustSimState::new("amd64").expect("amd64 state");
        {
            let ctx = donor.solver().borrow();
            let y = RustBV::symbolic(&ctx, "import_ptrs_donor_y", 32);
            ctx.assume_true(&y.ugt(&RustBV::concrete(3, 32), &ctx));
        }
        let donor_id = donor.state_id();
        mgr.sm.push(STASH_FOUND, donor);
        let ptrs = mgr
            ._export_z3_constraint_ptrs(donor_id)
            .expect("donor found");
        assert!(!ptrs.is_empty(), "donor must export at least one assertion");

        let state_id = undecided_state_in(&mut mgr, "import_ptrs_undecided_x");
        assert_eq!(
            mgr.import_z3_constraint_ptrs_checked(state_id, ptrs.clone())
                .expect("state found"),
            None,
        );
        assert!(
            !mgr.import_z3_constraint_ptrs(state_id, ptrs.clone())
                .expect("state found"),
        );

        unpin_rlimit(&mgr, state_id);
        assert_eq!(
            mgr.import_z3_constraint_ptrs_checked(state_id, ptrs)
                .expect("state found"),
            Some(true),
        );
    });
}

// ---------------------------------------------------------------------------
// Concrete memory get/set (angr-0jh0j.26)
// ---------------------------------------------------------------------------

/// Push `state` into `found` and hand back its id — the two-line preamble
/// every state-keyed accessor test below shares.
#[cfg(feature = "vex-engine-z3")]
fn stash_found(mgr: &mut RustExplorationManager, state: RustSimState) -> u64 {
    let state_id = state.state_id();
    mgr.sm.push(STASH_FOUND, state);
    state_id
}

/// `_get_state_memory` has three distinct answers and they must stay
/// distinct: bytes for a readable range, `Ok(None)` for an unreadable one
/// (unmapped / no SAT witness — a *memory* answer), and `Err` for both a
/// missing `state_id` and an over-cap `size` (caller bugs, which must not be
/// laundered through the `Ok(None)` channel — angr-0jh0j.11).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn get_state_memory_separates_unreadable_from_caller_error() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x1000, 0x1000, Permission::RW);
    let state_id = stash_found(&mut mgr, state);

    mgr._set_state_memory_concrete(state_id, 0x1010, b"hello")
        .expect("store into a mapped page");
    assert_eq!(
        mgr._get_state_memory(state_id, 0x1010, 5)
            .expect("state found"),
        Some(b"hello".to_vec()),
    );

    assert_eq!(
        mgr._get_state_memory(state_id, 0x9_0000, 4)
            .expect("state found"),
        None,
        "an unmapped range is an Ok(None) memory answer, not an error",
    );

    let err = mgr
        ._get_state_memory(state_id, 0x1010, crate::symbolic::MAX_CONCRETE_LOAD_BYTES + 1)
        .expect_err("an over-cap size must raise");
    assert!(
        err.to_string().contains("get_state_memory"),
        "the refusal names the op: {err}",
    );
    // The size check runs before the state lookup, so an unknown id still has
    // to raise on its own account for a legal size.
    assert!(
        mgr._get_state_memory(state_id + 999, 0x1010, 4).is_err(),
        "a missing state_id is an error, never Ok(None)",
    );
}

/// The `_automap` suffix on the concrete-store pair is the whole difference
/// between a lost write and a landed one: `memory_store` refuses an address
/// outside every mapped page and lazy region, and the callback-memory proxy
/// gets to *choose* such an address (angr-ijwp0). Pin both halves — plain
/// store refuses, automap widens the lazy region and the bytes read back.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_state_memory_concrete_automap_accepts_what_the_plain_store_refuses() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state_id = stash_found(&mut mgr, RustSimState::new("amd64").expect("amd64 state"));

    assert!(
        mgr._set_state_memory_concrete(state_id, 0x5_0000, b"xy")
            .is_err(),
        "precondition: the plain store must refuse an unmapped address",
    );
    mgr._set_state_memory_concrete_automap(state_id, 0x5_0000, b"xy")
        .expect("automap store must map the page and land the write");
    assert_eq!(
        mgr._get_state_memory(state_id, 0x5_0000, 2)
            .expect("state found"),
        Some(b"xy".to_vec()),
    );
}

// ---------------------------------------------------------------------------
// File-descriptor inspection (angr-0jh0j.26)
// ---------------------------------------------------------------------------

/// The fd-output family fans out from one `FileSystem` buffer per fd, so a
/// getter wired to the wrong fd would still look plausible in isolation.
/// Pin that `_get_state_stdout` is fd 1 specifically, that `_has_*` agrees
/// with `_get_*` per fd, and that a missing state answers `false` rather than
/// raising (the `find_state`-based getters deliberately have no `PyResult`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn state_fd_output_accessors_are_per_fd() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state_id = stash_found(&mut mgr, RustSimState::new("amd64").expect("amd64 state"));

    assert!(!mgr._has_state_fd_output(state_id, 1));
    assert!(
        mgr._append_state_fd_output(state_id, 1, b"out".to_vec())
            .expect("state found"),
    );
    assert!(
        mgr._append_state_fd_output(state_id, 2, b"err".to_vec())
            .expect("state found"),
    );

    assert_eq!(mgr._get_state_stdout(state_id).expect("state found"), b"out");
    assert_eq!(
        mgr._get_state_fd_output(state_id, 2).expect("state found"),
        b"err",
    );
    assert!(mgr._has_state_fd_output(state_id, 1));
    assert!(!mgr._has_state_fd_output(state_id, 3), "untouched fd is empty");
    assert!(
        !mgr._has_state_fd_output(state_id + 999, 1),
        "a missing state_id answers false rather than panicking",
    );
}

/// `_register_state_fd` is the inbound half of the callback fd sync, and its
/// `false` return is what keeps that sync idempotent across the repeated
/// callbacks sharing one cached state. Pin the round-trip through
/// `_get_state_open_fds` / `_get_state_fd_content`, the re-register refusal,
/// and `_has_state_extra_fds`, whose whole job is to stay `false` for the
/// three standard descriptors so the common case skips the fd export.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn register_state_fd_round_trips_and_refuses_a_duplicate() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state_id = stash_found(&mut mgr, RustSimState::new("amd64").expect("amd64 state"));

    assert!(
        !mgr._has_state_extra_fds(state_id),
        "precondition: a fresh state has nothing above stderr",
    );
    assert!(
        mgr._register_state_fd(state_id, 3, "/tmp/f".to_string(), 0, b"data".to_vec(), 2)
            .expect("state found"),
    );
    assert!(mgr._has_state_extra_fds(state_id));
    assert_eq!(
        mgr._get_state_fd_content(state_id, 3).expect("state found"),
        b"data",
    );

    let fds = mgr._get_state_open_fds(state_id).expect("state found");
    let entry = fds
        .iter()
        .find(|e| e.0 == 3)
        .expect("fd 3 present in the export");
    assert_eq!(
        (entry.1.as_str(), entry.2, entry.4, entry.5),
        ("/tmp/f", 2u64, 4usize, true),
        "name / position / content-len / is_open survive the round-trip: {entry:?}",
    );

    assert!(
        !mgr._register_state_fd(state_id, 3, "/tmp/other".to_string(), 0, vec![], 0)
            .expect("state found"),
        "re-registering a known fd changes nothing and reports false",
    );
    assert_eq!(
        mgr._get_state_fd_content(state_id, 3).expect("state found"),
        b"data",
        "the refused re-register must not have clobbered the content",
    );
}

/// angr-tqw60: `_state_fd0_is_dup2d` is how `_inject_rust_fds` tells a dup2'd
/// fd 0 from the pristine placeholder. It cannot be derived from
/// `_get_state_open_fds`, whose only provenance signal is the name — and a
/// guest `open("/dev/stdin")` + `dup2(fd, 0)` leaves fd 0 named exactly like
/// the placeholder, which is the shape asserted here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn state_fd0_is_dup2d_tracks_a_dup2_from_a_reopened_dev_stdin() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let pristine = stash_found(&mut mgr, RustSimState::new("amd64").expect("amd64 state"));

    let mut rewired = RustSimState::new("amd64").expect("amd64 state");
    let src = rewired
        .file_system()
        .open_with_content(
            "/dev/stdin".to_string(),
            crate::state::FdFlags::ReadOnly,
            b"REALIN".to_vec(),
        )
        .expect("fd space is not exhausted in tests");
    assert_eq!(rewired.file_system().dup2(src, 0), Some(0));
    let rewired = stash_found(&mut mgr, rewired);

    assert!(!mgr._state_fd0_is_dup2d(pristine));
    assert!(mgr._state_fd0_is_dup2d(rewired));
    assert!(
        !mgr._state_fd0_is_dup2d(u64::MAX),
        "a missing state_id answers false rather than panicking",
    );
    assert!(
        mgr._has_state_extra_fds(rewired),
        "the dup2'd state still needs its fd table synced inbound (angr-qmrrp)",
    );
}

// ---------------------------------------------------------------------------
// Call stack / history / heap getters (angr-0jh0j.26)
// ---------------------------------------------------------------------------

/// The three history-shaped getters each flatten a struct into a tuple, so a
/// swapped field would type-check silently. Pin the field order of
/// `CallStackEntry` and `HistoryEntry` as exported, plus the depth accessor
/// and the allocated/freed split `_get_state_heap_metadata` returns.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn call_stack_history_and_heap_getters_preserve_field_order() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.push_call(0x400_100, 0x400_200, 0x400_105, 0x7fff_0000);
    state.add_history_entry(0x400_100, 3, 0x400_200);
    let kept = state.heap_alloc(0x20);
    let freed = state.heap_alloc(0x10);
    assert_eq!(state.heap_free(freed), Some(0x10));
    let state_id = stash_found(&mut mgr, state);

    assert_eq!(
        mgr._get_state_call_stack(state_id).expect("state found"),
        vec![(0x400_100, 0x400_200, 0x400_105, 0x7fff_0000)],
        "(call_site, callee, return, stack_ptr) order",
    );
    assert_eq!(
        mgr._get_state_call_stack_depth(state_id)
            .expect("state found"),
        1,
    );
    assert_eq!(
        mgr._get_state_detailed_history(state_id)
            .expect("state found"),
        vec![(0x400_100, 3, 0x400_200)],
        "(addr, jumpkind, jump_target) order",
    );

    let (allocated, freed_list) = mgr._get_state_heap_metadata(state_id).expect("state found");
    assert_eq!(
        allocated,
        vec![(kept, 0x20)],
        "the freed allocation leaves `allocated`, the live one stays",
    );
    assert_eq!(freed_list, vec![freed]);
}

// ---------------------------------------------------------------------------
// stdin symbol eval (angr-0jh0j.26)
// ---------------------------------------------------------------------------

/// A Z3 const's identity is `(name, sort)`, so `_eval_stdin_symbol` must
/// rebuild the symbol at the width the caller recorded — a hardcoded width 8
/// minted a *different*, unconstrained const for the 32/64-bit symbols
/// `scanf("%d")` records, and eval then answered an arbitrary model value
/// (angr-ph300.18). Constrain a 32-bit stdin symbol to a value no 8-bit
/// const could produce, so a width regression cannot pass by luck.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn eval_stdin_symbol_reconstructs_the_symbol_at_the_recorded_width() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.record_stdin_symbol("stdin_scanf_0".to_string(), 32);
    {
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "stdin_scanf_0", 32);
        ctx.assume_true(&sym.eq(&RustBV::concrete(0x4142_4344, 32), &ctx));
    }
    let state_id = stash_found(&mut mgr, state);

    assert_eq!(
        mgr._get_state_stdin_symbols(state_id).expect("state found"),
        vec![("stdin_scanf_0".to_string(), 32)],
    );
    assert!(mgr._has_state_stdin_symbols(state_id));
    assert_eq!(
        mgr._eval_stdin_symbol(state_id, "stdin_scanf_0", 32),
        Some(0x4142_4344),
    );
    // Same name at the wrong width is a different const: unconstrained, and
    // 8 bits cannot hold the constrained value under any model.
    let narrow = mgr
        ._eval_stdin_symbol(state_id, "stdin_scanf_0", 8)
        .expect("an unconstrained const still evals");
    assert!(narrow <= 0xff, "width 8 yields an 8-bit value: {narrow:#x}");

    assert_eq!(
        mgr._eval_stdin_symbol(state_id + 999, "stdin_scanf_0", 32),
        None,
        "a missing state_id answers None rather than minting a bare context",
    );
}
