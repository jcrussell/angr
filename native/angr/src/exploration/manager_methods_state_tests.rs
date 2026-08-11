// Tests for exploration/manager_methods_state.rs — the stash-query half of the
// `#[pymethods]` surface (angr-03vl4.13).
//
// What is pinned here is the *absent-input* behaviour of the read-only
// accessors, because it is deliberately inconsistent across them and Python
// depends on the inconsistency:
//
//   * The stash-keyed readers (`get_state_pc`, `get_state_ids`, `stash_count`,
//     `get_state_predicate_info`) treat an unknown stash name exactly like an
//     empty one — no error, no declaration side effect. `RustStateProxy` and
//     the predicate cache poll stash names that may never have been declared,
//     so erroring (or auto-declaring, which `StashManager::declare_stash`
//     would) would either break the poll or litter the stash listing.
//   * The state-id-keyed readers (`get_state_pc_by_id`, `state_stash`,
//     `get_state_bbl_history_tail`) return `None` for an unknown id — the
//     "state is gone" signal `RustStateProxy.__repr__` renders.
//   * `find_state` consults the parked pending callback first, so an id-keyed
//     reader can resolve a state that no stash contains. The stash-keyed
//     readers cannot see it at all.
use super::*;

/// Every stash-keyed reader treats a never-declared stash as empty, and does
/// not bring it into existence by asking.
#[test]
fn stash_keyed_readers_treat_an_unknown_stash_as_empty_without_declaring_it() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let sid = mgr.create_state("active").expect("create state");

        assert_eq!(mgr.stash_count("no_such_stash"), 0);
        assert_eq!(mgr.get_state_ids("no_such_stash"), Vec::<u64>::new());
        assert_eq!(mgr.get_state_pc("no_such_stash", 0), None);
        assert!(mgr.get_state_predicate_info("no_such_stash").is_empty());
        assert!(
            mgr.sm.get("no_such_stash").is_none(),
            "a read must not declare the stash it missed — `stash_counts` would \
             then grow an entry per typo Python ever polled",
        );

        // The same index, out of range in a stash that *does* exist, is also
        // None rather than a panic: `get_state_pc`'s index defaults to 0, so
        // the first caller of a drained stash hits this path.
        assert_eq!(mgr.get_state_ids("active"), vec![sid]);
        assert_eq!(mgr.get_state_pc("active", 1), None);
    });
}

/// The id-keyed readers report an unknown state id as `None` rather than a
/// zero/empty stand-in, so Python can tell "gone" from "at pc 0".
#[test]
fn id_keyed_readers_return_none_for_an_unknown_state_id() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let ghost = u64::MAX;

        assert_eq!(mgr.get_state_pc_by_id(ghost), None);
        assert_eq!(mgr.state_stash(ghost), None);
        assert_eq!(mgr.get_state_bbl_history_tail(ghost, 8), None);
        assert_eq!(mgr.state_constraint_count(ghost), None);
        assert_eq!(mgr.get_state_root(ghost), None);
    });
}

/// `get_state_bbl_history_tail`'s `n` is a tail length where **0 means all** —
/// the opposite of the "0 entries" a caller might assume — and an oversized `n`
/// is clamped rather than erroring.
#[test]
fn bbl_history_tail_treats_n_zero_as_all_and_clamps_oversized_n() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let sid = mgr.create_state("active").expect("create state");

        {
            let state = mgr.sm.find_state_mut(sid).expect("state in active");
            for addr in [0x1000_u64, 0x2000, 0x3000, 0x4000] {
                state.add_to_history(addr);
            }
        }

        assert_eq!(
            mgr.get_state_bbl_history_tail(sid, 2),
            Some(vec![0x3000, 0x4000]),
            "a tail of 2 is the two most-recent blocks, not the two oldest",
        );
        assert_eq!(
            mgr.get_state_bbl_history_tail(sid, 0),
            Some(vec![0x1000, 0x2000, 0x3000, 0x4000]),
            "n == 0 is documented as `all entries`; returning an empty Vec here \
             would silently blind the caller to the whole history",
        );
        assert_eq!(
            mgr.get_state_bbl_history_tail(sid, 999),
            Some(vec![0x1000, 0x2000, 0x3000, 0x4000]),
            "an oversized n saturates (hist.len() - n underflow is guarded)",
        );
    });
}

/// `pending_callback_ids` enumerates a population disjoint from the stashes, so
/// a freshly created (stashed) state must not appear in it — the two listings
/// are added together by Python, and an overlap would double-count.
#[test]
fn a_stashed_state_does_not_appear_in_the_pending_callback_listing() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        assert!(mgr.pending_callback_ids().is_empty());

        let sid = mgr.create_state("active").expect("create state");
        assert_eq!(mgr.state_stash(sid).as_deref(), Some("active"));
        assert!(mgr.has_active_states());
        assert!(
            mgr.pending_callback_ids().is_empty(),
            "a stashed state is not parked, so it belongs to exactly one of the \
             two listings",
        );
    });
}

/// `get_state_root` reads the fork-lineage map, which is only populated for
/// forked states — an unforked root has no self-entry.
#[test]
fn get_state_root_has_no_self_entry_for_an_unforked_state() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let sid = mgr.create_state("active").expect("create state");

        assert_eq!(
            mgr.get_state_root(sid),
            None,
            "the roots map records fork edges only; callers wanting \
             root-or-self must use StashManager::root_or_self",
        );
    });
}
