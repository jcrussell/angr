// Tests for exploration/manager_methods_hooks.rs — the hook / SimProcedure
// registration corner of the `#[pymethods]` surface.
//
// The invariant under test: `hooks` and `simprocedures` are two halves of one
// dispatch decision. `run_loop_single.rs`'s step reads
// `self.simprocedures.get(&pc)` *inside* the `self.hooks.contains(&pc)` gate,
// so an address present in `simprocedures` but absent from `hooks` is inert —
// right up until a plain `add_hook` on that same address re-opens the gate and
// resurrects the stale SimProcedure. Every mutator that removes an address
// must therefore remove it from BOTH maps (`unregister_simprocedures` always
// did; `clear_hooks` did not — angr-sqfj8.30).
use super::*;

/// `clear_hooks` clears the SimProcedure table too, so a later plain
/// `add_hook` on a previously-registered address does NOT resurrect the old
/// SimProcedure dispatch (angr-sqfj8.30).
#[test]
fn clear_hooks_also_clears_simprocedures() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_simprocedure(0x40_1000, "strlen".to_string(), 1, false);
    mgr.add_hook(0x40_2000);
    assert!(mgr.hooks.contains(&0x40_1000) && mgr.hooks.contains(&0x40_2000));
    assert!(mgr.simprocedures.contains_key(&0x40_1000));

    mgr.clear_hooks();
    assert!(mgr.hooks.is_empty(), "clear_hooks emptied the hook set");
    assert!(
        mgr.simprocedures.is_empty(),
        "clear_hooks emptied the SimProcedure table too — a survivor would be \
         invisible until the address was re-hooked",
    );

    // The resurrection path the bug enabled: re-hook the same address as a
    // plain (non-SimProcedure) hook and confirm the dispatch gate now finds
    // no SimProcedure behind it.
    mgr.add_hook(0x40_1000);
    assert!(
        !mgr.simprocedures.contains_key(&0x40_1000),
        "re-hooking a cleared address does not resurrect its SimProcedure",
    );
}

/// The sibling that always upheld the both-maps invariant, pinned alongside
/// `clear_hooks` so the two can't drift apart again: `unregister_simprocedures`
/// removes only the named addresses, from both maps, leaving the rest intact.
#[test]
fn unregister_simprocedures_removes_from_both_maps() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_simprocedures(vec![
        (0x40_1000, "strlen".to_string(), 1, false),
        (0x40_2000, "memcpy".to_string(), 3, false),
    ]);
    mgr.unregister_simprocedures(vec![0x40_1000]);

    assert!(!mgr.hooks.contains(&0x40_1000));
    assert!(!mgr.simprocedures.contains_key(&0x40_1000));
    assert!(
        mgr.hooks.contains(&0x40_2000) && mgr.simprocedures.contains_key(&0x40_2000),
        "unregister is address-scoped; the untouched proc survives",
    );
}
