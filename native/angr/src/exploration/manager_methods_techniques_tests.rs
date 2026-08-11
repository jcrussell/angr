// Tests for exploration/manager_methods_techniques.rs — the native-technique
// and uniqueness-filter corner of the `#[pymethods]` surface (angr-03vl4.13).
//
// The contract these pin is the *stash-declaration side effect*. Each
// `register_*` here pushes a `NativeTechnique` AND declares the stash that
// technique will later move states into; the technique bodies in
// `native_technique.rs` assume the destination exists by the time they run, so
// a registrar that pushes without declaring leaves a landmine that only fires
// on the step that first tries to move a state. The two deliberate
// non-declarations (`register_length_limiter` with `drop=true`, which targets
// the `_DROP` sink, and `disable_uniqueness_filter`, which is not a registrar)
// are pinned here too so a future "declare it unconditionally, just in case"
// cleanup has to argue with a test.
use super::*;

/// A `drop=true` length limiter sends states to `_DROP` and must NOT declare
/// the `cut` stash; `drop=false` must.
#[test]
fn length_limiter_declares_cut_only_when_not_dropping() {
    let mut dropping = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    dropping.register_length_limiter(8, true);
    assert_eq!(dropping.native_technique_count(), 1);
    assert!(
        dropping.sm.get("cut").is_none(),
        "a dropping limiter never moves anything to `cut`, so declaring it \
         would leave a permanently-empty stash in every stash listing",
    );

    let mut cutting = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    cutting.register_length_limiter(8, false);
    assert!(
        cutting.sm.get("cut").is_some(),
        "a non-dropping limiter's destination must exist before the first step",
    );
}

/// `register_loop_bound` declares whichever stash it was handed, not just the
/// `"spinning"` default from its `#[pyo3(signature = ...)]`.
#[test]
fn loop_bound_declares_the_caller_supplied_discard_stash() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_loop_bound(4, "spinning");
    mgr.register_loop_bound(4, "my_own_sink");

    assert_eq!(mgr.native_technique_count(), 2);
    assert!(mgr.sm.get("spinning").is_some());
    assert!(
        mgr.sm.get("my_own_sink").is_some(),
        "the declaration follows the argument; a hardcoded \"spinning\" would \
         leave a custom sink undeclared",
    );
}

/// `register_timeout` and `register_merge_point` declare their destinations
/// too — the timeout stash by constant, the merge wait stash per address.
#[test]
fn timeout_and_merge_point_declare_their_destinations() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_timeout(1.0);
    assert!(mgr.sm.get(TIMEOUT_STASH).is_some());

    mgr.register_merge_point(0x40_1000, 10);
    mgr.register_merge_point(0x40_2000, 10);
    assert!(
        mgr.sm.get("merge_waiting_0x401000").is_some() && mgr.sm.get("merge_waiting_0x402000").is_some(),
        "the wait stash is per-address, so two merge points cannot share one \
         waiting room and cross-contaminate their callstack grouping",
    );
    assert_eq!(mgr.native_technique_count(), 3);
}

/// `clear_native_techniques` empties the technique list but deliberately leaves
/// the stashes it declared behind — dropping them would strand any state a
/// technique had already parked there.
#[test]
fn clear_native_techniques_keeps_the_stashes_it_declared() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_loop_bound(4, "spinning");
    mgr.register_timeout(1.0);
    mgr.clear_native_techniques();

    assert_eq!(mgr.native_technique_count(), 0);
    assert!(
        mgr.sm.get("spinning").is_some() && mgr.sm.get(TIMEOUT_STASH).is_some(),
        "un-declaring a stash would orphan whatever the technique already \
         moved into it",
    );
}

/// The uniqueness filter's enabled-ness is derived from the register list, and
/// `register_uniqueness_filter` resets the seen-set so a re-register does not
/// judge fresh register tuples against tuples read from different registers.
#[test]
fn uniqueness_filter_enabled_is_derived_and_re_register_resets_the_seen_set() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    assert!(!mgr.uniqueness_filter_enabled());

    mgr.register_uniqueness_filter(vec!["rax".to_string(), "rbx".to_string()]);
    assert!(mgr.uniqueness_filter_enabled());
    assert!(mgr.sm.get("not_unique").is_some());

    // Simulate a step having recorded tuples, then re-register on a different
    // register set: the stale tuples must not survive the switch.
    mgr.constraint_tracker.uniqueness_set.insert(0xdead_beef);
    assert_eq!(mgr.uniqueness_set_size(), 1);
    mgr.register_uniqueness_filter(vec!["rcx".to_string()]);
    assert_eq!(
        mgr.uniqueness_set_size(),
        0,
        "tuples hashed from rax/rbx are meaningless once the filter watches rcx",
    );

    // An empty register list is the off state, so `disable` is just a clear.
    mgr.constraint_tracker.uniqueness_set.insert(0xfeed_face);
    mgr.disable_uniqueness_filter();
    assert!(!mgr.uniqueness_filter_enabled());
    assert_eq!(mgr.uniqueness_set_size(), 0);
}
