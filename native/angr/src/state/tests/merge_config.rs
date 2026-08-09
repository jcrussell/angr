//! Merging the config-like map/set fields (hooks, options, env vars): union of
//! additions, shared-Arc preservation when nothing diverged, and the removal
//! tombstones that must neither resurrect a removal nor outlive a reinstate.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

/// angr-9ke6b.121 regression: the config-like fields a native proc or a Python
/// setter can mutate *after* a fork must not be carried from `self` alone.
/// `sim_options` / `hooks` / `environment` union; the four boolean SimOption
/// mirrors OR (an option a branch switched on stays on).
#[test]
fn test_merge_unions_config_like_fields_across_branches() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // Options: one shared, one per branch.
    a.set_option("SHORT_READS", true);
    a.set_option("SELF_ONLY", true);
    b.set_option("SHORT_READS", true);
    b.set_option("OTHER_ONLY", true);
    // Boolean mirrors: only the *other* branch enables them, so a self-only
    // carry would silently revert both.
    b.set_keep_ip_symbolic(true);
    b.set_force_eager_forks(true);

    // Hooks: one per branch.
    a.add_hook(0x400000);
    b.add_hook(0x500000);

    // Environment: shared key with a conflicting value (self wins, warned) plus
    // a key only the other branch's setenv created.
    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    b.setenv(b"PATH".to_vec(), b"/b".to_vec());
    b.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "ke6b121_m0", 1),
            RustBV::symbolic(&s, "ke6b121_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(merged.has_option("SHORT_READS"), "shared option survives");
    assert!(merged.has_option("SELF_ONLY"), "self-only option survives");
    assert!(
        merged.has_option("OTHER_ONLY"),
        "other-branch-only option must be unioned in, not reverted to self's set"
    );
    assert!(
        merged.keep_ip_symbolic(),
        "keep_ip_symbolic mirrors a SimOption; a branch that enabled it wins"
    );
    assert!(
        merged.force_eager_forks(),
        "force_eager_forks mirrors a SimOption; a branch that enabled it wins"
    );

    assert!(merged.is_hooked(0x400000), "self's hook survives");
    assert!(
        merged.is_hooked(0x500000),
        "other-branch-only hook must be unioned in"
    );

    let env = merged.environment();
    assert_eq!(
        env.get(b"PATH".as_slice()).map(|v| v.as_slice()),
        Some(b"/a".as_slice()),
        "conflicting environment value keeps the earlier (self) branch's"
    );
    assert_eq!(
        env.get(b"HOME".as_slice()).map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "other-branch-only environment key must be unioned in, not dropped"
    );
}

/// angr-9ke6b.121: the union must not allocate when nobody diverged — an
/// unmutated fork keeps sharing the parent's `Arc`s, so merging two pristine
/// forks leaves the sets pointer-identical to self's.
#[test]
fn test_merge_config_union_keeps_shared_arcs_when_nothing_diverged() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    a.set_option("SHORT_READS", true);
    a.add_hook(0x400000);
    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    let b = a.fork();

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "ke6b121b_m0", 1),
            RustBV::symbolic(&s, "ke6b121b_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(merged.has_option("SHORT_READS"));
    assert!(merged.is_hooked(0x400000));
    assert_eq!(
        merged.environment().get(b"PATH".as_slice()).unwrap(),
        b"/a".as_slice()
    );
    assert!(
        !merged.keep_ip_symbolic() && !merged.force_eager_forks(),
        "OR must not invent a flag no branch set"
    );
}

/// Regression: a plain set union can't distinguish "never touched" from
/// "explicitly removed" — `a.remove_hook(X)` followed by `a.merge(&[&b])`
/// must NOT resurrect `X` just because sibling `b` never touched it. Same
/// bug class for `set_option(name, false)` and `unsetenv`. This is the
/// removal-direction counterpart to
/// `test_merge_unions_config_like_fields_across_branches` above, which only
/// exercised additions.
#[test]
fn test_merge_does_not_resurrect_removed_hook_option_or_env_var() {
    Python::initialize();

    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.add_hook(0x400000);
    ancestor.set_option("SHORT_READS", true);
    ancestor.setenv(b"PATH".to_vec(), b"/a".to_vec());

    let mut a = ancestor.fork();
    let b = ancestor.fork();

    // `a` explicitly removes everything; `b` never touches any of it.
    a.remove_hook(0x400000);
    a.set_option("SHORT_READS", false);
    a.unsetenv(b"PATH");

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "resurrect_m0", 1),
            RustBV::symbolic(&s, "resurrect_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        !merged.is_hooked(0x400000),
        "a's explicit remove_hook must not be undone by b still having the hook"
    );
    assert!(
        !merged.has_option("SHORT_READS"),
        "a's explicit set_option(false) must not be undone by b still having it on"
    );
    assert!(
        merged.environment().get(b"PATH".as_slice()).is_none(),
        "a's explicit unsetenv must not be undone by b still having the var"
    );
}

/// Removal-then-merge must not regress the addition-direction fix
/// (`07f1f024e`): a hook/option/env-var added by `b` only, with `a` never
/// touching it, must still survive the merge.
#[test]
fn test_merge_still_unions_additions_alongside_removals() {
    Python::initialize();

    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.add_hook(0x400000);

    let mut a = ancestor.fork();
    let mut b = ancestor.fork();

    // `a` removes the shared hook; `b` independently adds a brand-new one.
    a.remove_hook(0x400000);
    b.add_hook(0x500000);
    b.set_option("OTHER_ONLY", true);
    b.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "resurrect_add_m0", 1),
            RustBV::symbolic(&s, "resurrect_add_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        !merged.is_hooked(0x400000),
        "a's removal still wins even when b also contributed an unrelated addition"
    );
    assert!(
        merged.is_hooked(0x500000),
        "b's brand-new hook must still be unioned in"
    );
    assert!(
        merged.has_option("OTHER_ONLY"),
        "b's brand-new option must still be unioned in"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"HOME".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "b's brand-new env var must still be unioned in"
    );
}

/// A hook/option/env-var removed and then re-added on the SAME branch before
/// merge must survive: the tombstone has to clear on re-add, or the merge
/// would incorrectly treat it as still-removed and drop it out from under a
/// branch that currently has it live.
#[test]
fn test_merge_reinstated_hook_option_env_survives_own_removal_tombstone() {
    Python::initialize();

    let ancestor = RustSimState::new("amd64").unwrap();
    let mut a = ancestor.fork();
    let b = ancestor.fork();

    a.add_hook(0x400000);
    a.remove_hook(0x400000);
    a.add_hook(0x400000); // back on before merge — tombstone must clear.

    a.set_option("SHORT_READS", true);
    a.set_option("SHORT_READS", false);
    a.set_option("SHORT_READS", true);

    a.setenv(b"PATH".to_vec(), b"/a".to_vec());
    a.unsetenv(b"PATH");
    a.setenv(b"PATH".to_vec(), b"/a2".to_vec());

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "reinstate_m0", 1),
            RustBV::symbolic(&s, "reinstate_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        merged.is_hooked(0x400000),
        "re-adding after remove_hook must clear the tombstone"
    );
    assert!(
        merged.has_option("SHORT_READS"),
        "re-enabling after set_option(false) must clear the tombstone"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"PATH".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/a2".as_slice()),
        "re-setenv after unsetenv must clear the tombstone"
    );
}
