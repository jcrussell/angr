//! The `arc-make-mut-cow` module invariant's peek-before-clone half.
//!
//! Every mutator of an `Arc`-wrapped config collection must check the read
//! path first, so a no-op call (re-adding a hook already present, disabling an
//! option never enabled, re-setting an env var to the value it already holds)
//! does NOT force `Arc::make_mut` to deep-clone the collection away from the
//! sibling states sharing it. `Arc::ptr_eq` against a forked sibling is the
//! only observable that distinguishes "skipped the clone" from "cloned and
//! produced an equal value" — asserting on the resulting *contents* would pass
//! either way, which is how these five mutators drifted from the invariant in
//! the first place (angr-0jh0j.47).

use super::super::*;

/// Fork `state`, run `mutate`, and report whether the field selected by
/// `field` still shares its allocation with the fork.
fn shares_after<T: ?Sized>(
    state: &mut RustSimState,
    field: impl Fn(&RustSimState) -> &Arc<T>,
    mutate: impl FnOnce(&mut RustSimState),
) -> bool {
    let forked = state.fork();
    let before = Arc::clone(field(&forked));
    mutate(state);
    Arc::ptr_eq(field(state), &before)
}

#[test]
fn test_add_hook_noop_skips_cow_clone() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.add_hook(0x400_000);

    assert!(
        shares_after(&mut state, |s| &s.hooks, |s| s.add_hook(0x400_000)),
        "re-adding an already-hooked address must not deep-clone the hook set"
    );
    // A real add still mutates (and therefore may clone).
    state.add_hook(0x400_010);
    assert!(state.is_hooked(0x400_010));
}

#[test]
fn test_remove_hook_noop_skips_cow_clone() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.add_hook(0x400_000);

    assert!(
        shares_after(&mut state, |s| &s.hooks, |s| s.remove_hook(0x400_020)),
        "removing an address that was never hooked must not deep-clone the hook set"
    );
    state.remove_hook(0x400_000);
    assert!(!state.is_hooked(0x400_000));
}

#[test]
fn test_set_option_noop_skips_cow_clone() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_option("SHORT_READS", true);

    assert!(
        shares_after(
            &mut state,
            |s| &s.sim_options,
            |s| s.set_option("SHORT_READS", true)
        ),
        "enabling an already-enabled option must not deep-clone the option set"
    );
    assert!(
        shares_after(
            &mut state,
            |s| &s.sim_options,
            |s| s.set_option("LAZY_SOLVES", false)
        ),
        "disabling an option that was never enabled must not deep-clone the option set"
    );
    state.set_option("SHORT_READS", false);
    assert!(!state.has_option("SHORT_READS"));
}

#[test]
fn test_setenv_noop_skips_cow_clone() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.setenv(b"PATH".to_vec(), b"/bin".to_vec());

    assert!(
        shares_after(
            &mut state,
            |s| &s.environment,
            |s| s.setenv(b"PATH".to_vec(), b"/bin".to_vec())
        ),
        "re-setting an env var to its current value must not deep-clone the environment"
    );
    // A changed value still lands.
    state.setenv(b"PATH".to_vec(), b"/usr/bin".to_vec());
    assert_eq!(state.getenv(b"PATH"), Some(&b"/usr/bin"[..]));
}

#[test]
fn test_unsetenv_noop_skips_cow_clone() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.setenv(b"PATH".to_vec(), b"/bin".to_vec());

    assert!(
        shares_after(&mut state, |s| &s.environment, |s| {
            assert!(!s.unsetenv(b"HOME"));
        }),
        "unsetting a key that was never set must not deep-clone the environment"
    );
    assert!(state.unsetenv(b"PATH"));
    assert_eq!(state.getenv(b"PATH"), None);
}
