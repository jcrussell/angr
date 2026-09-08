//! `RustSimState`'s environment map: the `entry_state(env=...)` seed
//! (`seed_environment`) and its interaction with the native
//! `setenv`/`unsetenv`/`clearenv` mutators.
//!
//! The seed exists because angr models the initial environment purely as a
//! memory-backed `envp` array that the native `getenv` family never reads
//! (angr-6cp06.12); `rust_manager.py::_seed_environ_to_rust` walks it and
//! pushes the concrete pairs through here.

use super::super::*;

#[test]
fn test_seed_environment_populates_empty_map() {
    let mut state = RustSimState::new("amd64").unwrap();
    assert_eq!(state.getenv(b"FLAG_CHECK"), None);

    let inserted = state.seed_environment(vec![
        (b"FLAG_CHECK".to_vec(), b"1".to_vec()),
        (b"PATH".to_vec(), b"/usr/bin".to_vec()),
    ]);

    assert_eq!(inserted, 2);
    assert_eq!(state.getenv(b"FLAG_CHECK"), Some(b"1".as_slice()));
    assert_eq!(state.getenv(b"PATH"), Some(b"/usr/bin".as_slice()));
}

#[test]
fn test_seed_environment_keeps_empty_value() {
    // `entry_state(env={"EMPTY": ""})` is a real shape — an empty value must
    // stay a *present* key, not collapse into a miss (which would make
    // `setenv(..., overwrite=0)` clobber it).
    let mut state = RustSimState::new("amd64").unwrap();
    assert_eq!(state.seed_environment(vec![(b"EMPTY".to_vec(), Vec::new())]), 1);
    assert_eq!(state.getenv(b"EMPTY"), Some(b"".as_slice()));
}

#[test]
fn test_seed_environment_does_not_overwrite_runtime_setenv() {
    // Mid-run re-add (merge / cross-manager transfer): the source `envp` array
    // still holds the pristine initial value, but the guest's own `setenv`
    // must win.
    let mut state = RustSimState::new("amd64").unwrap();
    state.setenv(b"PATH".to_vec(), b"/runtime".to_vec());

    let inserted = state.seed_environment(vec![
        (b"PATH".to_vec(), b"/initial".to_vec()),
        (b"HOME".to_vec(), b"/root".to_vec()),
    ]);

    assert_eq!(inserted, 1, "only the absent key is seeded");
    assert_eq!(state.getenv(b"PATH"), Some(b"/runtime".as_slice()));
    assert_eq!(state.getenv(b"HOME"), Some(b"/root".as_slice()));
}

#[test]
fn test_seed_environment_does_not_resurrect_unsetenv() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.setenv(b"PATH".to_vec(), b"/initial".to_vec());
    assert!(state.unsetenv(b"PATH"));

    assert_eq!(state.seed_environment(vec![(b"PATH".to_vec(), b"/initial".to_vec())]), 0);
    assert_eq!(state.getenv(b"PATH"), None);
}

#[test]
fn test_seed_environment_does_not_resurrect_clearenv() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.setenv(b"PATH".to_vec(), b"/initial".to_vec());
    state.clearenv();

    let inserted = state.seed_environment(vec![
        (b"PATH".to_vec(), b"/initial".to_vec()),
        (b"NEW".to_vec(), b"v".to_vec()),
    ]);

    assert_eq!(inserted, 1, "clearenv tombstones PATH; NEW was never present");
    assert_eq!(state.getenv(b"PATH"), None);
    assert_eq!(state.getenv(b"NEW"), Some(b"v".as_slice()));
}

#[test]
fn test_seed_environment_survives_fork() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.seed_environment(vec![(b"FLAG_CHECK".to_vec(), b"1".to_vec())]);

    let child = state.fork();
    assert_eq!(child.getenv(b"FLAG_CHECK"), Some(b"1".as_slice()));
}
