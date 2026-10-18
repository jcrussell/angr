//! The fresh-name counters count per key and are raised, never lowered, by a
//! restored snapshot's watermarks.

use super::*;

#[test]
fn keys_count_on_their_own() {
    let a0 = next("names_tests::a");
    let b0 = next("names_tests::b");
    assert_eq!(next("names_tests::a"), a0 + 1);
    assert_eq!(next("names_tests::b"), b0 + 1);
}

#[test]
fn reserve_raises_but_never_lowers() {
    let key = "names_tests::reserve";
    let first = next(key);
    let mark = first + 1000;
    reserve(&BTreeMap::from([(key.to_owned(), mark)]));
    assert_eq!(next(key), mark, "a restored watermark is the next index");
    reserve(&BTreeMap::from([(key.to_owned(), 0)]));
    assert_eq!(next(key), mark + 1, "an older watermark must not rewind");
    assert!(watermarks()[key] >= mark + 2);
}

#[test]
fn reserve_creates_a_counter_this_process_has_not_used() {
    let key = "names_tests::fresh";
    reserve(&BTreeMap::from([(key.to_owned(), 7)]));
    assert_eq!(next(key), 7);
}

/// A snapshot carries the counters, and restoring one written by a process
/// that had minted further than this one raises the local counter past it.
#[test]
fn a_restored_snapshot_reserves_its_name_counters() {
    use crate::symbolic::SymContext;
    let key = "names_tests::snapshot";
    let ctx = SymContext::new_mock();
    let mine = next(key);
    let mut snap = ctx.to_snapshot();
    assert_eq!(snap.name_counters.get(key), Some(&(mine + 1)));

    // As if written by a process that minted 5000 more names under `key`.
    snap.name_counters.insert(key.to_owned(), mine + 5000);
    SymContext::new_mock().restore_from_snapshot(&snap);
    assert_eq!(next(key), mine + 5000, "no local mint may reuse a restored name");
}
