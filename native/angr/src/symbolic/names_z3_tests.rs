//! The solver-facing half of `bv/names_tests.rs`: these tests lower or solve, so
//! they stay in `symbolic` (angr-mx00p.116.7).

// The fresh-name counters count per key and are raised, never lowered, by a
// restored snapshot's watermarks.

/// A snapshot carries the counters, and restoring one written by a process
/// that had minted further than this one raises the local counter past it.
#[test]
fn a_restored_snapshot_reserves_its_name_counters() {
    use crate::symbolic::SymContext;
    let key = "names_tests::snapshot";
    let ctx = SymContext::new_mock();
    let mine = crate::bv::names::next(key);
    let mut snap = ctx.to_snapshot();
    assert_eq!(snap.name_counters.get(key), Some(&(mine + 1)));

    // As if written by a process that minted 5000 more names under `key`.
    snap.name_counters.insert(key.to_owned(), mine + 5000);
    SymContext::new_mock().restore_from_snapshot(&snap);
    assert_eq!(crate::bv::names::next(key), mine + 5000, "no local mint may reuse a restored name");
}
