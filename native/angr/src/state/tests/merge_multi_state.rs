//! Merges whose answer depends on comparing more than two branches: keeping the
//! longest stdout branch of a filesystem merge, and the three-state overlay
//! union and its tiebreak.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-ph300.75: RustSimState::merge keeps the longest-stdout branch's
// FileSystem wholesale (documented stdout-only merge contract). This locks in
// that contract: the merged fs is the longest-stdout branch's fd table, ties
// keep the earlier branch, and a dropped branch carrying non-stdout fds
// (file offsets/writes) is a detectable warn condition (its state is lost).
#[test]
fn test_merge_filesystem_keeps_longest_stdout_branch() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // b writes more to stdout, so its FileSystem wins the merge. a opens a
    // non-std file fd and writes to it — that state is on the *dropped* branch.
    assert!(a.write_stdout(b"hi"));
    assert!(b.write_stdout(b"hello!"));
    let a_fd = a
        .file_system()
        .open("/tmp/only_on_a".to_string(), FdFlags::WriteOnly);
    assert!(a_fd > 2, "expected a non-std fd, got {a_fd}");
    assert!(a.write_fd(a_fd, b"branch-A-only"));

    // Precondition for the warn: a (the dropped branch) has fds above stderr,
    // b (the winner) does not.
    assert!(
        a.file_system_ref().has_fds_above_stderr(),
        "branch A must carry a non-std fd before merge"
    );
    assert!(
        !b.file_system_ref().has_fds_above_stderr(),
        "branch B must not carry a non-std fd"
    );

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30075_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30075_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    // The merged fs is b's (longest stdout): b's stdout survives and a's
    // non-std file fd is gone — the documented loss the warn announces.
    assert_eq!(
        merged.stdout_buffer(),
        b"hello!",
        "merged fs must be the longest-stdout branch (b)"
    );
    assert!(
        !merged.file_system_ref().has_fds_above_stderr(),
        "a's dropped non-std fd must NOT appear in the merged fs"
    );

    // Tie-break: equal stdout keeps self (a). Fresh states, both write the same
    // number of bytes, so a (self) wins and its non-std fd survives.
    let mut c = RustSimState::new("amd64").unwrap();
    let mut d = RustSimState::new("amd64").unwrap();
    assert!(c.write_stdout(b"eq"));
    assert!(d.write_stdout(b"eq"));
    let c_fd = c
        .file_system()
        .open("/tmp/only_on_c".to_string(), FdFlags::WriteOnly);
    assert!(c.write_fd(c_fd, b"branch-C-only"));
    let (n0, n1) = {
        let s = c.solver().borrow();
        (
            RustBV::symbolic(&s, "ph30075_n0", 1),
            RustBV::symbolic(&s, "ph30075_n1", 1),
        )
    };
    let merged_tie = c.merge(&[&d], &[n0, n1]);
    assert!(
        merged_tie.file_system_ref().has_fds_above_stderr(),
        "on a stdout tie the earlier branch (self=c) wins, keeping its non-std fd"
    );
}

// angr-qwyti.2 / angr-n0irt.18: systematic 3-state coverage of the overlay-union
// arms of RustSimState::merge (native/angr/src/state/fork.rs::merge). The
// pre-existing merge tests only exercised `hook_symbolic_memory` with 2 states,
// so two documented policies were asserted nowhere:
//   1. `symbolic_pages` and `addr_to_ast` share the identical union-with-
//      earlier-wins loop but were never touched by a test — a per-map typo
//      (e.g. inserting into the wrong map, or a `.contains_key`/`insert`
//      mismatch) would have gone undetected.
//   2. The 3-way tie-break "self > earlier-other > later-other": with only 2
//      states, a bug that iterated `others` in reverse (later-other winning a
//      conflict over earlier-other) passed silently. This needs >= 3 states
//      where the *same* key is present in two different `others` but NOT self.
//
// The test drives every overlay map through the same four-address table so a
// future map with the same policy can be slotted into the same asserts:
//   0x1000 present in all three  -> merged keeps self (a)          [self wins]
//   0x2000 present in b and c    -> merged keeps earlier-other (b) [tie-break]
//   0x3000 present in c only     -> survives                       [last-other]
//   0x4000 present in a only     -> survives                       [self-only]
// It also covers the `stdin_symbols` union arm (dedup by name, self-order-first),
// likewise untested before.
#[test]
fn test_merge_three_state_overlay_union_and_tiebreak() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();
    let mut c = RustSimState::new("amd64").unwrap();

    // Per-state, per-address entries. The *size* field (u32) of hook/addr_to_ast
    // entries and the *int value* of symbolic_pages ASTs double as a provenance
    // tag: 1 == came from a (self), 2 == b (earlier other), 3 == c (later other).
    Python::attach(|py| {
        let ast = |n: i64| n.into_pyobject(py).unwrap().into_any().unbind();

        // symbolic_pages: replace_symbolic_pages takes the whole map per state.
        let mut pa = std::collections::HashMap::new();
        pa.insert(0x1000u64, ast(1)); // all three -> self wins
        pa.insert(0x4000u64, ast(1)); // self-only -> survives
        a.replace_symbolic_pages(pa);
        let mut pb = std::collections::HashMap::new();
        pb.insert(0x1000u64, ast(2));
        pb.insert(0x2000u64, ast(2)); // b & c -> earlier-other (b) wins
        b.replace_symbolic_pages(pb);
        let mut pc = std::collections::HashMap::new();
        pc.insert(0x1000u64, ast(3));
        pc.insert(0x2000u64, ast(3));
        pc.insert(0x3000u64, ast(3)); // last-other-only -> survives
        c.replace_symbolic_pages(pc);

        // hook_symbolic_memory + addr_to_ast: same table, provenance in the size.
        for (st, tag) in [(&mut a, 1u32), (&mut b, 2), (&mut c, 3)] {
            st.set_hook_symbolic_memory(0x1000, py.None(), tag);
            st.set_addr_to_ast(0x1000, py.None(), tag);
        }
        for (st, tag) in [(&mut b, 2u32), (&mut c, 3)] {
            st.set_hook_symbolic_memory(0x2000, py.None(), tag);
            st.set_addr_to_ast(0x2000, py.None(), tag);
        }
        c.set_hook_symbolic_memory(0x3000, py.None(), 3);
        c.set_addr_to_ast(0x3000, py.None(), 3);
        a.set_hook_symbolic_memory(0x4000, py.None(), 1);
        a.set_addr_to_ast(0x4000, py.None(), 1);
    });

    // stdin_symbols union: b re-reads "sa" (dup of self) and adds "sb"; c adds
    // "sc". Merged must be self-order-first with dups dropped: [sa, sb, sc].
    a.record_stdin_symbol("sa".to_string(), 8);
    b.record_stdin_symbol("sa".to_string(), 8); // dup -> not re-added
    b.record_stdin_symbol("sb".to_string(), 8);
    c.record_stdin_symbol("sc".to_string(), 8);

    let (m0, m1, m2) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "qwyti2_m0", 1),
            RustBV::symbolic(&s, "qwyti2_m1", 1),
            RustBV::symbolic(&s, "qwyti2_m2", 1),
        )
    };
    let merged = a.merge(&[&b, &c], &[m0, m1, m2]);

    // --- hook_symbolic_memory ---
    let hook = merged.hook_symbolic_memory();
    assert_eq!(hook.len(), 4, "hook: every distinct addr must survive");
    assert_eq!(
        hook.get(&0x1000).map(|(_, s)| *s),
        Some(1),
        "hook 0x1000: self wins"
    );
    assert_eq!(
        hook.get(&0x2000).map(|(_, s)| *s),
        Some(2),
        "hook 0x2000: earlier-other (b) wins the 3-way tie-break, not later-other (c)"
    );
    assert_eq!(
        hook.get(&0x3000).map(|(_, s)| *s),
        Some(3),
        "hook 0x3000: last-other survives"
    );
    assert_eq!(
        hook.get(&0x4000).map(|(_, s)| *s),
        Some(1),
        "hook 0x4000: self-only survives"
    );

    // --- addr_to_ast (identical policy, distinct map) ---
    let a2a = merged.addr_to_ast();
    assert_eq!(
        a2a.len(),
        4,
        "addr_to_ast: every distinct addr must survive"
    );
    assert_eq!(
        a2a.get(&0x1000).map(|(_, s)| *s),
        Some(1),
        "addr_to_ast 0x1000: self wins"
    );
    assert_eq!(
        a2a.get(&0x2000).map(|(_, s)| *s),
        Some(2),
        "addr_to_ast 0x2000: earlier-other (b) wins the tie-break"
    );
    assert_eq!(
        a2a.get(&0x3000).map(|(_, s)| *s),
        Some(3),
        "addr_to_ast 0x3000: last-other survives"
    );
    assert_eq!(
        a2a.get(&0x4000).map(|(_, s)| *s),
        Some(1),
        "addr_to_ast 0x4000: self-only survives"
    );

    // --- symbolic_pages (no size; provenance read from the int AST value) ---
    let pages = merged.symbolic_pages();
    assert_eq!(
        pages.len(),
        4,
        "symbolic_pages: every distinct addr must survive"
    );
    Python::attach(|py| {
        let tag = |addr: u64| pages.get(&addr).unwrap().bind(py).extract::<i64>().unwrap();
        assert_eq!(tag(0x1000), 1, "symbolic_pages 0x1000: self wins");
        assert_eq!(
            tag(0x2000),
            2,
            "symbolic_pages 0x2000: earlier-other (b) wins the tie-break"
        );
        assert_eq!(tag(0x3000), 3, "symbolic_pages 0x3000: last-other survives");
        assert_eq!(tag(0x4000), 1, "symbolic_pages 0x4000: self-only survives");
    });

    // --- stdin_symbols union (dedup by name, self-order-first) ---
    let names: Vec<&str> = merged
        .stdin_symbols()
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["sa", "sb", "sc"],
        "stdin_symbols must union across branches, dedup by name, self-order-first"
    );
}
