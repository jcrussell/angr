//! Hand-written half of the `RustSimState::merge` field-policy census —
//! the **21** fields `#[derive(angr_macros::MergePolicy)]` generates no
//! `merge_field_<name>` for, so nothing but a hand-authored expected value
//! can check them.
//!
//! The mechanical half (22 fields, checked against their generated methods)
//! and the shared rationale for why this census exists at all live in
//! [`super::merge_property_mechanical`]; read that header first.
//!
//! Three of the four policy kinds here never get a generated method:
//! `delegate` (the field's own type owns the merge — `registers`, `memory`,
//! `solver`, `heap_metadata`), `computed` (the merged value is a function of
//! neither branch's value — fresh ids, tombstone sets) and `joint` (the
//! merge reads more than one field at once — a map and its removal
//! tombstone). The fourth kind is four fields whose policy *label* is
//! mechanical but which opt out via `#[merge_manual]`:
//! `native_resume_stack`, `stdin_symbols`, `inspection`, `last_time`.
//!
//! Every assertion below is reasoned from
//! `state/fork.rs::RustSimState::merge`'s actual logic, so a behaviour
//! change there must be reflected here by hand — that is the point: for
//! these fields there is no generated method to cross-check against.

use super::super::*;
use super::helpers::{base_pair, conds};


/// computed (2 of 5; the other 3 — `removed_hooks`/`removed_env_keys`/
/// `removed_sim_options` — are exercised alongside the `joint` fields they're
/// tombstones for, below): `state_id` gets a fresh monotonic id, `parent_id`
/// is `Some(self.state_id)` — not `None` and not `other`'s id.
#[test]
fn computed_fields_get_fresh_identity_not_inherited_from_either_branch() {
    let (a, b) = base_pair("amd64");

    let [m0, m1] = conds(&a, "computed_identity");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        merged.state_id() > a.state_id() && merged.state_id() > b.state_id(),
        "state_id must be freshly minted, strictly newer than either input"
    );
    assert_eq!(
        merged.parent_id(),
        Some(a.state_id()),
        "parent_id must be self's (a's) id — not None, and not b's"
    );
}

/// delegate: `registers`. `RegisterFile::merge` (its own independent
/// `#[derive(MergePolicy)]`, `arch/mod.rs`) has its own deep ITE-shape
/// coverage in `arch/mod_tests.rs`; this proves `RustSimState::merge` wires
/// its result in rather than dropping a branch-only concrete register write.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn delegate_registers_merge_preserves_a_branch_only_concrete_write() {
    let (a, mut b) = base_pair("amd64");
    b.set_register("rbx", RustBV::concrete(0x4242, 64));

    let [m0, m1] = conds(&a, "reg_delegate");

    let merged_a_active = a.merge(&[&b], &[m0.clone(), m1.clone()]);
    merged_a_active.solver().borrow().assume_true(&m0);
    let a_view = merged_a_active
        .get_register("rbx")
        .and_then(|v| merged_a_active.solver().borrow().eval(&v));
    assert_eq!(
        a_view,
        Some(0),
        "under a's own branch condition, rbx must read a's untouched (zero) value"
    );

    let merged_b_active = a.merge(&[&b], &[m0, m1.clone()]);
    merged_b_active.solver().borrow().assume_true(&m1);
    let b_view = merged_b_active
        .get_register("rbx")
        .and_then(|v| merged_b_active.solver().borrow().eval(&v));
    assert_eq!(
        b_view,
        Some(0x4242),
        "under b's own branch condition, rbx must read b's write — not be dropped"
    );
}

/// delegate: `memory`. Disjoint pages so no ITE is needed to prove the
/// point: `SymbolicMemory::merge` (own census in
/// `memory/tests/merge_sidecars.rs`) has to actually union b's page in, not
/// keep only self's.
#[test]
fn delegate_memory_merge_preserves_a_branch_only_store() {
    let (mut a, mut b) = base_pair("amd64");
    a.map_memory_data(0x5000_0000, &[0x11], Permission::RWX);
    b.map_memory_data(0x5000_1000, &[0x22], Permission::RWX);

    let [m0, m1] = conds(&a, "mem_delegate");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(
        merged
            .memory_load(0x5000_0000, 1)
            .ok()
            .and_then(|v| v.as_u128()),
        Some(0x11),
        "self's own page must survive the merge"
    );
    assert_eq!(
        merged
            .memory_load(0x5000_1000, 1)
            .ok()
            .and_then(|v| v.as_u128()),
        Some(0x22),
        "a page only b touched must be unioned in, not dropped"
    );
}

/// delegate: `solver`. Each branch asserts a *different* value for the same
/// shared symbol; the merged solver must let each branch's own condition
/// force its own value, proving both arms' constraints survived (not just
/// self's).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn delegate_solver_merge_preserves_each_branchs_constraint() {
    let (a, b) = base_pair("amd64");

    let x = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "solver_delegate_x", 32)
    };
    {
        let s = a.solver().borrow();
        let c = x.eq(&RustBV::concrete(5, 32), &s);
        s.assume_true(&c);
    }
    {
        let s = b.solver().borrow();
        let c = x.eq(&RustBV::concrete(9, 32), &s);
        s.assume_true(&c);
    }

    let [m0, m1] = conds(&a, "solver_delegate");

    let merged_a_active = a.merge(&[&b], &[m0.clone(), m1.clone()]);
    merged_a_active.solver().borrow().assume_true(&m0);
    assert_eq!(
        merged_a_active.solver().borrow().eval(&x),
        Some(5),
        "under a's own branch condition the merged solver must see a's constraint"
    );

    let merged_b_active = a.merge(&[&b], &[m0, m1.clone()]);
    merged_b_active.solver().borrow().assume_true(&m1);
    assert_eq!(
        merged_b_active.solver().borrow().eval(&x),
        Some(9),
        "under b's own branch condition the merged solver must see b's constraint too"
    );
}

/// delegate: `heap_metadata`. Deeper NativeRealloc-over-read regression
/// coverage lives in `merge_heap.rs`; this proves census inclusion with a
/// smaller setup (two non-aliasing allocations rather than a shared-address
/// one).
#[test]
fn delegate_heap_metadata_unions_allocation_records_across_branches() {
    let (mut a, mut b) = base_pair("amd64");
    let a_addr = a.heap_alloc(0x10);
    b.set_heap_brk(b.heap_brk() + 0x1000); // avoid aliasing a's block
    let b_addr = b.heap_alloc(0x20);
    assert_ne!(a_addr, b_addr, "sanity: the two allocations must not alias");

    let [m0, m1] = conds(&a, "heap_meta_delegate");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(
        merged.heap_metadata().alloc_size(a_addr),
        Some(0x10),
        "self's allocation must survive"
    );
    assert_eq!(
        merged.heap_metadata().alloc_size(b_addr),
        Some(0x20),
        "branch-only allocation must be unioned in via HeapMetadata::union_from, not dropped"
    );
}

/// joint (6 of 8; `fs` and the overlay maps get their own tests below):
/// `hooks`+`removed_hooks`, `sim_options`+`removed_sim_options`,
/// `environment`+`removed_env_keys`. Exhaustive 3-way-tie and
/// reinstate-after-remove coverage lives in `merge_config.rs`; this census
/// entry proves all six fields are wired into ONE merge together (addition
/// AND removal, on both hooks/options/env at once).
#[test]
fn joint_config_like_fields_union_additions_and_respect_removal_tombstones() {
    Python::initialize();
    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.add_hook(0x400000);
    ancestor.set_option("SHORT_READS", true);
    ancestor.setenv(b"PATH".to_vec(), b"/a".to_vec());

    let mut a = ancestor.fork();
    let mut b = ancestor.fork();

    // a removes everything the ancestor had; b adds something new instead.
    a.remove_hook(0x400000);
    a.set_option("SHORT_READS", false);
    a.unsetenv(b"PATH");
    b.add_hook(0x500000);
    b.set_option("OTHER_ONLY", true);
    b.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let [m0, m1] = conds(&a, "config_union");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(
        !merged.is_hooked(0x400000),
        "a's removal wins (removed_hooks tombstone)"
    );
    assert!(
        merged.is_hooked(0x500000),
        "b's brand-new hook is unioned in (hooks)"
    );
    assert!(
        !merged.has_option("SHORT_READS"),
        "a's set_option(false) wins (removed_sim_options tombstone)"
    );
    assert!(
        merged.has_option("OTHER_ONLY"),
        "b's brand-new option is unioned in (sim_options)"
    );
    assert!(
        merged.environment().get(b"PATH".as_slice()).is_none(),
        "a's unsetenv wins (removed_env_keys tombstone)"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"HOME".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "b's brand-new env var is unioned in (environment)"
    );
}

/// joint (1 of 8): `cgc_sinkholes`. Unlike the other `joint` fields above,
/// this one INTERSECTS rather than unions — a freed region only unblocks
/// `allocate()` reuse if every branch agreed it was freed (see the
/// `cgc_sinkholes` comment in `fork.rs::merge`). `cgc_allocation_base`'s
/// `min` policy is covered separately by
/// `mechanical_min_field_takes_the_lower_value`; deeper 3-branch coverage of
/// this field lives in `merge_heap.rs`.
#[test]
fn joint_cgc_sinkholes_intersects_freed_regions_across_branches() {
    Python::initialize();
    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.cgc_add_sinkhole(0x9000, 0x100);
    let mut a = ancestor.fork();
    let b = ancestor.fork();
    a.cgc_add_sinkhole(0xA000, 0x100); // self-only -> dropped

    let [m0, m1] = conds(&a, "cgc_sinkholes");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(
        merged.cgc_sinkholes(),
        &[(0x9000, 0x100)],
        "only the region freed on every branch survives; a self-only region must be dropped"
    );
}

/// joint (3 of 8): `symbolic_pages`, `hook_symbolic_memory`, `addr_to_ast` —
/// the three Python-AST overlay maps, all sharing the "union, earlier state
/// wins a conflict" policy. Deep 3-way tie-break coverage lives in
/// `merge_multi_state.rs`; this census entry proves all three maps are
/// wired into ONE merge together.
#[test]
fn joint_py_ast_overlay_maps_union_with_self_priority_on_conflict() {
    Python::initialize();
    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    Python::attach(|py| {
        a.set_hook_symbolic_memory(0x1000, py.None(), 4); // self-only key
        b.set_hook_symbolic_memory(0x1000, py.None(), 8); // conflict -> self (size 4) wins
        b.set_hook_symbolic_memory(0x2000, py.None(), 8); // other-only -> survives

        a.set_addr_to_ast(0x3000, py.None(), 4);
        b.set_addr_to_ast(0x4000, py.None(), 8);

        let ast = |n: i64| n.into_pyobject(py).unwrap().into_any().unbind();
        let mut pb = std::collections::HashMap::new();
        pb.insert(0x5000u64, ast(1));
        b.replace_symbolic_pages(pb);
    });

    let [m0, m1] = conds(&a, "overlay_union");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(
        merged.hook_symbolic_memory().get(&0x1000).map(|(_, s)| *s),
        Some(4),
        "hook_symbolic_memory conflict keeps self's entry"
    );
    assert!(
        merged.hook_symbolic_memory().contains_key(&0x2000),
        "hook_symbolic_memory other-only entry is unioned in"
    );
    assert!(
        merged.addr_to_ast().contains_key(&0x3000) && merged.addr_to_ast().contains_key(&0x4000),
        "addr_to_ast unions both branches' entries"
    );
    assert!(
        merged.symbolic_pages().contains_key(&0x5000),
        "symbolic_pages unions an other-only entry too"
    );
}

/// joint (1 of 8): `fs`. Longest-stdout-wins policy; full tie-break and
/// fd-drop-warn coverage lives in `merge_multi_state.rs`.
#[test]
fn joint_fs_keeps_the_longest_stdout_branch() {
    Python::initialize();
    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();
    assert!(a.write_stdout(b"hi"));
    assert!(b.write_stdout(b"hello!"));

    let [m0, m1] = conds(&a, "fs_longest_stdout");
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(
        merged.stdout_buffer(),
        b"hello!",
        "merged fs must be the longer-stdout branch (b), not self's"
    );
}

/// merge_manual (1 of 4): `stdin_symbols`. Labelled `union` but hand-written
/// since the union is set-like over a `Vec` keyed by name, not a bool OR.
#[test]
fn manual_stdin_symbols_unions_by_name_dedup_self_first() {
    Python::initialize();
    let mut ancestor = RustSimState::new("amd64").unwrap();
    ancestor.record_stdin_symbol("shared".to_string(), 8);
    let a = ancestor.fork();
    let mut b = ancestor.fork();
    b.record_stdin_symbol("shared".to_string(), 8); // dup -> not re-added
    b.record_stdin_symbol("only_b".to_string(), 16);

    let [m0, m1] = conds(&a, "stdin_union");
    let merged = a.merge(&[&b], &[m0, m1]);

    let names: Vec<&str> = merged
        .stdin_symbols()
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["shared", "only_b"],
        "union by name, dedup, self-order-first"
    );
}

/// merge_manual (1 of 4): `native_resume_stack`. Labelled `warn_on_diverge`
/// but hand-written since the divergence test is per-frame length, not a
/// plain `!=`. The divergence-DETECTION half
/// (`fork::native_resume_stack_diverges`) is unit-tested directly in
/// `merge_scalars.rs`; this proves the merged VALUE keeps self's frames.
#[test]
fn manual_native_resume_stack_keeps_selfs_frames_even_when_b_has_more() {
    let (mut a, mut b) = base_pair("amd64");
    a.push_native_resume_frame(NativeResumeFrame {
        proc_name: "malloc".to_string(),
        resume_tag: 7,
        saved_args: vec![],
        caller_return_addr: 0x401000,
    });
    b.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 1,
        saved_args: vec![],
        caller_return_addr: 0x400000,
    });
    b.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once_inner".to_string(),
        resume_tag: 2,
        saved_args: vec![],
        caller_return_addr: 0x400010,
    });

    let [m0, m1] = conds(&a, "native_resume");
    let merged = a.merge(&[&b], &[m0, m1]);
    let stack = merged.native_resume_stack();
    assert_eq!(
        stack.len(),
        1,
        "self (a)'s single frame wins; b's two extra frames are dropped, not unioned"
    );
    assert_eq!(
        stack[0].resume_tag, 7,
        "the surviving frame must be self (a)'s own frame, not one of b's"
    );
}

/// merge_manual (1 of 4): `inspection`. Labelled `warn_on_diverge` but
/// hand-written since the divergence check only compares `enabled_mask`
/// (angr-sqfj8.89) — the whole manager, including a losing branch's
/// `events`/`event_counts`, is otherwise dropped down to self's copy with no
/// warning. That drop was previously only implied by doc comments, never
/// pinned by a test.
#[test]
fn manual_inspection_keeps_selfs_events_and_drops_others_silently() {
    let (mut a, mut b) = base_pair("amd64");
    a.inspection_mut().enable(InspectEvent::MemRead);
    a.inspect_mem_read(0x1000, 4);
    b.inspection_mut().enable(InspectEvent::MemWrite);
    b.inspect_mem_write(0x2000, 8);

    let [m0, m1] = conds(&a, "inspection_drop");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(
        merged.inspection().event_counts()[InspectEvent::MemRead as usize],
        1,
        "self's recorded event count survives"
    );
    assert_eq!(
        merged.inspection().event_counts()[InspectEvent::MemWrite as usize],
        0,
        "b's recorded event/count is dropped silently (angr-sqfj8.89): only \
         enabled_mask is compared for divergence, not events/event_counts"
    );
    assert!(
        merged.inspection().is_enabled(InspectEvent::MemRead),
        "self's enabled mask survives"
    );
    assert!(
        !merged.inspection().is_enabled(InspectEvent::MemWrite),
        "b's enabled bit is not unioned in — the whole manager is self's clone"
    );
}

/// merge_manual (1 of 4): `last_time`. Labelled `max` but hand-written since
/// `Option<RustBV>` has no `Ord`; the maximum is built symbolically via
/// `uge`+`ite`. Deeper coverage (including the `None`-vs-`Some` fold arms)
/// lives in `merge_scalars.rs`; this is the census entry.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn manual_last_time_takes_the_symbolic_max_across_branches() {
    // b bigger than a: rules out "merge always keeps self's smaller value".
    let (mut a, mut b) = base_pair("amd64");
    a.set_last_time(RustBV::concrete(100, 64));
    b.set_last_time(RustBV::concrete(250, 64));

    let [m0, m1] = conds(&a, "last_time_max_b_bigger");
    let merged = a.merge(&[&b], &[m0, m1]);
    let last = merged.last_time().expect("last_time must survive the merge");
    let val = merged.solver().borrow().eval(last);
    assert_eq!(
        val,
        Some(250),
        "merge must take the later branch's time (b), not self's smaller one"
    );

    // a bigger than b: rules out "merge always takes other's value" — the
    // pairing missing from the original test, which only ever had b win.
    let (mut a2, mut b2) = base_pair("amd64");
    a2.set_last_time(RustBV::concrete(999, 64));
    b2.set_last_time(RustBV::concrete(3, 64));

    let [m0, m1] = conds(&a2, "last_time_max_a_bigger");
    let merged2 = a2.merge(&[&b2], &[m0, m1]);
    let last2 = merged2
        .last_time()
        .expect("last_time must survive the merge");
    let val2 = merged2.solver().borrow().eval(last2);
    assert_eq!(
        val2,
        Some(999),
        "merge must take the larger branch's time (a), not other's smaller one"
    );
}
