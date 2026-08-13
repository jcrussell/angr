//! Census property test for `RustSimState::merge`'s field-level policy
//! contract (Harness 1 of the
//! `we-keep-finding-bugs-optimized-acorn.md` plan).
//!
//! `#[derive(angr_macros::MergePolicy)]` on `RustSimState` (`state/mod.rs`)
//! proves every field carries a `#[merge_policy = "..."]` label, and for the
//! five mechanical policies (`self_wins`/`union`/`max`/`min`/
//! `warn_on_diverge`, minus any field opted out via `#[merge_manual]`)
//! generates a `merge_field_<name>` method whose body *is* the policy. But
//! nothing before this file checked that `RustSimState::merge`'s actual
//! output for a given field agrees with what that generated method computes
//! from the same two states — the derive only guarantees `merge` *calls*
//! `merge_field_<name>` (a private method it doesn't call is `dead_code`,
//! which CI's `-D warnings` rejects), not that the label matches production
//! intent. Nor did anything assert a *specific* value for the 21
//! hand-written fields (`delegate`/`computed`/`joint` policies, plus four
//! more whose policy label is mechanical but which opt out via
//! `#[merge_manual]`: `native_resume_stack`, `stdin_symbols`, `inspection`,
//! `last_time`).
//!
//! Of `RustSimState`'s 43 `#[merge_policy]` fields: **22** are mechanical
//! (get a generated `merge_field_<name>`), **21** are hand-written in
//! `state/fork.rs::RustSimState::merge`. (The originating plan estimated
//! 26/17 — see the module-level count-by-policy note in each section below
//! for the actual breakdown; the 17 there only counted `delegate` /
//! `computed` / `joint`, not the 4 additional mechanical-policy fields that
//! opt out via `#[merge_manual]`.)
//!
//! This file complements — does not replace — the deeper per-family coverage
//! in the sibling `merge_scalars.rs` / `merge_config.rs` / `merge_heap.rs` /
//! `merge_multi_state.rs` (3-way tie-breaks, tombstone reinstate-then-remove
//! sequencing, etc: those files exist first and stay authoritative for that
//! depth). It is additive to `memory/tests/merge_sidecars.rs`, which
//! censuses `SymbolicMemory`'s own independent
//! `#[derive(angr_macros::MergePolicy)]` one level below the `memory` field's
//! `delegate` label here — same census shape, one level up the struct tree.
//!
//! Layout:
//! - Mechanical fields (22): grouped one test per policy kind, each merging
//!   one pair of states that diverges on every field of that kind at once
//!   and asserting the merged value equals a direct call to that field's
//!   `merge_field_<name>` — resilient to a field being renamed, since the
//!   assertion is "does `merge()` agree with its own generated method", not
//!   a hand-coded literal alone (each test still separately pins the
//!   generated method's *own* output to a specific expected value, so the
//!   whole chain is non-vacuous).
//! - Hand-written fields (21): one small test per field or tightly-related
//!   field group, each with a hand-authored expected value reasoned from
//!   `state/fork.rs::RustSimState::merge`'s actual logic.

use std::collections::VecDeque;

use super::super::*;

/// Fork a common ancestor into two independent states so per-branch mutation
/// diverges cleanly from a shared starting point — the same shape every
/// sibling `merge_*.rs` test in this directory uses, and the one
/// `removed_hooks`/`removed_sim_options`/`removed_env_keys` (fork-point-
/// relative tombstones) depend on for correct semantics.
fn base_pair(arch: &str) -> (RustSimState, RustSimState) {
    Python::initialize();
    let ancestor = RustSimState::new(arch).unwrap();
    (ancestor.fork(), ancestor.fork())
}

/// `[cond_self, cond_other]` — a fresh pair of 1-bit symbolic merge
/// conditions rooted in `a`'s solver, exactly as every `a.merge(&[&b], ...)`
/// call in this file and its siblings builds them.
fn conds(a: &RustSimState, tag: &str) -> [RustBV; 2] {
    let s = a.solver().borrow();
    [
        RustBV::symbolic(&s, format!("{tag}_m0"), 1),
        RustBV::symbolic(&s, format!("{tag}_m1"), 1),
    ]
}

// =========================================================================
// Mechanical fields (22 total): self_wins(8) + union(4) + max(4) + min(1) +
// warn_on_diverge(5) = 22. `arch`/`vex_arch` (both self_wins) get their own
// test since diverging them needs two independently-constructed states —
// there is no setter, unlike every other field here.
// =========================================================================

/// `arch`/`vex_arch`: self_wins. Unlike every other field in this file,
/// there is no setter — the only way to diverge them is two independently
/// constructed states, not a fork of a common ancestor.
#[test]
fn mechanical_arch_and_vex_arch_self_wins_across_differing_archs() {
    Python::initialize();
    let a = RustSimState::new("amd64").unwrap();
    let b = RustSimState::new("x86").unwrap();

    let expected_arch = a.merge_field_arch(&[&b]);
    let expected_vex_arch = a.merge_field_vex_arch(&[&b]);
    assert_eq!(
        expected_arch.name(),
        "AMD64",
        "self_wins must compute self's arch, not other's"
    );
    assert_eq!(
        expected_vex_arch,
        a.vex_arch(),
        "self_wins must compute self's vex_arch"
    );
    assert_ne!(
        expected_vex_arch,
        b.vex_arch(),
        "a and b must actually differ for this to be non-vacuous"
    );

    let [m0, m1] = conds(&a, "arch_self_wins");
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(
        merged.arch().name(),
        expected_arch.name(),
        "merge()'s actual arch must equal merge_field_arch's direct computation"
    );
    assert_eq!(
        merged.vex_arch(),
        expected_vex_arch,
        "merge()'s actual vex_arch must equal merge_field_vex_arch's direct computation"
    );
}

/// self_wins (6 of the remaining 8; `arch`/`vex_arch` above): `pc`,
/// `history`, `detailed_history`, `max_history`, `track_history`,
/// `call_stack`. One populated pair, one merge, per-field comparison against
/// the generated `merge_field_*` method.
#[test]
fn mechanical_self_wins_fields_keep_selfs_value() {
    let (mut a, mut b) = base_pair("amd64");

    a.set_pc(0x1000);
    b.set_pc(0x2000);

    a.add_to_history(0x1100);
    b.add_to_history(0x2200);

    a.set_detailed_history(vec![HistoryEntry {
        addr: 0x1100,
        jumpkind: HistoryEntry::JK_BORING,
        jump_target: 0x1104,
    }]);
    b.set_detailed_history(vec![HistoryEntry {
        addr: 0x2200,
        jumpkind: HistoryEntry::JK_CALL,
        jump_target: 0x2300,
    }]);

    a.push_call(0x1000, 0x1010, 0x1005, 0x7fff_0000);
    b.push_call(0x2000, 0x2010, 0x2005, 0x7ffe_0000);

    a.set_max_history(50);
    b.set_max_history(99);

    a.set_track_history(false);
    b.set_track_history(true);

    let expected_pc = a.merge_field_pc(&[&b]);
    let expected_history: VecDeque<u64> = a.merge_field_history(&[&b]);
    let expected_detailed: VecDeque<HistoryEntry> = a.merge_field_detailed_history(&[&b]);
    let expected_max_history = a.merge_field_max_history(&[&b]);
    let expected_track_history = a.merge_field_track_history(&[&b]);
    let expected_call_stack: Vec<CallStackEntry> = a.merge_field_call_stack(&[&b]);

    assert_eq!(expected_pc, 0x1000, "self_wins must compute self's pc");
    assert_eq!(
        expected_history.iter().copied().collect::<Vec<_>>(),
        vec![0x1100]
    );
    assert_eq!(
        expected_detailed.iter().map(|h| h.addr).collect::<Vec<_>>(),
        vec![0x1100]
    );
    assert_eq!(
        expected_call_stack
            .iter()
            .map(|c| c.call_site_addr)
            .collect::<Vec<_>>(),
        vec![0x1000]
    );
    assert_eq!(expected_max_history, 50);
    assert!(!expected_track_history);

    let [m0, m1] = conds(&a, "self_wins_group");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert_eq!(merged.pc(), expected_pc);
    assert_eq!(
        merged.history().iter().copied().collect::<Vec<_>>(),
        expected_history.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        merged
            .detailed_history()
            .iter()
            .map(|h| h.addr)
            .collect::<Vec<_>>(),
        expected_detailed.iter().map(|h| h.addr).collect::<Vec<_>>()
    );
    assert_eq!(
        merged
            .call_stack()
            .iter()
            .map(|c| c.call_site_addr)
            .collect::<Vec<_>>(),
        expected_call_stack
            .iter()
            .map(|c| c.call_site_addr)
            .collect::<Vec<_>>()
    );
    assert_eq!(merged.max_history(), expected_max_history);
    assert_eq!(merged.track_history, expected_track_history);
}

/// union (4): `no_ip_concretization`, `no_symbolic_jump_resolution`,
/// `keep_ip_symbolic`, `force_eager_forks`. All four are boolean SimOption
/// mirrors that OR across branches.
#[test]
fn mechanical_union_bool_fields_take_logical_or() {
    let (a, mut b) = base_pair("amd64");
    b.set_no_ip_concretization(true);
    b.set_no_symbolic_jump_resolution(true);
    b.set_keep_ip_symbolic(true);
    b.set_force_eager_forks(true);

    assert!(
        a.merge_field_no_ip_concretization(&[&b]),
        "false || true must be true"
    );
    assert!(a.merge_field_no_symbolic_jump_resolution(&[&b]));
    assert!(a.merge_field_keep_ip_symbolic(&[&b]));
    assert!(a.merge_field_force_eager_forks(&[&b]));

    let [m0, m1] = conds(&a, "union_bool_group");
    let merged = a.merge(&[&b], &[m0, m1]);

    assert!(!a.no_ip_concretization(), "sanity: a alone stayed false");
    assert!(
        merged.no_ip_concretization(),
        "union must OR in b's true, not keep self's false"
    );
    assert!(merged.no_symbolic_jump_resolution());
    assert!(merged.keep_ip_symbolic());
    assert!(merged.force_eager_forks());
}

/// max (4): `heap_brk`, `posix_brk`, `mmap_base`, `tsc_counter`. Deliberately
/// alternates which branch is larger per field (`posix_brk` has `a` bigger)
/// so the test can't pass via a "always take other" bug hiding behind "max".
#[test]
fn mechanical_max_watermark_fields_take_the_higher_value() {
    let (mut a, mut b) = base_pair("amd64");

    a.set_heap_brk(0x1000);
    b.set_heap_brk(0x2000);
    a.set_posix_brk(0x9000);
    b.set_posix_brk(0x3000);
    a.set_mmap_base(0x4000);
    b.set_mmap_base(0x5000);
    a.set_tsc_counter(crate::vex::dirty::TSC_INITIAL + 5 * crate::vex::dirty::TSC_STEP);
    b.set_tsc_counter(crate::vex::dirty::TSC_INITIAL + 2 * crate::vex::dirty::TSC_STEP);

    let expected_heap_brk = a.merge_field_heap_brk(&[&b]);
    let expected_posix_brk = a.merge_field_posix_brk(&[&b]);
    let expected_mmap_base = a.merge_field_mmap_base(&[&b]);
    let expected_tsc = a.merge_field_tsc_counter(&[&b]);

    assert_eq!(expected_heap_brk, 0x2000, "max must take b's higher value");
    assert_eq!(
        expected_posix_brk, 0x9000,
        "max must keep a's higher value, not blindly take other's"
    );
    assert_eq!(expected_mmap_base, 0x5000);
    assert_eq!(
        expected_tsc,
        crate::vex::dirty::TSC_INITIAL + 5 * crate::vex::dirty::TSC_STEP
    );

    let [m0, m1] = conds(&a, "max_watermark_group");
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(merged.heap_brk(), expected_heap_brk);
    assert_eq!(merged.posix_brk(), expected_posix_brk);
    assert_eq!(merged.mmap_base(), expected_mmap_base);
    assert_eq!(merged.tsc_counter(), expected_tsc);
}

/// min (1): `cgc_allocation_base`. The CGC allocator grows downward, so the
/// anti-alias combinator is the opposite of the up-growing watermarks above.
#[test]
fn mechanical_min_field_takes_the_lower_value() {
    let (mut a, mut b) = base_pair("amd64");
    a.set_cgc_allocation_base(0xB800_0000);
    b.set_cgc_allocation_base(0xB700_0000);

    let expected = a.merge_field_cgc_allocation_base(&[&b]);
    assert_eq!(
        expected, 0xB700_0000,
        "min must take b's lower value (downward-growing allocator)"
    );

    let [m0, m1] = conds(&a, "min_field");
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(merged.cgc_allocation_base(), expected);
}

/// warn_on_diverge (5): `concretizer`, `getopt_optind`, `getopt_optchar`,
/// `getopt_extern`, `ctype_loc`. Value-wise these behave exactly like
/// self_wins (only the loudness of the drop differs, which is not
/// mechanically checkable here without fighting the process-global logger —
/// see `test_native_resume_stack_diverges_detects_depth_mismatch` in
/// `merge_scalars.rs` for how the divergence-detection half is tested
/// directly instead).
#[test]
fn mechanical_warn_on_diverge_fields_keep_selfs_value() {
    let (mut a, mut b) = base_pair("amd64");

    let a_conc = AddressConcretizer {
        max_solutions: 111,
        ..Default::default()
    };
    let b_conc = AddressConcretizer {
        max_solutions: 222,
        ..Default::default()
    };
    a.set_concretizer(a_conc.clone());
    b.set_concretizer(b_conc);

    a.set_getopt_cursor(3, 1);
    b.set_getopt_cursor(7, 2);

    let a_extern = GetoptExternAddrs {
        optind: Some(0x1000),
        optarg: Some(0x1008),
        optopt: Some(0x1010),
    };
    let b_extern = GetoptExternAddrs {
        optind: Some(0x2000),
        optarg: Some(0x2008),
        optopt: Some(0x2010),
    };
    a.set_getopt_extern(a_extern);
    b.set_getopt_extern(b_extern);

    let a_ctype = CtypeLocPtrs {
        b: Some(0x3000),
        tolower: Some(0x3008),
        toupper: Some(0x3010),
    };
    let b_ctype = CtypeLocPtrs {
        b: Some(0x4000),
        tolower: Some(0x4008),
        toupper: Some(0x4010),
    };
    a.set_ctype_loc(a_ctype);
    b.set_ctype_loc(b_ctype);

    let expected_conc = a.merge_field_concretizer(&[&b]);
    let expected_optind = a.merge_field_getopt_optind(&[&b]);
    let expected_optchar = a.merge_field_getopt_optchar(&[&b]);
    let expected_extern = a.merge_field_getopt_extern(&[&b]);
    let expected_ctype = a.merge_field_ctype_loc(&[&b]);

    assert_eq!(
        expected_conc, a_conc,
        "warn_on_diverge keeps self's value like self_wins"
    );
    assert_eq!(expected_optind, 3);
    assert_eq!(expected_optchar, 1);
    assert_eq!(expected_extern, a_extern);
    assert_eq!(expected_ctype, a_ctype);

    let [m0, m1] = conds(&a, "warn_on_diverge_group");
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(merged.concretizer, expected_conc);
    assert_eq!(merged.getopt_optind, expected_optind);
    assert_eq!(merged.getopt_optchar, expected_optchar);
    assert_eq!(merged.getopt_extern(), expected_extern);
    assert_eq!(merged.ctype_loc(), expected_ctype);
}

// =========================================================================
// Hand-written fields (21 total): delegate(4) + computed(5) + joint(8) +
// merge_manual-opted-out mechanical(4) = 21. No generated method exists to
// check against, so each assertion is a hand-authored expected value
// reasoned from `state/fork.rs::RustSimState::merge`.
// =========================================================================

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
