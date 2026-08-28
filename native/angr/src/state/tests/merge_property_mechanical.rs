//! Mechanical half of the census property test for `RustSimState::merge`'s
//! field-level policy contract (Harness 1 of the
//! `we-keep-finding-bugs-optimized-acorn.md` plan). The hand-written half is
//! [`super::merge_property_manual`]; this header carries the rationale both
//! halves share.
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
//! intent. Nor did anything assert a *specific* value for the fields with no
//! generated method at all; that gap is `merge_property_manual`'s.
//!
//! Of `RustSimState`'s 43 `#[merge_policy]` fields: **22** are mechanical
//! (get a generated `merge_field_<name>`) and are covered here; **21** are
//! hand-written in `state/fork.rs::RustSimState::merge` and are covered in
//! `merge_property_manual`. (The originating plan estimated 26/17 — see the
//! count-by-policy note in each file's section banner for the actual
//! breakdown; the 17 there only counted `delegate` / `computed` / `joint`,
//! not the 4 additional mechanical-policy fields that opt out via
//! `#[merge_manual]`.)
//!
//! Both halves complement — do not replace — the deeper per-family coverage
//! in the sibling `merge_scalars.rs` / `merge_config.rs` / `merge_heap.rs` /
//! `merge_multi_state.rs` (3-way tie-breaks, tombstone reinstate-then-remove
//! sequencing, etc: those files exist first and stay authoritative for that
//! depth). They are additive to `memory/tests/merge_sidecars.rs`, which
//! censuses `SymbolicMemory`'s own independent
//! `#[derive(angr_macros::MergePolicy)]` one level below the `memory` field's
//! `delegate` label — same census shape, one level up the struct tree.
//!
//! Layout here: one test per policy kind, each merging one pair of states
//! that diverges on every field of that kind at once and asserting the
//! merged value equals a direct call to that field's `merge_field_<name>` —
//! resilient to a field being renamed, since the assertion is "does
//! `merge()` agree with its own generated method", not a hand-coded literal
//! alone (each test still separately pins the generated method's *own*
//! output to a specific expected value, so the whole chain is non-vacuous).

use std::collections::VecDeque;

use super::super::*;
use super::helpers::{base_pair, conds};

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

