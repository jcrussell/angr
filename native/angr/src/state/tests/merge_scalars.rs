//! Merging the scalar/watermark state fields: other-only overlay union and max
//! brk, the TSC counter's fork-and-merge watermark rule, the symbolic max of
//! `last_time`, and native resume-stack depth-mismatch divergence detection.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-ph300.51: RustSimState::merge must union the three Python-AST overlay
// maps (not take only self's) and carry the furthest-advanced allocator
// watermarks — otherwise a branch-B symbolic overlay byte is silently lost and
// the merged state's next malloc can alias live allocations from B's ITE arm.
#[test]
fn test_merge_unions_other_only_overlays_and_takes_max_brk() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // Self (a) owns an overlay at 0x1000; other (b) owns a *distinct* overlay at
    // 0x2000 plus a conflicting one at 0x1000. b also advances the allocators.
    Python::attach(|py| {
        a.set_hook_symbolic_memory(0x1000, py.None(), 8);
        b.set_hook_symbolic_memory(0x1000, py.None(), 4); // conflict -> keep self
        b.set_hook_symbolic_memory(0x2000, py.None(), 8); // other-only -> survive
    });
    a.set_heap_brk(0x10_0000);
    b.set_heap_brk(0x20_0000); // b allocated further
    a.set_posix_brk(0x30_0000);
    b.set_posix_brk(0x11_0000); // a's is higher here
    a.set_mmap_base(0x40_0000);
    b.set_mmap_base(0x50_0000);

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30051_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "ph30051_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    // Other-only overlay survived; conflicting key kept self's size (8, not 4).
    assert_eq!(
        merged.hook_symbolic_memory().len(),
        2,
        "other-only overlay must survive the merge"
    );
    assert!(
        merged.hook_symbolic_memory().contains_key(&0x2000),
        "b's 0x2000 overlay must be present in the merged state"
    );
    assert_eq!(
        merged
            .hook_symbolic_memory()
            .get(&0x1000)
            .map(|(_, sz)| *sz),
        Some(8),
        "conflicting overlay must keep self's entry (size 8), not other's (4)"
    );

    // Allocator watermarks take the furthest-advanced value per field.
    assert_eq!(merged.heap_brk(), 0x20_0000, "heap_brk must be max(a, b)");
    assert_eq!(merged.posix_brk(), 0x30_0000, "posix_brk must be max(a, b)");
    assert_eq!(merged.mmap_base(), 0x50_0000, "mmap_base must be max(a, b)");
}

// angr-9ke6b.173: the simulated RDTSC counter moved from a process-wide
// AtomicU64 onto RustSimState. It must default to TSC_INITIAL, survive a fork
// (a child continues the parent's clock and then diverges independently), and
// merge as a max watermark so the merged state's next RDTSC can't read earlier
// than a value one of the arms already observed.
#[test]
fn test_tsc_counter_forks_and_merges_as_watermark() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    assert_eq!(
        a.tsc_counter(),
        crate::vex::dirty::TSC_INITIAL,
        "a fresh state starts at the fixed initial TSC, not wherever some \
         other state left a global counter"
    );

    a.set_tsc_counter(crate::vex::dirty::TSC_INITIAL + 5 * crate::vex::dirty::TSC_STEP);
    let mut child = a.fork();
    assert_eq!(
        child.tsc_counter(),
        a.tsc_counter(),
        "fork inherits the parent's clock"
    );

    // Diverge: the child executes more RDTSCs than the parent.
    child.set_tsc_counter(child.tsc_counter() + 3 * crate::vex::dirty::TSC_STEP);
    assert_ne!(child.tsc_counter(), a.tsc_counter());

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "tsc173_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "tsc173_m1", 1)
    };
    let expected = child.tsc_counter();
    let merged = a.merge(&[&child], &[m0, m1]);
    assert_eq!(
        merged.tsc_counter(),
        expected,
        "merge must take the furthest-advanced TSC across branches"
    );
}

// angr-sqfj8.88: `last_time` is `Option<RustBV>`, the symbolic analogue of the
// `tsc_counter` watermark above — merge must never let it read as though time
// ran backwards relative to any branch. Since `RustBV` has no `Ord`, this
// takes a real symbolic maximum (`uge` + `ite`) rather than dropping straight
// to `self`'s value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_takes_symbolic_max_of_last_time() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = a.fork();
    a.set_last_time(RustBV::concrete(100, 64));
    b.set_last_time(RustBV::concrete(250, 64));

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_88_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_88_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    let last = merged
        .last_time()
        .expect("last_time must survive the merge");
    let val = merged.solver().borrow().eval(last);
    assert_eq!(
        val,
        Some(250),
        "merge must take the later branch's time value, not silently keep \
         self's smaller one"
    );
}

/// angr-sqfj8.86: merge must actually detect a `native_resume_stack` length
/// mismatch across branches — before this fix nothing computed divergence
/// for this field at all, so `warn_config_divergence` never fired no matter
/// how far branches drifted. Exercises `native_resume_stack_diverges`
/// directly (the detection logic `RustSimState::merge` feeds into
/// `warn_config_divergence`) rather than trying to observe the resulting
/// `log::warn!` call: `log::set_logger` is a process-global one-shot
/// singleton, and `engine_tests.rs`'s `set_rust_log_level_accepts_levels_and_specs`
/// already claims that slot in the same test binary, so a competing logger
/// installed here would race it non-deterministically.
#[test]
fn test_native_resume_stack_diverges_detects_depth_mismatch() {
    let mut a = RustSimState::new("amd64").unwrap();
    let b = a.fork();
    assert!(
        !fork::native_resume_stack_diverges(&a, &[&b]),
        "equal-depth (both empty) stacks must not read as diverged"
    );

    a.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 0,
        saved_args: vec![RustBV::concrete(0x601000, 64)],
        caller_return_addr: 0x400600,
    });
    assert!(
        fork::native_resume_stack_diverges(&a, &[&b]),
        "a length-1 vs length-0 native_resume_stack must be detected as diverged"
    );

    // The merge itself still self-wins regardless of divergence detection
    // (see `RustSimState::merge`'s doc) — confirm that carry survives too.
    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_86_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "sqfj8_86_m1", 1)
    };
    let merged = a.merge(&[&b], &[m0, m1]);
    assert_eq!(
        merged.native_resume_stack().len(),
        1,
        "merge must keep self's frame (self-wins), not silently b's empty stack"
    );
}
