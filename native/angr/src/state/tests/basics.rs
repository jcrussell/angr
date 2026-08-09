//! State construction and fork basics: identity/lineage on fork, plus the
//! per-state config knobs (`getopt` cursor, extern-address allocator, native
//! resume stack) that must start at their documented default and stay isolated
//! across a fork.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

#[test]
fn test_state_creation() {
    let state = RustSimState::new("amd64").unwrap();
    assert_eq!(state.vex_arch(), VexArch::AMD64);
    assert_eq!(state.pc(), 0);
}

#[test]
fn test_state_fork() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.set_pc(0x1000);
    state1.set_register("rax", RustBV::concrete(42, 64));

    let state2 = state1.fork();

    // Both should have same values
    assert_eq!(state2.pc(), 0x1000);
    assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

    // Different state IDs
    assert_ne!(state1.state_id(), state2.state_id());

    // state2's parent should be state1
    assert_eq!(state2.parent_id(), Some(state1.state_id()));
}

#[test]
fn test_getopt_cursor_default_and_fork_isolation() {
    // Foundation slice for native getopt parity (bead angr-bhk0a): the
    // per-state getopt cursor must default to (optind=1, optchar=0) — the
    // glibc/Python `state.libc.getopt_optind`/`getopt_optchar` defaults — and
    // be copied (not shared) across fork so each path scans argv independently.
    let mut parent = RustSimState::new("amd64").unwrap();
    assert_eq!(parent.getopt_cursor(), (1, 0));

    parent.set_getopt_cursor(4, 2);
    let mut child = parent.fork();
    assert_eq!(child.getopt_cursor(), (4, 2));

    // Mutating the child must not disturb the parent (CoW isolation).
    child.set_getopt_cursor(9, 0);
    assert_eq!(child.getopt_cursor(), (9, 0));
    assert_eq!(parent.getopt_cursor(), (4, 2));
}

#[test]
fn test_getopt_extern_addrs_default_and_fork_isolation() {
    // bhk0a.1: the loader-resolved getopt(3) extern-global addresses
    // (optind/optarg/optopt) default to None (no init-push yet -> native proc
    // defers to Python) and are copied (not shared) across fork.
    let mut parent = RustSimState::new("amd64").unwrap();
    let d = parent.getopt_extern();
    assert_eq!((d.optind, d.optarg, d.optopt), (None, None, None));

    parent.set_getopt_extern(GetoptExternAddrs {
        optind: Some(0x601000),
        optarg: Some(0x601008),
        optopt: Some(0x601010),
    });
    let mut child = parent.fork();
    let c = child.getopt_extern();
    assert_eq!(
        (c.optind, c.optarg, c.optopt),
        (Some(0x601000), Some(0x601008), Some(0x601010))
    );

    // CoW isolation: mutating the child must not disturb the parent.
    child.set_getopt_extern(GetoptExternAddrs::default());
    assert_eq!(child.getopt_extern().optind, None);
    assert_eq!(parent.getopt_extern().optind, Some(0x601000));
}

#[test]
fn test_native_resume_stack_default_and_fork_isolation() {
    // angr-pn3w8 (S1): the native sub-call resume stack defaults to empty and
    // is copied (not shared) across fork so each path resumes its own pending
    // sub-calls. No dispatcher yet — this only exercises the per-state plumbing.
    let mut parent = RustSimState::new("amd64").unwrap();
    assert!(parent.native_resume_stack().is_empty());

    parent.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 0,
        saved_args: vec![
            RustBV::concrete(0x601000, 64),
            RustBV::concrete(0x400500, 64),
        ],
        caller_return_addr: 0x400600,
    });
    assert_eq!(parent.native_resume_stack().len(), 1);

    let mut child = parent.fork();
    assert_eq!(child.native_resume_stack().len(), 1);
    let frame = &child.native_resume_stack()[0];
    assert_eq!(frame.proc_name, "pthread_once");
    assert_eq!(frame.resume_tag, 0);
    assert_eq!(frame.saved_args.len(), 2);
    assert_eq!(frame.saved_args[0].as_u64(), Some(0x601000));

    // CoW isolation: popping in the child must not disturb the parent's stack.
    let popped = child.pop_native_resume_frame();
    assert!(popped.is_some());
    assert!(child.native_resume_stack().is_empty());
    assert_eq!(parent.native_resume_stack().len(), 1);
}
