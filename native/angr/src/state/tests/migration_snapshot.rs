//! Cross-context migration via snapshot: Multi-cell bytes, resume stack, Python
//! overlays, the wrong-context reattach error, and a migration across a real
//! OS thread.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-1ilq.1: the SAFE cross-worker migration twin of
// `test_translate_state_cross_context`. Instead of `translate_state` (which
// reads the *source* context cross-thread — unsound under work-stealing,
// hazard C), migration serializes the state to context-free bytes on the
// owner and rebuilds the ASTs in the stealing worker's own context. The same
// falsifiable claim must hold: a symbolic register pinned by a path constraint
// re-evaluates to its witness after the round-trip through a fresh context.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_via_snapshot_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xdead_beef, 64), &s)
    };
    state.add_constraint(constraint);
    let sid = state.state_id();

    let original = Context::thread_local();
    let target = Context::new(&Config::new());
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    // Owner serializes under the source context (current thread-local)...
    let payload = state.detach_for_migration();
    // ...stealer swaps its context in, then rebuilds — every AST is minted in
    // `target`, with no read of the source context.
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");
    let rax_t = migrated.get_register("rax").expect("rax present");
    let got = migrated.solver().borrow().eval(&rax_t);
    let sat = migrated.solver().borrow().is_sat();
    Context::set_thread_local(&original);

    assert_eq!(
        got,
        Some(0xdead_beef),
        "migrated rax must resolve via the transferred constraint",
    );
    assert!(sat, "migrated state's solver must remain SAT");
    assert_eq!(
        migrated.state_id(),
        sid,
        "migration preserves identity (same state_id)",
    );
}

// angr-sqfj8.71: `detach_for_migration` (state/migration.rs) must flush
// Multi cells before serializing, else bytes installed by the lazy
// symbolic-address store path (`store_symbolic_unified` resolving to
// `Multiple`) are silently absent on the stealing worker — the exact
// "silently losing data on state migration" shape the bug describes.
// `RustSimState::to_snapshot`/`to_serialized` now flush internally
// (`flush_memory`), so this is a compiler-enforced invariant
// (`SymbolicMemory::to_snapshot` requires a `MultiFlushed` proof), not just a
// call-site convention; this test proves the DATA survives the round trip.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_carries_multi_cell_bytes() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);

    let (addr_var, val_bv) = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "sqfj8_71_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        (addr_var, RustBV::concrete(0xCAFE_BABEu128, 32))
    };
    let concretizer = crate::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::concretize::AddressConcretizer::new()
    };
    let ctx = state.solver().clone();
    let ctx = ctx.borrow();
    state
        .memory_mut()
        .store_symbolic_unified(addr_var, val_bv, &ctx, &concretizer)
        .expect("Multi store must succeed");
    drop(ctx);
    assert_ne!(
        state.memory().multi_cell_count(),
        0,
        "precondition: the store must have installed Multi cells"
    );

    let target = Context::new(&Config::new());
    let payload = state.detach_for_migration();
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");

    let loaded = migrated
        .memory()
        .get_symbolic_object(0x1000)
        .expect("byte 0x1000 must be a symbolic object after migration, not silently dropped");
    assert_eq!(
        loaded.width(),
        32,
        "migrated Multi-cell byte must round-trip at its stored width"
    );
}

// angr-1ilq.1: migration parity with
// `test_translate_state_resume_stack_cross_context` over an A->B->A' round
// trip. Proves the snapshot transport carries `native_resume_stack` (the
// angr-1ilq.2 field) faithfully across two distinct contexts: a symbolic
// saved_arg re-proves its witness at every hop; a concrete saved_arg clones
// verbatim. Each hop serializes under the context that is then live, so a
// half-rebuilt foreign-context AST would surface here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_resume_stack_roundtrip() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_saved_arg", 64)
    };
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xCAFE_F00D_DEAD_BEEF, 64), &s)
    };
    state.add_constraint(constraint);
    state.push_native_resume_frame(NativeResumeFrame {
        proc_name: "ilq1_proc".to_string(),
        resume_tag: 7,
        saved_args: vec![x.clone(), RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x400600,
    });

    let original = Context::thread_local();

    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: migrated solver must remain SAT",
        );
        let stack = twin.native_resume_stack();
        assert_eq!(stack.len(), 1, "{label}: resume stack preserved");
        let frame = &stack[0];
        assert_eq!(frame.proc_name, "ilq1_proc", "{label}: proc_name preserved");
        assert_eq!(frame.resume_tag, 7, "{label}: resume_tag preserved");
        assert_eq!(
            frame.caller_return_addr, 0x400600,
            "{label}: caller_return_addr preserved",
        );
        assert_eq!(frame.saved_args.len(), 2, "{label}: saved_args count");
        assert_eq!(
            twin.solver().borrow().eval(&frame.saved_args[0]),
            Some(0xCAFE_F00D_DEAD_BEEF),
            "{label}: symbolic saved_arg must re-prove its witness",
        );
        assert_eq!(
            frame.saved_args[1].as_u64(),
            Some(0x602000),
            "{label}: concrete saved_arg clones verbatim",
        );
    };

    // A -> B: serialize under A, rebuild under a fresh B.
    let payload_b = state.detach_for_migration();
    let ctx_b = Context::new(&Config::new());
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        ctx_b.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );
    Context::set_thread_local(&ctx_b);
    let twin_b = payload_b.reattach(&ctx_b).expect("reattach A->B");
    verify(&twin_b, "A->B");

    // B -> A': serialize the twin under B, rebuild under another fresh context.
    let payload_a2 = twin_b.detach_for_migration();
    let ctx_a2 = Context::new(&Config::new());
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = payload_a2.reattach(&ctx_a2).expect("reattach B->A'");
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}

// angr-1ilq.1: the fidelity gap fix. `to_serialized` drops the Bucket-D
// `Py<PyAny>` overlays (symbolic_pages / hook_symbolic_memory / addr_to_ast),
// so a snapshot-only migration would silently lose Python-side symbolic state
// on callback-heavy workloads. Migration carries them as live `Send` handles.
// This proves they survive a cross-context round-trip alongside the Rust-side
// register/constraint state.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_preserves_py_overlays() {
    use pyo3::prelude::*;
    use z3::{Config, Context};

    Python::initialize();

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_overlay_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0x1234_5678, 64), &s)
    };
    state.add_constraint(constraint);

    // Populate one entry in each Bucket-D overlay map.
    Python::attach(|py| {
        let mut pages: std::collections::HashMap<u64, Py<PyAny>> = std::collections::HashMap::new();
        pages.insert(0x1000, py.None());
        state.replace_symbolic_pages(pages);
        state.set_hook_symbolic_memory(0x2000, py.None(), 8);
        state.set_addr_to_ast(0x3000, py.None(), 4);
    });

    let original = Context::thread_local();
    let target = Context::new(&Config::new());

    let payload = state.detach_for_migration();
    Context::set_thread_local(&target);
    let migrated = payload.reattach(&target).expect("reattach");
    // Rust-side state survived too.
    let rax_t = migrated.get_register("rax").expect("rax present");
    let got = migrated.solver().borrow().eval(&rax_t);
    Context::set_thread_local(&original);

    assert_eq!(got, Some(0x1234_5678), "migrated rax re-proves witness");
    assert_eq!(
        migrated.symbolic_pages().len(),
        1,
        "symbolic_pages carried across migration (not dropped like a bare snapshot)",
    );
    assert!(
        migrated.symbolic_pages().contains_key(&0x1000),
        "symbolic_pages key preserved",
    );
    assert_eq!(
        migrated.hook_symbolic_memory().len(),
        1,
        "hook_symbolic_memory carried across migration",
    );
    assert!(
        migrated.hook_symbolic_memory().contains_key(&0x2000),
        "hook_symbolic_memory key preserved",
    );
    assert_eq!(
        migrated.addr_to_ast().len(),
        1,
        "addr_to_ast carried across migration",
    );
    assert!(
        migrated.addr_to_ast().contains_key(&0x3000),
        "addr_to_ast key preserved",
    );
}

// angr-1ilq.1: reattach must FAIL FAST (not silently rebuild in the wrong
// context) when `target_ctx` is not the active thread-local. `from_serialized`
// mints ASTs in the active thread-local, so a stale thread-local would bind the
// rebuilt state to the wrong context — Z3 UB on the next query. The guard is
// unconditional (returns ContextMismatch), so this holds in release too.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_reattach_wrong_context_errors() {
    use z3::{Config, Context};

    let state = RustSimState::new("amd64").unwrap();
    let payload = state.detach_for_migration();

    // `target` is a fresh context that is NOT the active thread-local (we never
    // call set_thread_local), so reattach must reject it rather than rebuild.
    let target = Context::new(&Config::new());
    match payload.reattach(&target) {
        Err(SnapshotError::ContextMismatch) => {}
        other => panic!(
            "reattach with a non-active target_ctx must return ContextMismatch, got {:?}",
            other.map(|s| s.state_id()),
        ),
    }
}

// angr-1ilq.1: the end-to-end point of the bead — the payload is genuinely
// moved to ANOTHER OS thread and rebuilt there. Proves the `Send` transport
// works across a real thread boundary (not just a thread-local swap on one
// thread): the worker thread installs its own Z3 context, reattaches, and the
// migrated state re-proves its pinned witness in that thread's context.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_migrate_across_real_thread() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq1_thread_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0x5151_5151, 64), &s)
    };
    state.add_constraint(constraint);

    // Serialize on this (owner) thread, then MOVE the payload to a worker.
    let payload = state.detach_for_migration();
    let (got, sat) = std::thread::spawn(move || {
        // The worker installs its OWN Z3 context as the thread-local.
        let worker_ctx = Context::new(&Config::new());
        Context::set_thread_local(&worker_ctx);
        let migrated = payload.reattach(&worker_ctx).expect("reattach on worker");
        let rax_t = migrated.get_register("rax").expect("rax present");
        let got = migrated.solver().borrow().eval(&rax_t);
        let sat = migrated.solver().borrow().is_sat();
        (got, sat)
    })
    .join()
    .expect("worker thread panicked");

    assert_eq!(
        got,
        Some(0x5151_5151),
        "migrated rax must re-prove its witness in the worker thread's context",
    );
    assert!(
        sat,
        "migrated state's solver must remain SAT on the worker thread"
    );
}
