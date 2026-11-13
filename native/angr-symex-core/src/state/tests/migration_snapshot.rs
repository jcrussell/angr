//! Cross-context migration via snapshot, each rebuilt on a fresh worker
//! thread: Multi-cell bytes, resume stack, Python overlays, and the
//! wrong-context reattach error.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

/// Run `f` on a fresh OS thread and return its result. Each thread owns its
/// own thread-local Z3 context, so this is the production worker model: the
/// only thing that crosses is `Send` data (a migration payload, plain values).
fn on_worker<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    std::thread::scope(|s| {
        s.spawn(f)
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

/// The calling thread's thread-local Z3 context, as a comparable address.
fn thread_ctx_addr() -> usize {
    z3::Context::thread_local().get_z3_context().as_ptr() as usize
}

// angr-1ilq.1: migration serializes the state to context-free bytes on the
// owner and rebuilds the ASTs in the stealing worker's own context, never
// reading the source context cross-thread. A symbolic register pinned by a
// path constraint must re-evaluate to its witness after the round trip.
#[test]
fn test_migrate_via_snapshot_cross_context() {
    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        RustBV::symbol("ilq1_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        x.eq(&RustBV::concrete(0xdead_beef, 64))
    };
    state.add_constraint(constraint);
    let sid = state.state_id();
    let source_ctx = thread_ctx_addr();

    // Owner serializes under the source context (current thread-local)...
    let payload = state.detach_for_migration();
    // ...the stealer rebuilds in its own context — every AST is minted there,
    // with no read of the source context.
    let (target_ctx, got, sat, migrated_id) = on_worker(move || {
        let target = z3::Context::thread_local();
        let migrated = payload.reattach(&target).expect("reattach");
        let rax_t = migrated.get_register("rax").expect("rax present");
        let got = migrated.solver().eval(&rax_t);
        let sat = migrated.solver().is_sat();
        (thread_ctx_addr(), got, sat, migrated.state_id())
    });

    assert_ne!(source_ctx, target_ctx, "test bug: target == source context");
    assert_eq!(
        got,
        Some(0xdead_beef),
        "migrated rax must resolve via the transferred constraint",
    );
    assert!(sat, "migrated state's solver must remain SAT");
    assert_eq!(
        migrated_id, sid,
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
#[test]
fn test_migrate_carries_multi_cell_bytes() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);

    let (addr_var, val_bv) = {
        let ctx = state.solver();
        let addr_var = RustBV::symbol("sqfj8_71_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64))
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64))),
        );
        (addr_var, RustBV::concrete(0xCAFE_BABEu128, 32))
    };
    // The candidates sit a page apart: widen Range's limit past their span.
    let concretizer = crate::symbolic::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        write_range_limit: 0x10000,
        ..crate::symbolic::concretize::AddressConcretizer::new()
    };
    let ctx = state.solver().clone();
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

    let payload = state.detach_for_migration();
    let width = on_worker(move || {
        let migrated = payload
            .reattach(&z3::Context::thread_local())
            .expect("reattach");
        migrated
            .memory()
            .get_symbolic_object(0x1000)
            .map(|loaded| loaded.width())
    })
    .expect("byte 0x1000 must be a symbolic object after migration, not silently dropped");
    assert_eq!(
        width,
        32,
        "migrated Multi-cell byte must round-trip at its stored width"
    );
}

// angr-1ilq.1: an A->B->A' round trip. Proves the snapshot transport carries
// `native_resume_stack` (angr-1ilq.2) faithfully across two
// distinct worker contexts: a symbolic saved_arg re-proves its witness at
// every hop; a concrete saved_arg clones verbatim. Each hop serializes under
// the context that is then live, so a half-rebuilt foreign-context AST would
// surface here.
#[test]
fn test_migrate_resume_stack_roundtrip() {
    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        RustBV::symbol("ilq1_saved_arg", 64)
    };
    let constraint = {
        x.eq(&RustBV::concrete(0xCAFE_F00D_DEAD_BEEF, 64))
    };
    state.add_constraint(constraint);
    state.push_native_resume_frame(NativeResumeFrame {
        proc_name: "ilq1_proc".to_string(),
        resume_tag: 7,
        saved_args: vec![x.clone(), RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x400600,
    });

    fn verify(twin: &RustSimState, label: &str) {
        assert!(
            twin.solver().is_sat(),
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
            twin.solver().eval(&frame.saved_args[0]),
            Some(0xCAFE_F00D_DEAD_BEEF),
            "{label}: symbolic saved_arg must re-prove its witness",
        );
        assert_eq!(
            frame.saved_args[1].as_u64(),
            Some(0x602000),
            "{label}: concrete saved_arg clones verbatim",
        );
    }

    // A -> B: serialize under A, rebuild on worker B; B serializes its twin.
    let payload_b = state.detach_for_migration();
    let payload_a2 = on_worker(move || {
        let twin_b = payload_b
            .reattach(&z3::Context::thread_local())
            .expect("reattach A->B");
        verify(&twin_b, "A->B");
        twin_b.detach_for_migration()
    });

    // B -> A': rebuild B's payload on another fresh worker.
    on_worker(move || {
        let twin_a2 = payload_a2
            .reattach(&z3::Context::thread_local())
            .expect("reattach B->A'");
        verify(&twin_a2, "B->A'");
    });
}

// angr-1ilq.1: the fidelity gap fix. `to_serialized` drops the Bucket-D
// `Py<PyAny>` overlays (symbolic_pages / hook_symbolic_memory / addr_to_ast),
// so a snapshot-only migration would silently lose Python-side symbolic state
// on callback-heavy workloads. Migration carries them as live `Send` handles.
// This proves they survive a cross-context round-trip alongside the Rust-side
// register/constraint state.
#[test]
fn test_migrate_preserves_py_overlays() {
    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        RustBV::symbol("ilq1_overlay_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        x.eq(&RustBV::concrete(0x1234_5678, 64))
    };
    state.add_constraint(constraint);

    // Populate one entry in each Bucket-D overlay map. The overlays are opaque
    // to `state`; `()` stands in for the shell's AST.
    let mut pages = std::collections::HashMap::new();
    pages.insert(0x1000u64, ());
    state.replace_symbolic_pages(pages);
    state.set_hook_symbolic_memory(0x2000, (), 8);
    state.set_addr_to_ast(0x3000, (), 4);

    let payload = state.detach_for_migration();
    on_worker(move || {
        let migrated = payload
            .reattach(&z3::Context::thread_local())
            .expect("reattach");
        // Rust-side state survived too.
        let rax_t = migrated.get_register("rax").expect("rax present");
        let got = migrated.solver().eval(&rax_t);

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
    });
}

// angr-1ilq.1: reattach must FAIL FAST (not silently rebuild in the wrong
// context) when `target_ctx` is not the active thread-local. `from_serialized`
// mints ASTs in the active thread-local, so a stale thread-local would bind the
// rebuilt state to the wrong context — Z3 UB on the next query. The guard is
// unconditional (returns ContextMismatch), so this holds in release too.
#[test]
fn test_migrate_reattach_wrong_context_errors() {
    let state = RustSimState::new("amd64").unwrap();
    let payload = state.detach_for_migration();

    // `target` is a fresh context that is NOT the active thread-local (it is
    // never installed), so reattach must reject it rather than rebuild.
    // z3-rs exposes no safe constructor for a context it does not install, so
    // build one the way z3's own (crate-private) `Context::new` does.
    // SAFETY: `cfg` is a fresh config freed once the context exists; `raw` is a
    // fresh reference-counted context nothing else manages, so `from_raw` owns
    // it and deletes it on drop. Clearing the error handler matches z3-rs's
    // own contexts: errors surface as error codes instead of aborting.
    let target = unsafe {
        let cfg = z3_sys::Z3_mk_config().expect("Z3_mk_config");
        let raw = z3_sys::Z3_mk_context_rc(cfg).expect("Z3_mk_context_rc");
        z3_sys::Z3_del_config(cfg);
        z3_sys::Z3_set_error_handler(raw, None);
        z3::Context::from_raw(raw)
    };
    match payload.reattach(&target) {
        Err(SnapshotError::ContextMismatch) => {}
        other => panic!(
            "reattach with a non-active target_ctx must return ContextMismatch, got {:?}",
            other.map(|s| s.state_id()),
        ),
    }
}

// angr-az6ru.2: a state is `Send`, so it can be moved to another thread as it
// is. Its solver and its witness stay behind in the first thread's tables;
// the second thread builds its own from the constraint log, and when the
// state comes back the first thread's stale solver is brought up to date with
// what was added elsewhere. Every answer is the one a state that never moved
// gives.
#[test]
fn test_state_moved_between_threads_answers_from_its_log() {
    let mut state = RustSimState::new("amd64").unwrap();
    let x = RustBV::symbol("az6ru_moved_x", 8);
    let y = RustBV::symbol("az6ru_moved_y", 8);
    state.set_register("rax", x.zero_extend(56));
    state.add_constraint(x.ugt(&RustBV::concrete(250, 8)));
    // Builds this thread's solver and caches a witness here.
    assert!(state.satisfiable());
    assert_eq!(state.solver().eval_upto(&x, 16).len(), 5);
    assert!(state.solver().has_built_solver());

    let home_ctx = thread_ctx_addr();
    let (wx, wy) = (x.clone(), y.clone());
    let state = on_worker(move || {
        let (x, y) = (wx, wy);
        assert_ne!(thread_ctx_addr(), home_ctx);
        assert!(
            !state.solver().has_built_solver(),
            "the solver stayed on the thread that built it"
        );
        assert_eq!(state.solver().cached_models_for_test(), (false, false));
        assert!(state.satisfiable());
        assert_eq!(state.solver().eval_upto(&x, 16).len(), 5);
        // Narrow it here; the home thread's solver has not seen this.
        state.add_constraint(x.ult(&RustBV::concrete(253, 8)));
        state.add_constraint(y.eq(&x));
        assert_eq!(state.solver().eval_upto(&x, 16).len(), 2);
        state
    });

    assert!(state.solver().has_built_solver(), "the stale one, still here");
    let mut values = state.solver().eval_upto(&y, 16);
    values.sort_unstable();
    assert_eq!(values, vec![251, 252]);
    assert!(state.satisfiable());

    // A fork taken after the round trip carries the whole log.
    let child = state.fork();
    child.add_constraint(x.eq(&RustBV::concrete(252, 8)));
    assert_eq!(child.solver().eval_upto(&y, 16), vec![252]);
}
