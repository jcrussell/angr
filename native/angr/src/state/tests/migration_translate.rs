//! `translate_state` across two `SymContext`s: constraint/AST translation,
//! lineage-flag propagation, native resume-stack translation, and a
//! production-sized round-trip.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-ahypj: RustSimState::translate_state cross-context whole-state twin.
// Validates that translate_into composes correctly over a real state — a
// symbolic register pinned by a path constraint must re-evaluate to the same
// concrete witness once both the register overlay and the constraint have been
// Z3_translate'd into a fresh target context. Exercises shared-AST identity
// (the `rax` leaf in the register and inside the constraint must hash-cons to
// the same node in the target context) and identity preservation.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ahypj_rax", 64)
    };
    state.set_register("rax", x.clone());
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xdead_beef, 64), &s)
    };
    state.add_constraint(constraint);

    // angr-0xyq2 Fix 6: the filesystem carries context-bound RustBVs too
    // (fd content_sym + the file_contents registry) — pin a symbolic file
    // byte via a path constraint so a missed translate surfaces as a
    // foreign-context AST when the target-context solver evals it.
    let fbyte = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ahypj_file_byte", 8)
    };
    let fconstraint = {
        let s = state.solver().borrow();
        fbyte.eq(&RustBV::concrete(0x41, 8), &s)
    };
    state.add_constraint(fconstraint);
    state
        .file_system()
        .register_file_content("/tmp/ahypj", vec![fbyte, RustBV::concrete(0x42, 8)]);
    let fs_fd = state
        .file_system()
        .open("/tmp/ahypj".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");

    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    // translate_state asserts constraints into the new context's solver, which
    // builds against the thread-local — swap first (the target-worker model).
    Context::set_thread_local(&target);
    let translated = state.translate_state(&target);
    let rax_t = translated.get_register("rax").expect("rax present");
    let got = translated.solver().borrow().eval(&rax_t);
    let sat = translated.solver().borrow().is_sat();
    // Fix 6: the translated state's fs content BVs must live in the target
    // context — eval them through the target-context solver (a plain-cloned
    // foreign-context AST would misbehave here), mirroring the rax check.
    let content_t = translated
        .file_system_ref()
        .fd_content_sym(fs_fd)
        .expect("content_sym survives translate_state");
    let fd_byte = translated.solver().borrow().eval(&content_t[0]);
    let fd_concrete = content_t[1].as_u64();
    let registry_t = translated
        .file_system_ref()
        .file_content_for_path("/tmp/ahypj")
        .expect("file_contents registry survives translate_state");
    let reg_byte = translated.solver().borrow().eval(&registry_t[0]);
    Context::set_thread_local(&original);

    assert_eq!(
        got,
        Some(0xdead_beef),
        "translated rax must resolve via the transferred constraint",
    );
    assert!(sat, "translated state's solver must remain SAT");
    assert_eq!(
        fd_byte,
        Some(0x41),
        "translated fd content_sym byte must re-prove its witness",
    );
    assert_eq!(
        fd_concrete,
        Some(0x42),
        "concrete content_sym byte clones verbatim",
    );
    assert_eq!(
        reg_byte,
        Some(0x41),
        "translated file_contents registry byte must re-prove its witness",
    );
    assert_eq!(
        translated.state_id(),
        state.state_id(),
        "translate_state preserves identity (same state_id)",
    );
}

// angr-5khjd: SymContext::translate_into must propagate the source
// context's timeout_ms/deterministic/use_shared_lineage_solver atomic flags
// onto the target context, mirroring fork()'s three-line propagation
// (SymContext::new() defaults all three, so a naive translate silently
// reverts a deterministic/non-default-timeout/shared-lineage-solver source
// to defaults). Sets all three to non-default values on the source state
// before calling translate_state and asserts the translated twin carries
// the same values.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_propagates_lineage_flags() {
    use z3::{Config, Context};

    let state = RustSimState::new("amd64").unwrap();
    {
        let s = state.solver().borrow();
        s.set_timeout(12345);
        s.set_deterministic(true);
        s.set_use_shared_lineage_solver(true);
    }

    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    Context::set_thread_local(&target);
    let translated = state.translate_state(&target);
    let (timeout_ms, deterministic, shared_lineage_solver) = {
        let s = translated.solver().borrow();
        (
            s.timeout_ms(),
            s.is_deterministic(),
            s.use_shared_lineage_solver(),
        )
    };
    Context::set_thread_local(&original);

    assert_eq!(
        timeout_ms, 12345,
        "translate_state must propagate timeout_ms from the source context",
    );
    assert!(
        deterministic,
        "translate_state must propagate the deterministic flag from the source context",
    );
    assert!(
        shared_lineage_solver,
        "translate_state must propagate use_shared_lineage_solver from the source context",
    );
}

// angr-1ilq.2: translate_state must Z3_translate every RustBV in
// native_resume_stack.saved_args, not plain-clone the stack. A symbolic
// saved_arg cloned into a foreign worker's context is a dangling cross-context
// AST → UB once that worker touches it. This A->B->A' round-trip pushes a frame
// whose saved_args mix a symbolic arg (pinned by a path constraint) and a
// concrete arg, re-homes the whole state through a fresh context and back, and
// asserts the symbolic arg re-proves its witness via the transferred constraint
// at every hop (the concrete arg clones verbatim). Without the translate, the
// second hop reads a foreign-context AST.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_resume_stack_cross_context() {
    use z3::{Config, Context};

    let mut state = RustSimState::new("amd64").unwrap();
    // Symbolic saved_arg pinned to a known witness via a path constraint.
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "ilq2_saved_arg", 64)
    };
    let constraint = {
        let s = state.solver().borrow();
        x.eq(&RustBV::concrete(0xCAFE_F00D_DEAD_BEEF, 64), &s)
    };
    state.add_constraint(constraint);
    // Frame mixes a symbolic arg (must translate) and a concrete arg (clones
    // verbatim) — covers both arms of RustBV::translate_into.
    state.push_native_resume_frame(NativeResumeFrame {
        proc_name: "ilq2_proc".to_string(),
        resume_tag: 7,
        saved_args: vec![x.clone(), RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x400600,
    });

    let original = Context::thread_local();

    // Re-prove the frame survived translation into `twin`'s context.
    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: translated solver must remain SAT",
        );
        let stack = twin.native_resume_stack();
        assert_eq!(stack.len(), 1, "{label}: resume stack preserved");
        let frame = &stack[0];
        assert_eq!(frame.proc_name, "ilq2_proc", "{label}: proc_name preserved");
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

    // A -> B: translate the whole state into a fresh target context.
    let cfg_b = Config::new();
    let ctx_b = Context::new(&cfg_b);
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        ctx_b.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );
    Context::set_thread_local(&ctx_b);
    let twin_b = state.translate_state(&ctx_b);
    verify(&twin_b, "A->B");

    // B -> A': translate the twin BACK into another fresh context — the hop
    // that catches a half-translated (foreign-context) resume-stack AST.
    let cfg_a2 = Config::new();
    let ctx_a2 = Context::new(&cfg_a2);
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = twin_b.translate_state(&ctx_a2);
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}

// angr-9pwjd: production-sized translate_state round-trip validation.
//
// The synthetic kill-gate above proves the mechanism on a one-leaf state.
// This test scales it to a state shaped like a real mid-run state — many
// distinct symbolic leaves spread across both the register file and a
// symbolic memory region, each pinned by its own path constraint — and
// drives a full A->B->A round-trip. The literal Z3-bound trio
// (fairlight/sokohashv2/angry-reverser) cannot be captured into a pure-Rust
// test (no in-Rust real-binary loader; the engine is driven from Python),
// so we reconstruct an equivalently-shaped multi-leaf constrained state and
// assert the falsifiable claim the bead cares about: EVERY path constraint
// re-checks equal after translation, through a fresh context and back again.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_state_production_sized_roundtrip() {
    use z3::{Config, Context};

    const N_MEM_LEAVES: u64 = 64;
    const MEM_BASE: u64 = 0x10000;

    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(MEM_BASE, N_MEM_LEAVES * 8, Permission::RWX);

    // Pin a couple of registers to distinct symbolic leaves.
    let (rax, rbx) = {
        let s = state.solver().borrow();
        (
            RustBV::symbolic(&s, "pwjd_rax", 64),
            RustBV::symbolic(&s, "pwjd_rbx", 64),
        )
    };
    state.set_register("rax", rax.clone());
    state.set_register("rbx", rbx.clone());
    {
        let s = state.solver().borrow();
        let c_rax = rax.eq(&RustBV::concrete(0x1111_2222_3333_4444, 64), &s);
        let c_rbx = rbx.eq(&RustBV::concrete(0x5555_6666_7777_8888, 64), &s);
        drop(s);
        state.add_constraint(c_rax);
        state.add_constraint(c_rbx);
    }

    // Spread N distinct symbolic leaves across a symbolic memory region, each
    // pinned to a distinct witness. Mirrors a deep stdin/heap symbolic region.
    for i in 0..N_MEM_LEAVES {
        let leaf = {
            let s = state.solver().borrow();
            RustBV::symbolic(&s, format!("pwjd_mem_{i}"), 64)
        };
        state
            .memory_store(MEM_BASE + i * 8, leaf.clone())
            .expect("store leaf");
        let witness = 0xC0DE_0000_0000_0000u128 + i as u128;
        let s = state.solver().borrow();
        let c = leaf.eq(&RustBV::concrete(witness, 64), &s);
        drop(s);
        state.add_constraint(c);
    }

    // Capture the witnesses every constraint must re-prove after translation.
    let mut expected: Vec<(u64, u128)> = Vec::with_capacity(N_MEM_LEAVES as usize);
    for i in 0..N_MEM_LEAVES {
        let cell = state.memory_load(MEM_BASE + i * 8, 8).expect("load cell");
        let val = state.solver().borrow().eval(&cell).expect("cell evaluable");
        expected.push((MEM_BASE + i * 8, val));
    }
    let exp_rax = state
        .solver()
        .borrow()
        .eval(&state.get_register("rax").unwrap())
        .unwrap();
    let exp_rbx = state
        .solver()
        .borrow()
        .eval(&state.get_register("rbx").unwrap())
        .unwrap();

    // Helper: assert a translated twin re-proves every captured witness.
    let verify = |twin: &RustSimState, label: &str| {
        assert!(
            twin.solver().borrow().is_sat(),
            "{label}: translated solver must remain SAT",
        );
        for (addr, want) in &expected {
            let cell = twin.memory_load(*addr, 8).expect("load translated cell");
            let got = twin.solver().borrow().eval(&cell);
            assert_eq!(
                got,
                Some(*want),
                "{label}: mem leaf @{addr:#x} must re-prove its witness",
            );
        }
        assert_eq!(
            twin.solver()
                .borrow()
                .eval(&twin.get_register("rax").unwrap()),
            Some(exp_rax),
            "{label}: rax must re-prove its witness",
        );
        assert_eq!(
            twin.solver()
                .borrow()
                .eval(&twin.get_register("rbx").unwrap()),
            Some(exp_rbx),
            "{label}: rbx must re-prove its witness",
        );
        assert_eq!(
            twin.state_id(),
            state.state_id(),
            "{label}: translate_state preserves identity",
        );
    };

    let original = Context::thread_local();

    // A -> B: translate the whole state into a fresh target context.
    let cfg_b = Config::new();
    let ctx_b = Context::new(&cfg_b);
    Context::set_thread_local(&ctx_b);
    let twin_b = state.translate_state(&ctx_b);
    verify(&twin_b, "A->B");

    // B -> A': translate the twin BACK into another fresh context. This is
    // the round-trip that caught the iter15 add_constraint log bug — if the
    // translated solver did not seed its z3_assertions log, the second hop
    // would see zero constraints and the witnesses would not re-prove.
    let cfg_a2 = Config::new();
    let ctx_a2 = Context::new(&cfg_a2);
    Context::set_thread_local(&ctx_a2);
    let twin_a2 = twin_b.translate_state(&ctx_a2);
    verify(&twin_a2, "B->A'");

    Context::set_thread_local(&original);
}
