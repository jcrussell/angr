//! In-module unit tests for `callbacks/events.rs` (angr-c4xcs.7).
//!
//! `LoopExecutionEvent::from_run_result_with_forks` is a large flat match that
//! maps each `RunResult` variant onto the wire event returned to Python. The
//! mapping is pure (no Z3, no GIL — `DeferredFork`s carry `condition_ast=None`)
//! and easy to drift silently, so these tests pin the `event_type` string and
//! the variant-specific fields for every arm.

use super::*;

#[test]
fn test_max_blocks_maps_pc_only() {
    let ev = LoopExecutionEvent::from_run_result(RunResult::MaxBlocks { pc: 0xdead }, 3);
    assert_eq!(ev.event_type, "max_blocks");
    assert_eq!(ev.pc, Some(0xdead));
    assert_eq!(ev.addr, None);
    assert_eq!(ev.blocks_executed, 3);
}

#[test]
fn test_simprocedure_populates_name_and_return_addr() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::SimProcedure {
            addr: 0x1000,
            name: "strlen".to_string(),
            num_args: 1,
            return_addr: 0x2000,
        },
        0,
    );
    assert_eq!(ev.event_type, "simprocedure");
    assert_eq!(ev.pc, Some(0x1000));
    assert_eq!(ev.jumpkind.as_deref(), Some("Ijk_Call"));
    assert_eq!(ev.simprocedure_name.as_deref(), Some("strlen"));
    assert_eq!(ev.simprocedure_num_args, Some(1));
    assert_eq!(ev.simprocedure_return_addr, Some(0x2000));
}

#[test]
fn test_simprocedure_zero_return_addr_becomes_none() {
    // return_addr == 0 is the "unknown" sentinel and must map to None.
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::SimProcedure {
            addr: 0x1000,
            name: "malloc".to_string(),
            num_args: 1,
            return_addr: 0,
        },
        0,
    );
    assert_eq!(ev.simprocedure_return_addr, None);
}

#[test]
fn test_syscall_symbolic_num_stays_none() {
    // A symbolic syscall number (None) must survive onto the wire event so the
    // dispatch loop forces a Python callback (angr-gffd).
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::Syscall {
            num: None,
            pc: 0x40,
        },
        1,
    );
    assert_eq!(ev.event_type, "syscall");
    assert_eq!(ev.pc, Some(0x40));
    assert_eq!(ev.syscall_num, None);
    assert_eq!(ev.jumpkind.as_deref(), Some("Ijk_Sys_syscall"));
}

#[test]
fn test_symbolic_branch_targets_but_no_pc() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::SymbolicBranch {
            condition_id: 9,
            true_target: 0xaa,
            false_target: 0xbb,
        },
        2,
    );
    assert_eq!(ev.event_type, "symbolic_branch");
    assert_eq!(ev.pc, None);
    assert_eq!(ev.true_target, Some(0xaa));
    assert_eq!(ev.false_target, Some(0xbb));
}

#[test]
fn test_error_carries_message_regardless_of_kind() {
    // event_type is "error" for both RunErrorKind variants; the kind field is
    // intentionally dropped from the wire event (routing happens earlier).
    for kind in [RunErrorKind::Deadend, RunErrorKind::Fatal] {
        let ev = LoopExecutionEvent::from_run_result(
            RunResult::Error {
                message: "boom".to_string(),
                addr: 0x1234,
                kind,
            },
            0,
        );
        assert_eq!(ev.event_type, "error");
        assert_eq!(ev.addr, Some(0x1234));
        assert_eq!(ev.error.as_deref(), Some("boom"));
    }
}

#[test]
fn test_symbolic_jump_target_pc_is_first_target() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::SymbolicJumpTarget {
            targets: vec![0x10, 0x20, 0x30],
            condition_id: 5,
            jumpkind: "Ijk_Ret".to_string(),
        },
        0,
    );
    assert_eq!(ev.event_type, "symbolic_jump_target");
    // pc is the first concrete target.
    assert_eq!(ev.pc, Some(0x10));
    assert_eq!(ev.jump_targets, Some(vec![0x10, 0x20, 0x30]));
    assert_eq!(ev.jump_condition_id, Some(5));
    assert_eq!(ev.jumpkind.as_deref(), Some("Ijk_Ret"));
}

#[test]
fn test_unconstrained_jump_range_and_limit() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::UnconstrainedJump {
            min_target: 0x1000,
            max_target: 0x9000,
            limit: 257,
            jumpkind: "Ijk_Boring".to_string(),
        },
        0,
    );
    assert_eq!(ev.event_type, "unconstrained_jump");
    assert_eq!(ev.unconstrained_min, Some(0x1000));
    assert_eq!(ev.unconstrained_max, Some(0x9000));
    assert_eq!(ev.unconstrained_limit, Some(257));
}

#[test]
fn test_unmodeled_call_populates_addr_return_and_symbol() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::UnmodeledCall {
            addr: 0x555,
            return_addr: 0x666,
            symbol_name: Some("foo".to_string()),
        },
        0,
    );
    assert_eq!(ev.event_type, "unmodeled_call");
    assert_eq!(ev.unmodeled_call_addr, Some(0x555));
    assert_eq!(ev.unmodeled_call_return_addr, Some(0x666));
    assert_eq!(ev.unmodeled_call_symbol.as_deref(), Some("foo"));
    assert_eq!(ev.jumpkind.as_deref(), Some("Ijk_Call"));
}

#[test]
fn test_need_python_vex_uses_error_field_for_reason() {
    let ev = LoopExecutionEvent::from_run_result(
        RunResult::NeedPythonVEX {
            addr: 0x777,
            reason: "unsupported IROp".to_string(),
        },
        0,
    );
    assert_eq!(ev.event_type, "python_vex_fallback");
    assert_eq!(ev.addr, Some(0x777));
    // The reason is smuggled through the `error` field.
    assert_eq!(ev.error.as_deref(), Some("unsupported IROp"));
}

#[test]
fn test_from_run_result_with_forks_threads_forks_and_pushlevel() {
    let forks = vec![DeferredFork::new(0x1, true, 0x2, 0, 4, None)];
    let ev = LoopExecutionEvent::from_run_result_with_forks(
        RunResult::MaxDeferredForks { pc: 0x8000 },
        11,
        forks,
        4,
    );
    assert_eq!(ev.event_type, "max_deferred_forks");
    assert_eq!(ev.pc, Some(0x8000));
    assert_eq!(ev.blocks_executed, 11);
    assert_eq!(ev.push_level, 4);
    assert_eq!(ev.deferred_forks.len(), 1);
    assert_eq!(ev.deferred_forks[0].push_level, 4);
}
