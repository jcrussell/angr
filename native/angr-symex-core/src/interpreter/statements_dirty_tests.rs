//! Tests for `IRStmt::Dirty` execution, extracted from `statements_tests.rs`
//! following the prod-side split (angr-fs8kb.96, angr-fs8kb.62).
//!
//! Covers the two no-handler-anywhere policies
//! (`VEXInterpreter::dirty_python_dispatch`'s route-to-Python default and the
//! `InvalidIR` result-temp validation) and the false-guard poison-write
//! contract of `VEXInterpreter::dirty_guard_phase`, plus the memory-effect
//! routing and symbolic-result conversion of `handle_dirty_call` /
//! `dirty_python_dispatch` (angr-vmi2l.170) and the symbolic-guard `ITE`
//! (angr-vmi2l.171). Shared helpers live in
//! `statements_tests_support.rs`.

use super::*;
use crate::interpreter::TestHost;
use crate::interpreter::statements_tests_support::{make_irsb_with_temps, new_interp, with_test_host};
use angr_vex::ir::IRType;

/// angr-c7xno.44: a dirty helper that neither Rust nor Python models must
/// route the block to Python (`NeedPythonFallback`), not fabricate a fresh
/// unconstrained symbolic into the result tmp. Same policy — and same opt-in
/// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` escape hatch — as
/// `VEXInterpreter::vex_op_fallback` / `eval_ccall`. Default (gate unset)
/// behavior: fabricating silently diverges, because every downstream
/// condition over the unconstrained tmp explores both branches.
#[test]
fn unmodelled_dirty_call_routes_to_python_not_fabricate() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: Some(0),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_test_host(|cb| {
        // `with_test_host` builds a bare TestHost: no dirty_call callback,
        // which is the branch under test (production always registers one via
        // rust_manager.py::_setup_callbacks).
        assert!(!cb.has_dirty_call());
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        // `StmtResult` is not `Debug`, so describe the outcome by hand.
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "unmodelled dirty call must route to Python, got {}",
            match &res {
                Ok(_) => "Ok(..)".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "unmodelled dirty call fabricated instead of routing to Python"
        );
        assert!(
            interp.temps.first().is_none_or(Option::is_none),
            "no fabricated value may be written into the result tmp"
        );
    });
}

/// angr-03vl4.33: a dirty call naming a result temp that is absent from the
/// block's tyenv is malformed IR. `handle_dirty_call` used to default the
/// width to 64, which silently produced a wrong-width tmp on both the
/// native-handler and Python-callback write paths; it must now fail with
/// `InvalidIR` like the `IRStmt::LLSC` arm does for the same condition.
#[test]
fn dirty_call_result_tmp_missing_from_tyenv_is_invalid_ir() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // tyenv declares exactly one temp (t0); the dirty call names t5.
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: Some(5),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        match &res {
            Err(CbExecutionError::InvalidIR(msg)) => {
                assert!(
                    msg.contains("not in tyenv") && msg.contains('5'),
                    "InvalidIR message should name the missing temp, got {msg}"
                );
            }
            Ok(_) => panic!("missing tyenv entry silently accepted"),
            Err(e) => panic!("expected InvalidIR, got {e:?}"),
        }
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "the tyenv lookup must fail before any fabricate/fallback path"
        );
    });
}

// ---------------------------------------------------------------------------
// angr-fs8kb.57: a guarded dirty call must define its result temp even when
// the guard is false.
// ---------------------------------------------------------------------------

/// libvex_ir.h's `IRDirty` doc: "If at runtime the guard evaluates to false,
/// .tmp has an 0x555...555 bit pattern written to it. Hence conditional calls
/// that assign .tmp are allowed." `handle_dirty_call`'s `GuardClass::Never`
/// arm used to return `Continue` without touching the temp, so the very
/// downstream read VEX's guarantee licenses — modelled here by a following
/// `WrTmp` over `RdTmp(t0)` — failed with `UnknownTemp`.
#[test]
fn dirty_call_with_false_guard_defines_result_tmp() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64, IRType::I64]);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: Some(Box::new(IRExpr::Const(IRConst::U1(false)))),
        tmp: Some(0),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        assert!(
            matches!(res, Ok(StmtResult::Continue)),
            "a false guard must skip the call and continue, got {}",
            match &res {
                Ok(_) => "another StmtResult".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
        let written = interp.temps[0]
            .as_ref()
            .expect("false guard must still define the result tmp");
        assert_eq!(
            written.as_u64(),
            Some(0x5555_5555_5555_5555),
            "result tmp must carry VEX's guard-false poison pattern"
        );
        assert_eq!(written.width(), 64, "poison must take the tyenv width");
        // The whole point of the guarantee: a downstream unconditional read.
        let read_back = interp.execute_stmt_with_callbacks(
            cb,
            &IRStmt::WrTmp {
                tmp: 1,
                data: IRExpr::RdTmp(0),
            },
            &irsb,
        );
        assert!(
            matches!(read_back, Ok(StmtResult::Continue)),
            "reading the skipped dirty call's tmp must not fail, got {}",
            match &read_back {
                Ok(_) => "another StmtResult".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
    });
}

/// The poison is a `RustBV::Concrete`, whose `u128` payload cannot hold the
/// `0x555…555` pattern above 128 bits — masking it would write a silently
/// wrong *defined* value. Only `Ity_V256` can reach that, and it must route
/// the block to Python rather than guess.
#[test]
fn dirty_call_with_false_guard_declines_a_v256_result_tmp() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::V256]);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: Some(Box::new(IRExpr::Const(IRConst::U1(false)))),
        tmp: Some(0),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        match &res {
            Err(CbExecutionError::NeedPythonFallback(msg)) => assert!(
                msg.contains("256"),
                "the fallback reason should name the width, got {msg}"
            ),
            Ok(_) => panic!("a 256-bit poison was silently truncated"),
            Err(e) => panic!("expected NeedPythonFallback, got {e:?}"),
        }
        assert!(
            interp.temps[0].is_none(),
            "no truncated poison may reach the temp"
        );
    });
}

/// A guarded dirty call with no result temp has nothing to define; the false
/// guard just skips it. Guards the `if let Some(tmp)` in the `Never` arm
/// against a regression that unconditionally looks up a tyenv entry.
#[test]
fn dirty_call_with_false_guard_and_no_result_tmp_just_continues() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: Some(Box::new(IRExpr::Const(IRConst::U1(false)))),
        tmp: None,
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        assert!(
            matches!(res, Ok(StmtResult::Continue)),
            "a tmp-less guarded dirty call must continue, got {}",
            match &res {
                Ok(_) => "another StmtResult".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
    });
}

/// An x87 `fstp m80` dirty call (`storeF80le`, `mFx = Write`) on a dirty
/// helper `IRDirty` the native table does not cover.
fn store_f80_dirty(m_fx: angr_vex::ir::DirtyFx) -> angr_vex::ir::IRDirty {
    use angr_vex::ir::{IRCallee, IRDirty};
    IRDirty {
        cee: IRCallee {
            name: "x86g_dirtyhelper_storeF80le".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: None,
        mFx: m_fx,
        mAddr: Some(Box::new(IRExpr::Const(IRConst::U32(0x70_0000)))),
        mSize: 10,
        nFxState: 0,
        args: vec![
            IRExpr::Const(IRConst::U32(0x70_0000)),
            IRExpr::Const(IRConst::U64(0x4008_0000_0000_0000)),
        ],
    }
}

/// angr-vmi2l.170: the Python `dirty_call` callback runs a helper against the
/// host's SimState, not `rust_memory`, so a memory-effecting helper with no
/// native handler must route the block to Python and never reach the
/// callback, whichever effect it declares.
#[test]
fn memory_effect_dirty_call_routes_to_python_not_the_callback() {
    use angr_vex::ir::DirtyFx;
    for m_fx in [DirtyFx::Read, DirtyFx::Write, DirtyFx::Modify] {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x3000, &[]);
        let host = TestHost {
            dirty_result: Some(RustBV::concrete(0, 64)),
            ..TestHost::default()
        };
        let res =
            interp.execute_stmt_with_callbacks(&host, &IRStmt::Dirty(store_f80_dirty(m_fx)), &irsb);
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "{m_fx:?} dirty call must route to Python, got {}",
            match &res {
                Ok(_) => "Ok(..)".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
        assert!(
            host.dirty_calls.borrow().is_empty(),
            "{m_fx:?} dirty call reached the callback"
        );
    }
}

/// The control: the same helper with no memory effect still reaches the
/// callback, so the routing above is keyed on `mFx`, not on the name.
#[test]
fn effect_free_dirty_call_still_reaches_the_callback() {
    use angr_vex::ir::DirtyFx;
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[]);
    let host = TestHost {
        dirty_result: Some(RustBV::concrete(0, 64)),
        ..TestHost::default()
    };
    let res = interp.execute_stmt_with_callbacks(
        &host,
        &IRStmt::Dirty(store_f80_dirty(DirtyFx::None)),
        &irsb,
    );
    assert!(res.is_ok(), "effect-free dirty call failed");
    assert_eq!(
        *host.dirty_calls.borrow(),
        ["x86g_dirtyhelper_storeF80le".to_string()]
    );
}

/// A guarded dirty call over a symbolic tmp guard: `t0` is the I1 guard, `t1`
/// the I64 result temp.
fn symbolic_guard_dirty(m_fx: angr_vex::ir::DirtyFx, n_fx_state: u32) -> angr_vex::ir::IRDirty {
    angr_vex::ir::IRDirty {
        cee: angr_vex::ir::IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: Some(Box::new(IRExpr::RdTmp(0))),
        tmp: Some(1),
        mFx: m_fx,
        mAddr: None,
        mSize: 0,
        nFxState: n_fx_state,
        args: vec![],
    }
}

/// angr-vmi2l.171: a symbolic guard used to be pinned true, so every
/// guard-false state was dropped. VEX's semantics are
/// `tmp = ITE(guard, ret, 0x55..)`: the guard stays free and each side of it
/// reads its own value.
#[test]
fn dirty_call_with_symbolic_guard_writes_an_ite_and_keeps_both_sides() {
    let ctx = SymContext::new();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I1, IRType::I64]);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let guard = RustBV::symbol("dirty_guard", 1);
    interp.temps[0] = Some(guard.clone());
    let host = TestHost {
        dirty_result: Some(RustBV::concrete(7, 64)),
        ..TestHost::default()
    };
    let dirty = symbolic_guard_dirty(angr_vex::ir::DirtyFx::None, 0);
    let res = interp.execute_stmt_with_callbacks(&host, &IRStmt::Dirty(dirty), &irsb);
    assert!(res.is_ok(), "symbolic-guard dirty call failed");
    assert!(
        ctx.can_be_false(&guard),
        "the guard-false states must survive the call"
    );
    let written = interp.temps[1].clone().expect("result tmp must be defined");
    let one = RustBV::concrete(1, 1);
    let zero = RustBV::concrete(0, 1);
    for (side, expected) in [(&one, 7u64), (&zero, 0x5555_5555_5555_5555)] {
        assert!(
            !ctx.can_be_true(&guard.eq(side).and(&written.eq(&RustBV::concrete(expected as u128, 64)).not())),
            "guard == {side:?} must read {expected:#x}"
        );
    }
}

/// angr-vmi2l.171: the `ITE` conditions only the result temp, so a helper
/// with a memory or guest-state effect under a symbolic guard routes the
/// block to Python before it runs.
#[test]
fn dirty_call_with_symbolic_guard_and_effects_routes_to_python() {
    use angr_vex::ir::DirtyFx;
    for (m_fx, n_fx_state) in [(DirtyFx::Read, 0), (DirtyFx::None, 1)] {
        let ctx = SymContext::new();
        let mut interp = new_interp(&ctx);
        let irsb = make_irsb_with_temps(0x3000, &[IRType::I1, IRType::I64]);
        interp.temps.resize(irsb.tyenv.types.len(), None);
        interp.temps[0] = Some(RustBV::symbol("dirty_guard_fx", 1));
        let host = TestHost {
            dirty_result: Some(RustBV::concrete(7, 64)),
            ..TestHost::default()
        };
        let dirty = symbolic_guard_dirty(m_fx, n_fx_state);
        let res = interp.execute_stmt_with_callbacks(&host, &IRStmt::Dirty(dirty), &irsb);
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "{m_fx:?}/{n_fx_state}: must route to Python"
        );
        assert!(host.dirty_calls.borrow().is_empty(), "the callback must not run");
    }
}

/// angr-vmi2l.170: a symbolic callback result is the host's own value, not a
/// fresh `dirty_<name>` symbol unrelated to it; a width mismatch routes to
/// Python rather than writing a wrong-width temp.
#[test]
fn symbolic_dirty_result_is_the_hosts_value() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let dirty = IRDirty {
        cee: IRCallee {
            name: "x86g_dirtyhelper_RDMSR".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: Some(0),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![IRExpr::Const(IRConst::U32(0x10))],
    };
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64]);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let host_value = RustBV::symbol("rdmsr_0x10", 64);
    let host = TestHost {
        dirty_result: Some(host_value.clone()),
        ..TestHost::default()
    };
    let res = interp.execute_stmt_with_callbacks(&host, &IRStmt::Dirty(dirty.clone()), &irsb);
    assert!(res.is_ok(), "symbolic dirty call failed: {:?}", res.err());
    assert_eq!(
        interp.temps.first().cloned().flatten(),
        Some(host_value),
        "the result temp must hold the host's AST"
    );

    let mut interp = new_interp(&ctx);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let host = TestHost {
        dirty_result: Some(RustBV::symbol("rdmsr_0x10_narrow", 32)),
        ..TestHost::default()
    };
    let res = interp.execute_stmt_with_callbacks(&host, &IRStmt::Dirty(dirty), &irsb);
    assert!(
        matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
        "a wrong-width symbolic result must route to Python"
    );
}

/// angr-vmi2l.78: a `write_cr0` can clear CR0.PE in Python's globals, so it
/// closes the native `x86g_use_seg_selector` gate on every interpreter that
/// shares the flag, the manager's included.
#[test]
fn write_cr0_closes_the_seg_selector_gate() {
    use angr_vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let shared = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    interp.set_x86_protected_mode(std::sync::Arc::clone(&shared));
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I32]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "x86g_dirtyhelper_write_cr0".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: None,
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![IRExpr::Const(IRConst::U32(0))],
    };
    with_test_host(|cb| {
        // No dirty_call callback, so the block routes to Python; the gate
        // must close either way.
        drop(interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb));
    });
    assert!(!interp.x86_protected_mode());
    assert!(!shared.load(std::sync::atomic::Ordering::Relaxed));
}
