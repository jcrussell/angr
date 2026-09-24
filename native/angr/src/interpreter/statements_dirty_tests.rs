//! Tests for `IRStmt::Dirty` execution (extracted from `statements_tests.rs`
//! by angr-fs8kb.96, following the prod-side split in angr-fs8kb.62).
//!
//! Covers the two no-handler-anywhere policies
//! (`VEXInterpreter::dirty_python_dispatch`'s route-to-Python default and the
//! `InvalidIR` result-temp validation) and the false-guard poison-write
//! contract of `VEXInterpreter::dirty_guard_phase`. Shared helpers live in
//! `statements_tests_support.rs`.

use super::*;
use crate::interpreter::statements_tests_support::{make_irsb_with_temps, new_interp, with_python};
use crate::vex::ir::IRType;

/// angr-c7xno.44: a dirty helper that neither Rust nor Python models must
/// route the block to Python (`NeedPythonFallback`), not fabricate a fresh
/// unconstrained symbolic into the result tmp. Same policy — and same opt-in
/// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` escape hatch — as
/// `VEXInterpreter::vex_op_fallback` / `eval_ccall`. Default (gate unset)
/// behavior: fabricating silently diverges, because every downstream
/// condition over the unconstrained tmp explores both branches.
#[test]
fn unmodelled_dirty_call_routes_to_python_not_fabricate() {
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
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
    with_python(|cb| {
        // `with_python` builds a bare PythonCallbacks: no dirty_call callback,
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
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
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
    with_python(|cb| {
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
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
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
    with_python(|cb| {
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
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
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
    with_python(|cb| {
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
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
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
    with_python(|cb| {
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
