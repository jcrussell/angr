//! Error routing: which stash a `RunResult::Error` lands in per
//! [`RunErrorKind`], plus the segfault-message mirroring that decides whether a
//! native proc's memory error can be reported natively at all.

use super::*;

#[test]
fn error_deadend_kind_routes_to_deadended() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let sid = state.state_id();
        let inputs = PostStepInputs {
            result: RunResult::Error {
                message: "unliftable".to_string(),
                addr: 0x40_3000,
                kind: RunErrorKind::Deadend,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
                callbacks: None,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Deadended(s) => {
                assert_eq!(s.state_id(), sid);
                assert_eq!(s.pc(), 0x40_3000);
            }
            _ => panic!("expected Deadended"),
        }
    });
}

#[test]
fn error_fatal_kind_routes_to_errored() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let sid = state.state_id();
        let inputs = PostStepInputs {
            result: RunResult::Error {
                message: "boom".to_string(),
                addr: 0x40_4000,
                kind: RunErrorKind::Fatal,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
                callbacks: None,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Errored(s, msg) => {
                assert_eq!(s.state_id(), sid);
                assert_eq!(msg, "boom");
            }
            _ => panic!("expected Errored"),
        }
    });
}

/// angr-91vj9.9: a `Fatal` reported at pc 0 is deadended, not errored —
/// `ErrorRoute::NullAddressDeadend`. Pins the routing behaviour behind the
/// named variant (the classifier itself is unit-tested in `callbacks/events.rs`).
#[test]
fn error_fatal_at_null_addr_routes_to_deadended() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let sid = state.state_id();
        let inputs = PostStepInputs {
            result: RunResult::Error {
                message: "boom at null".to_string(),
                addr: 0,
                kind: RunErrorKind::Fatal,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
                callbacks: None,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Deadended(s) => assert_eq!(s.state_id(), sid),
            _ => panic!("expected Deadended for a Fatal error at pc 0"),
        }
    });
}

/// A native proc's unmapped-page error becomes a Python-identical
/// SimSegfaultException message — but only with STRICT_PAGE_ACCESS on, since
/// Python otherwise lazily initializes the page and keeps going (angr-gorvf.13).
#[test]
fn segfault_message_mirrors_python_strict_page_access() {
    use super::handlers::segfault_message;
    use crate::memory::MemoryError;
    use crate::procedures::ProcedureError;

    let mut state = RustSimState::new("amd64").unwrap();
    let unmapped = ProcedureError::Memory(MemoryError::Unmapped {
        addr: 0x1234,
        size: 4096,
    });

    // STRICT_PAGE_ACCESS off: Python services the read, so we must bounce.
    assert_eq!(segfault_message(&state, &unmapped), None);

    state.set_enforce_permissions(true);
    // Page-aligned, matching PrivilegedPagingMixin's `pageno * page_size`.
    assert_eq!(
        segfault_message(&state, &unmapped),
        Some("0x1000 (unmapped)".to_string())
    );

    // Every other decline still falls back to Python.
    for err in [
        ProcedureError::SymbolicArgument("n".into()),
        ProcedureError::NotImplemented,
        ProcedureError::Memory(MemoryError::UnmappedPageInRegion { page_addr: 0x1000 }),
    ] {
        assert_eq!(segfault_message(&state, &err), None);
    }
}
