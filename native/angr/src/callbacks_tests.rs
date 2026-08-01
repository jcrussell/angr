use super::*;

#[test]
fn test_callbacks_creation() {
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    assert!(!callbacks.is_ready());
}

#[test]
fn test_loop_execution_event() {
    let event = LoopExecutionEvent::from_run_result(
        RunResult::BlockEnd {
            next_addr: 0x1000,
            jumpkind: "Ijk_Boring".to_string(),
        },
        5,
    );
    assert_eq!(event.event_type, "block_end");
    assert_eq!(event.pc, Some(0x1000));
    assert_eq!(event.blocks_executed, 5);
}

/// angr-ph300.66: `call_memory_store_symbolic_full` must hard-error (not
/// silently `Ok(())`) when the callback is unset — module invariant 1
/// (`avoid-silent-no-op-callback-fallbacks`). A silent no-op would drop the
/// store and diverge Rust↔Python memory. The unset arm returns before any
/// claripy import, so this test does not need claripy on sys.path.
#[test]
fn store_symbolic_full_errors_when_callback_unset() {
    use crate::symbolic::RustBV;

    Python::initialize();
    let cb = PythonCallbacks::new();
    assert!(!cb.has_memory_store_symbolic_full());

    let addr = RustBV::concrete(0x1000, 64);
    let data = RustBV::concrete(0x41, 8);
    let res = cb.call_memory_store_symbolic_full(&addr, &data);
    let err = res.expect_err("unset full-store callback must error, not no-op");
    Python::attach(|py| {
        assert!(
            err.to_string()
                .contains("memory_store_symbolic_full callback not set"),
            "unexpected error message: {}",
            err.value(py)
        );
    });
}

/// angr-ph300.64: a multi-address store of *concrete* data must route to the
/// wired `memory_store_symbolic_full` callback (which stores across every
/// concretized candidate via Python's memory model), NOT degrade to the
/// first-address-only fallback that silently drops addrs[1..].
#[test]
fn concrete_multi_addr_store_prefers_full_callback() {
    use crate::symbolic::{RustBV, SymContext};
    use pyo3::types::{PyDict, PyList};

    let ctx = SymContext::new_mock();

    Python::initialize();
    Python::attach(|py| {
        // rustbv_to_claripy imports claripy; skip when it is not importable.
        if py.import("claripy").is_err() {
            return;
        }

        let globals = PyDict::new(py);
        py.run(
            c"_full = []
_first = []
def full_cb(addr_ast, data_ast):
    _full.append((addr_ast.op, data_ast.size()))
def store_cb(addr, data):
    _first.append(addr)
",
            Some(&globals),
            None,
        )
        .expect("define recorder callbacks");

        let full_cb = globals.get_item("full_cb").unwrap().unwrap();
        let store_cb = globals.get_item("store_cb").unwrap().unwrap();

        let mut cb = PythonCallbacks::new();
        cb.set_memory_store_symbolic_full(full_cb.unbind());
        cb.set_memory_store(store_cb.unbind());
        assert!(cb.has_memory_store_symbolic_full());

        // Symbolic address concretized to 17 candidates (> the in-Rust ITE
        // cap), concrete data — the exact `table[x]=const` shape from the bug.
        let addr_ast = RustBV::symbolic(&ctx, "store_addr", 64);
        let data = RustBV::concrete(0x41, 8);
        let addrs: Vec<u64> = (0..17).map(|i| 0x1000 + i * 8).collect();

        cb.call_memory_store_symbolic(&addrs, &data, &addr_ast)
            .expect("call_memory_store_symbolic");

        let full = globals.get_item("_full").unwrap().unwrap();
        let full = full.cast::<PyList>().unwrap();
        assert_eq!(full.len(), 1, "full callback must fire exactly once");

        let first = globals.get_item("_first").unwrap().unwrap();
        let first = first.cast::<PyList>().unwrap();
        assert_eq!(
            first.len(),
            0,
            "first-address-only fallback must NOT fire when full callback is wired"
        );
    });
}

/// angr-9ke6b.27: `clear_fields` (the body of `__clear__`, and of
/// `RustExplorationManager.__clear__`) must drop *every* `Py<PyAny>` callback
/// slot. A slot it misses keeps the `mgr -> _callbacks -> bound method -> mgr`
/// reference cycle alive and leaks the manager plus its `_state_cache`.
///
/// Completeness is enforced at compile time by the exhaustive destructure in
/// `clear_fields` (see `with_callback_fields!`); this pins the runtime half —
/// that the fields really are dropped and `inspect_enabled` really is reset,
/// so a future rewrite away from the macro still fails loudly.
#[test]
fn clear_fields_drops_every_callback_slot() {
    use std::sync::atomic::Ordering;

    Python::initialize();
    Python::attach(|py| {
        let sentinel: Py<PyAny> = py.None();

        macro_rules! fill_and_check {
            ($($field:ident),+ $(,)?) => {{
                let mut cb = PythonCallbacks::new();
                $(cb.$field = Some(sentinel.clone_ref(py));)+
                cb.inspect_enabled.store(u32::MAX, Ordering::Relaxed);
                assert!(cb.is_ready(), "every slot should be populated pre-clear");

                cb.clear_fields();

                $(assert!(
                    cb.$field.is_none(),
                    concat!("clear_fields left `", stringify!($field), "` set"),
                );)+
                assert_eq!(
                    cb.inspect_enabled.load(Ordering::Relaxed),
                    0,
                    "clear_fields must reset the inspect bitmask",
                );
                assert!(!cb.is_ready(), "cleared holder must not report ready");
            }};
        }
        with_callback_fields!(fill_and_check);
    });
}
