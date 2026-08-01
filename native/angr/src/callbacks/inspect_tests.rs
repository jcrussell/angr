//! In-module unit tests for `callbacks/inspect.rs` (angr-9ke6b.28).
//!
//! `state.inspect` dispatch has the *opposite* unset-callback contract from
//! `dispatch.rs`: a breakpoint that was never registered is normal, so every
//! `call_inspect_*` returns `Ok` when its callback is `None`. That makes the
//! module's real invariants easy to break silently, so they are pinned here:
//!
//! 1. Unset → `Ok(())` / `Ok(None)`, never an error and never a Python call.
//! 2. Errors raised *inside* a registered BP action propagate (the engine
//!    surfaces user-action failures rather than swallowing them).
//! 3. `inspect_event_enabled` reads the right bit of the `AtomicU32` mask,
//!    including the high bits added after the widening (`expr` at 16,
//!    `constraints` at 19, `vex_lift` at 20).
//! 4. The `Option<&Py<PyAny>>` value-injection channel on `mem_read` /
//!    `mem_write` maps a Python `None` return to `None` and anything else to
//!    `Some`.
//!
//! Only `call_inspect_constraints` touches claripy, and only on the *wired*
//! path; every assertion here stays claripy-free.

use super::*;
use pyo3::types::{PyDict, PyList};

/// Execute `src` in a fresh globals dict and hand it back so tests can read
/// the recorder lists the snippet defines.
fn defs<'py>(py: Python<'py>, src: &std::ffi::CStr) -> pyo3::Bound<'py, PyDict> {
    let globals = PyDict::new(py);
    py.run(src, Some(&globals), None)
        .expect("define test callbacks");
    globals
}

/// Pull a named object out of a `defs()` globals dict as an owned `Py<PyAny>`.
fn obj(globals: &pyo3::Bound<'_, PyDict>, name: &str) -> Py<PyAny> {
    globals
        .get_item(name)
        .unwrap()
        .unwrap_or_else(|| panic!("{name} not defined"))
        .unbind()
}

/// Read a recorder list defined by a `defs()` snippet.
fn recorder<'py>(globals: &pyo3::Bound<'py, PyDict>, name: &str) -> pyo3::Bound<'py, PyList> {
    globals
        .get_item(name)
        .unwrap()
        .unwrap_or_else(|| panic!("{name} not defined"))
        .cast_into::<PyList>()
        .expect("recorder must be a list")
}

/// The enabled-mask is the O(1) gate in front of every inspect crossing; an
/// off-by-one in the shift silently disables (or worse, spuriously enables) a
/// breakpoint type. Bits 16/19/20 exist only because the mask was widened from
/// `AtomicU16` to `AtomicU32` (angr-lge2), so they are checked explicitly.
#[test]
fn inspect_event_enabled_reads_the_right_bit() {
    Python::initialize();
    let cb = PythonCallbacks::new();

    // Default is 0: nothing enabled.
    for bit in [0u8, 1, 16, 19, 20, 31] {
        assert!(!cb.inspect_event_enabled(bit), "bit {bit} on by default");
    }

    for bit in [0u8, 1, 16, 19, 20, 31] {
        cb.py_set_inspect_enabled(1u32 << bit);
        assert!(cb.inspect_event_enabled(bit), "bit {bit} did not read back");
        assert_eq!(cb.py_get_inspect_enabled(), 1u32 << bit);
        let other = if bit == 0 { 1 } else { bit - 1 };
        assert!(
            !cb.inspect_event_enabled(other),
            "bit {bit} leaked into bit {other}"
        );
    }
}

/// Unset inspect callbacks are the common case (no breakpoint registered) and
/// must be a silent `Ok`, unlike the `dispatch.rs` callbacks.
#[test]
fn unset_inspect_callbacks_are_ok() {
    Python::initialize();
    Python::attach(|py| {
        let cb = PythonCallbacks::new();
        let none = py.None();
        let guard = crate::symbolic::RustBV::concrete(1, 1);

        assert!(
            cb.call_inspect_mem_read(1, "before", 0x1000, 4, None, "Iend_LE")
                .unwrap()
                .is_none()
        );
        assert!(
            cb.call_inspect_mem_write(1, "before", 0x1000, 4, None, "Iend_LE")
                .unwrap()
                .is_none()
        );
        cb.call_inspect_reg_read(1, "before", 16, 8, None).unwrap();
        cb.call_inspect_reg_write(1, "before", 16, 8, None).unwrap();
        cb.call_inspect_instruction(1, "before", 0x1000).unwrap();
        cb.call_inspect_irsb(1, "before", 0x1000).unwrap();
        cb.call_inspect_exit(1, "before", 0x1000, "Ijk_Boring", None)
            .unwrap();
        cb.call_inspect_call(1, "before", 0x1000).unwrap();
        cb.call_inspect_return(1, "after", 0x1000).unwrap();
        cb.call_inspect_tmp_read(1, "after", 3, None).unwrap();
        cb.call_inspect_tmp_write(1, "after", 3, None).unwrap();
        cb.call_inspect_statement(1, "before", 0).unwrap();
        cb.call_inspect_expr(1, "after", None).unwrap();
        cb.call_inspect_address_concretization(1, "after", "load", &none, Some(vec![0x1000]))
            .unwrap();
        cb.call_inspect_symbolic_variable(1, "after", "mem_1000", 64, &none)
            .unwrap();
        cb.call_inspect_fork(1, "after").unwrap();
        // Claripy-free precisely because the unset arm returns before the import.
        cb.call_inspect_constraints(1, "before", &guard, true)
            .unwrap();
        cb.call_inspect_vex_lift(-1, "before", 0x1000, None, Some(&[0x90]))
            .unwrap();
    });
}

/// Value injection (angr-uy32 / angr-inh0): a BP action that leaves
/// `mem_read_expr` unchanged returns Python `None` and must map to `None`; a
/// substituted AST must come back as `Some` so the caller can swap it in.
#[test]
fn mem_read_maps_none_return_to_none_and_object_to_some() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_calls = []
def bp(state_id, when, addr, size, value, endness):
    _calls.append((state_id, when, addr, size, value, endness))
    return 'substituted' if addr == 0x2000 else None
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_mem_read(obj(&globals, "bp"));

        assert!(
            cb.call_inspect_mem_read(7, "before", 0x1000, 4, None, "Iend_LE")
                .unwrap()
                .is_none(),
            "a None return means 'unchanged'"
        );
        let injected = cb
            .call_inspect_mem_read(7, "after", 0x2000, 4, None, "Iend_BE")
            .unwrap()
            .expect("substituted AST must be returned to the caller");
        assert_eq!(injected.extract::<String>(py).unwrap(), "substituted");

        let calls = recorder(&globals, "_calls");
        assert_eq!(calls.len(), 2);
        let first = calls.get_item(0).unwrap();
        assert_eq!(first.get_item(0).unwrap().extract::<i64>().unwrap(), 7);
        assert_eq!(
            first.get_item(1).unwrap().extract::<String>().unwrap(),
            "before"
        );
        assert_eq!(
            first.get_item(5).unwrap().extract::<String>().unwrap(),
            "Iend_LE"
        );
        // A `None` value_ast is passed through as Python None, not omitted.
        assert!(first.get_item(4).unwrap().is_none());
    });
}

/// `mem_write` shares the injection channel; both fire directions must reach
/// the BP with their `when` intact (before = pre-store override, after =
/// informational).
#[test]
fn mem_write_passes_both_fire_directions() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_whens = []
def bp(state_id, when, addr, size, value, endness):
    _whens.append(when)
    return None
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_mem_write(obj(&globals, "bp"));

        let value = py.None();
        cb.call_inspect_mem_write(1, "before", 0x1000, 8, Some(&value), "Iend_LE")
            .unwrap();
        cb.call_inspect_mem_write(1, "after", 0x1000, 8, Some(&value), "Iend_LE")
            .unwrap();

        let whens: Vec<String> = recorder(&globals, "_whens").extract().unwrap();
        assert_eq!(whens, vec!["before".to_string(), "after".to_string()]);
    });
}

/// An `Option<&Py<PyAny>>` of `None` must arrive as Python `None` in the guard
/// slot — dropping the argument would shift every later positional parameter.
#[test]
fn exit_passes_absent_guard_as_python_none() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_calls = []
def bp(*args):
    _calls.append(args)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_exit(obj(&globals, "bp"));

        cb.call_inspect_exit(3, "before", 0x400f00, "Ijk_Boring", None)
            .unwrap();

        let calls = recorder(&globals, "_calls");
        let args = calls.get_item(0).unwrap();
        assert_eq!(args.len().unwrap(), 5);
        assert_eq!(
            args.get_item(3).unwrap().extract::<String>().unwrap(),
            "Ijk_Boring"
        );
        assert!(args.get_item(4).unwrap().is_none());
    });
}

/// `address_concretization` carries the result list only on the AFTER fire;
/// BEFORE must pass Python `None`, not an empty list (the BP distinguishes
/// "not concretized yet" from "concretized to nothing").
#[test]
fn address_concretization_result_is_none_before_and_a_list_after() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_results = []
def bp(state_id, when, action, addr, result):
    _results.append(result)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_address_concretization(obj(&globals, "bp"));

        let addr_ast = py.None();
        cb.call_inspect_address_concretization(1, "before", "store", &addr_ast, None)
            .unwrap();
        cb.call_inspect_address_concretization(
            1,
            "after",
            "store",
            &addr_ast,
            Some(vec![0x1000, 0x1008]),
        )
        .unwrap();

        let results = recorder(&globals, "_results");
        assert!(results.get_item(0).unwrap().is_none());
        assert_eq!(
            results.get_item(1).unwrap().extract::<Vec<u64>>().unwrap(),
            vec![0x1000, 0x1008]
        );
    });
}

/// `vex_lift` mirrors `_cb_lift_block`: bytes + `size=None` on BEFORE, the
/// lifted size and no bytes on AFTER, with `state_id == -1` throughout.
#[test]
fn vex_lift_passes_bytes_before_and_size_after() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_calls = []
def bp(state_id, when, addr, size, buff):
    _calls.append((state_id, when, size, buff))
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_vex_lift(obj(&globals, "bp"));

        cb.call_inspect_vex_lift(-1, "before", 0x1000, None, Some(&[0x90, 0xc3]))
            .unwrap();
        cb.call_inspect_vex_lift(-1, "after", 0x1000, Some(2), None)
            .unwrap();

        let calls = recorder(&globals, "_calls");
        let before = calls.get_item(0).unwrap();
        assert_eq!(before.get_item(0).unwrap().extract::<i64>().unwrap(), -1);
        assert!(before.get_item(2).unwrap().is_none(), "BEFORE has no size");
        assert_eq!(
            before.get_item(3).unwrap().extract::<Vec<u8>>().unwrap(),
            vec![0x90, 0xc3]
        );
        let after = calls.get_item(1).unwrap();
        assert_eq!(after.get_item(2).unwrap().extract::<u32>().unwrap(), 2);
        assert!(
            after.get_item(3).unwrap().is_none(),
            "AFTER carries no bytes"
        );
    });
}

/// A BP action that raises must not be swallowed — the engine surfaces
/// user-action failures instead of continuing with a half-applied inspect.
#[test]
fn inspect_errors_from_the_bp_action_propagate() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"def boom(*args):
    raise ValueError('bp exploded')
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_instruction(obj(&globals, "boom"));
        cb.set_inspect_mem_read(obj(&globals, "boom"));

        let err = cb
            .call_inspect_instruction(1, "before", 0x1000)
            .expect_err("BP exception must propagate");
        assert!(err.to_string().contains("bp exploded"), "got {err}");

        let err = cb
            .call_inspect_mem_read(1, "before", 0x1000, 4, None, "Iend_LE")
            .expect_err("BP exception must propagate");
        assert!(err.to_string().contains("bp exploded"), "got {err}");
    });
}
