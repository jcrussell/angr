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
//! 5. `fire_constraints_bp` (the wired body of `call_inspect_constraints`)
//!    fires the BP with the exported guard, and *drops* the event rather than
//!    propagating when the export fails.
//!
//! Only `call_inspect_constraints` touches claripy, and only on the *wired*
//! path. The cargo-test interpreter has no claripy installed, so the two tests
//! for invariant 5 hand `fire_constraints_bp` a stand-in module object
//! (`CLARIPY_STUBS`) instead of going through the `py.import("claripy")` in
//! `call_inspect_constraints`; nothing here needs the real package.

use super::*;
use crate::callbacks::test_support::{defs, obj, recorder};

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
        assert!(!cb.inspect_bit_enabled(bit), "bit {bit} on by default");
    }

    for bit in [0u8, 1, 16, 19, 20, 31] {
        cb.py_set_inspect_enabled(1u32 << bit);
        assert!(cb.inspect_bit_enabled(bit), "bit {bit} did not read back");
        assert_eq!(cb.py_get_inspect_enabled(), 1u32 << bit);
        let other = if bit == 0 { 1 } else { bit - 1 };
        assert!(
            !cb.inspect_bit_enabled(other),
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

/// Stand-in `claripy` modules for the two `fire_constraints_bp` tests below.
///
/// `good` is the smallest object `assumed_guard_to_claripy` accepts: a `BVV`
/// factory whose result carries a `.length` (so the 1-bit-BV branch is taken)
/// and an `__eq__` that builds the `guard == BVV(bit, 1)` comparison. `boom`
/// raises out of `BVV`, which is the only claripy call a `Concrete` guard
/// makes — that is how the export is driven into failure without needing a
/// real (unexportable) guard shape.
const CLARIPY_STUBS: &std::ffi::CStr = c"_calls = []
def bp(state_id, when, added):
    _calls.append((state_id, when, [a.tag for a in added]))

class _Ast:
    def __init__(self, tag, length):
        self.tag = tag
        self.length = length
    def __eq__(self, other):
        return _Ast(('eq', self.tag, other.tag), None)

class _Good:
    def BVV(self, value, width):
        return _Ast(('BVV', int(value), width), width)

class _Boom:
    def BVV(self, value, width):
        raise ValueError('export exploded')

good = _Good()
boom = _Boom()
";

/// Positive control for `constraints_export_failure_drops_the_event`: with a
/// claripy that exports cleanly, the BP *is* called with the materialized
/// one-element `added_constraints` list. Without this the failure test could
/// pass for the wrong reason (a stub that never reaches the export at all).
#[test]
fn constraints_bp_fires_with_the_exported_guard() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(py, CLARIPY_STUBS);
        let mut cb = PythonCallbacks::new();
        cb.set_inspect_constraints(obj(&globals, "bp"));
        let guard = crate::symbolic::RustBV::concrete(1, 1);
        let claripy = globals.get_item("good").unwrap().unwrap();

        PythonCallbacks::fire_constraints_bp(
            py,
            &obj(&globals, "bp"),
            &claripy,
            7,
            "before",
            &guard,
            true,
        )
        .expect("wired export must succeed");

        let calls = recorder(&globals, "_calls");
        assert_eq!(calls.len(), 1, "BP must fire exactly once");
        let call = calls.get_item(0).unwrap();
        assert_eq!(call.get_item(0).unwrap().extract::<i64>().unwrap(), 7);
        assert_eq!(call.get_item(1).unwrap().extract::<String>().unwrap(), "before");
        // One constraint, and it is the `guard == BVV(1, 1)` comparison
        // `assumed_guard_to_claripy` builds for a 1-bit BV guard.
        let tags = call.get_item(2).unwrap();
        assert_eq!(tags.len().unwrap(), 1);
        let tag = format!("{}", tags.get_item(0).unwrap());
        assert!(tag.contains("eq") && tag.contains("BVV"), "got {tag}");
    });
}

/// angr-0jh0j.5: the `SILENT(cat-b)` arm — an export failure drops the event
/// (`Ok(())`, BP not called) rather than propagating. `constraints` is an
/// observation-only breakpoint on a hot fork-guard path, so turning this into
/// a propagated error would abort the fork over a failed *notification*.
#[test]
fn constraints_export_failure_drops_the_event() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(py, CLARIPY_STUBS);
        let guard = crate::symbolic::RustBV::concrete(1, 1);
        let claripy = globals.get_item("boom").unwrap().unwrap();

        for (when, is_true) in [("before", true), ("after", false)] {
            PythonCallbacks::fire_constraints_bp(
                py,
                &obj(&globals, "bp"),
                &claripy,
                7,
                when,
                &guard,
                is_true,
            )
            .expect("a failed export must not propagate out of an inspect hook");
        }

        assert_eq!(
            recorder(&globals, "_calls").len(),
            0,
            "BP must not fire with an unmaterialized constraint"
        );
    });
}

/// One row of the [`INSPECT_DISPATCH_CASES`] pairing table: an inspect event,
/// the setter for its callback slot, and a call of its dispatch method.
struct InspectDispatchCase {
    /// The `state.inspect` event name — the shared suffix of the
    /// `inspect_<event>` slot and the `call_inspect_<event>` method.
    event: &'static str,
    /// Wire the `inspect_<event>` slot (the generated `set_inspect_<event>`).
    set: fn(&mut PythonCallbacks, Py<PyAny>),
    /// Call `call_inspect_<event>` with placeholder arguments, discarding any
    /// value-injection return — the test only observes *whether* it fired.
    invoke: fn(&PythonCallbacks) -> PyResult<()>,
    /// `call_inspect_constraints` imports claripy *after* its slot check, and
    /// the cargo-test interpreter has no claripy (see the module doc), so a
    /// wired constraints slot is observable as an `Err` rather than as a BP
    /// fire. `true` makes an error count as "this slot was read".
    may_err_when_wired: bool,
}

/// Every Rust-dispatched `state.inspect` event, paired with the slot its
/// dispatch method is supposed to read.
///
/// A transposition *inside this table* cannot make the pairing test pass
/// vacuously: `each_call_inspect_dispatches_only_its_own_slot` requires the
/// diagonal to fire, so a row whose `set` and `invoke` name different events
/// fails at `i == j` rather than sliding through.
const INSPECT_DISPATCH_CASES: &[InspectDispatchCase] = &[
    InspectDispatchCase {
        event: "mem_read",
        set: PythonCallbacks::set_inspect_mem_read,
        invoke: |cb| {
            cb.call_inspect_mem_read(1, "before", 0x1000, 4, None, "Iend_LE")
                .map(drop)
        },
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "mem_write",
        set: PythonCallbacks::set_inspect_mem_write,
        invoke: |cb| {
            cb.call_inspect_mem_write(1, "before", 0x1000, 4, None, "Iend_LE")
                .map(drop)
        },
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "reg_read",
        set: PythonCallbacks::set_inspect_reg_read,
        invoke: |cb| cb.call_inspect_reg_read(1, "before", 16, 8, None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "reg_write",
        set: PythonCallbacks::set_inspect_reg_write,
        invoke: |cb| cb.call_inspect_reg_write(1, "before", 16, 8, None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "fork",
        set: PythonCallbacks::set_inspect_fork,
        invoke: |cb| cb.call_inspect_fork(1, "after"),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "exit",
        set: PythonCallbacks::set_inspect_exit,
        invoke: |cb| cb.call_inspect_exit(1, "before", 0x1000, "Ijk_Boring", None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "instruction",
        set: PythonCallbacks::set_inspect_instruction,
        invoke: |cb| cb.call_inspect_instruction(1, "before", 0x1000),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "irsb",
        set: PythonCallbacks::set_inspect_irsb,
        invoke: |cb| cb.call_inspect_irsb(1, "before", 0x1000),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "call",
        set: PythonCallbacks::set_inspect_call,
        invoke: |cb| cb.call_inspect_call(1, "before", 0x1000),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "return",
        set: PythonCallbacks::set_inspect_return,
        invoke: |cb| cb.call_inspect_return(1, "after", 0x1000),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "tmp_read",
        set: PythonCallbacks::set_inspect_tmp_read,
        invoke: |cb| cb.call_inspect_tmp_read(1, "after", 3, None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "tmp_write",
        set: PythonCallbacks::set_inspect_tmp_write,
        invoke: |cb| cb.call_inspect_tmp_write(1, "after", 3, None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "statement",
        set: PythonCallbacks::set_inspect_statement,
        invoke: |cb| cb.call_inspect_statement(1, "before", 0),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "expr",
        set: PythonCallbacks::set_inspect_expr,
        invoke: |cb| cb.call_inspect_expr(1, "after", None),
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "address_concretization",
        set: PythonCallbacks::set_inspect_address_concretization,
        invoke: |cb| {
            Python::attach(|py| {
                let addr = py.None();
                cb.call_inspect_address_concretization(1, "after", "load", &addr, Some(vec![0x1000]))
            })
        },
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "symbolic_variable",
        set: PythonCallbacks::set_inspect_symbolic_variable,
        invoke: |cb| {
            Python::attach(|py| {
                let expr = py.None();
                cb.call_inspect_symbolic_variable(1, "after", "mem_1000", 64, &expr)
            })
        },
        may_err_when_wired: false,
    },
    InspectDispatchCase {
        event: "constraints",
        set: PythonCallbacks::set_inspect_constraints,
        invoke: |cb| {
            let guard = crate::symbolic::RustBV::concrete(1, 1);
            cb.call_inspect_constraints(1, "before", &guard, true)
        },
        may_err_when_wired: true,
    },
    InspectDispatchCase {
        event: "vex_lift",
        set: PythonCallbacks::set_inspect_vex_lift,
        invoke: |cb| cb.call_inspect_vex_lift(-1, "before", 0x1000, None, Some(&[0x90])),
        may_err_when_wired: false,
    },
];

/// The `call_inspect_*` methods hand-copy the slot they read
/// (`self.inspect_<event>.as_ref()`), one layer below where
/// `inspect_test_entries!` made the *entry-point* naming compiler-enforced
/// (angr-0jh0j.6). Leaving `self.inspect_reg_read.as_ref()` in a body
/// copy-pasted into `call_inspect_reg_write` compiles cleanly and silently
/// routes every reg_write breakpoint to the reg_read callback — the failure
/// only ever surfaces as a user-reported misrouted breakpoint, because
/// `unset_inspect_callbacks_are_ok` exercises the all-`None` case only.
///
/// This is the N x N pairing check: wire exactly one slot, call all 18
/// dispatch methods, and require that precisely the matching one observes the
/// callback (angr-0jh0j.81).
#[test]
fn each_call_inspect_dispatches_only_its_own_slot() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(py, c"fired = []\ndef rec(*a):\n    fired.append(a)\n");
        let rec = obj(&globals, "rec");
        let fired = recorder(&globals, "fired");

        for wired in INSPECT_DISPATCH_CASES {
            let mut cb = PythonCallbacks::new();
            (wired.set)(&mut cb, rec.clone_ref(py));

            for called in INSPECT_DISPATCH_CASES {
                let before = fired.len();
                let res = (called.invoke)(&cb);
                let observed =
                    fired.len() > before || (called.may_err_when_wired && res.is_err());
                assert_eq!(
                    observed,
                    wired.event == called.event,
                    "wired inspect_{} but call_inspect_{} {} the slot",
                    wired.event,
                    called.event,
                    if observed { "read" } else { "did not read" },
                );
            }
        }
    });
}

/// Keeps [`INSPECT_DISPATCH_CASES`] from silently missing a dispatch method:
/// a new `call_inspect_<event>` needs a new [`InspectBit`] row anyway (the
/// enabled-mask gate), so cross-checking the table against `InspectBit::ALL`
/// turns "forgot to extend the pairing test" into a failing test.
#[test]
fn dispatch_pairing_table_covers_every_rust_dispatched_event() {
    // Dispatched from Python, so they have no `call_inspect_*` method — see
    // the `InspectBit::SimProcedure` doc comment.
    const PYTHON_DISPATCHED: &[&str] = &["simprocedure", "syscall", "dirty"];

    for event in InspectBit::ALL {
        let name = event.event_name();
        if PYTHON_DISPATCHED.contains(&name) {
            continue;
        }
        assert!(
            INSPECT_DISPATCH_CASES.iter().any(|c| c.event == name),
            "inspect event '{name}' has no INSPECT_DISPATCH_CASES row",
        );
    }
    assert_eq!(
        INSPECT_DISPATCH_CASES.len(),
        InspectBit::ALL.len() - PYTHON_DISPATCHED.len(),
        "INSPECT_DISPATCH_CASES has a row with no matching InspectBit",
    );
}
