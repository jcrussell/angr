//! In-module unit tests for `callbacks/dispatch.rs` (angr-9ke6b.28).
//!
//! Two contracts get pinned here, both cheap to break by accident:
//!
//! 1. **Unset callbacks hard-error** (module invariant 1,
//!    `avoid-silent-no-op-callback-fallbacks`). Every `call_*` that has no
//!    legitimate degraded mode must return `Err` naming the missing callback,
//!    never `Ok` with a fabricated value. A regression here is silent
//!    Rust↔Python divergence, not a crash.
//! 2. **The deliberate fallbacks that DO exist** (`call_memory_store_batch`,
//!    `call_memory_load_batch`, `call_batch_fetch_pages`,
//!    `call_memory_store_symbolic_value`) must actually reach the per-item
//!    callback rather than dropping the work.
//!
//! Nothing here needs claripy: every assertion is on a path that returns
//! before `rustbv_to_claripy` / `py.import("claripy")`.

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

/// Every dispatch callback with no legitimate degraded mode must surface a
/// named error when it is not wired, rather than fabricating a result.
#[test]
fn unset_dispatch_callbacks_hard_error() {
    Python::initialize();
    let cb = PythonCallbacks::new();
    let addr_bv = RustBV::concrete(0x1000, 64);

    // (expected substring of the error, probe)
    #[allow(clippy::type_complexity)]
    let probes: Vec<(&str, Box<dyn Fn() -> PyResult<()> + '_>)> = vec![
        (
            "memory_load callback not set",
            Box::new(|| cb.call_memory_load(0x1000, 4).map(|_| ())),
        ),
        (
            "memory_store callback not set",
            Box::new(|| cb.call_memory_store(0x1000, &[0u8; 4])),
        ),
        (
            "on_hook callback not set",
            Box::new(|| cb.call_on_hook(0x1000).map(|_| ())),
        ),
        (
            "on_syscall callback not set",
            Box::new(|| cb.call_on_syscall(60)),
        ),
        (
            "lift_block callback not set",
            Box::new(|| cb.call_lift_block(0x1000, None, None).map(|_| ())),
        ),
        (
            "get_register callback not set",
            Box::new(|| cb.call_get_register(16, 8).map(|_| ())),
        ),
        (
            "put_register callback not set",
            Box::new(|| cb.call_put_register(16, &[0u8; 8])),
        ),
        (
            "dirty_call callback not set",
            Box::new(|| {
                cb.call_dirty_call("amd64g_dirtyhelper_RDTSC", &[], 64)
                    .map(|_| ())
            }),
        ),
        (
            "fetch_page callback not set",
            Box::new(|| cb.call_fetch_page(0x1000).map(|_| ())),
        ),
        (
            "resolve_function callback not set",
            Box::new(|| cb.call_resolve_function(0x1000, None).map(|_| ())),
        ),
        (
            "memory_load_symbolic_full callback not set",
            Box::new(|| cb.call_memory_load_symbolic_full(&addr_bv, 8).map(|_| ())),
        ),
    ];

    for (expected, probe) in probes {
        let err = probe().expect_err(expected);
        assert!(
            err.to_string().contains(expected),
            "expected error containing {expected:?}, got {err}"
        );
    }
}

/// `call_memory_store_symbolic` hard-errors when `_full` is unwired rather
/// than storing only `addrs[0]` (the angr-ph300.64 divergent-memory bug).
/// Complements `concrete_multi_addr_store_prefers_full_callback` in
/// `callbacks_tests.rs`, which pins the wired half.
#[test]
fn store_symbolic_errors_when_full_callback_unset() {
    Python::initialize();
    let cb = PythonCallbacks::new();
    let addr_ast = RustBV::concrete(0x1000, 64);
    let data = RustBV::concrete(0x41, 8);

    let err = cb
        .call_memory_store_symbolic(&[0x1000, 0x1008], &data, &addr_ast)
        .expect_err("multi-addr store without _full must error, not drop addrs[1..]");
    assert!(
        err.to_string()
            .contains("memory_store_symbolic_full callback not set"),
        "unexpected error: {err}"
    );
}

/// The documented degraded mode of `call_memory_store_symbolic_value`: with no
/// symbolic-value callback it falls back to the byte-level `memory_store`.
/// Asserted with *concrete* data, where `bv_to_bytes` is exact — the symbolic
/// zero-fill case is the separate open bug angr-9ke6b.19 and is deliberately
/// not pinned here.
#[test]
fn store_symbolic_value_falls_back_to_byte_store() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_stores = []
def store_cb(addr, data):
    _stores.append((addr, bytes(data)))
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_memory_store(obj(&globals, "store_cb"));
        assert!(!cb.has_memory_store_symbolic_value());

        let value = RustBV::concrete(0x0102, 16);
        cb.call_memory_store_symbolic_value(0x2000, &value)
            .expect("fallback store");

        let stores = recorder(&globals, "_stores");
        assert_eq!(
            stores.len(),
            1,
            "fallback must reach memory_store exactly once"
        );
        let (addr, data): (u64, Vec<u8>) = stores.get_item(0).unwrap().extract().unwrap();
        assert_eq!(addr, 0x2000);
        // bv_to_bytes is little-endian.
        assert_eq!(data, vec![0x02, 0x01]);
    });
}

/// With no `memory_store_batch` wired, every store must still reach the
/// per-store callback — a fallback that dropped items would silently lose
/// writes. The empty-input fast path must not touch Python at all.
#[test]
fn store_batch_falls_back_to_individual_stores() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_stores = []
def store_cb(addr, data):
    _stores.append(addr)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_memory_store(obj(&globals, "store_cb"));

        // Empty batch is a no-op even with nothing wired at all.
        PythonCallbacks::new()
            .call_memory_store_batch(&[])
            .expect("empty batch must be a no-op");

        cb.call_memory_store_batch(&[(0x10, vec![1, 2]), (0x20, vec![3])])
            .expect("batch store");

        let stores = recorder(&globals, "_stores");
        let addrs: Vec<u64> = stores.extract().unwrap();
        assert_eq!(addrs, vec![0x10, 0x20]);
    });
}

/// Same contract on the load side, plus the tuple-arity handling: a 2-tuple
/// means "no symbolic AST", a 3-tuple with `None` means the same.
#[test]
fn load_batch_falls_back_to_individual_loads() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_loads = []
def load_cb(addr, size):
    _loads.append((addr, size))
    if addr == 0x10:
        return (b'\\x01' * size, False)
    return (b'\\x02' * size, True, None)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_memory_load(obj(&globals, "load_cb"));

        assert!(
            PythonCallbacks::new()
                .call_memory_load_batch(&[])
                .expect("empty batch must be a no-op")
                .is_empty()
        );

        let results = cb
            .call_memory_load_batch(&[(0x10, 2), (0x20, 1)])
            .expect("batch load");

        assert_eq!(recorder(&globals, "_loads").len(), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, vec![1, 1]);
        assert!(!results[0].1);
        assert!(results[0].2.is_none(), "2-tuple must yield no AST");
        assert_eq!(results[1].0, vec![2]);
        assert!(results[1].1);
        assert!(results[1].2.is_none(), "explicit None AST must yield None");
    });
}

/// A non-None third element must survive as `Some` — this is the value-injection
/// channel the interpreter relies on to keep symbolic loads symbolic.
#[test]
fn memory_load_preserves_symbolic_ast_when_present() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"def load_cb(addr, size):
    return (b'\\x00' * size, True, 'sentinel-ast')
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_memory_load(obj(&globals, "load_cb"));

        let (_data, is_symbolic, ast) = cb.call_memory_load(0x30, 4).expect("load");
        assert!(is_symbolic);
        let ast = ast.expect("non-None third element must be preserved");
        assert_eq!(ast.extract::<String>(py).unwrap(), "sentinel-ast");
    });
}

/// `call_batch_fetch_pages` degrades to `call_fetch_page` per page.
#[test]
fn batch_fetch_pages_falls_back_to_individual_fetches() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_pages = []
def fetch_cb(page_addr):
    _pages.append(page_addr)
    return (b'\\xcc' * 4096, 5, True)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_fetch_page(obj(&globals, "fetch_cb"));

        assert!(
            PythonCallbacks::new()
                .call_batch_fetch_pages(&[])
                .expect("empty fetch must be a no-op")
                .is_empty()
        );

        let pages = cb
            .call_batch_fetch_pages(&[0x1000, 0x2000])
            .expect("batch fetch");
        assert_eq!(recorder(&globals, "_pages").len(), 2);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].0.len(), 4096);
        assert_eq!(pages[0].1, 5);
        assert!(pages[0].2);
    });
}

/// `call_lift_block` builds four different positional-arg shapes depending on
/// `opt_level` / `dirty_bytes`. The Python endpoint reads them positionally, so
/// an arity change silently mis-binds `byte_string=`.
#[test]
fn lift_block_arg_shapes_match_opt_level_and_dirty_bytes() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"_calls = []
def lift_cb(*args):
    _calls.append(args)
    return 'irsb-json'
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_lift_block(obj(&globals, "lift_cb"));

        assert_eq!(cb.call_lift_block(0x1000, None, None).unwrap(), "irsb-json");
        cb.call_lift_block(0x1000, Some(1), None).unwrap();
        cb.call_lift_block(0x1000, None, Some(&[0xaa, 0xbb]))
            .unwrap();
        cb.call_lift_block(0x1000, Some(2), Some(&[0xcc])).unwrap();

        let calls = recorder(&globals, "_calls");
        let arities: Vec<usize> = (0..calls.len())
            .map(|i| calls.get_item(i).unwrap().len().unwrap())
            .collect();
        assert_eq!(arities, vec![1, 2, 3, 3]);

        // With dirty bytes but no opt_level, the opt_level slot must be None so
        // the bytes land in the third positional parameter.
        let third = calls.get_item(2).unwrap();
        assert!(third.get_item(1).unwrap().is_none());
        assert_eq!(
            third.get_item(2).unwrap().extract::<Vec<u8>>().unwrap(),
            vec![0xaa, 0xbb]
        );
    });
}

/// `call_resolve_function` has three legal outcomes; the malformed-tuple arm
/// must reject rather than truncate.
#[test]
fn resolve_function_handles_none_tuple_and_bad_arity() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"def resolve_cb(addr, symbol_name):
    if addr == 0:
        return None
    if addr == 1:
        return ('strlen', 1, False)
    return ('bad', 1)
",
        );
        let mut cb = PythonCallbacks::new();
        cb.set_resolve_function(obj(&globals, "resolve_cb"));
        assert!(cb.has_resolve_function());

        assert_eq!(cb.call_resolve_function(0, None).unwrap(), None);
        assert_eq!(
            cb.call_resolve_function(1, Some("strlen")).unwrap(),
            Some(("strlen".to_string(), 1, false))
        );
        let err = cb
            .call_resolve_function(2, None)
            .expect_err("a 2-tuple must be rejected, not silently truncated");
        assert!(
            err.to_string().contains("(name, num_args, no_return)"),
            "unexpected error: {err}"
        );
    });
}

/// Both page-universe predicates fail *open* when Python never installed a
/// snapshot — an unknown page must keep crossing exactly as before
/// (angr-gorvf.4.6 / .4.7).
#[test]
fn page_predicates_fail_open_without_snapshot() {
    Python::initialize();
    let cb = PythonCallbacks::new();
    assert!(cb.python_can_serve_page(0xdead_0000));
    assert!(cb.python_has_page(0xdead_0000));
}

/// The `has_*` predicates gate whole dispatch paths; they must track the
/// setters they name.
#[test]
fn has_predicates_track_their_setters() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(py, c"def noop(*args):\n    return None\n");
        let mut cb = PythonCallbacks::new();

        assert!(!cb.has_dirty_call());
        assert!(!cb.has_fetch_page());
        assert!(!cb.has_memory_store_symbolic_value());
        assert!(!cb.has_memory_store_symbolic_full());
        assert!(!cb.has_memory_load_symbolic_full());
        assert!(!cb.has_resolve_function());
        assert!(!cb.is_ready());

        cb.set_dirty_call(obj(&globals, "noop"));
        cb.set_fetch_page(obj(&globals, "noop"));
        cb.set_memory_store_symbolic_value(obj(&globals, "noop"));
        cb.set_memory_store_symbolic_full(obj(&globals, "noop"));
        cb.set_memory_load_symbolic_full(obj(&globals, "noop"));
        cb.set_resolve_function(obj(&globals, "noop"));

        assert!(cb.has_dirty_call());
        assert!(cb.has_fetch_page());
        assert!(cb.has_memory_store_symbolic_value());
        assert!(cb.has_memory_store_symbolic_full());
        assert!(cb.has_memory_load_symbolic_full());
        assert!(cb.has_resolve_function());
    });
}

/// `bv_to_bytes` is the fallback encoder for the byte-level store path.
#[test]
fn bv_to_bytes_is_little_endian_and_width_rounded() {
    assert_eq!(bv_to_bytes(&RustBV::concrete(0x0102, 16)), vec![0x02, 0x01]);
    assert_eq!(bv_to_bytes(&RustBV::concrete(0x41, 8)), vec![0x41]);
    // Width rounds up to whole bytes.
    assert_eq!(bv_to_bytes(&RustBV::concrete(1, 1)), vec![0x01]);
}

/// The shared decoder behind memory load / load-batch / register get / dirty
/// call (angr-9ke6b.24). All three tuple shapes are pinned in one place so a
/// future fix to the AST-optionality rule can't land on only some callers.
#[test]
fn extract_data_tuple_handles_all_three_tuple_shapes() {
    Python::initialize();
    Python::attach(|py| {
        let globals = defs(
            py,
            c"shapes = [(b'ab', False), (b'cd', True, None), (b'ef', True, 'sentinel-ast')]
",
        );
        let shapes = globals
            .get_item("shapes")
            .unwrap()
            .unwrap()
            .cast_into::<PyList>()
            .unwrap();
        let decode = |i: usize| {
            let tuple = shapes
                .get_item(i)
                .unwrap()
                .cast_into::<pyo3::types::PyTuple>()
                .unwrap();
            extract_data_tuple(&tuple).expect("decode")
        };

        let (data, is_symbolic, ast) = decode(0);
        assert_eq!(data, b"ab");
        assert!(!is_symbolic);
        assert!(ast.is_none(), "2-tuple means no AST");

        let (data, is_symbolic, ast) = decode(1);
        assert_eq!(data, b"cd");
        assert!(is_symbolic);
        assert!(ast.is_none(), "explicit None third element means no AST");

        let (data, is_symbolic, ast) = decode(2);
        assert_eq!(data, b"ef");
        assert!(is_symbolic);
        assert_eq!(
            ast.expect("non-None third element survives")
                .extract::<String>(py)
                .unwrap(),
            "sentinel-ast"
        );
    });
}
