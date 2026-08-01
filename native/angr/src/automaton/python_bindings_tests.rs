// Tests for python_bindings.rs — specifically ObjectMapper's identity model.
//
// The mapper buckets by Python `hash()` and disambiguates with `__eq__`, the
// same contract a Python dict/set uses. Regression coverage for angr-9ke6b.187:
// the previous `(hash, repr)`-tuple key merged `__eq__`-distinct objects that
// shared a hash and a repr.

use super::*;

/// Run `src` and return the named locals as Python objects.
fn eval_locals<'py>(py: Python<'py>, src: &str, names: &[&str]) -> Vec<Py<PyAny>> {
    let locals = pyo3::types::PyDict::new(py);
    py.run(
        std::ffi::CString::new(src).unwrap().as_c_str(),
        None,
        Some(&locals),
    )
    .unwrap();
    names
        .iter()
        .map(|n| locals.get_item(n).unwrap().unwrap().unbind())
        .collect()
}

/// Two instances of a class with a constant `__hash__`, a constant `__repr__`,
/// and identity `__eq__` — indistinguishable under the old (hash, repr) key.
const COLLIDING_BY_IDENTITY: &str = "
class Collider:
    def __hash__(self): return 7
    def __repr__(self): return '<Collider>'
    def __eq__(self, other): return self is other
a = Collider()
b = Collider()
";

/// Same hash and repr, but `__eq__` says the two instances *are* equal.
const COLLIDING_AND_EQUAL: &str = "
class Same:
    def __hash__(self): return 7
    def __repr__(self): return '<Same>'
    # `type(...)` rather than `isinstance(other, Same)`: the snippet runs with
    # only a locals dict, so the class name isn't in the method's globals.
    def __eq__(self, other): return type(other) is type(self)
a = Same()
b = Same()
";

#[test]
fn test_hash_repr_collision_keeps_states_distinct() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(py, COLLIDING_BY_IDENTITY, &["a", "b"]);
        let mut mapper = ObjectMapper::new();

        let id_a = mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .unwrap();
        let id_b = mapper
            .get_or_create_state_id(py, &PyState::new(objs[1].clone()))
            .unwrap();

        assert_ne!(
            id_a, id_b,
            "hash+repr collision must not merge __eq__-distinct states"
        );
        assert_eq!(mapper.id_to_state.len(), 2);
        // Both landed in the one hash bucket, so the __eq__ scan is what
        // separated them.
        assert_eq!(mapper.state_buckets.len(), 1);
        assert_eq!(mapper.state_buckets[&7], vec![id_a, id_b]);
    });
}

#[test]
fn test_hash_repr_collision_keeps_symbols_distinct() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(py, COLLIDING_BY_IDENTITY, &["a", "b"]);
        let mut mapper = ObjectMapper::new();

        let id_a = mapper
            .get_or_create_symbol_id(py, &PySymbol::new(objs[0].clone()))
            .unwrap();
        let id_b = mapper
            .get_or_create_symbol_id(py, &PySymbol::new(objs[1].clone()))
            .unwrap();

        assert_ne!(id_a, id_b);
        assert_eq!(mapper.id_to_symbol.len(), 2);
        // Each ID still round-trips back to its own object, not the other's.
        assert!(
            mapper
                .get_symbol_by_id(id_a)
                .unwrap()
                .bind(py)
                .is(objs[0].bind(py))
        );
        assert!(
            mapper
                .get_symbol_by_id(id_b)
                .unwrap()
                .bind(py)
                .is(objs[1].bind(py))
        );
    });
}

#[test]
fn test_eq_equal_objects_share_one_id() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(py, COLLIDING_AND_EQUAL, &["a", "b"]);
        let mut mapper = ObjectMapper::new();

        let id_a = mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .unwrap();
        let id_b = mapper
            .get_or_create_state_id(py, &PyState::new(objs[1].clone()))
            .unwrap();

        assert_eq!(id_a, id_b, "__eq__-equal objects must share a state ID");
        assert_eq!(mapper.id_to_state.len(), 1);
    });
}

#[test]
fn test_repeated_lookup_is_stable() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(py, "a = 'alpha'\nb = 'beta'", &["a", "b"]);
        let mut mapper = ObjectMapper::new();

        let first = mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .unwrap();
        let other = mapper
            .get_or_create_state_id(py, &PyState::new(objs[1].clone()))
            .unwrap();
        let again = mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .unwrap();

        assert_ne!(first, other);
        assert_eq!(first, again);
        assert_eq!(mapper.id_to_state.len(), 2);
    });
}

#[test]
fn test_raising_eq_propagates_instead_of_forking_a_state() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(
            py,
            "
class Boom:
    def __hash__(self): return 3
    def __eq__(self, other): raise RuntimeError('nope')
a = Boom()
b = Boom()
",
            &["a", "b"],
        );
        let mut mapper = ObjectMapper::new();

        // First insert never compares (empty bucket), so it succeeds.
        mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .unwrap();
        let err = mapper
            .get_or_create_state_id(py, &PyState::new(objs[1].clone()))
            .expect_err("a raising __eq__ must surface, not be swallowed");
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        assert_eq!(mapper.id_to_state.len(), 1);
    });
}

#[test]
fn test_unhashable_object_is_rejected() {
    Python::initialize();
    Python::attach(|py| {
        let objs = eval_locals(py, "a = ['unhashable']", &["a"]);
        let mut mapper = ObjectMapper::new();

        let err = mapper
            .get_or_create_state_id(py, &PyState::new(objs[0].clone()))
            .expect_err("hash() failure must propagate");
        assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        assert!(mapper.id_to_state.is_empty());
    });
}
