// Tests for python_bindings.rs — the Python/Rust identity-bridging layer.
//
// Two groups: `ObjectMapper`'s identity model, and the wrapper/automaton
// pyclasses (`PyState`/`PySymbol`/`PyEpsilon`, `PyEpsilonNFA`, `PyDFA`).
//
// The mapper buckets by Python `hash()` and disambiguates with `__eq__`, the
// same contract a Python dict/set uses. Regression coverage for angr-9ke6b.187:
// the previous `(hash, repr)`-tuple key merged `__eq__`-distinct objects that
// shared a hash and a repr.

use super::*;

/// Unwrap an [`IdLookup`] that is expected to have found an object.
///
/// Test-local rather than a method on `IdLookup`: production code must keep
/// handling `Reserved` and `Unassigned` separately (angr-9ke6b.190).
fn expect_found<'a>(lookup: IdLookup<'a>) -> &'a Py<PyAny> {
    match lookup {
        IdLookup::Found(obj) => obj,
        IdLookup::Reserved => panic!("expected an interned object, got the reserved sentinel"),
        IdLookup::Unassigned => panic!("expected an interned object, got an unassigned id"),
    }
}

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
        assert_eq!(mapper.states.interned.len(), 2);
        // Both landed in the one hash bucket, so the __eq__ scan is what
        // separated them.
        assert_eq!(mapper.states.buckets.len(), 1);
        assert_eq!(mapper.states.buckets[&7], vec![id_a, id_b]);
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
        assert_eq!(mapper.symbols.interned.len(), 2);
        // Each ID still round-trips back to its own object, not the other's.
        assert!(
            expect_found(mapper.lookup_symbol(id_a))
                .bind(py)
                .is(objs[0].bind(py))
        );
        assert!(
            expect_found(mapper.lookup_symbol(id_b))
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
        assert_eq!(mapper.states.interned.len(), 1);
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
        assert_eq!(mapper.states.interned.len(), 2);
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
        assert_eq!(mapper.states.interned.len(), 1);
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
        assert!(mapper.states.interned.is_empty());
    });
}

// ---------------------------------------------------------------------------
// Wrapper pyclasses: PyState / PySymbol / PyEpsilon
// ---------------------------------------------------------------------------

/// A `PyState` wrapping the Python string `name`.
fn state(py: Python<'_>, name: &str) -> PyState {
    PyState::new(pystr(py, name))
}

/// The Python string `name` as an owned object.
fn pystr(py: Python<'_>, name: &str) -> Py<PyAny> {
    pyo3::types::PyString::new(py, name).into_any().unbind()
}

/// A `PySymbol` wrapping `name`, as a `Bound` ready for `add_transition`.
fn symbol<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyAny> {
    Bound::new(py, PySymbol::new(pystr(py, name)))
        .unwrap()
        .into_any()
}

/// A `PyEpsilon` as a `Bound`, the other legal `add_transition` symbol.
fn epsilon(py: Python<'_>) -> Bound<'_, PyAny> {
    Bound::new(py, PyEpsilon::new()).unwrap().into_any()
}

#[test]
fn test_state_eq_hash_repr_delegate_to_wrapped_value() {
    Python::initialize();
    Python::attach(|py| {
        let a = state(py, "s");
        let same = Bound::new(py, state(py, "s")).unwrap().into_any();
        let other = Bound::new(py, state(py, "t")).unwrap().into_any();

        assert!(a.__eq__(&same, py).unwrap());
        assert!(!a.__eq__(&other, py).unwrap());
        assert_eq!(
            a.__hash__(py).unwrap(),
            pystr(py, "s").bind(py).hash().unwrap()
        );
        assert_eq!(a.__repr__(py).unwrap(), "State('s')");
        assert!(a.value().bind(py).eq(pystr(py, "s").bind(py)).unwrap());
    });
}

#[test]
fn test_state_and_symbol_wrappers_never_compare_equal() {
    Python::initialize();
    Python::attach(|py| {
        // Same wrapped value, different wrapper type: the automaton keeps
        // states and symbols in separate ID spaces, so the wrappers must not
        // claim equality just because their payloads match.
        let st = state(py, "x");
        let sym = PySymbol::new(pystr(py, "x"));
        let st_bound = Bound::new(py, state(py, "x")).unwrap().into_any();

        assert!(!st.__eq__(&symbol(py, "x"), py).unwrap());
        assert!(!sym.__eq__(&st_bound, py).unwrap());
        // …and a bare payload is not a wrapper either.
        assert!(!st.__eq__(pystr(py, "x").bind(py), py).unwrap());
    });
}

#[test]
fn test_symbol_eq_hash_repr_delegate_to_wrapped_value() {
    Python::initialize();
    Python::attach(|py| {
        let a = PySymbol::new(pystr(py, "s"));
        let same = symbol(py, "s");
        let other = symbol(py, "t");

        assert!(a.__eq__(&same, py).unwrap());
        assert!(!a.__eq__(&other, py).unwrap());
        assert_eq!(
            a.__hash__(py).unwrap(),
            pystr(py, "s").bind(py).hash().unwrap()
        );
        assert_eq!(a.__repr__(py).unwrap(), "Symbol('s')");
    });
}

#[test]
fn test_all_epsilons_are_equal_and_share_one_hash() {
    Python::initialize();
    Python::attach(|py| {
        let eps = PyEpsilon::new();

        assert!(eps.__eq__(&epsilon(py)));
        assert!(!eps.__eq__(&symbol(py, "x")));
        assert_eq!(eps.__hash__(), PyEpsilon::new().__hash__());
        assert_eq!(eps.__repr__(), "Epsilon()");
    });
}

// ---------------------------------------------------------------------------
// PyEpsilonNFA / PyDFA
// ---------------------------------------------------------------------------

/// `a -ε-> b -x-> c`, start `a`, final `c`: accepts exactly "x".
fn accepts_x(py: Python<'_>) -> PyEpsilonNFA {
    let mut nfa = PyEpsilonNFA::new();
    nfa.add_start_state(py, &state(py, "a")).unwrap();
    nfa.add_transition(py, &state(py, "a"), &epsilon(py), &state(py, "b"))
        .unwrap();
    nfa.add_transition(py, &state(py, "b"), &symbol(py, "x"), &state(py, "c"))
        .unwrap();
    nfa.add_final_state(py, &state(py, "c")).unwrap();
    nfa
}

#[test]
fn test_nfa_rejects_a_symbol_that_is_neither_symbol_nor_epsilon() {
    Python::initialize();
    Python::attach(|py| {
        let mut nfa = PyEpsilonNFA::new();
        let bare = pystr(py, "x");

        let err = nfa
            .add_transition(py, &state(py, "a"), bare.bind(py), &state(py, "b"))
            .expect_err("a bare payload must not pass as a Symbol");
        assert!(err.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn test_nfa_interns_equal_states_across_calls() {
    Python::initialize();
    Python::attach(|py| {
        // Every helper call builds a *fresh* PyState wrapping an equal string,
        // so a working interner must collapse them: 3 distinct states, 1 symbol.
        let nfa = accepts_x(py);

        assert_eq!(nfa.mapper.states.interned.len(), 3);
        assert_eq!(nfa.mapper.symbols.interned.len(), 1);
    });
}

#[test]
fn test_nfa_minimizes_to_a_dfa_accepting_one_symbol() {
    Python::initialize();
    Python::attach(|py| {
        let mut nfa = accepts_x(py);
        assert!(!nfa.is_empty());

        let dfa = nfa.minimize().unwrap();

        assert!(!dfa.is_empty());
        assert!(dfa.start_state().is_some());
        let finals = dfa.final_states(py).unwrap();
        assert_eq!(finals.bind(py).len(), 1);
        // The ε-transition collapsed `a` and `b`, so the minimal DFA is
        // start --x--> accept.
        assert_eq!(dfa.dfa.num_states(), 2);
    });
}

#[test]
fn test_nfa_without_a_final_state_minimizes_to_an_empty_dfa() {
    Python::initialize();
    Python::attach(|py| {
        let mut nfa = PyEpsilonNFA::new();
        nfa.add_start_state(py, &state(py, "a")).unwrap();
        nfa.add_transition(py, &state(py, "a"), &symbol(py, "x"), &state(py, "b"))
            .unwrap();
        assert!(nfa.is_empty());

        let dfa = nfa.minimize().unwrap();

        assert!(dfa.is_empty());
        assert_eq!(dfa.final_states(py).unwrap().bind(py).len(), 0);
    });
}

#[test]
fn test_dfa_minimize_is_idempotent() {
    Python::initialize();
    Python::attach(|py| {
        let dfa = accepts_x(py).minimize().unwrap();
        let again = dfa.minimize();

        assert_eq!(again.dfa.num_states(), dfa.dfa.num_states());
        assert_eq!(again.start_state(), dfa.start_state());
        assert!(!again.is_empty());
        // `minimize` clones the mapper, so the new DFA can still name its
        // symbols — losing it would silently degrade `to_networkx` labels.
        assert_eq!(again.mapper.symbols.interned.len(), 1);
    });
}

/// A `networkx` stand-in recording exactly what `to_networkx` asks for.
///
/// Using a stub rather than the real package keeps the test hermetic (the
/// cargo-test interpreter need not have networkx installed) and lets it assert
/// on the raw `add_node` / `add_edge` call sequence.
const NETWORKX_STUB: &str = "
import sys, types
_mod = types.ModuleType('networkx')
class MultiDiGraph:
    def __init__(self):
        self.recorded_nodes = []
        self.recorded_edges = []
    def add_node(self, node):
        self.recorded_nodes.append(node)
    def add_edge(self, src, dst, **kwargs):
        self.recorded_edges.append((src, dst, kwargs['label']))
_mod.MultiDiGraph = MultiDiGraph
sys.modules['networkx'] = _mod
";

#[test]
fn test_to_networkx_emits_every_node_and_labels_edges_with_python_symbols() {
    Python::initialize();
    Python::attach(|py| {
        let dfa = accepts_x(py).minimize().unwrap();
        py.run(
            std::ffi::CString::new(NETWORKX_STUB).unwrap().as_c_str(),
            None,
            None,
        )
        .unwrap();

        let graph = dfa.to_networkx(py).unwrap();

        let nodes: Vec<u32> = graph.getattr("recorded_nodes").unwrap().extract().unwrap();
        let edges: Vec<(u32, u32, String)> =
            graph.getattr("recorded_edges").unwrap().extract().unwrap();

        // Nodes are the dense id range, and every edge endpoint is among them —
        // the node-list/edge-list consistency `DFA::add_transition`'s state-id
        // auto-grow exists to preserve.
        assert_eq!(nodes, (0..dfa.dfa.num_states()).collect::<Vec<_>>());
        assert!(!edges.is_empty());
        for (src, dst, label) in &edges {
            assert!(nodes.contains(src) && nodes.contains(dst));
            // The label is the original Python payload, not the internal id.
            assert_eq!(label, "x");
        }
    });
}

/// angr-9ke6b.190: the reserved EPSILON marker and a never-assigned id are two
/// different bugs, and `lookup_symbol` must not collapse them into one "None".
#[test]
fn test_lookup_symbol_distinguishes_reserved_from_unassigned() {
    Python::initialize();
    Python::attach(|py| {
        let mut mapper = ObjectMapper::new();
        let obj = eval_locals(py, "a = object()", &["a"]).remove(0);
        let id = mapper
            .get_or_create_symbol_id(py, &PySymbol::new(obj.clone()))
            .unwrap();

        assert!(
            expect_found(mapper.lookup_symbol(id))
                .bind(py)
                .is(obj.bind(py))
        );
        assert!(matches!(mapper.lookup_symbol(EPSILON), IdLookup::Reserved));
        assert!(matches!(mapper.lookup_symbol(id + 1), IdLookup::Unassigned));
    });
}
