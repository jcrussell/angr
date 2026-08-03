//! PyO3 bindings for the automaton module.
//!
//! Provides a pyautomaton-compatible API for Python.

use crate::automaton::dfa::DFA;
use crate::automaton::epsilon_nfa::EpsilonNFA as RustEpsilonNFA;
use crate::automaton::state::StateId;
use crate::automaton::subset_construction::subset_construction;
use crate::automaton::symbol::{EPSILON, SymbolId};
use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySet;

/// A State wrapper that holds any Python object.
#[pyclass(
    name = "State",
    module = "angr.rustylib.automaton",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyState {
    /// The underlying Python value
    value: Py<PyAny>,
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl PyState {
    #[new]
    fn new(value: Py<PyAny>) -> Self {
        Self { value }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let repr = self.value.bind(py).repr()?;
        Ok(format!("State({repr})"))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<bool> {
        if let Ok(other_state) = other.extract::<PyRef<PyState>>() {
            self.value.bind(py).eq(other_state.value.bind(py))
        } else {
            Ok(false)
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.value.bind(py).hash()
    }

    #[getter]
    fn value(&self) -> Py<PyAny> {
        self.value.clone()
    }
}

/// A Symbol wrapper that holds any Python object.
#[pyclass(
    name = "Symbol",
    module = "angr.rustylib.automaton",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySymbol {
    /// The underlying Python value
    value: Py<PyAny>,
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl PySymbol {
    #[new]
    fn new(value: Py<PyAny>) -> Self {
        Self { value }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let repr = self.value.bind(py).repr()?;
        Ok(format!("Symbol({repr})"))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<bool> {
        if let Ok(other_sym) = other.extract::<PyRef<PySymbol>>() {
            self.value.bind(py).eq(other_sym.value.bind(py))
        } else {
            Ok(false)
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.value.bind(py).hash()
    }

    #[getter]
    fn value(&self) -> Py<PyAny> {
        self.value.clone()
    }
}

/// Marker for epsilon transitions.
#[pyclass(
    name = "Epsilon",
    module = "angr.rustylib.automaton",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyEpsilon;

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl PyEpsilon {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "Epsilon()"
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.is_instance_of::<PyEpsilon>()
    }

    fn __hash__(&self) -> isize {
        // Consistent hash for all Epsilon instances
        EPSILON_HASH as isize
    }
}

const EPSILON_HASH: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// A bidirectional interner assigning dense `u32` IDs to Python objects.
///
/// Identity follows Python's own dict/set contract: bucket by `hash()`, then
/// disambiguate within the bucket with `__eq__`. An earlier version keyed on
/// the tuple `(hash, repr)` as a *proxy* for equality with no `__eq__`
/// fallback, so two `__eq__`-distinct objects that happened to share a hash
/// and a `repr` (easy with a generic/truncated `__repr__`, or a custom
/// `__hash__`/`__eq__` pair that disagrees with `repr`) were silently merged
/// into one automaton state/symbol. See `angr-9ke6b.187`.
///
/// `StateId` and `SymbolId` are both `u32`, so one interner serves both; the
/// only difference is that the symbol side reserves `EPSILON` as a sentinel.
#[derive(Clone)]
struct IdInterner {
    /// Maps a Python object hash to every ID sharing that hash
    buckets: IndexMap<isize, Vec<u32>>,
    /// Maps IDs back to Python objects
    interned: Vec<Py<PyAny>>,
    /// ID reserved as a sentinel and never handed out (`EPSILON` for symbols);
    /// `None` when the whole `u32` range is assignable.
    reserved: Option<u32>,
    /// Plural noun for the "Too many …" exhaustion error
    kind: &'static str,
}

/// Look up `value` in a hash bucket, comparing candidates with Python `__eq__`.
///
/// `interned` maps an ID back to its Python object; `bucket` holds the IDs that
/// already hashed to the same value. Returns the matching ID, or `None` when
/// this is a genuinely new object. A raising `__eq__` propagates rather than
/// being swallowed into a false "distinct" verdict.
///
/// Free function rather than an `IdInterner` method because the caller holds a
/// mutable borrow of `buckets` across the scan and needs `interned` separately.
fn find_in_bucket(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    bucket: &[u32],
    interned: &[Py<PyAny>],
) -> PyResult<Option<u32>> {
    for &id in bucket {
        if value.eq(interned[id as usize].bind(py))? {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

impl IdInterner {
    fn new(reserved: Option<u32>, kind: &'static str) -> Self {
        Self {
            buckets: IndexMap::new(),
            interned: Vec::new(),
            reserved,
            kind,
        }
    }

    /// Return `value`'s existing ID, or assign it the next free one.
    fn get_or_create(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<u32> {
        let hash = value.hash()?;
        let bucket = self.buckets.entry(hash).or_default();

        if let Some(id) = find_in_bucket(py, value, bucket, &self.interned)? {
            return Ok(id);
        }
        let id = self.interned.len() as u32;
        if Some(id) == self.reserved {
            return Err(PyValueError::new_err(format!("Too many {}", self.kind)));
        }
        bucket.push(id);
        self.interned.push(value.clone().unbind());
        Ok(id)
    }

    /// Map an ID back to its Python object.
    ///
    /// The two "no object" outcomes stay distinguishable: the reserved sentinel
    /// (`EPSILON`) is never handed out at all, whereas an out-of-range ID means
    /// the caller mixed in an ID this interner never assigned. Both are bugs
    /// rather than expected control flow, and they have different causes, so
    /// callers get a three-way answer instead of a bare `Option` that collapses
    /// them (angr-9ke6b.190).
    fn lookup(&self, id: u32) -> IdLookup<'_> {
        if Some(id) == self.reserved {
            IdLookup::Reserved
        } else {
            match self.interned.get(id as usize) {
                Some(obj) => IdLookup::Found(obj),
                None => IdLookup::Unassigned,
            }
        }
    }
}

/// Outcome of [`IdInterner::lookup`].
enum IdLookup<'a> {
    /// The ID maps to this interned Python object.
    Found(&'a Py<PyAny>),
    /// The ID is the reserved sentinel (`EPSILON` for symbols), which is never
    /// handed out and therefore has no Python object behind it.
    Reserved,
    /// The ID was never assigned by this interner.
    Unassigned,
}

/// Helper struct for tracking Python object to ID mappings.
#[derive(Clone)]
struct ObjectMapper {
    /// Interner for automaton states
    states: IdInterner,
    /// Interner for transition symbols; reserves `EPSILON`
    symbols: IdInterner,
}

impl ObjectMapper {
    fn new() -> Self {
        Self {
            states: IdInterner::new(None, "states"),
            symbols: IdInterner::new(Some(EPSILON), "symbols"),
        }
    }

    fn get_or_create_state_id(&mut self, py: Python<'_>, state: &PyState) -> PyResult<StateId> {
        self.states.get_or_create(py, state.value.bind(py))
    }

    fn get_or_create_symbol_id(&mut self, py: Python<'_>, symbol: &PySymbol) -> PyResult<SymbolId> {
        self.symbols.get_or_create(py, symbol.value.bind(py))
    }

    fn lookup_symbol(&self, id: SymbolId) -> IdLookup<'_> {
        self.symbols.lookup(id)
    }
}

/// An Epsilon Non-deterministic Finite Automaton.
#[pyclass(name = "EpsilonNFA", module = "angr.rustylib.automaton")]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct PyEpsilonNFA {
    /// The underlying Rust NFA
    nfa: RustEpsilonNFA,
    /// Object mapper for Python <-> ID conversions
    mapper: ObjectMapper,
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl PyEpsilonNFA {
    #[new]
    fn new() -> Self {
        Self {
            nfa: RustEpsilonNFA::new(),
            mapper: ObjectMapper::new(),
        }
    }

    /// Add a transition.
    /// The symbol can be a Symbol or Epsilon.
    #[pyo3(signature = (source, symbol, destination))]
    fn add_transition(
        &mut self,
        py: Python<'_>,
        source: &PyState,
        symbol: &Bound<'_, PyAny>,
        destination: &PyState,
    ) -> PyResult<()> {
        let src_id = self.mapper.get_or_create_state_id(py, source)?;
        let dst_id = self.mapper.get_or_create_state_id(py, destination)?;

        if symbol.is_instance_of::<PyEpsilon>() {
            self.nfa.add_epsilon_transition(src_id, dst_id);
        } else if let Ok(sym) = symbol.extract::<PySymbol>() {
            let sym_id = self.mapper.get_or_create_symbol_id(py, &sym)?;
            self.nfa.add_transition(src_id, sym_id, dst_id);
        } else {
            return Err(PyValueError::new_err(
                "symbol must be a Symbol or Epsilon instance",
            ));
        }

        Ok(())
    }

    /// Add a start state.
    fn add_start_state(&mut self, py: Python<'_>, state: &PyState) -> PyResult<()> {
        let state_id = self.mapper.get_or_create_state_id(py, state)?;
        self.nfa.add_start_state(state_id);
        Ok(())
    }

    /// Add a final (accepting) state.
    fn add_final_state(&mut self, py: Python<'_>, state: &PyState) -> PyResult<()> {
        let state_id = self.mapper.get_or_create_state_id(py, state)?;
        self.nfa.add_final_state(state_id);
        Ok(())
    }

    /// Check if the NFA's language is empty.
    fn is_empty(&self) -> bool {
        self.nfa.is_empty()
    }

    /// Minimize the NFA by converting to DFA and minimizing.
    /// Returns a DeterministicFiniteAutomaton.
    fn minimize(&mut self) -> PyResult<PyDFA> {
        // Compute epsilon closures for efficiency
        self.nfa.compute_epsilon_closures();

        // Convert to DFA via subset construction. The epsilon-marker error is
        // unreachable while the alphabet excludes epsilon, but mapping it here
        // keeps a future invariant break a clean ValueError rather than a
        // PanicException (angr-9ke6b.191).
        let dfa =
            subset_construction(&self.nfa).map_err(|err| PyValueError::new_err(err.to_string()))?;

        // Minimize the DFA
        let minimized = dfa.minimize();

        Ok(PyDFA {
            dfa: minimized,
            mapper: self.mapper.clone(),
        })
    }
}

/// A Deterministic Finite Automaton.
#[pyclass(
    name = "DeterministicFiniteAutomaton",
    module = "angr.rustylib.automaton"
)]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct PyDFA {
    /// The underlying Rust DFA
    dfa: DFA,
    /// Object mapper for Python <-> ID conversions
    mapper: ObjectMapper,
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl PyDFA {
    /// Get the start state as an integer index.
    #[getter]
    fn start_state(&self) -> Option<u32> {
        self.dfa.start_state()
    }

    /// Get the final states as a set of integer indices.
    #[getter]
    fn final_states(&self, py: Python<'_>) -> PyResult<Py<PySet>> {
        let set = PySet::empty(py)?;
        for state in self.dfa.final_states().iter() {
            set.add(state)?;
        }
        Ok(set.unbind())
    }

    /// Check if the DFA's language is empty.
    fn is_empty(&self) -> bool {
        self.dfa.is_empty()
    }

    /// Convert to a NetworkX MultiDiGraph.
    fn to_networkx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Import networkx
        let nx = py.import("networkx")?;
        let graph = nx.call_method0("MultiDiGraph")?;

        // Add nodes
        for state in 0..self.dfa.num_states() {
            graph.call_method1("add_node", (state,))?;
        }

        // Add edges with labels
        for (src, sym, dst) in self.dfa.transitions() {
            // Get the original Python symbol for the label
            let label: Bound<'py, PyAny> = match self.mapper.lookup_symbol(sym) {
                IdLookup::Found(py_symbol) => py_symbol.bind(py).clone(),
                // SILENT(cat-b): both remaining arms fall back to the raw
                // numeric ID as the edge label. Neither is reachable for a DFA
                // built by `subset_construction` — it only iterates
                // `nfa.alphabet()`, which excludes EPSILON, and every symbol in
                // it came from `get_or_create_symbol_id` — so they are logged
                // separately rather than collapsed, and the loss is cosmetic
                // (a debug/visualisation label), not a wrong automaton.
                IdLookup::Reserved => {
                    log::warn!(
                        "to_networkx: DFA transition {src}->{dst} carries the EPSILON marker as a \
                         symbol; labelling it with the raw id"
                    );
                    sym.into_pyobject(py)?.into_any()
                }
                IdLookup::Unassigned => {
                    log::warn!(
                        "to_networkx: DFA transition {src}->{dst} uses symbol id {sym}, which was \
                         never interned; labelling it with the raw id"
                    );
                    sym.into_pyobject(py)?.into_any()
                }
            };

            // Create kwargs dict with label
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("label", label)?;

            graph.call_method("add_edge", (src, dst), Some(&kwargs))?;
        }

        Ok(graph)
    }

    /// Minimize the DFA (returns a new minimized DFA).
    fn minimize(&self) -> PyDFA {
        PyDFA {
            dfa: self.dfa.minimize(),
            mapper: self.mapper.clone(),
        }
    }
}

/// Register the automaton submodule.
pub fn automaton(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyState>()?;
    m.add_class::<PySymbol>()?;
    m.add_class::<PyEpsilon>()?;
    m.add_class::<PyEpsilonNFA>()?;
    m.add_class::<PyDFA>()?;
    Ok(())
}

#[cfg(test)]
#[path = "python_bindings_tests.rs"]
mod tests;
