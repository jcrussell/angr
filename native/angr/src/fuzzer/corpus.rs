use std::cell::RefCell;

use libafl::{
    Error,
    corpus::{Corpus, CorpusId, InMemoryCorpus, OnDiskCorpus, Testcase},
    inputs::BytesInput,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::{Deserialize, Serialize};

use crate::errors::MapPyErr;
use crate::fuzzer::delegate::delegate_two_variant;

/// Generate the postcard-backed `__getstate__` / `__setstate__` pickle pair.
///
/// `PyInMemoryCorpus` and `PyOnDiskCorpus` both pickle by round-tripping their
/// single `inner` field through postcard, so hand-written the two copies are
/// byte-identical and a change to the encoding (a version byte, a different
/// error type) is easy to land in one and forget in the other (angr-12jjk.27).
///
/// The pair is emitted as its own `#[pymethods]` block — a `macro_rules!`
/// invocation *inside* an existing block would be invisible to pyo3, which
/// parses the impl before declarative macros expand. That relies on the
/// `multiple-pymethods` feature (see `Cargo.toml`).
macro_rules! postcard_pickle_methods {
    ($ty:ty, $field:ident) => {
        #[pymethods]
        impl $ty {
            fn __getstate__(&self) -> PyResult<Vec<u8>> {
                postcard::to_stdvec(&self.$field).py_runtime_err()
            }

            fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
                self.$field = postcard::from_bytes(&state).py_runtime_err()?;
                Ok(())
            }
        }
    };
}

// A Send+Sync wrapper of InMemoryCorpus for use in PyInMemoryCorpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedCorpus<I> {
    data: Vec<u8>,
    _phantom: std::marker::PhantomData<I>,
}

impl<I: Serialize> TryFrom<&InMemoryCorpus<I>> for SerializedCorpus<I> {
    type Error = PyErr;

    fn try_from(value: &InMemoryCorpus<I>) -> Result<Self, Self::Error> {
        Ok(SerializedCorpus {
            data: postcard::to_stdvec(value).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to serialize corpus: {e}"
                ))
            })?,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<I: for<'de> Deserialize<'de>> TryFrom<&SerializedCorpus<I>> for InMemoryCorpus<I> {
    type Error = PyErr;

    fn try_from(value: &SerializedCorpus<I>) -> Result<Self, Self::Error> {
        let corpus: InMemoryCorpus<I> = postcard::from_bytes(&value.data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to deserialize corpus: {e}"
            ))
        })?;
        Ok(corpus)
    }
}

#[pyclass(
    module = "angr.rustylib.fuzzer",
    name = "InMemoryCorpus",
    from_py_object
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyInMemoryCorpus {
    inner: SerializedCorpus<BytesInput>,
}

impl TryFrom<&PyInMemoryCorpus> for InMemoryCorpus<BytesInput> {
    type Error = PyErr;

    fn try_from(value: &PyInMemoryCorpus) -> Result<Self, Self::Error> {
        InMemoryCorpus::<BytesInput>::try_from(&value.inner).py_runtime_err()
    }
}

impl TryFrom<&InMemoryCorpus<BytesInput>> for PyInMemoryCorpus {
    type Error = PyErr;

    fn try_from(value: &InMemoryCorpus<BytesInput>) -> Result<Self, Self::Error> {
        let serialized = SerializedCorpus::try_from(value).py_runtime_err()?;
        Ok(PyInMemoryCorpus { inner: serialized })
    }
}

#[pymethods]
impl PyInMemoryCorpus {
    #[new]
    fn py_new() -> PyResult<Self> {
        PyInMemoryCorpus::try_from(&InMemoryCorpus::default())
    }

    #[staticmethod]
    fn from_list(list: Vec<Vec<u8>>) -> PyResult<Self> {
        let mut corpus = InMemoryCorpus::default();
        for item in list {
            corpus
                .add(Testcase::new(BytesInput::from(item)))
                .py_type_err()?;
        }
        PyInMemoryCorpus::try_from(&corpus)
    }

    fn to_bytes_list(&self) -> PyResult<Vec<Vec<u8>>> {
        let deserialized = InMemoryCorpus::<BytesInput>::try_from(&self.inner)?;
        let mut result = Vec::new();
        for corpus_id in deserialized.ids() {
            if let Ok(testcase_ref) = deserialized.get(corpus_id) {
                let testcase = testcase_ref.borrow();
                if let Some(input) = testcase.input() {
                    result.push(input.as_ref().to_vec());
                }
            }
        }
        Ok(result)
    }

    fn __getitem__(&self, id: usize) -> PyResult<Vec<u8>> {
        let deserialized = InMemoryCorpus::<BytesInput>::try_from(self)?;
        let corpus_id = CorpusId::from(id);
        let testcase_ref = deserialized.get(corpus_id).py_runtime_err()?;
        let testcase = testcase_ref.borrow();
        match testcase.input().clone() {
            Some(input) => Ok(input.into_inner()),
            None => Err(PyRuntimeError::new_err("Testcase input is None")),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(InMemoryCorpus::<BytesInput>::try_from(self)?.count())
    }
}

postcard_pickle_methods!(PyInMemoryCorpus, inner);

// On DiskCorpus wrapper
#[pyclass(
    module = "angr.rustylib.fuzzer",
    name = "OnDiskCorpus",
    unsendable,
    from_py_object
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct PyOnDiskCorpus {
    pub(crate) inner: OnDiskCorpus<BytesInput>,
}

#[pymethods]
impl PyOnDiskCorpus {
    #[new]
    fn py_new(dir_path: String) -> PyResult<Self> {
        let corpus = OnDiskCorpus::new(&dir_path).py_runtime_err()?;
        Ok(PyOnDiskCorpus { inner: corpus })
    }

    fn add(&mut self, input: Vec<u8>) -> PyResult<usize> {
        let testcase = Testcase::new(BytesInput::from(input));
        let corpus_id = self.inner.add(testcase).py_runtime_err()?;
        Ok(corpus_id.into())
    }

    fn __getitem__(&self, id: usize) -> PyResult<Vec<u8>> {
        let corpus_id = CorpusId::from(id);
        let mut testcase = self.inner.get(corpus_id).py_runtime_err()?.borrow_mut();
        let input = testcase.input().clone().unwrap_or({
            testcase
                .load_input(&self.inner)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to load input for corpus id {corpus_id}: {e}"
                    ))
                })?
                .clone()
        });
        Ok(input.as_ref().clone())
    }

    fn __len__(&self) -> usize {
        self.inner.count()
    }

    fn to_bytes_list(&self) -> PyResult<Vec<Vec<u8>>> {
        let mut result = Vec::new();
        for corpus_id in self.inner.ids() {
            result.push(
                self.inner
                    .cloned_input_for_id(corpus_id)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to load input for corpus id {corpus_id}: {e}"
                        ))
                    })?
                    .as_ref()
                    .to_vec(),
            );
        }
        Ok(result)
    }
}

postcard_pickle_methods!(PyOnDiskCorpus, inner);

// Dynamic Corpus that can encapsulate InMemoryCorpus and OnDiskCorpus at runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum DynCorpus<I> {
    InMem(InMemoryCorpus<I>),
    OnDisk(OnDiskCorpus<I>),
}

// Have Dynamic Corpus implement Corpus<I> trait. Every method is a plain
// forward to the active variant, so the bodies are generated (angr-12jjk.25).
impl<I> Corpus<I> for DynCorpus<I>
where
    I: libafl::inputs::Input,
{
    delegate_two_variant! {
        DynCorpus { InMem, OnDisk }
        fn count(&self) -> usize;
        fn count_disabled(&self) -> usize;
        fn count_all(&self) -> usize;
        fn add(&mut self, testcase: Testcase<I>) -> Result<CorpusId, Error>;
        fn add_disabled(&mut self, testcase: Testcase<I>) -> Result<CorpusId, Error>;
        fn replace(&mut self, id: CorpusId, testcase: Testcase<I>) -> Result<Testcase<I>, Error>;
        fn remove(&mut self, id: CorpusId) -> Result<Testcase<I>, Error>;
        fn get(&self, id: CorpusId) -> Result<&RefCell<Testcase<I>>, Error>;
        fn get_from_all(&self, id: CorpusId) -> Result<&RefCell<Testcase<I>>, Error>;
        fn current(&self) -> &Option<CorpusId>;
        fn current_mut(&mut self) -> &mut Option<CorpusId>;
        fn next(&self, id: CorpusId) -> Option<CorpusId>;
        fn peek_free_id(&self) -> CorpusId;
        fn prev(&self, id: CorpusId) -> Option<CorpusId>;
        fn first(&self) -> Option<CorpusId>;
        fn last(&self) -> Option<CorpusId>;
        fn nth_from_all(&self, nth: usize) -> CorpusId;
        fn load_input_into(&self, testcase: &mut Testcase<I>) -> Result<(), Error>;
        fn store_input_from(&self, testcase: &Testcase<I>) -> Result<(), Error>;
    }
}

// Converts python objects into Rust enum
impl TryFrom<&PyInMemoryCorpus> for DynCorpus<BytesInput> {
    type Error = PyErr;

    fn try_from(value: &PyInMemoryCorpus) -> Result<Self, Self::Error> {
        let inner: InMemoryCorpus<BytesInput> =
            InMemoryCorpus::<BytesInput>::try_from(value).py_runtime_err()?;
        Ok(DynCorpus::InMem(inner))
    }
}

impl TryFrom<&PyOnDiskCorpus> for DynCorpus<BytesInput> {
    type Error = PyErr;

    fn try_from(value: &PyOnDiskCorpus) -> Result<Self, Self::Error> {
        Ok(DynCorpus::OnDisk(value.inner.clone()))
    }
}

// Converts Rust enum back into python object
impl DynCorpus<BytesInput> {
    pub(crate) fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self {
            DynCorpus::InMem(inner) => {
                let py_inmem = PyInMemoryCorpus::try_from(inner).py_runtime_err()?;
                let obj = Py::new(py, py_inmem)?; // Py<PyInMemoryCorpus>
                Ok(obj.into_bound(py).into_any().unbind())
            }
            DynCorpus::OnDisk(inner) => {
                let py_ondisk = PyOnDiskCorpus {
                    inner: inner.clone(),
                };
                let obj = Py::new(py, py_ondisk)?; // Py<PyOnDiskCorpus>
                Ok(obj.into_bound(py).into_any().unbind())
            }
        }
    }
}

#[cfg(test)]
#[path = "corpus_tests.rs"]
mod tests;
