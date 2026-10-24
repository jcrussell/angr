//! The `ExecutionConfig` Python class: the PyO3 wrapper over the
//! interpreter's [`ExecutionConfig`] (angr-mx00p.75).
//!
//! No engine entry point takes one from Python today; the class is kept so
//! the `angr.rustylib.vex_engine` surface is unchanged.

use pyo3::prelude::*;

use crate::interpreter::ExecutionConfig;

/// Configuration for the execution loop with deferred forks.
///
/// See [`ExecutionConfig`] for what each knob does.
#[pyclass(name = "ExecutionConfig", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyExecutionConfig {
    inner: ExecutionConfig,
}

#[pymethods]
impl PyExecutionConfig {
    /// Create a new execution config.
    ///
    /// Starts from [`ExecutionConfig::default`] (the single source of truth for
    /// every field, including the "eager prefetch off / deferred forks on" hot
    /// path) and overrides only the two Python-tunable arguments. The argument
    /// defaults mirror the struct default so `ExecutionConfig()` == `default()`.
    #[new]
    #[pyo3(signature = (max_deferred_forks=500, use_deferred_forks=true))]
    pub(crate) fn py_new(max_deferred_forks: u32, use_deferred_forks: bool) -> Self {
        PyExecutionConfig {
            inner: ExecutionConfig {
                max_deferred_forks,
                use_deferred_forks,
                ..ExecutionConfig::default()
            },
        }
    }

    /// See [`ExecutionConfig::max_deferred_forks`].
    #[getter]
    fn max_deferred_forks(&self) -> u32 {
        self.inner.max_deferred_forks
    }

    #[setter]
    fn set_max_deferred_forks(&mut self, value: u32) {
        self.inner.max_deferred_forks = value;
    }

    /// See [`ExecutionConfig::use_deferred_forks`].
    #[getter]
    fn use_deferred_forks(&self) -> bool {
        self.inner.use_deferred_forks
    }

    #[setter]
    fn set_use_deferred_forks(&mut self, value: bool) {
        self.inner.use_deferred_forks = value;
    }

    /// See [`ExecutionConfig::enable_eager_prefetch`].
    #[getter]
    fn enable_eager_prefetch(&self) -> bool {
        self.inner.enable_eager_prefetch
    }

    #[setter]
    fn set_enable_eager_prefetch(&mut self, value: bool) {
        self.inner.enable_eager_prefetch = value;
    }

    /// See [`ExecutionConfig::max_prefetch_batch`].
    #[getter]
    fn max_prefetch_batch(&self) -> usize {
        self.inner.max_prefetch_batch
    }

    #[setter]
    fn set_max_prefetch_batch(&mut self, value: usize) {
        self.inner.max_prefetch_batch = value;
    }

    /// See [`ExecutionConfig::max_symbolic_ip_targets`].
    #[getter]
    fn max_symbolic_ip_targets(&self) -> usize {
        self.inner.max_symbolic_ip_targets
    }

    #[setter]
    fn set_max_symbolic_ip_targets(&mut self, value: usize) {
        self.inner.max_symbolic_ip_targets = value;
    }

    pub(crate) fn __repr__(&self) -> String {
        let c = &self.inner;
        format!(
            "ExecutionConfig(max_deferred_forks={}, use_deferred_forks={}, eager_prefetch={}, max_prefetch_batch={})",
            c.max_deferred_forks,
            c.use_deferred_forks,
            c.enable_eager_prefetch,
            c.max_prefetch_batch
        )
    }
}

test_submod!("config_tests.rs" => config_tests);
