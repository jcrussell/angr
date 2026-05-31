// Conditional compilation for fuzzer module (requires optional deps)
#[cfg(feature = "fuzzer")]
pub mod fuzzer;
#[cfg(feature = "fuzzer")]
pub mod icicle;

#[cfg(feature = "automaton")]
pub mod automaton;
pub mod segmentlist;

// VEX Engine modules (requires vex-engine feature, enabled by default)
#[cfg(feature = "vex-engine")]
pub mod arch;
#[cfg(feature = "vex-engine")]
pub mod callbacks;
#[cfg(feature = "vex-engine")]
pub mod claripy_bridge;
#[cfg(feature = "vex-engine")]
pub mod concretize;
#[cfg(feature = "vex-engine")]
pub mod engine;
#[cfg(feature = "vex-engine")]
pub mod errors;
#[cfg(feature = "vex-engine")]
pub mod exploration;
#[cfg(feature = "vex-engine")]
pub mod interpreter_cb;
#[cfg(feature = "vex-engine")]
pub mod memory;
#[cfg(feature = "vex-engine")]
pub mod procedures;
#[cfg(feature = "vex-engine")]
pub mod solver;
#[cfg(feature = "vex-engine")]
pub mod stash;
#[cfg(feature = "vex-engine")]
pub mod state;
pub mod symbolic;
#[cfg(feature = "vex-engine")]
pub mod syscalls;
#[cfg(feature = "vex-engine")]
pub mod vex;

use pyo3::prelude::*;

fn import_submodule<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyModule>,
    package: &str,
    name: &str,
    import_func: impl FnOnce(&Bound<'py, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let submodule = PyModule::new(py, name)?;
    import_func(&submodule)?;

    // Add the submodule to sys.modules
    let sys_modules = PyModule::import(py, "sys")?.getattr("modules")?;
    sys_modules.set_item(format!("{package}.{name}"), submodule.clone())?;

    m.add_submodule(&submodule)?;
    Ok(())
}

#[pymodule]
fn rustylib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Fuzzer modules (optional)
    #[cfg(feature = "fuzzer")]
    {
        import_submodule(m.py(), m, "angr.rustylib", "fuzzer", fuzzer::fuzzer)?;
        import_submodule(m.py(), m, "angr.rustylib", "icicle", icicle::icicle)?;
    }

    // Segmentlist (always available)
    import_submodule(
        m.py(),
        m,
        "angr.rustylib",
        "segmentlist",
        segmentlist::segmentlist,
    )?;
    #[cfg(feature = "automaton")]
    import_submodule(
        m.py(),
        m,
        "angr.rustylib",
        "automaton",
        automaton::automaton,
    )?;
    m.add_class::<segmentlist::Segment>()?;
    m.add_class::<segmentlist::SegmentList>()?;

    // VEX Engine module (enabled by default)
    #[cfg(feature = "vex-engine")]
    import_submodule(m.py(), m, "angr.rustylib", "vex_engine", engine::vex_engine)?;

    Ok(())
}
