"""Shared fixtures and helpers for the Rust-engine test modules.

Consolidates boilerplate that was copy-pasted across the six
``tests/engines/test_*rust*.py`` files (angr-7gdp):

* the ``RUST_EXPLORATION_AVAILABLE`` availability guard,
* test-binary / examples directory resolution,
* the module-scoped ``fauxware_project`` fixture, and
* the ``BufferedStringIO`` stdout-capture helper.

Test modules import the names they need from here. When the Rust extension
is not built, every Rust symbol below is ``None`` and
``RUST_EXPLORATION_AVAILABLE`` is ``False`` — the per-module ``skipif`` /
``pytestmark`` guards then skip the affected tests before any ``None``
symbol is dereferenced, so skip behavior is identical to the old per-file
guards.
"""

from __future__ import annotations

import os

import pytest

import angr

# Single source of the BufferedStringIO helper (definition lives in the
# benchmark utilities; re-exported here so engine tests don't keep a copy).
from tests.benchmarks.test_utils import BufferedStringIO  # noqa: F401

# --- Availability guard ---------------------------------------------------
# Attempt the broadest import once. Modules import only the symbols they use.
try:
    from angr.rustylib.vex_engine import (
        ExplorationEvent,
        PythonCallbacks,
        RustSimState,
    )
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
    )

    from angr.exploration import RustExplorationManager

    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    _RustExplorationManager = None
    ExplorationEvent = None
    PythonCallbacks = None
    RustSimState = None
    RustExplorationManager = None
    RUST_EXPLORATION_AVAILABLE = False


# --- Path resolution ------------------------------------------------------
# In-repo binaries first, then the angr-examples fauxware fallbacks.
_BINARY_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "binaries"),
    os.path.expanduser("~/repos/angr-examples/examples/fauxware"),
    os.path.expanduser("~/angr-examples/examples/fauxware"),
]
TEST_BINARIES_DIR = next((d for d in _BINARY_CANDIDATES if os.path.isdir(d)), _BINARY_CANDIDATES[0])

# Honors ANGR_EXAMPLES_DIR (set by CI), else the default checkout location.
EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")


# --- Fixtures -------------------------------------------------------------
@pytest.fixture(scope="module")
def fauxware_project():
    """Load the fauxware test binary, shared across a module's tests.

    Module-scoped: callers must treat the project as read-only. Tests that
    mutate project state (e.g. installing hooks) override this with a
    function-scoped fixture of the same name in their own module.
    """
    binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(binary_path):
        pytest.skip("fauxware binary not found")
    return angr.Project(binary_path, auto_load_libs=False)


@pytest.fixture
def amd64_mgr():
    """A bare, project-less ``_RustExplorationManager("amd64")`` (angr-12jjk.8).

    The overwhelmingly common way a Rust-engine test starts is by constructing
    the raw PyO3 manager for the default arch and poking at it directly. This
    fixture is the single source of that construction.

    Function-scoped on purpose: every test gets its own manager, matching the
    per-test construction it replaces. Declare it **last** in a test's
    parameter list so it is instantiated after the test's other
    function-scoped fixtures, preserving the original ordering.

    Deliberately NOT applied everywhere: a test that must construct the
    manager *after* in-body setup — most often a ``monkeypatch.setenv`` of a
    knob the Rust ``__new__`` reads once, e.g. ``RUST_PARALLEL_WORKERS`` —
    keeps its explicit construction, because a fixture would run before the
    test body and read the un-patched environment.
    """
    return _RustExplorationManager("amd64")


# --- Factory monkey-patch -------------------------------------------------
class RustFactoryPatch:
    """Route ``factory.simulation_manager(...)`` / ``simgr(...)`` to
    :class:`RustExplorationManager` for the duration of a ``with`` block.

    Consolidates the two previously-duplicated monkey-patch implementations
    (``test_rust_integration._run_example`` and the
    ``_RustSimManagerFactoryPatch`` from ``test_callable_rust``) into one
    shared context manager (angr-k8pc).

    Args:
        fallback_to_python: when ``True``, calls originating from angr
            internals (analyses, exploration techniques) fall through to the
            original Python ``simulation_manager``. This is **load-bearing**
            for integration runs: the CFG jumptable resolver and exploration
            techniques build internal ``SimState`` objects carrying SimOptions
            like ``DO_RET_EMULATION`` that
            ``RustExplorationManager._check_raise_options`` rejects. When
            ``False`` (default — the callable smoke-test behavior), every call
            is routed to the Rust engine.

    Both behaviors strip the ``techniques`` / ``use_rust_engine`` kwargs that
    :meth:`Callable.perform_call` passes but ``RustExplorationManager`` does
    not accept; this is a harmless superset for the integration path (those
    kwargs are absent there).
    """

    def __init__(self, fallback_to_python: bool = False):
        self._fallback = fallback_to_python
        self._original_sm = None
        self._original_simgr = None

    def __enter__(self):
        from angr.factory import AngrObjectFactory

        self._original_sm = AngrObjectFactory.simulation_manager
        self._original_simgr = AngrObjectFactory.simgr
        original_sm = self._original_sm
        fallback = self._fallback

        def rust_simulation_manager(factory_self, thing=None, **kwargs):
            if fallback:
                import traceback

                for frame in traceback.extract_stack()[:-1]:
                    if "/angr/analyses/" in frame.filename or "/angr/exploration_techniques/" in frame.filename:
                        return original_sm(factory_self, thing, **kwargs)
            # Callable passes techniques=...; RustExplorationManager doesn't
            # accept it — strip silently for the smoke test.
            kwargs.pop("techniques", None)
            kwargs.pop("use_rust_engine", None)
            if thing is None:
                states = [factory_self.entry_state()]
            elif isinstance(thing, (list, tuple)):
                states = list(thing)
            else:
                states = [thing]
            return RustExplorationManager(factory_self.project, active_states=states)

        AngrObjectFactory.simulation_manager = rust_simulation_manager
        AngrObjectFactory.simgr = rust_simulation_manager
        return self

    def __exit__(self, exc_type, exc, tb):
        from angr.factory import AngrObjectFactory

        AngrObjectFactory.simulation_manager = self._original_sm
        AngrObjectFactory.simgr = self._original_simgr
