"""Shared fixtures and helpers for the Rust-engine test modules.

Consolidates boilerplate that was copy-pasted across the six
``tests/engines/test_*rust*.py`` files (angr-7gdp):

* the ``RUST_EXPLORATION_AVAILABLE`` availability guard,
* test-binary / examples directory resolution,
* the module-scoped ``fauxware_project`` fixture and its function-scoped
  ``hookable_project`` twin,
* the ``fork_solve_pbounce`` synthetic loader and its symbolic-stdin seed
  (angr-mx00p.68).

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
from angr import claripy

# --- Availability guard ---------------------------------------------------
# Attempt the broadest import once. Modules import only the symbols they use.
try:
    from angr.exploration import RustExplorationManager
    from angr.rustylib.vex_engine import (
        ExplorationEvent,
        PythonCallbacks,
        RustSimState,
    )
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
    )

    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    _RustExplorationManager = None
    ExplorationEvent = None
    PythonCallbacks = None
    RustSimState = None
    RustExplorationManager = None
    RUST_EXPLORATION_AVAILABLE = False


# --- Path resolution ------------------------------------------------------
# The angr/binaries checkout next to this repo, as ``tests.common.bin_location``
# defines it (not imported: tests.common raises at import without one).
BIN_LOCATION = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "binaries")

# Honors ANGR_EXAMPLES_DIR (set by CI), else the default checkout location.
EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")

# The fork's own bench tree, which carries the synthetic programs' sources.
_SYNTHETIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "synthetic_examples"
)


def _binary(*candidates: str) -> str:
    """The first candidate path that exists, else the first one (so a skip
    message names where the binary is expected). angr/binaries comes first."""
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


# angr/binaries' x86_64 fauxware, else angr-examples' byte-identical copy.
FAUXWARE_PATH = _binary(
    os.path.join(BIN_LOCATION, "tests", "x86_64", "fauxware"),
    os.path.join(EXAMPLES_DIR, "fauxware", "fauxware"),
)
TEST_BINARIES_DIR = os.path.dirname(FAUXWARE_PATH)

# Eight independent symbolic branches on stdin (256 leaves).
FORK_TREE_8_PATH = _binary(
    os.path.join(BIN_LOCATION, "tests", "x86_64", "fork_tree_8"),
    os.path.join(_SYNTHETIC_DIR, "cow_fork_scaling", "fork_tree_8"),
)
# angr/binaries carries the CGC CADET_00001 sample as sc1_0b32aa01_01.
CADET_00001_PATH = _binary(
    os.path.join(BIN_LOCATION, "tests", "cgc", "sc1_0b32aa01_01"),
    os.path.join(EXAMPLES_DIR, "CADET_00001", "CADET_00001"),
)
HOWTOUSE_DLL_PATH = _binary(
    os.path.join(BIN_LOCATION, "tests", "i386", "windows", "howtouse.dll"),
    os.path.join(EXAMPLES_DIR, "mma_howtouse", "howtouse.dll"),
)

# The bounded exhaustive find-all synthetic: 2^W = 8 leaves, one
# ``read(0, b, 32)``, a ``trap_point(x) -> x`` bounce site and a
# ``reach_target`` find address.
PBOUNCE_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
PBOUNCE_PATH = _binary(
    os.path.join(BIN_LOCATION, "tests", "x86_64", PBOUNCE_NAME),
    os.path.join(_SYNTHETIC_DIR, PBOUNCE_NAME, PBOUNCE_NAME),
)
PBOUNCE_SEED_BYTES = 32


# --- Loaders --------------------------------------------------------------
def load_fauxware(**kwargs) -> angr.Project:
    """A fresh fauxware project (``auto_load_libs=False`` plus ``kwargs``);
    skips the calling test when the binary is absent."""
    if not os.path.exists(FAUXWARE_PATH):
        pytest.skip("fauxware binary not found")
    return angr.Project(FAUXWARE_PATH, auto_load_libs=False, **kwargs)


def load_pbounce() -> angr.Project:
    """A fresh project over the pbounce synthetic; skips the calling test when
    the binary is absent."""
    if not os.path.exists(PBOUNCE_PATH):
        pytest.skip(f"synthetic bench binary not found: {PBOUNCE_PATH}")
    return angr.Project(PBOUNCE_PATH, auto_load_libs=False)


def pbounce_stdin_state(project):
    """A ``blank_state`` at ``main`` whose stdin is one fully symbolic
    ``PBOUNCE_SEED_BYTES`` packet, so the synthetic's single read returns
    symbolic content without an extra fork. Returns ``(state, stdin_bvs)``."""
    stdin_bvs = claripy.BVS("stdin", PBOUNCE_SEED_BYTES * 8)
    state = project.factory.blank_state(addr=project.loader.find_symbol("main").rebased_addr)
    state.posix.stdin.content.append((stdin_bvs, claripy.BVV(PBOUNCE_SEED_BYTES, state.arch.bits)))
    return state, stdin_bvs


class TrapPoint(angr.SimProcedure):
    """Identity hook for the synthetic's ``trap_point(x) -> x`` -- a pure
    Python bounce."""

    def run(self, x):  # pylint: disable=arguments-differ
        return x


def first_state_id(mgr):
    """The first state id of a high-level ``RustExplorationManager`` across
    the ``active``, ``deadended``, ``errored`` and ``unconstrained`` stashes,
    or ``None`` when all four are empty."""
    for stash in ("active", "deadended", "errored", "unconstrained"):
        ids = mgr._rust_mgr.get_state_ids(stash)
        if ids:
            return ids[0]
    return None


# --- Fixtures -------------------------------------------------------------
@pytest.fixture(scope="module")
def fauxware_project():
    """Load the fauxware test binary, shared across a module's tests.

    Module-scoped: callers must treat the project as read-only. Tests that
    mutate project state (e.g. installing hooks) override this with a
    function-scoped fixture of the same name in their own module.
    """
    return load_fauxware()


@pytest.fixture
def hookable_project():
    """A fresh fauxware project per test, for tests that install hooks and so
    must not share the module-scoped ``fauxware_project``."""
    return load_fauxware()


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
