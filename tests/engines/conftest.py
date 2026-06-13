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
        RustExplorationManager as _RustExplorationManager,
        ExplorationEvent,
        PythonCallbacks,
        RustSimState,
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
TEST_BINARIES_DIR = next(
    (d for d in _BINARY_CANDIDATES if os.path.isdir(d)), _BINARY_CANDIDATES[0]
)

# Honors ANGR_EXAMPLES_DIR (set by CI), else the default checkout location.
EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser(
    "~/repos/angr-examples/examples"
)


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
