"""Smoke test for :class:`angr.callable.Callable` under the Rust engine.

Lifts the pattern from the ``mma_howtouse`` benchmark, where ``solve.py``
uses ``project.factory.callable(addr)`` to invoke a function 45 times and
read back a concrete return value per call.

The test does not run all 45 invocations (the full sweep takes ~6.6s under
the Rust engine and is already covered by the benchmark suite); instead it
invokes the first few entries and checks them against the known flag
``MMA{fc7d90ca001fc8712497d88d9ee7efa9e9b32ed8}``.

``Callable.perform_call()`` builds its own SimulationManager internally via
``project.factory.simulation_manager(...)`` without threading a
``use_rust_engine`` kwarg through, so the test monkey-patches the factory
method to return a :class:`RustExplorationManager` — the same hook
``tests/benchmarks/run_single.py`` installs when invoking the bench under
the Rust engine.
"""

from __future__ import annotations

__package__ = __package__ or "tests.engines"  # pylint:disable=redefined-builtin

import os

import claripy
import pytest

import angr

# Availability guard and examples-dir resolution live in conftest (angr-7gdp).
from tests.engines.conftest import (
    EXAMPLES_DIR,
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

HOWTOUSE_DLL = os.path.join(EXAMPLES_DIR, "mma_howtouse", "howtouse.dll")
HOWTOUSE_BASE_ADDR = 0x10000000
HOWTOUSE_FUNC_ADDR = 0x10001130
EXPECTED_FLAG = "MMA{fc7d90ca001fc8712497d88d9ee7efa9e9b32ed8}"


class _RustSimManagerFactoryPatch:
    """Context manager that routes ``factory.simulation_manager(...)`` to
    :class:`RustExplorationManager` for the duration of the ``with`` block.

    Mirrors the patch installed by ``tests/benchmarks/run_single.py`` so that
    code paths invoking ``factory.simulation_manager`` internally (such as
    :meth:`Callable.perform_call`) end up exercising the Rust engine.
    """

    def __init__(self, project: angr.Project):
        self._project = project
        self._original_sm = None
        self._original_simgr = None

    def __enter__(self):
        from angr.factory import AngrObjectFactory

        self._original_sm = AngrObjectFactory.simulation_manager
        self._original_simgr = AngrObjectFactory.simgr

        def rust_simulation_manager(factory_self, thing=None, **kwargs):
            # Callable passes ``techniques=...``; RustExplorationManager
            # doesn't accept it — strip silently for the smoke test.
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


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestCallableRust:
    """``project.factory.callable(addr)`` returns the expected concrete byte
    when the underlying SimulationManager has been swapped for
    :class:`RustExplorationManager`.
    """

    def test_callable_returns_expected_bytes(self):
        if not os.path.exists(HOWTOUSE_DLL):
            pytest.skip(f"mma_howtouse DLL not found at {HOWTOUSE_DLL}")

        # Match the load options the benchmark's solve.py uses; the function
        # pointer table at 0x10001130 only resolves under this base address.
        proj = angr.Project(
            HOWTOUSE_DLL,
            load_options={"main_opts": {"base_addr": HOWTOUSE_BASE_ADDR}},
            auto_load_libs=False,
        )

        # Five characters is enough to exercise the Callable → Rust round
        # trip across multiple invocations without paying the full 45-call
        # bench cost; the bench corpus retains the long-running case.
        n_calls = 5
        expected = EXPECTED_FLAG[:n_calls]

        with _RustSimManagerFactoryPatch(proj):
            howtouse = proj.factory.callable(HOWTOUSE_FUNC_ADDR)
            for i in range(n_calls):
                ret = howtouse(i)
                assert ret is not None, f"Callable returned None for index {i}"
                concrete = claripy.backends.concrete.convert(ret).value
                ch = chr(concrete & 0xFF)
                assert ch == expected[i], f"index {i}: got {ch!r} (0x{concrete:x}), expected {expected[i]!r}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
