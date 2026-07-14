"""Shared engine-swap monkeypatch for the benchmark harnesses.

Benchmark ``solve.py`` scripts build their simulation managers through
``project.factory.simulation_manager()``. To measure the Rust engine we swap
that factory method so the bench transparently gets a
``RustExplorationManager`` instead of the Python ``SimulationManager``.

The exclusion rules below are load-bearing and easy to get subtly wrong (a
too-broad exclusion silently routes a whole bench back to the Python engine —
see angr-zbpw0), so they live here once and are shared by
``run_single.py`` and ``run_leak_check.py``.
"""

from __future__ import annotations

import traceback

# Callers whose simulation managers must stay on the Python engine.
#
# - analyses / exploration_techniques: angr internals (CFG, etc.); the bench is
#   not measuring them and the Rust manager has no technique support.
# - simos: angr resolves IFUNC / IRELATIVE relocations at load time by
#   *executing* the resolver through a Callable, which builds an internal
#   simulation manager. Those resolver states carry SimOptions (including
#   SYMBOL_FILL_UNCONSTRAINED_REGISTERS) that the Rust manager rejects, so a
#   static-glibc binary (busybox) would fail to even load. Scope this to
#   ``/angr/simos/`` and NOT to ``/angr/callable.py``: a Callable a *bench*
#   builds (mma_howtouse, flareon2015_10) must still run on Rust, else those
#   benches measure the Python engine under ``--engine rust`` (angr-zbpw0).
PYTHON_ENGINE_CALLERS = (
    "/angr/analyses/",
    "/angr/exploration_techniques/",
    "/angr/simos/",
)


def install_rust_engine_patch(make_manager):
    """Patch ``AngrObjectFactory.simulation_manager`` / ``.simgr`` to build a
    Rust manager via ``make_manager(project, states)``.

    Calls originating from :data:`PYTHON_ENGINE_CALLERS`, and calls passing a
    non-empty ``techniques=`` list (``Callable.perform_call()`` passes an empty
    one), fall back to the original Python factory rather than silently
    dropping behavior the Rust manager cannot honor.

    Returns a ``restore()`` callable that undoes the patch.
    """
    import angr

    original_sm = angr.factory.AngrObjectFactory.simulation_manager
    original_simgr = angr.factory.AngrObjectFactory.simgr

    def patched_simulation_manager(factory_self, thing=None, **kwargs):
        for frame in traceback.extract_stack()[:-1]:
            if any(caller in frame.filename for caller in PYTHON_ENGINE_CALLERS):
                return original_sm(factory_self, thing, **kwargs)

        techniques = kwargs.pop("techniques", None)
        if techniques:
            kwargs["techniques"] = techniques
            return original_sm(factory_self, thing, **kwargs)

        if thing is None:
            states = [factory_self.entry_state()]
        elif isinstance(thing, (list, tuple)):
            states = list(thing)
        else:
            states = [thing]
        return make_manager(factory_self.project, states)

    angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
    angr.factory.AngrObjectFactory.simgr = patched_simulation_manager

    def restore():
        angr.factory.AngrObjectFactory.simulation_manager = original_sm
        angr.factory.AngrObjectFactory.simgr = original_simgr

    return restore
