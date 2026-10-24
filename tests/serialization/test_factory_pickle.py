#!/usr/bin/env python3
# pylint:disable=protected-access,missing-class-docstring
"""Pickle compatibility of :class:`angr.factory.AngrObjectFactory` with angr master (angr-mx00p.62).

A factory pickled by angr master unpickles here, and one pickled here unpickles there. The ``engine="rust"``
setting lives on the project (``Project._rust_default``), so the factory's own pickled state stays the upstream
four-field tuple.
"""

from __future__ import annotations

__package__ = __package__ or "tests.serialization"  # pylint:disable=redefined-builtin

import os
import pickle
import threading
import unittest
from unittest import mock

import angr
from angr.factory import AngrObjectFactory
from angr.sim_manager import SimulationManager
from tests.common import bin_location

test_location = os.path.join(bin_location, "tests")


def _upstream_factory_getstate(self):
    """angr master's ``AngrObjectFactory.__getstate__``, verbatim."""
    return self.project, self.default_engine_factory, self.procedure_engine, self._default_cc


def _upstream_factory_setstate(self, state):
    """angr master's ``AngrObjectFactory.__setstate__``, verbatim."""
    self.project, self.default_engine_factory, self.procedure_engine, self._default_cc = state
    self._tls = threading.local()


def _project(**kwargs):
    return angr.Project(os.path.join(test_location, "x86_64", "fauxware"), auto_load_libs=False, **kwargs)


class TestFactoryPickleCompatibility(unittest.TestCase):
    def test_factory_state_has_the_upstream_shape(self):
        p = _project(engine="rust")
        assert p.factory.__getstate__() == _upstream_factory_getstate(p.factory)

    def test_upstream_pickle_loads_here(self):
        p = _project()
        project_getstate = angr.Project.__getstate__

        def upstream_project_getstate(project):
            # angr master's Project has no _rust_default attribute.
            state = project_getstate(project)
            del state["_rust_default"]
            return state

        with (
            mock.patch.object(AngrObjectFactory, "__getstate__", _upstream_factory_getstate),
            mock.patch.object(angr.Project, "__getstate__", upstream_project_getstate),
        ):
            data = pickle.dumps(p.factory, -1)

        factory = pickle.loads(data)
        assert factory.default_engine_factory is p.factory.default_engine_factory
        assert factory.project._rust_default is False
        assert type(factory.simulation_manager()) is SimulationManager

    def test_pickle_from_here_loads_upstream(self):
        for engine in (None, "rust"):
            p = _project(engine=engine)
            data = pickle.dumps(p.factory, -1)
            with mock.patch.object(AngrObjectFactory, "__setstate__", _upstream_factory_setstate):
                factory = pickle.loads(data)
            assert factory.default_engine_factory is p.factory.default_engine_factory
            assert factory.block(factory.project.entry).size > 0

    def test_rust_default_survives_a_roundtrip(self):
        for engine, expected in ((None, False), ("rust", True)):
            factory = pickle.loads(pickle.dumps(_project(engine=engine).factory, -1))
            assert factory.project._rust_default is expected


if __name__ == "__main__":
    unittest.main()
