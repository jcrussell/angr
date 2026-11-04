"""A Python SimProcedure a Rust state bounces to runs on that state's own plugins (angr-mx00p.136.38).

RustBackend keeps no export in ``_state_cache``, so the callback dispatcher
also looks up the SimState the backend stepped the id from and the lineage's
import template; without them a backend's callback ran on a blank state. A
mirror a callback cached mid-step is the next export, so the proc's plugin
edits carry over. And the proxy reads STRICT_PAGE_ACCESS from the Rust state:
a Rust fork's Python-side options are lost once its root leaves every stash.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import EXAMPLES_DIR, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

BABY_RE = os.path.join(EXAMPLES_DIR, "defcon2016quals_baby-re", "baby-re")
UNMAP = os.path.join(EXAMPLES_DIR, "unmapped_analysis", "unmap")
BABY_RE_FIND, BABY_RE_AVOID = 0x4028E9, 0x402941
UNMAP_ERRORS = [
    ("0x1000 (unmapped)", b"e6dba991c1745128787fbc7a9843306cb2bcc63e"),
    ("0x44444000 (unmapped)", b"275edf0657388c3a1197cdadfad7b96da0f977a3"),
]


def _baby_re():
    """baby-re with its scanf hooked by a Python proc that counts calls in ``state.globals``."""
    if not os.path.exists(BABY_RE):
        pytest.skip("angr-examples checkout not found")
    project = angr.Project(BABY_RE, auto_load_libs=False)
    flag_chars = [claripy.BVS(f"flag_{i}", 32) for i in range(13)]

    class Scanf(angr.SimProcedure):
        def run(self, fmt, ptr):  # pylint: disable=arguments-differ,unused-argument
            self.state.mem[ptr].dword = flag_chars[self.state.globals["scanf_count"]]
            self.state.globals["scanf_count"] += 1

    project.hook_symbol("__isoc99_scanf", Scanf(), replace=True)
    return project, flag_chars


def _flag(simgr, flag_chars):
    assert not simgr.errored, simgr.errored
    return "".join(chr(simgr.one_found.solver.eval(c)) for c in flag_chars)


@pytest.mark.parametrize("n", [None, 10_000], ids=["fast-path", "per-step"])
def test_a_python_proc_reads_the_globals_set_on_the_seed(n):
    project, flag_chars = _baby_re()
    backend = RustBackend(project)
    simgr = project.factory.simulation_manager(backend=backend)
    simgr.one_active.options.add(angr.options.LAZY_SOLVES)
    simgr.one_active.globals["scanf_count"] = 0
    simgr.explore(find=BABY_RE_FIND, avoid=BABY_RE_AVOID, **({} if n is None else {"n": n}))
    # The per-step run proves the proc's increment reaches the next step's export.
    assert backend.stats["fast_path_runs"] == (1 if n is None else 0)
    assert _flag(simgr, flag_chars) == "Math is hard!"
    assert simgr.one_found.globals["scanf_count"] == 13


def test_a_python_proc_reads_the_globals_set_on_an_export():
    project, flag_chars = _baby_re()
    backend = RustBackend(project)
    simgr = project.factory.simulation_manager(backend=backend)
    simgr.one_active.options.add(angr.options.LAZY_SOLVES)
    while not backend.stats["claimed"]:
        simgr.step()
    # The key exists on no import: only the export carries it.
    for state in simgr.active:
        state.globals["scanf_count"] = 0
    simgr.explore(find=BABY_RE_FIND, avoid=BABY_RE_AVOID)
    assert _flag(simgr, flag_chars) == "Math is hard!"


def _unmap_project():
    if not os.path.exists(UNMAP):
        pytest.skip("angr-examples checkout not found")
    return angr.Project(UNMAP, auto_load_libs=False)


def _errors(simgr):
    return sorted((str(e.error), e.state.posix.dumps(0)) for e in simgr.errored)


@pytest.mark.timeout(120)
def test_a_python_procs_unmapped_write_faults_in_a_rust_fork():
    """A block-granular run bounces the faulting native strncpy to Python, on a fork whose root is gone."""
    project = _unmap_project()
    state = project.factory.entry_state(add_options={angr.options.STRICT_PAGE_ACCESS})
    mgr = RustExplorationManager(project, [state])
    for _ in range(200):
        if not mgr.active:
            break
        mgr.run(n=1)
    assert not mgr.active
    assert _errors(mgr) == UNMAP_ERRORS


@pytest.mark.timeout(120)
def test_a_backend_explore_ends_on_the_unmapped_faults():
    project = _unmap_project()
    state = project.factory.entry_state(add_options={angr.options.STRICT_PAGE_ACCESS})
    simgr = project.factory.simulation_manager(state, backend=RustBackend(project))
    simgr.explore()
    assert _errors(simgr) == UNMAP_ERRORS
