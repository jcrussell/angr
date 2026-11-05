"""An export's plugins stay its own, and exports merge on a custom key (angr-mx00p.132.36).

Upstream's ``test_fauxware.py::_run_merge`` releases ``fs`` on every deadended
state, then merges them with ``merge_key=lambda s: s.addr``. Under Rust, a
root's cached frame was both its lineage's plugin template and the export
handed out for the root id, so releasing ``fs`` on it stripped ``fs`` from
every sibling exported afterwards (``_root_template``). And
``SimStateHistory.copy`` dropped the Rust history subclass, so
``SimState.merge`` refused two exports on "differing plugin classes".
"""

from __future__ import annotations

from angr.exploration.rust_manager import RustExplorationManager


def _deadended(project, rust: bool):
    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state]) if rust else project.factory.simulation_manager(state)
    mgr.run()
    return mgr


def test_releasing_a_plugin_on_one_export_leaves_its_siblings_alone(fauxware_project):
    for rust in (False, True):
        mgr = _deadended(fauxware_project, rust)
        assert len(mgr.deadended) > 2, rust
        for state in mgr.deadended:
            assert state.has_plugin("fs"), (rust, state)
            state.release_plugin("fs")


def test_exports_merge_on_a_custom_key_like_python(fauxware_project):
    counts = {}
    for rust in (False, True):
        mgr = _deadended(fauxware_project, rust)
        before = len(mgr.deadended)
        for state in mgr.deadended:
            state.release_plugin("fs")
            if 3 in state.posix.fd:
                state.posix.close(3)
        mgr.merge(stash="deadended", merge_key=lambda s: s.addr)
        merged = [s for s in mgr.deadended if s.history.merge_conditions is not None]
        assert merged, rust
        counts[rust] = (before, len(mgr.deadended))
    assert counts[True] == counts[False]


def test_merged_stdin_solves_under_each_merge_condition_like_python(fauxware_project):
    """Each native ``read`` is its own stdin packet, and merging keeps every path constraint.

    An export's stdin was one evaluated packet for every read, so ``content[2]``
    (the password) did not exist, and the merge exported solvers that deferred
    to Rust, so ``SimStateSolver.merge`` saw no constraints to put under the
    merge conditions (angr-mx00p.132.54).
    """
    for rust in (False, True):
        mgr = _deadended(fauxware_project, rust)
        for state in mgr.deadended:
            state.release_plugin("fs")
            if 3 in state.posix.fd:
                state.posix.close(3)
        mgr.merge(stash="deadended", merge_key=lambda s: s.addr)
        welcome = [b"Welcome" in s.posix.dumps(1) for s in mgr.deadended]
        path = mgr.deadended[welcome.index(True)]
        assert [d.length for d, _ in path.posix.stdin.content] == [64, 8, 64, 8], rust
        password = path.posix.stdin.content[2][0]
        solved = {
            b"SOSNEAKY" in path.solver.eval(password, cast_to=bytes, extra_constraints=(cond,))
            for cond in path.history.merge_conditions
        }
        assert solved == {True, False}, rust
