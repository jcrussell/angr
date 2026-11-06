# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: a Rust state has one options / globals store (angr-vmi2l.265, angr-vmi2l.261).

A proxy write (``px.globals[k] = v``, ``px.options.add(o)``) landed in a
Python-side store no export read, so ``mgr.active[i]`` reported the seed's
values; a fork's first access copied the lineage root's entry, not its
parent's; a SimProcedure's ``state.globals`` edits never reached the proxies;
and a snapshot export (``found_states``, ``get_state_by_id``) carried default
options and empty globals.
"""

from __future__ import annotations

import pytest

import angr
from angr import options as o
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

LOAD = 0x400000
# Four times: cmp <reg>,5 ; ja +0 ; jmp +0. Each ja forks; then jmp $.
FORKS = bytes.fromhex(
    "".join(["4883f805", "7700", "eb00", "4883fb05", "7700", "eb00", "4883f905", "7700", "eb00"]) + "4883fa057700ebfe"
)
ZF = "ZERO_FILL_UNCONSTRAINED_MEMORY"


def _forking_manager():
    proj = angr.load_shellcode(FORKS, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    state.globals["tag"] = "root"
    return RustExplorationManager(proj, [state])


class TestProxyWritesReachExports:
    def test_active_exports_carry_proxy_globals_and_options(self):
        mgr = _forking_manager()
        mgr.run(max_steps=2)
        written = {}
        for px in mgr.active_proxies():
            px.globals["tag"] = f"s{px.state_id}"
            px.options.add(ZF)
            written[px.state_id] = f"s{px.state_id}"
        assert len(written) > 1
        for state in mgr.active:
            sid = state.scratch.rust_mirror_id
            assert state.globals["tag"] == written[sid]
            assert o.ZERO_FILL_UNCONSTRAINED_MEMORY in state.options

    def test_grandchildren_inherit_from_their_parent_not_the_root(self):
        mgr = _forking_manager()
        mgr.run(max_steps=2)
        written = {}
        for px in mgr.active_proxies():
            px.globals["tag"] = f"s{px.state_id}"
            px.options.add(ZF)
            written[px.state_id] = f"s{px.state_id}"
        for _ in range(6):
            mgr.run(max_steps=1)
        tags = {}
        for px in mgr.active_proxies():
            tags[px.state_id] = px.globals["tag"]
            assert ZF in px.options
        new = set(tags) - set(written)
        assert new, "the steps after the writes must fork"
        assert all(tags[sid] in written.values() for sid in new)
        assert {tags[sid] for sid in new} != {written[min(written)]}, "every fork read the root's entry"
        for state in mgr.active:
            assert state.globals["tag"] == tags[state.scratch.rust_mirror_id]

    def test_a_write_after_the_next_step_stays_out_of_the_children(self):
        mgr = _forking_manager()
        mgr.run(max_steps=2)
        before = {px.state_id for px in mgr.active_proxies()}
        for px in mgr.active_proxies():
            px.globals["tag"] = "before"
        mgr.run(max_steps=3)
        mgr.run(max_steps=1)
        children = {px.state_id for px in mgr.active_proxies()} - before
        assert children
        for px in mgr.active_proxies():
            if px.state_id in before:
                px.globals["tag"] = "after"
        assert all(px.globals["tag"] == "before" for px in mgr.active_proxies() if px.state_id in children)


class TestSimProcedureWritesReachProxies:
    def test_a_hook_globals_write_reaches_the_proxy_and_the_export(self):
        proj = angr.load_shellcode(b"\x90" * 8 + b"\xeb\xfe", arch="amd64", load_address=LOAD)

        def hook(state):
            state.globals["hooked"] = state.globals.get("tag")

        proj.hook(LOAD + 2, hook, length=1)
        state = proj.factory.blank_state(addr=LOAD)
        state.globals["tag"] = "root"
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=4)
        (px,) = mgr.active_proxies()
        assert px.globals.get("hooked") == "root"
        assert mgr.active[0].globals.get("hooked") == "root"


class TestMirrorWritesReachTheStore:
    def test_globals_and_options_set_on_mgr_active_reach_the_proxy_and_a_hook(self):
        seen = []

        def hook(state):
            seen.append((state.globals.get("tag"), o.ZERO_FILL_UNCONSTRAINED_MEMORY in state.options))

        # cmp rax,5 ; ja +0 (forks) ; nop x4 (the hook on the third) ; jmp $
        proj = angr.load_shellcode(bytes.fromhex("4883f805770090909090ebfe"), arch="amd64", load_address=LOAD)
        proj.hook(LOAD + 8, hook, length=1)
        mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=LOAD)])
        mgr.run(max_steps=1)
        assert len(mgr.active) == 2
        # One side of the fork may already be past the hook.
        seen.clear()
        for export in mgr.active:
            export.globals["tag"] = "export"
            export.options.add(o.ZERO_FILL_UNCONSTRAINED_MEMORY)
        for px in mgr.active_proxies():
            assert px.globals["tag"] == "export"
        mgr.run(max_steps=4)
        assert seen
        assert all(call == ("export", True) for call in seen)
        for px in mgr.active_proxies():
            assert ZF in px.options


class TestSnapshotExportsKeepSeedMetadata:
    def test_found_states_and_get_state_by_id_keep_seed_options_and_globals(self):
        proj = angr.load_shellcode(b"\x90" * 4 + b"\xeb\xfe", arch="amd64", load_address=LOAD)
        state = proj.factory.blank_state(
            addr=LOAD,
            add_options={o.ZERO_FILL_UNCONSTRAINED_MEMORY, o.ZERO_FILL_UNCONSTRAINED_REGISTERS},
        )
        state.globals["k"] = 1
        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=LOAD + 4)
        assert mgr.found_states
        exports = [mgr.found_states[0], mgr.get_state_by_id(mgr._rust_mgr.get_state_ids("found")[0])]
        for export in exports:
            assert o.ZERO_FILL_UNCONSTRAINED_MEMORY in export.options
            assert o.ZERO_FILL_UNCONSTRAINED_REGISTERS in export.options
            assert dict(export.globals) == {"k": 1}
