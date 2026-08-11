"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import contextlib
import re

import pytest

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestCallStackProxy:
    """Tests for RustCallStackProxy on RustStateProxy."""

    def test_callstack_empty_on_unit_state(self):
        """A bare manager state has no frames; proxy reports depth 0."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        mgr = _RustExplorationManager("amd64")
        state_id = mgr.create_state("active")
        proxy = RustCallStackProxy(mgr, state_id)
        assert len(proxy) == 0
        assert list(proxy) == []
        assert proxy.top is None
        assert proxy.current_function_address == 0
        assert proxy.current_return_target == 0

    def test_callstack_proxy_after_explore(self, fauxware_project):
        """After exploration, found-state callstack frames match the
        snapshot exposed by the underlying Rust manager."""
        from angr.exploration.rust_state_proxy import (
            RustCallStackFrameProxy,
            RustCallStackProxy,
        )

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)
        assert len(mgr.found) > 0

        found_ids = mgr._rust_mgr.get_state_ids("found")
        sid = found_ids[0]
        proxy = RustCallStackProxy(mgr._rust_mgr, sid)
        # Same depth as the raw API, modulo direction.
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        assert len(proxy) == len(raw)
        # Iteration yields proxies with matching attrs in reverse order
        # (top-first).
        frames = list(proxy)
        assert all(isinstance(f, RustCallStackFrameProxy) for f in frames)
        if raw:
            top = frames[0]
            expected_top = raw[-1]  # raw is push-order; top is last
            assert top.call_site_addr == expected_top[0]
            assert top.func_addr == expected_top[1]
            assert top.ret_addr == expected_top[2]
            assert top.stack_ptr == expected_top[3]
            assert proxy.current_function_address == expected_top[1]

    def test_callstack_indexing_and_walk(self):
        """Frame __getitem__ and .next walk the same path."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        # Most-recent first: function 0x300 called from 0x200, etc. The
        # proxy reverses on read, so the fake mgr returns push order
        # (outermost first) — the mirror of the top-first list above.
        class _FakeMgr:
            def get_state_call_stack(self, _sid):
                return [
                    (0x050, 0x100, 0x055, 0x7200),  # bottom / outermost
                    (0x150, 0x200, 0x155, 0x7100),
                    (0x250, 0x300, 0x255, 0x7000),  # top / innermost
                ]

        proxy = RustCallStackProxy(_FakeMgr(), 0)
        assert len(proxy) == 3
        top = proxy[0]
        assert top.func_addr == 0x300
        assert top.next.func_addr == 0x200
        assert top.next.next.func_addr == 0x100
        assert top.next.next.next is None
        # Negative indexing
        assert proxy[-1].func_addr == 0x100

    def test_simstate_callstack_synced_after_explore(self, fauxware_project):
        """The angr SimState returned from mgr.found has its CallStack
        plugin populated from Rust's call_stack — not just the empty
        sentinel from the entry-state template."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)
        assert len(mgr.found) > 0
        found_state = mgr.found[0]

        # Cross-check against the raw Rust API.
        sid = mgr._rust_mgr.get_state_ids("found")[0]
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        if not raw:
            pytest.skip("Rust call stack empty at find — nothing to verify")

        # angr's CallStack iterates top-first; raw is push-order
        # (outermost first). Compare the angr-side top to raw[-1].
        cs_frames = list(found_state.callstack)
        # cs_frames includes the sentinel (zero-frame) at the bottom, so
        # the angr-side has len(raw) + 1 entries (worst case). Just check
        # the top frame matches the most recent Rust call.
        assert len(cs_frames) >= len(raw)
        top_call_site, top_func, top_ret, top_sp = raw[-1]
        assert found_state.callstack.func_addr == top_func
        assert found_state.callstack.call_site_addr == top_call_site
        assert found_state.callstack.ret_addr == top_ret
        assert found_state.callstack.stack_ptr == top_sp

    def test_simstate_fork_isolates_call_stack(self):
        """RustSimState.fork() must clone the call stack so that pushes
        on one side do not appear on the other.

        The Rust fork (state.rs:1496) clones `Vec<CallStackEntry>` on the
        new state. Regression guard: if a future refactor wraps the stack
        in `Arc` or otherwise shares storage without CoW-on-write, this
        test will catch the leak in either direction."""
        parent = RustSimState("amd64")
        # outermost → innermost
        parent.push_call_frame(0x1000, 0x2000, 0x1005, 0x7000)
        parent.push_call_frame(0x2008, 0x3000, 0x200D, 0x6F00)
        assert parent.get_call_stack() == [
            (0x1000, 0x2000, 0x1005, 0x7000),
            (0x2008, 0x3000, 0x200D, 0x6F00),
        ]

        child = parent.fork()
        # Child inherits the prefix.
        assert child.get_call_stack() == parent.get_call_stack()

        # Diverge: parent pushes one more, child pushes a different frame.
        parent.push_call_frame(0xAAAA, 0xBBBB, 0xAAAF, 0x6E00)
        child.push_call_frame(0xCCCC, 0xDDDD, 0xCCD1, 0x6E00)

        # Each side sees only its own divergence.
        assert parent.get_call_stack() == [
            (0x1000, 0x2000, 0x1005, 0x7000),
            (0x2008, 0x3000, 0x200D, 0x6F00),
            (0xAAAA, 0xBBBB, 0xAAAF, 0x6E00),
        ]
        assert child.get_call_stack() == [
            (0x1000, 0x2000, 0x1005, 0x7000),
            (0x2008, 0x3000, 0x200D, 0x6F00),
            (0xCCCC, 0xDDDD, 0xCCD1, 0x6E00),
        ]

    def test_callstack_proxy_reverses_after_fork(self):
        """After a Rust-level fork that produces divergent call stacks,
        the proxy's reverse-iteration semantics (innermost-frame-first)
        must hold independently for each state.

        Risk this guards against: drift in `_frames` reversal logic
        (rust_state_proxy.py:566) — a regression that returned frames in
        push order would silently report wrong top-frame data on every
        forked sibling."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        # Build a manager and add two RustSimStates with divergent stacks
        # to its `active` stash. The proxy is keyed off (mgr, state_id),
        # so registering both states lets us read them via the proxy.
        mgr = _RustExplorationManager("amd64")
        parent = RustSimState("amd64")
        parent.push_call_frame(0x100, 0x200, 0x105, 0x7000)
        parent.push_call_frame(0x208, 0x300, 0x20D, 0x6FF0)
        child = parent.fork()
        # Diverge at the top of the stack (post-fork divergence).
        parent.push_call_frame(0x310, 0x400, 0x315, 0x6FE0)
        child.push_call_frame(0x310, 0x500, 0x315, 0x6FE0)
        mgr.add_state("active", parent)
        mgr.add_state("active", child)
        # add_state forks internally; recover the assigned ids in order.
        ids = mgr.get_state_ids("active")
        assert len(ids) == 2
        parent_id, child_id = ids

        parent_proxy = RustCallStackProxy(mgr, parent_id)
        child_proxy = RustCallStackProxy(mgr, child_id)

        # Reverse-iteration: index 0 must be the innermost frame.
        assert parent_proxy[0].func_addr == 0x400
        assert parent_proxy[-1].func_addr == 0x200  # outermost
        assert child_proxy[0].func_addr == 0x500
        assert child_proxy[-1].func_addr == 0x200

        # Walk via .next from the top — depth and order match push history.
        parent_walk = []
        f = parent_proxy.top
        while f is not None:
            parent_walk.append(f.func_addr)
            f = f.next
        assert parent_walk == [0x400, 0x300, 0x200]

        child_walk = []
        f = child_proxy.top
        while f is not None:
            child_walk.append(f.func_addr)
            f = f.next
        assert child_walk == [0x500, 0x300, 0x200]

    def test_callstack_proxy_independent_across_states(self):
        """Two proxies on sibling states read their own state_id's frames.

        Each `RustCallStackProxy` reads live via
        `mgr.get_state_call_stack(state_id)` (no Python-side cache). If a
        refactor ever keyed the read only by a shared manager-level dict, a
        forked sibling could silently pick up the parent's frames. This pins
        the per-state-id read independence."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        mgr = _RustExplorationManager("amd64")
        a = RustSimState("amd64")
        a.push_call_frame(0x10, 0x20, 0x15, 0x7000)
        b = a.fork()
        b.push_call_frame(0x30, 0x40, 0x35, 0x6FF0)
        mgr.add_state("active", a)
        mgr.add_state("active", b)
        ids = mgr.get_state_ids("active")
        assert len(ids) == 2
        a_id, b_id = ids

        proxy_a = RustCallStackProxy(mgr, a_id)
        proxy_b = RustCallStackProxy(mgr, b_id)

        # Realize both views.
        list(proxy_a)
        list(proxy_b)

        assert len(proxy_a) == 1
        assert len(proxy_b) == 2
        # Independent reverse views — divergence does not leak.
        assert proxy_a[0].func_addr == 0x20
        assert proxy_b[0].func_addr == 0x40
        assert proxy_b[-1].func_addr == 0x20

    def test_callstack_proxy_reads_live_not_stale(self):
        """RustCallStackProxy must read frames live on every access, not
        cache them at first read.

        Regression guard for angr-qwyti.10: a `RustStateProxy` view memoizes
        its `callstack` sub-proxy, so a cached frame list would silently
        serve pre-step call frames after the underlying Rust state advances
        on the same state_id (the register-proxy stale-cache shape from
        angr-4rq7). The mgr below mutates its returned stack between reads;
        the proxy must reflect the change."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        class _MutatingMgr:
            def __init__(self):
                # push order (outermost first)
                self.stack = [(0x100, 0x200, 0x105, 0x7000)]

            def get_state_call_stack(self, _sid):
                return list(self.stack)

        mgr = _MutatingMgr()
        proxy = RustCallStackProxy(mgr, 0)
        assert len(proxy) == 1
        assert proxy.top.func_addr == 0x200

        # Simulate a Rust-side step that pushes another frame on the SAME id.
        mgr.stack.append((0x208, 0x300, 0x20D, 0x6FF0))
        # A cached proxy would still report depth 1 / top 0x200 here.
        assert len(proxy) == 2
        assert proxy.top.func_addr == 0x300
        assert proxy[-1].func_addr == 0x200


class TestInspectProxy:
    """Tests for the loud inspect proxy on RustStateProxy (no-manager case)."""

    def test_inspect_breakpoint_calls_raise_without_manager(self):
        """A standalone _NoOpInspectProxy (used when no manager is attached)
        raises NotImplementedError on every registration method.

        Silent no-ops would let breakpoint-based techniques register hooks
        that never fire. Raising loudly surfaces the unsupported path.
        """
        from angr.exploration.rust_state_proxy import _NoOpInspectProxy

        ins = _NoOpInspectProxy()
        with pytest.raises(NotImplementedError, match="rust_engine"):
            ins.b("mem_read", when="before", action=lambda s: None)
        with pytest.raises(NotImplementedError, match="rust_engine"):
            ins.make_breakpoint("mem_write")
        with pytest.raises(NotImplementedError, match="rust_engine"):
            ins.add_breakpoint("call", lambda s: None)
        with pytest.raises(NotImplementedError, match="rust_engine"):
            ins.remove_breakpoint("call", 0)
        with pytest.raises(NotImplementedError, match="rust_engine"):
            ins.action("call", lambda s: None)


class TestRustInspectMarshalling:
    """Tests for the state.inspect mem_read / mem_write marshalling layer.

    Covers angr-uq4n.2: the Rust → PyO3 → Python state.inspect.action()
    round-trip. The Rust dispatch instrumentation (uq4n.3) is not yet
    wired, so these tests invoke the Python callbacks directly with
    synthetic event payloads — the same shape the Rust side will use.
    """

    def test_callbacks_have_inspect_slots(self):
        """PythonCallbacks exposes set_inspect_mem_{read,write} + bitmask."""
        cbs = PythonCallbacks()
        assert hasattr(cbs, "set_inspect_mem_read")
        assert hasattr(cbs, "set_inspect_mem_write")
        assert hasattr(cbs, "set_inspect_enabled")
        assert hasattr(cbs, "get_inspect_enabled")
        assert cbs.get_inspect_enabled() == 0
        cbs.set_inspect_enabled(0b11)
        assert cbs.get_inspect_enabled() == 0b11
        cbs.set_inspect_enabled(0)
        assert cbs.get_inspect_enabled() == 0

    def test_inspect_proxy_registers_mem_read_bp(self, fauxware_project):
        """Registering a mem_read BP via state.inspect.b flips the bitmask."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr._callbacks.get_inspect_enabled() == 0

        proxy_inspect = mgr._get_inspect_proxy()
        bp = proxy_inspect.b("mem_read", when="before", action=lambda s: None)
        assert bp is not None
        assert mgr._callbacks.get_inspect_enabled() & 0b01 != 0
        # Adding a mem_write BP sets bit 1 too.
        proxy_inspect.b("mem_write", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & 0b10 != 0
        # Removing them clears the bitmask.
        proxy_inspect.remove_breakpoint("mem_read", bp)
        proxy_inspect._mgr._inspect_breakpoints["mem_write"].clear()
        proxy_inspect._mgr._update_inspect_bitmask()
        assert mgr._callbacks.get_inspect_enabled() == 0

    def test_inspect_proxy_accepts_constraints_and_vex_lift(self, fauxware_project):
        """constraints + vex_lift moved into the supported set in angr-4aach.

        reg_read, reg_write, instruction, irsb, exit moved into the
        supported set in angr-d46u; call/return moved in angr-4ai9;
        simprocedure/syscall/dirty moved in angr-xmfj; tmp_read/tmp_write
        moved in angr-64pi; statement moved in angr-t8vf; expr moved in
        angr-lge2; address_concretization/symbolic_variable moved in
        angr-vfst; fork moved in angr-ysml; constraints/vex_lift moved in
        angr-4aach. Registering a BP for the last two must no longer raise
        and must flip their bitmask bits.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()
        ins.b("constraints", when="before", action=lambda s: None)
        ins.b("vex_lift", when="after", action=lambda s: None)
        # Both are python-dispatched: the bit is still flipped for
        # consistency even though the Rust VEX sites don't read it.
        assert mgr._callbacks.get_inspect_enabled() & (1 << 19) != 0
        assert mgr._callbacks.get_inspect_enabled() & (1 << 20) != 0

    def _make_mgr_with_state_id(self, project):
        """Helper: build a manager and return (mgr, valid_state_id) for dispatch tests."""
        state = project.factory.entry_state()
        mgr = RustExplorationManager(project, [state])
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state"
        return mgr, state_ids[0], state

    def test_dispatch_mem_read_fires_bp(self, fauxware_project):
        """Calling _cb_inspect_mem_read invokes the user's BP action."""
        mgr, sid, state = self._make_mgr_with_state_id(fauxware_project)
        events = []

        def on_read(s):
            events.append(
                {
                    "addr": s.inspect.mem_read_address,
                    "length": s.inspect.mem_read_length,
                    "endness": s.inspect.mem_read_endness,
                }
            )

        mgr._get_inspect_proxy().b("mem_read", when="before", action=on_read)
        mgr._cb_inspect_mem_read(sid, "before", 0x401234, 4, None, "Iend_LE")

        assert len(events) == 1
        ev = events[0]
        assert ev["length"] == 4
        assert ev["endness"] == "Iend_LE"
        import claripy

        assert isinstance(ev["addr"], claripy.ast.bv.BV)
        assert state.solver.eval(ev["addr"]) == 0x401234

    def test_dispatch_mem_write_after_carries_value_ast(self, fauxware_project):
        """mem_write AFTER passes the stored value through to mem_write_expr."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []
        sym_val = claripy.BVS("written_value", 32)

        def on_write(s):
            seen.append(
                (
                    s.inspect.mem_write_address,
                    s.inspect.mem_write_length,
                    s.inspect.mem_write_expr,
                    s.inspect.mem_write_endness,
                )
            )

        mgr._get_inspect_proxy().b("mem_write", when="after", action=on_write)
        mgr._cb_inspect_mem_write(sid, "after", 0x402000, 4, sym_val, "Iend_LE")

        assert len(seen) == 1
        _addr, length, expr, endness = seen[0]
        assert length == 4
        assert endness == "Iend_LE"
        assert expr is sym_val

    def test_dispatch_skipped_when_no_bps(self, fauxware_project):
        """Dispatch with empty BP list is a no-op (no exception, nothing fired)."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        # No BPs registered — dispatch must return None (no override) for both
        # read and write so the Rust caller keeps the original load/store.
        assert mgr._cb_inspect_mem_read(sid, "before", 0x1000, 8, None, "Iend_LE") is None
        assert mgr._cb_inspect_mem_write(sid, "after", 0x1000, 8, None, "Iend_LE") is None

    def test_dispatch_reentrancy_guard(self, fauxware_project):
        """A BP action that triggers another inspect dispatch is suppressed.

        Without the guard, a user action that touched memory through the
        same RustExplorationManager could recursively call into the
        dispatcher and overwrite the in-flight inspect attributes
        mid-action.
        """
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        fire_count = [0]

        def reentrant_action(s):
            fire_count[0] += 1
            # Try to trigger another dispatch from inside the BP action.
            mgr._cb_inspect_mem_read(sid, "before", 0xDEAD, 4, None, "Iend_LE")

        mgr._get_inspect_proxy().b("mem_read", when="before", action=reentrant_action)
        mgr._cb_inspect_mem_read(sid, "before", 0x401234, 4, None, "Iend_LE")

        # Outer fire should run exactly once — the reentrant call returns early.
        assert fire_count[0] == 1

    def test_rust_call_inspect_mem_read_invokes_python(self, fauxware_project):
        """The Rust-side call_inspect_mem_read method round-trips into Python."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        hits = []

        def on_read(s):
            hits.append(s.inspect.mem_read_length)

        mgr._get_inspect_proxy().b("mem_read", when="before", action=on_read)

        # Invoke through the Rust PyO3 entry point — exercises both the
        # Rust callback dispatch path and the Python dispatcher.
        mgr._callbacks.call_inspect_mem_read(sid, "before", 0xCAFE, 8, None, "Iend_LE")
        assert hits == [8]

    def test_rust_call_inspect_skips_when_callback_unset(self):
        """call_inspect_mem_* is a no-op when no callback is registered."""
        cbs = PythonCallbacks()
        # No callback set — should not raise; returns None (no override).
        assert cbs.call_inspect_mem_read(-1, "before", 0x1000, 4, None, "Iend_LE") is None
        assert cbs.call_inspect_mem_write(-1, "after", 0x1000, 4, None, "Iend_LE") is None

    def test_mem_read_expr_override_returned(self, fauxware_project):
        """A BP that sets state.inspect.mem_read_expr surfaces the override
        as the callback's return value (value injection — angr-uy32)."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)
        injected = claripy.BVV(0xDEAD, 32)

        def on_read(s):
            s.inspect.mem_read_expr = injected

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        ret = mgr._cb_inspect_mem_read(sid, "after", 0x401234, 4, original, "Iend_LE")
        assert ret is injected

    def test_dispatch_constraints_fires_bp(self, fauxware_project):
        """Adding a constraint through the proxy solver fires the constraints BP.

        angr-4aach: RustSolverProxyPlugin.add dispatches the constraints
        event BP_BEFORE (with added_constraints) then BP_AFTER, mirroring
        Python's SimSolver.add.
        """
        import claripy

        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_constraints(s):
            seen.append(s.inspect.added_constraints)

        mgr._get_inspect_proxy().b("constraints", when="before", action=on_constraints)
        plugin = RustSolverProxyPlugin(mgr._rust_mgr, sid, python_mgr=mgr)
        x = claripy.BVS("x", 32)
        plugin.add(x == 42)

        assert len(seen) == 1
        assert len(seen[0]) == 1
        assert seen[0][0] is (x == 42) or str(seen[0][0]) == str(x == 42)

    def test_dispatch_constraints_before_override(self, fauxware_project):
        """A BP_BEFORE action may replace added_constraints (value injection)."""
        import claripy

        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        x = claripy.BVS("x", 32)
        replacement = [x == 7]

        def on_constraints(s):
            s.inspect.added_constraints = replacement

        mgr._get_inspect_proxy().b("constraints", when="before", action=on_constraints)
        plugin = RustSolverProxyPlugin(mgr._rust_mgr, sid, python_mgr=mgr)
        # The write-through should install the replacement, not the original.
        plugin.add(x == 99)
        cons = mgr._rust_mgr.export_state_constraints(sid)
        assert any(str(c) == str(x == 7) for c in cons)
        assert not any(str(c) == str(x == 99) for c in cons)

    def test_native_fork_guard_fires_constraints_bp(self, fauxware_project):
        """The native fork-guard add fires the constraints BP (angr-op0dn.14.4.1).

        No proxy solver is involved: exploring fauxware forks on the symbolic
        password compare, and `exploration::helpers::add_fork_guard_constraint`
        dispatches BP_BEFORE / BP_AFTER around the `assume_true`/`assume_false`
        that installs the branch guard on the continuing state.
        """
        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        seen = []

        def on_before(s):
            seen.append(("before", list(s.inspect.added_constraints or [])))

        def on_after(s):
            seen.append(("after", list(s.inspect.added_constraints or [])))

        ins = mgr._get_inspect_proxy()
        ins.b("constraints", when="before", action=on_before)
        ins.b("constraints", when="after", action=on_after)

        mgr.run(max_steps=30)

        whens = [w for w, _ in seen]
        assert "before" in whens and "after" in whens, f"no before/after pair: {whens}"
        # Every fired event carries the guard as a one-element list.
        assert all(len(cons) == 1 for _, cons in seen), seen
        # And the guard is a real (symbolic or concrete) claripy AST, not None.
        assert all(hasattr(cons[0], "op") for _, cons in seen), seen

    def test_native_fork_guard_no_bp_no_dispatch(self, fauxware_project):
        """With no constraints BP registered, the native guard add is silent."""
        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        # Bitmask gate keeps bit 19 clear -> Rust never calls into Python.
        assert not mgr._inspect_breakpoints.get("constraints")
        mgr.run(max_steps=10)
        assert mgr.stats["steps"] > 0

    def test_constraints_and_vex_lift_ffi_entry_points_exposed(self):
        """PythonCallbacks exposes the two newest call_inspect_* wrappers.

        angr-sqfj8.13: every other inspect event has a `call_inspect_<evt>`
        test entry point; constraints/vex_lift were the two gaps.
        """
        cbs = PythonCallbacks()
        for name in ("call_inspect_constraints", "call_inspect_vex_lift"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_ffi_call_inspect_constraints_marshals_guard(self, fauxware_project):
        """call_inspect_constraints exports the guard and applies the polarity.

        Drives the Rust `call_inspect_constraints` marshalling directly, the
        way the native fork-guard add does: `is_true=False` must reach the BP
        as `Not(guard)` in a one-element added_constraints list.
        """
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        mgr._get_inspect_proxy().b(
            "constraints", when="before", action=lambda s: seen.append(list(s.inspect.added_constraints or []))
        )

        x = claripy.BVS("x", 32)
        guard = x == 42
        mgr._callbacks.call_inspect_constraints(sid, "before", guard, True)
        mgr._callbacks.call_inspect_constraints(sid, "before", guard, False)

        assert len(seen) == 2
        assert all(len(cons) == 1 for cons in seen), seen
        assert str(seen[0][0]) == str(guard)
        # claripy folds Not(x == 42) to x != 42 on construction, so compare
        # against the same folded form rather than asserting op == "Not".
        assert str(seen[1][0]) == str(claripy.Not(guard))

    def test_ffi_call_inspect_vex_lift_marshals_size_and_buff(self, fauxware_project):
        """call_inspect_vex_lift passes size/buff through as the lift path does."""
        mgr, _sid, _ = self._make_mgr_with_state_id(fauxware_project)
        events = []

        def on_lift(s):
            events.append((s.inspect.vex_lift_addr, s.inspect.vex_lift_size, s.inspect.vex_lift_buff))

        mgr._get_inspect_proxy().b("vex_lift", when="before", action=on_lift)
        mgr._get_inspect_proxy().b("vex_lift", when="after", action=on_lift)

        # BEFORE: no size, raw bytes. AFTER: lifted size, no bytes.
        mgr._callbacks.call_inspect_vex_lift(-1, "before", 0x401234, None, b"\x90\x90")
        mgr._callbacks.call_inspect_vex_lift(-1, "after", 0x401234, 2, None)

        assert len(events) == 2
        (_, before_size, before_buff), (_, after_size, after_buff) = events
        assert before_size is None
        assert before_buff == b"\x90\x90"
        assert after_size == 2
        assert after_buff is None

    def test_dispatch_vex_lift_fires_before_and_after(self, fauxware_project):
        """_cb_lift_block dispatches vex_lift BP_BEFORE then BP_AFTER (angr-4aach)."""
        import claripy

        mgr, _sid, _ = self._make_mgr_with_state_id(fauxware_project)
        events = []

        def on_lift(s):
            events.append((s.inspect.vex_lift_addr, s.inspect.vex_lift_size))

        mgr._get_inspect_proxy().b("vex_lift", when="before", action=on_lift)
        mgr._get_inspect_proxy().b("vex_lift", when="after", action=on_lift)

        entry = fauxware_project.entry
        mgr._cb_lift_block(entry)

        # Two fires: before (size None) then after (concrete IRSB byte size).
        assert len(events) == 2
        before_addr, before_size = events[0]
        after_addr, after_size = events[1]
        assert before_size is None
        assert isinstance(after_size, int) and after_size > 0
        assert isinstance(before_addr, claripy.ast.bv.BV)
        assert isinstance(after_addr, claripy.ast.bv.BV)

    def test_native_lift_fires_vex_lift(self):
        """The native libVEX lift fires its own vex_lift pair (angr-op0dn.14.4.2).

        ``try_native_lift`` bypasses ``_cb_lift_block``, so on a feature-on
        build the Python-lift dispatch above never runs for a natively-lifted
        block — the Rust dispatch is the only thing keeping the event alive.
        Skipped on a build without ``--features libvex-ffi``.
        """
        import claripy

        import angr
        import angr.sim_options as o

        try:
            from angr.rustylib.vex_engine import libvex_ffi_enabled
        except ImportError:
            pytest.skip("build without --features libvex-ffi")
        if not libvex_ffi_enabled():
            pytest.skip("build without --features libvex-ffi")

        # Two-instruction blob: the cold block is served from the binary-region
        # store, so it lifts natively (see test_native_lift_serves_cold_blob).
        shellcode = bytes.fromhex("4831c0") + bytes.fromhex("c3")  # xor rax,rax ; ret
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.ZERO_FILL_UNCONSTRAINED_MEMORY, o.ZERO_FILL_UNCONSTRAINED_REGISTERS},
        )
        state.regs.rsp = 0x7FFFF0000

        mgr = RustExplorationManager(proj, [state], use_native_lift=True)
        mgr._rust_mgr.load_binary_regions([(0x1000, shellcode)])
        mgr.enable_profiling()

        events = []

        def on_lift(s):
            events.append((s.inspect.vex_lift_addr, s.inspect.vex_lift_size, s.inspect.vex_lift_buff))

        mgr._get_inspect_proxy().b("vex_lift", when="before", action=on_lift)
        mgr._get_inspect_proxy().b("vex_lift", when="after", action=on_lift)
        mgr.run(max_steps=5)

        assert mgr.stats.get("rust_native_lift_count", 0) > 0, "nothing lifted natively; test is vacuous"
        # Exactly one BEFORE/AFTER pair per lifted block — a native lift must
        # not also fall through to the _cb_lift_block dispatch.
        assert len(events) == 2 * mgr.stats["rust_native_lift_count"], events

        before_addr, before_size, before_buff = events[0]
        after_addr, after_size, _after_buff = events[1]
        assert isinstance(before_addr, claripy.ast.bv.BV)
        assert isinstance(after_addr, claripy.ast.bv.BV)
        assert before_size is None
        assert isinstance(after_size, int) and after_size > 0
        # The BEFORE fire carries the bytes handed to libVEX.
        assert isinstance(before_buff, bytes) and before_buff.startswith(shellcode[:3])

    def test_mem_read_expr_unchanged_returns_none(self, fauxware_project):
        """A read BP that does not touch mem_read_expr leaves the value
        unchanged — the callback returns None so the original load stands."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)

        def on_read(s):
            _ = s.inspect.mem_read_expr  # read only, no mutation

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        ret = mgr._cb_inspect_mem_read(sid, "after", 0x401234, 4, original, "Iend_LE")
        assert ret is None

    def test_mem_read_override_via_rust_callback(self, fauxware_project):
        """The Rust-side call_inspect_mem_read returns the user's override
        so the interpreter can substitute it for the loaded value."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)
        injected = claripy.BVV(0xBEEF, 32)

        def on_read(s):
            s.inspect.mem_read_expr = injected

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        ret = mgr._callbacks.call_inspect_mem_read(sid, "after", 0xCAFE, 4, original, "Iend_LE")
        assert ret is injected

    def test_mem_write_expr_override_returned(self, fauxware_project):
        """A BP_BEFORE that sets state.inspect.mem_write_expr surfaces the
        override as the callback's return value (value injection — angr-inh0).
        """
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)
        injected = claripy.BVV(0xDEAD, 32)

        def on_write(s):
            s.inspect.mem_write_expr = injected

        mgr._get_inspect_proxy().b("mem_write", when="before", action=on_write)
        ret = mgr._cb_inspect_mem_write(sid, "before", 0x402000, 4, original, "Iend_LE")
        assert ret is injected

    def test_mem_write_expr_unchanged_returns_none(self, fauxware_project):
        """A write BP that does not touch mem_write_expr leaves the value
        unchanged — the callback returns None so the original store stands."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)

        def on_write(s):
            _ = s.inspect.mem_write_expr  # read only, no mutation

        mgr._get_inspect_proxy().b("mem_write", when="before", action=on_write)
        ret = mgr._cb_inspect_mem_write(sid, "before", 0x402000, 4, original, "Iend_LE")
        assert ret is None

    def test_mem_write_override_via_rust_callback(self, fauxware_project):
        """The Rust-side call_inspect_mem_write returns the user's override
        so the interpreter can substitute it for the stored value."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0x1111, 32)
        injected = claripy.BVV(0xBEEF, 32)

        def on_write(s):
            s.inspect.mem_write_expr = injected

        mgr._get_inspect_proxy().b("mem_write", when="before", action=on_write)
        ret = mgr._callbacks.call_inspect_mem_write(sid, "before", 0xCAFE, 4, original, "Iend_LE")
        assert ret is injected

    def test_state_proxy_inspect_routes_to_manager(self, fauxware_project):
        """RustStateProxy.inspect returns the manager-wide proxy when bound."""
        from angr.exploration.rust_state_proxy import RustInspectProxy, RustStateProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state"
        proxy = RustStateProxy(mgr._rust_mgr, state_ids[0], fauxware_project, python_mgr=mgr)
        ins = proxy.inspect
        assert isinstance(ins, RustInspectProxy)
        # Identity: same proxy returned across calls / across state proxies.
        assert proxy.inspect is ins
        proxy2 = RustStateProxy(mgr._rust_mgr, state_ids[0], fauxware_project, python_mgr=mgr)
        assert proxy2.inspect is ins


class TestRustInspectMemReadDispatch:
    """Integration tests for mem_read inspect dispatch from IRExpr::Load
    in the Rust VEX interpreter (angr-uq4n.3).

    Mirror of TestRustInspectMemWriteDispatch — the marshalling-layer
    tests cover the dispatcher itself; these verify the full end-to-end
    path including the Rust-side bitmask gate and the IRExpr::Load hook.
    """

    def test_mem_read_fires_during_exploration(self, fauxware_project):
        """mem_read BP receives at least one event when running the engine.

        fauxware's entry block performs several VEX Load ops while
        reading the saved RBP and ELF rodata — the BP must fire on
        those concrete-address reads.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        events = []

        def on_read(s):
            events.append(
                (
                    s.inspect.mem_read_address,
                    s.inspect.mem_read_length,
                    s.inspect.mem_read_endness,
                )
            )

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        # Bitmask must reflect the BP — sanity-check before stepping.
        assert mgr._callbacks.get_inspect_enabled() & 0b01 != 0

        mgr.run(max_steps=5)

        # At least one load should have fired.
        assert len(events) > 0, "no mem_read events captured during run"
        _addr, length, endness = events[0]
        assert endness in ("Iend_LE", "Iend_BE")
        assert 0 < length <= 16

    def test_mem_read_skipped_when_no_bp(self, fauxware_project):
        """Without any mem_read BP, the bitmask gate keeps dispatch off
        and exploration still progresses normally."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Bitmask should be 0 with no BPs.
        assert mgr._callbacks.get_inspect_enabled() == 0

        mgr.run(max_steps=5)

        # Exploration must actually progress (a no-op run would also keep the
        # bitmask at 0, masking a regression that short-circuits stepping).
        assert mgr.stats["steps"] > 0
        # Gate must stay clear across the run — no event bit gets set when no
        # BP is registered.
        assert mgr._callbacks.get_inspect_enabled() == 0

    def test_mem_read_expr_injection_during_exploration(self, fauxware_project):
        """A mem_read BP_AFTER that overrides mem_read_expr drives the value
        back through the Rust interpreter (claripy->RustBV round-trip) without
        crashing, and exploration completes (angr-uy32 value injection)."""
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fired = [0]

        def on_read(s):
            fired[0] += 1
            length = s.inspect.mem_read_length
            # Override with a same-width concrete value so the width guard in
            # dispatch_mem_read_inspect accepts it and substitutes it for the
            # loaded value.
            if isinstance(length, int) and length > 0:
                s.inspect.mem_read_expr = claripy.BVV(0, length * 8)

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        mgr.run(max_steps=5)
        assert fired[0] > 0, "override BP must fire at least once"

    def test_mem_read_reentrancy_with_proxy_access(self, fauxware_project):
        """BP action that touches the firing state via the proxy must
        not deadlock or corrupt the exploration loop.

        The state owning the firing event is currently held by the
        interpreter, so the proxy's lookups raise PyValueError. The
        dispatcher catches and logs that — the contract is that
        exploration continues without deadlock or wrong-answer.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_read(s):
            fire_count[0] += 1
            with contextlib.suppress(Exception):
                _ = s.addr

        mgr._get_inspect_proxy().b("mem_read", when="after", action=on_read)
        mgr.run(max_steps=5)

        assert fire_count[0] > 0, "BP must fire at least once"


class TestRustInspectMemWriteDispatch:
    """Integration tests for mem_write inspect dispatch from IRStmt::Store
    in the Rust VEX interpreter (angr-uq4n.4).

    The marshalling-layer tests (TestRustInspectMarshalling) call the
    Python dispatcher directly; these tests verify the full end-to-end
    path including the Rust-side bitmask gate and the IRStmt::Store hook.
    """

    def test_mem_write_fires_during_exploration(self, fauxware_project):
        """mem_write BP receives at least one event when running the engine.

        fauxware's entry block writes the saved RBP and locals to the
        stack via VEX Store ops — the BP must fire on those concrete-
        address writes.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        events = []

        def on_write(s):
            events.append(
                (
                    s.inspect.mem_write_address,
                    s.inspect.mem_write_length,
                    s.inspect.mem_write_endness,
                )
            )

        mgr._get_inspect_proxy().b("mem_write", when="after", action=on_write)
        # Bitmask must reflect the BP — sanity-check before stepping.
        assert mgr._callbacks.get_inspect_enabled() & 0b10 != 0

        mgr.run(max_steps=5)

        # At least one stack-frame store should have fired.
        assert len(events) > 0, "no mem_write events captured during run"
        _addr, length, endness = events[0]
        # Endness should be a valid VEX endness string.
        assert endness in ("Iend_LE", "Iend_BE")
        # Length should be a positive integer up to register width.
        assert 0 < length <= 16

    def test_mem_write_skipped_when_no_bp(self, fauxware_project):
        """Without any mem_write BP, the bitmask gate keeps dispatch off
        and exploration still progresses normally.

        Regression for the zero-overhead common case: a manager with no
        breakpoints must not trigger the Python dispatcher even once.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Bitmask should be 0 with no BPs.
        assert mgr._callbacks.get_inspect_enabled() == 0

        # Run does not raise — we have no way to assert "0 fires" without
        # instrumenting the Rust side, but the dispatcher being off is
        # the contract being tested.
        mgr.run(max_steps=5)

        # Exploration must actually progress (a no-op run would also keep the
        # bitmask at 0, masking a regression that short-circuits stepping).
        assert mgr.stats["steps"] > 0
        # The gate stays clear across the run with no BP registered.
        assert mgr._callbacks.get_inspect_enabled() == 0

    def test_mem_write_reentrancy_with_proxy_access(self, fauxware_project):
        """BP action that touches the firing state via the proxy must
        not deadlock or corrupt the exploration loop.

        The state owning the firing event is currently held by the
        interpreter (popped from the active stash for the duration of
        the step), so the proxy's lookups raise PyValueError. The
        dispatcher catches and logs that — the test's contract is that
        exploration continues without deadlock or wrong-answer.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_write(s):
            fire_count[0] += 1
            # Try to touch the firing state via the proxy. With the state
            # currently held by the interpreter, this swallows a "state
            # not found" error inside the dispatcher. The contract is
            # that exploration keeps running and does not deadlock.
            with contextlib.suppress(Exception):
                _ = s.addr

        mgr._get_inspect_proxy().b("mem_write", when="after", action=on_write)
        # 5 steps is enough to hit a store; bound steps so a stuck loop
        # would still time out via pytest's default timeout.
        mgr.run(max_steps=5)

        assert fire_count[0] > 0, "BP must fire at least once"

    def test_mem_write_expr_injection_during_exploration(self, fauxware_project):
        """A mem_write BP_BEFORE that overrides mem_write_expr drives the value
        back through the Rust interpreter (claripy->RustBV round-trip) without
        crashing, and exploration completes (angr-inh0 value injection)."""
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fired = [0]

        def on_write(s):
            fired[0] += 1
            length = s.inspect.mem_write_length
            # Override with a same-width concrete value so the width guard in
            # dispatch_mem_write_inspect accepts it and substitutes it for the
            # stored value before commit.
            if isinstance(length, int) and length > 0:
                s.inspect.mem_write_expr = claripy.BVV(0, length * 8)

        mgr._get_inspect_proxy().b("mem_write", when="before", action=on_write)
        mgr.run(max_steps=5)
        assert fired[0] > 0, "override BP must fire at least once"


class TestRustInspectExtendedEvents:
    """Integration tests for the inspect events added in angr-d46u:
    reg_read, reg_write, instruction, irsb, exit.

    Mirrors the mem_read / mem_write coverage: bitmask flips on BP add,
    callback round-trip exercises the marshalling layer, and at least
    one event fires during a short exploration run.
    """

    def _make_mgr_with_state_id(self, project):
        state = project.factory.entry_state()
        mgr = RustExplorationManager(project, [state])
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state"
        return mgr, state_ids[0], state

    def test_callbacks_expose_extended_inspect_slots(self):
        """PythonCallbacks gained set_inspect_{reg_read,reg_write,instruction,irsb,exit}."""
        cbs = PythonCallbacks()
        for name in (
            "set_inspect_reg_read",
            "set_inspect_reg_write",
            "set_inspect_instruction",
            "set_inspect_irsb",
            "set_inspect_exit",
            "call_inspect_reg_read",
            "call_inspect_reg_write",
            "call_inspect_instruction",
            "call_inspect_irsb",
            "call_inspect_exit",
        ):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_bitmask_per_event(self, fauxware_project):
        """Registering a BP for each extended event flips its assigned bit."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr._callbacks.get_inspect_enabled() == 0

        proxy_inspect = mgr._get_inspect_proxy()
        # Bits per _INSPECT_EVENT_BITS: 2/3/5/6/7 for these events.
        expected = {
            "reg_read": 1 << 2,
            "reg_write": 1 << 3,
            "exit": 1 << 5,
            "instruction": 1 << 6,
            "irsb": 1 << 7,
        }
        for evt, bit in expected.items():
            mask_before = mgr._callbacks.get_inspect_enabled()
            proxy_inspect.b(evt, when="before", action=lambda s: None)
            mask_after = mgr._callbacks.get_inspect_enabled()
            assert mask_after & bit != 0, f"{evt} did not set bit {bit:#b}"
            assert mask_after != mask_before

    def test_dispatch_reg_read_fires_bp(self, fauxware_project):
        """_cb_inspect_reg_read invokes the user's BP with reg_read_* attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        events = []

        def on_read(s):
            events.append(
                (
                    s.inspect.reg_read_offset,
                    s.inspect.reg_read_length,
                    s.inspect.reg_read_expr,
                )
            )

        mgr._get_inspect_proxy().b("reg_read", when="after", action=on_read)
        val = claripy.BVV(0xDEADBEEF, 32)
        mgr._cb_inspect_reg_read(sid, "after", 16, 4, val)

        assert events == [(16, 4, val)]

    def test_dispatch_reg_write_fires_bp(self, fauxware_project):
        """_cb_inspect_reg_write invokes the user's BP with reg_write_* attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_write(s):
            seen.append(
                (
                    s.inspect.reg_write_offset,
                    s.inspect.reg_write_length,
                    s.inspect.reg_write_expr,
                )
            )

        mgr._get_inspect_proxy().b("reg_write", when="after", action=on_write)
        val = claripy.BVS("written_reg", 32)
        mgr._cb_inspect_reg_write(sid, "after", 32, 4, val)

        assert seen == [(32, 4, val)]

    def test_dispatch_instruction_fires_bp(self, fauxware_project):
        """_cb_inspect_instruction invokes the user's BP with the IMark addr."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        addrs = []

        def on_insn(s):
            addrs.append(s.inspect.instruction)

        mgr._get_inspect_proxy().b("instruction", when="before", action=on_insn)
        mgr._cb_inspect_instruction(sid, "before", 0x401234)
        assert addrs == [0x401234]

    def test_dispatch_irsb_fires_bp(self, fauxware_project):
        """_cb_inspect_irsb invokes the user's BP with the block address."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        addrs = []

        def on_irsb(s):
            addrs.append(s.inspect.address)

        mgr._get_inspect_proxy().b("irsb", when="before", action=on_irsb)
        mgr._cb_inspect_irsb(sid, "before", 0x400580)
        assert addrs == [0x400580]

    def test_dispatch_exit_fires_bp(self, fauxware_project):
        """_cb_inspect_exit invokes the user's BP with target/guard/jumpkind."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_exit(s):
            seen.append(
                (
                    s.inspect.exit_target,
                    s.inspect.exit_guard,
                    s.inspect.exit_jumpkind,
                )
            )

        mgr._get_inspect_proxy().b("exit", when="before", action=on_exit)
        guard = claripy.BVS("cond", 1)
        mgr._cb_inspect_exit(sid, "before", 0x401500, "Ijk_Boring", guard)

        assert len(seen) == 1
        target, g, jk = seen[0]
        assert isinstance(target, claripy.ast.bv.BV)
        # Project is AMD64 (fauxware) — addresses are 64-bit.
        assert target.size() == 64
        assert g is guard
        assert jk == "Ijk_Boring"

    def test_instruction_fires_during_exploration(self, fauxware_project):
        """instruction BP receives events for every IMark during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_insn(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("instruction", when="before", action=on_insn)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 6) != 0
        mgr.run(max_steps=3)
        assert fire_count[0] > 0, "no instruction events captured"

    def test_irsb_fires_during_exploration(self, fauxware_project):
        """irsb BP receives at least one event per stepped block."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        addrs = []

        def on_irsb(s):
            addrs.append(s.inspect.address)

        mgr._get_inspect_proxy().b("irsb", when="before", action=on_irsb)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 7) != 0
        mgr.run(max_steps=3)
        assert len(addrs) > 0, "no irsb events captured"
        # Entry block of fauxware is around 0x400580 (_start) — any
        # plausible code address suffices.
        assert all(a > 0 for a in addrs)

    def test_reg_read_fires_during_exploration(self, fauxware_project):
        """reg_read BP fires on VEX IRExpr::Get during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_read(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("reg_read", when="after", action=on_read)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 2) != 0
        mgr.run(max_steps=3)
        assert fire_count[0] > 0, "no reg_read events captured"

    def test_reg_write_fires_during_exploration(self, fauxware_project):
        """reg_write BP fires on VEX IRStmt::Put during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_write(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("reg_write", when="after", action=on_write)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 3) != 0
        mgr.run(max_steps=3)
        assert fire_count[0] > 0, "no reg_write events captured"

    def test_extended_events_skipped_when_no_bp(self, fauxware_project):
        """With no BPs for the extended events, exploration runs normally
        and the bitmask stays clear."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr._callbacks.get_inspect_enabled() == 0
        mgr.run(max_steps=3)
        # Exploration must progress and the gate must stay clear across the run
        # (a no-op run or a spuriously-set event bit would otherwise pass).
        assert mgr.stats["steps"] > 0
        assert mgr._callbacks.get_inspect_enabled() == 0

    def test_call_return_callback_slots_exposed(self):
        """angr-4ai9: PythonCallbacks gained set_inspect_{call,return}."""
        cbs = PythonCallbacks()
        for name in (
            "set_inspect_call",
            "set_inspect_return",
            "call_inspect_call",
            "call_inspect_return",
        ):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_call_return_bits_match_spec(self, fauxware_project):
        """Registering a `call`/`return` BP flips the expected bit (8/9)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("call", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 8) != 0
        ins.b("return", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 9) != 0

    def test_dispatch_call_fires_bp(self, fauxware_project):
        """_cb_inspect_call invokes the user's BP with function_address."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_call(s):
            seen.append(s.inspect.function_address)

        mgr._get_inspect_proxy().b("call", when="before", action=on_call)
        mgr._cb_inspect_call(sid, "before", 0x401ABC)

        assert len(seen) == 1
        addr = seen[0]
        assert isinstance(addr, claripy.ast.bv.BV)
        assert addr.size() == 64  # AMD64
        assert addr.concrete_value == 0x401ABC

    def test_dispatch_return_fires_bp(self, fauxware_project):
        """_cb_inspect_return invokes the user's BP with function_address."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_ret(s):
            seen.append(s.inspect.function_address)

        mgr._get_inspect_proxy().b("return", when="before", action=on_ret)
        mgr._cb_inspect_return(sid, "before", 0x4011A0)

        assert len(seen) == 1
        addr = seen[0]
        assert isinstance(addr, claripy.ast.bv.BV)
        assert addr.concrete_value == 0x4011A0

    def test_call_fires_before_and_after(self, fauxware_project):
        """A `call` BP registered for either phase sees only its phase."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        before_count = [0]
        after_count = [0]

        mgr._get_inspect_proxy().b(
            "call",
            when="before",
            action=lambda s: before_count.__setitem__(0, before_count[0] + 1),
        )
        mgr._get_inspect_proxy().b(
            "call",
            when="after",
            action=lambda s: after_count.__setitem__(0, after_count[0] + 1),
        )
        # Drive both phases for a single call event.
        mgr._cb_inspect_call(sid, "before", 0x40_0000)
        mgr._cb_inspect_call(sid, "after", 0x40_0000)
        assert before_count[0] == 1
        assert after_count[0] == 1

    def test_call_fires_during_exploration(self, fauxware_project):
        """call BP fires during exploration (fauxware has internal calls)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        targets = []

        def on_call(s):
            targets.append(s.inspect.function_address.concrete_value)

        mgr._get_inspect_proxy().b("call", when="before", action=on_call)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 8) != 0
        # fauxware's _start calls __libc_start_main almost immediately, so a
        # bounded run is guaranteed to dispatch at least one Ijk_Call.
        mgr.run(max_steps=20)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 8) != 0
        # The call BP must actually fire during exploration (not merely be
        # registered) and carry a real, non-zero callee address.
        assert len(targets) > 0, "call BP never fired during exploration"
        assert all(t > 0 for t in targets)

    def test_return_fires_during_exploration(self, fauxware_project):
        """return BP fires once Rust pops a frame (sym execution drives this)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seen = []

        def on_ret(s):
            seen.append(s.inspect.function_address.concrete_value)

        mgr._get_inspect_proxy().b("return", when="before", action=on_ret)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 9) != 0
        mgr.run(max_steps=20)
        # The return BP must actually fire (Rust pops a frame) and carry a
        # real, non-zero callee address — an empty/zero-address dispatch fails.
        assert len(seen) >= 1, "return BP never fired during exploration"
        assert all(t > 0 for t in seen)

    # ---- angr-64pi: tmp_read / tmp_write (VEX RdTmp/WrTmp) ----

    def test_tmp_callback_slots_exposed(self):
        """PythonCallbacks gained set_inspect_{tmp_read,tmp_write}."""
        cbs = PythonCallbacks()
        for name in (
            "set_inspect_tmp_read",
            "set_inspect_tmp_write",
            "call_inspect_tmp_read",
            "call_inspect_tmp_write",
        ):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_tmp_bits_match_spec(self, fauxware_project):
        """Registering a `tmp_read`/`tmp_write` BP flips bits 13/14."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("tmp_read", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 13) != 0
        ins.b("tmp_write", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 14) != 0

    def test_dispatch_tmp_read_fires_bp(self, fauxware_project):
        """_cb_inspect_tmp_read invokes the user's BP with tmp_read_* attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_read(s):
            seen.append(
                (
                    s.inspect.tmp_read_num,
                    s.inspect.tmp_read_expr,
                )
            )

        mgr._get_inspect_proxy().b("tmp_read", when="after", action=on_read)
        val = claripy.BVV(0xDEADBEEF, 32)
        mgr._cb_inspect_tmp_read(sid, "after", 7, val)
        assert seen == [(7, val)]

    def test_dispatch_tmp_write_fires_bp(self, fauxware_project):
        """_cb_inspect_tmp_write invokes the user's BP with tmp_write_* attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_write(s):
            seen.append(
                (
                    s.inspect.tmp_write_num,
                    s.inspect.tmp_write_expr,
                )
            )

        mgr._get_inspect_proxy().b("tmp_write", when="after", action=on_write)
        val = claripy.BVS("written_tmp", 64)
        mgr._cb_inspect_tmp_write(sid, "after", 3, val)
        assert seen == [(3, val)]

    def test_tmp_read_fires_during_exploration(self, fauxware_project):
        """tmp_read BP fires on VEX RdTmp during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_read(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("tmp_read", when="after", action=on_read)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 13) != 0
        mgr.run(max_steps=3)
        # Every IRSB has multiple RdTmps (binop args, store data, ...).
        assert fire_count[0] > 0, "no tmp_read events captured"

    def test_tmp_write_fires_during_exploration(self, fauxware_project):
        """tmp_write BP fires on VEX WrTmp during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_write(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("tmp_write", when="after", action=on_write)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 14) != 0
        mgr.run(max_steps=3)
        assert fire_count[0] > 0, "no tmp_write events captured"

    # ---- angr-t8vf: statement (per VEX IR statement) ----

    def test_statement_callback_slot_exposed(self):
        """PythonCallbacks gained set_inspect_statement / call_inspect_statement."""
        cbs = PythonCallbacks()
        for name in ("set_inspect_statement", "call_inspect_statement"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_statement_bit_matches_spec(self, fauxware_project):
        """Registering a `statement` BP flips bit 15."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("statement", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 15) != 0

    def test_dispatch_statement_fires_bp(self, fauxware_project):
        """_cb_inspect_statement invokes the user's BP with the statement attr."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_stmt(s):
            seen.append(s.inspect.statement)

        mgr._get_inspect_proxy().b("statement", when="before", action=on_stmt)
        mgr._cb_inspect_statement(sid, "before", 4)
        assert seen == [4]

    def test_statement_fires_during_exploration(self, fauxware_project):
        """statement BP fires per VEX IR statement during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        seen_indices = []

        def on_stmt(s):
            seen_indices.append(s.inspect.statement)

        mgr._get_inspect_proxy().b("statement", when="before", action=on_stmt)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 15) != 0
        mgr.run(max_steps=3)
        # Every IRSB has multiple statements; we should see indices starting at 0.
        assert len(seen_indices) > 0, "no statement events captured"
        assert min(seen_indices) == 0, f"stmt indices should start at 0, got min={min(seen_indices)}"

    # ---- angr-lge2: expr (per VEX IR expression eval) ----

    def test_expr_callback_slot_exposed(self):
        """PythonCallbacks gained set_inspect_expr / call_inspect_expr."""
        cbs = PythonCallbacks()
        for name in ("set_inspect_expr", "call_inspect_expr"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_expr_bit_matches_spec(self, fauxware_project):
        """Registering an `expr` BP flips bit 16 (u16→u32 widening)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("expr", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 16) != 0

    def test_dispatch_expr_fires_bp(self, fauxware_project):
        """_cb_inspect_expr invokes the user's BP with expr_result attr."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_expr(s):
            seen.append((s.inspect.expr, s.inspect.expr_result))

        mgr._get_inspect_proxy().b("expr", when="after", action=on_expr)
        sentinel = claripy.BVV(0xDEADBEEF, 32)
        mgr._cb_inspect_expr(sid, "after", sentinel)
        assert len(seen) == 1
        expr_attr, expr_result = seen[0]
        assert expr_attr is None
        assert expr_result is sentinel

    def test_expr_fires_during_exploration(self, fauxware_project):
        """expr BP fires per IR expression eval during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fire_count = [0]

        def on_expr(s):
            fire_count[0] += 1

        mgr._get_inspect_proxy().b("expr", when="after", action=on_expr)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 16) != 0
        mgr.run(max_steps=3)
        # eval_expr_with_callbacks fires for every IR expression in every
        # IRSB (constants, RdTmp, register reads, etc.) — even three steps
        # of fauxware should produce dozens of events.
        assert fire_count[0] > 0, "no expr events captured"

    # ---- angr-vfst: address_concretization ----

    def test_address_concretization_callback_slot_exposed(self):
        """PythonCallbacks gained set_inspect_address_concretization / call_*."""
        cbs = PythonCallbacks()
        for name in ("set_inspect_address_concretization", "call_inspect_address_concretization"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_address_concretization_bit_matches_spec(self, fauxware_project):
        """Registering an `address_concretization` BP flips bit 17."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("address_concretization", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 17) != 0

    def test_dispatch_address_concretization_before_fires_bp(self, fauxware_project):
        """_cb_inspect_address_concretization invokes the BP for BP_BEFORE."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_ac(s):
            seen.append(
                (
                    s.inspect.address_concretization_action,
                    s.inspect.address_concretization_expr,
                    s.inspect.address_concretization_result,
                    s.inspect.address_concretization_strategy,
                    s.inspect.address_concretization_memory,
                    s.inspect.address_concretization_add_constraints,
                )
            )

        mgr._get_inspect_proxy().b("address_concretization", when="before", action=on_ac)
        addr_ast = claripy.BVS("sym_addr", 64)
        mgr._cb_inspect_address_concretization(sid, "before", "load", addr_ast, None)
        assert len(seen) == 1
        action, expr_ast, result, strategy, memory, add_constraints = seen[0]
        assert action == "load"
        assert expr_ast is addr_ast
        assert result is None
        # MVP gap: strategy/memory/add_constraints are not surfaced.
        assert strategy is None
        assert memory is None
        assert add_constraints is None

    def test_dispatch_address_concretization_after_carries_result(self, fauxware_project):
        """_cb_inspect_address_concretization passes concretization result list on AFTER."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_ac(s):
            seen.append(
                (
                    s.inspect.address_concretization_action,
                    s.inspect.address_concretization_result,
                )
            )

        mgr._get_inspect_proxy().b("address_concretization", when="after", action=on_ac)
        addr_ast = claripy.BVS("sym_addr", 64)
        mgr._cb_inspect_address_concretization(
            sid,
            "after",
            "store",
            addr_ast,
            [0x400000, 0x400004, 0x400008],
        )
        assert len(seen) == 1
        action, result = seen[0]
        assert action == "store"
        assert result == [0x400000, 0x400004, 0x400008]

    # ---- angr-vfst: symbolic_variable ----

    def test_symbolic_variable_callback_slot_exposed(self):
        """PythonCallbacks gained set_inspect_symbolic_variable / call_*."""
        cbs = PythonCallbacks()
        for name in ("set_inspect_symbolic_variable", "call_inspect_symbolic_variable"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_symbolic_variable_bit_matches_spec(self, fauxware_project):
        """Registering a `symbolic_variable` BP flips bit 18."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("symbolic_variable", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 18) != 0

    def test_dispatch_symbolic_variable_fires_bp(self, fauxware_project):
        """_cb_inspect_symbolic_variable invokes BP with name/size/expr attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_sv(s):
            seen.append(
                (
                    s.inspect.symbolic_name,
                    s.inspect.symbolic_size,
                    s.inspect.symbolic_expr,
                )
            )

        mgr._get_inspect_proxy().b("symbolic_variable", when="after", action=on_sv)
        expr_ast = claripy.BVS("mem_400000_4", 32)
        mgr._cb_inspect_symbolic_variable(sid, "after", "mem_400000_4", 32, expr_ast)
        assert len(seen) == 1
        name, size, expr = seen[0]
        assert name == "mem_400000_4"
        assert size == 32
        assert expr is expr_ast

    # ---- angr-1c88c: end-to-end Rust dispatch (not the Python relay) ----

    def test_address_concretization_fires_end_to_end_from_rust(self):
        """angr-1c88c: drive the *native* dispatch site
        ``expressions.rs::dispatch_address_concretization_inspect`` end to end.

        The existing ``test_dispatch_address_concretization_*`` tests invoke
        ``mgr._cb_inspect_address_concretization`` directly — they cover the
        Python relay but never prove the Rust interpreter reaches the gate(17)
        + ``rustbv_to_claripy`` marshalling on its own. This test steps a real
        IRSB containing a load from a *symbolic* address that concretizes to an
        unmapped page: that is the only path that returns ``None`` from
        ``try_rust_memory_load`` (memory-layer ``load_symbolic_unified`` errors
        ``Unmapped``) and so falls through to the interpreter's
        ``load_symbolic_addr`` → ``dispatch_address_concretization_inspect``.

        Shellcode: ``mov rax, [rbx] ; ret`` with ``rbx`` a BVS constrained to
        a single unmapped address. The BP must fire with a claripy-AST ``expr``
        (proves the round-trip back through ``inspect_ast``), a BEFORE event
        carrying ``result is None`` and an AFTER event carrying the concretized
        address list.
        """
        import claripy

        import angr

        unmapped = 0xDEAD0000
        # mov rax, [rbx] ; ret
        proj = angr.load_shellcode(b"\x48\x8b\x03\xc3", "amd64")
        state = proj.factory.blank_state(addr=proj.entry)
        sym_addr = claripy.BVS("sym_load_addr", 64)
        state.solver.add(sym_addr == unmapped)
        state.regs.rbx = sym_addr

        mgr = RustExplorationManager(proj, [state])
        events = []

        def _on_before(s):
            events.append(
                (
                    "before",
                    s.inspect.address_concretization_action,
                    s.inspect.address_concretization_expr,
                    s.inspect.address_concretization_result,
                )
            )

        def _on_after(s):
            events.append(
                (
                    "after",
                    s.inspect.address_concretization_action,
                    s.inspect.address_concretization_expr,
                    s.inspect.address_concretization_result,
                )
            )

        ins = mgr._get_inspect_proxy()
        ins.b("address_concretization", when="before", action=_on_before)
        ins.b("address_concretization", when="after", action=_on_after)
        # Registering the BP must flip bit 17 so the native gate opens.
        assert mgr._callbacks.get_inspect_enabled() & (1 << 17) != 0

        mgr.run(max_steps=1)

        befores = [e for e in events if e[0] == "before"]
        afters = [e for e in events if e[0] == "after"]
        assert befores, f"native BP_BEFORE never fired; events={len(events)}"
        assert afters, f"native BP_AFTER never fired; events={len(events)}"

        # BEFORE: action="load", expr is a real claripy AST (marshalled back
        # from the Rust BV), result is None (concretizer has not run yet).
        _, b_action, b_expr, b_result = befores[0]
        assert b_action == "load"
        assert isinstance(b_expr, claripy.ast.Base), f"expr not claripy AST: {type(b_expr)}"
        assert b_expr.size() == 64
        assert b_result is None

        # AFTER: result carries the concretized address list (single unmapped
        # solution). This proves the Rust→Python result marshalling, not just
        # the expr round-trip.
        _, a_action, a_expr, a_result = afters[0]
        assert a_action == "load"
        assert isinstance(a_expr, claripy.ast.Base)
        assert a_result is not None
        assert unmapped in a_result, f"concretized result missing target: {a_result}"

    def test_symbolic_variable_fires_end_to_end_from_rust(self):
        """angr-1c88c: drive the *native* dispatch site
        ``expressions.rs::dispatch_symbolic_variable_inspect`` end to end.

        ``test_dispatch_symbolic_variable_fires_bp`` invokes
        ``mgr._cb_inspect_symbolic_variable`` directly — it proves the Python
        relay but never reaches the interpreter's fresh-symbol fallback. The
        native dispatch fires from ``mod.rs::load_from_callback`` when Python's
        ``memory_load`` callback returns ``is_symbolic=True`` with **no** AST:
        the engine mints a fresh ``mem_<addr>_<size>`` BVS and fires
        ``symbolic_variable`` BP_AFTER (gated on bit 18).

        Harness reuses the unmapped-symbolic-load shellcode spine (``mov rax,
        [rbx] ; ret`` with ``rbx`` constrained to an unmapped page) so the load
        routes ``load_symbolic_addr`` → Single → ``load_from_callback``. We
        patch ``_cb_memory_load`` *on the class* (it is captured as a bound
        method in ``_setup_callbacks`` at construction, so a post-construction
        instance override is too late — see the iter-45 gotcha) to return
        ``(bytes, True, None)`` for the unmapped target, forcing the
        no-AST fresh-symbol fallback.
        """
        import claripy

        import angr

        unmapped = 0xDEAD0000
        # mov rax, [rbx] ; ret
        proj = angr.load_shellcode(b"\x48\x8b\x03\xc3", "amd64")
        state = proj.factory.blank_state(addr=proj.entry)
        sym_addr = claripy.BVS("sym_load_addr", 64)
        state.solver.add(sym_addr == unmapped)
        state.regs.rbx = sym_addr

        orig_cb = RustExplorationManager._cb_memory_load

        def _patched_cb(self, addr, size):
            # Force the fresh-symbol fallback only for the unmapped target;
            # everything else (regs/code already in Rust) keeps real behavior.
            if addr == unmapped:
                return (bytes(size), True, None)
            return orig_cb(self, addr, size)

        RustExplorationManager._cb_memory_load = _patched_cb
        try:
            mgr = RustExplorationManager(proj, [state])
            seen = []

            def on_sv(s):
                seen.append(
                    (
                        s.inspect.symbolic_name,
                        s.inspect.symbolic_size,
                        s.inspect.symbolic_expr,
                    )
                )

            mgr._get_inspect_proxy().b("symbolic_variable", when="after", action=on_sv)
            # Registering the BP must flip bit 18 so the native gate opens.
            assert mgr._callbacks.get_inspect_enabled() & (1 << 18) != 0

            mgr.run(max_steps=1)
        finally:
            RustExplorationManager._cb_memory_load = orig_cb

        assert seen, f"native symbolic_variable BP_AFTER never fired; events={len(seen)}"
        name, size, expr = seen[0]
        # Fresh symbol minted by load_from_callback: mem_<addr>_<bytesize>.
        assert name == f"mem_{unmapped:x}_8", f"unexpected fresh-symbol name: {name}"
        assert size == 64
        assert isinstance(expr, claripy.ast.Base), f"expr not claripy AST: {type(expr)}"
        assert expr.size() == 64

    # ---- angr-xmfj: Python-dispatched simprocedure / syscall / dirty ----

    def test_python_dispatched_events_have_no_rust_slot(self):
        """simprocedure/syscall/dirty are dispatched from Python — they
        intentionally do NOT expose a `set_inspect_<event>` PyO3 slot,
        since the Rust engine never invokes them."""
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS,
        )

        cbs = PythonCallbacks()
        assert (
            frozenset(
                {
                    # `constraints` left this set in angr-op0dn.14.4.1 (it also
                    # fires natively from the fork-guard add) and `vex_lift` in
                    # angr-op0dn.14.4.2 (it also fires from the native libVEX
                    # lift), so both now have a `set_inspect_<event>` slot.
                    "simprocedure",
                    "syscall",
                    "dirty",
                }
            )
            == _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS
        )
        for evt in _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS:
            assert not hasattr(cbs, f"set_inspect_{evt}"), f"unexpected Rust slot for python-dispatched event {evt!r}"

    def test_simprocedure_syscall_dirty_bits_match_spec(self, fauxware_project):
        """Registering a `simprocedure`/`syscall`/`dirty` BP flips the
        expected bits (10/11/12). The Rust engine does not read these
        bits, but the bitmask stays consistent with the spec."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("simprocedure", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 10) != 0
        ins.b("syscall", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 11) != 0
        ins.b("dirty", when="before", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 12) != 0

    def test_dispatch_simprocedure_fires_bp(self, fauxware_project):
        """_cb_inspect_simprocedure invokes the user's BP with attrs."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_sp(s):
            seen.append(
                (
                    s.inspect.simprocedure_name,
                    s.inspect.simprocedure_addr,
                    s.inspect.simprocedure,
                    s.inspect.simprocedure_result,
                )
            )

        mgr._get_inspect_proxy().b("simprocedure", when="before", action=on_sp)
        sentinel = object()
        mgr._cb_inspect_simprocedure(sid, "before", "malloc", 0x401000, sentinel, None)
        assert len(seen) == 1
        name, addr, inst, result = seen[0]
        assert name == "malloc"
        assert addr == 0x401000
        assert inst is sentinel
        assert result is None

    def test_dispatch_syscall_fires_bp(self, fauxware_project):
        """_cb_inspect_syscall invokes the user's BP with syscall_name."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_sc(s):
            seen.append((s.inspect.syscall_name, s.inspect.simprocedure))

        mgr._get_inspect_proxy().b("syscall", when="after", action=on_sc)
        sentinel = object()
        mgr._cb_inspect_syscall(sid, "after", "read", sentinel)
        assert len(seen) == 1
        sc_name, inst = seen[0]
        assert sc_name == "read"
        assert inst is sentinel

    def test_dispatch_dirty_fires_bp(self, fauxware_project):
        """_cb_inspect_dirty invokes the user's BP with all four attrs."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        seen = []

        def on_dirty(s):
            seen.append(
                (
                    s.inspect.dirty_name,
                    s.inspect.dirty_handler,
                    s.inspect.dirty_args,
                    s.inspect.dirty_result,
                )
            )

        mgr._get_inspect_proxy().b("dirty", when="after", action=on_dirty)
        handler = lambda *a: None  # noqa: E731
        args = [claripy.BVV(0x10, 64), claripy.BVV(0x20, 64)]
        result = claripy.BVV(0xDEADBEEF, 64)
        mgr._cb_inspect_dirty(sid, "after", "amd64g_dirtyhelper_RDTSC", handler, args, result)
        assert len(seen) == 1
        name, hand, ar, res = seen[0]
        assert name == "amd64g_dirtyhelper_RDTSC"
        assert hand is handler
        assert ar == args
        assert res is result

    def test_dirty_result_override_returned(self, fauxware_project):
        """A BP that sets state.inspect.dirty_result surfaces the override
        as _cb_inspect_dirty's return value (short-circuit — angr-uy32)."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0xDEADBEEF, 64)
        injected = claripy.BVV(0xFEEDFACE, 64)

        def on_dirty(s):
            s.inspect.dirty_result = injected

        mgr._get_inspect_proxy().b("dirty", when="after", action=on_dirty)
        handler = lambda *a: None  # noqa: E731
        args = [claripy.BVV(0x10, 64)]
        ret = mgr._cb_inspect_dirty(sid, "after", "amd64g_dirtyhelper_RDTSC", handler, args, original)
        assert ret is injected

    def test_dirty_result_unchanged_returns_none(self, fauxware_project):
        """A dirty BP that does not touch dirty_result returns None so the
        handler's original return value stands."""
        import claripy

        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        original = claripy.BVV(0xDEADBEEF, 64)

        def on_dirty(s):
            _ = s.inspect.dirty_result  # read only

        mgr._get_inspect_proxy().b("dirty", when="after", action=on_dirty)
        handler = lambda *a: None  # noqa: E731
        ret = mgr._cb_inspect_dirty(sid, "after", "rdtsc", handler, [], original)
        assert ret is None

    def test_simprocedure_fires_during_exploration(self, fauxware_project):
        """simprocedure BP fires while running fauxware (libc procs hit).

        The BP is dispatched from the Python SimProcedure bounce, and stock
        fauxware bounces nothing at all now that open/read are served natively
        (angr-gorvf.15) — so a Python-only proc is hooked over ``puts`` to
        provide the bounce this test is about.
        """

        import angr as _angr

        class _PyOnlyPuts(_angr.SimProcedure):
            def run(self, _s):  # pylint: disable=arguments-differ
                return 0

        fauxware_project.hook_symbol("puts", _PyOnlyPuts(), replace=True)
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        names = []

        def on_sp(s):
            names.append(s.inspect.simprocedure_name)

        mgr._get_inspect_proxy().b("simprocedure", when="before", action=on_sp)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 10) != 0
        mgr.run(max_steps=30)
        # fauxware hits __libc_start_main, puts, read, strcmp, etc. very
        # early — at least one before-BP should have fired.
        assert names, "no simprocedure events captured during exploration"
        assert all(isinstance(n, str) for n in names)

    # ---- angr-ysml: fork (deferred-fork dispatch in stepping.rs) ----

    def test_fork_callback_slot_exposed(self):
        """PythonCallbacks gained set_inspect_fork / call_inspect_fork."""
        cbs = PythonCallbacks()
        for name in ("set_inspect_fork", "call_inspect_fork"):
            assert hasattr(cbs, name), f"PythonCallbacks missing {name}"

    def test_fork_bit_matches_spec(self, fauxware_project):
        """Registering a `fork` BP flips bit 4 (previously reserved)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        assert mgr._callbacks.get_inspect_enabled() == 0
        ins.b("fork", when="after", action=lambda s: None)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 4) != 0

    def test_dispatch_fork_fires_bp(self, fauxware_project):
        """_cb_inspect_fork invokes the user's BP (no attrs — fork has none)."""
        mgr, sid, _ = self._make_mgr_with_state_id(fauxware_project)
        fired = []

        def on_fork(s):
            # state is a RustStateProxy bound to the forked state id;
            # `state.inspect` exposes no fork-specific attrs (angr's
            # inspect_attributes table has none for fork).
            fired.append(s)

        mgr._get_inspect_proxy().b("fork", when="after", action=on_fork)
        mgr._cb_inspect_fork(sid, "after")
        assert len(fired) == 1

    def test_fork_fires_during_exploration(self, fauxware_project):
        """fork BP fires for each branch fauxware takes during exploration.

        fauxware has a symbolic input-driven branch (good_password vs.
        backdoor); the deferred-fork machinery in stepping.rs creates a
        sibling state per resolved branch, and the dispatch must fire on
        each one.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        fork_count = [0]

        def on_fork(s):
            fork_count[0] += 1

        mgr._get_inspect_proxy().b("fork", when="after", action=on_fork)
        assert mgr._callbacks.get_inspect_enabled() & (1 << 4) != 0
        mgr.run(max_steps=40)
        # fauxware has at least one symbolic branch within 40 steps;
        # the deferred-fork loop should fire the BP at least once.
        assert fork_count[0] > 0, "no fork events captured during exploration"


class TestRustInspectAllowlistConsistency:
    """CI-style guard tests that the inspect dispatch allowlist is a single
    source of truth, and that every event angr's SimInspector exposes
    is either honored (with dispatch wired) or loudly rejected.

    angr-ji7h: prevents silent pass-through when a future angr release
    adds a new state.inspect event.
    """

    def test_event_specs_keys_match_supported_set(self):
        """Derived views agree with the source-of-truth keys."""
        from angr.exploration.rust_state_proxy import (
            _INSPECT_EVENT_SPECS,
            _RUST_INSPECT_ATTRS_BY_EVENT,
            _RUST_INSPECT_EVENT_BITS,
            _RUST_INSPECT_SUPPORTED_EVENTS,
        )

        assert frozenset(_INSPECT_EVENT_SPECS) == _RUST_INSPECT_SUPPORTED_EVENTS
        assert set(_RUST_INSPECT_ATTRS_BY_EVENT) == set(_INSPECT_EVENT_SPECS)
        assert set(_RUST_INSPECT_EVENT_BITS) == set(_INSPECT_EVENT_SPECS)

    def test_event_bits_are_unique_and_in_range(self):
        """Every supported event has a unique bit position fitting in u32."""
        from angr.exploration.rust_state_proxy import _INSPECT_EVENT_SPECS

        bits = [spec["bit"] for spec in _INSPECT_EVENT_SPECS.values()]
        assert len(bits) == len(set(bits)), f"duplicate bits in specs: {bits}"
        # angr-4ai9 widened inspect_enabled from u8 to u16 to make room for
        # call/return (8/9); angr-lge2 widened from u16 to u32 to make room
        # for `expr` (bit 16) after `statement` filled bit 15.
        assert all(0 <= b < 32 for b in bits), "inspect_enabled is a u32 — bits must be in 0..=31"

    def test_every_supported_event_has_dispatch_method(self):
        """A supported event without a `_cb_inspect_<event>` method on
        RustExplorationManager would silently be enabled in the bitmask
        but never fire — assert one exists per event."""
        from angr.exploration.rust_state_proxy import _INSPECT_EVENT_SPECS

        for evt in _INSPECT_EVENT_SPECS:
            attr = f"_cb_inspect_{evt}"
            assert hasattr(RustExplorationManager, attr), (
                f"event {evt!r} is in _INSPECT_EVENT_SPECS but "
                f"RustExplorationManager.{attr} is missing — adding to "
                f"the allowlist without wiring dispatch produces a "
                f"silent pass-through"
            )

    def test_every_supported_event_has_rust_callback_slot(self):
        """PythonCallbacks must expose a `set_inspect_<event>` slot for
        every honored Rust-dispatched event; otherwise the dispatcher
        would never run even with a BP registered.

        Python-dispatched events (`dispatch_origin: 'python'`) are
        skipped because they fire from existing Python callback handlers
        in rust_callback_dispatch.py / rust_manager._cb_dirty_call and
        never traverse the PythonCallbacks slot.
        """
        from angr.exploration.rust_state_proxy import (
            _INSPECT_EVENT_SPECS,
            _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS,
        )

        cbs = PythonCallbacks()
        for evt in _INSPECT_EVENT_SPECS:
            if evt in _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS:
                continue
            slot = f"set_inspect_{evt}"
            assert hasattr(cbs, slot), (
                f"event {evt!r} is in _INSPECT_EVENT_SPECS but "
                f"PythonCallbacks.{slot} is missing — Rust-side "
                f"dispatch is not wired"
            )

    def test_every_angr_event_is_honored_or_rejected(self, fauxware_project):
        """The full safety guarantee: for every event angr exposes,
        either the Rust engine dispatches it (in _INSPECT_EVENT_SPECS),
        or registering a BP for it raises NotImplementedError. No silent
        accepts."""
        from angr.exploration.rust_state_proxy import _RUST_INSPECT_SUPPORTED_EVENTS
        from angr.state_plugins.inspect import EventType

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ins = mgr._get_inspect_proxy()

        unhandled = []
        for evt in EventType:
            if evt in _RUST_INSPECT_SUPPORTED_EVENTS:
                continue
            try:
                ins.b(evt, when="before", action=lambda s: None)
            except NotImplementedError:
                continue
            unhandled.append(evt)

        assert not unhandled, (
            f"event(s) {unhandled} are not in the supported set yet "
            f"registering them did NOT raise NotImplementedError — "
            f"this is a silent pass-through. Add them to "
            f"_INSPECT_EVENT_SPECS (with dispatch wired) or tighten "
            f"the check in RustInspectProxy._check_event."
        )

    def test_unsupported_event_message_lists_supported_set(self):
        """The NotImplementedError message must list every currently
        supported event so users know what they CAN use."""
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_SUPPORTED_EVENTS,
            _format_unsupported_event_msg,
        )

        msg = _format_unsupported_event_msg("call")
        for evt in _RUST_INSPECT_SUPPORTED_EVENTS:
            assert evt in msg, (
                f"supported event {evt!r} missing from rejection message — would mislead users about what is available"
            )

    def test_manager_breakpoint_storage_matches_specs(self, fauxware_project):
        """The manager-wide BP registry uses exactly the supported event
        keys. Adding a key here without a corresponding spec entry, or
        vice versa, breaks _update_inspect_bitmask."""
        from angr.exploration.rust_state_proxy import _RUST_INSPECT_SUPPORTED_EVENTS

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        assert set(mgr._inspect_breakpoints) == set(_RUST_INSPECT_SUPPORTED_EVENTS)
        assert set(mgr._INSPECT_EVENT_BITS) == set(_RUST_INSPECT_SUPPORTED_EVENTS)

    @staticmethod
    def _parse_doc_supported_events():
        """Return the set of event names listed in the "Supported events"
        grid table of ``docs/advanced-topics/rust_engine.rst``.

        The table is an RST simple table; each event row begins in column 0
        with a ``\\`\\`name\\`\\``` token, while BP-attribute continuation
        lines are indented. Extract the first backtick token of every
        non-indented, non-delimiter line inside the table body.
        """
        import pathlib

        rst = pathlib.Path(__file__).resolve().parents[3] / "docs" / "advanced-topics" / "rust_engine.rst"
        lines = rst.read_text().splitlines()

        # Find the "Supported events" section, then the two `====` delimiter
        # lines that bracket the table body.
        try:
            hdr = next(i for i, ln in enumerate(lines) if ln.strip() == "Supported events")
        except StopIteration:  # pragma: no cover - doc structure changed
            raise AssertionError("'Supported events' section missing from rust_engine.rst")

        delims = [
            i for i in range(hdr, len(lines)) if re.match(r"^=====+\s", lines[i]) or re.match(r"^=====+$", lines[i])
        ]
        assert len(delims) >= 3, "expected an RST simple table under 'Supported events' (3 '====' lines)"
        body = lines[delims[1] + 1 : delims[2]]

        events: set[str] = set()
        for ln in body:
            if not ln or ln[0].isspace():
                continue  # indented continuation line
            m = re.match(r"^``([a-z_]+)``", ln)
            if m:
                events.add(m.group(1))
        return events

    def test_doc_supported_events_table_matches_code(self):
        """The "Supported events" table in rust_engine.rst is the declared
        single source of truth for which inspect events the Rust engine
        dispatches. Assert it matches ``_RUST_INSPECT_SUPPORTED_EVENTS``
        exactly — a drift in either direction (a newly wired event missing
        from the doc, or a stale doc row for a removed event) means a user
        reading the doc would predict the wrong behavior. angr-c4xcs.2;
        this mirrors ``TestRustSimOptionMatrixConsistency`` in test_misc.py
        for the inspect-event half of rust_engine.rst."""
        from angr.exploration.rust_state_proxy import _RUST_INSPECT_SUPPORTED_EVENTS

        doc = self._parse_doc_supported_events()
        code = set(_RUST_INSPECT_SUPPORTED_EVENTS)
        assert doc == code, (
            "state.inspect 'Supported events' table disagrees with "
            "_RUST_INSPECT_SUPPORTED_EVENTS.\n"
            f"  doc-only:  {sorted(doc - code)}\n"
            f"  code-only: {sorted(code - doc)}"
        )


class TestStatePluginsProxy:
    """Tests for state.options, state.globals, and state.heap on RustStateProxy.

    Regression for angr-t3l3: prior implementation returned an empty set/dict
    that silently dropped writes (e.g., state.options.add(LAZY_SOLVES)).
    state.heap.mmap_base was completely unsupported via the proxy.
    """

    def test_options_seeded_from_source_state(self, fauxware_project):
        """options copied from the SimState passed to RustExplorationManager
        are visible through the proxy."""
        from angr import sim_options as o

        state = fauxware_project.factory.entry_state()
        state.options.add(o.LAZY_SOLVES)
        mgr = RustExplorationManager(fauxware_project, [state])

        # First active state's proxy should show LAZY_SOLVES.
        proxies = mgr.proxy.active
        assert len(proxies) >= 1
        assert o.LAZY_SOLVES in proxies[0].options

    def test_options_writes_persist(self, fauxware_project):
        """state.options.add(X) on a proxy persists for the same state_id.

        The Rust engine doesn't honor most SimOptions, but Python user code
        treats state.options as a live set; writes must round-trip.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        proxy.options.add("MY_FLAG")
        # Re-fetch the proxy — should see the same underlying set.
        proxy2 = mgr.proxy.active[0]
        assert "MY_FLAG" in proxy2.options

    def test_globals_seeded_and_mutable(self, fauxware_project):
        """state.globals copied from the source SimState; writes persist."""

        state = fauxware_project.factory.entry_state()
        state.globals["init_key"] = "init_value"
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        assert proxy.globals.get("init_key") == "init_value"
        proxy.globals["new_key"] = 42
        # Re-fetch
        proxy2 = mgr.proxy.active[0]
        assert proxy2.globals.get("new_key") == 42

    def test_options_inherited_by_forked_states(self, fauxware_project):
        """Children forked in Rust inherit options from their root state on
        first access."""
        from angr import sim_options as o

        state = fauxware_project.factory.entry_state()
        state.options.add(o.LAZY_SOLVES)
        mgr = RustExplorationManager(fauxware_project, [state])

        # Step a few times to trigger forks.
        mgr.run(max_steps=20)

        # Every state in any stash should see LAZY_SOLVES via parent walk.
        for stash in ("active", "deadended", "found"):
            for sid in mgr._rust_mgr.get_state_ids(stash):
                opts = mgr.get_state_options_py(sid)
                assert o.LAZY_SOLVES in opts, f"state {sid} in {stash} missing inherited LAZY_SOLVES"

    def test_heap_mmap_base_exposed(self, fauxware_project):
        """state.heap.mmap_base reads/writes through to the Rust state."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        # Default mmap_base is set on RustSimState construction (0xC100_0000).
        assert proxy.heap.mmap_base == 0xC100_0000
        # Writes round-trip.
        proxy.heap.mmap_base = 0xD000_0000
        proxy2 = mgr.proxy.active[0]
        assert proxy2.heap.mmap_base == 0xD000_0000

    def test_heap_allocations_and_freed_lists(self, fauxware_project):
        """state.heap.allocations / .freed return lists of (addr, size) and
        addresses respectively. Empty for a fresh state."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        assert proxy.heap.allocations == []
        assert proxy.heap.freed == []

    def test_options_globals_fallback_without_python_mgr(self):
        """When constructed without python_mgr, options/globals fall back to
        empty stand-ins (low-level unit-test path)."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        proxy = RustStateProxy(mgr, sid)
        # No python_mgr -> empty fallback (does not raise).
        assert proxy.options == set()
        assert proxy.globals == {}

    def test_scratch_bbl_addr_mirrors_pc(self, fauxware_project):
        """proxy.scratch.bbl_addr matches state.pc (most recently entered
        block)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]
        assert proxy.scratch.bbl_addr == proxy.addr
        assert proxy.scratch.ins_addr == proxy.addr

    def test_scratch_jumpkind_after_step(self, fauxware_project):
        """proxy.scratch.jumpkind reflects the last detailed-history entry."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Step a few blocks so detailed history accumulates.
        mgr.run(max_steps=5)
        for proxy in mgr.proxy.active:
            jk = proxy.scratch.jumpkind
            # Either no transitions yet (None) or a known Ijk_*.
            assert jk is None or jk.startswith("Ijk_"), f"unexpected jumpkind {jk!r}"

    def test_scratch_unsupported_attrs_are_none(self, fauxware_project):
        """SimStateScratch attributes the Rust engine doesn't persist
        (irsb, stmt_idx, tyenv, sim_procedure) read as None / empty."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]
        s = proxy.scratch
        assert s.irsb is None
        assert s.stmt_idx is None
        assert s.tyenv is None
        assert s.sim_procedure is None
        assert s.temps == []

    def test_scratch_proxy_cached_on_state_proxy(self, fauxware_project):
        """proxy.scratch returns the same RustScratchProxy on repeated reads
        (consistent with other lazy sub-proxies)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]
        assert proxy.scratch is proxy.scratch

    def test_scratch_jumpkind_empty_history(self):
        """A state with no recorded transitions reports jumpkind=None."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        proxy = RustStateProxy(mgr, sid)
        # Brand-new state — no history yet.
        assert proxy.scratch.jumpkind is None
        # bbl_addr defaults to 0 (initial pc); ins_addr mirrors it.
        assert proxy.scratch.bbl_addr == 0
        assert proxy.scratch.ins_addr == 0


class TestRegisterProxySymbolicRecovery:
    """angr-4pm1: RustRegisterProxy must recover Rust's claripy AST for symbolic
    registers via ``get_state_register_ast`` instead of minting an orphan
    ``claripy.BVS``.

    The orphan BVS had no identity link to Rust's symbol: constraints added
    through ``proxy.solver.add(proxy.regs.<sym_reg> == K)`` silently missed
    because the forked solver saw a constraint over a ghost symbol Y, while
    Rust's register still held the original X. The fix asks Rust for the AST
    of the underlying ``RustBV`` so the returned AST is the same Python object
    used for both constraint construction and evaluation.
    """

    @classmethod
    def setup_class(cls):
        """Initialize the Python/Rust shared Z3 context so claripy's z3
        backend and Rust's z3-rs binding hash-cons against the same Z3
        instance. Without this, an AST pointer obtained on the Python side
        will not match the ``z3::ast::BV::new_const`` that Rust creates on
        ``claripy_to_rustbv``, and constraints added in the proxy.solver fork
        end up over a phantom symbol — exactly the failure mode angr-4pm1 is
        meant to fix.
        """
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    @staticmethod
    def _setup_symbolic_rax(mgr):
        """Build a state with RAX bound to a fresh symbolic claripy AST,
        registered through ``set_state_register_symbolic_ast`` so the symbol
        is properly cached for FFI round-trip. Returns ``(state_id, sym)``
        where ``sym`` is the originating claripy BVS."""
        import claripy

        sym = claripy.BVS("sym_rax_proxy_4pm1", 64)
        sid = mgr.create_state("active")
        mgr.set_state_register_symbolic_ast(sid, "rax", sym)
        return sid, sym

    def test_symbolic_register_returns_ast_not_orphan(self):
        """proxy.regs.<sym_reg> returns the AST recovered from Rust — must
        report itself as symbolic and live in the proxy's cache (consistent
        repeated reads)."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid, _ = self._setup_symbolic_rax(mgr)
        proxy = RustStateProxy(mgr, sid)

        ast = proxy.regs.rax
        assert ast is not None
        assert hasattr(ast, "symbolic") and ast.symbolic, f"expected symbolic AST for rax, got {ast!r}"
        # Cached: the proxy must hand back the SAME Python object on
        # repeated reads. The constraint-add path relies on this — if the
        # second read returned a different BVS, the constraint added on the
        # first AST would not narrow the second.
        assert proxy.regs.rax is proxy.regs.rax

    def test_symbolic_register_constraint_round_trip(self):
        """proxy.solver.add(proxy.regs.<sym_reg> == K) followed by
        proxy.solver.eval(proxy.regs.<sym_reg>) returns K.

        Pre-fix: proxy.regs.rax was an orphan BVS Y; the constraint Y==K hit
        a ghost symbol in the forked solver, so eval(Y) could return anything
        the solver decided. Post-fix: Y is Rust's actual AST, so the
        constraint narrows the solver to K.
        """
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid, _ = self._setup_symbolic_rax(mgr)
        proxy = RustStateProxy(mgr, sid)

        rax = proxy.regs.rax
        proxy.solver.add(rax == 0x41)
        # Re-read via the proxy AND directly — both must report 0x41 since
        # the fork's constraint hits the symbol the proxy is caching.
        assert proxy.solver.eval(rax) == 0x41
        assert proxy.solver.eval(proxy.regs.rax) == 0x41

    def test_symbolic_register_prefetch_recovers_ast(self):
        """Prefetch path mirrors __getattr__: a register Rust reports as
        symbolic must land in the cache as an AST, not an orphan."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid, _ = self._setup_symbolic_rax(mgr)
        proxy = RustStateProxy(mgr, sid)

        proxy.regs.prefetch(["rax"])
        ast = proxy.regs.rax
        assert ast is not None
        assert hasattr(ast, "symbolic") and ast.symbolic
        # Same constraint round-trip after the prefetch warmed the cache.
        proxy.solver.add(ast == 0x42)
        assert proxy.solver.eval(ast) == 0x42

    def test_prefetch_alias_canonicalizes_no_orphan(self):
        """prefetch() with an ABI alias (e.g. "ip") must canonicalize before
        the FFI call, so alias and canonical name share one cache entry
        (angr-hv4lt.4).

        Pre-fix: prefetch(["ip"]) reached Rust verbatim, missed — the register
        file had no "ip" entry at all back then — and cached a fresh orphan BVS
        under "ip"; a subsequent proxy.regs.ip returned that ghost symbol
        instead of the real instruction pointer, so constraints never touched
        the actual register. Rust now defines "ip" as an alias of the PC
        (angr-690nc), but canonicalization still has to collapse the two
        spellings onto one cached AST.
        """
        import claripy

        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        # Seed a concrete PC via the canonical name.
        mgr.set_state_register_symbolic_ast(sid, "rip", claripy.BVV(0xDEADBEEF, 64))
        proxy = RustStateProxy(mgr, sid)

        proxy.regs.prefetch(["ip"])
        # Read via the alias: must be the real concrete PC, not an orphan BVS.
        ip = proxy.regs.ip
        assert ip is not None
        assert not (hasattr(ip, "symbolic") and ip.symbolic), f"expected concrete PC, got orphan {ip!r}"
        assert proxy.solver.eval(ip) == 0xDEADBEEF
        # Reading via the canonical name hits the dual-cached entry.
        assert proxy.solver.eval(proxy.regs.rip) == 0xDEADBEEF


class TestStateProxyRepr:
    """Tests for the enriched RustStateProxy.__repr__ (angr-4c20).

    repr should expose stash membership and constraint count so that
    `print(mgr.proxy.active[0])` returns something actionable in the REPL.
    """

    def test_repr_includes_stash_and_constraints(self, fauxware_project):

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        text = repr(proxy)
        assert text.startswith("<RustStateProxy ")
        assert f"id={proxy.state_id}" in text
        assert f"addr={hex(proxy.addr)}" in text
        assert "stash=active" in text
        assert "constraints=0" in text

    def test_repr_reflects_seeded_constraint(self, fauxware_project):
        """A state seeded into the manager with N constraints already attached
        should show constraints=N through repr."""
        import claripy

        state = fauxware_project.factory.entry_state()
        x = claripy.BVS("repr_test_x", 32)
        state.solver.add(x > 5)
        state.solver.add(x < 100)
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        text = repr(proxy)
        # constraint_count is read off the SymContext atomic and reflects
        # whatever was loaded during state seeding.
        match = re.search(r"constraints=(\d+)", text)
        assert match is not None, text
        assert int(match.group(1)) >= 2, text

    def test_repr_unknown_state_falls_back(self):
        """When the state_id has been GC'd (no longer in any stash and not
        the pending callback), repr still produces something readable rather
        than raising — stash/constraints fields just disappear."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        # Use the proxy before clearing the state, then drop the state.
        proxy = RustStateProxy(mgr, sid)
        # Capture the addr before clearing — proxy.addr does its own FFI lookup.
        baseline_repr = repr(proxy)
        assert f"id={sid}" in baseline_repr
        assert "stash=active" in baseline_repr

        # Clear the active stash. State is gone — stash/constraints disappear,
        # but the addr lookup uses cached value from the original creation
        # path, so repr still returns a valid string.
        mgr.clear_stash("active")
        text = repr(proxy)
        assert text.startswith("<RustStateProxy ")
        assert f"id={sid}" in text
        # Stash and constraints are absent now since the state is gone.
        assert "stash=" not in text
        assert "constraints=" not in text


class TestStateProxyCopySemantics:
    """``RustStateProxy.copy()`` is a Rust-side CoW deep fork (angr-d1dr).

    Replaces the v1.0 ``NotImplementedError`` raise with a real fork via
    ``RustSimState::fork`` + Python-side options/globals snapshot. The
    contract mirrors angr ``SimState.copy()``: mutations on the copy must
    not flow back into the source.
    """

    def test_proxy_copy_returns_independent_state(self, fauxware_project):

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()

        assert clone is not proxy
        assert clone._state_id != proxy._state_id
        assert clone.addr == proxy.addr

    def test_proxy_copy_register_write_does_not_affect_source(self, fauxware_project):
        """Writing a register on the copy leaves the source register intact."""
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()

        original_rax = proxy.regs.rax
        clone.regs.rax = claripy.BVV(0xDEADBEEF, 64)

        # Source is untouched
        assert proxy.regs.rax is original_rax or (
            proxy.regs.rax.concrete
            and original_rax.concrete
            and proxy.regs.rax.concrete_value == original_rax.concrete_value
        )
        # Copy got the new value
        assert clone.regs.rax.concrete
        assert clone.regs.rax.concrete_value == 0xDEADBEEF

    def test_proxy_copy_constraint_isolation(self, fauxware_project):
        """A constraint pushed into the clone's underlying state solver must
        not appear on the source's solver. Uses
        ``add_constraints_to_state`` (the path that actually persists onto
        the Rust state's solver) rather than ``proxy.add_constraints``
        (which routes through a per-proxy forked solver context and never
        touches the state's solver).
        """
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()

        before_src = mgr._rust_mgr.state_constraint_count(proxy._state_id)
        before_clone = mgr._rust_mgr.state_constraint_count(clone._state_id)

        x = claripy.BVS("d1dr_isolation", 64)
        mgr._rust_mgr.add_constraints_to_state(clone._state_id, [x == 7])

        after_src = mgr._rust_mgr.state_constraint_count(proxy._state_id)
        after_clone = mgr._rust_mgr.state_constraint_count(clone._state_id)

        assert after_src == before_src
        assert after_clone == before_clone + 1

    def test_proxy_add_constraints_visible_across_sub_proxies(self, fauxware_project):
        """angr-yodz: a constraint added via ``proxy.add_constraints`` is
        visible to the SAME proxy's posix.dumps(0) and solver reads, because
        all sub-proxies share one forked context. It must remain view-local
        — the underlying Rust state's solver is untouched.
        """
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        x = claripy.BVS("yodz_stdin", 8)
        # Wire a fake stdin var BEFORE first .posix access so the lazily-built
        # posix sub-proxy picks it up.
        proxy._stdin_vars = [(x, 0)]

        before = mgr._rust_mgr.state_constraint_count(proxy._state_id)
        proxy.add_constraints(x == 0x41)  # 'A'

        # posix.dumps(0) evaluates stdin vars under the shared context, so it
        # must honor the just-added constraint.
        assert proxy.posix.dumps(0) == b"A"
        # The solver sub-proxy reads the same value.
        assert proxy.solver.eval(x) == 0x41
        # View-local: the underlying state's solver gained no constraint.
        assert mgr._rust_mgr.state_constraint_count(proxy._state_id) == before

    def test_proxy_add_constraints_not_persistent_to_state(self, fauxware_project):
        """angr-yodz: add_constraints is view-local. A freshly built proxy
        for the same state does not see a constraint added through a prior
        proxy (it forks a clean context from the unchanged state solver).
        """
        import claripy

        from angr.exploration.rust_state_proxy import RustStateProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        x = claripy.BVS("yodz_viewlocal", 8)
        proxy.add_constraints(x == 0x41)

        fresh = RustStateProxy(mgr._rust_mgr, proxy._state_id, project=fauxware_project)
        # x is unconstrained from the fresh proxy's perspective: both 0x41 and
        # a different value are valid solutions.
        assert fresh.solver.satisfiable(extra_constraints=[x == 0x42])

    def test_proxy_copy_options_isolation(self, fauxware_project):
        """Mutating the copy's options set does not mutate the source's."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        # Seed a sentinel on the source first so the copy has something to
        # inherit from.
        proxy.options.add("d1dr_inherited")
        clone = proxy.copy()

        assert "d1dr_inherited" in clone.options

        clone.options.add("d1dr_clone_only")
        assert "d1dr_clone_only" in clone.options
        assert "d1dr_clone_only" not in proxy.options

    def test_proxy_copy_globals_isolation(self, fauxware_project):
        """Mutating the copy's globals dict does not mutate the source's."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        proxy.globals["d1dr_inherited"] = "before-copy"
        clone = proxy.copy()

        assert clone.globals.get("d1dr_inherited") == "before-copy"

        clone.globals["d1dr_clone_only"] = 42
        assert proxy.globals.get("d1dr_clone_only") is None

    def test_proxy_copy_lands_in_copies_stash(self, fauxware_project):
        """The cloned state is placed in the dedicated ``_copies`` stash so
        ``step()`` won't auto-advance it."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()

        assert mgr._rust_mgr.state_stash(clone._state_id) == "_copies"
        # And the active stash is unchanged.
        active_ids = mgr._rust_mgr.get_state_ids("active")
        assert clone._state_id not in active_ids
        assert proxy._state_id in active_ids

    def test_proxy_copy_without_manager_raises(self):
        """Constructing a proxy bypassing the high-level manager (low-level
        unit-test path) leaves nowhere to store options/globals snapshots;
        ``copy()`` must surface that explicitly rather than silently
        skipping the metadata mirror."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        proxy = RustStateProxy(mgr, sid)  # no python_mgr=

        with pytest.raises(NotImplementedError) as excinfo:
            proxy.copy()
        assert "python_mgr" in str(excinfo.value)

    def test_drop_copy_removes_state_from_copies_stash(self, fauxware_project):
        """``RustExplorationManager.drop_copy`` removes the state from the
        ``_copies`` stash and frees per-state metadata (angr-yhe0)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()
        clone_id = clone._state_id

        # Seed metadata so we can verify the manager cleared it.
        clone.options.add("yhe0_marker")
        clone.globals["yhe0_marker"] = 1

        before = mgr._rust_mgr.stash_count("_copies")
        assert before >= 1
        assert mgr._rust_mgr.state_stash(clone_id) == "_copies"

        assert mgr.drop_copy(clone_id) is True

        # State is gone from _copies and the index.
        assert mgr._rust_mgr.stash_count("_copies") == before - 1
        assert mgr._rust_mgr.state_stash(clone_id) is None
        # Python-side metadata cleared.
        assert clone_id not in mgr._py_state_options
        assert clone_id not in mgr._py_state_globals

        # Second drop is a no-op (idempotent).
        assert mgr.drop_copy(clone_id) is False

    def test_drop_copy_refuses_non_copies_state(self, fauxware_project):
        """``drop_copy`` only removes states that are *currently* in
        ``_copies`` — it must refuse to delete the active root state, even
        if the caller passes its id."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        # Active state id passed to drop_copy must NOT be honored.
        assert mgr.drop_copy(proxy._state_id) is False
        # And the active state is still there.
        assert mgr._rust_mgr.state_stash(proxy._state_id) == "active"

    def test_proxy_copy_del_drops_state_from_copies(self, fauxware_project):
        """When the copy-proxy is GC'd, its ``__del__`` must drop the
        underlying ``_copies`` entry so long-running spilling workflows
        don't leak (angr-yhe0)."""
        import gc

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        clone = proxy.copy()
        clone_id = clone._state_id
        assert clone._owns_copy is True

        baseline = mgr._rust_mgr.stash_count("_copies")
        assert baseline >= 1

        # Drop the only reference and force a GC cycle.
        del clone
        gc.collect()

        assert mgr._rust_mgr.stash_count("_copies") == baseline - 1
        assert mgr._rust_mgr.state_stash(clone_id) is None

    def test_proxy_for_active_state_does_not_drop_on_del(self, fauxware_project):
        """Proxies for live exploration states (not copies) must never
        auto-drop the underlying Rust state — only ``copy()``-minted proxies
        do."""
        import gc

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        assert proxy._owns_copy is False
        active_id = proxy._state_id

        del proxy
        gc.collect()

        # Active state is still there.
        assert mgr._rust_mgr.state_stash(active_id) == "active"

    def test_proxy_copy_chain_gc_releases_all_copies(self, fauxware_project):
        """A chain of N copies — when all proxies are dropped, the
        ``_copies`` stash returns to its pre-fork count. Validates the
        full Spiller-style workload pattern (angr-yhe0)."""
        import gc

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        baseline = mgr._rust_mgr.stash_count("_copies")

        clones = [proxy.copy() for _ in range(5)]
        assert mgr._rust_mgr.stash_count("_copies") == baseline + 5

        del clones
        gc.collect()

        assert mgr._rust_mgr.stash_count("_copies") == baseline
