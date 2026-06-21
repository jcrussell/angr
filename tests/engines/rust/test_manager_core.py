"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import pytest

import angr

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


class TestRustExplorationManagerUnit:
    """Unit tests for RustExplorationManager Rust class."""

    def test_creation(self):
        """Test creating an exploration manager."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.arch == "amd64"
        assert mgr.step_count == 0
        assert mgr.active_count() == 0
        assert mgr.found_count() == 0

    def test_create_state(self):
        """Test creating states in stashes."""
        mgr = _RustExplorationManager("amd64")

        # Create a state
        state_id = mgr.create_state("active")
        assert isinstance(state_id, int)  # create_state returns a state id
        assert mgr.active_count() == 1

        # Create another state
        mgr.create_state("active")
        # Note: Both states might have ID 0 since create_state creates fresh states
        assert mgr.active_count() == 2

    def test_stash_counts(self):
        """Test stash count reporting."""
        mgr = _RustExplorationManager("amd64")

        # Create states in different stashes
        mgr.create_state("active")
        mgr.create_state("active")
        mgr.create_state("found")

        counts = mgr.stash_counts()
        assert counts["active"] == 2
        assert counts["found"] == 1

    def test_find_avoid_addresses(self):
        """Test setting find/avoid addresses."""
        mgr = _RustExplorationManager("amd64")

        # Set find addresses
        mgr.set_find_addrs([0x1000, 0x2000, 0x3000])

        # Set avoid addresses
        mgr.set_avoid_addrs([0x4000, 0x5000])

        stats = mgr.stats()
        assert stats["find_addrs"] == 3
        assert stats["avoid_addrs"] == 2

    def test_register_simprocedures(self):
        """Test registering SimProcedures."""
        mgr = _RustExplorationManager("amd64")

        # Register some procedures
        mgr.register_simprocedure(0x401000, "strlen", 1, False)
        mgr.register_simprocedure(0x401100, "malloc", 1, False)
        mgr.register_simprocedure(0x401200, "exit", 1, True)

        stats = mgr.stats()
        assert stats["simprocedures"] == 3
        assert stats["hooks"] == 3

    def test_dcas_unsupported_metric_exposed(self):
        """DCAS visibility counter is exposed via stats() and get_fallback_stats()."""
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        assert "dcas_unsupported_count" in stats
        assert stats["dcas_unsupported_count"] == 0

        fb = mgr.get_fallback_stats()
        assert "dcas_unsupported_count" in fb
        assert fb["dcas_unsupported_count"] == 0

    def test_vecret_gsptr_fallback_counter_exposed(self):
        """angr-2iow: VECRET/GSPTR Python-fallback counter is wired through
        stats() and get_fallback_stats(). Starts at zero on a fresh manager
        and the reason marker is the shared `VECRET_GSPTR_REASON` constant
        in `native/angr/src/interpreter/mod.rs`.
        """
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        assert "vecret_gsptr_fallback_count" in stats
        assert stats["vecret_gsptr_fallback_count"] == 0

        fb = mgr.get_fallback_stats()
        assert "vecret_gsptr_fallback_count" in fb
        assert fb["vecret_gsptr_fallback_count"] == 0

    def test_per_category_fallback_counters_exposed(self):
        """Per-category fallback counters (angr-md0m) appear in stats() and
        get_fallback_stats(). All start at zero on a fresh manager.
        """
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        # Manager-side counters
        assert stats["simprocedure_python_fallback_count"] == 0
        assert stats["syscall_python_fallback_count"] == 0

        # Interpreter-side counters land under the rust_ prefix once the
        # ExecutionStats hashmap is merged. Before any run, they are zero.
        exec_stats = mgr.get_execution_stats()
        assert exec_stats["python_dirty_call_count"] == 0
        assert exec_stats["python_vex_op_fallback_count"] == 0
        # Op-family breakdown
        assert exec_stats["python_vex_unop_fallback_count"] == 0
        assert exec_stats["python_vex_binop_fallback_count"] == 0
        assert exec_stats["python_vex_triop_fallback_count"] == 0
        assert exec_stats["python_vex_qop_fallback_count"] == 0
        # Silent fabricate-fresh-symbolic BYPASS visibility counter (angr-s6miz).
        assert exec_stats["vex_bypass_fabricate_count"] == 0

        fb = mgr.get_fallback_stats()
        assert fb["simprocedure_python_fallback_count"] == 0
        assert fb["syscall_python_fallback_count"] == 0

    def test_simprocedure_fallback_by_name_empty(self):
        """angr-97l8: simprocedure_fallback_by_name is an empty dict on a
        fresh manager and is exposed via both stats() and get_fallback_stats().
        """
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        assert "simprocedure_fallback_by_name" in stats
        assert stats["simprocedure_fallback_by_name"] == {}

        fb = mgr.get_fallback_stats()
        assert "simprocedure_fallback_by_name" in fb
        assert fb["simprocedure_fallback_by_name"] == {}

    def test_add_rust_state(self):
        """Test adding an existing RustSimState."""
        mgr = _RustExplorationManager("amd64")

        # Create a RustSimState
        state = RustSimState("amd64")
        state.pc = 0x1000

        # Add to manager
        mgr.add_state("active", state)
        assert mgr.active_count() == 1

        # Check PC
        pc = mgr.get_state_pc("active", 0)
        assert pc == 0x1000

    def test_state_ids(self):
        """Test getting state IDs from stash."""
        mgr = _RustExplorationManager("amd64")

        # Create states
        id1 = mgr.create_state("active")
        id2 = mgr.create_state("active")

        # Get IDs
        ids = mgr.get_state_ids("active")
        assert len(ids) == 2
        assert id1 in ids
        assert id2 in ids

    def test_set_rust_log_level(self):
        """Test setting Rust log level from Python."""
        from angr.rustylib.vex_engine import set_rust_log_level

        # Should accept all valid levels without error
        for level in ("error", "warn", "info", "debug", "trace", "off"):
            set_rust_log_level(level)
        # Invalid level should raise
        with pytest.raises(ValueError):
            set_rust_log_level("invalid")

    def test_max_active_states_get_set(self):
        """Test get/set for max_active_states limit."""
        mgr = _RustExplorationManager("amd64")

        # Default is None (unlimited)
        assert mgr.get_max_active_states() is None

        # Set a limit
        mgr.set_max_active_states(5)
        assert mgr.get_max_active_states() == 5

        # Clear the limit
        mgr.set_max_active_states(None)
        assert mgr.get_max_active_states() is None

    def test_max_active_states_enforced(self, fauxware_project):
        """max_active_states must cap the active stash *throughout* stepping,
        not just at the final count.

        create_state pushes directly to the stash and bypasses the
        enforcement path (push_to_active_or_drop), so the limit can only be
        exercised by driving real forks. With limit=1, fauxware's symbolic
        auth branch must prune excess successors, and the active stash must
        never exceed 1 at any step — gutting the enforcement to a no-op
        would let the peak climb past the cap and leave pruned at 0.
        """
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=1)

        peak_active = 0
        for _ in range(80):
            counts = mgr.stash_counts()
            peak_active = max(peak_active, counts.get("active", 0))
            if counts.get("active", 0) == 0:
                break
            mgr.step()
        counts = mgr.stash_counts()
        peak_active = max(peak_active, counts.get("active", 0))

        assert peak_active <= 1, f"active stash peaked at {peak_active}, exceeding max_active_states=1"
        assert counts.get("pruned", 0) > 0, f"enforcement never pruned a fork; counts={counts}"

    def test_register_python_procedure_appears_in_listing(self):
        """register_python_procedure adds the procedure to the registry."""
        mgr = _RustExplorationManager("amd64")
        assert not mgr.has_native_procedure("custom_widget_init")

        def widget_init(args):
            return 0

        mgr.register_python_procedure(
            "custom_widget_init",
            num_args=0,
            no_return=False,
            callable=widget_init,
        )
        assert mgr.has_native_procedure("custom_widget_init")
        assert "custom_widget_init" in mgr.list_native_procedures()

    def test_register_python_procedure_invoked_via_simprocedure_hook(self):
        """A Python-registered native procedure runs when its hook fires.

        Binds a hook address to a Python-registered native procedure, drives
        the manager, and asserts the dispatcher actually invoked the callable
        (capturing its args) and wrote the return value to RAX — not merely
        that the procedure appears in the registry.
        """
        mgr = _RustExplorationManager("amd64")

        # Track invocations from Rust into our Python procedure.
        invocations = []

        def echo_args(args):
            invocations.append(tuple(args))
            return 0xDEADBEEF

        HOOK = 0x500500
        EXIT_HOOK = 0x600500
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "echo_proc",
            num_args=2,
            no_return=False,
            callable_=echo_args,
            arg_values=(0x1111, 0x2222),  # rdi, rsi
            return_addr=EXIT_HOOK,
        )

        # Registration landed.
        assert mgr.has_native_procedure("echo_proc")
        assert "echo_proc" in mgr.list_native_procedures()

        mgr.add_state("active", state)
        mgr.run(10)

        # Dispatcher fired: the callable ran with the two SysV arg-register
        # values and its return value landed in RAX.
        assert invocations == [(0x1111, 0x2222)], f"echo_proc not invoked with expected args; got {invocations}"
        stats = mgr.native_procedure_stats()
        assert stats["native_calls"] >= 1, f"expected native_calls>=1 after dispatch, got {stats}"
        deadended = mgr.get_state_ids("deadended")
        assert len(deadended) == 1, f"expected one deadended state after exit hook; stashes={mgr.stash_counts()}"
        rax = mgr.get_state_register(deadended[0], "rax")
        assert rax == 0xDEADBEEF, f"expected RAX=0xDEADBEEF from echo_proc return; got {rax:#x}"

    def test_x86_native_procedure_returns_to_eax_not_edx(self):
        """Native procedure return value must land in EAX (offset 8), not EDX
        (offset 16) for 32-bit x86 (Cdecl). Locks the regression fixed in
        commit 5329d8222 — Cdecl previously used offset 16 (RAX in amd64) which
        routed native results to EDX in 32-bit binaries, leaving EAX with stale
        data and masking forks driven by the symbolic return value.
        """
        mgr = _RustExplorationManager("x86")

        # Minimal callbacks (lift_block won't fire — we hook every PC we visit).
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFF0000
        EAX_POISON = 0xDEADBEEF
        EDX_POISON = 0xCAFEBABE

        # Native strlen and exit are pre-registered in the procedure registry;
        # binding them to addresses makes the dispatcher invoke them on PC hit.
        # Both hook addresses are outside any loaded binary region, so the
        # `is_in_binary` gate lets native dispatch fire.
        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState("x86")

        # Map memory: 4KB string region + 4KB stack page. Permission 7 = RWX.
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)

        # "hello" + null = length 5.
        state.memory_store(STRING_ADDR, b"hello\x00")

        # Cdecl: arg passed on stack at [esp+4], return address at [esp].
        # Place return address pointing at the exit hook so the state deadends.
        # 32-bit little-endian: 4 bytes each.
        state.memory_store(STACK_BASE, EXIT_HOOK.to_bytes(4, "little"))
        state.memory_store(STACK_BASE + 4, STRING_ADDR.to_bytes(4, "little"))

        state.set_register("esp", STACK_BASE)
        state.set_register("eax", EAX_POISON)
        state.set_register("edx", EDX_POISON)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        # ~3 dispatcher iterations: strlen → ret to exit → exit deadends.
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"expected exactly one deadended state after exit hook fired; stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]

        eax = mgr.get_state_register(sid, "eax")
        edx = mgr.get_state_register(sid, "edx")

        # EAX must hold the strlen result, NOT the poison value.
        assert eax == 5, (
            f"strlen('hello') landed in wrong register: eax={eax!r} edx={edx!r}. "
            f"Expected eax=5 (Cdecl return register = EAX, offset 8). "
            f"If eax==0xdeadbeef and edx==5, Cdecl is using offset 16 (EDX) — "
            f"the angr-4pkm regression has come back."
        )
        # EDX must remain poisoned — strlen does not write it under Cdecl.
        assert edx == EDX_POISON, (
            f"EDX was clobbered: edx={edx:#x}; expected {EDX_POISON:#x}. "
            f"Native procedure return register is bleeding into the wrong slot."
        )

    # ------------------------------------------------------------------
    # Edge-case tests for register_python_procedure (angr-x9bx).
    # The PyO3 API exposes registration but invocation is dispatcher-driven,
    # so each test stands up an amd64 state at a hook PC and runs mgr.run().
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_amd64_python_proc_test(
        mgr,
        hook_addr,
        proc_name,
        num_args,
        no_return,
        callable_,
        arg_values=None,
        return_addr=0xDEADC0DE,
    ):
        """Common amd64 fixture: callbacks, hook binding, stack/regs setup.

        Registers `callable_` as a native procedure under `proc_name`, binds
        `hook_addr` to it, and produces a `RustSimState` with PC=hook_addr
        and SystemV arg registers (RDI, RSI, RDX, RCX, R8, R9) populated
        from `arg_values`. Returns the state — caller adds it to the stash.
        """
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        mgr.register_python_procedure(proc_name, num_args=num_args, no_return=no_return, callable=callable_)
        mgr.register_simprocedure(hook_addr, proc_name, num_args=num_args, no_return=no_return)

        STACK_BASE = 0x7FFF0000
        state = RustSimState("amd64")
        state.map_memory(STACK_BASE, 0x1000, 7)
        # SystemV: return addr at [rsp].
        state.memory_store(STACK_BASE, return_addr.to_bytes(8, "little"))
        state.set_register("rsp", STACK_BASE)

        SYSV_ARG_REGS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        for reg, val in zip(SYSV_ARG_REGS, arg_values or ()):
            state.set_register(reg, val)
        state.pc = hook_addr
        return state

    def test_python_procedure_symbolic_arg_falls_back_to_python(self, fauxware_project):
        """A symbolic argument must trigger the Python SimProcedure fallback,
        not invoke the registered native callable.

        extract_concrete_arg (procedures/mod.rs) returns ProcedureError::
        SymbolicArgument when the BV is symbolic; the dispatcher catches
        the Err, increments native_proc_stats.python_fallbacks, and emits
        a need_simprocedure event so Python can handle it.
        """
        import claripy

        mgr = _RustExplorationManager("amd64")
        invocations = []

        def proc(args):
            invocations.append(tuple(args))
            return 0

        HOOK = 0x500000
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "sym_proc",
            num_args=1,
            no_return=False,
            callable_=proc,
        )

        # Replace RDI with a fresh symbolic BV via the shared Z3 context.
        sym = claripy.BVS("sym_arg0", 64)
        ast_ptr = claripy.backends.z3.convert(sym).as_ast().value
        state.set_register_symbolic("rdi", ast_ptr, 64)

        mgr.add_state("active", state)
        event = mgr.run(5)

        # Native call attempted then errored → python_fallbacks bumped, event
        # emitted asking Python to take over the SimProcedure.
        stats = mgr.native_procedure_stats()
        assert stats["python_fallbacks"] >= 1, f"expected python_fallbacks>=1 after symbolic arg, got stats={stats}"
        assert stats["native_calls"] == 0
        assert invocations == [], f"python callable must NOT run for symbolic args; invocations={invocations}"
        assert event.event_type == "need_callback", (
            f"expected need_callback (SimProcedure fallback); got {event.event_type}"
        )
        assert event.callback_reason == "simprocedure"

        # angr-97l8: the by-name fallback map must record the procedure
        # under its registered name so users can see which procedures need
        # native handlers.
        mgr_stats = mgr.stats()
        by_name = mgr_stats["simprocedure_fallback_by_name"]
        assert by_name.get("sym_proc", 0) >= 1, (
            f"simprocedure_fallback_by_name should record 'sym_proc' fallback; got {by_name}"
        )
        assert sum(by_name.values()) == mgr_stats["simprocedure_python_fallback_count"], (
            f"sum of by-name counts must equal scalar fallback counter; "
            f"by_name={by_name} scalar={mgr_stats['simprocedure_python_fallback_count']}"
        )

        # angr-ilsr: SymbolicArgument fallbacks must be attributed to the
        # "symbolic" bucket, not "not_implemented" or "other". Both the
        # mgr.stats() and native_procedure_stats() views must agree.
        sym_by_name = stats["symbolic_fallbacks_by_name"]
        assert sym_by_name.get("sym_proc", 0) >= 1, (
            f"native_procedure_stats symbolic_fallbacks_by_name should record 'sym_proc'; got {sym_by_name}"
        )
        assert stats["not_implemented_fallbacks_by_name"].get("sym_proc", 0) == 0
        assert stats["other_fallbacks_by_name"].get("sym_proc", 0) == 0
        assert mgr_stats["native_proc_symbolic_fallbacks_by_name"].get("sym_proc", 0) >= 1
        # The three buckets must sum to native_proc_fallbacks.
        assert (
            stats["symbolic_fallbacks"] + stats["not_implemented_fallbacks"] + stats["other_fallbacks"]
            == stats["python_fallbacks"]
        ), (
            f"sum(symbolic+not_implemented+other) must equal python_fallbacks; "
            f"got symbolic={stats['symbolic_fallbacks']} "
            f"not_implemented={stats['not_implemented_fallbacks']} "
            f"other={stats['other_fallbacks']} total={stats['python_fallbacks']}"
        )

    def test_python_procedure_num_args_truncates_at_registered_count(self):
        """The dispatcher extracts exactly `num_args` values from the calling
        convention. Extra args sitting in unused registers (e.g. RDX when
        num_args=2) must not leak into the callable.
        """
        mgr = _RustExplorationManager("amd64")
        invocations = []

        def proc(args):
            invocations.append(tuple(args))
            return 0

        HOOK = 0x500100
        EXIT_HOOK = 0x600100
        # Two-arg procedure, but populate three arg regs.
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "two_arg_proc",
            num_args=2,
            no_return=False,
            callable_=proc,
            arg_values=(0xAAAA, 0xBBBB, 0xCCCC),  # rdi, rsi, rdx
            return_addr=EXIT_HOOK,
        )

        mgr.add_state("active", state)
        mgr.run(10)

        assert len(invocations) == 1, f"expected one invocation, got {invocations}"
        assert invocations[0] == (0xAAAA, 0xBBBB), (
            f"only the first num_args=2 values should reach the callable; "
            f"got {invocations[0]} (RDX={hex(0xCCCC)} should not appear)"
        )

    def test_python_procedure_zero_args_passes_empty_list(self):
        """num_args=0 must result in the callable receiving an empty list,
        regardless of what's sitting in the arg registers.
        """
        mgr = _RustExplorationManager("amd64")
        invocations = []

        def proc(args):
            invocations.append(tuple(args))
            return 7

        HOOK = 0x500200
        EXIT_HOOK = 0x600200
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "no_arg_proc",
            num_args=0,
            no_return=False,
            callable_=proc,
            arg_values=(0xDEAD, 0xBEEF),  # ignored
            return_addr=EXIT_HOOK,
        )

        mgr.add_state("active", state)
        mgr.run(10)

        assert invocations == [()]

    @pytest.mark.parametrize(
        "bad_return,label",
        [
            (-1, "negative int (signed → u64 extract fails)"),
            (1 << 70, "value larger than u64::MAX"),
            ("not_an_int", "non-integer return"),
            (3.14, "float return"),
        ],
    )
    def test_python_procedure_invalid_return_falls_back(self, bad_return, label):
        """Returning a value that won't fit u64 (negative, too large, or
        non-int) must surface as a Python fallback rather than corrupting
        RAX. The dispatcher's match arm catches ProcedureError::Other and
        bumps python_fallbacks.
        """
        mgr = _RustExplorationManager("amd64")
        invocations = []

        def proc(args):
            invocations.append(tuple(args))
            return bad_return

        HOOK = 0x500300
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "bad_ret_proc",
            num_args=0,
            no_return=False,
            callable_=proc,
        )

        mgr.add_state("active", state)
        event = mgr.run(5)

        # The callable did run (extract_concrete_arg passed), but the return
        # extraction failed, so the dispatcher fell back.
        assert invocations == [()], f"({label}) expected one invocation, got {invocations}"
        stats = mgr.native_procedure_stats()
        assert stats["python_fallbacks"] >= 1, f"({label}) expected python_fallbacks>=1, got stats={stats}"
        assert stats["native_calls"] == 0, f"({label}) failed return must not count as a successful native call"
        assert event.event_type == "need_callback", f"({label}) expected need_callback fallback; got {event.event_type}"

        # angr-ilsr: PythonNativeProcedure surfaces ProcedureError::Other for
        # bad return values, which must land in the "other" bucket, not
        # "symbolic" or "not_implemented".
        assert stats["other_fallbacks_by_name"].get("bad_ret_proc", 0) >= 1, (
            f"({label}) bad-return fallback must be recorded under "
            f"other_fallbacks_by_name; got {stats['other_fallbacks_by_name']}"
        )
        assert stats["symbolic_fallbacks_by_name"].get("bad_ret_proc", 0) == 0
        assert stats["not_implemented_fallbacks_by_name"].get("bad_ret_proc", 0) == 0

    def test_python_override_bypasses_native_strlen(self):
        """``set_python_override("strlen")`` must skip the native strlen
        registered by default and emit a ``need_simprocedure`` event so
        Python can handle the call instead. Locks the contract documented
        in ``procedures/mod.rs`` (**Dispatch priority** step 2): a Python
        override always wins over the native implementation regardless of
        registration order. angr-o0vm regression.
        """
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")

        STRLEN_HOOK = 0x500500
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFF0000

        mgr = _RustExplorationManager("amd64")
        mgr.set_callbacks(callbacks)
        # Bind an external address to the "strlen" SimProcedure name. The
        # default registry has a native NativeStrlen so this would normally
        # fire on the fast path.
        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        # Force the dispatcher to bypass NativeStrlen by name.
        assert mgr.has_native_procedure("strlen"), (
            "precondition: strlen must be registered natively (else the override path is moot)"
        )
        mgr.set_python_override("strlen")

        state = RustSimState("amd64")
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")
        state.memory_store(STACK_BASE, (0xDEADC0DE).to_bytes(8, "little"))
        state.set_register("rsp", STACK_BASE)
        state.set_register("rdi", STRING_ADDR)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        event = mgr.run(5)

        # Native must NOT have run (override skipped it before dispatch).
        nstats = mgr.native_procedure_stats()
        assert nstats["native_calls"] == 0, (
            f"native strlen must not fire when Python override is set; native_procedure_stats={nstats}"
        )
        # And the python_fallbacks counter on native_proc_stats stays at 0
        # because native was *bypassed*, not *attempted-and-failed*.
        assert nstats["python_fallbacks"] == 0, (
            f"override path bypasses native entirely (no fallback bookkeeping); native_procedure_stats={nstats}"
        )

        # The dispatcher must have routed to the Python SimProcedure path —
        # that bumps the generic simprocedure_python_fallback_count, NOT
        # native_proc_stats.python_fallbacks (those are distinct counters,
        # per the dispatch-priority docs).
        mgr_stats = mgr.stats()
        assert mgr_stats["simprocedure_python_fallback_count"] >= 1, (
            f"override should route through the Python SimProcedure path; got stats={mgr_stats}"
        )
        assert mgr_stats["simprocedure_fallback_by_name"].get("strlen", 0) >= 1, (
            f"by-name fallback bookkeeping must record 'strlen'; got {mgr_stats['simprocedure_fallback_by_name']}"
        )

        # And the engine emitted a need_simprocedure event so Python knows
        # to take over.
        assert event.event_type == "need_callback", (
            f"expected need_callback to hand SimProcedure to Python; got {event.event_type}"
        )
        assert event.callback_reason == "simprocedure"

        # Removing the override restores the native fast path. Wire up a
        # fresh state at the same hook and confirm native fires this time.
        mgr.remove_python_override("strlen")
        state2 = RustSimState("amd64")
        state2.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state2.map_memory(STACK_BASE, 0x1000, 7)
        state2.memory_store(STRING_ADDR, b"hi\x00")
        state2.memory_store(STACK_BASE, (0xDEADC0DE).to_bytes(8, "little"))
        state2.set_register("rsp", STACK_BASE)
        state2.set_register("rdi", STRING_ADDR)
        state2.pc = STRLEN_HOOK
        mgr.add_state("active", state2)
        mgr.run(5)

        nstats2 = mgr.native_procedure_stats()
        assert nstats2["native_calls"] >= 1, (
            f"after remove_python_override, native strlen must fire; native_procedure_stats={nstats2}"
        )

    def test_python_procedure_re_registration_overrides_prior(self):
        """Registering a procedure under an existing name must replace the
        prior callable (HashMap.insert semantics). The dispatcher should
        invoke the most recently registered one.
        """
        mgr = _RustExplorationManager("amd64")

        first_calls = []
        second_calls = []

        def first(args):
            first_calls.append(tuple(args))
            return 0x1111

        def second(args):
            second_calls.append(tuple(args))
            return 0x2222

        HOOK = 0x500400
        EXIT_HOOK = 0x600400
        # First registration via the helper, then override.
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)
        state = self._setup_amd64_python_proc_test(
            mgr,
            HOOK,
            "swap_proc",
            num_args=0,
            no_return=False,
            callable_=first,
            return_addr=EXIT_HOOK,
        )
        # Re-register same name with a different callable.
        mgr.register_python_procedure(
            "swap_proc",
            num_args=0,
            no_return=False,
            callable=second,
        )

        mgr.add_state("active", state)
        mgr.run(10)

        assert first_calls == [], f"prior callable must not run; got {first_calls}"
        assert second_calls == [()], f"override callable must run once; got {second_calls}"

        # RAX should hold the override's return value (0x2222), not 0x1111.
        deadended = mgr.get_state_ids("deadended")
        assert len(deadended) == 1, f"stashes={mgr.stash_counts()}"
        rax = mgr.get_state_register(deadended[0], "rax")
        assert rax == 0x2222, f"expected RAX=0x2222 from override callable; got {rax:#x}"


class TestRustSimStateIntegration:
    """Tests for RustSimState functionality used by exploration."""

    def test_state_fork(self):
        """Test O(1) state forking."""
        state1 = RustSimState("amd64")
        state1.pc = 0x1000
        state1.set_register("rax", 42)

        # Fork
        state2 = state1.fork()

        # Check values preserved
        assert state2.pc == 0x1000
        assert state2.get_register("rax") == 42

        # Check independent
        state2.pc = 0x2000
        state2.set_register("rax", 100)

        assert state1.pc == 0x1000
        assert state1.get_register("rax") == 42

    def test_state_memory(self):
        """Test memory operations."""
        state = RustSimState("amd64")

        # Map memory
        state.map_memory(0x1000, 0x1000, 7)  # RWX

        # Store
        state.memory_store(0x1000, bytes([0x41, 0x42, 0x43, 0x44]))

        # Load
        data = state.memory_load(0x1000, 4)
        assert data == bytes([0x41, 0x42, 0x43, 0x44])

    def test_state_memory_fork(self):
        """Test CoW memory forking."""
        state1 = RustSimState("amd64")
        state1.map_memory(0x1000, 0x1000, 7)
        state1.memory_store(0x1000, bytes([0xAA, 0xBB]))

        # Fork
        state2 = state1.fork()

        # Modify fork
        state2.memory_store(0x1000, bytes([0xCC, 0xDD]))

        # Check original unchanged
        assert state1.memory_load(0x1000, 2) == bytes([0xAA, 0xBB])
        assert state2.memory_load(0x1000, 2) == bytes([0xCC, 0xDD])


class TestRustExplorationPython:
    """Tests for Python RustExplorationManager wrapper."""

    def test_python_wrapper_creation(self, fauxware_project):
        """Test creating Python wrapper."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        assert mgr.stash_counts()["active"] == 1, "entry_state should produce exactly 1 active state"

    def test_strict_page_access_propagates_to_rust(self, fauxware_project):
        """A SimState with STRICT_PAGE_ACCESS option should flip the Rust
        memory model's enforce_permissions flag automatically (mirrors angr)."""
        from angr import sim_options as o

        # No option → flag stays off (default behavior).
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_enforce_permissions(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        strict_state = fauxware_project.factory.entry_state(add_options={o.STRICT_PAGE_ACCESS})
        strict_mgr = RustExplorationManager(fauxware_project, [strict_state])
        strict_ids = strict_mgr._rust_mgr.get_state_ids("active")
        assert strict_ids, "expected an active state to be added"
        assert strict_mgr._rust_mgr.state_enforce_permissions(strict_ids[0]) is True

    def test_set_ip_register_syncs_state_pc(self, fauxware_project):
        """Writing the IP register via set_state_register_symbolic_ast must
        keep the state's pc field in sync (angr-4rq7).

        The RustRegisterProxy write-through gate routes
        ``state.regs.ip = target`` into ``set_state_register_symbolic_ast(
        state_id, 'rip', ...)`` -> ``RustSimState::set_register``. Before
        the fix, set_register only updated the register file and left
        ``self.pc`` stale (often 0 on a freshly-forked state), so
        ``get_state_pc_by_id`` and the next block fetch read 0x0 ("Lift
        error at 0x0") while ``get_register('rip')`` read correctly. This
        asserts the two stay consistent after an IP write.
        """
        import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        ids = mgr._rust_mgr.get_state_ids("active")
        assert ids, "expected an active state to be added"
        sid = ids[0]

        # Baseline: pc and the IP register agree on the entry state.
        assert mgr._rust_mgr.get_state_pc_by_id(sid) == mgr._rust_mgr.get_state_register(sid, "rip")

        # Write a new IP through the same FFI the register proxy uses.
        new_pc = 0x4006ED
        mgr._rust_mgr.set_state_register_symbolic_ast(sid, "rip", claripy.BVV(new_pc, 64))

        # Both the register file and the pc field must reflect the write.
        assert mgr._rust_mgr.get_state_register(sid, "rip") == new_pc
        assert mgr._rust_mgr.get_state_pc_by_id(sid) == new_pc, (
            "set_register('rip') must sync state.pc — otherwise the next block fetch lifts at a stale/zero address"
        )

    def test_strict_page_access_blocks_nx_block_fetch(self):
        """Fetching a basic block from a mapped non-executable page must
        surface as a permission error (state lands in `errored`) when BOTH
        enforce_permissions and enforce_nx are on. Lift callback must NOT
        be invoked at the NX address — the check fires before lifting.
        Matches Python heavy VEX engine, which only raises non-executable
        when STRICT_PAGE_ACCESS AND ENABLE_NX are both set."""
        from angr.rustylib.vex_engine import PythonCallbacks

        mgr = _RustExplorationManager("amd64")
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda addr, size: (bytes(size), False, None))
        callbacks.set_memory_store(lambda addr, data: None)
        lift_addrs = []

        def lift(addr):
            lift_addrs.append(addr)
            return "{}"

        callbacks.set_lift_block(lift)
        mgr.set_callbacks(callbacks)

        state = RustSimState("amd64")
        state.set_enforce_permissions(True)
        state.set_enforce_nx(True)
        state.map_memory(0x1000, 0x1000, 6)  # RW, no X
        state.pc = 0x1000
        mgr.add_state("active", state)

        mgr.run(10)

        counts = mgr.stash_counts()
        assert counts.get("errored", 0) == 1, f"expected the NX block fetch to error the state, got stashes={counts}"
        assert 0x1000 not in lift_addrs, "permission check must fire before lift_block is dispatched"

    def test_strict_page_access_alone_does_not_block_nx_fetch(self):
        """STRICT_PAGE_ACCESS without ENABLE_NX must not raise on a fetch
        from a non-executable page — matches Python's heavy VEX engine,
        which guards the SimSegfaultError on `o.ENABLE_NX in options`
        (angr/engines/vex/heavy/heavy.py:115-124). The lift callback IS
        invoked because no early permission rejection fires."""
        from angr.rustylib.vex_engine import PythonCallbacks

        mgr = _RustExplorationManager("amd64")
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda addr, size: (bytes(size), False, None))
        callbacks.set_memory_store(lambda addr, data: None)
        lift_addrs = []

        def lift(addr):
            lift_addrs.append(addr)
            return "{}"

        callbacks.set_lift_block(lift)
        mgr.set_callbacks(callbacks)

        state = RustSimState("amd64")
        state.set_enforce_permissions(True)
        # enforce_nx left OFF — NX should NOT fire.
        state.map_memory(0x1000, 0x1000, 6)  # RW, no X
        state.pc = 0x1000
        mgr.add_state("active", state)

        mgr.run(10)

        counts = mgr.stash_counts()
        assert counts.get("errored", 0) == 0, f"NX should not fire without enforce_nx; got stashes={counts}"
        assert 0x1000 in lift_addrs, "lift_block should have been dispatched at 0x1000 since NX is off"

    def test_enable_nx_propagates_to_rust(self, fauxware_project):
        """A SimState with ENABLE_NX option should flip the Rust memory
        model's enforce_nx flag automatically (mirrors angr)."""
        from angr import sim_options as o

        # No option → flag stays off.
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_enforce_nx(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        nx_state = fauxware_project.factory.entry_state(add_options={o.ENABLE_NX})
        nx_mgr = RustExplorationManager(fauxware_project, [nx_state])
        nx_ids = nx_mgr._rust_mgr.get_state_ids("active")
        assert nx_ids, "expected an active state to be added"
        assert nx_mgr._rust_mgr.state_enforce_nx(nx_ids[0]) is True

    def test_no_ip_concretization_propagates_to_rust(self, fauxware_project):
        """A SimState with NO_IP_CONCRETIZATION option should flip the
        Rust state's no_ip_concretization flag automatically."""
        from angr import sim_options as o

        # No option → flag stays off.
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_no_ip_concretization(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        nic_state = fauxware_project.factory.entry_state(add_options={o.NO_IP_CONCRETIZATION})
        nic_mgr = RustExplorationManager(fauxware_project, [nic_state])
        nic_ids = nic_mgr._rust_mgr.get_state_ids("active")
        assert nic_ids, "expected an active state to be added"
        assert nic_mgr._rust_mgr.state_no_ip_concretization(nic_ids[0]) is True

    def test_no_ip_concretization_routes_symbolic_jump_to_unconstrained(self):
        """With NO_IP_CONCRETIZATION, a `jmp rax` against a symbolic rax must
        skip enumeration and land the state in the `unconstrained` stash —
        matches engines/successors.py:292-296 (max_targets=0, no warning)."""
        import claripy

        import angr
        from angr import sim_options as o

        # AMD64: `ff e0` = jmp rax. With rax unconstrained-symbolic, Python's
        # symbolic-IP path would normally enumerate concretizations; with
        # NO_IP_CONCRETIZATION it pushes one unconstrained successor instead.
        proj = angr.load_shellcode(b"\xff\xe0", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.NO_IP_CONCRETIZATION},
        )
        state.regs.rax = claripy.BVS("sym_jmp_target", 64)
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=2)

        # State lands in unconstrained, not enumerated into many active forks.
        assert len(mgr.unconstrained) == 1, (
            f"expected 1 unconstrained state with NO_IP_CONCRETIZATION, "
            f"got stashes={ {k: len(v) for k, v in mgr.stashes.items() if v} }"
        )
        # Active stash should not have forked into many enumerated targets.
        assert len(mgr.active) == 0, (
            "active stash must be empty — NO_IP_CONCRETIZATION should have short-circuited enumeration"
        )

    def test_no_symbolic_jump_resolution_propagates_to_rust(self, fauxware_project):
        """A SimState with NO_SYMBOLIC_JUMP_RESOLUTION option should flip the
        Rust state's no_symbolic_jump_resolution flag automatically. Mirrors
        the propagation pattern used for NO_IP_CONCRETIZATION (angr-yl5n)."""
        from angr import sim_options as o

        # No option → flag stays off.
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_no_symbolic_jump_resolution(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        nsjr_state = fauxware_project.factory.entry_state(add_options={o.NO_SYMBOLIC_JUMP_RESOLUTION})
        nsjr_mgr = RustExplorationManager(fauxware_project, [nsjr_state])
        nsjr_ids = nsjr_mgr._rust_mgr.get_state_ids("active")
        assert nsjr_ids, "expected an active state to be added"
        assert nsjr_mgr._rust_mgr.state_no_symbolic_jump_resolution(nsjr_ids[0]) is True

    def test_no_symbolic_jump_resolution_routes_symbolic_jump_to_unconstrained(self):
        """With NO_SYMBOLIC_JUMP_RESOLUTION, a `jmp rax` against a symbolic
        rax must skip enumeration and land the state in the `unconstrained`
        stash — matches engines/successors.py:234-239 (early elif route to
        unconstrained_successors before AddressConcretizer is invoked)."""
        import claripy

        import angr
        from angr import sim_options as o

        # AMD64: `ff e0` = jmp rax. With rax unconstrained-symbolic, Python's
        # symbolic-IP path would normally enumerate concretizations; with
        # NO_SYMBOLIC_JUMP_RESOLUTION the state goes to unconstrained.
        proj = angr.load_shellcode(b"\xff\xe0", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.NO_SYMBOLIC_JUMP_RESOLUTION},
        )
        state.regs.rax = claripy.BVS("sym_jmp_target", 64)
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=2)

        assert len(mgr.unconstrained) == 1, (
            f"expected 1 unconstrained state with NO_SYMBOLIC_JUMP_RESOLUTION, "
            f"got stashes={ {k: len(v) for k, v in mgr.stashes.items() if v} }"
        )
        assert len(mgr.active) == 0, (
            "active stash must be empty — NO_SYMBOLIC_JUMP_RESOLUTION should have short-circuited enumeration"
        )

    def test_symbolic_syscall_num_forces_python_fallback(self):
        """A symbolic syscall register must skip the native syscall registry
        and route to the Python callback (angr-gffd). Before the fix,
        `get_syscall_num` returned ``unwrap_or(0)``; on amd64 syscall 0 is
        ``read`` (NativeReadSyscall) so a symbolic ``rax`` silently dispatched
        to ``read`` with whatever happened to be in the arg registers.

        We assert the Python fallback runs (``syscall_python_fallback_count``
        is bumped) — that is the load-bearing observation. With
        ``NO_SYMBOLIC_SYSCALL_RESOLUTION`` set, Python's
        ``engines/successors.py::_resolve_syscall`` (line 352) returns
        ``(syscall_num, None)`` and the unknown-syscall stub takes over, so
        no enumeration happens.
        """
        import claripy

        import angr
        from angr import sim_options as o

        # AMD64: `0f 05` = syscall. Pad so PC+2 stays mapped.
        shellcode = b"\x0f\x05" + b"\x90" * 0x100
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.NO_SYMBOLIC_SYSCALL_RESOLUTION},
        )
        # Fresh unconstrained syscall number.
        state.regs.rax = claripy.BVS("sym_syscall_num", 64)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] >= 1, (
            "Python syscall fallback must run for a symbolic syscall number — "
            "otherwise Rust silently dispatched to native syscall 0 (read on "
            f"amd64). Got {stats['syscall_python_fallback_count']}."
        )
        # angr-0wam: the per-num breakdown must record symbolic syscalls under
        # the -1 sentinel key (no concrete num was consulted).
        by_num = stats["syscall_python_fallback_by_num"]
        assert by_num.get(-1, 0) >= 1, (
            f"symbolic syscall must increment syscall_python_fallback_by_num[-1] (got {by_num})"
        )

    def test_concrete_syscall_num_still_uses_native_dispatch(self):
        """Sanity check that the angr-gffd `Option<u64>` plumbing did not
        regress the concrete-syscall fast path. A concrete ``rax = 60`` (exit
        on amd64) must dispatch through ``NativeExitSyscall`` without bumping
        the Python fallback counter."""
        import angr

        shellcode = b"\x0f\x05" + b"\x90" * 0x100
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = 60  # exit

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "concrete syscall must take the native fast path, not the Python "
            f"fallback (got {stats['syscall_python_fallback_count']})"
        )
        # NativeExitSyscall routes the state to deadended.
        assert len(mgr.deadended) == 1, (
            f"native exit(60) must deadend the state, got stashes={ {k: len(v) for k, v in mgr.stashes.items() if v} }"
        )

    def test_mips_native_syscall_clears_a3_error_register(self):
        """angr-pfbu: Linux MIPS syscalls report success/failure in $a3
        (0=success, non-zero=error) separately from the $v0 return value;
        glibc branches on it. The Rust native dispatcher previously wrote only
        the return register, leaving $a3 holding its pre-syscall value (syscall
        arg 4). A successful native syscall could then take the binary's errno
        path.

        Here a MIPS32 ``syscall`` runs ``getpid`` (4020, a no-arg success) with
        $a3 pre-seeded to a sentinel. After the step $a3 must be cleared to 0,
        mirroring ``SimCCO32LinuxSyscall.linux_syscall_update_error_reg``.
        """
        import angr

        # MIPS32 little-endian `syscall` = 0x0000000c -> bytes 0c 00 00 00.
        # Only the syscall is mapped so the successor deadends immediately and
        # freezes $a3 (see the error-path test for why padding is avoided).
        shellcode = b"\x0c\x00\x00\x00"
        proj = angr.load_shellcode(shellcode, arch="mipsel", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.v0 = 4020  # getpid
        state.regs.a3 = 0x12345678  # sentinel: must be overwritten to 0

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "MIPS getpid must take the native fast path, not the Python "
            f"fallback (got {stats['syscall_python_fallback_count']})"
        )
        # The post-syscall state may land in any stash (the synthetic shellcode
        # has no valid return target), so search all of them for the single
        # state and read its $a3.
        a3 = self._find_single_state_register(mgr, "a3")
        assert a3 == 0, (
            f"MIPS $a3 must be cleared to 0 after a successful syscall, got {a3:#x} "
            "(stale arg4 value means glibc would take the errno path)"
        )

    def test_mips_native_syscall_error_sets_a3_nonzero(self):
        """angr-pfbu: a native MIPS syscall returning a negative errno must set
        $a3 non-zero and leave $v0 holding the *positive* errno, mirroring
        ``linux_syscall_update_error_reg`` (which negates the return value on
        the error path). ``close`` of an unopened fd returns -1 -> $a3=all-ones,
        $v0=1.
        """
        import angr

        # Only the 4-byte `syscall` is mapped: the successor at 0x1004 lifts
        # into unmapped memory and deadends *immediately*, freezing $v0/$a3 at
        # their post-syscall values. (Padding the shellcode with executable
        # nops lets the engine run past the syscall and clobber $v0 before we
        # can read it — $a3 happens to survive, but $v0 does not.)
        shellcode = b"\x0c\x00\x00\x00"
        proj = angr.load_shellcode(shellcode, arch="mipsel", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.v0 = 4006  # close
        state.regs.a0 = 9999  # never-opened fd -> -EBADF path
        state.regs.a3 = 0  # sentinel: must be set non-zero

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        a3 = self._find_single_state_register(mgr, "a3")
        v0 = self._find_single_state_register(mgr, "v0")
        assert a3 != 0, f"MIPS $a3 must be non-zero on a syscall error, got {a3:#x}"
        assert v0 == 1, f"MIPS $v0 must hold the positive errno (1) on the error path, got {v0:#x}"

    @staticmethod
    def _find_single_state_register(mgr, reg):
        """Read a register from the lone post-step state regardless of which
        stash it landed in. Returns the register value (u128) or raises if no
        state is found."""
        for stash in mgr.stashes:
            for sid in mgr._rust_mgr.get_state_ids(stash):
                return mgr._rust_mgr.get_state_register(sid, reg)
        raise AssertionError(f"no state found in any stash to read {reg}")

    def test_keep_ip_symbolic_propagates_to_rust(self, fauxware_project):
        """A SimState with KEEP_IP_SYMBOLIC option should flip the Rust
        state's keep_ip_symbolic flag automatically — mirrors angr-yl5n's
        propagation test for NO_IP_CONCRETIZATION."""
        from angr import sim_options as o

        # No option → flag stays off.
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_keep_ip_symbolic(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        kis_state = fauxware_project.factory.entry_state(add_options={o.KEEP_IP_SYMBOLIC})
        kis_mgr = RustExplorationManager(fauxware_project, [kis_state])
        kis_ids = kis_mgr._rust_mgr.get_state_ids("active")
        assert kis_ids, "expected an active state to be added"
        assert kis_mgr._rust_mgr.state_keep_ip_symbolic(kis_ids[0]) is True

    def test_short_reads_option_propagates_to_rust(self, fauxware_project):
        """A SimState with SHORT_READS should set the Rust state's
        symex-relevant SimOption mirror so native SimProcedures can read it
        via has_option (angr-kzjv6)."""
        from angr import sim_options as o

        # No option → has_option("SHORT_READS") is False.
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_has_option(plain_ids[0], "SHORT_READS") is False

        # Option present → has_option flips on for the Rust state.
        sr_state = fauxware_project.factory.entry_state(add_options={o.SHORT_READS})
        sr_mgr = RustExplorationManager(fauxware_project, [sr_state])
        sr_ids = sr_mgr._rust_mgr.get_state_ids("active")
        assert sr_ids, "expected an active state to be added"
        assert sr_mgr._rust_mgr.state_has_option(sr_ids[0], "SHORT_READS") is True

    def test_set_option_has_option_roundtrip_and_fork(self):
        """RustSimState.set_option / has_option roundtrip, and the option set
        is carried across fork (angr-kzjv6)."""
        state = RustSimState("amd64")
        assert state.has_option("SHORT_READS") is False
        state.set_option("SHORT_READS", True)
        assert state.has_option("SHORT_READS") is True
        # Forked child inherits the snapshot.
        child = state.fork()
        assert child.has_option("SHORT_READS") is True
        # Removal works and is independent on the child (CoW).
        child.set_option("SHORT_READS", False)
        assert child.has_option("SHORT_READS") is False
        assert state.has_option("SHORT_READS") is True

    def test_keep_ip_symbolic_leaves_ip_register_symbolic_after_fork(self):
        """With KEEP_IP_SYMBOLIC, after a `jmp rax` against a symbolic rax
        constrained to a small set of concrete addresses, each forked state's
        IP register should still hold a symbolic expression (not the
        concretized value). Mirrors engines/successors.py:326-331's
        `regs.ip = target` branch which skips `add_constraints(cond)`.

        Without the option, each fork has `rip` pinned to a concrete u64 (the
        Rust manager API's `get_state_register("rip")` returns Some(value)).
        With the option, the IP register is symbolic and the API returns None.
        """
        import claripy

        import angr
        from angr import sim_options as o

        # AMD64: `ff e0` = jmp rax. Two concrete jump targets are mapped so
        # the concretizer has somewhere to land (otherwise the jumps go to
        # unmapped memory and the test exits the Rust path early).
        shellcode = b"\xff\xe0" + b"\x00" * 0x2000  # pad so 0x2000/0x3000 are in-range
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        def _new_state(*, with_keep):
            opts = {o.KEEP_IP_SYMBOLIC} if with_keep else set()
            state = proj.factory.blank_state(addr=0x1000, add_options=opts)
            sym_target = claripy.BVS("sym_jmp_target", 64)
            state.regs.rax = sym_target
            # Restrict rax to two concrete in-range targets so the
            # concretizer returns Multiple(2) instead of TooMany.
            state.solver.add(claripy.Or(sym_target == 0x2000, sym_target == 0x2010))
            return state

        # Use step(1) (exactly one step pass) instead of run(max_steps=1):
        # run() keeps stepping until termination so each fork would proceed
        # into the zero-padded region (decoded as add [rax], al) and either
        # deadend or fall back, washing the forks out of the active stash.
        # step(1) gives us the post-fork snapshot.

        # --- Baseline: WITHOUT KEEP_IP_SYMBOLIC ---
        base_mgr = RustExplorationManager(proj, [_new_state(with_keep=False)])
        base_mgr.step(1)
        base_ids = base_mgr._rust_mgr.get_state_ids("active")
        assert len(base_ids) == 2, f"expected 2 forks for jmp rax over 2 targets, got {len(base_ids)}"
        for sid in base_ids:
            rip = base_mgr._rust_mgr.get_state_register(sid, "rip")
            assert rip is not None, "without KEEP_IP_SYMBOLIC, rip should be concretized to a u64"
            assert rip in (0x2000, 0x2010), f"unexpected concretized rip 0x{rip:x}"

        # --- With KEEP_IP_SYMBOLIC ---
        kis_mgr = RustExplorationManager(proj, [_new_state(with_keep=True)])
        kis_mgr.step(1)
        kis_ids = kis_mgr._rust_mgr.get_state_ids("active")
        assert len(kis_ids) == 2, (
            f"expected 2 forks for jmp rax over 2 targets with KEEP_IP_SYMBOLIC, got {len(kis_ids)}"
        )
        for sid in kis_ids:
            rip = kis_mgr._rust_mgr.get_state_register(sid, "rip")
            assert rip is None, f"with KEEP_IP_SYMBOLIC, rip must stay symbolic, got concrete 0x{rip:x}"
            assert kis_mgr._rust_mgr.state_keep_ip_symbolic(sid) is True, (
                "keep_ip_symbolic flag should propagate through fork"
            )

    def test_solver_stats_populated(self, fauxware_project):
        """mgr.get_solver_stats() returns populated counters after exploration.

        Verifies the observability surface for angr-2usz: per-process Z3 query
        counters (sat/unsat/timeout, total time, per-site breakdown) reach
        Python via the instance method.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.reset_solver_stats()

        baseline = mgr.get_solver_stats()
        assert isinstance(baseline, dict)
        assert "z3_check_count" in baseline
        assert "z3_sat_count" in baseline
        assert "z3_unsat_count" in baseline
        assert "z3_timeout_count" in baseline
        assert "z3_check_time_ns" in baseline
        assert baseline["z3_check_count"] == 0
        assert baseline["z3_sat_count"] == 0
        assert baseline["z3_unsat_count"] == 0
        assert baseline["z3_timeout_count"] == 0

        mgr.explore(find=0x4006ED, num_find=1)

        stats = mgr.get_solver_stats()
        assert stats["z3_check_count"] >= 1, f"expected at least one solver query, got {stats['z3_check_count']}"
        # A completed check must take nonzero wall-clock time.
        assert stats["z3_check_time_ns"] > 0
        # Sat + unsat + timeout should account for every check.
        assert (stats["z3_sat_count"] + stats["z3_unsat_count"] + stats["z3_timeout_count"]) == stats["z3_check_count"]

    def test_z3_ast_cache_counters(self, fauxware_project):
        """angr-zdho: `z3_ast_cache_hit` + `z3_ast_cache_miss` are exposed via
        `get_solver_stats()` and at least one is non-zero after exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.reset_solver_stats()

        baseline = mgr.get_solver_stats()
        assert "z3_ast_cache_hit" in baseline
        assert "z3_ast_cache_miss" in baseline
        assert baseline["z3_ast_cache_hit"] == 0
        assert baseline["z3_ast_cache_miss"] == 0

        mgr.explore(find=0x4006ED, num_find=1)

        stats = mgr.get_solver_stats()
        assert stats["z3_ast_cache_miss"] >= 1, (
            f"exploration with at least one Z3 query should produce cache misses; got {stats['z3_ast_cache_miss']}"
        )

    def test_fork_counters_exposed_and_non_summable(self, fauxware_project):
        """angr-95up.2: `solver_fork_count` and `deferred_fork_count` are
        exposed via `get_execution_stats()` as non-negative u64s AND they
        measure distinct, non-summable concepts.

        Semantics (mirrored from counter docstrings in
        `native/angr/src/interpreter/mod.rs`):

        - `deferred_fork_count` counts deferred forks PRESENTED to
          post-block processing (input length of the `deferred_forks`
          Vec, summed across all processing sites). Not every entry
          produces a solver clone: callback-resume entries with no
          stored/reconstructed condition are skipped, and stepping-path
          entries missing a condition route through a conservative
          `state.fork()` that is NOT tallied by `solver_fork_count`.
        - `solver_fork_count` counts solver Z3-clone operations whose
          cost is timed by `solver_fork_time_ns`. It includes a
          pre-callback state-snapshot fork on the SimProcedure path that
          is NOT a deferred fork, and excludes the conservative-fork
          P15 fallback. There is no "total fork count" the two should
          sum to.

        Both counters are also profiling-gated at their increment sites
        (they live inside the same `if let Some(start) = ..._fork_start`
        block as their `*_fork_time_ns` siblings), so a run that did not
        call `enable_profiling()` will leave the counters at zero even
        when the underlying fork ops happen. Test runs with profiling
        enabled and only requires structural well-formedness, not a
        specific count (count is a property of the binary's symbolic
        branch geometry, not the counter wiring).
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.enable_profiling()
        mgr.explore(find=0x4006ED, num_find=1)

        exec_stats = mgr._rust_mgr.get_execution_stats()
        for key in ("solver_fork_count", "deferred_fork_count", "solver_fork_time_ns", "deferred_fork_time_ns"):
            assert key in exec_stats, f"missing key {key}"
            # u64 counters surface as Python ints; isinstance covers the
            # non-negativity contract (a `>= 0` arm would be tautological).
            assert isinstance(exec_stats[key], int), f"{key} not int"

        # fauxware's symbolic auth branch is guaranteed to fork, so the
        # profiling-gated fork instrumentation must actually execute. The
        # *_fork_time_ns timers live inside the same `if let Some(start) =
        # ..._fork_start` blocks as the counters, so a positive timer proves
        # the block ran — catching a regression that breaks the profiling
        # gate (leaving the whole block dead), which the type-only checks
        # above would miss. The *_fork_count values are intentionally NOT
        # asserted nonzero: fauxware's forks route through the conservative
        # `state.fork()` path that is timed but not tallied (see docstring),
        # so the counts legitimately stay 0 here.
        assert exec_stats["deferred_fork_time_ns"] > 0, (
            f"deferred-fork block must have run on a branching binary; got {exec_stats['deferred_fork_time_ns']}"
        )
        assert exec_stats["solver_fork_time_ns"] > 0, (
            f"solver-fork block must have run on a branching binary; got {exec_stats['solver_fork_time_ns']}"
        )

    def test_analyze_constraint_sharing(self, fauxware_project):
        """angr-zdho: `analyze_constraint_sharing()` reports pointer-vs-structural
        sharing across every state's assumed-constraint RustBV graph.

        Invariants:
          - `unique_pointers >= unique_shapes` (hash-cons can only collapse,
            never split)
          - `structural_duplicates == unique_pointers - unique_shapes`
          - keys are present and integer-valued
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, num_find=1)

        sharing = mgr._rust_mgr.analyze_constraint_sharing()
        for key in (
            "total_visits",
            "unique_pointers",
            "unique_shapes",
            "structural_duplicates",
            "states_analyzed",
            "constraints_analyzed",
        ):
            assert key in sharing, f"missing key {key} in {sharing}"
            assert isinstance(sharing[key], int), f"{key} not int"

        assert sharing["unique_pointers"] >= sharing["unique_shapes"], f"hash-cons can never split: {sharing}"
        assert sharing["structural_duplicates"] == sharing["unique_pointers"] - sharing["unique_shapes"], (
            f"structural_duplicates accounting wrong: {sharing}"
        )
        # fauxware should produce at least one constraint to walk.
        assert sharing["constraints_analyzed"] >= 1
        assert sharing["total_visits"] >= sharing["unique_pointers"]

    def test_basic_explore(self, fauxware_project):
        """Test basic exploration with find address."""

        # Find the "Welcome" message address
        # In fauxware, this is typically around 0x4006ed
        find_addr = 0x4006ED

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Run exploration
        mgr.explore(find=find_addr)
        # Exploration completed — verify at least one state was found or explored
        assert len(mgr.found) > 0 or len(mgr.deadended) > 0, "exploration should find states or deadend some"

    def test_state_export_cache_invalidated_on_step(self, fauxware_project):
        """step() / explore() must clear the rust_fully_synced sentinel on
        cached Python state mirrors so that the next stash read re-syncs
        from the (now-advanced) Rust state. Rust remains the single source
        of truth; the cache only short-circuits redundant syncs between
        re-entries.

        Note: stash properties now return ``_LazySimStateRef`` wrappers
        (kwpi.b). Touching ``wrapper.scratch`` triggers materialization and
        re-syncs the cached SimState, so post-step sentinel state has to be
        observed via ``_state_cache`` directly to avoid the re-sync.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.step(n=1)

        active = mgr.active
        assert len(active) >= 1, "fauxware should still have at least one active state after one step"
        first = active[0]
        # First read materializes the SimState and sets the sentinel.
        assert getattr(first.scratch, "rust_fully_synced", False), (
            "first read should set the rust_fully_synced sentinel"
        )
        cached_state = mgr._state_cache[first._lazy_state_id]

        # Re-read without stepping: cache hit must preserve wrapper identity
        # and not re-sync the cached SimState.
        again = mgr.active
        assert again[0] is first, "repeated mgr.active access must return the cached lazy ref"
        assert getattr(again[0].scratch, "rust_fully_synced", False)

        # step() invalidates: cached mirror is now stale until next read.
        # Observe the sentinel via the cached SimState directly; reading it
        # through the wrapper would re-trigger materialization.
        mgr.step(n=1)
        assert not getattr(cached_state.scratch, "rust_fully_synced", False), (
            "step() must clear rust_fully_synced on cached SimStates"
        )

        # explore() must also invalidate at its top.
        state2 = fauxware_project.factory.entry_state()
        mgr2 = RustExplorationManager(fauxware_project, [state2])
        mgr2.step(n=1)
        cached_wrapper = mgr2.active[0]
        # Force materialization to seed the cache + set the sentinel.
        assert getattr(cached_wrapper.scratch, "rust_fully_synced", False)
        mgr2_cached = mgr2._state_cache[cached_wrapper._lazy_state_id]
        mgr2.explore(max_steps=1)
        assert not getattr(mgr2_cached.scratch, "rust_fully_synced", False), (
            "explore() must clear rust_fully_synced on cached SimStates"
        )

    def test_stash_access(self, fauxware_project):
        """Test accessing stashes."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Access stashes
        active = mgr.active
        found = mgr.found
        avoid = mgr.avoid
        deadended = mgr.deadended

        assert isinstance(active, list)
        assert isinstance(found, list)
        assert isinstance(avoid, list)
        assert isinstance(deadended, list)

    def _proxy_with_multiple_active(self, project):
        """Step a fauxware manager until >=2 active states exist; return
        (mgr, RustSimulationManagerProxy)."""
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        state = project.factory.entry_state()
        mgr = RustExplorationManager(project, [state])
        for _ in range(20):
            mgr.step()
            if mgr.stash_counts().get("active", 0) >= 2:
                break
        proxy = RustSimulationManagerProxy(rust_mgr=mgr._rust_mgr, project=project)
        return mgr, proxy

    def test_stash_assignment_limits_subset(self, fauxware_project):
        """simgr.stashes['active'] = [subset] keeps only the assigned proxies (angr-wxuo)."""
        _mgr, proxy = self._proxy_with_multiple_active(fauxware_project)
        active = proxy.stashes["active"]
        assert len(active) >= 2, "fauxware should fork into >=2 active states"

        keep = active[0]
        proxy.stashes["active"] = [keep]

        remaining = proxy.stashes["active"]
        assert len(remaining) == 1
        assert remaining[0].state_id == keep.state_id

    def test_stash_assignment_reorders(self, fauxware_project):
        """Assignment honors caller-specified ordering (angr-wxuo)."""
        _mgr, proxy = self._proxy_with_multiple_active(fauxware_project)
        active = proxy.stashes["active"]
        assert len(active) >= 2

        reversed_ids = [s.state_id for s in reversed(active)]
        proxy.stashes["active"] = list(reversed(active))

        got_ids = [s.state_id for s in proxy.stashes["active"]]
        assert got_ids == reversed_ids

    def test_stash_assignment_rejects_non_proxy(self, fauxware_project):
        """Assigning non-RustStateProxy objects raises (angr-wxuo)."""
        _mgr, proxy = self._proxy_with_multiple_active(fauxware_project)
        with pytest.raises(TypeError):
            proxy.stashes["active"] = [123]
        with pytest.raises(TypeError):
            proxy.stashes["active"] = "not-a-list"

    def test_stash_assignment_rejects_foreign_state(self, fauxware_project):
        """Assigning a proxy from a different manager raises (angr-wxuo)."""
        _mgr_a, proxy_a = self._proxy_with_multiple_active(fauxware_project)

        state_b = fauxware_project.factory.entry_state()
        mgr_b = RustExplorationManager(fauxware_project, [state_b])
        proxy_b = self._make_proxy(mgr_b, fauxware_project)
        foreign = proxy_b.stashes["active"][0]

        with pytest.raises(ValueError):
            proxy_a.stashes["active"] = [foreign]

    def _make_proxy(self, mgr, project):
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        return RustSimulationManagerProxy(rust_mgr=mgr._rust_mgr, project=project)

    def test_stash_assignment_empty_clears(self, fauxware_project):
        """Empty-list assignment still clears the stash (angr-wxuo regression)."""
        _mgr, proxy = self._proxy_with_multiple_active(fauxware_project)
        assert len(proxy.stashes["active"]) >= 1
        proxy.stashes["active"] = []
        assert proxy.stashes["active"] == []

    def test_max_active_states_python(self, fauxware_project):
        """Test max_active_states limit via Python wrapper."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=2)

        # Run exploration with limited active states
        mgr.explore(find=0x4006ED, num_find=1)

        # Active states should never exceed the limit
        counts = mgr.stash_counts()
        assert counts.get("active", 0) <= 2, f"active count {counts['active']} exceeds max_active_states=2"

    def test_max_active_states_prunes_forks(self, fauxware_project):
        """Excess forks should be pruned when max_active_states is reached.

        With limit=1, fauxware's symbolic strcmp branches must produce at
        least one pruned state — otherwise the limit is being silently
        ignored (the bug fixed by angr-jmiz).
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=1)
        mgr.explore(find=0x4006ED, num_find=1)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) <= 1, f"active count {counts.get('active', 0)} exceeds max_active_states=1"
        assert counts.get("pruned", 0) > 0, f"expected pruned states with max_active_states=1, got counts={counts}"

    def test_max_active_states_default_cap(self, fauxware_project):
        """A manager built without max_active_states gets the default safety cap.

        angr-o4q3: an unbounded active stash can OOM the process on a divergent
        explore. The default must be a finite backstop, not None/unlimited.
        """
        from angr.exploration.rust_manager import DEFAULT_MAX_ACTIVE_STATES

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr._rust_mgr.get_max_active_states() == DEFAULT_MAX_ACTIVE_STATES

    def test_max_active_states_default_disable(self, fauxware_project):
        """Passing max_active_states=None explicitly disables the default cap."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=None)
        assert mgr._rust_mgr.get_max_active_states() is None

    def test_progress_callback(self, fauxware_project):
        """Test that progress callback fires during exploration."""

        progress_reports = []

        def on_progress(info):
            progress_reports.append(info)

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_progress_callback(on_progress, interval_steps=1)
        mgr.explore(find=0x4006ED, num_find=1)

        # Progress callback should have fired at least once
        assert len(progress_reports) > 0, "progress callback never fired"
        # Each report should have the expected keys
        report = progress_reports[0]
        assert "step_count" in report
        assert "active_count" in report
        assert "found_count" in report
        assert "elapsed_seconds" in report

    def test_progress_callback_deadended_count(self, fauxware_project):
        """deadended_count in the progress payload must match the actual stash.

        Regression for angr-8mln: the payload read counts.get("deadened", 0)
        but the Rust stash key is "deadended", so deadended_count was always 0.
        """
        progress_reports = []

        def on_progress(info):
            progress_reports.append(info)

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_progress_callback(on_progress, interval_steps=1)
        # A never-matching find predicate routes through the predicate path,
        # which keeps terminal states (drop_terminal_states=False), so the
        # deadended stash fills up and the progress callback can observe it.
        # Exploration runs until the active stash drains.
        mgr.explore(find=lambda s: False)

        assert len(progress_reports) > 0, "progress callback never fired"
        # Every tick must carry the deadended_count key...
        assert all("deadended_count" in r for r in progress_reports)
        # ...and fauxware deadends states, so at least one tick must report
        # a nonzero count (the typo made this permanently 0).
        assert max(r["deadended_count"] for r in progress_reports) > 0, (
            "deadended_count never went nonzero — stash-key typo regression"
        )
        # The deadended stash only grows during run(), so each report's count
        # must be consistent with the final live stash size.
        final_deadended = len(mgr._rust_mgr.get_state_ids("deadended"))
        assert final_deadended > 0
        assert progress_reports[-1]["deadended_count"] <= final_deadended


class TestExplorationEvent:
    """Tests for ExplorationEvent class."""

    def test_event_attributes(self):
        """Test event has expected attributes."""
        mgr = _RustExplorationManager("amd64")

        # Set up minimal callbacks
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        # Run should return an event
        event = mgr.run(1)

        # With no active states, run() deterministically takes the
        # empty-active termination branch and must return an `active_empty`
        # event with zeroed counts/steps. Assert the actual values, not just
        # attribute presence — a run() that returned the wrong event with
        # garbage values would pass a hasattr-only check.
        assert event.event_type == "active_empty", f"expected active_empty on no-state run; got {event.event_type}"
        assert event.found_count == 0, f"found_count must be 0; got {event.found_count}"
        assert event.active_count == 0, f"active_count must be 0; got {event.active_count}"
        assert event.steps_taken == 0, f"steps_taken must be 0 (no active states to step); got {event.steps_taken}"


class TestRustEdgeCases:
    """Edge-case tests for Rust state and solver."""

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def test_load_shellcode_blob_in_binary_control_flow(self):
        """In-blob jump targets in a load_shellcode/blob must resolve, not deadend.

        Regression for angr-1lzq. The Blob main object produced by
        ``load_shellcode`` has ``binary is None`` (stream-backed, no file path),
        so ``_load_binary_regions`` skipped it and the Rust engine received zero
        concrete regions. With no region, ``is_in_binary`` was false for every
        in-blob address, so an in-bounds jump was misclassified as an unmodeled
        call and the successor IP collapsed to 0x0 ("Lift error at 0x0"),
        deadending after one step. A second latent bug truncated the last byte
        of every executable region (cle's ``max_addr`` is inclusive, so the byte
        count is ``max_addr - min_addr + 1``).

        ``\xeb\xfe`` is ``jmp $`` (self-loop). Once the blob region is registered
        the state must stay active at the loop address across steps instead of
        deadending to 0x0.
        """
        proj = angr.load_shellcode(b"\xeb\xfe", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])

        for _ in range(3):
            mgr.step(n=1)
            active = mgr.active_proxies()
            assert len(active) == 1, f"expected the self-loop to stay active, got {mgr.stash_counts()}"
            assert active[0].addr == 0x1000, f"self-loop drifted off 0x1000 to {hex(active[0].addr)}"

    def test_load_shellcode_blob_conditional_loop_cold_flags(self):
        """A cold conditional loop in a blob must run to completion, not deadend.

        Regression for angr-g6dg. In a blank_state-backed blob, ``dec``'s VEX
        calls ``amd64g_calculate_rflags_c`` to preserve the carry in cc_ndep.
        For the INC/DEC cc_op category the carry is preserved from cc_ndep, but
        the Rust ccall layer only handled Copy/Sub/Add/Logic and fell through to
        a fresh unconstrained symbolic carry, poisoning the flags threaded into
        later iterations.

        ``mov ecx,3; dec ecx; jnz $-2; jmp $`` counts down concretely and parks
        on the trailing ``jmp $`` self-loop pad at 0x1009. The state must reach
        and stay active at 0x1009 across steps, never collapsing to 0x0.
        """
        # b9 03 00 00 00  mov ecx, 3
        # ff c9           dec ecx          (0x1005)
        # 75 fc           jnz 0x1005       (0x1007)
        # eb fe           jmp $            (0x1009, self-loop landing pad)
        shellcode = b"\xb9\x03\x00\x00\x00\xff\xc9\x75\xfc\xeb\xfe"
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])

        for _ in range(6):
            mgr.step(n=1)
            active = mgr.active_proxies()
            assert len(active) == 1, f"expected one active state, got {mgr.stash_counts()}"
            assert active[0].addr == 0x1009, f"loop drifted off the pad to {hex(active[0].addr)}"

    def test_block_granular_step_observes_mid_chain_address(self):
        """Block-granular stepping makes a chained-through block boundary observable.

        Regression for angr-bmyx. The VEX interpreter chains basic blocks within
        one ``step(n=1)``, so a state runs THROUGH an interior block boundary
        without ever stopping there — which breaks bare step-loop idioms like
        CADET solve.py phase 3 (``while True: sm.step(); break if any active.addr
        == TARGET``). ``set_block_granular(True)`` breaks the chain at every
        block boundary so each ``step()`` advances exactly one block.

        Layout (3 blocks, each a direct jmp):
          0x1000: eb 02  jmp 0x1004   (block A)
          0x1004: eb 02  jmp 0x1008   (block B — the transient mid-chain target)
          0x1008: eb fe  jmp $        (block C — self-loop landing pad)
        The 0x1002/0x1006 nop pads are dead (jumped over).
        """
        shellcode = b"\xeb\x02\x90\x90\xeb\x02\x90\x90\xeb\xfe"

        # Default (chaining on): step(n=1) runs A->B->C and parks on the C
        # self-loop; the 0x1004 boundary is never observable.
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        chained = RustExplorationManager(proj, [proj.factory.blank_state(addr=0x1000)])
        seen_chained = set()
        for _ in range(3):
            chained.step(n=1)
            seen_chained.update(p.addr for p in chained.active_proxies())
        assert 0x1004 not in seen_chained, (
            f"chained stepping should run past 0x1004, but observed it: {sorted(map(hex, seen_chained))}"
        )
        assert 0x1008 in seen_chained, (
            f"chained stepping should reach the self-loop pad: {sorted(map(hex, seen_chained))}"
        )

        # Block-granular: each step advances exactly one block, so the 0x1004
        # boundary becomes observable.
        proj2 = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        bg = RustExplorationManager(proj2, [proj2.factory.blank_state(addr=0x1000)])
        prev = bg.set_block_granular(True)
        assert prev is False, "block-granular should default to off"

        bg.step(n=1)
        active = bg.active_proxies()
        assert len(active) == 1, f"expected one active state, got {bg.stash_counts()}"
        assert active[0].addr == 0x1004, f"step 1 should stop at the 0x1004 boundary, got {hex(active[0].addr)}"

        bg.step(n=1)
        active = bg.active_proxies()
        assert active[0].addr == 0x1008, f"step 2 should advance one block to 0x1008, got {hex(active[0].addr)}"

        # Restoring chaining returns the prior (enabled) value.
        assert bg.set_block_granular(False) is True

    def test_materialize_unconstrained_forks_toggle(self):
        """The unconstrained-fork materialization flag round-trips (angr-ckdy).

        ``set_materialize_unconstrained_forks(True)`` keeps the loop-exit
        deferred forks alive at an ``UnconstrainedJump`` (instead of dropping
        them in deferred mode) so a bare step-loop that bypasses ``explore()``
        (CADET solve.py phase 3) keeps progressing toward a target behind a
        symbolic loop exit. This pins the setter/getter plumbing through the
        PyO3 boundary; the end-to-end CADET egg-hunt convergence it enables is
        covered (env-gated, heavy) in ``test_misc.py``.
        """
        shellcode = b"\xeb\xfe"  # jmp $ — minimal self-loop, never used to step
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=0x1000)])

        # Default off.
        assert mgr._rust_mgr.materialize_unconstrained_forks() is False
        # Setter returns the prior value and flips the flag.
        assert mgr.set_materialize_unconstrained_forks(True) is False
        assert mgr._rust_mgr.materialize_unconstrained_forks() is True
        # Restoring returns the prior (enabled) value.
        assert mgr.set_materialize_unconstrained_forks(False) is True
        assert mgr._rust_mgr.materialize_unconstrained_forks() is False

    def test_page_boundary_store_load(self):
        """Store 6 bytes at offset 4090, crossing a 4096-byte page boundary."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x2000, 7)  # Map 2 pages

        data_in = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])
        state.memory_store(0x1000 + 4090, data_in)
        data_out = state.memory_load(0x1000 + 4090, 6)
        assert data_out == data_in

    def test_page_boundary_cow_fork(self):
        """Page-boundary store preserves CoW isolation after fork."""
        state1 = RustSimState("amd64")
        state1.map_memory(0x1000, 0x2000, 7)
        state1.memory_store(0x1000 + 4090, bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]))

        state2 = state1.fork()
        state2.memory_store(0x1000 + 4090, bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66]))

        assert state1.memory_load(0x1000 + 4090, 6) == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
        assert state2.memory_load(0x1000 + 4090, 6) == bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])

    def test_many_symbolic_variables(self):
        """Solver with >50 symbolic variables and constraints."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        symbols = []
        for i in range(60):
            s = claripy.BVS(f"var_{i}", 32)
            symbols.append(s)
            ctx.add_constraint_ast(s >= i)
            ctx.add_constraint_ast(s <= i + 100)

        assert ctx.satisfiable()
        assert ctx.num_constraints() == 120

        # Eval boundary values
        val_first = ctx.eval(symbols[0])
        val_last = ctx.eval(symbols[59])
        assert val_first is not None and 0 <= val_first <= 100
        assert val_last is not None and 59 <= val_last <= 159

    def test_solver_push_pop(self):
        """Push/pop preserves solver state correctly."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        ctx.push()
        ctx.add_constraint_ast(x == 15)
        assert ctx.eval(x) == 15

        ctx.pop()
        # After pop, x == 15 constraint is gone
        val = ctx.eval(x)
        assert 10 <= val <= 20

    def test_floordiv_mod_unsigned_semantics(self):
        """claripy // and % are UNSIGNED; the Rust import must agree (angr-p05r).

        For x == 0xFFFFFFFE (sign bit set), unsigned x//3 == 0x55555554 and
        x % 3 == 2.  A signed sdiv/srem import would produce different models,
        so this round-trips the dunder ASTs through the slow path and checks
        the Rust solver matches claripy's unsigned semantics.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        x = claripy.BVS("x", 32)

        # // (unsigned floordiv)
        ctx = RustSolverContext()
        ctx.add_constraint_ast(x == 0xFFFFFFFE)
        assert ctx.eval(x // 3) == 0x55555554
        # signed sdiv would give -2 // 3 == 0 (0x00000000), so guard against it
        assert ctx.eval(x // 3) != 0

        # % (unsigned mod)
        ctx2 = RustSolverContext()
        ctx2.add_constraint_ast(x == 0xFFFFFFFE)
        assert ctx2.eval(x % 3) == 2

        # SDiv / SMod must stay signed: SDiv(-2, 3) == 0, SMod(-2, 3) == -2.
        ctx3 = RustSolverContext()
        ctx3.add_constraint_ast(x == 0xFFFFFFFE)
        assert ctx3.eval(claripy.SDiv(x, 3)) == 0
        assert ctx3.eval(claripy.SMod(x, 3)) == 0xFFFFFFFE

    def test_deep_fork_chain(self):
        """Fork a state 20 levels deep and verify independence."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)
        state.set_register("rax", 0)

        states = [state]
        for i in range(1, 21):
            child = states[-1].fork()
            child.set_register("rax", i)
            child.memory_store(0x1000, bytes([i]))
            states.append(child)

        # Verify each level is independent
        for i, s in enumerate(states):
            assert s.get_register("rax") == i
            if i > 0:
                assert s.memory_load(0x1000, 1) == bytes([i])


class TestCallablePredicates:
    """Tests for callable find/avoid predicates with RustStateProxy."""

    def test_find_lambda_by_address(self, fauxware_project):
        """Callable find predicate matching by address works."""

        ACCEPTED = 0x4006ED
        REJECTED = 0x4006FD

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=lambda s: s.addr == ACCEPTED, avoid=REJECTED, max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state reaching accepted()"
        for s in mgr.found:
            assert s.addr == ACCEPTED

    def test_avoid_lambda_by_address(self, fauxware_project):
        """Callable avoid predicate matching by address works."""

        ACCEPTED = 0x4006ED
        REJECTED = 0x4006FD

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=ACCEPTED, avoid=lambda s: s.addr == REJECTED, max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state reaching accepted()"

    def test_solver_proxy_eval_returns_single_value(self):
        """RustSolverProxy.eval() returns a single value, not a tuple."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        from angr.exploration.rust_state_proxy import RustSolverProxy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        # RustSolverProxy normally lazily forks from rust_mgr, but we can
        # inject a pre-built solver context for unit testing.
        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        result = proxy.eval(x)
        assert isinstance(result, int), f"eval() should return int, got {type(result)}"
        assert 10 <= result <= 20

    def test_solver_proxy_eval_upto_returns_tuple(self):
        """RustSolverProxy.eval_upto() returns a tuple."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        from angr.exploration.rust_state_proxy import RustSolverProxy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 12)

        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        results = proxy.eval_upto(x, 5)
        assert isinstance(results, tuple), f"eval_upto() should return tuple, got {type(results)}"
        assert len(results) == 3  # exactly 10, 11, 12
        assert set(results) == {10, 11, 12}

    def test_solver_proxy_eval_cast_to_bytes(self):
        """RustSolverProxy.eval() with cast_to=bytes works."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        from angr.exploration.rust_state_proxy import RustSolverProxy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x41424344)

        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        result = proxy.eval(x, cast_to=bytes)
        assert isinstance(result, bytes), f"eval(cast_to=bytes) should return bytes, got {type(result)}"
        assert result == b"ABCD"
