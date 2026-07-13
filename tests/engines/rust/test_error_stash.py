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


class TestErroredStash:
    """Tests for the errored stash and RustErrorRecord."""

    def test_errored_returns_error_records(self, fauxware_project):
        """errored property returns RustErrorRecord objects with error details.

        De-vacuified (angr-9drh4): the prior version explored to a find
        address, which never populates the errored stash, so every per-record
        assertion lived inside a never-iterated loop — the test passed for any
        implementation. The high-level RustExplorationManager cannot be driven
        into the errored stash by unmapped/bad accesses (those are swallowed to
        zero buffers or routed to deadended), and loader/lazy pages are all
        mapped RWX (perms=7), so ENABLE_NX never fires on code pages. The one
        non-executable page the high-level sync DOES map is the stack page
        (perms=6, see rust_state_sync.py::_sync_stack_page / _setup_stack_region).
        Pointing the PC into the eagerly-mapped stack page with ENABLE_NX (and
        its STRICT_PAGE_ACCESS gate) on therefore produces a deterministic NX
        permission violation that lands in errored as a real RustErrorRecord.
        """
        from angr import sim_options as o
        from angr.exploration import RustErrorRecord

        state = fauxware_project.factory.entry_state(add_options={o.STRICT_PAGE_ACCESS, o.ENABLE_NX})
        # Jump into the (non-executable, perms=6) stack page. The fetch fails
        # the NX check before lifting, stashing the state into `errored`.
        sp = state.solver.eval(state.regs.sp)
        state.regs.pc = sp
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=5)

        errored = mgr.errored
        assert isinstance(errored, list)
        # Unconditional: the NX block fetch from the stack page MUST error,
        # so the per-record assertions below are actually exercised. Without
        # this a regression that stops populating errored (or routes NX
        # violations elsewhere) would silently re-vacuify the test.
        assert len(errored) >= 1, f"expected the NX stack-page fetch to error a state, got stashes={mgr.stash_counts()}"

        # Errored states should be RustErrorRecord instances with full detail.
        for record in errored:
            assert isinstance(record, RustErrorRecord)
            assert hasattr(record, "state")
            assert hasattr(record, "error")
            assert hasattr(record, "addr")
            assert hasattr(record, "error_class")
            assert hasattr(record, "registers")
            assert hasattr(record, "last_statements")
            assert hasattr(record, "constraint_count")
            assert isinstance(record.error, Exception)
            assert isinstance(record.error_class, str)
            assert isinstance(record.registers, dict)
            assert isinstance(record.last_statements, list)
            assert isinstance(record.constraint_count, int)
            assert repr(record).startswith("<State errored")

    def test_error_record_has_state(self):
        """RustErrorRecord wraps a state with error info."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "test error", 0x401000)
        assert str(record.error) == "test error"
        assert record.addr == 0x401000
        assert record.state is None
        assert "test error" in repr(record)
        assert "0x401000" in repr(record)

    def test_error_record_reraise(self):
        """RustErrorRecord.reraise() re-raises the stored error."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "test error", 0x401000)
        with pytest.raises(RuntimeError, match="test error"):
            record.reraise()

    def test_error_record_classifies_message(self):
        """error_class taxonomy matches CbExecutionError variant prefixes."""
        from angr.exploration import RustErrorRecord

        cases = {
            "memory error: unmapped 0xdead": "memory",
            "operation error: shift overflow": "operation",
            "invalid VEX IR: bogus": "invalid_ir",
            "unsupported: CAS instruction": "unsupported",
            "type mismatch: expected I64, got I32": "type_mismatch",
            "unknown temporary t42": "unknown_temp",
            "callback error: python raised": "callback",
            "lift error: bytes": "lift",
            "need lift at 0x401000": "need_lift",
            "need Python fallback: VECRET": "need_python_fallback",
            "resolve_function error: bad addr": "resolve_function",
            "something with timeout in it": "timeout",
            "page is unmapped at 0x0": "unmapped",
            "rust panic in interpreter": "rust_panic",
            "totally novel error": "unknown",
        }
        for msg, expected in cases.items():
            record = RustErrorRecord(None, msg, 0x400000)
            assert record.error_class == expected, (msg, record.error_class)

    def test_error_record_repr_includes_class(self):
        """repr() includes the classified error class for grep-friendliness."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "unsupported: CAS", 0x400000)
        assert "class=unsupported" in repr(record)

    def test_error_record_defaults_when_state_none(self):
        """Snapshot/history/constraint-count default cleanly when state is None."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "lift error: x", 0x400000)
        assert record.constraint_count == 0
        assert record.registers == {}
        assert record.last_statements == []


class TestErrorRecovery:
    """Tests that the engine degrades gracefully instead of panicking."""

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def test_unmapped_memory_load_returns_error(self):
        """Loading from unmapped memory should raise, not crash."""
        state = RustSimState("amd64")
        # No memory mapped — load should fail gracefully with ValueError
        with pytest.raises(ValueError, match="unmapped"):
            state.memory_load(0xDEAD0000, 4)

    def test_mapped_memory_load_succeeds(self):
        """Loading from mapped memory should return data."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)  # RWX
        data = state.memory_load(0x1000, 4)
        assert len(data) == 4

    def test_symbolic_load_to_unmapped_address_does_not_stay_active(self):
        """Symbolic load address pinned to a single unmapped value must NOT
        keep the state in `active`. It should land in `errored` (or at minimum
        `deadended`) — silently falling back to address 0 / a fresh symbol
        would diverge from Python angr (which raises SimUnsatError /
        SimMemoryAddressError under STRICT_PAGE_ACCESS).

        Locks down current behaviour for angr-ho6i: if the engine ever silently
        keeps the state in `active`, this test fails and a real bug is filed.
        """
        import claripy

        from angr import sim_options as o

        # mov rax, [rdi]   -> 48 8b 07
        # ret              -> c3
        # rdi will be constrained to 0xDEADBEEF, which is OUTSIDE the only
        # mapped region [0x1000, 0x2000) (the loaded shellcode page).
        shellcode = bytes.fromhex("488b07c3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.STRICT_PAGE_ACCESS},
        )

        rdi_sym = claripy.BVS("rdi_addr", 64)
        state.regs.rdi = rdi_sym
        state.solver.add(rdi_sym == 0xDEADBEEF)
        assert state.solver.satisfiable(), (
            "Sanity: rdi==0xDEADBEEF is satisfiable on its own; the *load* "
            "from that address is what should fail, not the constraint set."
        )

        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=10)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) == 0, (
            f"State remained in active despite load from unmapped 0xDEADBEEF "
            f"under STRICT_PAGE_ACCESS; stashes={counts} — engine silently "
            f"continued instead of erroring or dead-ending."
        )

    def test_dcas_cmpxchg16b_no_python_fallback(self):
        """`cmpxchg16b` lifts to a VEX `CAS` with `old_hi`/`expdHi`/`dataHi`
        populated. The Rust callback interpreter now handles DCAS natively
        (angr-ufez); the previous behavior was to return
        `DCAS_UNSUPPORTED_REASON` and let Python's VEX engine handle it.

        This test locks down the new path: profiling shows the block executed
        in Rust (`rust_blocks_executed > 0`), and the DCAS-unsupported counter
        stays at zero with no PythonVEXFallback recorded for the block.
        """

        # cmpxchg16b [rdi]   -> 48 0f c7 0f
        # ret                -> c3
        shellcode = bytes.fromhex("480fc70fc3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        # rdi must be 16-byte aligned or VEX raises Ijk_SigSEGV before CAS.
        state.regs.rdi = 0x2000
        state.memory.store(0x2000, b"\x00" * 16)
        state.regs.rax = 0
        state.regs.rdx = 0
        state.regs.rbx = 0xDEADBEEF
        state.regs.rcx = 0xCAFEBABE
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state])
        mgr.enable_profiling()
        mgr.run(max_steps=1)

        # DCAS handled natively → the unsupported counter stays at zero, no
        # PythonVEXFallback recorded for the block.
        stats = mgr.stats
        fb = mgr._rust_mgr.get_fallback_stats()
        assert stats["dcas_unsupported_count"] == 0, f"DCAS now native — counter must stay zero; stats={stats}, fb={fb}"
        assert fb["dcas_unsupported_count"] == 0
        assert not any("double compare-and-swap" in reason for reason in fb["addresses"].values()), (
            f"DCAS reason still recorded in fallback addrs: {fb['addresses']}"
        )
        # Block was actually run by the Rust interpreter (not silently skipped):
        # the DCAS path issues exactly 2 stores (lo + hi halves). Any DCAS that
        # falls back early or returns Unsupported produces 0.
        assert stats["rust_step_count"] > 0, f"Rust did not step the state; stats={stats}"
        assert stats["rust_store_stmt_count"] >= 2, (
            f"DCAS lo/hi stores not fired; rust_store_stmt_count={stats['rust_store_stmt_count']}; stats={stats}"
        )

    def test_dcas_cmpxchg16b_no_match_keeps_memory(self):
        """DCAS failure path — when (rdx:rax) does NOT match `[m128]`, the
        store is suppressed and memory keeps its original value (set up via
        Python prior to Rust taking over). This locks down the cmp-false
        branch of `execute_cas_stmt`.
        """

        shellcode = bytes.fromhex("480fc70fc3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rdi = 0x2000
        # Memory holds 0xAAAA...BBBB pattern, expected says 0:0 → mismatch.
        state.memory.store(0x2000, b"\xaa" * 8 + b"\xbb" * 8)
        state.regs.rax = 0
        state.regs.rdx = 0
        state.regs.rbx = 0xDEADBEEF
        state.regs.rcx = 0xCAFEBABE
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        # save_unconstrained=True: the shellcode is `cmpxchg16b [rdi]; ret`,
        # and the trailing ret runs against an empty call stack — angr-3uye.2
        # routes that to the unconstrained stash. Without saving it, the
        # post-step state would be dropped before we can inspect memory.
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.enable_profiling()
        mgr.run(max_steps=1)

        assert mgr.stats["dcas_unsupported_count"] == 0
        assert mgr.stats["rust_step_count"] > 0
        # cmp-false path: zero CAS-driven stores.
        assert mgr.stats["rust_store_stmt_count"] == 0, (
            f"cmp-false DCAS should not store; got {mgr.stats['rust_store_stmt_count']}"
        )
        survived = list(mgr.active) + list(mgr.found) + list(mgr.deadended) + list(mgr.unconstrained)
        assert survived, "DCAS block produced no survived state"
        s = survived[0]

        # Memory is unchanged (no store fired): exercises the cmp-false path.
        mem_lo = s.solver.eval(s.memory.load(0x2000, 8, endness="Iend_LE"))
        mem_hi = s.solver.eval(s.memory.load(0x2008, 8, endness="Iend_LE"))
        assert mem_lo == 0xAAAAAAAAAAAAAAAA, f"mem_lo changed: 0x{mem_lo:x}"
        assert mem_hi == 0xBBBBBBBBBBBBBBBB, f"mem_hi changed: 0x{mem_hi:x}"

    def test_lock_cmpxchg_qword_match_updates_memory(self):
        """Single-word `LOCK CMPXCHG [mem], reg` lifts to a VEX `CAS` with
        all of `oldHi`/`expdHi`/`dataHi` absent — the non-DCAS branch of
        `execute_cas_stmt`. When RAX matches `[m64]`, the data is stored
        and memory ends up holding the source register's value.

        Locks down the angr-4enc fix to the IRSB serializer: pyvex reports
        `oldHi=0xFFFFFFFF` (IRTemp_INVALID sentinel) for single-word CAS
        and the Python-side JSON bridge has to map the sentinel to None
        so the Rust validation accepts the all-Some/all-None shape.
        Before the fix this test errored with "CAS: oldHi/expdHi/dataHi
        must be all-Some (DCAS) or all-None (single)".
        """
        import claripy

        # lock cmpxchg qword ptr [rdi], rcx  -> f0 48 0f b1 0f
        # ret                                -> c3
        shellcode = bytes.fromhex("f0480fb10fc3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rdi = 0x2000
        state.memory.store(0x2000, claripy.BVV(0x1234, 64), endness="Iend_LE")
        state.regs.rax = 0x1234  # matches [rdi]
        state.regs.rcx = 0xDEADBEEFCAFEBABE  # new value to write on match
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.enable_profiling()
        mgr.run(max_steps=1)

        # Block ran natively, the single-word CAS path fired exactly one
        # store (no hi half).
        assert mgr.stats["rust_step_count"] > 0
        assert mgr.stats["dcas_unsupported_count"] == 0
        assert mgr.stats["rust_store_stmt_count"] >= 1, (
            f"single-word CAS should issue >=1 store on match; got {mgr.stats['rust_store_stmt_count']}"
        )

        survived = list(mgr.active) + list(mgr.found) + list(mgr.deadended) + list(mgr.unconstrained)
        assert survived, "CMPXCHG block produced no survived state"
        s = survived[0]
        mem = s.solver.eval(s.memory.load(0x2000, 8, endness="Iend_LE"))
        assert mem == 0xDEADBEEFCAFEBABE, f"mem not updated: 0x{mem:x}"

    def test_lock_cmpxchg_qword_no_match_keeps_memory(self):
        """Mismatch branch of the non-DCAS CAS path: when RAX does NOT
        match `[m64]`, the store is suppressed and memory keeps its
        original value. The CPU also writes the loaded value back to RAX
        (so subsequent compares see the actual contents).
        """
        import claripy

        shellcode = bytes.fromhex("f0480fb10fc3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rdi = 0x2000
        state.memory.store(0x2000, claripy.BVV(0xAAAAAAAAAAAAAAAA, 64), endness="Iend_LE")
        state.regs.rax = 0x1234  # NO match (mem holds 0xAA...)
        state.regs.rcx = 0xDEADBEEF
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.enable_profiling()
        mgr.run(max_steps=1)

        assert mgr.stats["rust_step_count"] > 0
        # cmp-false: zero CAS-driven stores.
        assert mgr.stats["rust_store_stmt_count"] == 0, (
            f"cmp-false single CAS should not store; got {mgr.stats['rust_store_stmt_count']}"
        )

        survived = list(mgr.active) + list(mgr.found) + list(mgr.deadended) + list(mgr.unconstrained)
        assert survived, "CMPXCHG block produced no survived state"
        s = survived[0]
        # Memory is unchanged.
        mem = s.solver.eval(s.memory.load(0x2000, 8, endness="Iend_LE"))
        assert mem == 0xAAAAAAAAAAAAAAAA, f"mem changed: 0x{mem:x}"
        # CPU semantics: on mismatch, RAX is loaded with current `[m64]`.
        rax = s.solver.eval(s.regs.rax)
        assert rax == 0xAAAAAAAAAAAAAAAA, f"rax not updated to loaded value: 0x{rax:x}"

    def test_arm_ldrex_lifts_to_llsc_load_linked(self):
        """ARM LDREX lifts to VEX `LLSC` with `storedata=None` — the
        load-linked branch of the LLSC statement handler. The handler
        synthesizes an `IRExpr::Load` from the address temp and writes
        the loaded value into the result temp (which the lifted IR
        then puts into the destination register).
        """
        import claripy

        # ldrex r0, [r1]: 0xE1910F9F -> bytes LE: 9F 0F 91 E1
        # bx lr        : 0xE12FFF1E -> bytes LE: 1E FF 2F E1
        shellcode = bytes.fromhex("9f0f91e11eff2fe1")
        proj = angr.load_shellcode(shellcode, arch="ARM", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.r1 = 0x2000
        state.memory.store(0x2000, claripy.BVV(0xDEADBEEF, 32), endness="Iend_LE")
        state.regs.sp = 0x7FFFFE00
        state.regs.lr = 0x9000

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.enable_profiling()
        mgr.run(max_steps=2)

        assert mgr.stats["rust_step_count"] > 0
        survived = list(mgr.active) + list(mgr.found) + list(mgr.deadended) + list(mgr.unconstrained)
        assert survived, "LDREX block produced no survived state"
        s = survived[0]
        r0 = s.solver.eval(s.regs.r0)
        assert r0 == 0xDEADBEEF, f"r0 not loaded by LDREX: 0x{r0:x}"

    def test_arm_strex_lifts_to_llsc_store_conditional(self):
        """ARM STREX lifts to VEX `LLSC` with `storedata=Some(...)` —
        the store-conditional branch of the LLSC statement handler.
        In single-state symex the store always succeeds, so `[Rn]`
        receives Rt's value. The lifted IR also writes the *inverted*
        LLSC success bit into Rd (the status register), so a successful
        STREX leaves Rd = 0.
        """
        import claripy

        # strex r0, r2, [r1]: 0xE1810F92 -> bytes LE: 92 0F 81 E1
        # bx lr             : 0xE12FFF1E -> bytes LE: 1E FF 2F E1
        shellcode = bytes.fromhex("920f81e11eff2fe1")
        proj = angr.load_shellcode(shellcode, arch="ARM", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.r1 = 0x2000
        state.regs.r2 = claripy.BVV(0xCAFEBABE, 32)
        state.memory.store(0x2000, claripy.BVV(0xAAAAAAAA, 32), endness="Iend_LE")
        state.regs.sp = 0x7FFFFE00
        state.regs.lr = 0x9000

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.enable_profiling()
        mgr.run(max_steps=2)

        assert mgr.stats["rust_step_count"] > 0
        survived = list(mgr.active) + list(mgr.found) + list(mgr.deadended) + list(mgr.unconstrained)
        assert survived, "STREX block produced no survived state"
        s = survived[0]
        mem = s.solver.eval(s.memory.load(0x2000, 4, endness="Iend_LE"))
        assert mem == 0xCAFEBABE, f"STREX did not store r2 to [r1]: 0x{mem:x}"
        r0 = s.solver.eval(s.regs.r0)
        assert r0 == 0, f"STREX success status (inverted LLSC bit) not 0: 0x{r0:x}"

    def test_unsat_state_pruned_during_step(self):
        """State with pre-existing contradictory constraints (x>100 AND x<50)
        must NOT remain in `active` after stepping. Locks down behaviour for
        angr-eldx — engine eagerly checks fork satisfiability and prunes
        UNSAT children, so an UNSAT parent is evicted via the prune path
        rather than silently advancing.
        """
        import claripy

        # mov rax, 1     -> 48 c7 c0 01 00 00 00
        # cmp rax, 0     -> 48 83 f8 00
        # jne +1         -> 75 01
        # nop            -> 90
        # ret            -> c3
        shellcode = bytes.fromhex("48c7c0010000004883f8007501 90c3".replace(" ", ""))
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        x = claripy.BVS("contradiction", 32)
        state.solver.add(x > 100)
        state.solver.add(x < 50)
        assert not state.solver.satisfiable(), "Sanity: Python solver also sees the contradiction."

        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=20)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) == 0, (
            f"State with contradictory constraints (x>100 AND x<50) "
            f"remained in active after step; stashes={counts} — engine "
            f"failed to detect UNSAT and continued executing."
        )

    def test_register_invalid_name(self):
        """Getting a nonexistent register should raise, not crash."""
        state = RustSimState("amd64")
        with pytest.raises(ValueError, match="cannot read register"):
            state.get_register("nonexistent_register_xyz")

    def test_register_set_get_roundtrip(self):
        """Setting and getting a register preserves the value."""
        state = RustSimState("amd64")
        state.set_register("rax", 0xCAFEBABE)
        assert state.get_register("rax") == 0xCAFEBABE

    def test_solver_eval_after_unsat(self):
        """Evaluating an expression on an UNSAT solver should return None, not crash."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)  # contradicts
        assert not ctx.satisfiable()
        # Contract (solver.rs eval / solving_ops.rs): eval on an UNSAT solver
        # returns None rather than a spurious witness.
        assert ctx.eval(x) is None

    def test_solver_min_on_unsat(self):
        """min() on UNSAT solver should return None, not crash."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        result = ctx.min(x, signed=False)
        # UNSAT must short-circuit to None, not a garbage extremum.
        assert result is None

    def test_solver_eval_upto_on_unsat(self):
        """eval_upto() on UNSAT solver returns empty list, not crash."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        results = ctx.eval_upto(x, 5)
        # UNSAT yields no solutions: the list must be empty, not spurious values.
        assert results == []

    def test_solver_many_constraints(self):
        """Solver handles many constraints without crashing."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        # Add 100 redundant but consistent constraints
        for i in range(100):
            ctx.add_constraint_ast(x >= i)
        ctx.add_constraint_ast(x <= 200)
        assert ctx.satisfiable()
        val = ctx.eval(x)
        assert 99 <= val <= 200

    def test_solver_fork_many_times(self):
        """Forking solver many times should not leak or crash."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 0)

        forks = []
        for i in range(50):
            f = ctx.fork()
            f.add_constraint_ast(x == i)
            forks.append(f)

        # All forks should be independent
        for i, f in enumerate(forks):
            assert f.eval(x) == i

    def test_state_fork_preserves_registers(self):
        """Forking a state preserves register values."""
        state = RustSimState("amd64")
        state.set_register("rax", 42)
        state.set_register("rbx", 99)

        forked = state.fork()
        assert forked.get_register("rax") == 42
        assert forked.get_register("rbx") == 99

        # Modifying fork doesn't affect original
        forked.set_register("rax", 100)
        assert state.get_register("rax") == 42
        assert forked.get_register("rax") == 100

    def test_state_fork_preserves_memory(self):
        """Forking a state preserves memory contents (CoW)."""
        state = RustSimState("amd64")
        state.map_memory_data(0x1000, b"\x41\x42\x43\x44", 7)

        forked = state.fork()
        data = forked.memory_load(0x1000, 4)
        assert bytes(data) == b"\x41\x42\x43\x44"

    def test_exploration_manager_empty_run(self):
        """Running with no callbacks set should raise, not crash silently."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000])
        # No callbacks set — run should raise RuntimeError
        with pytest.raises(RuntimeError, match="callbacks not set"):
            mgr.run(100)

    # angr-2i4n: error path coverage — make the strict, user-facing failure
    # modes explicit so we notice if the engine ever starts silently
    # papering over them (phantom symbolic zero, unbounded Z3 hang, etc.).

    def test_unmapped_concrete_store_returns_error(self):
        """Storing to unmapped memory must raise — not silently succeed.

        Mirror of test_unmapped_memory_load_returns_error for the write
        side: the store path has its own start_page/end_page check
        (memory.rs ~750) and we want a regression that catches a future
        change that would auto-map on demand without permissions.
        """
        state = RustSimState("amd64")
        with pytest.raises(ValueError, match="unmapped"):
            state.memory_store(0xDEAD0000, b"\x41\x42\x43\x44")

    def test_cross_page_permission_write_returns_error(self):
        """Cross-page write where the second page is R-only must surface a
        permission violation, not silently overwrite the read-only page.

        The Rust unit tests in memory.rs already cover this via
        SymbolicMemory directly (test_permission_enforcement_cross_page_write
        and test_permission_enforcement_unaligned_store_two_pages_*); this
        test locks the same invariant down at the Python/RustSimState
        boundary so that a regression in the FFI wiring (e.g. forgetting
        to forward set_enforce_permissions) does not go unnoticed.
        """
        state = RustSimState("amd64")
        # Page 0 RW, page 1 R-only.
        state.map_memory(0x1000, 0x1000, 0x6)  # R|W
        state.map_memory(0x2000, 0x1000, 0x4)  # R only
        state.set_enforce_permissions(True)

        # 4-byte write straddling 0x1FFE..0x2002 hits the R-only page.
        with pytest.raises(ValueError, match="permission"):
            state.memory_store(0x1FFE, b"\x11\x22\x33\x44")

        # Sanity: a write entirely within the RW page still succeeds.
        state.memory_store(0x1000, b"\x55\x66\x77\x88")

    # angr-8e81: narrow exception types in _cb_memory_load / _cb_memory_store.
    # The callbacks used to swallow `except Exception`, masking unrelated bugs
    # (e.g. AttributeError, RuntimeError) by returning a zero buffer. These
    # tests pin down the new contract: only Sim*/Claripy errors are caught;
    # everything else propagates.

    def _build_load_store_manager(self):

        shellcode = bytes.fromhex("c3")  # ret
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])
        return mgr, state

    def test_cb_memory_load_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() is the expected failure mode
        and must keep being swallowed (zero buffer fallback)."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped read")

        state.memory.load = boom

        mgr._set_callback_state(state)

        result = mgr._cb_memory_load(0x1000, 4)
        assert result == (bytes(4), False, None)

    def test_cb_memory_load_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions (e.g. RuntimeError from a real bug)
        must propagate out of _cb_memory_load instead of being silently
        masked as a zero buffer."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected bug")

        state.memory.load = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected bug"):
            mgr._cb_memory_load(0x1000, 4)

    def test_cb_memory_store_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() is the expected failure
        mode and must keep being swallowed."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped write")

        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        mgr._cb_memory_store(0x1000, b"\x41\x42\x43\x44")

    def test_cb_memory_store_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_store."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected bug")

        state.memory.store = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected bug"):
            mgr._cb_memory_store(0x1000, b"\x41\x42\x43\x44")

    # angr-2f7o: extend the angr-8e81 contract to the remaining callbacks.
    # Each callback must:
    #   1. swallow Sim*/Claripy/PyVEX errors (legitimate symbolic-engine failures)
    #   2. propagate RuntimeError (or any other unrelated exception) so real
    #      bugs surface instead of being masked as zero buffers / empty pages.

    def _put_state_in_default_cache(self, mgr, state):
        """Inject `state` so _get_default_state() returns it. Several
        callbacks (fetch_page, sync_constraints, batch_fetch_pages) read
        through _state_cache rather than _callback_state."""
        mgr._state_cache[id(state)] = state

    def test_cb_lift_block_swallows_pyvex_error(self):
        """PyVEXError from the lifter must keep falling back to '{}'."""
        from pyvex.errors import PyVEXError

        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise PyVEXError("simulated lift failure")

        mgr._project.factory.block = boom

        result = mgr._cb_lift_block(0x1000)
        assert result == "{}"

    def test_cb_lift_block_swallows_sim_engine_error(self):
        """SimEngineError from block construction must keep falling back."""
        from angr.errors import SimEngineError

        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimEngineError("simulated engine failure")

        mgr._project.factory.block = boom

        result = mgr._cb_lift_block(0x1000)
        assert result == "{}"

    def test_cb_lift_block_propagates_unrelated_exceptions(self):
        """Non-(SimEngine/Claripy/PyVEX) exceptions must propagate."""
        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected lift bug")

        mgr._project.factory.block = boom

        with pytest.raises(RuntimeError, match="unexpected lift bug"):
            mgr._cb_lift_block(0x1000)

    # angr-kwwd: SMC support — when Rust signals dirty_bytes, _cb_lift_block
    # must lift those bytes (via byte_string=) instead of the cle static
    # buffer. Without this, the engine executes stale instructions after a
    # store overwrites in-binary code.
    def test_cb_lift_block_uses_dirty_bytes_when_provided(self):
        """dirty_bytes lifts the supplied buffer, not the project's static binary."""
        import json

        # Project's static binary at 0x1000 is "ret" (0xc3, 1 byte).
        proj = angr.load_shellcode(b"\xc3", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])

        # dirty_bytes is "nop; ret" (0x90 0xc3); the lift must reflect the
        # extra nop, proving cle's stale buffer was bypassed.
        result = mgr._cb_lift_block(0x1000, dirty_bytes=b"\x90\xc3")
        assert result and result != "{}", f"expected non-empty IRSB, got {result!r}"
        irsb = json.loads(result)
        imarks = [s for s in irsb["statements"] if s.get("tag") == "Ist_IMark"]
        # Two IMarks (nop at 0x1000, ret at 0x1001) prove both bytes were lifted.
        assert len(imarks) == 2, f"expected 2 IMarks (nop+ret), got {len(imarks)}: {imarks}"
        assert imarks[0]["addr"] == 0x1000
        assert imarks[1]["addr"] == 0x1001

    def test_cb_lift_block_static_binary_when_no_dirty_bytes(self):
        """Without dirty_bytes the lift comes from the project's static binary."""
        import json

        proj = angr.load_shellcode(b"\xc3", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])

        result = mgr._cb_lift_block(0x1000)
        assert result and result != "{}"
        irsb = json.loads(result)
        imarks = [s for s in irsb["statements"] if s.get("tag") == "Ist_IMark"]
        # Static binary is just 0xc3 (1 byte ret) — exactly one IMark.
        assert len(imarks) == 1, f"expected 1 IMark (ret only), got {len(imarks)}"
        assert imarks[0]["addr"] == 0x1000

    def test_smc_rust_passes_dirty_bytes_to_python_lift(self):
        """End-to-end SMC: a Rust-side store to in-binary code marks the page
        dirty; the next lift must call _cb_lift_block with dirty_bytes from
        rust_memory (not None / not the cle static buffer).

        This locks down the angr-kwwd contract: without the fix, _cb_lift_block
        is called with dirty_bytes=None and lifts cle's stale binary buffer,
        causing the engine to execute the pre-store instructions.
        """
        import angr.sim_options as o

        # Layout at 0x1000:
        #   0x1000: 48 b8 10 10 00 00 00 00 00 00  mov rax, 0x1010
        #   0x100a: c6 00 c3                       mov byte [rax], 0xc3
        #   0x100d: ff e0                          jmp rax            (PC -> 0x1010)
        #   0x100f: 00                             padding
        #   0x1010: 90 c3                          nop ; ret  (static)
        #   After the store: 0x1010 = c3 c3 (ret ; ret)
        shellcode = (
            bytes.fromhex("48b81010000000000000")  # mov rax, 0x1010
            + bytes.fromhex("c600c3")  # mov byte [rax], 0xc3
            + bytes.fromhex("ffe0")  # jmp rax
            + b"\x00"  # padding to align 0x1010
            + bytes.fromhex("90c3")  # nop ; ret (static at 0x1010)
        )
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        state.regs.rsp = 0x7FFFF0000
        state.memory.store(0x7FFFF0000, b"\x00" * 8)

        # use_native_lift=False: this test pins the *callback* dirty-bytes
        # contract, and the native libVEX seam bypasses _cb_lift_block
        # entirely (it reads the live bytes straight out of rust_memory, so
        # it gets SMC freshness without the dirty_bytes plumbing — see
        # test_smc_native_lift_executes_stored_bytes). Without this the test
        # records zero lifts at 0x1010 on a --features libvex-ffi build.
        mgr = RustExplorationManager(proj, [state], use_native_lift=False)
        # load_shellcode's cle Blob has obj.binary == None, so the default
        # _load_binary_regions skips it and `is_in_binary` returns False
        # everywhere — the engine then treats the jmp at 0x100d as an
        # UnmodeledCall and never lifts 0x1010. Register the shellcode
        # bytes manually so 0x1000-0x1011 counts as "in binary".
        # See bd memory load-shellcode-blob-binary-none.
        mgr._rust_mgr.load_binary_regions([(0x1000, shellcode)])

        # Wrap _cb_lift_block to record (addr, dirty_bytes) for each call,
        # then re-install on the Rust side (set_callbacks takes a clone, so
        # we must push the updated PythonCallbacks back).
        recorded: list[tuple[int, bytes | None]] = []
        original = mgr._cb_lift_block

        def wrapper(addr, opt_level=None, dirty_bytes=None):
            recorded.append((addr, dirty_bytes))
            return original(addr, opt_level, dirty_bytes)

        mgr._callbacks.set_lift_block(wrapper)
        mgr._rust_mgr.set_callbacks(mgr._callbacks)

        mgr.run(max_steps=10)

        smc_lifts = [(a, b) for (a, b) in recorded if a == 0x1010]
        assert smc_lifts, (
            f"expected at least one lift at 0x1010 after SMC store; "
            f"all lifts: {[(hex(a), b is not None) for a, b in recorded]}"
        )
        _last_addr, last_bytes = smc_lifts[-1]
        assert last_bytes is not None, (
            "Rust failed to pass dirty_bytes for SMC lift at 0x1010; "
            f"all lifts: {[(hex(a), b is not None) for a, b in recorded]}"
        )
        assert last_bytes[0] == 0xC3, (
            f"dirty_bytes[0] should be 0xc3 (the byte just stored), got "
            f"0x{last_bytes[0]:02x} (full: {last_bytes[:8].hex()})"
        )

    def test_smc_native_lift_executes_stored_bytes(self):
        """The native libVEX seam must honor SMC too.

        ``try_native_lift`` sources block bytes from ``rust_memory``, so a
        store to in-binary code is visible to it without any dirty_bytes
        plumbing — but that is only true if the lift really reads the state's
        memory and not the (stale) cle binary image. Distinguish the two
        behaviorally: the static byte at 0x1010 is an infinite self-loop
        (``jmp $``), and the store overwrites it with ``ret``. Executing the
        stale image loops forever (a state is still active at max_steps);
        executing the stored bytes returns and drains the active stash.

        No-op on a build without ``--features libvex-ffi`` (the manager's
        native-lift default is inert there and the callback path, covered by
        test_smc_rust_passes_dirty_bytes_to_python_lift, serves the lift).
        """
        import angr.sim_options as o

        # Same layout as the callback test, except 0x1010 is `eb fe` (jmp $)
        # statically; the store at 0x100a rewrites its first byte to 0xc3,
        # turning it into `ret` (the trailing 0xfe is then unreachable).
        shellcode = (
            bytes.fromhex("48b81010000000000000")  # mov rax, 0x1010
            + bytes.fromhex("c600c3")  # mov byte [rax], 0xc3
            + bytes.fromhex("ffe0")  # jmp rax
            + b"\x00"  # padding to align 0x1010
            + bytes.fromhex("ebfe")  # jmp $ (static at 0x1010)
        )
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        state.regs.rsp = 0x7FFFF0000
        state.memory.store(0x7FFFF0000, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state], use_native_lift=True)
        # See load-shellcode-blob-binary-none (as in the callback test).
        mgr._rust_mgr.load_binary_regions([(0x1000, shellcode)])
        # The interpreter lift counters are only published under profiling.
        mgr.enable_profiling()
        mgr.run(max_steps=10)

        assert not mgr.stashes.get("active"), (
            "a state is still active after 10 steps — the lift at 0x1010 saw "
            "the stale `jmp $` from the binary image instead of the stored "
            f"`ret`; stashes: { {k: len(v) for k, v in mgr.stashes.items() if v} }"
        )

        # On a feature-on build the assertion above is only meaningful if the
        # native seam actually served a lift — otherwise this silently
        # degrades into a second copy of the callback-path test.
        try:
            from angr.rustylib.vex_engine import libvex_ffi_enabled
        except ImportError:
            libvex_ffi_enabled = None
        if libvex_ffi_enabled is not None and libvex_ffi_enabled():
            assert mgr.stats.get("rust_native_lift_count", 0) > 0, (
                "libvex-ffi build with use_native_lift=True lifted nothing "
                "natively; the SMC assertion above did not exercise the "
                "native seam"
            )

    def test_cb_fetch_page_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning empty page."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped page")

        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        result = mgr._cb_fetch_page(0x1000)
        assert result == (bytes(4096), 0, False)

    def test_cb_fetch_page_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_fetch_page."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected fetch bug")

        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        with pytest.raises(RuntimeError, match="unexpected fetch bug"):
            mgr._cb_fetch_page(0x1000)

    def test_rust_ctx_missing_counter_starts_at_zero(self):
        """angr-h0dv: defensive `rust_ctx_missing` counter on a fresh
        manager. Path A (rust_solver_ctx attach in
        rust_callback_dispatch.py::_install_rust_solver_on_callback_state)
        covers every live callback site; non-zero here signals a callback
        site forgot to attach the Rust solver context. The legacy Path B
        constraint-sync counters (cb_sync_calls, cb_sync_constraints,
        cb_sync_failures, pending_ast_sync_calls) were retired in this
        bead after a 20-bench soak proved they stayed at 0."""
        mgr, _ = self._build_load_store_manager()
        stats = mgr.stats
        assert stats["rust_ctx_missing"] == 0
        # Retired counters must not reappear in stats.
        for retired in (
            "cb_sync_calls",
            "cb_sync_constraints",
            "cb_sync_failures",
            "pending_ast_sync_calls",
        ):
            assert retired not in stats, f"retired counter resurfaced: {retired}"
        # _cb_sync_constraints method is also gone.
        assert not hasattr(mgr, "_cb_sync_constraints")
        # _sync_rust_constraints_to_python lived on the state-sync mixin.
        assert not hasattr(mgr, "_sync_rust_constraints_to_python")

    def test_cb_memory_store_batch_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must keep being swallowed
        per-store in the batch loop (other stores still attempted)."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch store")

        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        mgr._cb_memory_store_batch([(0x1000, b"\x41\x42\x43\x44")])

    def test_cb_memory_store_batch_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_store_batch."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch store bug")

        state.memory.store = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected batch store bug"):
            mgr._cb_memory_store_batch([(0x1000, b"\x41\x42\x43\x44")])

    def test_cb_memory_load_batch_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning a
        per-entry zero buffer in the batch loop."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch load")

        state.memory.load = boom

        mgr._set_callback_state(state)

        result = mgr._cb_memory_load_batch([(0x1000, 4)])
        assert result == [(bytes(4), False, None)]

    def test_cb_memory_load_batch_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_load_batch."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch load bug")

        state.memory.load = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected batch load bug"):
            mgr._cb_memory_load_batch([(0x1000, 4)])

    def test_cb_batch_fetch_pages_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning empty
        pages per-entry in the batch loop."""
        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch page")

        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        result = mgr._cb_batch_fetch_pages([0x1000])
        assert result == [(bytes(4096), 0, True)]

    def test_cb_batch_fetch_pages_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_batch_fetch_pages."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch page bug")

        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        with pytest.raises(RuntimeError, match="unexpected batch page bug"):
            mgr._cb_batch_fetch_pages([0x1000])

    def test_cb_memory_store_symbolic_value_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must keep being swallowed."""
        import claripy

        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic store")

        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        ast = claripy.BVS("sym_store", 32)
        mgr._cb_memory_store_symbolic_value(0x1000, ast)

    def test_cb_memory_store_symbolic_value_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of
        _cb_memory_store_symbolic_value."""
        import claripy

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic store bug")

        state.memory.store = boom

        mgr._set_callback_state(state)

        ast = claripy.BVS("sym_store", 32)
        with pytest.raises(RuntimeError, match="unexpected symbolic store bug"):
            mgr._cb_memory_store_symbolic_value(0x1000, ast)

    # angr-b1qq: wire memory_{store,load}_symbolic_full so the existing
    # interpreter fallbacks (statements.rs:368/444/710/1174,
    # expressions.rs:149) can actually delegate to Python instead of returning
    # CbExecutionError::Unsupported on TooLarge symbolic addresses.

    def test_symbolic_full_callbacks_are_wired(self):
        """_init_callbacks must bind both *_symbolic_full callbacks. Without
        them the Rust TooLarge branches at interpreter/statements.rs:368/
        444/710/1174 and expressions.rs:149 hard-error instead of falling
        back to Python's memory model."""
        mgr, _ = self._build_load_store_manager()
        # Bound-method presence is the contract: _init_callbacks must have
        # called set_memory_{store,load}_symbolic_full with the corresponding
        # method. The setters are unconditional in PythonCallbacks (see
        # callbacks.rs:560/566).
        assert callable(getattr(mgr, "_cb_memory_store_symbolic_full", None))
        assert callable(getattr(mgr, "_cb_memory_load_symbolic_full", None))

    def test_cb_memory_store_symbolic_full_round_trip(self):
        """Storing a symbolic value at a (pinned) symbolic address via the
        full callback must land in Python state memory and be loadable back.
        Uses 0x4000 because 0x1000 is already mapped by load_shellcode."""
        import claripy

        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)

        target_addr = 0x4000
        state.memory.map_region(target_addr, 0x100, 7)
        addr = claripy.BVS("sym_addr", 64)
        state.solver.add(addr == target_addr)
        data = claripy.BVV(0xCAFEBABE, 32)

        mgr._cb_memory_store_symbolic_full(addr, data)

        loaded = state.memory.load(
            target_addr, 4, endness=state.arch.memory_endness, inspect=False, disable_actions=True
        )
        assert state.solver.eval(loaded) == 0xCAFEBABE

    def test_cb_memory_load_symbolic_full_returns_stored_ast(self):
        """Loading via the full callback must return a claripy AST that
        evaluates to the previously stored value at a symbolic-but-pinned
        address."""
        import claripy

        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)

        target_addr = 0x4000
        state.memory.map_region(target_addr, 0x100, 7)
        state.memory.store(
            target_addr,
            claripy.BVV(0xDEADBEEF, 32),
            endness=state.arch.memory_endness,
            inspect=False,
            disable_actions=True,
        )

        addr = claripy.BVS("sym_load_addr", 64)
        state.solver.add(addr == target_addr)

        result = mgr._cb_memory_load_symbolic_full(addr, 4)
        assert result is not None
        assert state.solver.eval(result) == 0xDEADBEEF

    def test_cb_memory_store_symbolic_full_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must be swallowed (matches
        the angr-8e81 / angr-2f7o convention used by the other symbolic
        callbacks). Otherwise Rust's flush_stores() path would propagate the
        error and tear down exploration."""
        import claripy

        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic-address store")

        state.memory.store = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        data = claripy.BVS("sym_data", 32)
        # Should not raise.
        mgr._cb_memory_store_symbolic_full(addr, data)

    def test_cb_memory_store_symbolic_full_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate so real bugs aren't
        masked as silent no-ops."""
        import claripy

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic-address store bug")

        state.memory.store = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        data = claripy.BVS("sym_data", 32)
        with pytest.raises(RuntimeError, match="unexpected symbolic-address store bug"):
            mgr._cb_memory_store_symbolic_full(addr, data)

    def test_cb_memory_load_symbolic_full_swallows_sim_memory_error(self):
        """SimMemoryError must be swallowed and a fresh symbolic AST returned
        so Rust can keep going (it'll wrap the result in a sym_pyref_*
        placeholder via expressions.rs)."""
        import claripy

        from angr.errors import SimMemoryError

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic-address load")

        state.memory.load = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        result = mgr._cb_memory_load_symbolic_full(addr, 4)
        # Returns something valid for Rust to consume — width must match.
        assert result is not None
        assert getattr(result, "length", None) == 32

    def test_cb_memory_load_symbolic_full_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate."""
        import claripy

        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic-address load bug")

        state.memory.load = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        with pytest.raises(RuntimeError, match="unexpected symbolic-address load bug"):
            mgr._cb_memory_load_symbolic_full(addr, 4)

    # Phase 1.4 (angr-5zw8): _cb_memory_store_symbolic_full must route
    # MultiwriteAnnotation-tagged addresses through the Rust Multi-cell
    # store path (state_memory_store_symbolic_multi) instead of falling
    # back to Python's state.memory.store.  These tests pin the routing
    # contract.  End-to-end correctness of the Rust Multi-cell helpers
    # themselves is covered by Rust unit tests in
    # native/angr/src/memory/tests.rs (Phase 1.3 / angr-aija).

    def test_try_multi_cell_store_skips_when_no_annotation(self):
        """No MultiwriteAnnotation on addr → fall through (return False).
        This is the common case: every other symbolic-store callback hits
        this path and the existing Python state.memory.store handles it."""
        import claripy

        mgr, _ = self._build_load_store_manager()
        addr = claripy.BVS("plain_addr", 64)
        data = claripy.BVV(0xCAFEBABE, 32)
        assert mgr._try_multi_cell_store(addr, data) is False

    def test_try_multi_cell_store_skips_when_state_id_unknown(self):
        """MultiwriteAnnotation present but no stepping state id → fall
        through (return False).  Outside an active step there is no
        Rust state to apply the Multi-cell store to."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, _ = self._build_load_store_manager()
        # Force no stepping state id (mirrors the off-path callback case).
        mgr._get_stepping_state_id = lambda: None
        addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
        data = claripy.BVV(0xCAFEBABE, 32)
        assert mgr._try_multi_cell_store(addr, data) is False

    def test_try_multi_cell_store_routes_to_pyo3_when_annotated(self):
        """MultiwriteAnnotation present AND a stepping state id is
        available → call into Rust's state_memory_store_symbolic_multi
        with the original addr/data ASTs.  Asserts the routing only;
        the Rust side is unit-tested separately."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, _ = self._build_load_store_manager()
        captured = []

        def fake_store(state_id, addr_ast, data_ast):
            captured.append((state_id, addr_ast, data_ast))
            return True

        mgr._get_stepping_state_id = lambda: 1234
        mgr._rust_state_memory_store_symbolic_multi = fake_store

        addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
        data = claripy.BVV(0xCAFEBABE, 32)
        assert mgr._try_multi_cell_store(addr, data) is True
        assert len(captured) == 1
        sid, fwd_addr, fwd_data = captured[0]
        assert sid == 1234
        # Identity check: forwarded ASTs are the exact ones we passed in
        # (the annotation lives on the AST; identity preserves it).
        assert fwd_addr is addr
        assert fwd_data is data

    def test_try_multi_cell_store_falls_through_on_pyo3_failure(self):
        """If state_memory_store_symbolic_multi returns False the caller
        must fall through to the Python state path so the write is not
        silently lost.  _try_multi_cell_store returns False in that case."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, _ = self._build_load_store_manager()
        mgr._get_stepping_state_id = lambda: 1234
        mgr._rust_state_memory_store_symbolic_multi = lambda *_args, **_kw: False

        addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
        data = claripy.BVV(0xCAFEBABE, 32)
        assert mgr._try_multi_cell_store(addr, data) is False

    def test_try_multi_cell_store_falls_through_on_pyo3_exception(self):
        """Errors from the Rust side must be swallowed so the Python
        fallback can still apply the store."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, _ = self._build_load_store_manager()
        mgr._get_stepping_state_id = lambda: 1234

        def boom(*_args, **_kw):
            raise RuntimeError("simulated Rust-side failure")

        mgr._rust_state_memory_store_symbolic_multi = boom

        addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
        data = claripy.BVV(0xCAFEBABE, 32)
        assert mgr._try_multi_cell_store(addr, data) is False

    def test_cb_memory_store_symbolic_full_routes_through_multi(self):
        """Integration: _cb_memory_store_symbolic_full must hand
        annotated stores to _try_multi_cell_store; on success it must
        return without invoking state.memory.store.  Pinned via a stub
        state.memory.store that would otherwise record a call."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)
        mgr._get_stepping_state_id = lambda: 1234
        mgr._rust_state_memory_store_symbolic_multi = lambda *_args, **_kw: True

        store_calls = []
        original_store = state.memory.store
        state.memory.store = lambda *a, **kw: store_calls.append((a, kw))

        try:
            addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
            data = claripy.BVV(0xCAFEBABE, 32)
            mgr._cb_memory_store_symbolic_full(addr, data)
            assert store_calls == [], "MultiwriteAnnotation-tagged store leaked to Python state.memory.store"
        finally:
            state.memory.store = original_store

    def test_cb_memory_store_symbolic_full_falls_back_when_multi_fails(self):
        """If the Multi-cell PyO3 path returns False, the Python state
        path must still receive the store so the write is preserved."""
        import claripy

        from angr.storage.memory_mixins.address_concretization_mixin import (
            MultiwriteAnnotation,
        )

        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)
        mgr._get_stepping_state_id = lambda: 1234
        mgr._rust_state_memory_store_symbolic_multi = lambda *_args, **_kw: False

        store_calls = []
        original_store = state.memory.store
        state.memory.store = lambda *a, **kw: store_calls.append((a, kw))

        try:
            addr = claripy.BVS("sym_addr", 64).annotate(MultiwriteAnnotation())
            data = claripy.BVV(0xCAFEBABE, 32)
            mgr._cb_memory_store_symbolic_full(addr, data)
            assert len(store_calls) == 1, "Phase 1.4 fallback dropped the store on Multi-cell False"
        finally:
            state.memory.store = original_store

    # angr-2q5k: targeted coverage for LoadG with symbolic address.
    # Commit 0c90d4962 added resolve_loadg_load (expressions.rs:546) which
    # routes the three not-Single concretization shapes — Strided / TooLarge /
    # Failed — through fallback_load_symbolic_full (mod.rs:929).  Plain Load
    # tests already exist; LoadG was uncovered.
    #
    # The test below builds an IRSB by hand (LDle:I64 -> LoadG with
    # always-true guard) and runs it through the low-level
    # _RustExplorationManager.  The inner Load returns a fresh BVS via the
    # memory_load callback, so the LoadG address is symbolic and the
    # resolve_loadg_load dispatch must fire.

    def test_loadg_symbolic_address_dispatches_through_resolve_loadg_load(self):
        """A hand-built IRSB containing a LoadG with a symbolic address must
        dispatch through `resolve_loadg_load` (interpreter/expressions.rs:546).

        The address comes from an inner `Iex_Load` whose `memory_load`
        callback returns `(zeros, True, claripy_ast)` — Rust's bridge picks
        up the AST and the LoadG sees a symbolic temp.  Whichever
        ConcretizationResult shape the engine picks (Single / Multiple /
        Strided / TooLarge / Failed) MUST result in either:

          * `memory_load` firing for the resolved concrete address, OR
          * `memory_load_symbolic_full` firing for the symbolic address.

        Both paths route through `resolve_loadg_load`; failing to fire either
        means the LoadG was dropped (the regression caught by 0c90d4962
        before it returned `Unsupported`).
        """
        import json

        import claripy

        mgr = _RustExplorationManager("amd64")
        callbacks = PythonCallbacks()

        sym_addr = claripy.BVS("loadg_sym_addr", 64)
        load_calls = []

        def cb_load(addr, size):
            load_calls.append((addr, size))
            # The Load at 0x2000 returns the symbolic AST that becomes the
            # LoadG's address.  Other reads (e.g. page prefetch) get zeros.
            if addr == 0x2000 and size == 8:
                return (b"\x00" * size, True, sym_addr)
            return (b"\x00" * size, False, None)

        callbacks.set_memory_load(cb_load)
        callbacks.set_memory_store(lambda a, d: None)

        full_calls = []

        def cb_full(addr_ast, size):
            full_calls.append((addr_ast, size))
            return claripy.BVV(0xDEADBEEF, size * 8)

        callbacks.set_memory_load_symbolic_full(cb_full)

        # Hand-built IRSB:
        #   tmp 0 (I64) = LDle:I64(Const(0x2000))   ; -> symbolic via cb_load
        #   tmp 2 (I64) = Const(0)                   ; LoadG alt
        #   tmp 3 (I1)  = Const(true)                ; LoadG always-true guard
        #   LoadG dst=tmp1 addr=tmp0 alt=tmp2 guard=tmp3 cvt=Ident64 end=LE
        irsb = {
            "addr": 0x1000,
            "arch": "AMD64",
            "statements": [
                {"tag": "Ist_IMark", "addr": 0x1000, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Load",
                        "end": "Iend_LE",
                        "ty": "Ity_I64",
                        "addr": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 0x2000}},
                    },
                },
                {"tag": "Ist_WrTmp", "tmp": 2, "data": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 0}}},
                {"tag": "Ist_WrTmp", "tmp": 3, "data": {"tag": "Iex_Const", "con": {"tag": "Ico_U1", "value": True}}},
                {
                    "tag": "Ist_LoadG",
                    "dst": 1,
                    "addr": {"tag": "Iex_RdTmp", "tmp": 0},
                    "alt": {"tag": "Iex_RdTmp", "tmp": 2},
                    "guard": {"tag": "Iex_RdTmp", "tmp": 3},
                    "cvt": "ILGop_Ident64",
                    "end": "Iend_LE",
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 0x1010}},
            "jumpkind": "Ijk_Boring",
            "offsIP": 184,  # AMD64 RIP register offset
            "tyenv": {"types": ["Ity_I64", "Ity_I64", "Ity_I64", "Ity_I1"]},
        }
        irsb_json = json.dumps(irsb)

        def cb_lift(addr):
            return irsb_json if addr == 0x1000 else "{}"

        callbacks.set_lift_block(cb_lift)
        mgr.set_callbacks(callbacks)

        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)  # IRSB region
        state.pc = 0x1000
        mgr.add_state("active", state)

        mgr.run(2)

        # The LoadG must have been dispatched through resolve_loadg_load.
        # Either the Python full callback fires (Strided / TooLarge / Failed
        # branch) or the concrete callback fires for a non-0x2000 address
        # (Single / Multiple branch resolving the symbolic temp).
        loadg_dispatched = len(full_calls) > 0 or any(addr != 0x2000 for addr, _ in load_calls)
        assert loadg_dispatched, (
            f"LoadG with symbolic address never reached resolve_loadg_load. "
            f"memory_load_symbolic_full fired {len(full_calls)} times; "
            f"memory_load fired {len(load_calls)} times: {load_calls}. "
            f"Expected at least one cb_full call OR a cb_load call to a "
            f"non-0x2000 address (the LoadG would resolve the symbolic "
            f"temp and call back for the chosen concrete address)."
        )

    def test_loadg_symbolic_full_callback_returned_value_flows_through(self):
        """When the LoadG fallback fires, the value the Python
        `_cb_memory_load_symbolic_full` callback returns must flow back into
        the LoadG dst temp via either the handle table or the claripy
        bridge.  This test directly exercises the callback return-path used
        by `fallback_load_symbolic_full` (interpreter/mod.rs:947) by
        invoking the manager-level callback the way Rust would.
        """
        import claripy

        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)

        # Map a region and store a known value so the load returns it.
        target = 0x4100
        state.memory.map_region(target, 0x100, 7)
        state.memory.store(
            target, claripy.BVV(0xCAFED00D, 32), endness=state.arch.memory_endness, inspect=False, disable_actions=True
        )

        # Pin a symbolic address to `target` — the same shape the LoadG
        # fallback feeds the callback (a symbolic AST that Python's
        # concretization strategies can still resolve).
        addr = claripy.BVS("loadg_full_addr", 64)
        state.solver.add(addr == target)

        result = mgr._cb_memory_load_symbolic_full(addr, 4)
        assert result is not None, (
            "memory_load_symbolic_full must return an AST when Python's "
            "memory model can resolve the address; returning None would "
            "force Rust to fabricate a fresh symbolic placeholder via the "
            "sym_pyref_* fallback (interpreter/mod.rs:972)."
        )
        assert state.solver.eval(result) == 0xCAFED00D

    def test_z3_solver_timeout_does_not_hang(self):
        """A tight Z3 timeout must bound `satisfiable()` wall-clock — even
        on a constraint set Z3 would otherwise grind on forever (factoring a
        128-bit semiprime). Locks down that set_timeout() actually flows
        into the Z3 solver params and that the result path handles
        SatResult::Unknown by returning (rather than panicking or looping).

        The wall-clock bound is intentionally generous (5s) to avoid CI
        flakes on slow runners; the *contract* under test is "does not
        hang", not "respects timeout to the millisecond".
        """
        import time

        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        ctx.set_timeout(50)  # 50ms — far smaller than the 5s assertion bound

        # Force Z3 to factor a 128-bit semiprime where both factors are
        # constrained > 2^60. Practically intractable for Z3's BV theory
        # under a 50ms budget.
        x = claripy.BVS("x_factor", 128)
        y = claripy.BVS("y_factor", 128)
        # An arbitrary 128-bit composite whose factorization is non-obvious
        # to Z3 within 50ms.
        n = 0xC0DEBABE_DEADBEEF_FEEDFACE_CAFEF00D
        ctx.add_constraint_ast(x * y == n)
        ctx.add_constraint_ast(x > (1 << 60))
        ctx.add_constraint_ast(y > (1 << 60))
        ctx.add_constraint_ast(x < (1 << 100))
        ctx.add_constraint_ast(y < (1 << 100))

        start = time.monotonic()
        # We do NOT assert on the boolean result: under a tight timeout
        # Z3 may return Unknown, which the engine collapses to false. We
        # only require that the call returns at all and within bound.
        _ = ctx.satisfiable()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"satisfiable() took {elapsed:.2f}s with a 50ms Z3 timeout — "
            f"timeout config is not being honoured (expected < 5s)."
        )
