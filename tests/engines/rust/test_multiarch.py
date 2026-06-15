"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import os

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


class TestMultiArchSupport:
    """Tests for MIPS, ARM, and big-endian architecture support."""

    def test_mips32_state_creation(self):
        """MIPS32 state creation and register operations."""
        state = RustSimState("mips32")
        state.set_register("v0", 0xDEAD)
        state.set_register("a0", 0xBEEF)
        state.set_register("sp", 0x7FFF0000)

        assert state.get_register("v0") == 0xDEAD
        assert state.get_register("a0") == 0xBEEF
        assert state.get_register("sp") == 0x7FFF0000

    def test_mips32_exploration_manager(self):
        """MIPS32 exploration manager creation and stash ops."""
        mgr = _RustExplorationManager("mips32")
        assert mgr.arch == "mips32"

        mgr.create_state("active")
        assert mgr.active_count() == 1
        mgr.set_find_addrs([0x400000])
        mgr.set_avoid_addrs([0x400100])

    def test_mips32_big_endian(self):
        """MIPS32 big-endian state creation."""
        state = RustSimState("mips32", little_endian=False)
        state.set_register("v0", 0x12345678)
        assert state.get_register("v0") == 0x12345678

    def test_mips32_fork_isolation(self):
        """MIPS32 forked states have independent registers."""
        state1 = RustSimState("mips32")
        state1.set_register("t0", 100)
        state2 = state1.fork()
        state2.set_register("t0", 200)
        assert state1.get_register("t0") == 100
        assert state2.get_register("t0") == 200

    def test_mips32_full_register_family_dispatch(self):
        """MIPS32 FPU, HI/LO, and previously-untested GPRs round-trip via
        the standard register dispatch.

        Covers angr-w2gj.1: every entry in `offsets32` is now reachable
        through ``set_register``/``get_register`` (full-ABI MIPS32). The
        previous test set hit only a0/v0/sp/t0 — this exercises the
        register families that the dead-code audit (bd memory
        `dead-code-bd-ticket-policy`) flagged as unreachable.
        """
        state = RustSimState("mips32")

        # FPU registers (64-bit each on MIPS32 VEX guest state).
        # f0 (canonical) and $f31 (alias) are at opposite ends of the
        # F0-F31 block, and f12 is in the middle (first FP argument
        # register in O32).
        state.set_register("f0", 0x1122334455667788)
        state.set_register("f12", 0xCAFEBABEDEADBEEF)
        state.set_register("$f31", 0xFFEEDDCCBBAA9988)
        assert state.get_register("f0") == 0x1122334455667788
        assert state.get_register("f12") == 0xCAFEBABEDEADBEEF
        assert state.get_register("f31") == 0xFFEEDDCCBBAA9988
        assert state.get_register("$f0") == 0x1122334455667788

        # HI/LO: 32-bit multiplication result registers.
        state.set_register("hi", 0xDEADBEEF)
        state.set_register("lo", 0xCAFEBABE)
        assert state.get_register("hi") == 0xDEADBEEF
        assert state.get_register("lo") == 0xCAFEBABE

        # FPU control registers (32-bit).
        state.set_register("fir", 0x12345678)
        state.set_register("fcsr", 0x01020304)
        assert state.get_register("fir") == 0x12345678
        assert state.get_register("fcsr") == 0x01020304

        # Previously-untested GPRs: k0/k1 (kernel scratch, $26/$27)
        # were dead-code-flagged because no test exercised them, even
        # though they were already in the alias table.
        state.set_register("k0", 0xABCD1234)
        state.set_register("k1", 0x5678EF90)
        assert state.get_register("k0") == 0xABCD1234
        assert state.get_register("k1") == 0x5678EF90
        # And via the $N alias path.
        assert state.get_register("$26") == 0xABCD1234
        assert state.get_register("$27") == 0x5678EF90

    def test_mips64_full_register_family_dispatch(self):
        """MIPS64 FPU, HI/LO, and FPU-control registers round-trip via
        the standard register dispatch.

        Covers angr-w2gj.2: every entry in ``offsets64`` is now
        reachable through ``set_register``/``get_register`` (full-ABI
        MIPS64). The previous MIPS64 test set only exercised v0 — this
        exercises the register families that the dead-code audit
        (bd memory ``dead-code-bd-ticket-policy``) flagged as
        unreachable.
        """
        state = RustSimState("mips64")

        # FPU registers (64-bit each on MIPS64 VEX guest state).
        # f0 (canonical) and $f31 (alias) bracket the F0-F31 block;
        # f12 is the first FP argument register in the N64 ABI.
        state.set_register("f0", 0x1122334455667788)
        state.set_register("f12", 0xCAFEBABEDEADBEEF)
        state.set_register("$f31", 0xFFEEDDCCBBAA9988)
        assert state.get_register("f0") == 0x1122334455667788
        assert state.get_register("f12") == 0xCAFEBABEDEADBEEF
        assert state.get_register("f31") == 0xFFEEDDCCBBAA9988
        assert state.get_register("$f0") == 0x1122334455667788

        # HI/LO: 64-bit multiplication result registers.
        state.set_register("hi", 0xDEADBEEFCAFEBABE)
        state.set_register("lo", 0x0123456789ABCDEF)
        assert state.get_register("hi") == 0xDEADBEEFCAFEBABE
        assert state.get_register("lo") == 0x0123456789ABCDEF

        # FPU control registers (32-bit).
        state.set_register("fir", 0x12345678)
        state.set_register("fcsr", 0x01020304)
        assert state.get_register("fir") == 0x12345678
        assert state.get_register("fcsr") == 0x01020304

        # Previously-untested GPRs: k0/k1 (kernel scratch, $26/$27).
        state.set_register("k0", 0xABCD1234DEADBEEF)
        state.set_register("k1", 0x5678EF90CAFEBABE)
        assert state.get_register("k0") == 0xABCD1234DEADBEEF
        assert state.get_register("k1") == 0x5678EF90CAFEBABE
        # And via the $N alias path.
        assert state.get_register("$26") == 0xABCD1234DEADBEEF
        assert state.get_register("$27") == 0x5678EF90CAFEBABE

    def test_arm32_full_register_family_dispatch(self):
        """ARM32 VFP (d/q), FPSCR, and high GPRs round-trip via the standard
        register dispatch.

        Covers angr-9n19: the prior ARM32 unit tests touched only
        r0/r1/sp/lr. ``native/angr/src/arch/arm.rs`` defines d0-d31, q0-q15,
        fpscr, and r8-r12 (plus the fp/ip aliases) that were verified only by
        Rust-side offset assertions — this exercises them through the same
        ``set_register``/``get_register`` path real explorations use.
        """
        state = RustSimState("arm")

        # High GPRs r8-r12 (callee-saved / scratch) and the fp/ip aliases.
        state.set_register("r8", 0x08080808)
        state.set_register("r9", 0x09090909)
        state.set_register("r10", 0x10101010)
        state.set_register("r11", 0x11111111)
        state.set_register("r12", 0x12121212)
        assert state.get_register("r8") == 0x08080808
        assert state.get_register("r9") == 0x09090909
        assert state.get_register("r10") == 0x10101010
        # fp aliases r11, ip aliases r12.
        assert state.get_register("fp") == 0x11111111
        assert state.get_register("ip") == 0x12121212

        # VFP double registers (64-bit): d0 (first), d15 (mid), d31 (last).
        state.set_register("d0", 0x1122334455667788)
        state.set_register("d15", 0xCAFEBABEDEADBEEF)
        state.set_register("d31", 0xFFEEDDCCBBAA9988)
        assert state.get_register("d0") == 0x1122334455667788
        assert state.get_register("d15") == 0xCAFEBABEDEADBEEF
        assert state.get_register("d31") == 0xFFEEDDCCBBAA9988

        # NEON quad registers (128-bit) alias pairs of d registers: q0 overlaps
        # d0/d1. The low 64 bits land in d0, the high 64 bits in d1 (the guest
        # state is little-endian).
        qval = (0x0123456789ABCDEF << 64) | 0xFEDCBA9876543210
        state.set_register("q0", qval)
        assert state.get_register("q0") == qval
        assert state.get_register("d0") == 0xFEDCBA9876543210
        assert state.get_register("d1") == 0x0123456789ABCDEF

        # FPSCR (32-bit VFP status/control register).
        state.set_register("fpscr", 0x0A0B0C0D)
        assert state.get_register("fpscr") == 0x0A0B0C0D

    def test_aarch64_full_register_family_dispatch(self):
        """AArch64 SIMD (q/v/d), CC thunks, and high GPRs round-trip via the
        standard register dispatch.

        Covers angr-9n19: the prior AArch64 unit test touched only x0/x1/sp.
        ``native/angr/src/arch/arm64.rs`` defines x8-x30 (+ w aliases),
        q0-q31/v0-v31 (128-bit), d0-d31 (low-64 views), and the VEX
        condition-code thunks (cc_op/cc_dep1/cc_dep2/cc_ndep). AArch64's VEX
        guest represents flags as those thunks, not a flat NZCV register, so
        this pins the thunk names. All were previously checked only by
        Rust-side offset assertions.
        """
        state = RustSimState("aarch64")

        # High GPRs x8-x30 and the fp/lr aliases (x29/x30).
        state.set_register("x8", 0x0808080808080808)
        state.set_register("x16", 0x1616161616161616)
        state.set_register("x28", 0x2828282828282828)
        state.set_register("x29", 0x2929292929292929)
        state.set_register("x30", 0x3030303030303030)
        assert state.get_register("x8") == 0x0808080808080808
        assert state.get_register("x16") == 0x1616161616161616
        assert state.get_register("x28") == 0x2828282828282828
        assert state.get_register("fp") == 0x2929292929292929  # fp == x29
        assert state.get_register("lr") == 0x3030303030303030  # lr == x30

        # 32-bit w-views read the low half of the x register.
        assert state.get_register("w8") == 0x08080808
        assert state.get_register("w30") == 0x30303030

        # SIMD quad registers (128-bit); q and v are aliases of the same slot,
        # and d is the low-64 view.
        qval = (0x0011223344556677 << 64) | 0x8899AABBCCDDEEFF
        state.set_register("q5", qval)
        assert state.get_register("q5") == qval
        assert state.get_register("v5") == qval  # v5 aliases q5
        assert state.get_register("d5") == 0x8899AABBCCDDEEFF

        # d31 / q31 at the end of the SIMD block.
        state.set_register("d31", 0x1234567890ABCDEF)
        assert state.get_register("d31") == 0x1234567890ABCDEF

        # VEX condition-code thunks (AArch64 has no flat NZCV register).
        state.set_register("cc_op", 0x1111111111111111)
        state.set_register("cc_dep1", 0x2222222222222222)
        state.set_register("cc_dep2", 0x3333333333333333)
        state.set_register("cc_ndep", 0x4444444444444444)
        assert state.get_register("cc_op") == 0x1111111111111111
        assert state.get_register("cc_dep1") == 0x2222222222222222
        assert state.get_register("cc_dep2") == 0x3333333333333333
        assert state.get_register("cc_ndep") == 0x4444444444444444

    def test_arm_state_creation(self):
        """ARM32 state creation and register operations."""
        state = RustSimState("arm")
        state.set_register("r0", 0x1234)
        state.set_register("r1", 0x5678)
        state.set_register("sp", 0x7FFF0000)
        state.set_register("lr", 0x8000)

        assert state.get_register("r0") == 0x1234
        assert state.get_register("r1") == 0x5678
        assert state.get_register("sp") == 0x7FFF0000
        assert state.get_register("lr") == 0x8000

    def test_arm_exploration_manager(self):
        """ARM exploration manager creation."""
        mgr = _RustExplorationManager("arm")
        assert mgr.arch == "arm"
        mgr.create_state("active")
        assert mgr.active_count() == 1

    def test_arm64_state_creation(self):
        """ARM64/AArch64 state creation and register operations."""
        state = RustSimState("aarch64")
        state.set_register("x0", 0xDEADBEEF)
        state.set_register("x1", 0xCAFEBABE)
        state.set_register("sp", 0x7FFFFFFFE000)
        assert state.get_register("x0") == 0xDEADBEEF
        assert state.get_register("x1") == 0xCAFEBABE
        assert state.get_register("sp") == 0x7FFFFFFFE000

    def test_x86_state_creation(self):
        """x86 (32-bit) state creation and register operations."""
        state = RustSimState("x86")
        state.set_register("eax", 0xDEADBEEF)
        state.set_register("esp", 0x7FFF0000)
        assert state.get_register("eax") == 0xDEADBEEF
        assert state.get_register("esp") == 0x7FFF0000

    def test_x86_segment_selectors_dispatch(self):
        """x86 (32-bit) segment selectors round-trip via the standard
        register dispatch.

        Covers angr-5spy.1: the six 16-bit segment selectors
        (``cs``/``ds``/``es``/``fs``/``gs``/``ss``) in
        ``arch/x86.rs`` are reachable through
        ``set_register``/``get_register``. Each selector sits at its
        own offset (288, 290, 292, 294, 296, 298) so writes must not
        bleed into adjacent slots — a regression here would be a
        wrong-offset bug, not just unreachability.
        """
        state = RustSimState("x86")

        # Distinct 16-bit values so a cross-talk bug surfaces as a
        # wrong read on one of the neighbouring slots.
        selectors = {
            "cs": 0x0023,
            "ds": 0x002B,
            "es": 0x002B,
            "fs": 0x0053,
            "gs": 0x0063,
            "ss": 0x002B,
        }
        for name, value in selectors.items():
            state.set_register(name, value)
        for name, value in selectors.items():
            assert state.get_register(name) == value, name

        # "Selector-aware code path": overwrite GS to a non-default
        # value (Linux i386 sets %gs for TLS) and confirm the read
        # reflects the new value without disturbing CS/SS.
        state.set_register("gs", 0x0033)
        assert state.get_register("gs") == 0x0033
        assert state.get_register("cs") == selectors["cs"]
        assert state.get_register("ss") == selectors["ss"]

        # Selectors are 16-bit: writing a value that fits in 16 bits
        # and then reading must mask cleanly. (set_register builds a
        # bitvector sized from register_size; reads should round-trip
        # the low 16 bits.)
        state.set_register("fs", 0xFFFF)
        assert state.get_register("fs") == 0xFFFF

    def test_x86_segment_bases_dispatch(self):
        """x86 (32-bit) segment-base entries round-trip via the standard
        register dispatch.

        Covers angr-5spy.2: the four segment-base slots — ``fs_const``
        (320, 4B placeholder), ``gs_const`` (324, 4B placeholder),
        ``ldt`` (304, 8B per archinfo), and ``gdt`` (312, 8B per
        archinfo) — are reachable through
        ``set_register``/``get_register`` after wiring them into
        ``arch/x86.rs::ALIASES``. ``fs_const``/``gs_const`` are
        Rust-only placeholders (archinfo does not define them for
        x86, so Python-side parity is not applicable); ``ldt``/``gdt``
        exist on both sides of the FFI boundary.

        A wrong-offset bug would surface as cross-talk between
        neighbouring slots (writing ``fs_const`` would corrupt
        ``gs_const`` or vice versa).
        """
        state = RustSimState("x86")

        # Distinct values at each slot so a misaligned offset surfaces
        # as a cross-talk failure on the assertion side.
        state.set_register("fs_const", 0xCAFE_BABE)
        state.set_register("gs_const", 0xDEAD_BEEF)
        state.set_register("ldt", 0x1122334455667788)
        state.set_register("gdt", 0xAABBCCDDEEFF0011)

        assert state.get_register("fs_const") == 0xCAFE_BABE
        assert state.get_register("gs_const") == 0xDEAD_BEEF
        assert state.get_register("ldt") == 0x1122334455667788
        assert state.get_register("gdt") == 0xAABBCCDDEEFF0011

        # Rewriting fs_const must not disturb gs_const (or vice versa),
        # since the two slots are adjacent (320/324, both 4B) and a
        # wrong-width write would bleed across the boundary.
        state.set_register("fs_const", 0x1234_5678)
        assert state.get_register("fs_const") == 0x1234_5678
        assert state.get_register("gs_const") == 0xDEAD_BEEF

        # ldt/gdt are 8B and adjacent (304/312); writing one must not
        # bleed into the other.
        state.set_register("ldt", 0x0)
        assert state.get_register("ldt") == 0x0
        assert state.get_register("gdt") == 0xAABBCCDDEEFF0011

    def test_amd64_gs_const_routes_to_archinfo_offset(self):
        """amd64 ``gs_const`` lives at archinfo offset 1032, not the
        SSEROUND slot at 216.

        Covers angr-a68t: the Rust ``ALIASES`` table used to alias
        ``gs_const`` to offset 216, the SSEROUND slot. Every
        ``arch_prctl(ARCH_SET_GS, addr)`` therefore silently corrupted
        SSEROUND and left VEX's gs_const offset (1032) reading zero.
        This test pins the offsets via the Rust register dispatch and
        confirms a ``gs_const`` write does not touch ``sseround`` (and
        vice versa), so a regression that re-collides them surfaces as
        cross-talk.
        """
        state = RustSimState("amd64")

        state.set_register("gs_const", 0x1122334455667788)
        state.set_register("sseround", 0xDEAD_BEEF)

        assert state.get_register("gs_const") == 0x1122334455667788
        assert state.get_register("sseround") == 0xDEAD_BEEF

        # Rewriting one must not touch the other (regression would
        # surface as gs_const reading sseround's new value, or
        # sseround flipping when gs_const is updated).
        state.set_register("gs_const", 0xAABBCCDDEEFF0011)
        assert state.get_register("gs_const") == 0xAABBCCDDEEFF0011
        assert state.get_register("sseround") == 0xDEAD_BEEF

        state.set_register("sseround", 0x0)
        assert state.get_register("sseround") == 0x0
        assert state.get_register("gs_const") == 0xAABBCCDDEEFF0011

        # fs_const at archinfo offset 208 must remain independent
        # of both gs_const (1032) and sseround (216).
        state.set_register("fs_const", 0xFEED_FACE_CAFE_BABE)
        assert state.get_register("fs_const") == 0xFEED_FACE_CAFE_BABE
        assert state.get_register("gs_const") == 0xAABBCCDDEEFF0011
        assert state.get_register("sseround") == 0x0

    def test_unsupported_arch_raises(self):
        """Unknown architecture raises an error."""
        with pytest.raises(ValueError, match="unknown architecture"):
            RustSimState("pdp11")

    def test_x86_explore_blob(self, tmp_path):
        """End-to-end x86 (32-bit) exploration on a hand-assembled blob.

        x86 register/CC plumbing in ``arch/x86.rs`` is exercised by unit
        tests (state creation, segment selectors, segment bases) but no
        end-to-end Rust-engine test ran on a real x86 instruction stream
        — exactly the gap that let the Cdecl return-register bug
        (5329d8222) hide for months on amd64. This test ships a small
        i386 program as raw bytes, loads it via cle's Blob backend, and
        drives ``eax`` through ``RustExplorationManager.explore`` to
        confirm VEX interpretation, register sync, conditional branch
        resolution, and solver evaluation all work on x86.

        The program computes ``2*eax + 16`` and branches to ``found`` if
        the result equals 100; ``eax == 42`` is the canonical solution.
        Promotes x86 (32-bit) from Skeleton (no e2e coverage) to
        Experimental — tested in the arch support matrix.
        """
        import claripy

        # i386 little-endian:
        #   0x00: 01 C0       add eax, eax           ; eax = 2*eax
        #   0x02: 83 C0 10    add eax, 0x10          ; eax = 2*eax + 16
        #   0x05: 83 F8 64    cmp eax, 0x64          ; cmp eax, 100
        #   0x08: 74 06       je  +6  -> 0x10        ; found
        #   0x0a: EB 0C       jmp +12 -> 0x18        ; avoid
        #   0x0c: 90 90 90 90 (pad to 0x10)
        #   0x10: 90 90 90 90 90 90 90 90  (found region)
        #   0x18: 90          (avoid)
        code = bytes.fromhex(
            "01C0"  # add eax, eax
            "83C010"  # add eax, 0x10
            "83F864"  # cmp eax, 0x64
            "7406"  # je  +6  -> 0x10
            "EB0C"  # jmp +12 -> 0x18
            "90909090"  # pad 0x0c..0x0f
            "9090909090909090"  # found region 0x10..0x17
            "90"  # avoid 0x18
        )
        assert len(code) == 0x19
        blob_path = tmp_path / "x86_branch.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "x86", "base_addr": 0x400000},
            auto_load_libs=False,
        )
        assert proj.arch.name == "X86"
        assert proj.arch.bits == 32

        state = proj.factory.blank_state(addr=0x400000)
        eax = claripy.BVS("eax", 32)
        state.regs.eax = eax

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400010, avoid=0x400018, num_find=1, max_steps=100)

        assert len(mgr.found) >= 1, f"x86 exploration did not reach 0x400010; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        # 2*eax + 16 == 100  ⇒  eax == 42.
        assert found.solver.eval(eax) == 42, f"Expected eax==42 to reach found, got {found.solver.eval(eax)}"

    def test_arm32_explore_real_binary(self):
        """End-to-end ARM32 (ARMEL) exploration on a real binary.

        Until this landed, ARM had only state-creation unit tests. The Cdecl
        x86 return-register bug (5329d8222) was latent for months precisely
        because no end-to-end x86 test ran — same risk class for ARM, so
        this exercises VEX interpretation, register sync, calling-convention
        plumbing, and find/avoid stashing in one shot.

        Uses the angr-examples Android license_validation binary
        (load address 0x401760, find=0x401840, avoid=0x401854).
        """
        import claripy

        binary_path = os.path.expanduser("~/repos/angr-examples/examples/android_arm_license_validation/validate")
        if not os.path.exists(binary_path):
            pytest.skip(f"ARM binary not found at {binary_path}")

        proj = angr.Project(binary_path, auto_load_libs=False)
        assert proj.arch.name == "ARMEL"

        state = proj.factory.blank_state(addr=0x401760)
        concrete_addr = 0xFFE00000
        code = claripy.BVS("code", 10 * 8)
        state.memory.store(concrete_addr, code, endness="Iend_BE")
        state.regs.r0 = concrete_addr

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x401840, avoid=0x401854, num_find=1, max_steps=2000)

        assert len(mgr.found) >= 1, f"ARM exploration did not reach 0x401840; counts={mgr.stash_counts()}"
        # Sanity: solver still has a model for the symbolic input.
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"

    def test_armeb_explore_blob(self, tmp_path):
        """End-to-end ARM Big-Endian (ARMEB) exploration on a hand-assembled blob.

        ARMEL (little-endian) has both a real-binary integration test
        (``test_arm32_explore_real_binary``) and the ``arm_le_branch``
        benchmark, but ARMEB had only state-creation coverage —
        Skeleton in the arch matrix. archinfo's ``armeb`` maps to
        ``ARMEL`` with ``memory_endness=Iend_BE`` and
        ``instruction_endness=Iend_BE`` (BE32 model), so this test
        ships the same code words as ``arm_le_branch`` packed
        big-endian and routes them through the Rust engine.

        Confirms instruction fetch, register sync, comparison, branch
        resolution, and solver evaluation all work on a BE memory
        layout. Promotes ARMEB from Skeleton to Experimental in the
        support matrix; same risk class as the latent Cdecl x86 bug
        (5329d8222) — without an end-to-end BE test, byte-order
        regressions in instruction-fetch or register handling hide
        silently.
        """
        import struct

        import claripy

        # ARM (AL condition = 0xE), packed big-endian for ARMEB BE32:
        #   0x00: ADD  r0, r0, r0      0xE0800000  ; r0 = 2*r0
        #   0x04: ADD  r0, r0, #16     0xE2800010  ; r0 = 2*r0 + 16
        #   0x08: MOV  r1, #100        0xE3A01064
        #   0x0c: CMP  r0, r1          0xE1500001
        #   0x10: BEQ  +0  -> 0x18     0x0A000000
        #   0x14: B    +4  -> 0x20     0xEA000001
        #   0x18: NOP  (found)         0xE1A00000
        #   0x1c: NOP                  0xE1A00000
        #   0x20: NOP  (avoid)         0xE1A00000
        code = struct.pack(
            ">IIIIIIIII",
            0xE0800000,
            0xE2800010,
            0xE3A01064,
            0xE1500001,
            0x0A000000,
            0xEA000001,
            0xE1A00000,
            0xE1A00000,
            0xE1A00000,
        )
        blob_path = tmp_path / "armeb_branch.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "armeb", "base_addr": 0x10000},
            auto_load_libs=False,
        )
        assert proj.arch.name == "ARMEL"
        assert proj.arch.memory_endness == "Iend_BE"
        assert proj.arch.instruction_endness == "Iend_BE"

        state = proj.factory.blank_state(addr=0x10000)
        r0 = claripy.BVS("r0", 32)
        state.regs.r0 = r0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x10018, avoid=0x10020, num_find=1, max_steps=100)

        assert len(mgr.found) >= 1, f"ARMEB exploration did not reach 0x10018; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        # 2*r0 + 16 == 100 ⇒ r0 == 42.
        assert found.solver.eval(r0) == 42, f"Expected r0==42 to reach found, got {found.solver.eval(r0)}"

    def test_aarch64_explore_blob(self, tmp_path):
        """End-to-end AArch64 exploration on a hand-assembled blob.

        No AArch64 binaries ship with angr-examples and no cross-compiler
        is available locally, so this test ships seven AArch64 instructions
        as raw bytes and loads them via cle's Blob backend. The program
        compares ``w0`` against 42 and branches to either ``found`` or
        ``avoid``. Symbolic execution must drive ``w0`` to 42 to reach
        the find address.

        Promotes AArch64 from Skeleton to Experimental in the support
        matrix; same risk class as the latent Cdecl x86 bug
        (5329d8222) — without an end-to-end test, register-offset or
        calling-convention bugs hide for months.
        """
        import struct

        import claripy

        # AArch64 little-endian:
        #   0x00: MOV w1, #42         52800541
        #   0x04: CMP w0, w1          6b01001f  (SUBS wzr, w0, w1)
        #   0x08: B.EQ +8 -> 0x10     54000040
        #   0x0c: B  +12 -> 0x18      14000003
        #   0x10: NOP  (found)        d503201f
        #   0x14: NOP                 d503201f
        #   0x18: NOP  (avoid)        d503201f
        code = struct.pack(
            "<IIIIIII",
            0x52800541,
            0x6B01001F,
            0x54000040,
            0x14000003,
            0xD503201F,
            0xD503201F,
            0xD503201F,
        )
        blob_path = tmp_path / "aarch64_branch.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "aarch64", "base_addr": 0x400000},
        )
        assert proj.arch.name == "AARCH64"

        state = proj.factory.blank_state(addr=0x400000)
        x0 = claripy.BVS("x0", 64)
        state.regs.x0 = x0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400010, avoid=0x400018, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, f"AArch64 exploration did not reach 0x400010; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        assert found.solver.eval(x0) == 42, f"Expected x0==42 to reach found, got {found.solver.eval(x0)}"

    def test_aarch64_concrete_branch_no_spurious_fork(self, tmp_path):
        """arm64g_calculate_condition must not fork on concrete flags (angr-37d4).

        Before arm64g ccalls were implemented, the ``is_cond_ccall``
        catch-all returned a *fresh unconstrained* symbolic for every
        ``arm64g_calculate_condition``, so a B.EQ whose flags are fully
        concrete still produced two successors — the infeasible ``avoid``
        path got explored, diverging from the Python engine.

        Here ``w0`` is *concrete* (42), so CMP w0,#42 sets Z=1 concretely
        and B.EQ is concretely taken. With the ccall computed properly the
        false branch (0x400018) is infeasible and never created, so the
        ``avoid`` stash stays empty.
        """
        import struct

        # Same blob as test_aarch64_explore_blob.
        code = struct.pack(
            "<IIIIIII",
            0x52800541,
            0x6B01001F,
            0x54000040,
            0x14000003,
            0xD503201F,
            0xD503201F,
            0xD503201F,
        )
        blob_path = tmp_path / "aarch64_concrete_branch.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "aarch64", "base_addr": 0x400000},
        )

        state = proj.factory.blank_state(addr=0x400000)
        state.regs.x0 = 42  # concrete -> B.EQ concretely taken

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400010, avoid=0x400018, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, f"concrete x0==42 did not reach found; counts={mgr.stash_counts()}"
        # The key assertion: the concretely-infeasible avoid branch must not
        # have been forked into existence.
        assert len(mgr.avoided) == 0, (
            f"spurious fork: concrete-flag B.EQ created the infeasible avoid branch; counts={mgr.stash_counts()}"
        )

    def test_aarch64_neon_mla_blob(self, tmp_path):
        """End-to-end AArch64 NEON exploration on a hand-assembled blob.

        Validates angr-bkcs.2: NEON SIMD ops (Iop_Add8x16, Iop_Mul8x16) run
        natively through the Rust engine without falling back to Python.
        The program performs a byte-wise multiply-accumulate on the low byte
        of a symbolic w0 and branches based on the result. The MLA NEON
        instruction lifts to ``t = Mul8x16(...); t' = Add8x16(...)`` in VEX —
        if either op had no real handler, dispatch would panic at
        ``IROp::NeonUnimplemented`` and the test would fail loudly.

        Solver must drive ``w0 & 0xFF`` to one of the residues r where
        ``r + r * r ≡ 20 (mod 256)``; r=4 (4 + 16 = 20) is the canonical
        solution.
        """
        import struct

        import claripy

        # AArch64 little-endian — verified individually via pyvex:
        #   0x400000: FMOV S0, W0       1E270000  ; q0 = [0,0,0, w0_low32]
        #   0x400004: FMOV S1, W0       1E270001  ; q1 = [0,0,0, w0_low32]
        #   0x400008: MLA V0.16B,V0,V1  4E219400  ; v0 += v0 * v1 (per byte)
        #                                         ; -> Iop_Mul8x16 + Iop_Add8x16
        #   0x40000C: UMOV W0, V0.B[0]  0E013C00  ; w0 = byte 0 (zero-extended)
        #   0x400010: MOVZ W1, #20      52800281
        #   0x400014: CMP W0, W1        6B01001F
        #   0x400018: B.EQ +8 -> 0x20   54000040
        #   0x40001C: B +8 -> 0x24      14000002
        #   0x400020: NOP (found)       D503201F
        #   0x400024: NOP (avoid)       D503201F
        code = struct.pack(
            "<IIIIIIIIII",
            0x1E270000,
            0x1E270001,
            0x4E219400,
            0x0E013C00,
            0x52800281,
            0x6B01001F,
            0x54000040,
            0x14000002,
            0xD503201F,
            0xD503201F,
        )
        blob_path = tmp_path / "aarch64_neon_mla.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "aarch64", "base_addr": 0x400000},
        )
        assert proj.arch.name == "AARCH64"

        state = proj.factory.blank_state(addr=0x400000)
        x0 = claripy.BVS("x0", 64)
        state.regs.x0 = x0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400020, avoid=0x400024, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, f"AArch64 NEON exploration did not reach 0x400020; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"

        # Solver picks any residue r with r + r*r ≡ 20 (mod 256).
        val = found.solver.eval(x0) & 0xFF
        expected = (val + val * val) & 0xFF
        assert expected == 20, (
            f"low-byte residue {val} did not satisfy r + r*r == 20 (mod 256); got r + r*r = {expected}"
        )

    def test_aarch64_explore_real_elf(self, tmp_path):
        """End-to-end AArch64 exploration on a hand-assembled ELF binary.

        The existing aarch64 blob test covers only a single compare + branch
        through the cle Blob backend. This test goes further:

          * loads through cle's ELF backend (parses e_machine / e_entry /
            PT_LOAD), exercising the full ELF loader path on AArch64
          * exercises BL / RET (calling-convention plumbing — X30 set on
            BL, branch target read back from X30 on RET) in addition to
            the conditional branch
          * runs from the ELF's e_entry (proj.entry) rather than a
            hand-picked address, confirming e_entry decoding

        No AArch64 binaries ship with angr-examples and no cross-compiler
        is available locally, so the ELF is constructed inline. The
        program calls a subroutine ``double_it`` that returns ``2 * w0``,
        then checks the result equals 84 — symbolic execution must drive
        ``w0`` to 42 to reach the find address.

        Promotes AArch64 from blob-only to real-ELF coverage in the
        support matrix; pairs with angr-gxhf.1.
        """
        import struct

        import claripy

        # AArch64 little-endian instructions (verified against ARMv8 ARM):
        #   double_it (at 0x400078):
        #     0x400078: add w0, w0, w0       0B000000  ; w0 = 2*w0
        #     0x40007c: ret                  D65F03C0  ; return via x30
        #   entry (at 0x400080):
        #     0x400080: bl double_it (-8)    97FFFFFE
        #     0x400084: movz w1, #84         52800A81
        #     0x400088: cmp w0, w1           6B01001F  ; subs wzr, w0, w1
        #     0x40008c: b.eq found (+8)      54000040
        #     0x400090: b avoid (+12)        14000003
        #   found (at 0x400094):
        #     0x400094: nop                  D503201F
        #     0x400098: nop                  D503201F
        #   avoid (at 0x40009C):
        #     0x40009c: nop                  D503201F
        code = struct.pack(
            "<IIIIIIIIII",
            0x0B000000,  # add w0, w0, w0
            0xD65F03C0,  # ret
            0x97FFFFFE,  # bl -8 -> double_it
            0x52800A81,  # movz w1, #84
            0x6B01001F,  # cmp w0, w1
            0x54000040,  # b.eq +8 -> found
            0x14000003,  # b +12 -> avoid
            0xD503201F,  # nop (found)
            0xD503201F,  # nop
            0xD503201F,  # nop (avoid)
        )

        # Minimal ELF64 (AArch64) header. PT_LOAD covers file [0, 160] →
        # vaddr [0x400000, 0x4000A0). Entry = 0x400080. No section headers.
        BASE = 0x400000
        EHDR_SIZE = 64
        PHDR_SIZE = 56
        TOTAL = EHDR_SIZE + PHDR_SIZE + len(code)
        ENTRY = BASE + EHDR_SIZE + PHDR_SIZE + 8  # skip subroutine

        # ELF64 header (little-endian)
        ehdr = (
            b"\x7fELF"
            + bytes(
                [
                    2,  # EI_CLASS = ELF64
                    1,  # EI_DATA = LSB
                    1,  # EI_VERSION
                    0,  # EI_OSABI = System V
                    0,  # EI_ABIVERSION
                ]
            )
            + b"\x00" * 7
        )  # EI_PAD
        ehdr += struct.pack(
            "<HHIQQQIHHHHHH",
            2,  # e_type = ET_EXEC
            0xB7,  # e_machine = EM_AARCH64
            1,  # e_version
            ENTRY,  # e_entry
            EHDR_SIZE,  # e_phoff
            0,  # e_shoff
            0,  # e_flags
            EHDR_SIZE,  # e_ehsize
            PHDR_SIZE,  # e_phentsize
            1,  # e_phnum
            0,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )
        assert len(ehdr) == EHDR_SIZE

        # PT_LOAD program header
        phdr = struct.pack(
            "<IIQQQQQQ",
            1,  # p_type = PT_LOAD
            5,  # p_flags = PF_R | PF_X
            0,  # p_offset
            BASE,  # p_vaddr
            BASE,  # p_paddr
            TOTAL,  # p_filesz
            TOTAL,  # p_memsz
            0x1000,  # p_align
        )
        assert len(phdr) == PHDR_SIZE

        elf_bytes = ehdr + phdr + code
        elf_path = tmp_path / "aarch64_call.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "AARCH64"
        assert proj.entry == ENTRY, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {ENTRY:#x}"

        state = proj.factory.blank_state(addr=proj.entry)
        x0 = claripy.BVS("x0", 64)
        state.regs.x0 = x0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400094, avoid=0x40009C, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, f"AArch64 ELF exploration did not reach 0x400094; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        # The doubled input must equal 84, so the input must be 42.
        assert found.solver.eval(x0) == 42, f"Expected x0==42 (so 2*x0==84) to reach found, got {found.solver.eval(x0)}"

    def test_mips32_explore_blob(self, tmp_path):
        """End-to-end MIPS32 (big-endian) exploration on a hand-assembled blob.

        MIPS32 has no calling convention defined in the Rust engine
        (default_cc_for_arch falls through to SystemVAMD64 — see the
        ``invariant-mips-no-calling-convention`` memory), but the VEX
        interpreter, register sync, and branch handling can still be
        exercised without a SimProcedure call. This test loads seven
        MIPS instructions as a blob and asserts that the engine drives
        ``a0`` to 42 to reach the find address.

        Promotes MIPS32 from Skeleton to Experimental in the support
        matrix.
        """
        import struct

        import claripy

        # MIPS32 big-endian, with delay slots:
        #   0x00: ADDIU t0, zero, 42     2408002A
        #   0x04: BEQ a0, t0, +3         10880003   -> on equal, target 0x14
        #   0x08: NOP (delay slot)       00000000
        #   0x0c: B +2 (BEQ zero,zero)   10000002   -> target 0x18 (avoid)
        #   0x10: NOP (delay slot)       00000000
        #   0x14: NOP (found)            00000000
        #   0x18: NOP (avoid)            00000000
        code = struct.pack(
            ">IIIIIII",
            0x2408002A,
            0x10880003,
            0x00000000,
            0x10000002,
            0x00000000,
            0x00000000,
            0x00000000,
        )
        blob_path = tmp_path / "mips32_branch.bin"
        blob_path.write_bytes(code)

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": "mips", "base_addr": 0x400000},
        )
        assert proj.arch.name == "MIPS32"
        assert proj.arch.memory_endness == "Iend_BE"

        state = proj.factory.blank_state(addr=0x400000)
        a0 = claripy.BVS("a0", 32)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400014, avoid=0x400018, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, f"MIPS32 exploration did not reach 0x400014; counts={mgr.stash_counts()}"
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        assert found.solver.eval(a0) == 42, f"Expected a0==42 to reach found, got {found.solver.eval(a0)}"

    def test_mips32_explore_le_real_elf(self, tmp_path):
        """End-to-end MIPS32 little-endian exploration on a hand-assembled ELF.

        ``test_mips32_explore_blob`` covers MIPS32 BE via the cle Blob
        backend. This test fills two distinct gaps:

          * MIPS32 LE end-to-end — at port time only the BE path was
            smoke-tested past block 0; LE has never been driven through
            the interpreter on a multi-block control flow.
          * cle's ELF loader on MIPS32 — Blob bypasses e_machine / EI_DATA
            parsing; an actual ELF32 LE header forces cle's ELF backend
            to recognise EM_MIPS + EI_DATA=LSB and report MIPS32/Iend_LE.

        No MIPS LE binaries ship with angr-examples and no cross-compiler
        is available locally, so the ELF is constructed inline (same
        pattern as ``test_aarch64_explore_real_elf``).

        Pairs with angr-gxhf.2.
        """
        import struct

        import claripy

        # MIPS32 instruction encodings are endian-agnostic at decode time;
        # the storage byte order changes with EI_DATA. Same opcodes as the
        # BE blob test, packed little-endian.
        #   0x00: ADDIU t0, zero, 42     2408002A
        #   0x04: BEQ a0, t0, +3         10880003   -> on equal, target 0x14
        #   0x08: NOP (delay slot)       00000000
        #   0x0c: B +2 (BEQ zero,zero)   10000002   -> target 0x18 (avoid)
        #   0x10: NOP (delay slot)       00000000
        #   0x14: NOP (found)            00000000
        #   0x18: NOP (avoid)            00000000
        code = struct.pack(
            "<IIIIIII",
            0x2408002A,
            0x10880003,
            0x00000000,
            0x10000002,
            0x00000000,
            0x00000000,
            0x00000000,
        )

        # Minimal ELF32 (MIPS LE) header. PT_LOAD covers file [0, 84+len(code)]
        # → vaddr [0x400000, ...). Entry = first instruction.
        BASE = 0x400000
        EHDR_SIZE = 52  # ELF32 header
        PHDR_SIZE = 32  # ELF32 program header
        TOTAL = EHDR_SIZE + PHDR_SIZE + len(code)
        ENTRY = BASE + EHDR_SIZE + PHDR_SIZE

        # ELF32 header (little-endian)
        ehdr = (
            b"\x7fELF"
            + bytes(
                [
                    1,  # EI_CLASS = ELF32
                    1,  # EI_DATA = LSB (little-endian)
                    1,  # EI_VERSION
                    0,  # EI_OSABI = System V
                    0,  # EI_ABIVERSION
                ]
            )
            + b"\x00" * 7
        )  # EI_PAD
        ehdr += struct.pack(
            "<HHIIIIIHHHHHH",
            2,  # e_type = ET_EXEC
            0x08,  # e_machine = EM_MIPS
            1,  # e_version
            ENTRY,  # e_entry
            EHDR_SIZE,  # e_phoff
            0,  # e_shoff
            0x50001000,  # e_flags = EF_MIPS_ARCH_32 | EF_MIPS_ABI_O32
            EHDR_SIZE,  # e_ehsize
            PHDR_SIZE,  # e_phentsize
            1,  # e_phnum
            0,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )
        assert len(ehdr) == EHDR_SIZE

        # ELF32 PT_LOAD program header (field order differs from ELF64!)
        phdr = struct.pack(
            "<IIIIIIII",
            1,  # p_type = PT_LOAD
            0,  # p_offset
            BASE,  # p_vaddr
            BASE,  # p_paddr
            TOTAL,  # p_filesz
            TOTAL,  # p_memsz
            5,  # p_flags = PF_R | PF_X
            0x1000,  # p_align
        )
        assert len(phdr) == PHDR_SIZE

        elf_bytes = ehdr + phdr + code
        elf_path = tmp_path / "mips32le_branch.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "MIPS32"
        assert proj.arch.memory_endness == "Iend_LE", (
            f"cle ELF loader did not pick up EI_DATA=LSB for MIPS32: got {proj.arch.memory_endness}"
        )
        assert proj.entry == ENTRY, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {ENTRY:#x}"

        state = proj.factory.blank_state(addr=proj.entry)
        a0 = claripy.BVS("a0", 32)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        # Branch targets are at +0x14 / +0x18 from the first instruction,
        # which sits at ENTRY (just past the ELF header / phdr).
        mgr.explore(
            find=ENTRY + 0x14,
            avoid=ENTRY + 0x18,
            num_find=1,
            max_steps=50,
        )

        assert len(mgr.found) >= 1, (
            f"MIPS32 LE ELF exploration did not reach {ENTRY + 0x14:#x}; counts={mgr.stash_counts()}"
        )
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        assert found.solver.eval(a0) == 42, f"Expected a0==42 to reach found, got {found.solver.eval(a0)}"

    def test_mips32_symbolic_register_survives_disk_init_cache(self, tmp_path):
        """User-set symbolic registers must survive the disk init cache (angr-g9hy).

        Bug: `_state_has_user_symbolic` only scanned memory, so a user mutation
        like ``state.regs.a0 = claripy.BVS("a0", 32)`` did not invalidate the
        cache. The cached blank_state replaced the user's state, dropping the
        symbolic ``a0`` — Rust saw concrete 0 for every read. The chain
        ``$t0 = $t0 + $a0`` over many MIPS blocks then collapsed ``$t0`` to
        concrete 0, the comparator BEQ became concrete-false, and the test
        binary never reached its FOUND address.

        Reproduces by running the failing-before-fix multi-block accumulator
        layout: N=30 blocks each doing ``ADDU $t0,$t0,$a0; B +1; NOP`` followed
        by a comparator ``BEQ $t0, N*5``. Before the fix, FOUND was empty for
        N>=30 (deterministic).
        """
        import struct

        import claripy

        def addiu(rt, rs, imm):
            return 0x24000000 | (rs << 21) | (rt << 16) | (imm & 0xFFFF)

        def addu(rd, rs, rt):
            return (rs << 21) | (rt << 16) | (rd << 11) | 0x21

        def beq(rs, rt, off):
            return 0x10000000 | (rs << 21) | (rt << 16) | (off & 0xFFFF)

        def b_(off):
            return beq(0, 0, off)

        N = 30
        code = [addiu(8, 0, 0)]  # init: t0 = 0
        for _ in range(N):
            code.append(addu(8, 8, 4))  # addu t0, t0, a0
            code.append(b_(1))  # b +1 -> skip nop_pad
            code.append(0)  # nop (delay slot)
            code.append(0)  # nop_pad (B target)
        target = N * 5
        code.append(addiu(9, 0, target))  # addiu t1, zero, N*5
        code.append(beq(8, 9, 3))  # beq t0, t1, +3 -> FOUND
        code.append(0)  # delay
        code.append(b_(2))  # b +2 -> AVOID
        code.append(0)  # delay
        found_idx = len(code)
        code.append(0)  # FOUND
        avoid_idx = len(code)
        code.append(0)  # AVOID

        code_bytes = struct.pack("<" + "I" * len(code), *code)
        BASE = 0x400000
        EHDR_SIZE = 52
        PHDR_SIZE = 32
        TOTAL = EHDR_SIZE + PHDR_SIZE + len(code_bytes)
        ENTRY = BASE + EHDR_SIZE + PHDR_SIZE

        ehdr = b"\x7fELF" + bytes([1, 1, 1, 0, 0]) + b"\x00" * 7
        ehdr += struct.pack(
            "<HHIIIIIHHHHHH",
            2,
            0x08,
            1,
            ENTRY,
            EHDR_SIZE,
            0,
            0x50001000,
            EHDR_SIZE,
            PHDR_SIZE,
            1,
            0,
            0,
            0,
        )
        phdr = struct.pack(
            "<IIIIIIII",
            1,
            0,
            BASE,
            BASE,
            TOTAL,
            TOTAL,
            5,
            0x1000,
        )
        elf_path = tmp_path / "mips32_accum.elf"
        elf_path.write_bytes(ehdr + phdr + code_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "MIPS32"

        state = proj.factory.blank_state(addr=proj.entry)
        a0 = claripy.BVS("a0", 32)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        find_addr = ENTRY + 4 * found_idx
        avoid_addr = ENTRY + 4 * avoid_idx
        mgr.explore(find=find_addr, avoid=avoid_addr, num_find=1, max_steps=500)

        assert mgr.found, f"Symbolic accumulator collapsed before reaching find — stashes={mgr.stash_counts()}"
        result = mgr.found[0].solver.eval(a0)
        assert result == 5, f"Expected a0==5 (so t0 == N*5 satisfies BEQ), got {result}"

    def test_mips64_explore_le_real_elf(self, tmp_path):
        """End-to-end MIPS64 little-endian exploration on a hand-assembled ELF.

        Closes the last gap in the gxhf epic (promote ARM64 / MIPS32 / MIPS64
        from Skeleton to Experimental). Before this test, MIPS64 was wired
        up (N64 calling convention defined as of angr-gzk8 commit 8bd93978c,
        register table present) but never exercised end-to-end on a binary
        through the Rust interpreter — the same risk class as the latent
        Cdecl x86 EAX/EDX bug (commit 5329d8222).

        MIPS64 instruction encodings for these base opcodes match MIPS32
        (registers are still 5 bits; ADDIU sign-extends to 64-bit on
        MIPS64), so the same program shape as the MIPS32 LE test works.
        The difference is the ELF: ELF64 header (64 B) + PT_LOAD program
        header (56 B) + EM_MIPS + EI_CLASS=ELF64 + EF_MIPS_ARCH_64.

        No MIPS64 binaries ship with angr-examples and no cross-compiler
        is available locally, so the ELF is constructed inline (same
        pattern as ``test_aarch64_explore_real_elf``).

        Pairs with angr-gxhf.3.
        """
        import struct

        import claripy

        # MIPS64 instruction encodings — same as MIPS32 for these opcodes
        # since registers are still 5 bits. ADDIU sign-extends the 16-bit
        # immediate to 64 bits on MIPS64. Stored little-endian.
        #   0x00: ADDIU t0, zero, 42     2408002A  ; $t0 = sign-extend(42)
        #   0x04: BEQ a0, t0, +3         10880003  ; on equal -> +0x14 (found)
        #   0x08: NOP (delay slot)       00000000
        #   0x0c: B +2 (BEQ zero,zero)   10000002  ; -> +0x18 (avoid)
        #   0x10: NOP (delay slot)       00000000
        #   0x14: NOP (found)            00000000
        #   0x18: NOP (avoid)            00000000
        code = struct.pack(
            "<IIIIIII",
            0x2408002A,
            0x10880003,
            0x00000000,
            0x10000002,
            0x00000000,
            0x00000000,
            0x00000000,
        )

        # Minimal ELF64 (MIPS64 LE) header. PT_LOAD covers file
        # [0, 120+len(code)] → vaddr [0x400000, ...). Entry = first instr.
        BASE = 0x400000
        EHDR_SIZE = 64  # ELF64 header
        PHDR_SIZE = 56  # ELF64 program header
        TOTAL = EHDR_SIZE + PHDR_SIZE + len(code)
        ENTRY = BASE + EHDR_SIZE + PHDR_SIZE

        # ELF64 header (little-endian)
        ehdr = (
            b"\x7fELF"
            + bytes(
                [
                    2,  # EI_CLASS = ELF64
                    1,  # EI_DATA = LSB (little-endian)
                    1,  # EI_VERSION
                    0,  # EI_OSABI = System V
                    0,  # EI_ABIVERSION
                ]
            )
            + b"\x00" * 7
        )  # EI_PAD
        ehdr += struct.pack(
            "<HHIQQQIHHHHHH",
            2,  # e_type = ET_EXEC
            0x08,  # e_machine = EM_MIPS
            1,  # e_version
            ENTRY,  # e_entry
            EHDR_SIZE,  # e_phoff
            0,  # e_shoff
            0x60000000,  # e_flags = EF_MIPS_ARCH_64 (N64 implied by EI_CLASS=ELF64)
            EHDR_SIZE,  # e_ehsize
            PHDR_SIZE,  # e_phentsize
            1,  # e_phnum
            0,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )
        assert len(ehdr) == EHDR_SIZE

        # ELF64 PT_LOAD program header (field order: p_type, p_flags first)
        phdr = struct.pack(
            "<IIQQQQQQ",
            1,  # p_type = PT_LOAD
            5,  # p_flags = PF_R | PF_X
            0,  # p_offset
            BASE,  # p_vaddr
            BASE,  # p_paddr
            TOTAL,  # p_filesz
            TOTAL,  # p_memsz
            0x1000,  # p_align
        )
        assert len(phdr) == PHDR_SIZE

        elf_bytes = ehdr + phdr + code
        elf_path = tmp_path / "mips64le_branch.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "MIPS64"
        assert proj.arch.bits == 64
        assert proj.arch.memory_endness == "Iend_LE", (
            f"cle ELF loader did not pick up EI_DATA=LSB for MIPS64: got {proj.arch.memory_endness}"
        )
        assert proj.entry == ENTRY, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {ENTRY:#x}"

        state = proj.factory.blank_state(addr=proj.entry)
        a0 = claripy.BVS("a0", 64)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        # Branch targets are at +0x14 / +0x18 from the first instruction,
        # which sits at ENTRY (just past the ELF header / phdr).
        mgr.explore(
            find=ENTRY + 0x14,
            avoid=ENTRY + 0x18,
            num_find=1,
            max_steps=50,
        )

        assert len(mgr.found) >= 1, (
            f"MIPS64 LE ELF exploration did not reach {ENTRY + 0x14:#x}; counts={mgr.stash_counts()}"
        )
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        assert found.solver.eval(a0) == 42, f"Expected a0==42 to reach found, got {found.solver.eval(a0)}"

    def test_mips64_explore_be_real_elf(self, tmp_path):
        """End-to-end MIPS64 big-endian exploration on a hand-assembled ELF.

        Closes the last gap in the ``ig3o`` Arch BE validation campaign
        (sibling of ``test_armeb_explore_blob``, angr-ig3o.1). MIPS64
        BE was wired up (register table, N64 calling convention) and
        the LE path was end-to-end tested via
        ``test_mips64_explore_le_real_elf``, but BE memory + instruction
        layout had never been driven through the Rust interpreter on a
        multi-block control flow — the same risk class as the latent
        Cdecl x86 EAX/EDX bug (commit 5329d8222).

        MIPS64 base-integer instruction encodings match MIPS32 for the
        opcodes used here; the storage byte order changes with EI_DATA.
        The ELF wrapper is ELF64 with ``EI_CLASS=ELF64`` and
        ``EI_DATA=MSB``, exercising the N64 calling convention path on
        BE memory.

        No MIPS64 binaries ship with angr-examples and no cross-compiler
        is available locally, so the ELF is constructed inline (same
        pattern as ``test_mips64_explore_le_real_elf``).

        Pairs with the ``mips64_be_branch`` synthetic benchmark.
        """
        import struct

        import claripy

        # MIPS64 instruction encodings — same as MIPS32 for these opcodes
        # since registers are still 5 bits. ADDIU sign-extends the 16-bit
        # immediate to 64 bits on MIPS64. Stored big-endian.
        #   0x00: ADDIU t0, zero, 42     2408002A  ; $t0 = sign-extend(42)
        #   0x04: BEQ a0, t0, +3         10880003  ; on equal -> +0x14 (found)
        #   0x08: NOP (delay slot)       00000000
        #   0x0c: B +2 (BEQ zero,zero)   10000002  ; -> +0x18 (avoid)
        #   0x10: NOP (delay slot)       00000000
        #   0x14: NOP (found)            00000000
        #   0x18: NOP (avoid)            00000000
        code = struct.pack(
            ">IIIIIII",
            0x2408002A,
            0x10880003,
            0x00000000,
            0x10000002,
            0x00000000,
            0x00000000,
            0x00000000,
        )

        # Minimal ELF64 (MIPS64 BE) header. PT_LOAD covers file
        # [0, 120+len(code)] → vaddr [0x400000, ...). Entry = first instr.
        BASE = 0x400000
        EHDR_SIZE = 64  # ELF64 header
        PHDR_SIZE = 56  # ELF64 program header
        TOTAL = EHDR_SIZE + PHDR_SIZE + len(code)
        ENTRY = BASE + EHDR_SIZE + PHDR_SIZE

        # ELF64 header (big-endian)
        ehdr = (
            b"\x7fELF"
            + bytes(
                [
                    2,  # EI_CLASS = ELF64
                    2,  # EI_DATA = MSB (big-endian)
                    1,  # EI_VERSION
                    0,  # EI_OSABI = System V
                    0,  # EI_ABIVERSION
                ]
            )
            + b"\x00" * 7
        )  # EI_PAD
        ehdr += struct.pack(
            ">HHIQQQIHHHHHH",
            2,  # e_type = ET_EXEC
            0x08,  # e_machine = EM_MIPS
            1,  # e_version
            ENTRY,  # e_entry
            EHDR_SIZE,  # e_phoff
            0,  # e_shoff
            0x60000000,  # e_flags = EF_MIPS_ARCH_64 (N64 implied by EI_CLASS=ELF64)
            EHDR_SIZE,  # e_ehsize
            PHDR_SIZE,  # e_phentsize
            1,  # e_phnum
            0,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )
        assert len(ehdr) == EHDR_SIZE

        # ELF64 PT_LOAD program header (field order: p_type, p_flags first)
        phdr = struct.pack(
            ">IIQQQQQQ",
            1,  # p_type = PT_LOAD
            5,  # p_flags = PF_R | PF_X
            0,  # p_offset
            BASE,  # p_vaddr
            BASE,  # p_paddr
            TOTAL,  # p_filesz
            TOTAL,  # p_memsz
            0x1000,  # p_align
        )
        assert len(phdr) == PHDR_SIZE

        elf_bytes = ehdr + phdr + code
        elf_path = tmp_path / "mips64be_branch.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "MIPS64"
        assert proj.arch.bits == 64
        assert proj.arch.memory_endness == "Iend_BE", (
            f"cle ELF loader did not pick up EI_DATA=MSB for MIPS64: got {proj.arch.memory_endness}"
        )
        assert proj.arch.instruction_endness == "Iend_BE"
        assert proj.entry == ENTRY, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {ENTRY:#x}"

        state = proj.factory.blank_state(addr=proj.entry)
        a0 = claripy.BVS("a0", 64)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        # Branch targets are at +0x14 / +0x18 from the first instruction,
        # which sits at ENTRY (just past the ELF header / phdr).
        mgr.explore(
            find=ENTRY + 0x14,
            avoid=ENTRY + 0x18,
            num_find=1,
            max_steps=50,
        )

        assert len(mgr.found) >= 1, (
            f"MIPS64 BE ELF exploration did not reach {ENTRY + 0x14:#x}; counts={mgr.stash_counts()}"
        )
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        assert found.solver.eval(a0) == 42, f"Expected a0==42 to reach found, got {found.solver.eval(a0)}"

    # ------------------------------------------------------------------
    # SimProcedure round-trip tests (angr-orc9). One per non-amd64 arch:
    # exercise the calling-convention path through the dispatcher
    # (arg extraction → native procedure → return-value placement →
    # return-address handoff). Same regression-class as the latent
    # Cdecl x86 EAX/EDX bug (commit 5329d8222) — without these tests,
    # a register-offset or return-addr error stays silent until end-to-end
    # binary work happens to depend on it.
    # ------------------------------------------------------------------

    def test_arm_native_procedure_round_trip(self):
        """ARMEABI: native strlen runs, return lands in r0, PC = LR.

        ARM uses BL which stores the return address in LR (R14, offset 64),
        not on the stack. The dispatcher must read LR after a native
        procedure returns, not pop a stack frame.
        """
        mgr = _RustExplorationManager("arm")

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFF0000

        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState("arm")
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")

        # ARMEABI: r0 = arg0; LR = return addr.
        state.set_register("r0", STRING_ADDR)
        state.set_register("lr", EXIT_HOOK)
        state.set_register("sp", STACK_BASE)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"expected exactly one deadended state after exit hook fired; stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]
        r0 = mgr.get_state_register(sid, "r0")
        assert r0 == 5, (
            f"strlen('hello') should return 5 in r0, got {r0!r}. native_calls={mgr.native_procedure_stats()}"
        )
        # SP must be untouched: ARM doesn't push the return address.
        sp = mgr.get_state_register(sid, "sp")
        assert sp == STACK_BASE, (
            f"ARM SP changed from {STACK_BASE:#x} to {sp:#x}; "
            f"the dispatcher should NOT pop a return address from the stack "
            f"because BL stores it in LR."
        )

    def test_aarch64_native_procedure_round_trip(self):
        """AArch64: native strlen runs, return lands in x0, PC = X30 (LR).

        AArch64 uses BL which stores the return address in X30 (offset 256),
        not on the stack. Mirrors the ARM round-trip test.
        """
        mgr = _RustExplorationManager("aarch64")

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFFFFFFE000

        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState("aarch64")
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")

        # AArch64: x0 = arg0; X30 (LR) = return addr.
        state.set_register("x0", STRING_ADDR)
        state.set_register("x30", EXIT_HOOK)
        state.set_register("sp", STACK_BASE)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"expected exactly one deadended state after exit hook fired; stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]
        x0 = mgr.get_state_register(sid, "x0")
        assert x0 == 5, (
            f"strlen('hello') should return 5 in x0, got {x0!r}. native_calls={mgr.native_procedure_stats()}"
        )
        sp = mgr.get_state_register(sid, "sp")
        assert sp == STACK_BASE, (
            f"AArch64 SP changed from {STACK_BASE:#x} to {sp:#x}; "
            f"the dispatcher should NOT pop a return address from the stack."
        )

    def test_mips32_native_procedure_round_trip(self):
        """MIPS32 (O32): native strlen runs, return lands in $v0, PC = $ra.

        MIPS uses JAL which stores the return address in $ra (R31, offset 132),
        not on the stack. Args are passed in $a0-$a3 (R4-R7). Return value
        in $v0 (R2, offset 16). Locks the calling-convention bug class — if
        MIPS falls back to SystemVAMD64, the dispatcher would read RDI=72
        = MIPS R12 instead of $a0=24, and the native procedure would see
        garbage args.
        """
        mgr = _RustExplorationManager("mips32")

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFF0000

        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState("mips32")
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")

        # MIPS O32: $a0 = arg0; $ra = return addr.
        state.set_register("a0", STRING_ADDR)
        state.set_register("ra", EXIT_HOOK)
        state.set_register("sp", STACK_BASE)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"expected exactly one deadended state after exit hook fired; stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]
        v0 = mgr.get_state_register(sid, "v0")
        assert v0 == 5, (
            f"strlen('hello') should return 5 in $v0, got {v0!r}. native_calls={mgr.native_procedure_stats()}"
        )
        sp = mgr.get_state_register(sid, "sp")
        assert sp == STACK_BASE, (
            f"MIPS SP changed from {STACK_BASE:#x} to {sp:#x}; "
            f"the dispatcher should NOT pop a return address from the stack."
        )

    def test_mips64_native_procedure_round_trip(self):
        """MIPS64 (N64): native strlen runs, return lands in $v0, PC = $ra.

        Guards against the silent SystemV_AMD64 fallback that motivated
        angr-gzk8: before MipsN64 was added to default_cc_for_arch, a
        MIPS64 SimProcedure would extract its first arg from offset 72
        (AMD64 RDI) instead of offset 48 (MIPS64 $a0), and the dispatcher
        would have popped a return address off the stack instead of
        leaving $ra/$sp alone.

        Args: $a0-$a7 (R4-R11, VEX offsets 48..104). Return value: $v0
        (R2, offset 32). Return addr: $ra (R31, offset 264).
        """
        mgr = _RustExplorationManager("mips64")

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000
        STACK_BASE = 0x7FFF0000

        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState("mips64")
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")

        # MIPS N64: $a0 = arg0; $ra = return addr.
        state.set_register("a0", STRING_ADDR)
        state.set_register("ra", EXIT_HOOK)
        state.set_register("sp", STACK_BASE)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"expected exactly one deadended state after exit hook fired; stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]
        v0 = mgr.get_state_register(sid, "v0")
        assert v0 == 5, (
            f"strlen('hello') should return 5 in $v0, got {v0!r}. native_calls={mgr.native_procedure_stats()}"
        )
        sp = mgr.get_state_register(sid, "sp")
        assert sp == STACK_BASE, (
            f"MIPS64 SP changed from {STACK_BASE:#x} to {sp:#x}; "
            f"the dispatcher should NOT pop a return address from the stack."
        )
