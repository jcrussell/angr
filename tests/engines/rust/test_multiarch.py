"""Multi-architecture coverage for the Rust engine.

Most of this module is a **table-driven sweep**: the per-architecture facts
live in :mod:`tests.engines.rust.arch_specs` and each assertion runs against
every row of :data:`~tests.engines.rust.arch_specs.ARCH_SPECS`. Before that
table existed, coverage of a minority arch depended on someone remembering to
hand-author a test for it, so arch-specific bugs slipped through
disproportionately (BE ``htonl``/``htons``/``pipe`` byte order, 32-bit
``scanf``/``sprintf`` length modifiers, unmapped NEON unsigned vector compares
— all angr-9ke6b findings). Adding an assertion or an architecture now buys the
whole matrix at once; ``test_arch_offset_parity.py`` is the same idiom applied
to the register tables.

Tests that are *genuinely* arch-specific — x86 segment selectors, the amd64
``gs_const``/``sseround`` collision, the AArch64 NEON MLA blob, the AArch64
concrete-flag ccall, the MIPS32 disk-init-cache regression — stay hand-written
below the sweeps rather than being bent into table columns.
"""

from __future__ import annotations

import os
import struct

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
from tests.engines.rust.arch_specs import ARCH_SPECS, ArchSpec, EndianVariant, build_elf, spec_for

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


def _explore_cases() -> list[tuple[ArchSpec, EndianVariant, str]]:
    """``(spec, variant, loader)`` rows for the end-to-end explore sweep.

    Every (endianness flavour x loader backend) cell is covered. The two
    loaders are not redundant: cle's Blob backend takes the architecture name
    verbatim, so only the ELF rows exercise ``e_machine`` / ``EI_DATA`` /
    ``e_entry`` decoding.
    """
    return [(spec, variant, loader) for spec in ARCH_SPECS for variant in spec.variants for loader in ("blob", "elf")]


def _explore_id(case: tuple[ArchSpec, EndianVariant, str]) -> str:
    _spec, variant, loader = case
    return f"{variant.label}-{loader}"


EXPLORE_CASES = _explore_cases()
EXPLORE_IDS = [_explore_id(c) for c in EXPLORE_CASES]

#: The link-register architectures, i.e. the rows with a
#: :class:`~tests.engines.rust.arch_specs.ProcSpec`.
PROC_SPECS = [spec for spec in ARCH_SPECS if spec.proc is not None]


def _probe_values(spec: ArchSpec) -> dict[str, int]:
    """Distinct per-register test values, width-masked using archinfo.

    Widths come from archinfo rather than from the spec table so a row cannot
    encode a wrong width; the values are pairwise distinct so a wrong offset
    surfaces as cross-talk between two probes rather than a clean pass.
    """
    import archinfo

    registers = archinfo.arch_from_id(spec.archinfo_id).registers
    values = {}
    for i, name in enumerate(spec.probe_regs):
        size = registers[name][1]
        values[name] = (0x1122334455667788 + 0x1111111111111111 * i) & ((1 << (size * 8)) - 1)
    return values


def _load_explore_project(spec: ArchSpec, variant: EndianVariant, loader: str, tmp_path):
    """Load ``spec``'s explore program through ``loader``; return (proj, entry)."""
    code = spec.code_for(variant)
    if loader == "blob":
        blob_path = tmp_path / f"{variant.label}.bin"
        blob_path.write_bytes(code)
        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": variant.cle_arch, "base_addr": spec.base_addr},
            auto_load_libs=False,
        )
        return proj, spec.base_addr

    elf_bytes, entry = build_elf(spec, variant, code)
    elf_path = tmp_path / f"{variant.label}.elf"
    elf_path.write_bytes(elf_bytes)
    proj = angr.Project(str(elf_path), auto_load_libs=False)
    assert proj.entry == entry, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {entry:#x}"
    return proj, entry


class TestMultiArchSupport:
    """Tests for MIPS, ARM, x86 and big-endian architecture support."""

    # ------------------------------------------------------------------
    # Table-driven sweeps (angr-9ke6b.215). Each of these runs on every row
    # of ARCH_SPECS; a new architecture only has to add a row.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("spec", ARCH_SPECS, ids=lambda s: s.rust_arch)
    def test_state_creation_round_trips_registers(self, spec):
        """Fresh state creation plus a register round-trip on every arch.

        Replaces the per-arch ``test_*_state_creation`` copies (which existed
        only for mips32/arm/arm64/x86). Writing all probes *before* reading any
        of them back makes a wrong offset visible as cross-talk, which a
        write/read-one-at-a-time loop would miss.
        """
        state = RustSimState(spec.rust_arch)
        values = _probe_values(spec)

        for name, value in values.items():
            state.set_register(name, value)
        for name, value in values.items():
            assert state.get_register(name) == value, (
                f"{spec.rust_arch}.{name}: wrote {value:#x}, read back {state.get_register(name):#x}"
            )

    @pytest.mark.parametrize("spec", ARCH_SPECS, ids=lambda s: s.rust_arch)
    def test_fork_isolation(self, spec):
        """Forked states have independent registers on every arch.

        Generalizes the former mips32-only ``test_mips32_fork_isolation``; the
        Rust ``RegisterFile`` shares its buffer via ``Arc`` and relies on
        ``Arc::make_mut`` copy-on-write, so a broken CoW path would leak writes
        across the parent/child boundary on *all* arches at once.
        """
        name = spec.probe_regs[0]
        parent = RustSimState(spec.rust_arch)
        parent.set_register(name, 100)
        child = parent.fork()
        child.set_register(name, 200)

        assert parent.get_register(name) == 100, f"{spec.rust_arch}: child write leaked into parent"
        assert child.get_register(name) == 200, f"{spec.rust_arch}: child write was lost"

    @pytest.mark.parametrize("spec", ARCH_SPECS, ids=lambda s: s.rust_arch)
    def test_exploration_manager_creation(self, spec):
        """Manager construction and stash ops work on every arch.

        Generalizes the former mips32-only / arm-only copies. ``set_find_addrs``
        / ``set_avoid_addrs`` are included because they were only ever
        exercised on mips32.
        """
        mgr = _RustExplorationManager(spec.rust_arch)
        assert mgr.arch == spec.rust_arch

        mgr.create_state("active")
        assert mgr.active_count() == 1
        mgr.set_find_addrs([spec.base_addr])
        mgr.set_avoid_addrs([spec.base_addr + 0x100])

    @pytest.mark.parametrize(("spec", "variant", "loader"), EXPLORE_CASES, ids=EXPLORE_IDS)
    def test_explore_solves_for_42(self, spec, variant, loader, tmp_path):
        """End-to-end exploration drives the input register to 42, everywhere.

        Replaces the seven hand-authored ``*_explore_blob`` /
        ``*_explore_*_real_elf`` copies, which between them covered only
        x86-blob, armeb-blob, aarch64-blob, aarch64-elf, mips32be-blob,
        mips32le-elf, mips64le-elf and mips64be-elf. The sweep is a strict
        superset: it also covers amd64, ARMEL, and the missing loader half of
        every other row.

        Exercises, per cell: cle's loader (``e_machine`` / ``EI_DATA`` /
        ``e_entry`` decoding on the ELF rows), instruction fetch in the
        variant's byte order, VEX interpretation, register sync, conditional
        branch resolution, and solver evaluation. This is the same risk class
        as the latent Cdecl x86 EAX/EDX bug (commit 5329d8222): without an
        end-to-end test per arch, register-offset and byte-order regressions
        stay silent for months.
        """
        import claripy

        proj, entry = _load_explore_project(spec, variant, loader, tmp_path)
        assert proj.arch.name == spec.proj_arch_name
        assert proj.arch.bits == spec.bits
        assert proj.arch.memory_endness == variant.memory_endness, (
            f"{variant.label}/{loader}: memory_endness is {proj.arch.memory_endness}, expected {variant.memory_endness}"
        )
        assert proj.arch.instruction_endness == variant.instruction_endness

        state = proj.factory.blank_state(addr=entry)
        sym = claripy.BVS(spec.input_reg, spec.input_bits)
        setattr(state.regs, spec.input_reg, sym)

        mgr = RustExplorationManager(proj, [state])
        find, avoid = entry + spec.find_off, entry + spec.avoid_off
        mgr.explore(find=find, avoid=avoid, num_find=1, max_steps=100)

        assert len(mgr.found) >= 1, (
            f"{variant.label}/{loader}: exploration did not reach {find:#x}; counts={mgr.stash_counts()}"
        )
        found = mgr.found[0]
        assert found.solver.satisfiable(), f"{variant.label}/{loader}: found state's solver became unsat"
        got = found.solver.eval(sym)
        assert got == 42, f"{variant.label}/{loader}: expected {spec.input_reg}==42 to reach found, got {got}"

    @pytest.mark.parametrize("spec", PROC_SPECS, ids=lambda s: s.rust_arch)
    def test_native_procedure_round_trip(self, spec):
        """Native strlen runs, the result lands in the return register, SP is untouched.

        Replaces the four copies (arm / aarch64 / mips32 / mips64). Each of
        these architectures passes the return address in a link register
        (ARM ``lr``, AArch64 ``x30``, MIPS ``$ra``) rather than on the stack,
        so the dispatcher must *not* pop a stack frame — hence the SP
        assertion. Locks the calling-convention bug class that motivated
        angr-gzk8 and angr-orc9: with a silent fall-through to SystemVAMD64,
        a MIPS64 SimProcedure would read its first argument from offset 72
        (AMD64 RDI) instead of 48 ($a0) and see garbage.
        """
        proc = spec.proc
        mgr = _RustExplorationManager(spec.rust_arch)

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        STRLEN_HOOK = 0x500000
        EXIT_HOOK = 0x600000
        STRING_ADDR = 0x2000

        mgr.register_simprocedure(STRLEN_HOOK, "strlen", num_args=1, no_return=False)
        mgr.register_simprocedure(EXIT_HOOK, "exit", num_args=1, no_return=True)

        state = RustSimState(spec.rust_arch)
        state.map_memory(STRING_ADDR & ~0xFFF, 0x1000, 7)
        state.map_memory(proc.stack_base, 0x1000, 7)
        state.memory_store(STRING_ADDR, b"hello\x00")

        state.set_register(proc.arg_reg, STRING_ADDR)
        state.set_register(proc.ret_addr_reg, EXIT_HOOK)
        state.set_register("sp", proc.stack_base)
        state.pc = STRLEN_HOOK

        mgr.add_state("active", state)
        mgr.run(10)

        deadended_ids = mgr.get_state_ids("deadended")
        assert len(deadended_ids) == 1, (
            f"{spec.rust_arch}: expected exactly one deadended state after the exit hook fired; "
            f"stashes={mgr.stash_counts()}"
        )
        sid = deadended_ids[0]
        ret = mgr.get_state_register(sid, proc.ret_reg)
        assert ret == 5, (
            f"{spec.rust_arch}: strlen('hello') should return 5 in {proc.ret_reg}, got {ret!r}. "
            f"native_calls={mgr.native_procedure_stats()}"
        )
        sp = mgr.get_state_register(sid, "sp")
        assert sp == proc.stack_base, (
            f"{spec.rust_arch}: SP changed from {proc.stack_base:#x} to {sp:#x}; the dispatcher should NOT pop a "
            f"return address from the stack because the return address arrives in {proc.ret_addr_reg}."
        )

    # ------------------------------------------------------------------
    # Hand-written, genuinely arch-specific tests.
    # ------------------------------------------------------------------

    def test_mips_alias_constructors_do_not_panic(self):
        """Every MIPS32 arch alias that ``arch_from_name`` accepts must also
        resolve a calling convention, so the ``#[new]`` constructor succeeds
        rather than panicking in ``default_cc_for_arch``.

        Regression for angr-n0irt.6: ``arch_from_name`` accepted ``"mipsle"``
        but the MIPS O32 calling convention's alias list (then a per-CC
        ``ARCH_ALIASES`` const, now the MIPS32 row of ``ALL_ARCHES`` in
        ``arch/mod.rs``) listed ``"mipsbe"`` instead, so
        ``_RustExplorationManager("mipsle")`` passed the arch lookup then
        panicked (PyO3 catch_unwind -> PanicException) on the very next line —
        a *recognized* alias failing more violently than an unrecognized one.
        """
        for alias in ("mips", "mips32", "mipsel", "mipsle"):
            mgr = _RustExplorationManager(alias)
            assert mgr.arch == alias

    def test_unknown_arch_raises_valueerror(self):
        """A genuinely-unrecognized arch name raises a clean ValueError, not a
        PanicException — the graceful failure `mipsle` used to bypass."""
        with pytest.raises(ValueError, match="unsupported architecture"):
            _RustExplorationManager("nonexistent-arch")

    def test_mips32_big_endian(self):
        """MIPS32 big-endian memory lays out multi-byte values MSB-first.

        Regression for angr-a116s.7: the prior body only did an
        endian-agnostic register round-trip (RegisterFile::put_reg/get_reg
        are pure offset+size ops with no byte ordering), so silently
        dropping the ``little_endian=False`` override was invisible.
        MIPS32's arch default is little-endian, so the override is the only
        thing that selects BE — exercise a memory store whose physical byte
        layout differs under BE vs LE to make the override load-bearing.
        """
        be = RustSimState("mips32", little_endian=False)
        be.map_memory(0x1000, 0x1000, 7)  # RWX
        # Raw input bytes pack little-endian into the value 0x11223344,
        # which the SymbolicMemory then lays out per its endness.
        be.memory_store(0x1000, b"\x44\x33\x22\x11")
        # Big-endian: most-significant byte (0x11) lands at the lowest addr.
        assert be.memory_load(0x1000, 1)[0] == 0x11
        # A full-width round-trip is endness-symmetric, so it still recovers
        # the original input bytes regardless of endianness.
        assert list(be.memory_load(0x1000, 4)) == [0x44, 0x33, 0x22, 0x11]

        # Contrast with the little-endian default: LSB (0x44) at lowest addr.
        # If the override were dropped, `be` above would behave like this and
        # the MSB assertion would fail.
        le = RustSimState("mips32")
        le.map_memory(0x1000, 0x1000, 7)
        le.memory_store(0x1000, b"\x44\x33\x22\x11")
        assert le.memory_load(0x1000, 1)[0] == 0x44

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

    def test_mips64_fpu_offsets_match_archinfo(self):
        """MIPS64 FPU register offsets are ground-truth-checked against
        archinfo, not just self-consistent round-trips.

        Covers angr-mpln0: the whole ``offsets64`` FPU block (F0-F31 +
        FIR/FCCR/FEXR/FENR/FCSR) was +8 off the VEX guest layout. Because
        get/set share the same table, an in-isolation round-trip (the
        angr-w2gj.2 test) passed despite every name aliasing the adjacent
        register. This test writes each FPU name and reads the raw register
        file at the *archinfo* offset — the interpreter's ground truth — so
        a wrong offset fails here. The raw guest state is stored host-native
        little-endian regardless of arch endianness.
        """
        import archinfo

        arch = archinfo.ArchMIPS64()
        state = RustSimState("mips64")

        fpu_names = [f"f{i}" for i in range(32)] + [
            "fir",
            "fccr",
            "fexr",
            "fenr",
            "fcsr",
        ]
        # Distinct per-register value so a +8 alias lands the wrong bytes.
        for idx, name in enumerate(fpu_names):
            off, size = arch.registers[name]
            value = (0x1000000000000000 + idx) & ((1 << (size * 8)) - 1)
            state.set_register(name, value)
            raw = state.get_registers_raw()
            got = int.from_bytes(raw[off : off + size], "little")
            assert got == value, (
                f"{name}: wrote {value:#x}, archinfo offset {off} holds {got:#x} (offset table drifted from VEX layout)"
            )

        # fcsr must NOT clobber guest_ULR (VEX offset 576) — the pre-fix
        # +8 drift wrote fcsr straight into the TLS/userlocal register.
        state2 = RustSimState("mips64")
        state2.set_register("fcsr", 0xDEADBEEF)
        raw2 = state2.get_registers_raw()
        assert int.from_bytes(raw2[576:584], "little") == 0, "fcsr write leaked into guest_ULR (offset 576)"

    @pytest.mark.parametrize("arch_name", ["MIPS32", "MIPS64"])
    def test_mips_sync_reg_map_matches_archinfo(self, arch_name):
        """MIPS GPR sync-back tables ground-truth-checked against archinfo.

        Covers angr-hrcds: `_get_reg_map_and_return_regs` /
        `_get_arch_register_names` (angr/exploration/rust_state_sync.py) had
        no MIPS arms, so a MIPS Python-SimProcedure bounce dropped every
        register the proc wrote — including the $v0/$v1 return value — on
        sync-back, silently returning a stale result on a "Supported" arch.
        A wrong offset or missing name here re-breaks that soundness path.
        """
        import archinfo

        from angr.exploration.rust_state_sync import RustStateSyncMixin

        arch = getattr(archinfo, "Arch" + arch_name)(archinfo.Endness.LE)

        # No "Unknown architecture" warning, and non-None tables.
        with self._assert_no_unknown_arch_warning():
            reg_map, return_regs = RustStateSyncMixin._get_reg_map_and_return_regs(arch)
        assert reg_map is not None and return_regs is not None

        # Return value lives in $v0/$v1 — the whole point of the fix.
        assert return_regs == {"v0", "v1"}

        # Every mapped (offset, size) must match the VEX guest layout archinfo
        # reports, else a SimProc write lands in the wrong register on sync-back.
        for name, (off, size) in reg_map.items():
            assert arch.registers[name] == (off, size), (
                f"{arch_name} {name}: sync table has ({off}, {size}), archinfo has {arch.registers[name]}"
            )

        # The name table used by export_callback_bundle must cover the same set.
        names = RustStateSyncMixin._get_arch_register_names(RustStateSyncMixin, arch)
        assert set(names) == set(reg_map)
        assert "v0" in names and "v1" in names

    @staticmethod
    def _assert_no_unknown_arch_warning():
        import contextlib

        from angr.exploration import rust_state_sync

        @contextlib.contextmanager
        def _cm():
            seen = []
            orig = rust_state_sync.l.warning
            rust_state_sync.l.warning = lambda msg, *a, **k: seen.append(msg)
            try:
                yield
            finally:
                rust_state_sync.l.warning = orig
            assert not any("Unknown architecture" in str(m) for m in seen), seen

        return _cm()

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
        # fp aliases r11. r12 has no "ip" alias on purpose (angr-itm3u):
        # archinfo reserves "ip" for the instruction pointer (R15T) on every
        # arch, so the ARM-ABI reading of "ip" as r12 would silently redirect
        # a PC write into a scratch register.
        assert state.get_register("fp") == 0x11111111
        with pytest.raises(ValueError):
            state.get_register("ip")

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

    def test_arm32_offsets_match_archinfo(self):
        """ARM32 register offsets are ground-truth-checked against archinfo,
        not just self-consistent round-trips.

        Covers angr-ihfe5: the ``mod offsets`` table omitted the 16-byte VEX
        block at 108-127 (emnote/cmstart/cmlen/nraddr/ip_at_syscall), so every
        field from D0 on was shifted -16 vs the real VEX/pyvex layout
        (D0=112 not 128, fpscr=368 not 384, tpidruro=372 not 388,
        itstate=376 not 392). Because get/set share the same table, the
        paired ``test_arm32_full_register_family_dispatch`` round-trip passed
        despite every name aliasing the wrong slot, and IR-level PUTs to the
        real fpscr/tpidruro/itstate offsets fell past the 380-byte buffer and
        were silently dropped. This writes each name and reads the raw
        register file at the *archinfo* offset — the interpreter's ground
        truth — so a wrong offset fails here.
        """
        import archinfo

        arch = archinfo.ArchARM()
        state = RustSimState("arm")

        # Every register whose offset the -16 drift moved, plus the newly
        # named 108-127 block. Values are per-register-distinct so a -16 alias
        # lands in the wrong slot.
        names = (
            [f"d{i}" for i in range(32)]
            + ["fpscr", "tpidruro", "itstate"]
            + ["emnote", "cmstart", "cmlen", "nraddr", "ip_at_syscall"]
        )
        for idx, name in enumerate(names):
            off, size = arch.registers[name]
            value = (0x1122334455667788 + idx) & ((1 << (size * 8)) - 1)
            state.set_register(name, value)
            raw = state.get_registers_raw()
            got = int.from_bytes(raw[off : off + size], "little")
            assert got == value, (
                f"{name}: wrote {value:#x}, archinfo offset {off} holds {got:#x} "
                "(ARM offset table drifted from VEX layout)"
            )

        # tpidruro (VEX 388) is the TLS base a `mrc p15,0,rN,c13,c0,3` GET
        # reads. Pre-fix its Rust slot was 372, aliasing d31's high word; a
        # write there must now land at 388 and leave the old 372 slot alone.
        state2 = RustSimState("arm")
        state2.set_register("tpidruro", 0x76543210)
        raw2 = state2.get_registers_raw()
        assert int.from_bytes(raw2[388:392], "little") == 0x76543210
        assert int.from_bytes(raw2[372:376], "little") == 0, "tpidruro write leaked into the pre-fix -16 slot (372)"

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

        Covers angr-5spy.2: ``ldt`` (304, 8B per archinfo) and ``gdt``
        (312, 8B per archinfo) are reachable through
        ``set_register``/``get_register`` after wiring them into
        ``arch/x86.rs::ALIASES``.

        Covers angr-rfxc7: x86 has **no** ``fs_const``/``gs_const`` —
        that pair is amd64-only (``arch_prctl`` is an amd64 syscall).
        The former Rust-only placeholders sat on VEX's live
        ``guest_EMNOTE``/``guest_CMSTART`` (320/324), so writing them
        by name silently clobbered guest state; they are now named
        after the real VEX fields instead.

        A wrong-offset bug would surface as cross-talk between
        neighbouring slots (writing ``ldt`` would corrupt ``gdt`` or
        vice versa).
        """
        state = RustSimState("x86")

        # Distinct values at each slot so a misaligned offset surfaces
        # as a cross-talk failure on the assertion side.
        state.set_register("ldt", 0x1122334455667788)
        state.set_register("gdt", 0xAABBCCDDEEFF0011)
        state.set_register("emnote", 0xCAFE_BABE)
        state.set_register("cmstart", 0xDEAD_BEEF)

        assert state.get_register("ldt") == 0x1122334455667788
        assert state.get_register("gdt") == 0xAABBCCDDEEFF0011
        assert state.get_register("emnote") == 0xCAFE_BABE
        assert state.get_register("cmstart") == 0xDEAD_BEEF

        # x86 must reject the amd64-only segment-base names outright.
        with pytest.raises(Exception):
            state.set_register("fs_const", 0x1234)

        # Rewriting emnote must not disturb cmstart (or vice versa),
        # since the two slots are adjacent (320/324, both 4B) and a
        # wrong-width write would bleed across the boundary.
        state.set_register("emnote", 0x1234_5678)
        assert state.get_register("emnote") == 0x1234_5678
        assert state.get_register("cmstart") == 0xDEAD_BEEF

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

    def test_arm32_explore_real_binary(self):
        """End-to-end ARM32 (ARMEL) exploration on a real binary.

        The synthetic ARM rows of ``test_explore_solves_for_42`` are
        hand-assembled straight-line code; this one runs a compiler-produced
        binary with a real prologue, library calls and a symbolic buffer in
        memory, so it catches things a nine-instruction blob cannot.

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

    def test_aarch64_concrete_branch_no_spurious_fork(self, tmp_path):
        """arm64g_calculate_condition must not fork on concrete flags (angr-37d4).

        Before arm64g ccalls were implemented, the ``is_cond_ccall``
        catch-all returned a *fresh unconstrained* symbolic for every
        ``arm64g_calculate_condition``, so a B.EQ whose flags are fully
        concrete still produced two successors — the infeasible ``avoid``
        path got explored, diverging from the Python engine.

        Here ``w0`` is *concrete* (42), so CMP w0,#42 sets Z=1 concretely
        and B.EQ is concretely taken. With the ccall computed properly the
        false branch is infeasible and never created, so the ``avoid``
        stash stays empty. Reuses the arm64 row's explore program.
        """
        spec = spec_for("arm64")
        variant = spec.variants[0]
        blob_path = tmp_path / "aarch64_concrete_branch.bin"
        blob_path.write_bytes(spec.code_for(variant))

        proj = angr.Project(
            str(blob_path),
            main_opts={"backend": "blob", "arch": variant.cle_arch, "base_addr": spec.base_addr},
            auto_load_libs=False,
        )

        state = proj.factory.blank_state(addr=spec.base_addr)
        state.regs.x0 = 42  # concrete -> B.EQ concretely taken

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(
            find=spec.base_addr + spec.find_off,
            avoid=spec.base_addr + spec.avoid_off,
            num_find=1,
            max_steps=50,
        )

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
        """End-to-end AArch64 exploration across a BL / RET boundary.

        The ELF rows of ``test_explore_solves_for_42`` already cover cle's
        ELF backend (e_machine / e_entry / PT_LOAD) on every arch. What is
        unique here is the *call*: the program calls a subroutine
        ``double_it`` that returns ``2 * w0``, so BL must set X30 and RET
        must branch to it. That calling-convention plumbing has no
        equivalent in the straight-line sweep programs.

        No AArch64 binaries ship with angr-examples and no cross-compiler is
        available locally, so the ELF is constructed inline via
        ``arch_specs.build_elf``. Pairs with angr-gxhf.1.
        """
        import claripy

        # AArch64 little-endian instructions (verified against ARMv8 ARM):
        #   double_it:
        #     +0x00: add w0, w0, w0       0B000000  ; w0 = 2*w0
        #     +0x04: ret                  D65F03C0  ; return via x30
        #   entry (== double_it + 8):
        #     +0x08: bl double_it (-8)    97FFFFFE
        #     +0x0c: movz w1, #84         52800A81
        #     +0x10: cmp w0, w1           6B01001F  ; subs wzr, w0, w1
        #     +0x14: b.eq found (+8)      54000040
        #     +0x18: b avoid (+12)        14000003
        #   found (== entry + 0x14):
        #     +0x1c: nop                  D503201F
        #     +0x20: nop                  D503201F
        #   avoid (== entry + 0x1c):
        #     +0x24: nop                  D503201F
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

        spec = spec_for("arm64")
        variant = spec.variants[0]
        # e_entry skips the two-instruction subroutine at the top of `code`.
        elf_bytes, entry = build_elf(spec, variant, code, entry_skip=8)
        elf_path = tmp_path / "aarch64_call.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "AARCH64"
        assert proj.entry == entry, f"e_entry not parsed: proj.entry={proj.entry:#x} vs expected {entry:#x}"

        state = proj.factory.blank_state(addr=proj.entry)
        x0 = claripy.BVS("x0", 64)
        state.regs.x0 = x0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=entry + 0x14, avoid=entry + 0x1C, num_find=1, max_steps=50)

        assert len(mgr.found) >= 1, (
            f"AArch64 ELF exploration did not reach {entry + 0x14:#x}; counts={mgr.stash_counts()}"
        )
        found = mgr.found[0]
        assert found.solver.satisfiable(), "found state's solver became unsat"
        # The doubled input must equal 84, so the input must be 42.
        assert found.solver.eval(x0) == 42, f"Expected x0==42 (so 2*x0==84) to reach found, got {found.solver.eval(x0)}"

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
        spec = spec_for("mips32")
        # The little-endian MIPS32 row; the cache bug is endness-independent.
        variant = next(v for v in spec.variants if v.little)
        elf_bytes, entry = build_elf(spec, variant, code_bytes)
        elf_path = tmp_path / "mips32_accum.elf"
        elf_path.write_bytes(elf_bytes)

        proj = angr.Project(str(elf_path), auto_load_libs=False)
        assert proj.arch.name == "MIPS32"

        state = proj.factory.blank_state(addr=proj.entry)
        a0 = claripy.BVS("a0", 32)
        state.regs.a0 = a0

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=entry + 4 * found_idx, avoid=entry + 4 * avoid_idx, num_find=1, max_steps=500)

        assert mgr.found, f"Symbolic accumulator collapsed before reaching find — stashes={mgr.stash_counts()}"
        result = mgr.found[0].solver.eval(a0)
        assert result == 5, f"Expected a0==5 (so t0 == N*5 satisfies BEQ), got {result}"
