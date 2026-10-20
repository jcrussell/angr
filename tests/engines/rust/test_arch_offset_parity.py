"""Ground-truth parity between the Rust ``Arch`` register tables and archinfo.

Every other register test in this package is *same-table round-trip*::

    state.set_register("f0", V); assert state.get_register("f0") == V

Both halves resolve through the identical Rust offset table, so a consistently
wrong offset passes. That blind spot is what produced the MIPS64 FPU (+8),
ARM32 VFP (-16) and ARM64 FPCR offset errors, each caught only by hand-reading
the tables (angr-j4f1l).

This module cross-checks against a *foreign* source of truth:
``archinfo.Arch.registers`` — the ``{name: (offset, size)}`` mapping generated
from the VEX guest-state layout that pyvex actually lifts against. Comparing to
archinfo therefore subsumes the "lift a block and check the IR offset" smoke
test: the IR offsets pyvex emits *are* these numbers.

Rust exposes no ``register_offset`` binding to Python, so the offset/size pair
is recovered behaviorally (:func:`_rust_offset_size`): write all-ones through
``set_register(name, ...)`` and diff the flat guest-state dump from
``get_registers_raw()``. The span of changed bytes is exactly Rust's
``(register_offset(name), register_size(name))``.

This generalizes the two hand-written ground-truth checks in
``test_multiarch.py`` (``test_mips64_fpu_offsets_match_archinfo`` /
``test_arm32_offsets_match_archinfo``), which stay as named anchors for the
MIPS64 FPU and ARM32 offset regressions (angr-mpln0, angr-ihfe5).

Registers that Rust does not know cannot be probed (``set_register`` raises
``ValueError``), so the offset check covers the intersection — 90-170 names per
arch. Those names are not silently skipped: each must appear in
:data:`KNOWN_MISSING` with a stated reason, so a *missing* register name is a
hard failure rather than an invisible one (angr-keda8). Genuine offset/size
disagreements live in :data:`KNOWN_DRIFT`, each pinned to an issue id. Both tables
are asserted to *still* describe reality, so fixing an entry reddens this test and
forces the entry's removal — neither can silently rot into a permanent
allowlist.
"""

from __future__ import annotations

import functools

import archinfo
import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustSimState

# Rust arch name -> archinfo arch id. Shared with test_multiarch.py's
# table-driven sweeps so a new architecture is registered in exactly one place
# (angr-9ke6b.215).
from tests.engines.rust.arch_specs import ARCHES

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# (rust_arch, register) -> (rust_offset, rust_size, issue id, why).
#
# Each entry is a REAL disagreement with archinfo, tracked as its own bug. The
# parity test asserts the drift is still exactly this, so fixing the bug
# fails this module until the entry is deleted.
KNOWN_DRIFT: dict[tuple[str, str], tuple[int, int, str, str]] = {}

# archinfo arch id -> issue ids for guest-state sizes smaller than the last
# archinfo register slot. Same self-cleaning contract as KNOWN_DRIFT.
KNOWN_STATE_SIZE_DRIFT: dict[str, str] = {}


def _numbered(prefix: str, stop: int, suffix: str = "", start: int = 0) -> frozenset[str]:
    """``{prefix}{i}{suffix}`` for ``i`` in ``range(start, stop)`` — register-file spellings."""
    return frozenset(f"{prefix}{i}{suffix}" for i in range(start, stop))


# rust_arch -> ((names, reason), ...): archinfo register names the Rust tables
# deliberately (or, when an issue id is cited, accidentally) do not implement.
#
# The parity test may only skip a name listed here. Without this table an
# archinfo name Rust never defined would disappear into a ``pytest.skip``,
# letting a missing alias (such as 'pc' on x86/amd64, 'bp' on ARM/ARM64/MIPS
# or 'lr' on MIPS) go unnoticed (angr-keda8).
#
# Same self-cleaning contract as KNOWN_DRIFT:
# :func:`test_known_missing_groups_are_still_entirely_missing` asserts every
# name in a group is *still* absent, so implementing one of them reddens this
# module and forces the group to be edited rather than left to rot.
KNOWN_MISSING: dict[str, tuple[tuple[frozenset[str], str], ...]] = {
    "amd64": (
        (
            frozenset({"_bp", "bph", "dih", "sih"}),
            (
                "high-byte (bits 8-15) slices of rbp/rdi/rsi plus archinfo's 16-bit `_bp` "
                "spelling; the Rust table stops at the 8-bit low, 16-, 32- and 64-bit forms"
            ),
        ),
        (
            frozenset({"cr0", "cr2", "cr3", "cr4", "cr8"}),
            "privileged control registers; the interpreter models user-mode state only",
        ),
        (
            frozenset({"cs_seg", "ds_seg", "es_seg", "fs_seg", "gs_seg", "ss_seg"}),
            (
                "segment selectors; Rust models flat addressing and reaches TLS through "
                "fs_const/gs_const, which it does implement"
            ),
        ),
        (
            frozenset({"fpu_regs", "fpu_tags"}),
            (
                "aggregate x87 rows (the whole 64-byte register file / 8-byte tag block); "
                "archinfo exposes no per-st* names and the Rust table has no aggregate entries"
            ),
        ),
        (_numbered("mm", 8), "MMX aliases over the x87 register file (mm0 is fpu_regs+0)"),
        (_numbered("r", 16, "w", start=8), "16-bit sub-registers of r8-r15"),
        (
            frozenset(f"xmm{i}{half}" for i in range(16) for half in ("lq", "hq")),
            "64-bit halves of the xmm slots",
        ),
        (
            _numbered("ymm", 16) | _numbered("ymm", 16, "hx"),
            "AVX 256-bit views and their upper 128-bit halves; Rust's SIMD support stops at 128-bit xmm",
        ),
    ),
    "x86": (
        (
            frozenset({"dih", "dil", "sih", "sil"}),
            "8-bit slices of edi/esi; the Rust x86 table stops at the 16- and 32-bit forms",
        ),
        (
            frozenset({"fpu_regs", "fpu_tags"}),
            (
                "aggregate x87 rows (register file / tag block); archinfo exposes no "
                "per-st* names and the Rust table has no aggregate entries"
            ),
        ),
        (_numbered("mm", 8), "MMX aliases over the x87 register file (mm0 is fpu_regs+0)"),
    ),
    "arm": (
        (
            frozenset({"a1", "a2", "a3", "a4", "sb", "sl"}) | _numbered("v", 9, start=1),
            "AAPCS role aliases of r0-r3 (a1-a4) and r4-r11 (v1-v8), with sb/sl as the alternate names for r9/r10",
        ),
        (_numbered("s", 32), "VFP single-precision halves of d0-d15"),
    ),
    "arm64": (
        (frozenset({"ip0", "ip1"}), "AAPCS64 intra-procedure-call scratch aliases of x16/x17"),
        (_numbered("r", 31), "legacy r-spelling of x0-x30"),
        (frozenset({"wsp"}), "32-bit view of sp"),
        (
            _numbered("b", 32) | _numbered("h", 32) | _numbered("s", 32),
            "byte/half/single lane views of v0-v31; Rust exposes the d/q/v spellings",
        ),
    ),
    "mips32": (
        (
            frozenset({"ulr", "cond", "cp0_status"}),
            (
                "VEX guest-state tail past the FP file (TLS shadow, condition word, CP0 "
                "status) that the Rust MIPS table does not model"
            ),
        ),
        (frozenset({"dspcontrol"}) | _numbered("ac", 4), "MIPS DSP-ASE accumulators and control word"),
        (_numbered("f", 32, "_lo"), "low 32-bit halves of the 64-bit FP registers; Rust exposes f0-f31 whole"),
    ),
    "mips64": (
        (
            frozenset({"ulr", "cond", "cp0_status"}),
            (
                "VEX guest-state tail past the FP file (TLS shadow, condition word, CP0 "
                "status) that the Rust MIPS table does not model"
            ),
        ),
        (_numbered("f", 32, "_lo"), "low 32-bit halves of the 64-bit FP registers; Rust exposes f0-f31 whole"),
    ),
}


def _known_missing_reason(rust_arch: str, name: str) -> str | None:
    """Why ``rust_arch`` is expected to lack ``name``, or ``None`` if it is not expected to."""
    for names, reason in KNOWN_MISSING.get(rust_arch, ()):
        if name in names:
            return reason
    return None


# set_register truncates to Rust's width, so an all-ones write paints exactly
# min(register_size(name), _PROBE_MAX_BYTES) bytes with 0xFF.
_ALL_ONES = (1 << 128) - 1

# The probe's reach, in bytes. ``set_register`` takes a ``u128``, so the value
# it writes has at most 16 non-zero bytes no matter how wide the register is —
# a 64-byte register like amd64/x86 ``fpreg`` keeps implicit zeros above the
# payload, which a fresh state already holds, so the changed-byte span stops at
# 16. This used to read 64 because the byte-store loop shift-wrapped mod 128 and
# repeated the low 16 bytes four times (angr-0jh0j.2); the span shrinking is the
# fix landing, not the offset table drifting.
_PROBE_MAX_BYTES = 16


@functools.cache
def _rust_offset_size(arch: str, name: str) -> tuple[int, int] | None:
    """Recover Rust's ``(register_offset, register_size)`` for ``name``.

    Returns ``None`` when the Rust table has no entry for the name (its
    ``set_register`` raises) or when the write is unobservable in the flat
    dump. Probes twice — all-ones, then all-zeros — so a register that already
    holds the probe pattern in a fresh state is still located.

    Cached: the probe is a pure function of ``(arch, name)`` and all three
    tests in this module walk the same ~1000 pairs.
    """
    for value in (_ALL_ONES, 0):
        state = RustSimState(arch)
        before = bytes(state.get_registers_raw())
        try:
            state.set_register(name, value)
        except ValueError:
            return None
        after = bytes(state.get_registers_raw())
        changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
        if changed:
            return changed[0], changed[-1] - changed[0] + 1
    return None


def _parity_cases() -> list[tuple[str, str, str, int, int]]:
    """(rust_arch, archinfo_id, reg_name, archinfo_offset, archinfo_size) rows."""
    cases = []
    for rust_arch, arch_id in ARCHES.items():
        registers = archinfo.arch_from_id(arch_id).registers
        for name, (offset, size) in sorted(registers.items()):
            cases.append((rust_arch, arch_id, name, offset, size))
    return cases


@pytest.mark.parametrize(
    ("rust_arch", "arch_id", "name", "arch_offset", "arch_size"),
    _parity_cases(),
    ids=lambda v: str(v),
)
def test_register_offset_and_size_match_archinfo(rust_arch, arch_id, name, arch_offset, arch_size):
    """Rust's offset/size for a register equals archinfo's VEX guest layout."""
    probed = _rust_offset_size(rust_arch, name)
    if probed is None:
        missing_reason = _known_missing_reason(rust_arch, name)
        assert missing_reason is not None, (
            f"{rust_arch} has no register named {name!r}, and KNOWN_MISSING does not "
            "say why. A missing archinfo alias is usually a real bug (angr-9ke6b.217 "
            "'pc', angr-03vl4.1 'bp', angr-03vl4.2 'lr'): add the row to that arch's "
            "ALIASES table in native/angr/src/arch/, or — if the omission is "
            "deliberate — add a KNOWN_MISSING group here stating the reason."
        )
        pytest.skip(f"{rust_arch} has no register named {name!r} ({missing_reason})")

    drift = KNOWN_DRIFT.get((rust_arch, name))
    if drift is not None:
        expected_offset, expected_size, bead, why = drift
        assert probed == (expected_offset, expected_size), (
            f"{rust_arch}.{name}: known drift ({bead}: {why}) changed shape — "
            f"expected {(expected_offset, expected_size)}, got {probed}. "
            f"If the bead was fixed, drop this KNOWN_DRIFT entry."
        )
        assert probed != (arch_offset, arch_size), (
            f"{rust_arch}.{name} now matches archinfo — {bead} is fixed; delete its KNOWN_DRIFT entry in this module."
        )
        return

    expected = (arch_offset, min(arch_size, _PROBE_MAX_BYTES))
    assert probed == expected, (
        f"{rust_arch}.{name}: Rust table says offset/size {probed}, "
        f"archinfo {arch_id} says {(arch_offset, arch_size)} "
        f"(probe reach {expected}). "
        "The Rust register table has drifted from the VEX guest-state layout."
    )


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_known_missing_groups_are_still_entirely_missing(rust_arch, arch_id):
    """Every :data:`KNOWN_MISSING` name is still absent from the Rust table.

    The self-cleaning half of the allowlist, mirroring what :data:`KNOWN_DRIFT`
    does for offset drift: implementing one name in a group reddens this test,
    so the group has to be edited and the name handed back to the real parity
    check instead of staying permanently exempt. Also rejects a name archinfo
    itself no longer defines, which would exempt nothing at all.
    """
    registers = archinfo.arch_from_id(arch_id).registers
    for names, reason in KNOWN_MISSING.get(rust_arch, ()):
        unknown = sorted(n for n in names if n not in registers)
        assert not unknown, (
            f"{rust_arch}: KNOWN_MISSING group ({reason}) lists {unknown}, which "
            f"archinfo {arch_id} does not define — the group has rotted; drop them."
        )
        implemented = sorted(n for n in names if _rust_offset_size(rust_arch, n) is not None)
        assert not implemented, (
            f"{rust_arch}: KNOWN_MISSING group ({reason}) claims {implemented} are "
            "absent, but the Rust table implements them now. Drop them from the group "
            "so the parity check covers their offset and size."
        )


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_guest_state_size_covers_every_archinfo_register(rust_arch, arch_id):
    """Rust's guest-state buffer holds every register Rust itself exposes.

    Scoped to the *intersection*: Rust deliberately implements a subset of the
    VEX guest state (MIPS stops before archinfo's CP0 tail), so requiring the
    full archinfo extent would flag a design choice as a bug. What is always a
    bug is exposing a register name whose archinfo slot lies past the end of
    the buffer — the write lands out of bounds and reads back as zero (the
    ARM64 fpcr / ip_at_syscall case, angr-zxzi3).
    """
    registers = archinfo.arch_from_id(arch_id).registers
    needed = max(
        offset + size for name, (offset, size) in registers.items() if _rust_offset_size(rust_arch, name) is not None
    )
    actual = len(RustSimState(rust_arch).get_registers_raw())

    bead = KNOWN_STATE_SIZE_DRIFT.get(rust_arch)
    if bead is not None:
        assert actual < needed, (
            f"{rust_arch} guest state is now {actual} bytes (>= {needed}) — "
            f"{bead} is fixed; drop its KNOWN_STATE_SIZE_DRIFT entry."
        )
        return

    assert actual >= needed, (
        f"{rust_arch} guest state is {actual} bytes but archinfo {arch_id} "
        f"places registers up to byte {needed}; the tail registers are unreachable."
    )


@pytest.mark.parametrize("rust_arch", ["amd64", "x86"])
def test_register_wider_than_u128_reads_and_writes_cleanly(rust_arch):
    """``fpreg`` (64 B) survives the name-keyed accessors without corruption.

    ``fpreg`` is deliberately absent from the bulk ``register_names()`` u128
    export channel for its width, but ``set_register`` / ``get_register``
    resolve the offset table directly and reached it anyway. Reading composed
    64 bytes into a ``u128``, shifting past bit 127 — a panic under the
    ``release-checked`` profile CI runs ``cargo test`` with, 16-byte-cycle
    garbage otherwise (angr-0jh0j.1); writing did the same in reverse and
    painted the low 16 bytes four times across the register (angr-0jh0j.2).

    Contract now: the write stores the value it was given (implicit zeros above
    the ``u128`` payload, not a wrapped repeat), and the read declines with a
    clean ``ValueError`` because no ``u128`` can carry 512 bits back to Python.
    """
    state = RustSimState(rust_arch)
    offset, size = archinfo.arch_from_id(ARCHES[rust_arch]).registers["fpreg"]
    assert size > _PROBE_MAX_BYTES, "fpreg must be wider than a u128 for this test to bite"

    state.set_register("fpreg", _ALL_ONES)
    raw = bytes(state.get_registers_raw())[offset : offset + size]
    assert raw[:_PROBE_MAX_BYTES] == b"\xff" * _PROBE_MAX_BYTES
    assert raw[_PROBE_MAX_BYTES:] == bytes(size - _PROBE_MAX_BYTES), (
        "bytes above the u128 payload must stay zero, not repeat the low 16"
    )

    with pytest.raises(ValueError, match="cannot read register fpreg"):
        state.get_register("fpreg")


# ---------------------------------------------------------------------------
# The other direction, and the registers the engine treats specially.
#
# Rust's tables are the engine's source of truth (``arch_register_info``);
# these tests are what make that safe. Everything above checks archinfo ->
# Rust. The tests below check Rust -> archinfo (every Rust name is archinfo's
# or a listed Rust-only spelling), the special-purpose registers (sp / ip / bp
# / return value) and the default calling convention against angr's.
# ---------------------------------------------------------------------------


@functools.cache
def _rust_info(rust_arch: str) -> dict:
    from angr.rustylib import vex_engine

    info = vex_engine.arch_register_info(rust_arch)
    assert info is not None, f"arch_register_info({rust_arch!r}) returned None"
    return info


def _rust_offsets(rust_arch: str) -> dict[str, tuple[int, int]]:
    return {name: (offset, size) for name, offset, size, _ in _rust_info(rust_arch)["registers"]}


# rust_arch -> ((names, reason), ...): spellings the Rust tables accept that
# archinfo does not define. Self-cleaning: a name archinfo starts defining, or
# Rust stops accepting, fails test_rust_only_names_are_still_rust_only.
RUST_ONLY: dict[str, tuple[tuple[frozenset[str], str], ...]] = {
    "amd64": (
        (frozenset({"eip"}), "32-bit view of rip, kept as an alias of the instruction pointer"),
        (frozenset({"spl"}), "low byte of rsp (the 8-bit form of the x86-64 register set)"),
    ),
    "arm": (
        (_numbered("q", 16), "NEON 128-bit views over d-register pairs; archinfo's ARM32 table has no qN"),
        (frozenset({"r15t"}), "Thumb-mode spelling of pc"),
    ),
    "mips32": ((_numbered("$", 32) | _numbered("$f", 32), "numeric assembler spellings of the GPRs and FPRs"),),
    "mips64": ((_numbered("$", 32) | _numbered("$f", 32), "numeric assembler spellings of the GPRs and FPRs"),),
}


def _rust_only_reason(rust_arch: str, name: str) -> str | None:
    for names, reason in RUST_ONLY.get(rust_arch, ()):
        if name in names:
            return reason
    return None


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_every_rust_register_name_is_archinfos(rust_arch, arch_id):
    """Every name the Rust table resolves is an archinfo name at the same offset.

    The archinfo -> Rust test above cannot see a name only Rust defines. Such a
    name is harmless only if it is a deliberate extra spelling; anything else is
    a typo or a register that Python code reading ``arch_register_info`` would
    look up in archinfo and not find.
    """
    registers = archinfo.arch_from_id(arch_id).registers
    stray = sorted(
        name
        for name in _rust_offsets(rust_arch)
        if name not in registers and _rust_only_reason(rust_arch, name) is None
    )
    assert not stray, (
        f"{rust_arch}: Rust resolves {stray}, which archinfo {arch_id} does not define. "
        "Rename to archinfo's spelling, or add a RUST_ONLY group stating why it exists."
    )


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_rust_only_names_are_still_rust_only(rust_arch, arch_id):
    registers = archinfo.arch_from_id(arch_id).registers
    rust = _rust_offsets(rust_arch)
    for names, reason in RUST_ONLY.get(rust_arch, ()):
        now_archinfo = sorted(n for n in names if n in registers)
        gone = sorted(n for n in names if n not in rust)
        assert not now_archinfo, f"{rust_arch}: RUST_ONLY ({reason}) lists {now_archinfo}, which archinfo defines now"
        assert not gone, f"{rust_arch}: RUST_ONLY ({reason}) lists {gone}, which Rust no longer resolves"


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_special_registers_match_archinfo(rust_arch, arch_id):
    """sp / ip / bp / return-value registers sit at archinfo's offsets."""
    arch = archinfo.arch_from_id(arch_id)
    info = _rust_info(rust_arch)
    offsets = _rust_offsets(rust_arch)
    actual = {
        "sp": offsets[info["sp"]][0],
        "ip": offsets[info["ip"]][0],
        "bp": offsets[info["bp"]][0],
        "ret": offsets[info["cc_ret"]][0],
    }
    expected = {"sp": arch.sp_offset, "ip": arch.ip_offset, "bp": arch.bp_offset, "ret": arch.ret_offset}
    assert actual == expected


# rust_arch -> why the Rust default calling convention has no FP return
# register where angr's has one.
NO_FP_RETURN = {
    "x86": "cdecl returns floats in x87 st0, a stack-relative slot the engine does not model as a register",
}


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_calling_convention_matches_angrs(rust_arch, arch_id):
    """The default and syscall calling conventions use angr's registers.

    Compared by archinfo offset, so a different spelling of the same register
    (AArch64 ``q0`` vs angr's ``d0``) agrees. The Rust syscall window may be a
    prefix of angr's: a syscall needing more arguments than the window holds
    falls back to Python rather than reading a wrong register.
    """
    from angr.calling_conventions import DEFAULT_CC, SYSCALL_CC

    arch = archinfo.arch_from_id(arch_id)
    info = _rust_info(rust_arch)
    rust = _rust_offsets(rust_arch)

    def rust_off(names):
        return [rust[n][0] for n in names]

    def angr_off(names):
        return [arch.registers[n][0] for n in names]

    cc = DEFAULT_CC[arch.name]["Linux"]
    assert rust_off(info["cc_args"]) == angr_off(cc.ARG_REGS)
    assert rust[info["cc_ret"]][0] == arch.registers[cc.RETURN_VAL.reg_name][0]
    angr_fp_ret = getattr(cc.FP_RETURN_VAL, "reg_name", None)
    if info["cc_fp_ret"] is None:
        assert angr_fp_ret is None or rust_arch in NO_FP_RETURN, (
            f"{rust_arch}: angr returns floats in {angr_fp_ret}, Rust's CC has no FP return register"
        )
    else:
        assert rust[info["cc_fp_ret"]][0] == arch.registers[angr_fp_ret][0]

    syscall_args = rust_off(info["cc_syscall_args"])
    angr_syscall_args = angr_off(SYSCALL_CC[arch.name]["Linux"].ARG_REGS)
    assert syscall_args == angr_syscall_args[: len(syscall_args)], (
        f"{rust_arch}: syscall arg registers {info['cc_syscall_args']} are not a prefix of angr's"
    )


@pytest.mark.parametrize(("rust_arch", "arch_id"), [("mips32", "mips32"), ("mips64", "mips64")])
def test_big_endian_tables_match_archinfo(rust_arch, arch_id):
    """The one Rust table serves both byte orders; archinfo's BE layout agrees."""
    be = archinfo.arch_from_id(arch_id, "Iend_BE").registers
    le = archinfo.arch_from_id(arch_id).registers
    shared = {n for n in _rust_offsets(rust_arch) if n in be}
    assert {n: be[n] for n in shared} == {n: le[n] for n in shared}
    state = RustSimState(rust_arch, little_endian=False)
    assert len(state.get_registers_raw()) == len(RustSimState(rust_arch).get_registers_raw())


@pytest.mark.parametrize(("rust_arch", "arch_id"), sorted(ARCHES.items()))
def test_sync_registers_are_archinfo_registers(rust_arch, arch_id):
    """``Arch::sync_registers`` names archinfo registers at the engine's offsets.

    ``rust_state_sync`` reads them off the Python SimState with
    ``getattr(state.regs, name)`` and writes the bytes back at the engine's
    offset, so a name archinfo lacks, or one at a different slot, drops or
    misplaces a SimProcedure's register write.
    """
    registers = archinfo.arch_from_id(arch_id).registers
    rust = _rust_offsets(rust_arch)
    sync = _rust_info(rust_arch)["sync"]
    assert sync, f"{rust_arch}: no sync registers"
    assert {name: registers.get(name) for name in sync} == {name: rust[name] for name in sync}


def test_arch_register_info_is_none_for_an_arch_the_engine_lacks():
    from angr.rustylib import vex_engine

    assert vex_engine.arch_register_info("PPC32") is None
