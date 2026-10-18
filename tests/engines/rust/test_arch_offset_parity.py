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
angr-mpln0 and angr-ihfe5 regressions.

Registers that Rust does not know cannot be probed (``set_register`` raises
``ValueError``), so the offset check covers the intersection — 90-170 names per
arch. Those names are not silently skipped: each must appear in
:data:`KNOWN_MISSING` with a stated reason, so a *missing* register name is a
hard failure rather than an invisible one (angr-keda8). Genuine offset/size
disagreements live in :data:`KNOWN_DRIFT`, each pinned to a bead. Both tables
are asserted to *still* describe reality, so fixing a bead reddens this test and
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

# (rust_arch, register) -> (rust_offset, rust_size, bead, why).
#
# Each entry is a REAL disagreement with archinfo, filed as its own bug. The
# parity test asserts the drift is still exactly this, so closing the bead
# fails this module until the entry is deleted.
KNOWN_DRIFT: dict[tuple[str, str], tuple[int, int, str, str]] = {}

# archinfo arch id -> beads for guest-state sizes smaller than the last
# archinfo register slot. Same self-cleaning contract as KNOWN_DRIFT.
KNOWN_STATE_SIZE_DRIFT: dict[str, str] = {}


def _numbered(prefix: str, stop: int, suffix: str = "", start: int = 0) -> frozenset[str]:
    """``{prefix}{i}{suffix}`` for ``i`` in ``range(start, stop)`` — register-file spellings."""
    return frozenset(f"{prefix}{i}{suffix}" for i in range(start, stop))


# rust_arch -> ((names, reason), ...): archinfo register names the Rust tables
# deliberately (or, when a bead is cited, accidentally) do not implement.
#
# The parity test may only skip a name listed here. Without this table an
# archinfo name Rust never defined just disappeared into a ``pytest.skip``,
# which is how three missing aliases reached master unnoticed — angr-9ke6b.217
# ('pc' on x86/amd64), angr-03vl4.1 ('bp' on ARM/ARM64/MIPS) and angr-03vl4.2
# ('lr' on MIPS). See angr-keda8.
#
# Same self-cleaning contract as KNOWN_DRIFT:
# :func:`test_known_missing_groups_are_still_entirely_missing` asserts every
# name in a group is *still* absent, so implementing one of them reddens this
# module and forces the group to be edited rather than left to rot.
KNOWN_MISSING: dict[str, tuple[tuple[frozenset[str], str], ...]] = {
    "amd64": (
        (
            frozenset({"_bp", "bph", "dih", "sih"}),
            "high-byte (bits 8-15) slices of rbp/rdi/rsi plus archinfo's 16-bit `_bp` "
            "spelling; the Rust table stops at the 8-bit low, 16-, 32- and 64-bit forms",
        ),
        (
            frozenset({"cr0", "cr2", "cr3", "cr4", "cr8"}),
            "privileged control registers; the interpreter models user-mode state only",
        ),
        (
            frozenset({"cs_seg", "ds_seg", "es_seg", "fs_seg", "gs_seg", "ss_seg"}),
            "segment selectors; Rust models flat addressing and reaches TLS through "
            "fs_const/gs_const, which it does implement",
        ),
        (
            frozenset({"fpu_regs", "fpu_tags"}),
            "aggregate x87 rows (the whole 64-byte register file / 8-byte tag block); "
            "archinfo exposes no per-st* names and the Rust table has no aggregate entries",
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
            "aggregate x87 rows (register file / tag block); archinfo exposes no "
            "per-st* names and the Rust table has no aggregate entries",
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
            "VEX guest-state tail past the FP file (TLS shadow, condition word, CP0 "
            "status) that the Rust MIPS table does not model",
        ),
        (frozenset({"dspcontrol"}) | _numbered("ac", 4), "MIPS DSP-ASE accumulators and control word"),
        (_numbered("f", 32, "_lo"), "low 32-bit halves of the 64-bit FP registers; Rust exposes f0-f31 whole"),
    ),
    "mips64": (
        (
            frozenset({"ulr", "cond", "cp0_status"}),
            "VEX guest-state tail past the FP file (TLS shadow, condition word, CP0 "
            "status) that the Rust MIPS table does not model",
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
