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

Registers that Rust does not know are skipped (``set_register`` raises
``ValueError``); the gate covers the intersection, which is 90-170 names per
arch. Genuine disagreements live in :data:`KNOWN_DRIFT`, each pinned to a bead.
That table is asserted to *still* mismatch, so fixing a bead reddens this test
and forces the entry's removal — the table cannot silently rot into a
permanent allowlist.
"""

from __future__ import annotations

import archinfo
import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustSimState

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# Rust arch name -> archinfo arch id.
ARCHES = {
    "amd64": "AMD64",
    "x86": "X86",
    "arm": "ARMEL",
    "arm64": "AArch64",
    "mips32": "MIPS32",
    "mips64": "MIPS64",
}

# (rust_arch, register) -> (rust_offset, rust_size, bead, why).
#
# Each entry is a REAL disagreement with archinfo, filed as its own bug. The
# parity test asserts the drift is still exactly this, so closing the bead
# fails this module until the entry is deleted.
KNOWN_DRIFT: dict[tuple[str, str], tuple[int, int, str, str]] = {}

# archinfo arch id -> beads for guest-state sizes smaller than the last
# archinfo register slot. Same self-cleaning contract as KNOWN_DRIFT.
KNOWN_STATE_SIZE_DRIFT: dict[str, str] = {}

# Wider than any register we probe; set_register truncates to Rust's width, so
# an all-ones write paints exactly register_size(name) bytes with 0xFF.
_ALL_ONES = (1 << 128) - 1


def _rust_offset_size(arch: str, name: str) -> tuple[int, int] | None:
    """Recover Rust's ``(register_offset, register_size)`` for ``name``.

    Returns ``None`` when the Rust table has no entry for the name (its
    ``set_register`` raises) or when the write is unobservable in the flat
    dump. Probes twice — all-ones, then all-zeros — so a register that already
    holds the probe pattern in a fresh state is still located.
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
        pytest.skip(f"{rust_arch} has no register named {name!r}")

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

    assert probed == (arch_offset, arch_size), (
        f"{rust_arch}.{name}: Rust table says offset/size {probed}, "
        f"archinfo {arch_id} says {(arch_offset, arch_size)}. "
        "The Rust register table has drifted from the VEX guest-state layout."
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
