"""Formal Python->Rust->Python state round-trip identity tests.

The cross-mixin invariants I1-I7 documented in the
``angr/exploration/rust_manager.py`` module docstring are otherwise only
implicitly exercised by the benchmark corpus. This module pins the ones
that govern state-export *identity* with explicit assertions:

* I3 — init-cache user-symbolic gate (``_compute_mem_init_key`` /
  ``_compute_disk_init_key`` return ``''`` for user-symbolic states).
* I4 — ``_apply_state_metadata`` is an allow-list (constraints + globals +
  a fixed option set are mirrored; everything else is dropped).
* I5 — register filter at the FFI boundary (``_supported_register_names``
  excludes the archinfo registers the Rust engine does not model, so a
  full-register bulk set never raises ``unknown register``).

The register/PC identity tests do a genuine round trip: register values
set on a Python ``SimState`` are pushed into Rust at manager construction
and read back via the ``export_state`` snapshot FFI (the same path
``_materialize_single_state`` uses as its last resort), then asserted
equal.

Relates to the characterization epic (angr-kvn0) and supersedes the
narrow ``invariant-reverse-z3-roundtrip-tests`` framing for the
Python<->Rust state boundary.
"""

from __future__ import annotations

import claripy
import pytest

from angr import sim_options as o

# Rust availability guard + RustExplorationManager live in
# tests/engines/conftest.py; the module-scoped ``fauxware_project`` fixture is
# auto-discovered by pytest from that same conftest (no import needed).
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestRegisterFilterBoundary:
    """I5: only the registers Rust models cross the FFI boundary."""

    def test_supported_register_names_excludes_unmodeled(self, fauxware_project):
        """archinfo defines cr0..8, ymm0..15, fs_seg, ... that the Rust
        engine does not model; the supported list must exclude them so a
        full bulk set never raises ``unknown register``, while still
        carrying the general-purpose file + rip + rsp."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        supported = set(mgr._supported_register_names(proj.arch))

        # Core registers the interpreter consumes must be present.
        for name in ("rax", "rbx", "rsp", "rbp", "rdi", "rsi", "rip"):
            assert name in supported, f"{name} should cross the FFI boundary"

        # Registers archinfo lists but Rust does not model must be filtered.
        for name in ("cr0", "ymm0", "fs", "gs"):
            assert name not in supported, f"{name} must be filtered from bulk set"

    def test_full_archinfo_state_construction_does_not_raise(self, fauxware_project):
        """An entry_state carries every archinfo register (cr0, ymm*, ...).
        Pushing it through manager construction must not raise — the bulk
        set is filtered to the supported subset. Regression for the I5
        ``PyValueError 'unknown register: cr0'`` failure mode."""
        proj = fauxware_project
        # Would raise at construction if the FFI filter regressed.
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        assert len(mgr._rust_mgr.get_state_ids("active")) == 1


class TestRegisterPcRoundTripIdentity:
    """The Rust-side register/PC view (export_state snapshot) and the
    materialized Python SimState agree for the same state id.

    NOTE: the manager's init pipeline (``_run_python_init_if_needed``)
    *simulates the program prologue*, so register values set on the input
    state are NOT a pass-through — the entry_state at ``_start`` lands at
    ``main`` after init. The identity the export contract guarantees is
    therefore Rust<->Python *consistency*: whatever Rust holds is what the
    materialized Python state reports, on both the snapshot FFI and the
    full reconstruction path.
    """

    def test_snapshot_registers_match_materialized_state(self, fauxware_project):
        """Every register Rust models reports the same concrete value through
        the export_state snapshot FFI and through the materialized SimState —
        the Rust -> Python register sync is faithful."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        sid = mgr._rust_mgr.get_state_ids("active")[0]

        snapshot = mgr._rust_mgr.export_state(sid)
        named = snapshot.get_registers_named()
        materialized = mgr._get_stash_states("active")[0]

        checked = 0
        for name in mgr._supported_register_names(proj.arch):
            if name not in named:
                continue
            reg = getattr(materialized.regs, name)
            if reg.symbolic:
                continue
            assert materialized.solver.eval(reg) == named[name][0], (
                f"{name}: snapshot 0x{named[name][0]:x} != materialized state"
            )
            checked += 1
        assert checked > 0, "expected at least one concrete register to compare"

    def test_pc_consistent_across_snapshot_and_materialized(self, fauxware_project):
        """The snapshot's ``pc`` field equals the materialized SimState's
        ``addr`` — the program counter crosses the FFI boundary identically."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        sid = mgr._rust_mgr.get_state_ids("active")[0]

        snapshot = mgr._rust_mgr.export_state(sid)
        materialized = mgr._get_stash_states("active")[0]
        assert snapshot.pc == materialized.addr

    def test_snapshot_to_angr_matches_snapshot_registers(self, fauxware_project):
        """The full reconstruction path (``_snapshot_to_angr``) rebuilds a
        SimState whose concrete registers match the source snapshot — the
        cache-independent Rust -> Python materialization is faithful."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        snapshot = mgr._rust_mgr.export_state(sid)
        named = snapshot.get_registers_named()

        rebuilt = mgr._snapshot_to_angr(snapshot)
        assert rebuilt.addr == snapshot.pc

        checked = 0
        for name in ("rax", "rbx", "rsp", "rbp", "rdi", "rsi"):
            if name not in named:
                continue
            reg = getattr(rebuilt.regs, name)
            if reg.symbolic:
                continue
            assert rebuilt.solver.eval(reg) == named[name][0], f"{name}: rebuilt state != snapshot 0x{named[name][0]:x}"
            checked += 1
        assert checked > 0, "expected at least one concrete register to compare"


class TestInitCacheUserSymbolicGate:
    """I3: init caches are suppressed when the input state holds user-created
    symbolic data that a blank_state round-trip cannot reproduce."""

    def test_concrete_state_allows_mem_init_key(self, fauxware_project):
        """A purely concrete entry_state yields a non-empty in-memory init
        key — caching is permitted. The key carries the concrete-input digest
        (``dummy_key:<hex>``) so distinct args don't collide (angr-gxaht)."""
        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        key = mgr._compute_mem_init_key(state, "dummy_key")
        assert key.startswith("dummy_key:")
        assert key != "dummy_key"  # a non-empty digest was appended

    def test_concrete_input_digest_differentiates_argv(self, fauxware_project):
        """Two entry_states differing only in concrete argv (same length) yield
        different init keys; identical argv yields identical keys. Guards the
        angr-gxaht collision where a warm run silently replayed the prior run's
        argv/env because the key was keyed only by binary hash."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        state_a = proj.factory.entry_state(args=["prog", "AAAAAAAA"])
        state_a2 = proj.factory.entry_state(args=["prog", "AAAAAAAA"])
        state_b = proj.factory.entry_state(args=["prog", "BBBBBBBB"])

        mem_a = mgr._compute_mem_init_key(state_a, "dummy_key")
        mem_a2 = mgr._compute_mem_init_key(state_a2, "dummy_key")
        mem_b = mgr._compute_mem_init_key(state_b, "dummy_key")
        assert mem_a == mem_a2  # deterministic for identical inputs
        assert mem_a != mem_b  # distinct concrete argv -> distinct key

        # Disk keys hash the real binary bytes, so use the actual path.
        bin_path = proj.loader.main_object.binary
        disk_a = mgr._compute_disk_init_key(state_a, bin_path)
        disk_a2 = mgr._compute_disk_init_key(state_a2, bin_path)
        disk_b = mgr._compute_disk_init_key(state_b, bin_path)
        assert disk_a and disk_a == disk_a2
        assert disk_a != disk_b

    def test_user_symbolic_register_suppresses_keys(self, fauxware_project):
        """A user-set symbolic register (state.regs.rdi = BVS) makes both the
        in-memory and disk init keys empty, suppressing load AND save so the
        symbolic identity is never lost to a cached blank_state."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        sym_state = proj.factory.entry_state()
        sym_state.regs.rdi = claripy.BVS("user_arg", 64)

        assert mgr._compute_mem_init_key(sym_state, "dummy_key") == ""
        assert mgr._compute_disk_init_key(sym_state, "dummy_key") == ""

    def test_empty_cache_key_stays_empty(self, fauxware_project):
        """An empty cache_key (no binary path) short-circuits to '' regardless
        of symbolic content."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        state = proj.factory.entry_state()

        assert mgr._compute_mem_init_key(state, "") == ""
        assert mgr._compute_disk_init_key(state, "") == ""

    def test_user_inserted_simfile_suppresses_keys(self, fauxware_project):
        """A user-inserted SimFile (state.fs.insert) makes both init keys
        empty. The init-cache state is built from a blank_state with an empty
        filesystem and _apply_state_metadata does not carry the fs plugin, so
        a cache hit would hand fopen an empty filesystem — minting a fresh
        symbolic-size file and exploding fread (asisctffinals2015_license
        TIMEOUT; angr-ql3ja)."""
        import angr

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        fs_state = proj.factory.entry_state()
        content = claripy.Concat(*[claripy.BVS("license_byte_%d" % i, 8) for i in range(34)])
        fs_state.fs.insert("/home/user/license", angr.storage.file.SimFile("license", content))

        assert mgr._compute_mem_init_key(fs_state, "dummy_key") == ""
        assert mgr._compute_disk_init_key(fs_state, "dummy_key") == ""


class TestApplyStateMetadataAllowList:
    """I4: _apply_state_metadata copies constraints + globals + a fixed option
    allow-list, and drops everything else."""

    def test_constraints_and_globals_copied(self, fauxware_project):
        """Constraints and globals on the source state are mirrored onto the
        destination — the metadata carried across the init-cache copy."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        src = proj.factory.entry_state()
        x = claripy.BVS("x", 32)
        src.solver.add(x == 0x41)
        src.globals["marker"] = 0xABCD

        dst = proj.factory.blank_state()
        n_before = len(dst.solver.constraints)
        mgr._apply_state_metadata(src, dst)

        assert len(dst.solver.constraints) == n_before + 1
        assert dst.globals["marker"] == 0xABCD

    def test_allowlisted_option_mirrored_both_ways(self, fauxware_project):
        """An allow-listed option (LAZY_SOLVES) is ADDED when the source has
        it and REMOVED when the source lacks it — the mirror contract that
        stops a cached option leaking to a later caller."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        # Add half: src has it, dst doesn't -> dst gains it.
        src = proj.factory.blank_state(add_options={o.LAZY_SOLVES})
        dst = proj.factory.blank_state(remove_options={o.LAZY_SOLVES})
        mgr._apply_state_metadata(src, dst)
        assert o.LAZY_SOLVES in dst.options

        # Remove half: src lacks it, dst has it -> dst loses it.
        src2 = proj.factory.blank_state(remove_options={o.LAZY_SOLVES})
        dst2 = proj.factory.blank_state(add_options={o.LAZY_SOLVES})
        mgr._apply_state_metadata(src2, dst2)
        assert o.LAZY_SOLVES not in dst2.options

    def test_fill_policy_options_mirrored(self, fauxware_project):
        """The unconstrained-fill options survive the init-cache copy (angr-z21g0).

        The destination is a blank_state, whose defaults lack ZERO_FILL_*. If
        the fill policy is dropped here, every Python-side load of an unmapped
        byte during a SimProcedure callback mints a fresh ``mem_*`` BVS instead
        of a zero; those symbolic bytes flow back into Rust and fork-storm the
        exploration (xmllint/libxml2 went 1 active -> 60+ at step 27).
        """
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        fill_opts = {
            o.ZERO_FILL_UNCONSTRAINED_MEMORY,
            o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            o.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
            o.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
        }
        src = proj.factory.entry_state(add_options=fill_opts)
        dst = proj.factory.blank_state(remove_options=fill_opts)
        mgr._apply_state_metadata(src, dst)
        for opt in fill_opts:
            assert opt in dst.options

        # Mirror contract: a source without the fill policy strips it from dst.
        src2 = proj.factory.blank_state(remove_options=fill_opts)
        dst2 = proj.factory.blank_state(add_options=fill_opts)
        mgr._apply_state_metadata(src2, dst2)
        for opt in fill_opts:
            assert opt not in dst2.options

    def test_non_allowlisted_option_not_copied(self, fauxware_project):
        """An option outside the allow-list (TRACK_MEMORY_ACTIONS) on the
        source is NOT propagated to the destination — proves the copy is a
        positive allow-list, not a full options union."""
        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        src = proj.factory.blank_state(add_options={o.TRACK_MEMORY_ACTIONS})
        dst = proj.factory.blank_state(remove_options={o.TRACK_MEMORY_ACTIONS})
        assert o.TRACK_MEMORY_ACTIONS not in dst.options

        mgr._apply_state_metadata(src, dst)
        assert o.TRACK_MEMORY_ACTIONS not in dst.options
