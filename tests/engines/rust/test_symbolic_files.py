# pylint: disable=missing-class-docstring,no-self-use
"""End-to-end tests for bounded symbolic file content (angr-0xyq2 Phase 3).

Phase 1 gave the Rust ``FileSystem`` a path-keyed symbolic-content registry,
Phase 2 made native ``open()``/``read()``/``fread()`` attach and serve it, and
Phase 3 (under test here) exports eligible ``state.fs._files`` entries into
that registry at ``_add_rust_state`` time
(``RustExplorationManager._export_fs_files_to_rust`` →
``set_fs_cwd`` / ``register_file_content`` on the PyO3 manager).

Test vehicle: fauxware's ``authenticate(username, password)``::

    if (strcmp(password, sneaky) == 0) return 1;   // backdoor
    pwfile = open(username, O_RDONLY);
    read(pwfile, stored_pw, 8);
    if (strcmp(password, stored_pw) == 0) return 1; // 0x4006df
    return 0;                                       // 0x4006e6

A ``call_state`` with a CONCRETE username (``"pwfile"``) and password
(``"MYSECRET"``) makes the whole function body run natively: the backdoor
strcmp is concretely false, ``open("pwfile")`` resolves the exported registry
entry, ``read`` serves the symbolic bytes natively, and the second strcmp is
the lone (native) symbolic branch. Reaching the return-1 block therefore
constrains the file bytes to ``b"MYSECRET"`` purely through native branching —
no Python fallback ever fires, which the counter assertions pin down.

The found-state assertions evaluate on the Rust solver: found-state
materialization attaches ``RustSolverFallback`` (``rust_state_export.py``), so
both ``found.solver.eval(content_bvs)`` and ``SimFile.concretize()`` (which
evaluates through ``found.solver``) see the natively-added constraints. The
claripy bridge preserves BVS identity by hash and name+width on import, so the
original Python ASTs are the same symbols the Rust constraints mention.
"""

from __future__ import annotations

import claripy
import pytest

from angr.storage.file import SimFile

# Availability guard + module-scoped fauxware_project fixture live in
# tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# fauxware (amd64) authenticate() landmarks — see module docstring.
AUTH_ADDR = 0x400664
# "return 1" block reached only when strcmp(password, stored_pw) == 0.
FILE_MATCH_RET1 = 0x4006DF
# "return 0" block (file password mismatch).
MISMATCH_RET0 = 0x4006E6

PASSWORD = b"MYSECRET"  # 8 bytes, != "SOSNEAKY" so the backdoor is dead


def _auth_state(project, simfile):
    """``call_state`` at authenticate() with concrete args and ``simfile``
    inserted at the (cwd-relative) path the guest will open()."""
    state = project.factory.call_state(AUTH_ADDR, b"pwfile\x00", PASSWORD + b"\x00")
    state.fs.insert("pwfile", simfile)
    return state


def _explore_match(project, simfile):
    """Explore to the file-password-match return-1 block."""
    mgr = RustExplorationManager(project, [_auth_state(project, simfile)])
    # symfile_* counters are process-wide atomics; reset for a per-run delta.
    mgr.reset_solver_stats()
    mgr.explore(find=FILE_MATCH_RET1, avoid=MISMATCH_RET0, num_find=1)
    return mgr


def _assert_no_read_fallbacks(stats):
    """The whole run stayed native: no read/fread Python bounce anywhere.

    Counter names from ``stats_api.rs`` (native-proc per-reason breakdown) and
    the manager-level simprocedure/syscall fallback tallies.
    """
    for by_name_key in (
        "native_proc_symbolic_fallbacks_by_name",
        "native_proc_not_implemented_fallbacks_by_name",
        "native_proc_other_fallbacks_by_name",
        "simprocedure_fallback_by_name",
    ):
        by_name = stats[by_name_key]
        assert "read" not in by_name and "fread" not in by_name, f"{by_name_key} shows a read/fread bounce: {by_name}"
    # Observed-zero totals for this fully-native run — any nonzero value means
    # some proc/syscall bounced to Python and the native serve regressed.
    assert stats["native_proc_fallbacks"] == 0, stats["native_proc_other_fallbacks_by_name"]
    assert stats["simprocedure_python_fallback_count"] == 0, stats["simprocedure_fallback_by_name"]
    assert stats["syscall_python_fallback_count"] == 0, stats["syscall_python_fallback_by_num"]


class TestSymbolicFileNativeServe:
    def test_symbolic_content_constrained_by_native_branch(self, fauxware_project):
        """Registered symbolic content is served natively and the find-path
        constraint (added purely by the native strcmp branch) pins it."""
        content = claripy.BVS("symfile_content", 8 * 8)
        simfile = SimFile("pwfile", content=content, size=8, has_end=True)

        mgr = _explore_match(fauxware_project, simfile)
        assert len(mgr.found) == 1, "file-match path not found"
        found = mgr.found[0]

        # Solved via the ORIGINAL claripy BVS: bridge import preserved its
        # identity, so the natively-added strcmp constraints bind it.
        assert found.solver.eval(content, cast_to=bytes) == PASSWORD

        # And via the found state's own fs plugin (concretize() evaluates
        # through found.solver → RustSolverFallback → the Rust solver).
        found_file = found.fs.get("pwfile")
        assert found_file is not None
        assert found_file.concretize() == PASSWORD

        stats = mgr.stats
        assert stats["symfile_reads_native"] >= 1, "read was not served from the Rust registry"
        assert stats["symfile_write_demotions"] == 0
        # angr-4ref8: an eligible file was handed to the native registry and no
        # scope-gate reason fired for it.
        assert stats["symfile_exports"] >= 1
        assert stats["symfile_export_skip_endness"] == 0
        _assert_no_read_fallbacks(stats)

    def test_no_content_simfile_fresh_symbolic_bytes(self, fauxware_project):
        """The no-content ``SimFile(name, size=N)`` form (fresh symbolic bytes
        minted by the export's load) works through the same native flow."""
        simfile = SimFile("pwfile", size=8, has_end=True)

        mgr = _explore_match(fauxware_project, simfile)
        assert len(mgr.found) == 1, "file-match path not found"
        found = mgr.found[0]

        found_file = found.fs.get("pwfile")
        assert found_file is not None
        assert found_file.concretize() == PASSWORD

        stats = mgr.stats
        assert stats["symfile_reads_native"] >= 1
        _assert_no_read_fallbacks(stats)

    @pytest.mark.parametrize(
        ("ineligible_kwargs", "skip_reason"),
        [
            pytest.param({"has_end": False}, "has_end", id="unbounded"),
            pytest.param({"has_end": True, "file_exists": False}, "file_exists", id="nonexistent"),
            pytest.param({"has_end": True, "endness": "Iend_LE"}, "endness", id="little_endian"),
        ],
    )
    def test_ineligible_file_skipped_still_explores_via_fallback(
        self, fauxware_project, ineligible_kwargs, skip_reason
    ):
        """A file failing the v1 scope gate (unbounded ``has_end=False``,
        ``file_exists=False``, or a little-endian load model) is skipped:
        nothing is registered (``symfile_reads_native == 0``), the guest read
        bounces to Python (fallback counters > 0), and exploration still
        completes — the mismatch/return-0 path, i.e. exactly the pre-Phase-3
        behavior for these shapes (the natively-minted fd is not mirrored into
        Python, angr-8j16, so the Python file model cannot pin the content)."""
        content = claripy.BVS("symfile_ineligible", 8 * 8)
        simfile = SimFile("pwfile", content=content, size=8, **ineligible_kwargs)

        state = _auth_state(fauxware_project, simfile)
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.reset_solver_stats()
        mgr.explore(find=MISMATCH_RET0, num_find=1)

        assert len(mgr.found) == 1, "exploration should still complete via the Python fallback"
        assert mgr.found[0].addr == MISMATCH_RET0

        stats = mgr.stats
        assert stats["symfile_reads_native"] == 0, "ineligible file must not be served from the registry"
        assert stats["simprocedure_python_fallback_count"] > 0, "expected the guest read to bounce to Python"
        assert stats["simprocedure_fallback_by_name"].get("read", 0) > 0, stats["simprocedure_fallback_by_name"]
        # angr-4ref8: the scope-gate rejection is attributed to its reason
        # counter, and nothing was exported for this state.
        assert stats[f"symfile_export_skip_{skip_reason}"] >= 1, f"expected symfile_export_skip_{skip_reason} to fire"
        assert stats["symfile_exports"] == 0

    def test_parallel_workers_content_survives_migration(self, fauxware_project, monkeypatch):
        """workers=2: registered content survives cross-worker state migration
        (serde snapshot rebuilds symbolic leaves by name+width) and the found
        state still solves to the pinned password."""
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", "2")

        content = claripy.BVS("symfile_parallel", 8 * 8)
        simfile = SimFile("pwfile", content=content, size=8, has_end=True)

        mgr = RustExplorationManager(fauxware_project, [_auth_state(fauxware_project, simfile)])
        assert mgr.stats["parallel_real_workers"] == 2
        mgr.reset_solver_stats()
        mgr.explore(find=FILE_MATCH_RET1, avoid=MISMATCH_RET0, num_find=1)

        assert len(mgr.found) == 1
        found = mgr.found[0]
        assert found.solver.eval(content, cast_to=bytes) == PASSWORD
        assert found.fs.get("pwfile").concretize() == PASSWORD
        assert mgr.stats["symfile_reads_native"] >= 1


class TestFsExportApiContract:
    """Error contract of the raw PyO3 surface (set_fs_cwd /
    register_file_content) — errors, never panics."""

    def test_set_fs_cwd_unknown_state_errors(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match="not found"):
            mgr._rust_mgr.set_fs_cwd(10**9, "/nowhere")

    def test_register_file_content_unknown_state_errors(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match="not found"):
            mgr._rust_mgr.register_file_content(10**9, "/tmp/f", [claripy.BVV(0x41, 8)])

    def test_register_file_content_rejects_non_byte_width(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        state_ids = mgr._rust_mgr.get_state_ids("active")
        assert state_ids, "seed state should be in the active stash"
        with pytest.raises(ValueError, match="width"):
            mgr._rust_mgr.register_file_content(state_ids[0], "/tmp/f", [claripy.BVV(0x4142, 16)])
