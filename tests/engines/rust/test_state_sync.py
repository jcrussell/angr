"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import claripy
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


class TestMmapBaseSync:
    """Tests that the Rust per-state mmap_base mirrors back to Python's
    state.heap.mmap_base on stash export.

    Regression for angr-0cnm: previously the native mmap syscall handler
    bumped Rust's mmap_base on addr=0 calls, but nothing pushed that bump
    back to the angr SimState — so a subsequent Python-side fallback
    allocation would overlap a Rust-allocated region.
    """

    def test_get_state_mmap_base_default(self):
        """The Rust manager's mmap_base getter returns the documented default
        (heap_base 0xC0000000 + heap_size 0x00800000 * 2 = 0xC1000000)."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        assert mgr.get_state_mmap_base(sid) == 0xC100_0000

    def test_set_state_mmap_base_round_trips(self):
        """Setter advances the value and getter reads it back — proves the
        FFI accessor pair is wired to the same RustSimState field that the
        native mmap syscall handler bumps."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        mgr.set_state_mmap_base(sid, 0xC100_5000)
        assert mgr.get_state_mmap_base(sid) == 0xC100_5000

    def test_get_state_mmap_base_unknown_state_raises(self):
        """Unknown state IDs surface a ValueError (matches the timeout API)."""
        mgr = _RustExplorationManager("amd64")
        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr.get_state_mmap_base(999_999)

    def test_export_path_syncs_rust_mmap_base_into_state_heap(self, fauxware_project):
        """End-to-end: a Rust-side mmap_base advance is visible on the angr
        SimState returned by mgr.active.

        Pre-fix this fails — state.heap.mmap_base stays at the default
        0xC1000000 even though Rust bumped its internal counter, leading to
        the silent-corruption scenario in the bead description.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        # Sanity: the Python default matches the Rust default so the test
        # detects only sync changes, not a base-address mismatch.
        assert state.heap.mmap_base == 0xC100_0000

        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        assert len(active_ids) == 1
        sid = active_ids[0]

        # Simulate what NativeMmapSyscall does on a successful addr=0 mmap:
        # bump the per-state mmap_base by one page.
        bumped = 0xC100_1000
        mgr._rust_mgr.set_state_mmap_base(sid, bumped)
        assert mgr._rust_mgr.get_state_mmap_base(sid) == bumped

        # Pull the state back via the public stash API. _get_stash_states
        # is the path mgr.active / mgr.found go through.
        states = mgr._get_stash_states("active")
        assert len(states) == 1
        synced = states[0]

        assert synced.heap.mmap_base == bumped, (
            f"state.heap.mmap_base = 0x{synced.heap.mmap_base:x} but Rust's "
            f"mmap_base advanced to 0x{bumped:x} — sync did not run on stash "
            f"export and a Python-side mmap fallback would now overlap a "
            f"Rust-allocated region."
        )

    def test_export_path_does_not_clobber_higher_python_mmap_base(self, fauxware_project):
        """The sync takes max(rust, python) — a Python-side advance that
        outpaced Rust must not be reverted.

        Scenario: Python-side SimProcedure bumped state.heap.mmap_base; Rust's
        per-state field was not yet updated (drift in the opposite direction).
        On stash export we must keep the Python value, not overwrite it with
        the smaller Rust value.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        sid = active_ids[0]

        # Python advanced its mmap_base; Rust still at default.
        cached = mgr._state_cache[sid]
        cached.heap.mmap_base = 0xC100_8000
        assert mgr._rust_mgr.get_state_mmap_base(sid) == 0xC100_0000

        states = mgr._get_stash_states("active")
        assert len(states) == 1
        # Python's higher value wins — not clobbered by Rust's smaller value.
        assert states[0].heap.mmap_base == 0xC100_8000


class TestMmapMapFixedNative:
    """End-to-end: mmap(addr, len, prot, MAP_FIXED|..., -1, 0) on a
    range that collides with an existing mapping succeeds natively,
    discarding the colliding pages and remapping at the requested
    address. Matches Linux mmap(2) semantics and diverges from the
    Python posix mmap procedure (which returns -1 on collision).

    Regression for angr-ttr7: previously the native fast path bailed
    to Python on any collision, including the MAP_FIXED case where
    Python would just return -1.
    """

    MAP_PRIVATE = 0x02
    MAP_FIXED = 0x10
    MAP_ANONYMOUS = 0x20

    def _build_syscall_state(self, target_addr, length, prot, flags):
        """A blank amd64 state at a `syscall` instruction with rax=9 (mmap)
        and the standard amd64 syscall ABI registers set for an mmap call."""

        shellcode = b"\x0f\x05" + b"\x90" * 0x100
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = 9  # mmap
        state.regs.rdi = target_addr  # addr
        state.regs.rsi = length  # length
        state.regs.rdx = prot  # prot
        state.regs.r10 = flags  # flags
        state.regs.r8 = 0xFFFFFFFF_FFFFFFFF  # fd = -1
        state.regs.r9 = 0  # offset
        return proj, state

    def test_map_fixed_collision_no_python_fallback(self):
        """MAP_FIXED + collision: the syscall runs through the native
        fast path. The load-bearing assertion is that
        ``syscall_python_fallback_count`` stays at zero — before the
        fix, every MAP_FIXED collision routed back to Python.

        The Rust unit tests in ``syscalls/mmap.rs`` cover the unmap +
        remap behavior comprehensively; this Python-level test exists
        to lock in the integration contract (no fallback)."""

        target = 0x4000_0000
        length = 0x1000
        flags = self.MAP_FIXED | self.MAP_PRIVATE | self.MAP_ANONYMOUS

        proj, state = self._build_syscall_state(target, length, 0x5, flags)
        mgr = RustExplorationManager(proj, [state])

        # Pre-seed the colliding page in all active Rust states with RW.
        # A flat "any collision → fall back" path would bump the
        # fallback counter on this; the native unmap+remap path
        # absorbs it.
        mgr._rust_mgr.active_states_map_memory(target, b"\x00" * length, 0x3)

        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "MAP_FIXED collision must take the native fast path; got "
            f"{stats['syscall_python_fallback_count']} Python fallback(s)."
        )

    def test_map_fixed_clean_addr_no_python_fallback(self):
        """MAP_FIXED on a non-colliding addr also takes the native
        path (regression guard for the is_fixed shortcut)."""

        target = 0x4000_0000
        length = 0x1000
        flags = self.MAP_FIXED | self.MAP_PRIVATE | self.MAP_ANONYMOUS

        proj, state = self._build_syscall_state(target, length, 0x3, flags)
        mgr = RustExplorationManager(proj, [state])

        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "MAP_FIXED without collision must take the native path; got "
            f"{stats['syscall_python_fallback_count']} Python fallback(s)."
        )


class TestPosixBrkSync:
    """Tests that the Rust per-state posix_brk mirrors back to Python's
    state.posix.brk on stash export.

    Regression for angr-as3c (mirrors angr-0cnm for mmap_base): NativeBrkSyscall
    bumps Rust's posix_brk on a concrete grow, but nothing pushed that bump
    back to the angr SimState — so a Python-side fallback (symbolic brk arg
    or set_brk collision retry) would read a stale state.posix.brk and hand
    out heap addresses overlapping a Rust-allocated region.
    """

    def test_get_state_posix_brk_default(self):
        """Default posix_brk matches Python's posix.brk default (0x1B00000)."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        assert mgr.get_state_posix_brk(sid) == 0x1B0_0000

    def test_set_state_posix_brk_round_trips(self):
        """Setter advances the value and getter reads it back — proves the
        FFI accessor pair is wired to the same RustSimState field that
        NativeBrkSyscall mutates."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        mgr.set_state_posix_brk(sid, 0x1B0_5000)
        assert mgr.get_state_posix_brk(sid) == 0x1B0_5000

    def test_get_state_posix_brk_unknown_state_raises(self):
        """Unknown state IDs surface a ValueError (matches the mmap_base API)."""
        mgr = _RustExplorationManager("amd64")
        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr.get_state_posix_brk(999_999)

    def test_init_push_aligns_rust_posix_brk_with_python(self, fauxware_project):
        """At state creation, the angr loader sets state.posix.brk to a value
        derived from the binary's last address (e.g. 0x602000 for fauxware) —
        distinct from Rust's hardcoded default 0x1B00000. _add_rust_state
        pushes Python's brk into Rust so subsequent NativeBrkSyscall calls
        compare against the correct base.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        py_brk = state.posix.brk
        assert isinstance(py_brk, int)
        # fauxware sits below 0x1B00000, so Rust's default would otherwise
        # over-shoot Python's actual brk and break any sync semantics.
        assert py_brk < 0x1B0_0000

        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        assert mgr._rust_mgr.get_state_posix_brk(sid) == py_brk

    def test_export_path_syncs_rust_posix_brk_into_state_posix(self, fauxware_project):
        """End-to-end: a Rust-side posix_brk advance is visible on the angr
        SimState returned by mgr.active.

        Pre-fix this fails — state.posix.brk stays at the loader-set value
        even though Rust bumped its internal counter, leading to the silent
        heap-collision scenario in the bead description.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        starting = state.posix.brk
        assert isinstance(starting, int)

        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        assert len(active_ids) == 1
        sid = active_ids[0]
        # Init push aligned the two sides at the loader-set base.
        assert mgr._rust_mgr.get_state_posix_brk(sid) == starting

        # Simulate what NativeBrkSyscall does on a concrete brk(addr) grow:
        # bump the per-state posix_brk by one page past the loader base.
        bumped = starting + 0x1000
        mgr._rust_mgr.set_state_posix_brk(sid, bumped)
        assert mgr._rust_mgr.get_state_posix_brk(sid) == bumped

        # Pull the state back via the public stash API. _get_stash_states
        # is the path mgr.active / mgr.found go through.
        states = mgr._get_stash_states("active")
        assert len(states) == 1
        synced = states[0]

        assert synced.posix.brk == bumped, (
            f"state.posix.brk = {synced.posix.brk!r} but Rust's posix_brk "
            f"advanced to 0x{bumped:x} — sync did not run on stash export "
            f"and a Python-side brk fallback would now overlap a "
            f"Rust-allocated region."
        )

    def test_export_path_does_not_clobber_higher_python_posix_brk(self, fauxware_project):
        """The sync takes max(rust, python) — a Python-side advance that
        outpaced Rust must not be reverted.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        starting = state.posix.brk

        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        sid = active_ids[0]

        # Python advanced its posix.brk; Rust still at the loader-set base.
        cached = mgr._state_cache[sid]
        higher = starting + 0x6000
        cached.posix.brk = higher
        assert mgr._rust_mgr.get_state_posix_brk(sid) == starting

        states = mgr._get_stash_states("active")
        assert len(states) == 1
        # Python's higher value wins — not clobbered by Rust's smaller value.
        assert states[0].posix.brk == higher

    def test_export_path_leaves_symbolic_python_posix_brk_alone(self, fauxware_project):
        """If Python's set_brk has rewritten state.posix.brk as a claripy BV
        (concrete BVV after a concrete grow, or symbolic If(...) after a
        symbolic grow), the sync must not replace it with a raw int — that
        would break downstream Python code that expects a BV.
        """
        import claripy

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        sid = active_ids[0]

        # Python's set_brk would have left a BVV here. Rust's int posix_brk
        # is bigger but we still must not overwrite a BV with a raw int.
        cached = mgr._state_cache[sid]
        cached.posix.brk = claripy.BVV(state.posix.brk + 0x2000, proj.arch.bits)
        mgr._rust_mgr.set_state_posix_brk(sid, state.posix.brk + 0x5000)

        states = mgr._get_stash_states("active")
        assert len(states) == 1
        # BV is preserved — sync skipped because posix.brk is not an int.
        assert isinstance(states[0].posix.brk, claripy.ast.BV)
        assert states[0].posix.brk is cached.posix.brk


class TestHeapBrkSync:
    """Tests that the Rust per-state heap_brk (malloc bump allocator) mirrors
    back to Python's state.heap.heap_location on stash export.

    Regression for angr-um39j (mirrors TestPosixBrkSync for the brk syscall and
    TestMmapBaseSync for mmap): native heap-allocating SimProcedures
    (malloc/calloc/realloc/strdup/fopen) bump Rust's heap_brk via heap_alloc,
    but nothing pushed that bump back to the angr SimState — so a later
    Python-side fallback SimProcedure would read a stale state.heap.heap_location
    and hand out heap addresses overlapping a Rust-allocated region. Latent
    until angr-blq01 made native fopen/calloc/realloc actually run.
    """

    def test_get_state_heap_brk_default(self):
        """Default heap_brk matches Python's heap.heap_location default
        (0xC0000000)."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        assert mgr.get_state_heap_brk(sid) == 0xC000_0000

    def test_set_state_heap_brk_round_trips(self):
        """Setter advances the value and getter reads it back — proves the
        FFI accessor pair is wired to the same RustSimState field that
        heap_alloc mutates."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        mgr.set_state_heap_brk(sid, 0xC000_5000)
        assert mgr.get_state_heap_brk(sid) == 0xC000_5000

    def test_get_state_heap_brk_unknown_state_raises(self):
        """Unknown state IDs surface a ValueError (matches the posix_brk API)."""
        mgr = _RustExplorationManager("amd64")
        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr.get_state_heap_brk(999_999)

    def test_init_push_aligns_rust_heap_brk_with_python(self, fauxware_project):
        """At state creation, _add_rust_state pushes Python's
        state.heap.heap_location into Rust so subsequent native allocations
        don't collide with a Python-side allocation. Default matches, so
        advance Python first to prove the push runs.

        Uses a non-entry blank_state: an entry-point state runs Python
        init-to-main, which produces a fresh state and discards a bump on the
        seed. A blank_state inside the main binary (not the entry) is returned
        by _run_python_init_if_needed unmodified, so the bump survives.
        """
        proj = fauxware_project
        state = proj.factory.blank_state(addr=proj.entry + 0x10)
        assert state.addr != proj.entry
        bumped = state.heap.heap_location + 0x4000
        state.heap.heap_location = bumped

        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        assert mgr._rust_mgr.get_state_heap_brk(sid) == bumped

    def test_export_path_syncs_rust_heap_brk_into_state_heap(self, fauxware_project):
        """End-to-end: a Rust-side heap_brk advance is visible on the angr
        SimState returned by mgr.active.

        Pre-fix this fails — state.heap.heap_location stays at the default
        even though Rust bumped its internal counter, leading to the silent
        heap-collision scenario in the bead description.
        """
        proj = fauxware_project
        state = proj.factory.entry_state()
        starting = state.heap.heap_location
        assert isinstance(starting, int)

        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        assert len(active_ids) == 1
        sid = active_ids[0]
        assert mgr._rust_mgr.get_state_heap_brk(sid) == starting

        # Simulate what heap_alloc does on a native malloc: bump the per-state
        # heap_brk by one page past the base.
        bumped = starting + 0x1000
        mgr._rust_mgr.set_state_heap_brk(sid, bumped)
        assert mgr._rust_mgr.get_state_heap_brk(sid) == bumped

        states = mgr._get_stash_states("active")
        assert len(states) == 1
        synced = states[0]

        assert synced.heap.heap_location == bumped, (
            f"state.heap.heap_location = {synced.heap.heap_location!r} but "
            f"Rust's heap_brk advanced to 0x{bumped:x} — sync did not run on "
            f"stash export and a Python-side malloc fallback would now overlap "
            f"a Rust-allocated region."
        )

    def test_export_path_does_not_clobber_higher_python_heap_brk(self, fauxware_project):
        """The sync takes max(rust, python) — a Python-side advance that
        outpaced Rust must not be reverted."""
        proj = fauxware_project
        state = proj.factory.entry_state()
        starting = state.heap.heap_location

        mgr = RustExplorationManager(proj, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        sid = active_ids[0]

        # Python advanced heap_location; Rust still at base.
        cached = mgr._state_cache[sid]
        higher = starting + 0x6000
        cached.heap.heap_location = higher
        assert mgr._rust_mgr.get_state_heap_brk(sid) == starting

        states = mgr._get_stash_states("active")
        assert len(states) == 1
        # Python's higher value wins — not clobbered by Rust's smaller value.
        assert states[0].heap.heap_location == higher


class TestCallbackHeapSync:
    """The SimProcedure-callback bounce must round-trip the heap break pointer.

    Regression for angr-op0dn.14.1.3 (found by the M6.5a fallback census): the
    heap sync helpers were export-path-only, so a bounced malloc/calloc/
    operator-new read the *default* heap_location off its freshly copied
    callback state — even when native allocations had already moved Rust's
    heap_brk past it — and its own bump never reached Rust. Either half hands
    out overlapping heap blocks.

    Both directions are exercised through a Python-only SimProcedure (the Rust
    engine has no handler for it, so hitting it forces the bounce).
    """

    @staticmethod
    def _alloc_proc(record):
        class AllocProc(angr.SimProcedure):
            def run(self):
                record.append(self.state.heap.allocate(0x30))
                return 0

        return AllocProc

    @pytest.fixture
    def hookable_project(self):
        """Function-scoped: these tests install hooks, so they must not share
        the module-scoped ``fauxware_project``."""
        import os

        return angr.Project(os.path.join(TEST_BINARIES_DIR, "fauxware"), auto_load_libs=False)

    def test_bounced_alloc_sees_rust_heap_brk(self, hookable_project):
        """Inbound: a native allocation already moved Rust's heap_brk, so the
        address the bounced proc hands out must sit above it.

        Pre-fix the proc allocates from the default 0xC0000000 base and returns
        an address Rust already gave away.
        """
        proj = hookable_project
        allocated = []
        proj.hook_symbol("puts", self._alloc_proc(allocated)())

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        # Stand in for a prior native malloc that bumped Rust's bump allocator.
        native_brk = 0xC000_4000
        mgr._rust_mgr.set_state_heap_brk(sid, native_brk)

        mgr.run(max_steps=60)

        assert allocated, "puts hook never bounced — test would vacuously pass"
        assert min(allocated) >= native_brk, (
            f"bounced allocator handed out 0x{min(allocated):x}, below Rust's "
            f"heap_brk 0x{native_brk:x} — it would overlap a native allocation"
        )

    def test_bounced_alloc_bump_reaches_rust(self, hookable_project):
        """Outbound: the bump the bounced proc made must land in Rust, so a
        later native malloc starts above it.

        Checked against Rust directly (not the exported SimState) — the export
        path takes max(rust, python), which would mask a missing write-back.
        """
        proj = hookable_project
        allocated = []
        proj.hook_symbol("puts", self._alloc_proc(allocated)())

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=60)

        assert allocated
        expected = max(allocated) + 0x30
        brks = [
            mgr._rust_mgr.get_state_heap_brk(sid)
            for stash in ("active", "deadended", "found", "avoid")
            for sid in mgr._rust_mgr.get_state_ids(stash)
        ]
        assert brks, "no surviving Rust states to check"
        assert max(brks) >= expected, (
            f"Rust's heap_brk topped out at 0x{max(brks):x} but the bounced "
            f"allocator ran up to 0x{expected:x} — the bump never reached Rust, "
            f"so a native malloc would overlap it"
        )


class TestCallbackPosixSync:
    """The SimProcedure-callback bounce must round-trip fd output buffers.

    Regression for angr-op0dn.14.1.4 (gap 2 of the M6.5a fallback census): the
    bounce had no outbound posix channel at all, so a bounced
    fwrite/fputc/fprintf wrote the callback state's posix stream and that
    output was invisible to ``posix.dumps(1)`` read back from the Rust state —
    and to any find predicate keyed on stdout. The inbound half was
    export-path-only, so a bounced proc could not see the native output that
    preceded it either.
    """

    @staticmethod
    def _write_proc(payload, seen):
        class WriteProc(angr.SimProcedure):
            def run(self):
                # What the bounced proc sees of the native output so far.
                seen.append(self.state.posix.dumps(1))
                self.state.posix.stdout.write(None, claripy.BVV(payload), events=False)
                return 0

        return WriteProc

    @pytest.fixture
    def hookable_project(self):
        import os

        return angr.Project(os.path.join(TEST_BINARIES_DIR, "fauxware"), auto_load_libs=False)

    def test_bounced_write_reaches_rust(self, hookable_project):
        """Outbound: what the bounced proc wrote to stdout must land in Rust.

        Checked against Rust's own fd buffer, not the exported SimState, so a
        Python-side plugin that happens to carry the bytes cannot mask a
        missing write-back.
        """
        proj = hookable_project
        payload = b"BOUNCED-OUTPUT"
        proj.hook_symbol("puts", self._write_proc(payload, [])())

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=60)

        outs = [
            bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            for stash in ("active", "deadended", "found", "avoid")
            for sid in mgr._rust_mgr.get_state_ids(stash)
        ]
        assert outs, "no surviving Rust states to check"
        assert any(payload in o for o in outs), f"the bounced proc's stdout write never reached Rust: {outs!r}"

    def test_bounced_proc_sees_native_stdout(self, hookable_project):
        """Inbound: native output written before the bounce must be visible to
        a bounced proc that reads ``state.posix.dumps(1)``."""
        proj = hookable_project
        seen = []
        proj.hook_symbol("puts", self._write_proc(b"X", seen)())

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        # Stand in for a native puts/printf that ran before the bounce.
        native = b"NATIVE-STDOUT"
        mgr._rust_mgr.append_state_fd_output(sid, 1, native)

        mgr.run(max_steps=60)

        assert seen, "puts hook never bounced — test would vacuously pass"
        assert any(s.startswith(native) for s in seen), (
            f"bounced proc did not see the native stdout that preceded it: {seen!r}"
        )

    def test_repeated_bounce_does_not_duplicate_stdout(self, hookable_project):
        """The cached callback state is reused across bounces, so the inbound
        injection must be idempotent — no duplicated native output."""
        proj = hookable_project
        seen = []
        proj.hook_symbol("puts", self._write_proc(b"", seen)())

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        mgr._rust_mgr.append_state_fd_output(sid, 1, b"AB")

        mgr.run(max_steps=60)

        assert seen
        for s in seen:
            assert s.count(b"AB") == 1, f"native stdout duplicated across bounces: {s!r}"


class TestCallbackFdTableSync:
    """The SimProcedure-callback bounce must round-trip the fd *table*, not
    just the fd output bytes.

    Regression for angr-op0dn.14.1.5: a bounced open/fopen/dup allocates an fd
    in ``state.posix`` and a ``SimFile`` in ``state.fs``, and Rust's FileSystem
    never learned about it — so ``get_state_open_fds`` omitted the fd and a
    later *native* read/write on it hit a closed fd.
    """

    O_WRONLY = 1

    @pytest.fixture
    def hookable_project(self):
        import os

        return angr.Project(os.path.join(TEST_BINARIES_DIR, "fauxware"), auto_load_libs=False)

    @staticmethod
    def _open_proc(path, opened):
        class OpenProc(angr.SimProcedure):
            def run(self):
                opened.append(self.state.posix.open(path, claripy.BVV(TestCallbackFdTableSync.O_WRONLY, 32)))
                return 0

        return OpenProc

    def _run_with_open_hook(self, proj, path):
        opened = []
        proj.hook_symbol("puts", self._open_proc(path, opened)())
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        mgr.run(max_steps=60)
        assert opened, "puts hook never bounced — test would vacuously pass"
        return mgr, opened[0]

    def _rust_fds(self, mgr, fd):
        """Every (state_id, fd_info) pair where a Rust state knows ``fd``."""
        return [
            (sid, info)
            for stash in ("active", "deadended", "found", "avoid")
            for sid in mgr._rust_mgr.get_state_ids(stash)
            for info in mgr._rust_mgr.get_state_open_fds(sid)
            if info[0] == fd
        ]

    def test_bounced_open_reaches_rust_fd_table(self, hookable_project):
        """The fd a bounced proc opened must show up in ``get_state_open_fds``."""
        path = b"/tmp/bounced-open.txt"
        mgr, fd = self._run_with_open_hook(hookable_project, path)

        hits = self._rust_fds(mgr, fd)
        assert hits, f"fd {fd} opened by the bounced proc never reached Rust's fd table"
        _, info = hits[0]
        assert info[1] == path.decode(), f"wrong name on the adopted fd: {info!r}"
        assert info[5], "adopted fd is not marked open"

    def test_native_write_on_bounced_fd_works(self, hookable_project):
        """A native write on the bounced-open fd must succeed, not hit a closed
        fd. ``append_state_fd_output`` is the same ``RustSimState::write_fd``
        choke point every native proc writes through, so its False return is
        exactly the pre-fix failure."""
        mgr, fd = self._run_with_open_hook(hookable_project, b"/tmp/bounced-write.txt")

        sid = self._rust_fds(mgr, fd)[0][0]
        assert mgr._rust_mgr.append_state_fd_output(sid, fd, b"NATIVE"), "native write on the bounced fd was refused"
        assert bytes(mgr._rust_mgr.get_state_fd_content(sid, fd)) == b"NATIVE"

    def test_no_extra_fds_skips_the_ffi(self, hookable_project):
        """A proc that opened nothing must not cross the FFI boundary at all."""
        proj = hookable_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        state = proj.factory.entry_state()

        class Tripwire:
            def get_state_open_fds(self, _state_id):
                raise AssertionError("fd-table diff crossed the FFI with no fds above stderr")

        mgr._rust_mgr = Tripwire()
        mgr._sync_state_posix_fds_to_rust(state, 1)


class TestStateMetadataStorage:
    """Tests for per-state metadata moved from Python ``_state_metadata`` dict
    into Rust ``RustSimState`` (angr-p8o3).

    The previous implementation kept three Python-side maps
    (``symbolic_pages``, ``hook_symbolic_memory``, ``addr_to_ast``) keyed by
    state ID. They now live on each ``RustSimState`` so the storage and the
    state lifetime are unified — when Rust drops the state, the metadata is
    freed automatically.
    """

    def test_addr_to_ast_round_trip(self):
        """set_state_addr_to_ast then get_state_addr_to_ast returns the same
        AST object and size."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        ast = claripy.BVS("sym_addr_round_trip", 32)
        mgr.set_state_addr_to_ast(sid, 0x4000, ast, 4)

        out = mgr.get_state_addr_to_ast(sid)
        assert 0x4000 in out
        recovered_ast, recovered_size = out[0x4000]
        # Identity preserved — Rust holds a strong PyObject ref, not a clone.
        assert recovered_ast is ast
        assert recovered_size == 4

    def test_hook_symbolic_memory_round_trip(self):
        """Hook symbolic memory entries survive a round trip."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        ast = claripy.BVS("hook_round_trip", 64)
        mgr.set_state_hook_symbolic_memory(sid, 0x5000, ast, 8)

        out = mgr.get_state_hook_symbolic_memory(sid)
        assert 0x5000 in out
        recovered_ast, recovered_size = out[0x5000]
        assert recovered_ast is ast
        assert recovered_size == 8

    def test_symbolic_pages_replace_whole_dict(self):
        """set_state_symbolic_pages replaces the entire map. A second call
        overwrites the previous contents — matches the old
        ``_state_md(sid).symbolic_pages = pages`` assignment semantics.
        """
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        first = {0x1000: claripy.BVS("page_first", 8 * 4096)}
        mgr.set_state_symbolic_pages(sid, first)
        assert dict(mgr.get_state_symbolic_pages(sid)) == first

        second = {0x2000: claripy.BVS("page_second", 8 * 4096)}
        mgr.set_state_symbolic_pages(sid, second)
        # Old entry gone, new one present.
        out = mgr.get_state_symbolic_pages(sid)
        assert 0x1000 not in out
        assert 0x2000 in out
        assert out[0x2000] is second[0x2000]

    def test_unknown_state_returns_empty(self):
        """Reads for an unknown state ID return an empty dict — preserves the
        old ``_state_metadata.get(sid)`` falsy semantics that callbacks rely
        on with ``if md and md.X``.
        """
        mgr = _RustExplorationManager("amd64")
        assert dict(mgr.get_state_addr_to_ast(424242)) == {}
        assert dict(mgr.get_state_hook_symbolic_memory(424242)) == {}
        assert dict(mgr.get_state_symbolic_pages(424242)) == {}

    def test_setter_unknown_state_raises(self):
        """Unknown state IDs on the setter side surface ValueError — matches
        every other ``set_state_*`` method on the manager."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        ast = claripy.BVS("nope", 8)
        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr.set_state_addr_to_ast(424242, 0x1, ast, 1)

    def test_clear_state_metadata_drops_all_three_maps(self):
        """clear_state_metadata removes every map for the state — replaces
        the previous ``_state_metadata.pop(sid, None)`` cleanup."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        mgr.set_state_addr_to_ast(sid, 0x10, claripy.BVS("a", 8), 1)
        mgr.set_state_hook_symbolic_memory(sid, 0x20, claripy.BVS("b", 8), 1)
        mgr.set_state_symbolic_pages(sid, {0x1000: claripy.BVS("c", 8 * 4096)})

        mgr.clear_state_metadata(sid)

        assert dict(mgr.get_state_addr_to_ast(sid)) == {}
        assert dict(mgr.get_state_hook_symbolic_memory(sid)) == {}
        assert dict(mgr.get_state_symbolic_pages(sid)) == {}

    def test_clear_state_metadata_unknown_state_is_noop(self):
        """clear_state_metadata on a missing state returns silently — matches
        ``dict.pop(sid, None)`` semantics it replaces."""
        mgr = _RustExplorationManager("amd64")
        # Should not raise.
        mgr.clear_state_metadata(424242)

    def test_explicit_clear_after_state_drop_is_safe(self):
        """``clear_state_metadata`` must tolerate a state ID that no longer
        matches any stash entry — a stale ID from a state that was already
        moved/dropped should be a no-op, not a panic.
        """
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        # Move the state out so the manager's stash lookup will miss it.
        mgr.move_states("active", "deadended", None)
        # Stale ID — this is still in `deadended` so technically not stale,
        # but the API must accept any u64. Use a guaranteed-missing ID too.
        mgr.clear_state_metadata(sid)
        mgr.clear_state_metadata(0xDEAD_BEEF_DEAD_BEEF)

    def test_fork_does_not_alias_metadata(self):
        """``RustSimState.fork`` clones the per-state metadata maps so that
        parent and child have independent storage. Catches the regression
        where a missing fork-time clone would leave both states pointing at
        the same backing HashMap.
        """
        import claripy

        ast_parent = claripy.BVS("parent_only", 32)
        # We need to set the entry through the manager API. Wire the state
        # in via create_state isn't enough since we want the .fork() path,
        # so do it directly through a manager + a fresh state.
        mgr = _RustExplorationManager("amd64")
        parent_sid = mgr.create_state("active")
        mgr.set_state_addr_to_ast(parent_sid, 0x9000, ast_parent, 4)

        # Sanity: parent entry visible.
        assert 0x9000 in mgr.get_state_addr_to_ast(parent_sid)

        # Exercise the real fork path: ``fork_state_to_stash`` calls
        # ``RustSimState.fork``, which must clone the parent's metadata into
        # the child's own backing map. The child therefore sees the planted
        # entry...
        child_sid = mgr.fork_state_to_stash(parent_sid, "active")
        assert 0x9000 in mgr.get_state_addr_to_ast(child_sid), (
            "fork must clone the parent's addr_to_ast entry into the child"
        )

        # ...but the maps are independent: a write on the child must not
        # appear on the parent, and vice-versa. A missing fork-time clone
        # (shared HashMap) would leak each write across the boundary.
        ast_child = claripy.BVS("child_only", 32)
        mgr.set_state_addr_to_ast(child_sid, 0xA000, ast_child, 4)
        assert 0xA000 not in mgr.get_state_addr_to_ast(parent_sid), "child write aliased into the parent's metadata map"

        ast_parent2 = claripy.BVS("parent_only_2", 32)
        mgr.set_state_addr_to_ast(parent_sid, 0xB000, ast_parent2, 4)
        assert 0xB000 not in mgr.get_state_addr_to_ast(child_sid), "parent write aliased into the child's metadata map"

    # ------------------------------------------------------------------
    # angr-nsg9: lifecycle tests for the metadata-storage refactor.
    # The Python `_state_metadata` dict was replaced with per-state Rust
    # storage. These tests pin the cleanup, fork-duplication, and
    # eviction-order contracts so a regression in any of them surfaces
    # as a leak (or silent staleness) rather than a generic crash.
    # ------------------------------------------------------------------

    def test_cleanup_state_cache_prunes_predicate_eval_cache(self, fauxware_project):
        """`_cleanup_state_cache` must drop `_predicate_eval_cache` entries
        whose state no longer exists in any Rust stash (angr-iu40). This
        change-detection cache was formerly pruned per-state by the removed
        `_cleanup_state_refs`; folding it into the cache-cleanup path keeps
        the (addr, stdout_len) map bounded over long explorations instead of
        growing one entry per dead state.
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        live_ids = mgr._rust_mgr.get_state_ids("active")
        assert len(live_ids) >= 1
        live_sid = live_ids[0]

        # A live state's eval-cache entry must survive; a phantom id's must go.
        dead_sid = 0xDEAD_BEEF_DEAD_BEEF
        mgr._predicate_eval_cache = {live_sid: (0x1000, 0), dead_sid: (0x2000, 5)}

        mgr._cleanup_state_cache()

        assert dead_sid not in mgr._predicate_eval_cache, (
            "stale predicate-eval entry for a non-stash state must be pruned"
        )
        assert live_sid in mgr._predicate_eval_cache, "predicate-eval entry for a live active state must be preserved"

    def test_cleanup_state_cache_evicts_oldest_first(self, fauxware_project):
        """When `_state_cache` grows beyond `_max_state_cache_size`,
        `_cleanup_state_cache` (manager version) evicts in insertion order
        — Python dict preserves it since 3.7+, so the oldest entries leave
        first while the newest stay. Pinned state ids (roots, current
        callback, stepping target) are skipped.

        Note: metadata is NOT scrubbed here — per-state Rust metadata is
        freed when ``RustSimState`` itself drops (once the state leaves
        every stash).
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        # Reset the cache so the manager's pre-populated entry state
        # doesn't take up one of our slots and offset eviction order.
        # Also clear roots so nothing is pinned during the assertion.
        mgr._state_cache.clear()
        mgr._state_roots = {}
        mgr._current_callback_state_id = None
        mgr._current_stepping_state_id = None
        mgr._max_state_cache_size = 2

        # Insert 5 states. Each must be live in 'active' so the manager's
        # liveness check (Step 1) does not drop them prematurely.
        ordered_sids = []
        sentinel_state = proj.factory.entry_state()
        for _ in range(5):
            sid = mgr._rust_mgr.create_state("active")
            mgr._state_cache[sid] = sentinel_state
            ordered_sids.append(sid)

        assert len(mgr._state_cache) == 5

        mgr._cleanup_state_cache()

        # Cache is back at the cap, oldest 3 evicted, newest 2 retained.
        assert len(mgr._state_cache) == 2
        retained = set(mgr._state_cache.keys())
        evicted = [s for s in ordered_sids if s not in retained]
        assert evicted == ordered_sids[:3], (
            f"expected oldest 3 evicted in insertion order; got evicted={evicted}, retained={retained}"
        )

    def test_cleanup_state_cache_drops_dead_states(self, fauxware_project):
        """Step 1 of ``_cleanup_state_cache``: any state id whose state
        no longer exists in active/found must be removed from
        ``_state_cache`` regardless of insertion order. This prevents the
        cache from holding a strong ref to a state the manager already
        deadended/errored — a Python-side leak the per-state metadata
        refactor was meant to eliminate.
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        mgr._state_cache.clear()
        mgr._state_roots = {}
        mgr._current_callback_state_id = None
        mgr._current_stepping_state_id = None

        sentinel_state = proj.factory.entry_state()
        live_sid = mgr._rust_mgr.create_state("active")
        # An id that was never in any stash — guaranteed-dead.
        dead_sid = 0xDEAD_BEEF_DEAD_BEEF
        mgr._state_cache[live_sid] = sentinel_state
        mgr._state_cache[dead_sid] = sentinel_state

        mgr._cleanup_state_cache()

        assert live_sid in mgr._state_cache
        assert dead_sid not in mgr._state_cache, "states absent from active/found stashes must be dropped from cache"

    def test_cleanup_state_cache_skips_pinned(self, fauxware_project):
        """``_cleanup_state_cache`` Step 2: pinned ids (roots, current
        callback state, stepping target) are exempt from eviction even
        when the cache is over cap. This is what keeps the in-flight
        callback state alive across cache pressure.
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        mgr._state_cache.clear()
        mgr._max_state_cache_size = 1

        # Three states; the first one is the "in-flight callback" — pin it.
        sentinel = proj.factory.entry_state()
        sid_pinned = mgr._rust_mgr.create_state("active")
        sid_b = mgr._rust_mgr.create_state("active")
        sid_c = mgr._rust_mgr.create_state("active")
        mgr._state_cache[sid_pinned] = sentinel
        mgr._state_cache[sid_b] = sentinel
        mgr._state_cache[sid_c] = sentinel
        mgr._current_callback_state_id = sid_pinned
        mgr._current_stepping_state_id = None
        mgr._state_roots = {}

        mgr._cleanup_state_cache()

        assert sid_pinned in mgr._state_cache, "current callback state must not be evicted under cache pressure"

    def test_cleanup_state_cache_prunes_state_roots(self, fauxware_project):
        """`_cleanup_state_cache` must drop `_state_roots` entries whose key
        state no longer exists in any Rust stash. Without this, the dict
        grows monotonically across `explore()` calls and root pinning bloats
        `_state_cache` indirectly (every dead root pinned into the live set).
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        # Live state + a guaranteed-dead id that has a root-mapping entry.
        live_sid = mgr._rust_mgr.create_state("active")
        dead_sid = 0xDEAD_BEEF_DEAD_BEEF
        dead_root = 0xDEAD_BEEF_DEAD_BEEE
        mgr._state_roots[live_sid] = live_sid
        mgr._state_roots[dead_sid] = dead_root

        mgr._cleanup_state_cache()

        assert live_sid in mgr._state_roots
        assert dead_sid not in mgr._state_roots, "_state_roots entry for a dead state must be pruned"

    def test_cleanup_state_cache_prunes_predicate_matched_ids(self, fauxware_project):
        """`_cleanup_state_cache` must shrink `_predicate_matched_ids` to
        only ids that still exist in some Rust stash. The set otherwise
        grows monotonically over the manager's lifetime — fine for a one-
        shot script, leaky for orchestrators that drive many explore()s.
        """

        proj = fauxware_project
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        live_sid = mgr._rust_mgr.create_state("active")
        dead_sid = 0xDEAD_BEEF_DEAD_BEEF
        mgr._predicate_matched_ids = {live_sid, dead_sid}

        mgr._cleanup_state_cache()

        assert live_sid in mgr._predicate_matched_ids
        assert dead_sid not in mgr._predicate_matched_ids

    def test_state_fork_clones_metadata_via_dispatcher(self, fauxware_project):
        """The Rust dispatcher forks states on symbolic branches, and the
        forked state's metadata must be a clone of the parent's, not a
        shared reference. We exercise this through a real fauxware run
        (which forks at the password compare) and verify that whatever
        metadata the parent had is also visible on each forked descendant
        — the no-aliasing claim is then proven by the per-state-isolation
        invariants pinned upstream.
        """
        import claripy

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        # Plant metadata on the entry state before exploration begins.
        entry_sids = mgr._rust_mgr.get_state_ids("active")
        assert len(entry_sids) == 1
        entry_sid = entry_sids[0]
        ast = claripy.BVS("fork_meta", 32)
        mgr._rust_mgr.set_state_addr_to_ast(entry_sid, 0x9000, ast, 4)

        # Run far enough for the symbolic branch in fauxware to fork.
        for _ in range(20):
            mgr.run(max_steps=15)
            if not mgr._rust_mgr.has_active_states():
                break

        # Gather every state the run produced across all stashes. On a
        # fauxware run the password compare forks the entry, so at least one
        # descendant (an id other than the original entry) must exist. We
        # read a CHILD's metadata, not just the parent's — the parent's
        # owned-by-value map is untouched by the clone path, so checking it
        # alone proves nothing about whether the fork wired metadata through
        # to children.
        all_sids = []
        for stash in mgr._rust_mgr.stash_counts():
            all_sids.extend(mgr._rust_mgr.get_state_ids(stash))
        descendants = [sid for sid in all_sids if sid != entry_sid]
        assert descendants, "fauxware run produced no forked descendant to inspect"

        # Every forked descendant must carry a CLONE of the planted metadata:
        # the same (0x9000 -> ast) entry with Python object identity
        # preserved. If fork dropped the child's metadata (children built with
        # empty maps) or aliased+overwrote it, no descendant carries 0x9000
        # and this fails.
        cloned = False
        for sid in descendants:
            meta = dict(mgr._rust_mgr.get_state_addr_to_ast(sid))
            if 0x9000 in meta:
                recovered_ast, _ = meta[0x9000]
                # clone_ref preserves Python object identity.
                assert recovered_ast is ast, (
                    f"forked child {sid}'s metadata AST is not the cloned "
                    "parent AST — fork shared the HashMap and overwrote it"
                )
                cloned = True
        assert cloned, (
            "no forked descendant carried the parent's planted metadata — "
            "clone_py_metadata did not wire the entry through the fork chain"
        )


class TestStashOperations:
    """Tests for stash management operations."""

    def test_move_states_all(self):
        """move_states without filter moves all states."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        mgr.create_state("active")
        mgr.create_state("active")
        assert mgr.active_count() == 3

        count = mgr.move_states("active", "found", None)
        assert count == 3
        assert mgr.active_count() == 0
        assert mgr.found_count() == 3

    def test_move_states_empty_source(self):
        """move_states from empty stash returns 0."""
        mgr = _RustExplorationManager("amd64")
        count = mgr.move_states("active", "found", None)
        assert count == 0

    def test_move_state_by_id(self):
        """move_state moves a specific state by ID."""
        mgr = _RustExplorationManager("amd64")
        id1 = mgr.create_state("active")
        id2 = mgr.create_state("active")

        result = mgr.move_state(id1, "active", "found")
        assert result is True
        assert mgr.active_count() == 1
        assert mgr.found_count() == 1

        # The remaining state should be id2
        remaining = mgr.get_state_ids("active")
        assert id2 in remaining

    def test_move_state_nonexistent(self):
        """move_state returns False for nonexistent state ID."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        result = mgr.move_state(999999, "active", "found")
        assert result is False

    def test_clear_stash(self):
        """clear_stash removes all states from a stash."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("found")
        mgr.create_state("found")
        assert mgr.found_count() == 2

        mgr.clear_stash("found")
        assert mgr.found_count() == 0

    def test_clear_empty_stash(self):
        """clear_stash on empty stash is a no-op."""
        mgr = _RustExplorationManager("amd64")
        mgr.clear_stash("nonexistent")  # Should not raise

    def test_stash_counts_multiple(self):
        """stash_counts includes all stash names."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        mgr.create_state("found")
        mgr.create_state("deadended")

        counts = mgr.stash_counts()
        assert counts["active"] == 1
        assert counts["found"] == 1
        assert counts["deadended"] == 1

    def test_get_state_ids_empty(self):
        """get_state_ids on empty stash returns empty list."""
        mgr = _RustExplorationManager("amd64")
        ids = mgr.get_state_ids("active")
        assert ids == []


class TestHooksAndProcedures:
    """Tests for hook and SimProcedure registration."""

    def test_register_hook(self):
        """Registering a hook at an address."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x401000, "test_hook", 0, False)
        stats = mgr.stats()
        assert stats["hooks"] == 1

    def test_register_multiple_hooks(self):
        """Multiple hooks at different addresses."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x401000, "hook1", 1, False)
        mgr.register_simprocedure(0x402000, "hook2", 2, False)
        mgr.register_simprocedure(0x403000, "hook3", 0, True)
        stats = mgr.stats()
        assert stats["simprocedures"] == 3
        assert stats["hooks"] == 3

    def test_set_find_avoid_addrs(self):
        """Setting find and avoid addresses."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000, 0x2000])
        mgr.set_avoid_addrs([0x3000])

        stats = mgr.stats()
        assert stats["find_addrs"] == 2
        assert stats["avoid_addrs"] == 1

    def test_empty_find_avoid(self):
        """Empty find/avoid lists."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([])
        mgr.set_avoid_addrs([])
        stats = mgr.stats()
        assert stats["find_addrs"] == 0
        assert stats["avoid_addrs"] == 0

    def test_simproc_dispatch_name_prefers_display_name(self):
        """angr-gbk6: dispatch name follows display_name, not class.

        SimLibrary instantiates unimplemented libc symbols as
        ``ReturnUnconstrained(display_name=<symbol>)``. Keying off the class
        name would route every libc stub through the same "ReturnUnconstrained"
        slot, so the per-symbol native registry on the Rust side would never
        match. The helper must surface the per-instance display_name.
        """
        from angr.exploration.rust_callback_dispatch import _simproc_dispatch_name
        from angr.procedures.posix.getenv import getenv
        from angr.procedures.stubs.ReturnUnconstrained import ReturnUnconstrained

        stub = ReturnUnconstrained(display_name="setenv")
        assert _simproc_dispatch_name(stub) == "setenv"

        # First-class SimProc keeps its class name (display_name defaults to
        # type(self).__name__ in SimProcedure.__init__).
        real = getenv()
        assert _simproc_dispatch_name(real) == "getenv"

    def test_register_simprocedures_uses_display_name_for_stubs(self, fauxware_project, monkeypatch):
        """angr-gbk6: _register_simprocedures wires stubs to Rust by symbol.

        Without preferring display_name, every ReturnUnconstrained-backed
        libc stub on a real binary registers under "ReturnUnconstrained" and
        the Rust-side native procedure registry never sees the symbol name
        (so e.g. native ``setenv`` stays dormant). Capture the tuples handed
        to Rust and assert the stub address went over as "setenv".
        """
        from angr.procedures.stubs.ReturnUnconstrained import ReturnUnconstrained

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        stub_addr = 0x4F1000
        stub = ReturnUnconstrained(display_name="setenv")
        proj._sim_procedures[stub_addr] = stub
        try:
            captured = []

            class _SpyRustMgr:
                def __init__(self, inner):
                    self._inner = inner

                def register_simprocedures(self, procs):
                    captured.extend(procs)
                    return self._inner.register_simprocedures(procs)

                def __getattr__(self, item):
                    return getattr(self._inner, item)

            monkeypatch.setattr(mgr, "_rust_mgr", _SpyRustMgr(mgr._rust_mgr))
            mgr._register_simprocedures()
        finally:
            proj._sim_procedures.pop(stub_addr, None)

        names_by_addr = {addr: name for (addr, name, _na, _nr) in captured}
        assert names_by_addr.get(stub_addr) == "setenv", (
            f"expected stub to register as 'setenv', got {names_by_addr.get(stub_addr)!r}; full capture={captured}"
        )


class TestStateManagement:
    """Tests for state creation and management."""

    def test_state_pc_get_set(self):
        """Get and set PC on states via manager."""
        mgr = _RustExplorationManager("amd64")
        state = RustSimState("amd64")
        state.pc = 0x401000
        mgr.add_state("active", state)

        pc = mgr.get_state_pc("active", 0)
        assert pc == 0x401000

    def test_multiple_states_different_pcs(self):
        """Multiple states with different PCs."""
        mgr = _RustExplorationManager("amd64")

        for addr in [0x1000, 0x2000, 0x3000]:
            state = RustSimState("amd64")
            state.pc = addr
            mgr.add_state("active", state)

        assert mgr.active_count() == 3
        # The "different PCs" claim is only meaningful if each PC round-trips
        # back distinctly — active_count alone passes even if every PC were
        # zeroed or collapsed. Read them back via get_state_pc.
        assert [mgr.get_state_pc("active", i) for i in range(3)] == [0x1000, 0x2000, 0x3000]

    def test_has_active_states(self):
        """has_active_states reflects stash contents."""
        mgr = _RustExplorationManager("amd64")
        assert not mgr.has_active_states()

        mgr.create_state("active")
        assert mgr.has_active_states()

    def test_drop_terminal_states_toggle(self):
        """set_drop_terminal_states flips the observable stats flag both ways."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.stats()["drop_terminal_states"] is False  # default
        mgr.set_drop_terminal_states(True)
        assert mgr.stats()["drop_terminal_states"] is True
        mgr.set_drop_terminal_states(False)
        assert mgr.stats()["drop_terminal_states"] is False


class TestWideConcreteMemoryRoundTrip:
    """Python-boundary regression for the >16-byte concrete load corruption
    fixed in commit df4bb4cd3 (bd angr-tk7yv).

    ``load_concrete`` / ``concat_into`` packed concrete bytes into a u128
    (16 bytes); for size>16 the shift wrapped mod 128 and OR-ed the high
    chunk over the low chunk, so a 24-byte read of xmllint rodata
    ``-maxmem\\0--debug\\0--shell\\0`` came back as
    ``-msxmmm\\0--debug\\0-msxmmm\\0`` (chunk0 | chunk2). The Rust-side fix
    has a unit test (``test_wide_concrete_load_exact``); this pins the
    user-facing FFI path (``mgr.eval_memory`` -> ``get_state_memory`` ->
    ``_get_state_memory``) that the bug actually corrupted.
    """

    def _mgr_and_oracle(self, fauxware_project):
        """A manager over fauxware plus the loader's concrete image bytes.

        eval_memory reads Rust's own state memory; the loaded binary image
        is the reliably concrete-backed region in Rust (a Python-side
        state.memory.store does not eagerly mirror into Rust). The loaded
        code/rodata at the entry point is the same kind of concrete image
        memory the original xmllint corruption was read from.
        """
        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        return mgr, sid, proj

    @pytest.mark.parametrize("size", [8, 16, 17, 24, 31, 32, 48])
    def test_eval_memory_matches_loader_image(self, fauxware_project, size):
        """eval_memory round-trips the loaded image byte-for-byte at sizes
        spanning the 16-byte u128 boundary. Pre-fix, size>16 wrapped the
        u128 shift mod 128 and OR-ed the high chunk over the low chunk,
        corrupting the leading bytes."""
        mgr, sid, proj = self._mgr_and_oracle(fauxware_project)
        addr = proj.entry
        expected = proj.loader.memory.load(addr, size)
        assert len(expected) == size
        got = mgr.eval_memory(sid, addr, size)
        assert got == expected, f"{size}-byte concrete read mismatch: {got!r} != {expected!r}"

    def test_eval_memory_wide_read_not_chunk_or_folded(self, fauxware_project):
        """A 24-byte read whose chunk0 and chunk1 differ must not collapse
        to (chunk0 | chunk1...). Asserts the high bytes survive distinctly
        rather than being OR-folded over the low 16 — the exact failure
        mode of the pre-fix u128 packing."""
        mgr, sid, proj = self._mgr_and_oracle(fauxware_project)
        addr = proj.entry
        expected = proj.loader.memory.load(addr, 24)
        got = mgr.eval_memory(sid, addr, 24)
        assert got == expected, f"wide read corrupted: {got!r} != {expected!r}"
        # Guard the regression directly: bytes 16..24 are not (low | high).
        folded = bytes(expected[i] | expected[i + 16] for i in range(8))
        assert got[:8] == expected[:8], (
            f"leading bytes OR-folded with high chunk: got {got[:8]!r}, OR-fold would be {folded!r}"
        )


class TestSymbolicRegisterSync:
    """angr-5rjbq: a callback that assigns a *symbolic* value to a non-return
    register (e.g. ``state.regs.ecx = ebp - 0x70004``, as flareon2015_5's hooks
    do) must have that AST synced to Rust. It used to be dropped: only return
    registers (eax/rax) took the symbolic-sync path, so Rust kept executing with
    the stale pre-callback register value."""

    @staticmethod
    def _mgr(fauxware_project):
        proj = fauxware_project
        state = proj.factory.entry_state()
        return RustExplorationManager(proj, [state]), proj

    def _capture(self, monkeypatch, fauxware_project, reg, make_new_val):
        mgr, proj = self._mgr(fauxware_project)
        old_state = proj.factory.blank_state(addr=proj.entry)
        new_state = old_state.copy()
        setattr(new_state.regs, reg, make_new_val(old_state))

        synced = {}
        monkeypatch.setattr(mgr, "_sync_symbolic_register_to_rust", lambda name, value: synced.__setitem__(name, value))
        concrete = mgr._extract_register_changes(old_state, new_state)
        return synced, concrete

    def test_symbolic_nonreturn_register_is_synced(self, monkeypatch, fauxware_project):
        """rdi := (rbp - 0x70004) — a symbolic address expression — reaches Rust."""
        synced, _ = self._capture(monkeypatch, fauxware_project, "rdi", lambda s: s.regs.rbp - 0x70004)
        assert "rdi" in synced, f"symbolic write to non-return register rdi was dropped; synced={list(synced)}"
        assert synced["rdi"].symbolic

    def test_unchanged_symbolic_register_is_not_resynced(self, monkeypatch, fauxware_project):
        """An already-symbolic register the callback never touched must not be
        re-exported on every callback (that would thrash the AST bridge)."""
        mgr, proj = self._mgr(fauxware_project)
        state = proj.factory.blank_state(addr=proj.entry)
        assert state.regs.rdi.symbolic, "precondition: blank_state rdi is unconstrained"

        synced = {}
        monkeypatch.setattr(mgr, "_sync_symbolic_register_to_rust", lambda name, value: synced.__setitem__(name, value))
        mgr._extract_register_changes(state, state.copy())
        assert "rdi" not in synced, "untouched symbolic register was needlessly re-synced"

    def test_concrete_register_change_still_concrete(self, monkeypatch, fauxware_project):
        """The concrete diff path is unaffected: rdi := 0x1234 stays a concrete change."""
        synced, concrete = self._capture(monkeypatch, fauxware_project, "rdi", lambda s: 0x1234)
        assert "rdi" not in synced, "concrete value took the symbolic-AST path"
        assert any(int.from_bytes(data, "little") == 0x1234 for (_off, _sz, data) in concrete), (
            f"concrete rdi change missing from diff: {concrete}"
        )
