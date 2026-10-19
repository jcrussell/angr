"""Native libVEX lift parity with pyvex (angr-mx00p.33).

The native lifter (``native/angr/src/vex/libvex_lifter.rs``) calls the same
``vex_lift`` in the loaded libpyvex that pyvex does, through upstream's
runtime-resolved ``ailment::vex_ffi`` struct mirrors. That needs an imported
pyvex, which ``cargo test`` does not have, so the checks live here:

* IRSB parity over a block corpus (``native_lift_corpus.json``): the native
  lift and the pyvex lift must render to the same ``vex::ir`` tree.
* The enum tags and struct layouts the Rust side hard-codes, against pyvex's
  cffi library, which is the ABI authority for the libpyvex it loads.
"""

from __future__ import annotations

import json
import os

import archinfo
import pytest
import pyvex
from pyvex.native import ffi, pvc

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

CORPUS = os.path.join(os.path.dirname(__file__), "native_lift_corpus.json")


def _vex_engine():
    from angr.rustylib import vex_engine

    return vex_engine


def _pyvex_debug(data: bytes, addr: int, arch: archinfo.Arch) -> str:
    """The pyvex lift at the native lifter's settings, as a Rust ``{:#?}``."""
    irsb = pyvex.lift(data, addr, arch, opt_level=1, strict_block_end=False, cross_insn_opt=True)
    return _vex_engine().irsb_debug_for_test(irsb)


def _corpus():
    with open(CORPUS) as f:
        return json.load(f)


@pytest.mark.parametrize("entry", _corpus(), ids=lambda e: f"{e['arch']}@{e['addr']:#x}")
def test_corpus_native_matches_pyvex(entry):
    arch = archinfo.arch_from_id(entry["arch"])
    data = bytes.fromhex(entry["bytes"])
    native = _vex_engine().native_lift_debug_for_test(arch, data, entry["addr"])
    assert native == _pyvex_debug(data, entry["addr"], arch)


def test_native_lift_reads_ret_jumpkind():
    native = _vex_engine().native_lift_debug_for_test(archinfo.ArchAMD64(), b"\xc3", 0x1000)
    assert "jumpkind: Ret" in native


def test_native_lift_rejects_unknown_guest():
    class NoGuest:
        name = "AMD64"
        vex_arch = "VexArchTILEGX"
        vex_archinfo = archinfo.ArchAMD64().vex_archinfo

    with pytest.raises(ValueError, match="no libVEX guest"):
        _vex_engine().native_lift_debug_for_test(NoGuest(), b"\xc3", 0x1000)


def test_enum_tags_match_pyvex():
    tags, _ = _vex_engine().native_lift_tables_for_test()
    mismatched = {name: (tag, getattr(pvc, name, None)) for name, tag in tags if getattr(pvc, name, None) != tag}
    assert not mismatched, f"Rust tag != pyvex tag: {mismatched}"


@pytest.mark.parametrize("prefix", ["Ijk_", "Ity_", "Ico_", "ILGop_", "Iend_"])
def test_enum_families_are_complete(prefix):
    """A tag pyvex grows must be taught to the Rust tables, not decoded as a fallback."""
    tags, _ = _vex_engine().native_lift_tables_for_test()
    known = {name for name, _ in tags}
    missing = sorted(n for n in dir(pvc) if n.startswith(prefix) and n not in known)
    # ILGop_INVALID has no mapping on purpose (`loadg_op`'s `Unknown`).
    assert [n for n in missing if n != "ILGop_INVALID"] == []


def test_struct_layouts_match_pyvex():
    _, layouts = _vex_engine().native_lift_tables_for_test()
    mismatched = {}
    for name, value in layouts:
        struct, _, field = name.partition(".")
        expected = ffi.offsetof(struct, field) if field else ffi.sizeof(struct)
        if value != expected:
            mismatched[name] = (value, expected)
    assert not mismatched, f"Rust layout != pyvex cffi layout: {mismatched}"


@pytest.mark.parametrize("opt_level", [0, 2])
def test_native_lift_honors_opt_level(opt_level):
    """An opt-level override reaches libVEX the way ``factory.block(opt_level=)`` does."""
    arch = archinfo.ArchAMD64()
    for entry in _corpus()[:10]:
        data = bytes.fromhex(entry["bytes"])
        native = _vex_engine().native_lift_debug_for_test(arch, data, entry["addr"], opt_level)
        irsb = pyvex.lift(data, entry["addr"], arch, opt_level=opt_level, strict_block_end=False, cross_insn_opt=True)
        assert native == _vex_engine().irsb_debug_for_test(irsb)


# nop ; rdmsr ; nop ; jmp $ -- libVEX stops at RDMSR with Ijk_NoDecode and
# pyvex's AMD64Spotter lifts the rest.
SPOTTER_CODE = bytes.fromhex("900f3290ebfe")


def test_native_lift_stops_at_spotter_instruction():
    native = _vex_engine().native_lift_debug_for_test(archinfo.ArchAMD64(), SPOTTER_CODE, 0x400000)
    assert "jumpkind: NoDecode" in native


def _explore(project, use_native_lift, find, max_steps=10, opt_level=None):
    from angr.exploration.rust_manager import RustExplorationManager

    state = project.factory.blank_state(addr=0x400000)
    mgr = RustExplorationManager(project, [state], use_native_lift=use_native_lift)
    mgr.enable_profiling()
    if opt_level is not None:
        mgr._rust_mgr.set_vex_opt_level(opt_level)
    mgr.explore(find=find, max_steps=max_steps)
    return mgr, dict(mgr._rust_mgr.get_execution_stats())


@pytest.mark.parametrize("use_native_lift", [False, True])
def test_spotter_block_runs(use_native_lift):
    """A block pyvex's Spotter finishes lifts through the callback and runs
    (angr-mx00p.43: the native lift used to serve libVEX's NoDecode block;
    angr-mx00p.44: the callback's JSON rejected the Spotter's int U1 guard
    and None Dirty fields)."""
    import angr

    project = angr.load_shellcode(SPOTTER_CODE, arch="amd64", load_address=0x400000)
    mgr, stats = _explore(project, use_native_lift, find=0x400000 + len(SPOTTER_CODE) - 2)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    assert stats.get("native_lift_count", 0) == 0
    if use_native_lift:
        assert stats["native_lift_fallback_count"] >= 1


def test_native_lift_serves_opt_level_override():
    """A global opt-level override no longer forces the callback."""
    import angr

    code = bytes.fromhex("4831c04883c001ebfe")  # xor rax,rax ; add rax,1 ; jmp $
    project = angr.load_shellcode(code, arch="amd64", load_address=0x400000)
    mgr, stats = _explore(project, True, find=0x400007, opt_level=0)
    assert len(mgr.found) == 1
    assert stats["native_lift_count"] >= 1 and stats["native_lift_fallback_count"] == 0


def test_native_lift_off_with_patches():
    """``lift_vex`` lifts patched bytes, which native lift does not read
    (angr-mx00p.46: native lift ran the unpatched instruction)."""
    import angr

    code = bytes.fromhex("4831c04883c001ebfe")
    project = angr.load_shellcode(code, arch="amd64", load_address=0x400000)
    project.kb.patches.add_patch(0x400003, bytes.fromhex("4883c002"))  # add rax,2
    mgr, stats = _explore(project, True, find=0x400007)
    assert stats.get("native_lift_count", 0) == 0
    assert mgr.found[0].solver.eval(mgr.found[0].regs.rax) == 2


def test_native_lift_off_with_postprocessor(monkeypatch):
    """A registered pyvex postprocessor rewrites what the callback lifts."""
    from pyvex.lifting import lift_function

    import angr
    from angr.exploration.rust_manager import _native_lift_settings

    monkeypatch.setitem(lift_function.postprocessors, "AMD64", [object])
    code = bytes.fromhex("4831c0ebfe")
    project = angr.load_shellcode(code, arch="amd64", load_address=0x400000)
    assert _native_lift_settings(project) is None


def test_lift_callback_holds_lift_lock(monkeypatch):
    """The callback's pyvex lift runs under the engine's lift lock, so it
    cannot overlap a native lift on another worker (angr-mx00p.45)."""
    import angr
    from angr.exploration.rust_manager import RustExplorationManager

    held = []
    original = RustExplorationManager._cb_lift_block

    def recording(self, *args, **kwargs):
        held.append(_vex_engine().lift_lock_is_held_for_test())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RustExplorationManager, "_cb_lift_block", recording)
    code = bytes.fromhex("4831c0ebfe")
    project = angr.load_shellcode(code, arch="amd64", load_address=0x400000)
    _explore(project, False, find=0x400003)
    assert held and all(held)
    assert not _vex_engine().lift_lock_is_held_for_test()
