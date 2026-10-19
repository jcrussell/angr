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

from angr.exploration.rust_irsb_serializer import serialize_irsb
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
    return _vex_engine().irsb_json_debug_for_test(serialize_irsb(irsb))


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
