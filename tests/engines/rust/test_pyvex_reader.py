"""Tests for ``native/angr/src/vex/pyvex_reader.rs``, which reads the pyvex
``IRSB`` the ``lift_block`` callback returns into the engine's IR.

Each test builds pyvex objects, reads them through
``vex_engine._testing.irsb_debug_for_test(irsb, pretty=False)`` and checks the Rust IR's
``{:?}`` rendering. Ported from the Rust tests of the JSON bridge the reader
replaced (angr-mx00p.33); the pure string helpers keep Rust tests in
``pyvex_reader_tests.rs``.
"""

from __future__ import annotations

import archinfo
import pytest
import pyvex
from pyvex import const as C
from pyvex import expr as E
from pyvex import stmt as S
from pyvex.enums import IRCallee, IRRegArray

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


def _t(n):
    return E.RdTmp.get_instance(n)


def _read(statements, types=("Ity_I64",) * 8, arch=None, jumpkind="Ijk_Boring"):
    from angr.rustylib import vex_engine

    arch = arch or archinfo.ArchAMD64()
    irsb = pyvex.IRSB.empty_block(
        arch,
        0x1000,
        statements=[S.IMark(0x1000, 4, 0), *statements],
        nxt=E.Const(C.U64(0x1004)),
        tyenv=pyvex.IRTypeEnv(arch, types=list(types)),
        jumpkind=jumpkind,
    )
    return vex_engine._testing.irsb_debug_for_test(irsb, pretty=False)


def _read_expr(expr):
    out = _read([S.WrTmp(0, expr)])
    start = out.index("WrTmp { tmp: 0, data: ") + len("WrTmp { tmp: 0, data: ")
    return out[start:]


def test_lifted_block_reads(fauxware_project):
    from angr.rustylib import vex_engine

    irsb = fauxware_project.factory.block(fauxware_project.entry).vex
    out = vex_engine._testing.irsb_debug_for_test(irsb, pretty=False)
    assert out.startswith(f"IRSB {{ addr: {fauxware_project.entry}, arch: AMD64, statements: [IMark")
    assert "jumpkind:" in out and "tyenv: TypeEnv { types: [" in out


def test_block_fields():
    out = _read([S.Put(E.Const(C.U64(42)), 16)], types=["Ity_I64", "Ity_I32"], jumpkind="Ijk_Call")
    assert "Put { offset: 16, data: Const(U64(42)) }" in out
    assert "next: Const(U64(4100)), jumpkind: Call, offsIP: 184" in out
    assert out.endswith("tyenv: TypeEnv { types: [I64, I32] } }")


def test_binop_get_const():
    assert _read_expr(E.Binop("Iop_Add64", [E.Get(16, "Ity_I64"), E.Const(C.U64(8))])).startswith(
        "Binop { op: Add(I64), left: Get { offset: 16, ty: I64 }, right: Const(U64(8)) }"
    )


def test_exit_takes_constant_destination():
    out = _read([S.Exit(_t(0), C.U64(0x2000), "Ijk_Boring", 184)])
    assert "Exit { guard: RdTmp(0), dst: 8192, jk: Boring, offsIP: 184 }" in out


def test_store_endness_and_unknown_endness_fallback():
    assert "Store { addr: RdTmp(3), data: RdTmp(4), endness: Big }" in _read([S.Store(_t(3), _t(4), "Iend_BE")])
    # parse_endness's logged little-endian fallback, reached from a Store.
    assert "endness: Little" in _read([S.Store(_t(3), _t(4), "Iend_ME")])


def test_storeg_and_loadg():
    out = _read([S.StoreG("Iend_LE", _t(0), _t(1), _t(2))])
    assert "StoreG { addr: RdTmp(0), data: RdTmp(1), guard: RdTmp(2), endness: Little }" in out
    out = _read([S.LoadG("Iend_LE", "ILGop_16Sto32", 4, _t(0), _t(1), _t(2))])
    assert "LoadG { dst: 4, addr: RdTmp(0), alt: RdTmp(1), guard: RdTmp(2), cvt: WidenS { src_bits: 16 }" in out


def test_cas_single_width_maps_invalid_old_hi_to_none():
    out = _read([S.CAS(_t(0), _t(1), None, _t(2), None, 3, 0xFFFFFFFF, "Iend_LE")])
    assert "CAS { old_hi: None, old_lo: 3, addr: RdTmp(0), expdHi: None, expdLo: RdTmp(2), dataHi: None" in out


def test_cas_double_width():
    out = _read([S.CAS(_t(0), _t(1), _t(5), _t(2), _t(6), 3, 4, "Iend_LE")])
    assert "CAS { old_hi: Some(4), old_lo: 3, addr: RdTmp(0), expdHi: Some(RdTmp(6))" in out
    assert "dataHi: Some(RdTmp(5)), dataLo: RdTmp(1)" in out


def test_llsc_load_linked_and_store_conditional():
    assert "LLSC { storedata: None, result: 2, addr: RdTmp(0), endness: Little }" in _read(
        [S.LLSC(_t(0), None, 2, "Iend_LE")]
    )
    assert "LLSC { storedata: Some(RdTmp(1)), result: 2" in _read([S.LLSC(_t(0), _t(1), 2, "Iend_LE")])


def test_mbe_reads_its_event():
    """The JSON bridge's serializer dropped MBE's event, which its schema
    required, so every fence block failed to deserialize."""
    assert "MBE(Fence)" in _read([S.MBE("Imbe_Fence")])


def test_dirty_with_result_tmp_and_memory_effect():
    cee = IRCallee(0, "amd64g_dirtyhelper_CPUID", 0)
    out = _read([S.Dirty(cee, E.Const(C.U1(1)), [_t(0)], 3, "Ifx_Read", _t(1), 8, 0)])
    assert (
        'Dirty(IRDirty { cee: IRCallee { name: "amd64g_dirtyhelper_CPUID", addr: 0, mcx_mask: 0 }, '
        "guard: Some(Const(U1(true))), tmp: Some(3), mFx: Read, mAddr: Some(RdTmp(1)), mSize: 8" in out
    )


def test_python_built_dirty_reads():
    """A Spotter's Dirty: an int U1 guard, no result temp (IRTemp_INVALID) and
    None for the fields libVEX would write as 0 (angr-mx00p.44)."""
    cee = IRCallee(0, "helper", 0)
    out = _read([S.Dirty(cee, E.Const(C.U1(1)), [], 0xFFFFFFFF, None, None, None, None)])
    assert "guard: Some(Const(U1(true))), tmp: None, mFx: None, mAddr: None, mSize: 0, nFxState: 0" in out


def test_spotter_block_reads(fauxware_project):
    import angr
    from angr.rustylib import vex_engine

    project = angr.load_shellcode(bytes.fromhex("900f3290ebfe"), arch="amd64", load_address=0x400000)
    irsb = project.factory.block(0x400000).vex
    assert "Dirty(IRDirty" in vex_engine._testing.irsb_debug_for_test(irsb, pretty=False)


def test_puti_and_geti_descr():
    descr = IRRegArray(136, "Ity_F64", 8)
    assert (
        "PutI { descr: IRRegArray { base: 136, elemTy: F64, nElems: 8 }, ix: RdTmp(0), bias: 7, data: RdTmp(1) }"
        in (_read([S.PutI(descr, _t(0), _t(1), 7)]))
    )
    assert _read_expr(E.GetI(descr, _t(0), 1)).startswith(
        "GetI { descr: IRRegArray { base: 136, elemTy: F64, nElems: 8 }, ix: RdTmp(0), bias: 1 }"
    )
    # A negative bias wraps the way the native lifter's `as u32` does.
    assert "bias: 4294967295" in _read([S.PutI(descr, _t(0), _t(1), -1)])


def test_abihint_and_noop():
    out = _read([S.AbiHint(_t(0), 128, _t(1)), S.NoOp()])
    assert "AbiHint { base: RdTmp(0), len: 128, nia: RdTmp(1) }, NoOp" in out


def test_load_unop_triop_qop_ite():
    assert _read_expr(E.Load("Iend_BE", "Ity_I32", _t(1))).startswith("Load { addr: RdTmp(1), ty: I32, endness: Big }")
    assert _read_expr(E.Unop("Iop_Not64", [_t(1)])).startswith("Unop { op: Not(I64), arg: RdTmp(1) }")
    assert _read_expr(E.Triop("Iop_AddF64", [_t(1), _t(2), _t(3)])).startswith(
        "Triop { op: "
    ) and "arg1: RdTmp(1), arg2: RdTmp(2), arg3: RdTmp(3) }" in _read_expr(E.Triop("Iop_AddF64", [_t(1), _t(2), _t(3)]))
    assert "arg1: RdTmp(1), arg2: RdTmp(2), arg3: RdTmp(3), arg4: RdTmp(4) }" in _read_expr(
        E.Qop("Iop_64x4toV256", [_t(1), _t(2), _t(3), _t(4)])
    )
    # pyvex's ITE takes (cond, iffalse, iftrue); the arms must not swap.
    assert _read_expr(E.ITE(_t(1), _t(2), _t(3))).startswith(
        "ITE { cond: RdTmp(1), iftrue: RdTmp(3), iffalse: RdTmp(2) }"
    )


def test_ccall():
    out = _read_expr(E.CCall("Ity_I64", IRCallee(0, "amd64g_calculate_condition", 0), [_t(1), _t(2)]))
    assert out.startswith(
        'CCall { cee: IRCallee { name: "amd64g_calculate_condition", addr: 0, mcx_mask: 0 }, retty: I64, '
        "args: [RdTmp(1), RdTmp(2)] }"
    )


def test_vecret_and_gsptr():
    assert _read_expr(E.VECRET()).startswith("VECRET")
    assert _read_expr(E.GSPTR()).startswith("GSPTR")


@pytest.mark.parametrize(
    ("con", "rendered"),
    [
        (C.U1(1), "U1(true)"),
        (C.U8(0x7A), "U8(122)"),
        (C.U16(0x7AAB), "U16(31403)"),
        (C.U32(0x7AABCDEE), "U32(2058079726)"),
        (C.U64(0x7AABCDEEF0011223), f"U64({0x7AABCDEEF0011223})"),
        (C.F32(1.5), "F32(1.5)"),
        (C.F64(1.5), "F64(1.5)"),
        (C.V128((0xFF << 120) | 0xFF), f"V128({(0xFF << 120) | 0xFF})"),
        (C.V256(1 << 64 | 2), "V256([2, 1, 0, 0])"),
    ],
)
def test_constants(con, rendered):
    assert _read_expr(E.Const(con)).startswith(f"Const({rendered})")


def test_unknown_class_is_an_error():
    class Bogus:
        pass

    with pytest.raises(ValueError, match=r"unknown pyvex statement class .Bogus."):
        _read([Bogus()])
