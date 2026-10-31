#!/usr/bin/env python3
"""The Rust integer IROps (angr.rustylib.vex_irop) against irop.py, which is the oracle.

Every op Rust serves runs on edge and random concrete operands at its libVEX signature (pyvex.expr.op_arg_types),
then on symbolic operands, alone and mixed with edge constants, through both implementations. Rust must refuse
wherever irop.py raises, serve wherever it does, and agree with it: the same width and concreteness, equal values for
concrete results, a solver-proved equivalence for symbolic ones, and no variables the arguments do not carry. Width 0
is not a VEX type, so the edges stop at 1 bit.
"""

# pylint: disable=missing-function-docstring
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import pyvex

from angr import claripy, load_shellcode
from angr.engines.vex.claripy import datalayer, irop
from angr.errors import UnsupportedIROpError
from angr.rustylib import vex_irop  # pylint:disable=no-name-in-module

SOLVER_TIMEOUT_MS = 60_000
DIVIDES = ("Iop_Div", "Iop_Mod")
SHIFTS = ("Iop_Shl", "Iop_Shr", "Iop_Sar")


def _signature(name):
    ret, args = pyvex.expr.op_arg_types(name)
    return pyvex.get_type_size(ret), [pyvex.get_type_size(a) for a in args]


def _edges(bits):
    if bits == 1:
        return [0, 1]
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    ladder = [0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    values = [0, 1, 2, half - 1, half, half + 1, mask - 1, mask, *ladder]
    if bits == 128:
        values += [(1 << 64) - 1, 1 << 64]
    return sorted({v & mask for v in values})


def _pool(rng, bits, single_bits=False):
    values = _edges(bits)
    if bits > 1:
        if single_bits:
            values += [1 << i for i in range(bits)]
        values += [rng.getrandbits(bits) for _ in range(8)]
    return list(dict.fromkeys(values))


def _sample(rng, pool, k):
    return rng.sample(pool, min(k, len(pool)))


def _concrete_cases(name, rng, arg_bits):
    if len(arg_bits) == 1:
        return [(v,) for v in _pool(rng, arg_bits[0], single_bits=True)]
    lbits, rbits = arg_bits
    lmask, rmask = (1 << lbits) - 1, (1 << rbits) - 1
    lpool, rpool = _pool(rng, lbits), _pool(rng, rbits)
    if name.startswith(SHIFTS):
        width = lbits
        amounts = [*range(width), width, width + 1, 2 * width, 255]
        return [(v, a & rmask) for v in _sample(rng, lpool, 5) for a in dict.fromkeys(amounts)]
    cases = [(rng.choice(lpool), rng.choice(rpool)) for _ in range(60)]
    cases += [(a, b) for a in _sample(rng, lpool, 6) for b in _sample(rng, rpool, 6)]
    # Compares and borrows flip at equal or adjacent operands, which independent draws almost never hit.
    if lbits == rbits:
        cases += [(a, (a + d) & rmask) for a in _sample(rng, lpool, 6) for d in (-1, 0, 1)]
    if name.startswith(DIVIDES):
        rmin = 1 << (rbits - 1)
        divisors = [0, 1, rmask, rmin, rmin - 1, 3, rmask - 2]
        cases += [(a, d) for a in _sample(rng, lpool, 6) for d in divisors]
        cases += [((1 << (lbits - 1)) & lmask, rmask)]  # MIN / -1
        if lbits > rbits:
            # A dividend whose high half reaches the divisor: the quotient does not fit (cf. angr-vmi2l.58).
            for d in (1, 3, rmin, rmask):
                for hi in (d, d + 1, rmask):
                    cases.append((((hi & rmask) << rbits) | rng.getrandbits(rbits), d))
    return list(dict.fromkeys(cases))


def _symbolic_cases(name, arg_bits):
    syms = [claripy.BVS(f"a{i}", bits) for i, bits in enumerate(arg_bits)]
    if 1 in arg_bits:
        flag = claripy.If(claripy.BoolS("flag"), claripy.BVV(1, 1), claripy.BVV(0, 1))
        yield [flag if bits == 1 else s for s, bits in zip(syms, arg_bits)]
    big_divide = name.startswith(DIVIDES) and max(arg_bits) > 64
    if not big_divide:
        yield syms
    for pos, bits in enumerate(arg_bits):
        if big_divide and pos == 0:
            continue
        values = _edges(bits)[:3] + _edges(bits)[-2:]
        if pos == 1 and name.startswith(SHIFTS):
            values += [arg_bits[0] - 1, arg_bits[0], arg_bits[0] + 1]
        for value in dict.fromkeys(values):
            args = list(syms)
            args[pos] = claripy.BVV(value, bits)
            yield args
    # x^x, x-x, x==x and friends: the same symbol in both positions.
    if len(arg_bits) == 2 and arg_bits[0] == arg_bits[1] and not big_divide:
        yield [syms[0], syms[0]]


def _raised(func, *args):
    """Call func; return (result, None), or (None, the exception's class) if it raises."""
    try:
        return func(*args), None
    except Exception as e:  # pylint:disable=broad-exception-caught
        return None, type(e)


def _python(name, args):
    return _raised(lambda: irop.vexop_to_simop(name).calculate(*args))


def _seam(name, args):
    """The engine's op dispatch, with Rust in front of irop.py."""
    return _raised(lambda: datalayer.ClaripyDataMixin._perform_vex_expr_Op(SimpleNamespace(state=None), name, args))


def _check(name, args, ret_bits):
    """Compare the two implementations on one call; return whether Rust served it."""
    where = f"{name}{tuple(args)}"
    expected, raised = _python(name, args)
    actual = vex_irop.irop(name, tuple(args))
    if raised is not None:
        assert actual is None, f"{where}: irop.py raises but Rust returned {actual}"
        assert _seam(name, args)[1] is raised, f"{where}: the engine does not raise irop.py's {raised.__name__}"
        return False
    if actual is None and name.startswith(SHIFTS) and args[0].symbolic and not args[1].symbolic:
        # irop.py leaves the shift unfolded where Rust would fold it to the fill, so Rust declines.
        assert args[1].concrete_value >= args[0].size(), f"{where}: Rust refused an in-range shift"
        return False
    assert actual is not None, f"{where}: Rust refused a call irop.py serves"
    assert isinstance(actual, claripy.ast.BV) and isinstance(expected, claripy.ast.BV), where
    assert actual.size() == expected.size() == ret_bits, where
    assert actual.symbolic == expected.symbolic, f"{where}: Rust {actual} vs irop.py {expected}"
    if not actual.symbolic:
        assert actual.concrete_value == expected.concrete_value, f"{where}: Rust {actual} != irop.py {expected}"
    elif not actual.identical(expected):
        solver = claripy.Solver(timeout=SOLVER_TIMEOUT_MS)
        solver.add(actual != expected)
        try:
            differs = solver.satisfiable()
        except claripy.ClaripyError as e:
            pytest.fail(f"{where}: the equivalence check did not finish: {e}")
        assert not differs, f"{where}: Rust {actual} != irop.py {expected}"
    allowed = set().union(*(a.variables for a in args))
    assert actual.variables <= allowed, f"{where}: Rust {actual} has variables its arguments do not"
    return True


@pytest.mark.parametrize("name", sorted(datalayer._RUST_IROP_NAMES))
def test_rust_irop_matches_python(name):
    ret_bits, arg_bits = _signature(name)
    rng = random.Random(name)
    served = 0
    for values in _concrete_cases(name, rng, arg_bits):
        served += _check(name, [claripy.BVV(v, b) for v, b in zip(values, arg_bits)], ret_bits)
    for args in _symbolic_cases(name, arg_bits):
        served += _check(name, args, ret_bits)
    assert served


def test_rust_irop_refuses_unserved_calls():
    x32 = claripy.BVS("x", 32)
    # Narrower than the signature: irop.py extends it, Rust leaves it to irop.py.
    narrow = (claripy.BVS("y", 16), x32)
    assert vex_irop.irop("Iop_Add32", narrow) is None
    assert _seam("Iop_Add32", narrow)[0].identical(_python("Iop_Add32", narrow)[0])
    assert vex_irop.irop("Iop_Add32", (claripy.BoolS("b"), x32)) is None
    assert vex_irop.irop("Iop_Add32", (3, x32)) is None
    assert vex_irop.irop("Iop_Add32", (x32,)) is None
    assert vex_irop.irop("Iop_Add32", (x32, x32, x32)) is None
    assert vex_irop.irop("Iop_NoSuchOp", (x32, x32)) is None
    # irop.py raises SimZeroDivisionException for a literal zero divisor, whatever the dividend.
    assert vex_irop.irop("Iop_DivU32", (x32, claripy.BVV(0, 32))) is None
    # irop.py leaves a symbolic value's shift by at least the width unfolded.
    assert vex_irop.irop("Iop_Shl32", (x32, claripy.BVV(32, 8))) is None


def test_rust_irop_exclusions():
    supported = set(vex_irop.SUPPORTED)
    # irop.py's results pass through clarirs' shift folds, which truncate the amount (angr-mx00p.105).
    assert not supported & {"Iop_Shl64", "Iop_Shr64", "Iop_Sar64"}
    # irop.py computes it as an unsigned remainder (angr-mx00p.124).
    assert "Iop_ModS128" not in supported
    # FP-sorted on the Python side.
    assert not {n for n in supported if n.startswith("Iop_Reinterp")}
    assert datalayer._RUST_IROP_NAMES.issubset(irop.operations)


def test_engine_honours_irop_operations_edits(monkeypatch):
    # Rust serves only what irop.py implements at call time, so an op removed from irop.operations stays unsupported.
    monkeypatch.delitem(irop.operations, "Iop_Add32")
    args = (claripy.BVS("x", 32), claripy.BVV(1, 32))
    assert _seam("Iop_Add32", args)[1] is UnsupportedIROpError


class RelocAnnotation(claripy.Annotation):
    """A relocatable annotation, which every node built over the annotated one inherits."""

    relocatable = True
    eliminatable = False

    def __hash__(self):
        return hash(RelocAnnotation)

    def __eq__(self, other):
        return isinstance(other, RelocAnnotation)


def test_rust_irop_keeps_annotations():
    x = claripy.BVS("x", 32).annotate(RelocAnnotation())
    args = (x, claripy.BVV(5, 32))
    expected = _python("Iop_Add32", args)[0]
    actual = vex_irop.irop("Iop_Add32", args)
    assert actual is not None and actual.identical(expected)
    assert set(actual.annotations) == set(expected.annotations) == {RelocAnnotation()}
    # A literal's annotations would not survive the import, so irop.py serves it.
    args = (claripy.BVS("y", 32), claripy.BVV(5, 32).annotate(RelocAnnotation()))
    assert vex_irop.irop("Iop_Add32", args) is None


# mov r8d, edi ; add r8d, esi ; mov r9d, edi ; sub r9d, esi ; mov r10d, edi ; imul r10d, esi ; movzx r11d, dil ;
# movsx r12d, si ; mov r13d, edi ; shr r13d, 3 ; mov eax, edi ; xor edx, edx ; div ecx ; cmp edi, esi ; setb bl ; int3
INT_OPS = bytes.fromhex("4189f84101f04189f94129f14189fa440fafd6440fb6df440fbfe64189fd41c1ed0389f831d2f7f139f70f92c3cc")


@pytest.mark.parametrize("rust", [True, False], ids=["rust", "python"])
def test_engine_dispatches_irops(monkeypatch, rust):
    calls = []
    real = vex_irop.irop

    def counting(name, args):
        result = real(name, args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(vex_irop, "irop", counting)
    monkeypatch.setattr(datalayer, "RUST_IROPS", rust)
    project = load_shellcode(INT_OPS, arch="amd64")
    state = project.factory.blank_state(add_options={"SYMBOL_FILL_UNCONSTRAINED_REGISTERS"})
    a, b = claripy.BVS("a", 64), claripy.BVS("b", 64)
    # A symbolic divisor makes the quotient's equivalence proof too slow for Z3.
    state.regs.rdi, state.regs.rsi, state.regs.rcx = a, b, 7
    succ = project.factory.successors(state, num_inst=15, opt_level=0).flat_successors[0]
    a32, b32, c32 = a[31:0], b[31:0], claripy.BVV(7, 32)
    expected = {
        "r8": claripy.ZeroExt(32, a32 + b32),
        "r9": claripy.ZeroExt(32, a32 - b32),
        "r10": claripy.ZeroExt(32, a32 * b32),
        "r11": claripy.ZeroExt(56, a[7:0]),
        "r12": claripy.ZeroExt(32, claripy.SignExt(16, b[15:0])),
        "r13": claripy.ZeroExt(32, claripy.LShR(a32, 3)),
        "rax": claripy.ZeroExt(32, a32 // c32),
        "rdx": claripy.ZeroExt(32, a32 % c32),
        "bl": claripy.If(claripy.ULT(a32, b32), claripy.BVV(1, 8), claripy.BVV(0, 8)),
    }
    for reg, value in expected.items():
        assert not succ.solver.satisfiable([succ.registers.load(reg) != value]), reg
    if rust:
        assert any(calls), calls
    else:
        assert not calls
