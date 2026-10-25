#!/usr/bin/env python3
"""The Rust ccalls (angr.rustylib.vex_ccall) against ccall.py, which is the oracle.

For every helper Rust serves, each condition and cc_op is run on edge and random concrete operands, and once with
symbolic operands, through both implementations. Rust must refuse wherever ccall.py raises, serve wherever it does
(bar the cc_ops in UNPORTED), and agree with it: equal values for concrete operands, a solver-proved equivalence for
symbolic ones.
"""

# pylint: disable=missing-function-docstring
from __future__ import annotations

import random

import pytest

from angr import claripy, load_shellcode
from angr.engines.vex.claripy import ccall, datalayer
from angr.rustylib import vex_ccall  # pylint:disable=no-name-in-module

# prefix -> (shellcode arch, operand bits, return bits)
ARCHES = {
    "amd64g": ("amd64", 64, 64),
    "x86g": ("x86", 32, 32),
    "armg": ("armel", 32, 32),
    "arm64g": ("aarch64", 64, 64),
}

# amd64 BMI/ADX cc_ops (ANDN32 .. ADOXQ) that only ccall.py implements; Rust refuses them, so they fall back.
UNPORTED = {("amd64g", cc_op) for cc_op in range(53, 65)}

# libVEX feeds some thunk fields only 0/1 (old carry, old overflow, shifter carry-out) or a 2-bit pack (MUL's old
# C:V). ccall.py and Rust disagree outside that contract, so operands are drawn inside it: prefix -> cc_op -> the
# bound of each of (dep1, dep2, dep3), None for a full-width operand.
ARM_LIMITS = {
    "armg": {3: (None, None, 1), 4: (None, None, 1), 5: (None, 1, 1), 6: (None, None, 3), 7: (None, None, 3)},
    "arm64g": {5: (None, None, 1), 6: (None, None, 1), 7: (None, None, 1), 8: (None, None, 1)},
}

STATES = {}


def _state(prefix):
    if prefix not in STATES:
        STATES[prefix] = load_shellcode(b"\x00" * 8, arch=ARCHES[prefix][0]).factory.blank_state()
    return STATES[prefix]


def _selectors(name):
    """The concrete leading arguments: (cond, cc_op) for x86 conditions, (cond << 4 | cc_op,) for ARM ones."""
    prefix = name.split("_", 1)[0]
    if name.endswith("_condition"):
        if prefix in ("armg", "arm64g"):
            return [((cond << 4) | cc_op,) for cond in range(16) for cc_op in range(16)]
        return [(cond, cc_op) for cond in range(18) for cc_op in range(66)]
    return [(cc_op,) for cc_op in range(66)]


def _cc_op(name, selector):
    if name.endswith("_condition"):
        return selector[0] & 0xF if name.startswith("arm") else selector[1]
    return selector[0]


def _limits(prefix, cc_op):
    return ARM_LIMITS.get(prefix, {}).get(cc_op, (None, None, None))


def _operands(rng, bits, limits):
    mask = (1 << bits) - 1
    edges = [0, 1, 2, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    edges += [mask, mask >> 1, (mask >> 1) + 1]
    out = []
    for limit in limits:
        if limit is not None:
            out.append(rng.randint(0, limit))
        elif rng.random() < 0.5:
            out.append(rng.choice(edges) & mask)
        else:
            out.append(rng.getrandbits(bits))
    # Carry, borrow and overflow flip at operands that are equal or adjacent, which independent draws almost never hit.
    if limits[1] is None and rng.random() < 0.5:
        out[1] = (out[0] + rng.choice((-1, 0, 1))) & mask
    return out


def _python(state, name, args):
    try:
        return getattr(ccall, name)(state, *args)
    except Exception:  # pylint:disable=broad-exception-caught
        return None


@pytest.mark.parametrize("name", vex_ccall.SUPPORTED)
def test_rust_ccall_matches_python(name):
    prefix = name.split("_", 1)[0]
    _, bits, ret_bits = ARCHES[prefix]
    state = _state(prefix)
    rng = random.Random(name)
    served = 0
    for selector in _selectors(name):
        cc_op = _cc_op(name, selector)
        limits = _limits(prefix, cc_op)
        head = [claripy.BVV(v, bits) for v in selector]
        cases = [[claripy.BVV(v, bits) for v in _operands(rng, bits, limits)] for _ in range(6)]
        symbols = [claripy.BVS(f"dep{i}", bits) for i in range(3)]
        cases.append(symbols)
        for deps in cases:
            args = (*head, *deps)
            expected = _python(state, name, args)
            actual = vex_ccall.ccall(name, args, ret_bits)
            where = f"{name}{tuple(selector)} deps={deps}"
            if expected is None:
                assert actual is None, f"{where}: ccall.py raises but Rust returned {actual}"
                continue
            if actual is None:
                assert (prefix, cc_op) in UNPORTED, f"{where}: Rust refused a call ccall.py serves"
                continue
            served += 1
            assert actual.size() == expected.size() == ret_bits, where
            solver = claripy.Solver()
            for symbol, limit in zip(symbols, limits):
                if limit is not None:
                    solver.add(claripy.ULE(symbol, limit))
            solver.add(actual != expected)
            assert not solver.satisfiable(), f"{where}: Rust {actual} != ccall.py {expected}"
    assert served


def test_rust_ccall_refuses_unserved_calls():
    bv = claripy.BVV(0, 32)
    # Not a ported helper: it reads the GDT/LDT from memory.
    assert vex_ccall.ccall("x86g_use_seg_selector", (bv, bv, bv, bv), 64) is None
    assert vex_ccall.ccall("no_such_helper", (bv,), 32) is None
    # A symbolic cc_op is ccall.py's to concretize.
    args = (claripy.BVV(4, 64), claripy.BVS("cc_op", 64), *(claripy.BVV(0, 64),) * 3)
    assert vex_ccall.ccall("amd64g_calculate_condition", args, 64) is None
    # ARM32 NV is not a condition a ccall can carry.
    args = (claripy.BVV(0xF2, 32), *(claripy.BVV(0, 32),) * 3)
    assert vex_ccall.ccall("armg_calculate_condition", args, 32) is None


# cmp rdi, rsi ; setb al ; setl cl ; int3
CMP_SETCC = bytes.fromhex("4839f70f92c00f9cc1cc")


@pytest.mark.parametrize("rust", [True, False], ids=["rust", "python"])
def test_engine_dispatches_ccalls(monkeypatch, rust):
    calls = []
    real = vex_ccall.ccall

    def counting(name, args, ret_bits):
        result = real(name, args, ret_bits)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(vex_ccall, "ccall", counting)
    monkeypatch.setattr(datalayer, "RUST_CCALLS", rust)
    project = load_shellcode(CMP_SETCC, arch="amd64")
    state = project.factory.blank_state(add_options={"SYMBOL_FILL_UNCONSTRAINED_REGISTERS"})
    a, b = claripy.BVS("a", 64), claripy.BVS("b", 64)
    state.regs.rdi, state.regs.rsi = a, b
    succ = project.factory.successors(state, num_inst=3, opt_level=0).flat_successors[0]
    below = claripy.ULT(a, b)
    less = claripy.SLT(a, b)
    assert not succ.solver.satisfiable([succ.regs.al[0] != claripy.If(below, claripy.BVV(1, 1), claripy.BVV(0, 1))])
    assert not succ.solver.satisfiable([succ.regs.cl[0] != claripy.If(less, claripy.BVV(1, 1), claripy.BVV(0, 1))])
    if rust:
        assert calls and all(calls), calls
    else:
        assert not calls
