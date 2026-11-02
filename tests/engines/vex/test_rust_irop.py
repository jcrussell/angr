#!/usr/bin/env python3
"""The Rust integer IROps (angr.rustylib.vex_irop) against irop.py, which is the oracle.

Every op Rust serves runs on edge and random concrete operands at its libVEX signature (pyvex.expr.op_arg_types),
then on symbolic operands, alone and mixed with edge constants, through both implementations. A vector operand is
built lane by lane from the lane width's edges, and a shift amount or lane index walks past the lane width or count. Rust must refuse
wherever irop.py raises, serve wherever it does, and agree with it: the same width and concreteness, equal values for
concrete results, a solver-proved equivalence for symbolic ones, and no variables the arguments do not carry. Width 0
is not a VEX type, so the edges stop at 1 bit.
"""

# pylint: disable=missing-function-docstring
from __future__ import annotations

import math
import random
import struct
from types import SimpleNamespace

import pytest
import pyvex

from angr import claripy, load_shellcode
from angr.engines.vex.claripy import datalayer, irop
from angr.errors import UnsupportedIROpError
from angr.rustylib import vex_irop  # pylint:disable=no-name-in-module
from angr.rustylib.claripy.annotation import UninitializedAnnotation  # pylint:disable=no-name-in-module

SOLVER_TIMEOUT_MS = 60_000
DIVIDES = ("Iop_Div", "Iop_Mod")
SHIFTS = ("Iop_Shl", "Iop_Shr", "Iop_Sar")
MULS = ("Iop_Mul", "Iop_MullS", "Iop_MullU")


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


def _lanes(name):
    """The lane width and count of a vector op, from irop.py's parse of its name; None for a scalar op."""
    op = irop.operations[name]
    if op._vector_size is None:
        return None
    return op._vector_size, op._vector_count


def _arg_lane(name, bits):
    """The lane width of a vector operand of `bits` bits, or None if the operand is a scalar."""
    lanes = _lanes(name)
    if lanes is None:
        return None
    lane = irop.operations[name]._from_size or lanes[0]
    return lane if bits % lane == 0 and bits > lane else None


def _join(lane, values):
    """Pack lane values, lane 0 lowest, into one integer."""
    return sum((v & ((1 << lane) - 1)) << (i * lane) for i, v in enumerate(values))


# Bit patterns of float edges: +-0, +-1, 0.5, +-inf, quiet and signalling NaNs (both signs), the smallest and largest
# subnormal, the smallest normal, the largest finite value, and values that round differently by mode.
FLOAT_EDGES = {
    32: [
        0,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x3F000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFC00000,
        0x7FA00000,
        0x7FC00001,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0x3EAAAAAB,
        0x4B800001,
        0xCF000000,
    ],
    64: [
        0,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x3FE0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        0xFFF8000000000000,
        0x7FF4000000000000,
        0x7FF8000000000001,
        0x0000000000000001,
        0x000FFFFFFFFFFFFF,
        0x0010000000000000,
        0x7FEFFFFFFFFFFFFF,
        0x3FD5555555555555,
        0x4340000000000001,
        0xC1E0000000000000,
    ],
}
# libVEX's IRRoundingMode values: nearest-even, -inf, +inf, zero, nearest-away.
ROUNDING_MODES = [0, 1, 2, 3, 4]


def _float_lanes(name):
    op = irop.operations[name]
    return op._float and op._vector_size in FLOAT_EDGES


def _takes_rm(name):
    """A vector float op whose first operand is the VEX rounding mode."""
    _, args = pyvex.expr.op_arg_types(name)
    return _float_lanes(name) and len(args) > 1 and args[0] == "Ity_I32"


def _scalar_float(name):
    """A scalar float op: its operands after any rounding mode are FP-sorted, and so is its result unless it is
    Iop_CmpF*'s condition code."""
    op = irop.operations[name]
    return op._float and op._vector_size is None


def _scalar_rm(name):
    """Whether a scalar float op's first operand is the VEX rounding mode (Iop_{Abs,Neg,Cmp}F*, Iop_F32toF64 and
    Iop_I32{S,U}toF64 take none)."""
    args = pyvex.expr.op_arg_types(name)[1]
    return len(args) > 1 and args[0] == "Ity_I32"


def _float_operands(name):
    """Whether each operand after any rounding mode is a float (an int-to-float conversion's is an integer)."""
    args = pyvex.expr.op_arg_types(name)[1]
    return [a.startswith("Ity_F") for a in args[1 if _scalar_rm(name) else 0 :]]


def _int_to_float_edges(bits):
    """The integer edges plus the integers around 2^24 and 2^53, where F32 and F64 stop being exact."""
    near = [(1 << p) + d for p in (24, 53) for d in (-1, 0, 1, 2, 3) if p < bits]
    return sorted(set(_edges(bits)) | {v & ((1 << bits) - 1) for v in near} | {-v & ((1 << bits) - 1) for v in near})


def _operand_pool(rng, bits, is_float):
    edges = FLOAT_EDGES[bits] if is_float else _int_to_float_edges(bits)
    return edges + [rng.getrandbits(bits) for _ in range(4)]


def _operand(value, bits, is_float, as_fpv):
    return _float_arg(value, bits, as_fpv) if is_float else claripy.BVV(value, bits)


def _float_arg(value, bits, as_fpv):
    """The float with IEEE bits `value`, as claripy's raw_to_fp (bvToFP) or as an FPV."""
    if not as_fpv:
        return claripy.BVV(value, bits).raw_to_fp()
    fmt = ">I" if bits == 32 else ">Q"
    unpacked = struct.unpack(">f" if bits == 32 else ">d", struct.pack(fmt, value))[0]
    return claripy.FPV(unpacked, claripy.FSORT_FLOAT if bits == 32 else claripy.FSORT_DOUBLE)


def _scalar_float_cases(name, rng, arg_bits):
    """Any rounding mode, then the operands, each an edge or random bits, a float alternately bvToFP and FPV."""
    rm = _scalar_rm(name)
    float_bits = arg_bits[1:] if rm else arg_bits
    floats = _float_operands(name)
    modes = ROUNDING_MODES if rm else [None]
    pools = [modes] + [_operand_pool(rng, b, f) for b, f in zip(float_bits, floats)]
    values = [tuple(rng.choice(p) for p in pools) for _ in range(60)]
    if len(pools) == 3:
        values += [(m, a, b) for m in modes for a in _sample(rng, pools[1], 5) for b in _sample(rng, pools[2], 5)]
        # 1.0 and the next float up round apart by mode.
        one = 0x3F800000 if float_bits[0] == 32 else 0x3FF0000000000000
        values += [(m, one, one + 1) for m in modes]
        # Each edge against itself and its negation: +0 == -0, a NaN is unordered even with itself.
        sign = 1 << (float_bits[0] - 1)
        values += [(m, e, f) for m in modes for e in FLOAT_EDGES[float_bits[0]] for f in (e, e ^ sign)]
    elif len(pools) == 4:
        values += [
            (m, a, b, c)
            for m in modes
            for a in _sample(rng, pools[1], 3)
            for b in _sample(rng, pools[2], 3)
            for c in _sample(rng, pools[3], 3)
        ]
        values += [(m, *triple) for m in modes for triple in _fused_triples(float_bits[0])]
    else:
        values += [(m, a) for m in modes for a in pools[1]]
    for i, case in enumerate(dict.fromkeys(values)):
        args = [claripy.BVV(case[0], arg_bits[0])] if rm else []
        args += [
            _operand(v, b, f, (i + pos) % 2 == 0) for pos, (v, b, f) in enumerate(zip(case[1:], float_bits, floats))
        ]
        yield args


def _fused_triples(bits):
    """Multiply-add operands whose fused and unfused results differ: (1 + u)(1 - u) +- 1 is -+u^2 exactly but 0
    rounded per step, and 1 * 1 +- u^2 rounds away from 1 only under a directed mode (u is one ulp of 1). A NaN addend
    of either sign: x86 vfmsub returns it unnegated, irop.py's fpFMA negates it first."""
    mant = 23 if bits == 32 else 52
    exp = 8 if bits == 32 else 11
    one = ((1 << (exp - 1)) - 1) << mant
    sign = 1 << (bits - 1)
    tiny = ((1 << (exp - 1)) - 1 - 2 * mant) << mant  # u^2
    below = (((1 << (exp - 1)) - 2) << mant) | ((1 << mant) - 2)  # 1 - u
    nan = (((1 << exp) - 1) << mant) | (1 << (mant - 1))
    return [
        *((one + 1, below, c) for c in (one, one | sign)),
        *((one, one, c) for c in (tiny, tiny | sign)),
        *((one, one, c) for c in (nan, nan | sign)),
    ]


def _scalar_float_symbolic_cases(name, arg_bits):
    rm = _scalar_rm(name)
    float_bits = arg_bits[1:] if rm else arg_bits
    modes = [claripy.BVV(m, arg_bits[0]) for m in ROUNDING_MODES] if rm else []
    if not all(_float_operands(name)):
        # An int-to-float conversion of a symbolic integer, under each mode (a symbolic mode is pinned separately).
        for mode in modes or [None]:
            yield [*([mode] if rm else []), claripy.BVS("i", float_bits[0])]
        return
    sort = claripy.FSORT_FLOAT if float_bits[0] == 32 else claripy.FSORT_DOUBLE
    for mode in modes or [None]:
        lead = [mode] if rm else []
        syms = [claripy.FPS(f"f{i}", sort) for i in range(len(float_bits))]
        yield [*lead, *syms]
        # A raw_to_fp of a bitvector symbol, as the Python engine reads a float register.
        yield [*lead, *(claripy.BVS(f"b{i}", float_bits[0]).raw_to_fp() for i in range(len(float_bits)))]
        for pos in range(len(syms)):
            for value in FLOAT_EDGES[float_bits[0]][::3]:
                args = list(syms)
                args[pos] = _float_arg(value, float_bits[0], value % 2 == 0)
                yield [*lead, *args]
        if len(syms) == 2:
            yield [*lead, syms[0], syms[0]]


def _past_int_range(name, arg):
    """Whether a scalar float-to-int conversion's literal operand is infinite, NaN, or rounds outside the target
    range under some mode, where libVEX leaves the result undefined."""
    op = irop.operations[name]
    if arg.symbolic or op._from_type != "F" or op._to_type != "I" or op._vector_size is not None:
        return False
    value = arg.concrete_value
    if not math.isfinite(value):
        return True
    lo, end = (-(1 << (op._to_size - 1)), 1 << (op._to_size - 1)) if op._to_signed == "S" else (0, 1 << op._to_size)
    return math.floor(value) < lo or math.ceil(value) >= end


def _vector_pool(rng, bits, lane, edges=None):
    count = bits // lane
    edges = edges or _edges(lane)
    values = [_join(lane, [e] * count) for e in edges]
    values += [_join(lane, [rng.choice(edges) for _ in range(count)]) for _ in range(10)]
    values += [rng.getrandbits(bits) for _ in range(4)]
    return list(dict.fromkeys(values))


def _split(lane, value, count):
    return [(value >> (i * lane)) & ((1 << lane) - 1) for i in range(count)]


def _vector_cases(name, rng, arg_bits):
    lane, count = _lanes(name)
    generic = irop.operations[name]._generic_name
    pools = []
    for pos, bits in enumerate(arg_bits):
        arg_lane = _arg_lane(name, bits)
        if pos == 0 and _takes_rm(name):
            pools.append(ROUNDING_MODES)
        elif arg_lane is not None and _float_lanes(name) and arg_lane in FLOAT_EDGES:
            pools.append(_vector_pool(rng, bits, arg_lane, FLOAT_EDGES[arg_lane]))
        elif arg_lane is not None:
            pools.append(_vector_pool(rng, bits, arg_lane))
        elif generic in ("ShlN", "ShrN", "SarN") and pos == 1:
            pools.append([a & ((1 << bits) - 1) for a in (*range(lane), lane, lane + 1, 2 * lane, 255)])
        elif generic in ("GetElem", "SetElem") and pos == 1:
            pools.append([i & ((1 << bits) - 1) for i in (*range(count), count, 255)])
        else:
            pools.append(_pool(rng, bits))
    if generic in ("Shl", "Shr", "Sar") and len(arg_bits) == 2:
        # A lane-wise shift: each amount lane walks past the lane width.
        amounts = [*range(lane), lane, lane + 1, 2 * lane, (1 << lane) - 1]
        pools[1] = [_join(lane, [rng.choice(amounts) for _ in range(count)]) for _ in range(12)]
        pools[1] += [_join(lane, [a] * count) for a in amounts]
    if len(pools) == 1:
        return [(v,) for v in pools[0]]
    cases = [tuple(rng.choice(p) for p in pools) for _ in range(60)]
    if len(pools) == 2:
        cases += [(a, b) for a in _sample(rng, pools[0], 6) for b in _sample(rng, pools[1], 6)]
        lbits, rbits = arg_bits
        arg_lane = _arg_lane(name, lbits)
        if lbits == rbits and arg_lane is not None and generic not in ("Shl", "Shr", "Sar"):
            # Compares, saturation and borrows flip at equal or adjacent lanes.
            n = lbits // arg_lane
            for a in _sample(rng, pools[0], 6):
                for d in (-1, 0, 1):
                    cases.append((a, _join(arg_lane, [x + d for x in _split(arg_lane, a, n)])))
    return list(dict.fromkeys(cases))


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
    if _lanes(name) is not None:
        return _vector_cases(name, rng, arg_bits)
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
    if _takes_rm(name):
        # The concrete cases cover every mode; a symbolic one makes the float equivalence proofs slow.
        syms[0] = claripy.BVV(0, arg_bits[0])
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
        arg_lane = _arg_lane(name, bits)
        if arg_lane is not None:
            values += [_join(arg_lane, [e] * (bits // arg_lane)) for e in _edges(arg_lane)]
        if pos == 1 and name.startswith(SHIFTS):
            width = _lanes(name)[0] if _lanes(name) else arg_bits[0]
            values += [width - 1, width, width + 1]
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


def _symbolic_times_power_of_two(args):
    """Whether one of a multiply's operands is symbolic and the other a literal power of two."""
    a, b = args
    return any(x.symbolic and not y.symbolic and _is_power_of_two(y.concrete_value) for x, y in ((a, b), (b, a)))


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


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
        # A scalar shift by any literal: Rust rewrites it to Concat/Extract, irop.py keeps the shift (angr-mx00p.100.12).
        lanes = _lanes(name)
        if lanes is not None:
            lane = lanes[0]
            amounts = [args[1].concrete_value]
            if args[1].size() > 8:  # a lane-wise shift
                amounts = _split(lane, args[1].concrete_value, args[1].size() // lane)
            assert max(amounts) >= lane, f"{where}: Rust refused an in-range shift"
        return False
    if actual is None and name.startswith(MULS) and _symbolic_times_power_of_two(args):
        # Rust rewrites it to a shift and then Concat/Extract; irop.py keeps the Mul (angr-mx00p.100.12).
        return False
    if actual is None and irop.operations[name]._generic_name in ("GetElem", "SetElem") and not args[1].symbolic:
        # libVEX leaves an index past the lane count undefined; irop.py's reading of it stands.
        assert args[1].concrete_value >= _lanes(name)[1], f"{where}: Rust refused an in-range index"
        return False
    if actual is None and irop.operations[name]._generic_name == "Cmp" and expected.symbolic:
        # Rust folds a NaN literal against a symbol to unordered where irop.py keeps the ite chain, so Rust declines.
        assert any(not a.symbolic and claripy.fpIsNaN(a).is_true() for a in args), (
            f"{where}: Rust refused a compare without a NaN literal"
        )
        return False
    if actual is None and _past_int_range(name, args[-1]):
        # libVEX leaves an out-of-range, infinite or NaN conversion undefined; irop.py's reading of it stands.
        return False
    assert actual is not None, f"{where}: Rust refused a call irop.py serves"
    if isinstance(expected, claripy.ast.FP):
        return _check_float(where, args, actual, expected, ret_bits)
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


def _check_float(where, args, actual, expected, ret_bits):
    """_check for an FP-sorted result: concrete results agree on their IEEE bits, symbolic ones are equal or both NaN
    for every input."""
    assert isinstance(actual, claripy.ast.FP) and isinstance(expected, claripy.ast.FP), where
    assert actual.size() == expected.size() == ret_bits, where
    assert actual.symbolic == expected.symbolic, f"{where}: Rust {actual} vs irop.py {expected}"
    if not actual.symbolic:
        assert actual.raw_to_bv().concrete_value == expected.raw_to_bv().concrete_value, (
            f"{where}: Rust {actual} != irop.py {expected}"
        )
    elif not actual.identical(expected):
        solver = claripy.Solver(timeout=SOLVER_TIMEOUT_MS)
        same = claripy.Or(
            claripy.fpEQ(actual, expected), claripy.And(claripy.fpIsNaN(actual), claripy.fpIsNaN(expected))
        )
        solver.add(claripy.Not(same))
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
    if _scalar_float(name):
        for args in _scalar_float_cases(name, rng, arg_bits):
            served += _check(name, args, ret_bits)
        for args in _scalar_float_symbolic_cases(name, arg_bits):
            served += _check(name, args, ret_bits)
        assert served
        return
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
    # The same per lane, for the vector shifts; libVEX leaves both undefined.
    v128 = claripy.BVS("v", 128)
    assert vex_irop.irop("Iop_ShlN32x4", (v128, claripy.BVV(32, 8))) is None
    assert vex_irop.irop("Iop_Shr32x4", (v128, claripy.BVV(32 << 64, 128))) is None
    assert vex_irop.irop("Iop_ShlN32x4", (v128, claripy.BVV(31, 8))) is not None
    # libVEX leaves a lane index past the lane count undefined: irop.py reads lane 0 or leaves the vector as is.
    assert vex_irop.irop("Iop_GetElem32x4", (v128, claripy.BVV(4, 8))) is None
    assert vex_irop.irop("Iop_SetElem32x4", (v128, claripy.BVV(4, 8), x32)) is None
    assert vex_irop.irop("Iop_SetElem32x4", (v128, claripy.BVV(3, 8), x32)) is not None


@pytest.mark.parametrize(
    "name, amount",
    [
        ("Iop_Shl64", 3),
        ("Iop_Shr64", 61),
        ("Iop_Sar64", 3),
        ("Iop_Shl32", 0),
        ("Iop_Shr32", 5),
        ("Iop_Mul64", 8),
        ("Iop_Mul32", 1),
        ("Iop_MullU32", 4),
        ("Iop_MullS64", 2),
    ],
)
def test_rust_irop_keeps_a_literal_shift_s_shape(name, amount):
    """A symbolic value shifted by a literal, or multiplied by a literal power of two, keeps irop.py's term: Rust's
    Concat/Extract form is equal as a bitvector, but the jump-table resolver's VSA loses the table bound on it
    (angr-mx00p.100.12)."""
    _, (bits, amount_bits) = _signature(name)
    x = claripy.BVS("x", bits)
    literal = claripy.BVV(amount, amount_bits)
    calls = [(x, literal)]
    if name.startswith(MULS):
        calls.append((literal, x))
    for args in calls:
        assert vex_irop.irop(name, args) is None
        assert _seam(name, args)[0].identical(_python(name, args)[0])
    # A product by a non-power of two has no shift form, so Rust serves it.
    if name.startswith(MULS):
        assert vex_irop.irop(name, (x, claripy.BVV(6, amount_bits))) is not None


def test_rust_irop_exclusions():
    supported = set(vex_irop.SUPPORTED)
    # FP-sorted on the Python side.
    assert not {n for n in supported if n.startswith("Iop_Reinterp")}
    # irop.py builds the estimates from fresh symbols.
    assert not {n for n in supported if "Est" in n or "Step" in n}
    assert {"Iop_Add8x16", "Iop_QNarrowBin16Sto8Ux16", "Iop_SetElem32x4"} <= supported
    assert {"Iop_Add64F0x2", "Iop_Add32Fx4", "Iop_Mul64F0x2", "Iop_CmpLT64F0x2"} <= supported
    # clarirs folds fpSub of a NaN with the NaN's sign flipped (angr-mx00p.127).
    assert not {n for n in supported if n.startswith("Iop_Sub") and "F" in n[7:]}
    # irop.py's float Min/Max match neither x86 nor AArch64 on +-0 and NaN (angr-mx00p.128).
    assert not {n for n in supported if n.startswith(("Iop_Min", "Iop_Max", "Iop_PwMin", "Iop_PwMax")) and "F" in n}
    # libVEX leaves an out-of-range lane undefined, and Rust and irop.py fill it differently (angr-mx00p.129).
    assert not {n for n in supported if n.startswith("Iop_F32toI32") and "x" in n[12:]}
    assert datalayer._RUST_IROP_NAMES.issubset(irop.operations)


UNMAPPED = "not mapped in vex/opcode_map"
# angr-mx00p.101.16: angr-bv's float row is F32 and F64, and Rust has no stable
# f16/f128 for the concrete folds, so a half or quad lane or operand stays in irop.py.
F16_F128 = "stays: angr-bv's float row has no F16 or F128"
# Every irop.py op Rust does not serve, with the reason; test_rust_irop_unserved_is_pinned keeps it exact.
UNSERVED = {
    # Excluded from SERVED in vex_irop.rs.
    **dict.fromkeys(
        ("Iop_ReinterpF32asI32", "Iop_ReinterpF64asI64", "Iop_ReinterpI32asF32", "Iop_ReinterpI64asF64"),
        "excluded: FP-sorted on the Python side",
    ),
    **dict.fromkeys(
        (
            "Iop_SubF32",
            "Iop_SubF64",
            "Iop_Sub32F0x4",
            "Iop_Sub64F0x2",
            "Iop_Sub32Fx2",
            "Iop_Sub32Fx4",
            "Iop_Sub32Fx8",
            "Iop_Sub64Fx2",
            "Iop_Sub64Fx4",
        ),
        "excluded: clarirs fpSub NaN sign",
    ),
    **dict.fromkeys(
        (
            "Iop_Max32F0x4",
            "Iop_Max64F0x2",
            "Iop_Min32F0x4",
            "Iop_Min64F0x2",
            "Iop_Max32Fx2",
            "Iop_Max32Fx4",
            "Iop_Max32Fx8",
            "Iop_Max64Fx2",
            "Iop_Max64Fx4",
            "Iop_Min32Fx2",
            "Iop_Min32Fx4",
            "Iop_Min32Fx8",
            "Iop_Min64Fx2",
            "Iop_Min64Fx4",
        ),
        "excluded: Min/Max per-guest semantics",
    ),
    **dict.fromkeys(
        (
            "Iop_F32toI32Sx2_RZ",
            "Iop_F32toI32Sx4",
            "Iop_F32toI32Sx4_RZ",
            "Iop_F32toI32Sx8",
            "Iop_F32toI32Ux2_RZ",
            "Iop_F32toI32Ux4_RZ",
        ),
        "excluded: libVEX-undefined out-of-range lane",
    ),
    # Mapped, but in a family vex_irop does not serve.
    **dict.fromkeys(
        ("Iop_RSqrtEst32F0x4", "Iop_RSqrtEst32Fx2", "Iop_RSqrtEst32Fx4", "Iop_RSqrtEst32Fx8", "Iop_RSqrtEst64Fx2"),
        "stays: irop.py builds the estimate from fresh symbols",
    ),
    # Scalar integer.
    **dict.fromkeys(
        ("Iop_Max32U", "Iop_QAdd32S", "Iop_QSub32S"),
        "stays: irop.py's Max/QAdd/QSub are lane-only and raise on the scalar form",
    ),
    **dict.fromkeys(
        (
            "Iop_CmpNEZ8",
            "Iop_CmpNEZ16",
            "Iop_CmpNEZ32",
            "Iop_CmpNEZ64",
        ),
        UNMAPPED,
    ),
    # Scalar F16 and F128.
    **dict.fromkeys(
        (
            "Iop_AbsF16",
            "Iop_AddF16",
            "Iop_SubF16",
            "Iop_NegF16",
            "Iop_CmpF16",
            "Iop_F16toF32",
            "Iop_F16toF64",
            "Iop_F32toF16",
            "Iop_F64toF16",
            "Iop_AbsF128",
            "Iop_AddF128",
            "Iop_SubF128",
            "Iop_MulF128",
            "Iop_DivF128",
            "Iop_NegF128",
            "Iop_CmpF128",
            "Iop_RoundF128toInt",
            "Iop_F128HItoF64",
            "Iop_F128LOtoF64",
            "Iop_F64HLtoF128",
            "Iop_F128toF32",
            "Iop_F128toF64",
            "Iop_F32toF128",
            "Iop_F64toF128",
            "Iop_F128toI32S",
            "Iop_F128toI32U",
            "Iop_F128toI64S",
            "Iop_F128toI64U",
            "Iop_F128toI128S",
            "Iop_I32StoF128",
            "Iop_I32UtoF128",
            "Iop_I64StoF128",
            "Iop_I64UtoF128",
            "Iop_I128StoF128",
            "Iop_I128UtoF128",
            "Iop_ReinterpF128asI128",
            "Iop_ReinterpI128asF128",
        ),
        F16_F128,
    ),
    # Vector float.
    **dict.fromkeys(
        (
            "Iop_Abs16Fx8",
            "Iop_Add16Fx8",
            "Iop_Sub16Fx8",
            "Iop_Neg16Fx8",
            "Iop_CmpEQ16Fx8",
            "Iop_CmpLE16Fx8",
            "Iop_CmpLT16Fx8",
            "Iop_F16toF32x4",
            "Iop_F16toF32x8",
            "Iop_F16toF64x2",
            "Iop_F32toF16x4",
            "Iop_F32toF16x4_DEP",
            "Iop_F32toF16x8",
            "Iop_F64toF16x2_DEP",
        ),
        F16_F128,
    ),
    # Vector integer lanes.
    **dict.fromkeys(
        (
            "Iop_CatEvenLanes8x8",
            "Iop_CatEvenLanes8x16",
            "Iop_CatEvenLanes16x4",
            "Iop_CatEvenLanes16x8",
            "Iop_CatEvenLanes32x4",
            "Iop_CatOddLanes8x8",
            "Iop_CatOddLanes8x16",
            "Iop_CatOddLanes16x4",
            "Iop_CatOddLanes16x8",
            "Iop_CatOddLanes32x4",
            "Iop_CmpNEZ8x4",
            "Iop_CmpNEZ8x8",
            "Iop_CmpNEZ8x16",
            "Iop_CmpNEZ8x32",
            "Iop_CmpNEZ16x2",
            "Iop_CmpNEZ16x4",
            "Iop_CmpNEZ16x8",
            "Iop_CmpNEZ16x16",
            "Iop_CmpNEZ32x2",
            "Iop_CmpNEZ32x4",
            "Iop_CmpNEZ32x8",
            "Iop_CmpNEZ32x16",
            "Iop_CmpNEZ64x2",
            "Iop_CmpNEZ64x4",
            "Iop_CmpNEZ64x8",
            "Iop_CmpNEZ128x1",
            "Iop_Add8x4",
            "Iop_Sub8x4",
            "Iop_Add16x2",
            "Iop_Sub16x2",
            "Iop_HAdd8Sx4",
            "Iop_HAdd8Ux4",
            "Iop_HAdd16Sx2",
            "Iop_HAdd16Ux2",
            "Iop_HSub8Sx4",
            "Iop_HSub8Ux4",
            "Iop_HSub16Sx2",
            "Iop_HSub16Ux2",
            "Iop_QAdd8Sx4",
            "Iop_QAdd8Ux4",
            "Iop_QAdd16Sx2",
            "Iop_QAdd16Ux2",
            "Iop_QSub8Sx4",
            "Iop_QSub8Ux4",
            "Iop_QSub16Sx2",
            "Iop_QSub16Ux2",
            "Iop_Ctz8x16",
            "Iop_Ctz16x8",
            "Iop_Ctz32x4",
            "Iop_Ctz64x2",
            "Iop_PopCount8x16",
            "Iop_Add128x1",
            "Iop_Sub128x1",
            "Iop_Perm16x8",
            "Iop_Perm8x32",
            "Iop_Perm16x16",
            "Iop_Perm64x4",
            "Iop_Perm8x64",
            "Iop_Perm16x32",
            "Iop_Perm32x16",
            "Iop_Perm64x8",
            "Iop_Max64Sx8",
            "Iop_Max64Ux8",
            "Iop_Min64Sx8",
            "Iop_Min64Ux8",
        ),
        UNMAPPED,
    ),
    # Vector integer lanes, AVX-512 compares and tests.
    **dict.fromkeys(
        (
            "Iop_Cmp8Sx16",
            "Iop_Cmp8Ux16",
            "Iop_Cmp8Sx32",
            "Iop_Cmp8Ux32",
            "Iop_Cmp8Sx64",
            "Iop_Cmp8Ux64",
            "Iop_Cmp16Sx8",
            "Iop_Cmp16Ux8",
            "Iop_Cmp16Sx16",
            "Iop_Cmp16Ux16",
            "Iop_Cmp16Sx32",
            "Iop_Cmp16Ux32",
            "Iop_Cmp32Sx4",
            "Iop_Cmp32Ux4",
            "Iop_Cmp32Sx8",
            "Iop_Cmp32Ux8",
            "Iop_Cmp32Sx16",
            "Iop_Cmp32Ux16",
            "Iop_Cmp64Sx2",
            "Iop_Cmp64Ux2",
            "Iop_Cmp64Sx4",
            "Iop_Cmp64Ux4",
            "Iop_Cmp64Sx8",
            "Iop_Cmp64Ux8",
            "Iop_Test8x16",
            "Iop_Test8x32",
            "Iop_Test8x64",
            "Iop_Test16x8",
            "Iop_Test16x16",
            "Iop_Test16x32",
            "Iop_Test32x4",
            "Iop_Test32x8",
            "Iop_Test32x16",
            "Iop_Test64x2",
            "Iop_Test64x4",
            "Iop_Test64x8",
            "Iop_TestN8x16",
            "Iop_TestN8x32",
            "Iop_TestN8x64",
            "Iop_TestN16x8",
            "Iop_TestN16x16",
            "Iop_TestN16x32",
            "Iop_TestN32x4",
            "Iop_TestN32x8",
            "Iop_TestN32x16",
            "Iop_TestN64x2",
            "Iop_TestN64x4",
            "Iop_TestN64x8",
            "Iop_Ternlog32x16",
            "Iop_Ternlog64x8",
        ),
        UNMAPPED,
    ),
    # Whole-register vector ops.
    **dict.fromkeys(
        (
            "Iop_AndV512",
            "Iop_OrV512",
            "Iop_NotV512",
            "Iop_ShlV128",
            "Iop_ShrV128",
            "Iop_SarV128",
            "Iop_SliceV128",
            "Iop_V128HLtoV256",
            "Iop_V256HLtoV512",
            "Iop_64x4toV256",
            "Iop_V128to32",
            "Iop_V256to64_0",
            "Iop_V256to64_1",
            "Iop_V256to64_2",
            "Iop_V256to64_3",
            "Iop_V256toV128_0",
            "Iop_V256toV128_1",
            "Iop_V512toV256_0",
            "Iop_V512toV256_1",
            "Iop_V512to64_0",
            "Iop_V512to64_1",
            "Iop_V512to64_2",
            "Iop_V512to64_3",
            "Iop_V512to64_4",
            "Iop_V512to64_5",
            "Iop_V512to64_6",
            "Iop_V512to64_7",
            "Iop_ExpandBitsToInt",
            "Iop_ExpandBitsToV128",
            "Iop_ExpandBitsToV256",
            "Iop_ExpandBitsToV512",
        ),
        UNMAPPED,
    ),
}


def test_rust_irop_unserved_is_pinned():
    """Serving an irop.py op, or dropping one, means editing UNSERVED (angr-mx00p.100.3)."""
    unserved = set(irop.operations) - set(vex_irop.SUPPORTED)
    assert sorted(unserved - UNSERVED.keys()) == [], "unserved but not pinned"
    assert sorted(UNSERVED.keys() - unserved) == [], "pinned but served, or not an irop.py op"


def test_engine_honours_irop_operations_edits(monkeypatch):
    # Rust serves only what irop.py implements at call time, so an op removed from irop.operations stays unsupported.
    monkeypatch.delitem(irop.operations, "Iop_Add32")
    args = (claripy.BVS("x", 32), claripy.BVV(1, 32))
    assert _seam("Iop_Add32", args)[1] is UnsupportedIROpError


def test_engine_refuses_float_irops_without_fp_support():
    # irop.py refuses every float op without SUPPORT_FLOATING_POINT, so Rust must not serve one (angr-mx00p.101.1).
    engine = SimpleNamespace(state=SimpleNamespace(options=set()))
    v128 = (claripy.BVV(0, 32), claripy.BVV(1, 128), claripy.BVV(2, 128))
    assert "Iop_Add32Fx4" in datalayer._RUST_IROP_NAMES
    with pytest.raises(UnsupportedIROpError):
        datalayer.ClaripyDataMixin._perform_vex_expr_Op(engine, "Iop_Add32Fx4", v128)
    args = (claripy.BVS("x", 32), claripy.BVV(1, 32))
    served = datalayer.ClaripyDataMixin._perform_vex_expr_Op(engine, "Iop_Add32", args)
    assert served.identical(_python("Iop_Add32", args)[0])


def test_rust_irop_honours_a_symbolic_rounding_mode():
    # irop.py rounds to nearest-even under a symbolic mode; Rust keeps the mode's every value.
    a, b = claripy.BVS("a", 128), claripy.BVS("b", 128)
    rm = claripy.BVS("rm", 32)
    result = vex_irop.irop("Iop_Add32Fx4", (rm, a, b))
    assert result is not None and rm.variables <= result.variables
    cases = {
        "Iop_AddF32": (claripy.FPS("x", claripy.FSORT_FLOAT), claripy.FPS("y", claripy.FSORT_FLOAT)),
        "Iop_DivF64": (claripy.FPS("x", claripy.FSORT_DOUBLE), claripy.FPS("y", claripy.FSORT_DOUBLE)),
        "Iop_F64toF32": (claripy.FPS("x", claripy.FSORT_DOUBLE),),
        "Iop_I64StoF32": (claripy.BVS("i", 64),),
        "Iop_I64UtoF64": (claripy.BVS("i", 64),),
        # 2^53 + 1 is the smallest integer F64 cannot hold; it rounds apart by mode.
        "Iop_I64StoF64": (claripy.BVV((1 << 53) + 1, 64),),
    }
    for name, operands in cases.items():
        result = vex_irop.irop(name, (rm, *operands))
        assert isinstance(result, claripy.ast.FP) and rm.variables <= result.variables, name
        for mode in ROUNDING_MODES:
            pinned = vex_irop.irop(name, (claripy.BVV(mode, 32), *operands))
            solver = claripy.Solver(timeout=SOLVER_TIMEOUT_MS)
            solver.add(rm == mode)
            solver.add([claripy.Not(claripy.fpEQ(result, pinned)), claripy.Not(claripy.fpIsNaN(pinned))])
            assert not solver.satisfiable(), (name, mode)


def test_rust_irop_float_to_int_range():
    # libVEX leaves an out-of-range, infinite or NaN float-to-int conversion undefined, so irop.py's reading stands.
    rz = claripy.BVV(3, 32)
    f64 = lambda v: claripy.FPV(v, claripy.FSORT_DOUBLE)  # noqa: E731
    f32 = lambda v: claripy.FPV(v, claripy.FSORT_FLOAT)  # noqa: E731
    assert vex_irop.irop("Iop_F64toI32S", (rz, f64(2.0**31))) is None
    assert vex_irop.irop("Iop_F64toI32S", (rz, f64(2.0**31 - 1))).concrete_value == 2**31 - 1
    assert vex_irop.irop("Iop_F64toI32S", (rz, f64(-(2.0**31)))).concrete_value == 2**31
    assert vex_irop.irop("Iop_F64toI16S", (rz, f64(2.0**15))) is None
    assert vex_irop.irop("Iop_F64toI16S", (rz, f64(-(2.0**15)))).concrete_value == 2**15
    assert vex_irop.irop("Iop_F32toI64U", (rz, f32(-1.0))) is None
    assert vex_irop.irop("Iop_F32toI64U", (rz, f32(-0.0))).concrete_value == 0
    assert vex_irop.irop("Iop_F64toI64U", (rz, f64(math.nan))) is None
    assert vex_irop.irop("Iop_F32toI32U", (rz, f32(math.inf))) is None
    # Some mode rounds 2^31 - 0.5 up to 2^31, so Rust declines it whatever the mode.
    assert vex_irop.irop("Iop_F64toI32S", (rz, f64(2.0**31 - 0.5))) is None
    assert vex_irop.irop("Iop_F64toI32S", (claripy.BVS("rm", 32), f64(2.0**31 - 0.5))) is None
    # A symbolic operand is served.
    assert vex_irop.irop("Iop_F64toI32S", (rz, claripy.FPS("x", claripy.FSORT_DOUBLE))) is not None


def test_rust_irop_round_to_int():
    # Rounding toward +inf keeps -0.3's sign, and mode 4 rounds a tie away from zero.
    for name, sort in (("Iop_RoundF64toInt", claripy.FSORT_DOUBLE), ("Iop_RoundF32toInt", claripy.FSORT_FLOAT)):
        for mode, value, want in ((2, -0.3, -0.0), (4, 2.5, 3.0), (4, -2.5, -3.0), (0, 2.5, 2.0)):
            args = (claripy.BVV(mode, 32), claripy.FPV(value, sort))
            result = vex_irop.irop(name, args)
            assert result.raw_to_bv().concrete_value == claripy.FPV(want, sort).raw_to_bv().concrete_value
            assert result.raw_to_bv().concrete_value == _python(name, args)[0].raw_to_bv().concrete_value


def test_rust_irop_scalar_float_operand_sorts():
    rm, one = claripy.BVV(0, 32), claripy.FPV(1.0, claripy.FSORT_DOUBLE)
    # irop.py raises for a bitvector float operand, so Rust leaves it to irop.py.
    assert vex_irop.irop("Iop_AddF64", (rm, claripy.BVV(1, 64), one)) is None
    assert _python("Iop_AddF64", (rm, claripy.BVV(1, 64), one))[1] is not None
    # The rounding mode is a bitvector.
    assert vex_irop.irop("Iop_AddF64", (one, one, one)) is None
    # A mode libVEX does not define: irop.py raises.
    assert vex_irop.irop("Iop_AddF64", (claripy.BVV(5, 32), one, one)) is None
    # A float of the other width.
    assert vex_irop.irop("Iop_AddF32", (rm, one, one)) is None
    # A bvToFP keeps a NaN's payload through the round trip.
    snan = claripy.BVV(0x7FF4000000000001, 64).raw_to_fp()
    actual, expected = vex_irop.irop("Iop_MulF64", (rm, snan, one)), _python("Iop_MulF64", (rm, snan, one))[0]
    assert actual.raw_to_bv().concrete_value == expected.raw_to_bv().concrete_value


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


class PinnedAnnotation(claripy.Annotation):
    """Neither relocatable nor eliminatable: claripy does not simplify through it."""

    relocatable = False
    eliminatable = False

    def __hash__(self):
        return hash(PinnedAnnotation)

    def __eq__(self, other):
        return isinstance(other, PinnedAnnotation)


# The ops the Python engine reached with an UninitializedAnnotation literal on the drop-in survey.
ANNOTATED_LITERAL_OPS = [
    "Iop_And32",
    "Iop_Or32",
    "Iop_Xor8",
    "Iop_Add8",
    "Iop_Add32",
    "Iop_Add64",
    "Iop_Sub32",
    "Iop_Shl32",
    "Iop_Shr32",
    "Iop_CmpEQ32",
    "Iop_CmpNE32",
    "Iop_CmpNE64",
    "Iop_32to1",
    "Iop_32to8",
    "Iop_64to1",
    "Iop_64to8",
    "Iop_1Uto8",
    "Iop_1Uto32",
    "Iop_1Uto64",
    "Iop_8Uto64",
]


@pytest.mark.parametrize("name", ANNOTATED_LITERAL_OPS)
def test_rust_irop_carries_a_literal_s_annotations(name):
    ret_bits, arg_bits = _signature(name)
    rng = random.Random(name)
    for pos in range(len(arg_bits)):
        for annotation in (RelocAnnotation(), UninitializedAnnotation()):
            for other in ("symbolic", "concrete"):
                args = []
                for i, bits in enumerate(arg_bits):
                    # A shift amount stays below the width: Rust leaves a symbolic value's wider shift to irop.py.
                    value = claripy.BVV(
                        rng.getrandbits(bits) % (arg_bits[0] if name.startswith(SHIFTS) else 1 << bits), bits
                    )
                    if i == pos:
                        args.append(value.annotate(annotation))
                    elif other == "symbolic":
                        args.append(claripy.BVS(f"a{i}", bits))
                    else:
                        args.append(value)
                expected = _python(name, args)[0]
                if name.startswith(SHIFTS) and args[0].symbolic and pos == 1:
                    # Rust leaves a symbolic value's literal shift to irop.py (angr-mx00p.100.12).
                    assert vex_irop.irop(name, tuple(args)) is None
                    continue
                assert _check(name, args, ret_bits), f"{name}{tuple(args)}: Rust refused"
                actual = vex_irop.irop(name, tuple(args))
                assert set(actual.annotations) == set(expected.annotations), f"{name}{tuple(args)}"
                assert any(type(a) is type(annotation) for a in actual.annotations)
    # claripy does not fold through a pinned annotation, so irop.py serves the call.
    args = [claripy.BVV(1, bits) for bits in arg_bits]
    args[0] = args[0].annotate(PinnedAnnotation())
    assert vex_irop.irop(name, tuple(args)) is None


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
