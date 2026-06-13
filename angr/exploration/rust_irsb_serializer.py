"""JSON serialization of pyvex IRSB blocks for the Rust engine."""

from __future__ import annotations

import json

_MASK64 = 0xFFFFFFFFFFFFFFFF

_CONST_TYPES = frozenset(
    (
        "U1",
        "U8",
        "U16",
        "U32",
        "U64",
        "U128",
        "F32",
        "F32i",
        "F64",
        "F64i",
        "V128",
        "V256",
    )
)

_EXPR_FIELDS = {
    # 'val' fields, 'str' fields, 'expr' fields
    "RdTmp": (("tmp",), (), ()),
    "Get": (("offset",), ("ty",), ()),
    "Load": ((), ("ty", "end"), ("addr",)),
    "Unop": None,  # special: singular 'arg' not 'args'
    "Binop": None,  # special: has 'op' + 'args'
    "Triop": None,  # same pattern
    "Qop": None,  # same pattern
    "ITE": ((), (), ("cond", "iftrue", "iffalse")),
}

_STMT_FIELDS = {
    # (val_fields, str_fields, expr_fields)
    "IMark": (("addr", "len", "delta"), (), ()),
    "WrTmp": (("tmp",), (), ("data",)),
    "Put": (("offset",), (), ("data",)),
    "Store": ((), ("end",), ("addr", "data")),
    "StoreG": ((), ("end",), ("addr", "data", "guard")),
    "AbiHint": (("len",), (), ("base", "nia")),
    "MBE": ((), (), ()),
    "NoOp": ((), (), ()),
}


def _serialize_const(con):
    """Serialize a pyvex constant (Ico_ prefix)."""
    name = type(con).__name__
    if name == "V128":
        val = con.value if isinstance(con.value, int) else 0
        return {"tag": "Ico_V128", "low": val & _MASK64, "high": (val >> 64) & _MASK64}
    if name == "V256":
        val = con.value if isinstance(con.value, int) else 0
        return {
            "tag": "Ico_V256",
            "value": [val & _MASK64, (val >> 64) & _MASK64, (val >> 128) & _MASK64, (val >> 192) & _MASK64],
        }
    return {"tag": f"Ico_{name}", "value": con.value}


def _serialize_descr(descr):
    """Serialize a VEX array descriptor (GetI/PutI)."""
    return {"base": descr.base, "elemTy": str(descr.elemTy), "nElems": descr.nElems}


def _serialize_cee(cee):
    """Serialize a callee descriptor (CCall/Dirty)."""
    return {"name": cee.name if hasattr(cee, "name") else str(cee), "addr": 0, "mcx_mask": getattr(cee, "mcx_mask", 0)}


def _serialize_expr(expr):
    if expr is None:
        return None
    if isinstance(expr, (int, float, str, bool)):
        return expr

    name = type(expr).__name__

    # Direct constants (Ico_ prefix)
    if hasattr(expr, "value") and name in _CONST_TYPES:
        return _serialize_const(expr)

    result = {"tag": f"Iex_{name}"}

    # Const expression — wraps a constant
    if hasattr(expr, "con"):
        con = expr.con
        con_result = (
            _serialize_const(con) if hasattr(con, "value") else {"tag": f"Ico_{type(con).__name__}", "value": 0}
        )
        result["con"] = con_result
        return result

    # Table-driven common expressions
    fields = _EXPR_FIELDS.get(name)
    if fields is not None:
        val_f, str_f, expr_f = fields
        for f in val_f:
            if hasattr(expr, f):
                result[f] = getattr(expr, f)
        for f in str_f:
            if hasattr(expr, f):
                result[f] = str(getattr(expr, f))
        for f in expr_f:
            if hasattr(expr, f):
                result[f] = _serialize_expr(getattr(expr, f))
        return result

    # Unop: singular 'arg' key
    if name == "Unop":
        result["op"] = expr.op
        if hasattr(expr, "args") and expr.args:
            result["arg"] = _serialize_expr(expr.args[0])
        return result

    # Binop/Triop/Qop: 'op' + 'args' list
    if hasattr(expr, "op"):
        result["op"] = expr.op
        if hasattr(expr, "args"):
            result["args"] = [_serialize_expr(a) for a in expr.args]
        return result

    # GetI: descr + ix + bias
    if name == "GetI":
        if hasattr(expr, "descr"):
            result["descr"] = _serialize_descr(expr.descr)
        if hasattr(expr, "ix"):
            result["ix"] = _serialize_expr(expr.ix)
        if hasattr(expr, "bias"):
            result["bias"] = expr.bias
        return result

    # CCall: callee + retty + args
    if name == "CCall":
        if hasattr(expr, "cee"):
            result["cee"] = _serialize_cee(expr.cee)
        if hasattr(expr, "retty"):
            result["retty"] = str(expr.retty)
        if hasattr(expr, "args"):
            result["args"] = [_serialize_expr(a) for a in expr.args]
        return result

    # Fallback
    for f in ("offset", "tmp"):
        if hasattr(expr, f):
            result[f] = getattr(expr, f)
    if hasattr(expr, "ty"):
        result["ty"] = str(expr.ty)
    return result


def _serialize_stmt(stmt):
    name = type(stmt).__name__
    result = {"tag": f"Ist_{name}"}

    # Table-driven common statements
    fields = _STMT_FIELDS.get(name)
    if fields is not None:
        val_f, str_f, expr_f = fields
        for f in val_f:
            if hasattr(stmt, f):
                result[f] = getattr(stmt, f)
        for f in str_f:
            if hasattr(stmt, f):
                result[f] = str(getattr(stmt, f))
        for f in expr_f:
            if hasattr(stmt, f):
                result[f] = _serialize_expr(getattr(stmt, f))
        return result

    # PutI: descr + ix + bias + data
    if name == "PutI":
        if hasattr(stmt, "descr"):
            result["descr"] = _serialize_descr(stmt.descr)
        if hasattr(stmt, "ix"):
            result["ix"] = _serialize_expr(stmt.ix)
        if hasattr(stmt, "bias"):
            result["bias"] = stmt.bias
        if hasattr(stmt, "data"):
            result["data"] = _serialize_expr(stmt.data)
        return result

    # LoadG: guarded load
    if name == "LoadG":
        for f in ("dst",):
            if hasattr(stmt, f):
                result[f] = getattr(stmt, f)
        for f in ("cvt", "end"):
            if hasattr(stmt, f):
                result[f] = str(getattr(stmt, f))
        for f in ("addr", "alt", "guard"):
            if hasattr(stmt, f):
                result[f] = _serialize_expr(getattr(stmt, f))
        return result

    # Exit: guard + constant dst + jk + offsIP
    if name == "Exit":
        if hasattr(stmt, "guard"):
            result["guard"] = _serialize_expr(stmt.guard)
        if hasattr(stmt, "dst"):
            result["dst"] = _serialize_const(stmt.dst)
        if hasattr(stmt, "jk"):
            result["jk"] = str(stmt.jk)
        if hasattr(stmt, "offsIP"):
            result["offsIP"] = stmt.offsIP
        return result

    # CAS: compare-and-swap
    if name == "CAS":
        result["end"] = str(stmt.end) if hasattr(stmt, "end") else "Iend_LE"
        result["oldLo"] = getattr(stmt, "oldLo", 0)
        # Single-word CAS leaves oldHi=IRTemp_INVALID (0xFFFFFFFF) on the
        # pyvex side; mirror libpyvex_ffi's convention by mapping the
        # sentinel to None so the Rust bridge's all-Some/all-None DCAS
        # validation accepts the converted shape.
        old_hi = getattr(stmt, "oldHi", None)
        result["oldHi"] = None if old_hi == 0xFFFFFFFF else old_hi
        for f in ("addr", "dataLo", "dataHi", "expdLo", "expdHi"):
            if hasattr(stmt, f):
                result[f] = _serialize_expr(getattr(stmt, f))
        return result

    # LLSC: load-linked/store-conditional
    if name == "LLSC":
        result["end"] = str(stmt.end) if hasattr(stmt, "end") else "Iend_LE"
        if hasattr(stmt, "addr"):
            result["addr"] = _serialize_expr(stmt.addr)
        if hasattr(stmt, "storedata"):
            result["storedata"] = _serialize_expr(stmt.storedata) if stmt.storedata else None
        if hasattr(stmt, "result"):
            result["result"] = stmt.result
        return result

    # Dirty: helper call with side effects
    if name == "Dirty":
        if hasattr(stmt, "cee"):
            result["cee"] = _serialize_cee(stmt.cee)
        result["guard"] = _serialize_expr(stmt.guard) if hasattr(stmt, "guard") and stmt.guard else None
        if hasattr(stmt, "args"):
            result["args"] = [_serialize_expr(a) for a in stmt.args]
        result["tmp"] = getattr(stmt, "tmp", None)
        result["mFx"] = str(stmt.mFx) if hasattr(stmt, "mFx") and stmt.mFx else "Ifx_None"
        result["mAddr"] = _serialize_expr(stmt.mAddr) if hasattr(stmt, "mAddr") and stmt.mAddr else None
        result["mSize"] = getattr(stmt, "mSize", 0)
        result["nFxState"] = getattr(stmt, "nFxState", 0)
        return result

    # Fallback for unknown statements
    for f in ("tmp", "offset", "len", "delta"):
        if hasattr(stmt, f):
            result[f] = getattr(stmt, f)
    if hasattr(stmt, "addr"):
        addr = stmt.addr
        result["addr"] = addr if isinstance(addr, int) else _serialize_expr(addr)
    for f in ("data", "guard", "dst"):
        if hasattr(stmt, f):
            result[f] = _serialize_expr(getattr(stmt, f))
    return result


def serialize_irsb(irsb) -> str:
    """Serialize a pyvex IRSB to JSON for the Rust VEX interpreter."""
    data = {
        "addr": irsb.addr,
        "arch": irsb.arch.name if hasattr(irsb.arch, "name") else str(irsb.arch),
        "statements": [_serialize_stmt(s) for s in irsb.statements],
        "next": _serialize_expr(irsb.next),
        "jumpkind": str(irsb.jumpkind),
        "offsIP": irsb.offsIP,
        "tyenv": {"types": [str(t) for t in irsb.tyenv.types] if irsb.tyenv else []},
    }
    return json.dumps(data)
