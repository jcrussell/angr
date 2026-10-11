"""Runtime shim: claripy 9.2.209 against the z3-solver 5.x Python bindings.

The Rust engine links z3-sys 0.13, whose minimum Z3 is newer than the
``z3-solver==4.13.0.0`` that standalone ``claripy==9.2.209`` was written
against, so the venv carries z3-solver 5.1 (``pyproject.toml``). Z3 5.0
changed the C signature of ``Z3_fpa_get_numeral_sign`` from ``int *`` to
``bool *`` and the Python bindings now declare the ``ctypes`` argtype
accordingly; claripy's ``BackendZ3._abstract_fp_val`` and
``_abstract_fp_encoded_val`` still pass ``byref(c_int())`` and raise
``ctypes.ArgumentError`` on every concrete FP model value.

:func:`install` wraps the binding so a ``c_int`` out-parameter is accepted
and filled, which is the entire incompatibility the fork's test-suite hits.
It is idempotent and a no-op when the binding already accepts ``c_int``
(z3-solver 4.x). This shim goes away with the upstream angr 10 merge, which
replaces standalone claripy with ``angr.claripy`` and drops the ``z3``
Python package entirely.
"""

from __future__ import annotations

import ctypes

_INSTALLED_ATTR = "_angr_z3_compat_wrapped"


def _needs_sign_shim(z3, z3core) -> bool:
    """True when the binding's out-parameter is declared ``bool *``.

    z3core wrappers carry their ctypes function as the ``_elems`` default
    argument (``Elementaries.f``); the module-level ``_lib`` is not exposed.
    Fall back to the Z3 major version when that layout is not recognised.
    """
    defaults = getattr(z3core.Z3_fpa_get_numeral_sign, "__defaults__", None) or ()
    cfunc = getattr(defaults[-1], "f", None) if defaults else None
    argtypes = getattr(cfunc, "argtypes", None)
    if argtypes:
        return argtypes[-1] is ctypes.POINTER(ctypes.c_bool)
    return int(z3.get_version_string().split(".")[0]) >= 5


def install() -> bool:
    """Wrap ``z3.Z3_fpa_get_numeral_sign`` to accept an ``int`` out-param.

    Returns True when the wrapper is (or already was) in place, False when
    z3 is unavailable or the shim is not needed for the installed version.
    """
    try:
        import z3
        from z3 import z3core
    except ImportError:
        return False

    original = z3core.Z3_fpa_get_numeral_sign
    if getattr(original, _INSTALLED_ATTR, False):
        return True
    if not _needs_sign_shim(z3, z3core):
        return False

    def Z3_fpa_get_numeral_sign(ctx, ast, out):
        sign = ctypes.c_bool()
        ok = original(ctx, ast, ctypes.byref(sign))
        # `byref(x)` yields a CArgObject whose `_obj` is `x`; a raw pointer
        # exposes `.contents`. Either way write the sign back as 0/1.
        target = getattr(out, "_obj", None)
        if target is None:
            target = out.contents
        target.value = int(sign.value)
        return ok

    setattr(Z3_fpa_get_numeral_sign, _INSTALLED_ATTR, True)
    z3core.Z3_fpa_get_numeral_sign = Z3_fpa_get_numeral_sign
    z3.Z3_fpa_get_numeral_sign = Z3_fpa_get_numeral_sign
    return True
