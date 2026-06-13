"""``@_deprecated`` decorator for the Rust exploration engine public API.

Marks a name on the Rust engine public surface as deprecated. The decorated
callable still runs; the wrapper emits a ``DeprecationWarning`` whose message
carries the version metadata required by the API stability policy
(:ref:`rust-engine-api-stability` in ``docs/advanced-topics/rust_engine.rst``).

The warning fires **once per decorated callable per process** so test logs
and long-running sessions stay legible. ``stacklevel=2`` so the warning
points at the caller, not at this wrapper.

This decorator is intentionally underscore-prefixed (private). It is meant to
be applied from inside ``angr/exploration/`` (and the Rust pyclass shim
layer) when an existing public name is being phased out. Downstream code
should not import or apply it.

Example
-------

.. code-block:: python

    from angr.exploration._deprecation import _deprecated

    class RustExplorationManager:
        @_deprecated(version="9.3", removed_in="9.4", replacement="run")
        def explore(self, *args, **kwargs):
            return self.run(*args, **kwargs)

The first call from any context emits::

    DeprecationWarning: RustExplorationManager.explore is deprecated since
    9.3 and will be removed in 9.4. Use ``run`` instead.

See :ref:`rust-engine-api-stability` for when and how to apply this decorator
during a minor-release deprecation cycle.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_warned: set[object] = set()

P = ParamSpec("P")
R = TypeVar("R")


def _deprecated(
    *,
    version: str,
    removed_in: str,
    replacement: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a Rust-engine public callable as deprecated.

    Parameters
    ----------
    version
        The minor release in which deprecation started (e.g. ``"9.3"``).
        Anchors the warning to a release the user can look up.
    removed_in
        The minor release in which the name is targeted for removal
        (e.g. ``"9.4"``). Must be later than ``version``; per the policy
        doc, removal happens no earlier than the next minor release.
    replacement
        Optional. The name a caller should migrate to. When given, the
        warning message ends with ``Use ``<replacement>`` instead.``.
    """

    def outer(func: Callable[P, R]) -> Callable[P, R]:
        qualname = getattr(func, "__qualname__", func.__name__)

        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            if func not in _warned:
                _warned.add(func)
                message = f"{qualname} is deprecated since {version} and will be removed in {removed_in}."
                if replacement is not None:
                    message += f" Use ``{replacement}`` instead."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return inner

    return outer


__all__ = ["_deprecated"]
