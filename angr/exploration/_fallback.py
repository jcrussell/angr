"""Fallback helpers that never swallow ``MemoryError`` (angr-mx00p.132.66).

A cat-(b)/(c) fallback that logs and continues turns an out-of-memory error
into a silently wrong state (angr-mx00p.132.65), so every broad handler in
this package re-raises ``MemoryError``; ``tools/audit_memory_error_swallow.py``
gates it.
"""

from __future__ import annotations


class suppress_recoverable:  # pylint: disable=invalid-name
    """``contextlib.suppress(Exception)`` that lets ``MemoryError`` through."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        return exc_type is not None and issubclass(exc_type, Exception) and not issubclass(exc_type, MemoryError)


__all__ = ["suppress_recoverable"]
