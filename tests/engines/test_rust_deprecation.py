"""Tests for ``angr.exploration._deprecation._deprecated``.

The decorator is the Rust-engine half of the deprecation cycle described in
``docs/advanced-topics/rust_engine.rst`` (:ref:`rust-engine-api-stability`).
These tests verify the warning fires with the documented version metadata so
the policy doc's promise holds at runtime.

Parent epic: angr-9cps. This is sub-task .4 (decorator + warning test).
"""

from __future__ import annotations

import warnings

import pytest

from angr.exploration._deprecation import _deprecated, _warned


@pytest.fixture(autouse=True)
def _reset_warned_set():
    """Each test gets a clean ``_warned`` set so the once-per-callable
    deduplication does not leak between tests."""
    snapshot = set(_warned)
    _warned.clear()
    try:
        yield
    finally:
        _warned.clear()
        _warned.update(snapshot)


def test_warning_fires_on_first_call():
    @_deprecated(version="9.3", removed_in="9.4")
    def old_api():
        return 42

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = old_api()

    assert result == 42, "decorated callable must still return the wrapped value"
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    message = str(caught[0].message)
    assert "9.3" in message, f"deprecation version missing from {message!r}"
    assert "9.4" in message, f"removed_in version missing from {message!r}"
    assert "old_api" in message, f"callable name missing from {message!r}"


def test_warning_includes_replacement_when_given():
    @_deprecated(version="9.3", removed_in="9.4", replacement="new_api")
    def old_api():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_api()

    assert len(caught) == 1
    assert "new_api" in str(caught[0].message)


def test_warning_fires_once_per_callable():
    @_deprecated(version="9.3", removed_in="9.4")
    def old_api():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            old_api()

    assert len(caught) == 1, "warning must dedupe per callable so test logs and long-running sessions stay legible"


def test_separate_callables_warn_independently():
    @_deprecated(version="9.3", removed_in="9.4")
    def first():
        return None

    @_deprecated(version="9.3", removed_in="9.4")
    def second():
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first()
        second()

    assert len(caught) == 2
    assert "first" in str(caught[0].message)
    assert "second" in str(caught[1].message)


def test_preserves_wrapped_metadata():
    @_deprecated(version="9.3", removed_in="9.4")
    def old_api(x, y):
        """The old API docstring."""
        return x + y

    assert old_api.__name__ == "old_api"
    assert old_api.__doc__ == "The old API docstring."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert old_api(2, 3) == 5
