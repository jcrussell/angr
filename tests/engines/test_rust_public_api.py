"""Snapshot tests for the ``angr.exploration`` public API surface.

These tests pin the public surface declared in
``angr/exploration/_public_api.py`` against what the runtime classes actually
expose. They fail when a public name is added, removed, or renamed without a
corresponding inventory update — that is the point.

A failure here is **not** a bug in the test. Either:

1. The change was intentional and the inventory needs to be updated in the
   same PR (along with a version bump / deprecation cycle per the
   ``:ref:`rust-engine-api-stability``` policy in
   ``docs/advanced-topics/rust_engine.rst``); or
2. The change was unintentional — revert or rename to a private name
   (leading underscore).

Parent epic: angr-9cps. This is sub-task .3 (drift enforcement).
"""

from __future__ import annotations

import pytest

# Availability guard lives in conftest — if the Rust extension didn't build,
# skip rather than ImportError at collection time (angr-7gdp).
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


_REMEDIATION_HINT = (
    "Public surface drift detected. If this change is intentional, update "
    "angr/exploration/_public_api.py in the same PR and follow the policy "
    "at docs/advanced-topics/rust_engine.rst (:ref:`rust-engine-api-stability`) "
    "— version bump + deprecation cycle for removals/renames. If "
    "unintentional, rename to a leading-underscore private name."
)


def _resolve_classes() -> dict[str, type]:
    """Map each class name in the inventory to the actual runtime class.

    Importing all of these at module import time would couple this test
    file's collection to internal module layout. Resolving lazily here
    keeps the failure modes localized to the offending row.
    """
    from angr.exploration import RustErrorRecord, RustExplorationManager
    from angr.exploration.rust_state_proxy import (
        RustCallStackFrameProxy,
        RustCallStackProxy,
        RustHeapProxy,
        RustHistoryProxy,
        RustInspectProxy,
        RustMemoryProxy,
        RustPosixProxy,
        RustRegisterProxy,
        RustScratchProxy,
        RustSimulationManagerProxy,
        RustSolverProxy,
        RustStateProxy,
    )

    return {
        "RustErrorRecord": RustErrorRecord,
        "RustExplorationManager": RustExplorationManager,
        "RustStateProxy": RustStateProxy,
        "RustSolverProxy": RustSolverProxy,
        "RustRegisterProxy": RustRegisterProxy,
        "RustMemoryProxy": RustMemoryProxy,
        "RustHeapProxy": RustHeapProxy,
        "RustScratchProxy": RustScratchProxy,
        "RustHistoryProxy": RustHistoryProxy,
        "RustPosixProxy": RustPosixProxy,
        "RustCallStackProxy": RustCallStackProxy,
        "RustCallStackFrameProxy": RustCallStackFrameProxy,
        "RustInspectProxy": RustInspectProxy,
        "RustSimulationManagerProxy": RustSimulationManagerProxy,
    }


def _actual_public_attrs(cls_name: str, cls: type) -> set[str]:
    """Compute the public attribute surface a caller would observe.

    For most classes ``dir(cls)`` is sufficient because every public name is
    a method or ``@property``. ``RustErrorRecord`` is the outlier — its
    public attributes are plain instance attributes set in ``__init__``, so
    we build a stand-in instance and merge ``vars()`` with ``dir(cls)``.
    """
    if cls_name == "RustErrorRecord":
        # Construct with a benign payload; init writes the attrs we need to
        # see. The taxonomy classification path needs a real message prefix
        # so it doesn't fall through to "unknown" — pick "memory error" from
        # the _ERROR_CLASS_PREFIXES table.
        instance = cls(state=None, message="memory error: snapshot", addr=0)
        names = set(vars(instance)) | set(dir(cls))
    else:
        names = set(dir(cls))
    return {n for n in names if not n.startswith("_")}


class TestRustEngineVersion:
    """The ``__rust_engine_version__`` attribute exists, is a non-empty
    string, and matches what the Rust submodule exposes.

    The value is sourced from ``native/angr/Cargo.toml`` via
    ``CARGO_PKG_VERSION`` at compile time; both import paths must agree.
    Intentionally **not** in ``__all__`` (follows the standard
    ``__version__`` convention — excluded from ``import *`` but
    accessible by explicit name).
    """

    def test_attribute_is_non_empty_string(self):
        import angr.exploration

        version = angr.exploration.__rust_engine_version__
        assert isinstance(version, str) and version, (
            f"__rust_engine_version__ must be a non-empty string, got {version!r}"
        )

    def test_attribute_matches_rustylib_submodule(self):
        from angr.rustylib.vex_engine import __version__ as rust_version

        import angr.exploration

        assert angr.exploration.__rust_engine_version__ == rust_version, (
            f"Version drift between angr.exploration.__rust_engine_version__ "
            f"({angr.exploration.__rust_engine_version__!r}) and "
            f"angr.rustylib.vex_engine.__version__ ({rust_version!r})"
        )

    def test_attribute_not_in_dunder_all(self):
        """Follows the standard ``__version__`` convention: explicit access
        only, never via ``from angr.exploration import *``.
        """
        import angr.exploration

        assert "__rust_engine_version__" not in angr.exploration.__all__, (
            "__rust_engine_version__ should follow the __version__ convention "
            "(excluded from __all__). If you intentionally added it, update "
            "this test and the policy doc."
        )


class TestModuleExports:
    """The ``angr.exploration.__all__`` contract."""

    def test_dunder_all_matches_inventory(self):
        """``__all__`` and ``MODULE_EXPORTS`` must agree, ordered or not."""
        import angr.exploration
        from angr.exploration._public_api import MODULE_EXPORTS

        actual = set(angr.exploration.__all__)
        pinned = set(MODULE_EXPORTS)
        missing = pinned - actual
        extra = actual - pinned
        assert not missing and not extra, (
            f"angr.exploration.__all__ drifted from MODULE_EXPORTS.\n"
            f"  added (in __all__, not in inventory): {sorted(extra)}\n"
            f"  removed (in inventory, not in __all__): {sorted(missing)}\n"
            f"{_REMEDIATION_HINT}"
        )

    def test_every_exported_name_resolves(self):
        """``from angr.exploration import *`` must not be a lie."""
        import angr.exploration
        from angr.exploration._public_api import MODULE_EXPORTS

        unresolved = [n for n in MODULE_EXPORTS if not hasattr(angr.exploration, n)]
        assert not unresolved, (
            f"MODULE_EXPORTS lists names not actually reachable from "
            f"angr.exploration: {unresolved}\n{_REMEDIATION_HINT}"
        )


class TestTypedExceptions:
    """The ``TYPED_EXCEPTIONS`` block is the documented error taxonomy."""

    def test_typed_exceptions_are_in_module_exports(self):
        from angr.exploration._public_api import MODULE_EXPORTS, TYPED_EXCEPTIONS

        leaked = set(TYPED_EXCEPTIONS) - set(MODULE_EXPORTS)
        assert not leaked, (
            f"TYPED_EXCEPTIONS contains names not in MODULE_EXPORTS: {sorted(leaked)}\n{_REMEDIATION_HINT}"
        )

    def test_typed_exceptions_are_exception_subclasses(self):
        import angr.exploration
        from angr.exploration._public_api import TYPED_EXCEPTIONS

        bad = []
        for name in TYPED_EXCEPTIONS:
            obj = getattr(angr.exploration, name, None)
            if not (isinstance(obj, type) and issubclass(obj, BaseException)):
                bad.append(name)
        assert not bad, f"TYPED_EXCEPTIONS lists names that are not Exception subclasses: {bad}\n{_REMEDIATION_HINT}"


class TestClassPublicAttrs:
    """Per-class snapshot of the public attribute surface."""

    def test_inventory_covers_every_module_export_class(self):
        """Every class-typed name in ``MODULE_EXPORTS`` must be pinned.

        Catches the case where a new public class is added to
        ``__init__.py`` but the inventory is updated only for the export
        list, not the class-attr block.
        """
        import angr.exploration
        from angr.exploration._public_api import CLASS_PUBLIC_ATTRS, MODULE_EXPORTS

        export_classes = {
            n
            for n in MODULE_EXPORTS
            if isinstance(getattr(angr.exploration, n, None), type)
            and not issubclass(getattr(angr.exploration, n), BaseException)
        }
        missing = export_classes - set(CLASS_PUBLIC_ATTRS)
        assert not missing, f"Exported classes lack a CLASS_PUBLIC_ATTRS entry: {sorted(missing)}\n{_REMEDIATION_HINT}"

    @pytest.mark.parametrize(
        "class_name",
        # Importable at parametrize-time: just list the keys without
        # touching the runtime classes (which need the Rust .so loaded).
        # _public_api.py is a pure-Python data module.
        list(__import__("angr.exploration._public_api", fromlist=["CLASS_PUBLIC_ATTRS"]).CLASS_PUBLIC_ATTRS),
    )
    def test_class_public_attrs_match_inventory(self, class_name):
        """Per-class drift detector — a row at a time so failures localize."""
        from angr.exploration._public_api import CLASS_PUBLIC_ATTRS

        classes = _resolve_classes()
        assert class_name in classes, (
            f"CLASS_PUBLIC_ATTRS lists {class_name!r} but _resolve_classes() "
            f"in this test file has no entry for it. Add the import + entry "
            f"so the snapshot can be checked."
        )
        cls = classes[class_name]
        actual = _actual_public_attrs(class_name, cls)
        pinned = set(CLASS_PUBLIC_ATTRS[class_name])

        added = actual - pinned
        removed = pinned - actual
        assert not added and not removed, (
            f"{class_name} public surface drifted.\n"
            f"  added (on class, not in inventory): {sorted(added)}\n"
            f"  removed (in inventory, not on class): {sorted(removed)}\n"
            f"{_REMEDIATION_HINT}"
        )
