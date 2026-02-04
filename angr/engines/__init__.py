from __future__ import annotations

from .engine import SimEngine
from .failure import SimEngineFailure
from .hook import HooksMixin
from .procedure import ProcedureEngine, ProcedureMixin
from .soot import SootMixin
from .successors import SimSuccessors, SuccessorsEngine
from .syscall import SimEngineSyscall
from .unicorn import SimEngineUnicorn
from .vex import HeavyResilienceMixin, HeavyVEXMixin, SimInspectMixin, SuperFastpathMixin, TrackActionsMixin
from .ail import AILMixin


class UberEngine(
    SimEngineFailure,
    SimEngineSyscall,
    HooksMixin,
    SimEngineUnicorn,
    SuperFastpathMixin,
    TrackActionsMixin,
    SimInspectMixin,
    HeavyResilienceMixin,
    SootMixin,
    AILMixin,
    HeavyVEXMixin,
):
    """
    The default execution engine for angr. This engine includes mixins for most
    common functionality in angr, including VEX IR, unicorn, syscall handling,
    and simprocedure handling.

    For some performance-sensitive applications, you may want to create a custom
    engine with only the necessary mixins.
    """


__all__ = [
    "HeavyResilienceMixin",
    "HeavyVEXMixin",
    "HooksMixin",
    "ProcedureEngine",
    "ProcedureMixin",
    "SimEngine",
    "SimEngineFailure",
    "SimEngineSyscall",
    "SimEngineUnicorn",
    "SimInspectMixin",
    "SimSuccessors",
    "SootMixin",
    "SuccessorsEngine",
    "SuperFastpathMixin",
    "TrackActionsMixin",
    "UberEngine",
]


# Rust VEX Engine (optional, requires rustylib with vex-engine feature)
try:
    from .rust_vex import RustVEXMixin, RustVEXEngineWrapper, RUST_ENGINE_AVAILABLE

    # UberEngineRust is similar to UberEngine but includes RustVEXMixin for
    # potential future use. Currently, HeavyVEXMixin handles all VEX execution
    # because it comes before RustVEXMixin in the MRO (via TrackActionsMixin).
    #
    # To enable Rust VEX execution, use RustVEXEngine directly via the wrapper.
    class UberEngineRust(
        SimEngineFailure,
        SimEngineSyscall,
        HooksMixin,
        TrackActionsMixin,
        SimInspectMixin,
        HeavyResilienceMixin,
        RustVEXMixin,  # Not used in MRO due to HeavyVEXMixin from TrackActionsMixin
    ):
        """
        Execution engine that includes both Python and Rust VEX capabilities.

        Currently uses Python VEX execution (HeavyVEXMixin via TrackActionsMixin).
        The RustVEXMixin is included for direct access via the rust_engine property.

        For pure Rust execution, use RustVEXEngineWrapper directly.
        """

    __all__.extend(["RustVEXMixin", "RustVEXEngineWrapper", "UberEngineRust", "RUST_ENGINE_AVAILABLE"])

except ImportError:
    # Rust engine not available
    RUST_ENGINE_AVAILABLE = False
    __all__.append("RUST_ENGINE_AVAILABLE")


try:
    from .pcode import HeavyPcodeMixin

    class UberEnginePcode(
        SimEngineFailure, SimEngineSyscall, HooksMixin, HeavyPcodeMixin
    ):  # pylint:disable=abstract-method
        pass

    __all__.append("UberEnginePcode")

except ImportError:
    pass
