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

    # UberEngineRust uses RustVEXMixin for VEX execution when available.
    # RustVEXMixin comes before TrackActionsMixin in the MRO so Rust execution
    # is tried first. When Rust can't handle execution (symbolic addresses, etc.),
    # it falls back to HeavyVEXMixin via the super() chain.
    #
    # Enable RUST_VEX_LOOP sim_option for multi-block execution with deferred forks.
    class UberEngineRust(
        SimEngineFailure,
        SimEngineSyscall,
        HooksMixin,
        RustVEXMixin,  # First: try Rust VEX execution
        TrackActionsMixin,  # Fallback: Python VEX via HeavyVEXMixin
        SimInspectMixin,
        HeavyResilienceMixin,
    ):
        """
        Execution engine that uses Rust VEX execution with Python fallback.

        RustVEXMixin.process_successors() is called first due to MRO ordering.
        When Rust can't handle execution (symbolic addresses, unsupported ops),
        it falls back to HeavyVEXMixin via TrackActionsMixin.

        Multi-block execution with deferred forks is enabled by default when
        using RustSimSolver, reducing Python/Rust round trips. Without
        RustSimSolver, falls back to single-block mode. To explicitly disable
        multi-block mode, add RUST_VEX_SINGLE to state.options.

        Example:
            import angr
            from angr import sim_options as o
            from angr.state_plugins.rust_solver import RustSimSolver

            proj = angr.Project("/path/to/binary", auto_load_libs=False)
            state = proj.factory.entry_state()
            state.register_plugin('solver', RustSimSolver())  # Enable multi-block

            from angr.engines import UberEngineRust
            engine = UberEngineRust(proj)
            engine.configure_deferred_forks(enabled=True, max_forks=50)

            successors = engine.process(state)
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
