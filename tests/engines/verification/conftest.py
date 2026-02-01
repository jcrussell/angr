"""
Pytest configuration and fixtures for verification tests.
"""
from __future__ import annotations

import pytest
import logging

import claripy

# Set up logging
logging.getLogger("angr").setLevel(logging.WARNING)
logging.getLogger("claripy").setLevel(logging.WARNING)

# Import Rust engine availability
try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE, RustVEXEngineWrapper
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "rust_engine: mark test as requiring Rust VEX engine"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require Rust engine if it's not available."""
    if RUST_ENGINE_AVAILABLE:
        return

    skip_rust = pytest.mark.skip(reason="Rust VEX engine not available")
    for item in items:
        if "rust_engine" in item.keywords:
            item.add_marker(skip_rust)


@pytest.fixture(scope="session")
def rust_available():
    """Check if Rust engine is available."""
    return RUST_ENGINE_AVAILABLE


@pytest.fixture
def rust_engine_amd64():
    """Create AMD64 Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    return RustVEXEngineWrapper("amd64")


@pytest.fixture
def rust_engine_x86():
    """Create x86 Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    return RustVEXEngineWrapper("x86")


@pytest.fixture
def rust_engine_arm():
    """Create ARM Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    try:
        return RustVEXEngineWrapper("arm")
    except Exception:
        pytest.skip("ARM Rust engine not available")


@pytest.fixture
def rust_engine_arm64():
    """Create ARM64 Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    try:
        return RustVEXEngineWrapper("arm64")
    except Exception:
        pytest.skip("ARM64 Rust engine not available")


@pytest.fixture
def python_engine_amd64():
    """Create AMD64 Python VEX engine."""
    from angr import SimState, load_shellcode
    from angr.engines import HeavyVEXMixin

    project = load_shellcode(b"\xc3", arch="AMD64")
    engine = HeavyVEXMixin(project)
    state = SimState(project=project)
    engine.state = state
    return engine


@pytest.fixture
def python_engine_x86():
    """Create x86 Python VEX engine."""
    from angr import SimState, load_shellcode
    from angr.engines import HeavyVEXMixin

    project = load_shellcode(b"\xc3", arch="X86")
    engine = HeavyVEXMixin(project)
    state = SimState(project=project)
    engine.state = state
    return engine


@pytest.fixture
def edge_case_values():
    """Generate edge case values for various bit widths."""
    def _generate(width: int) -> list[int]:
        mask = (1 << width) - 1
        values = [
            0,                              # Zero
            1,                              # One
            mask,                           # All ones
            1 << (width - 1),               # Only MSB set
            (1 << (width - 1)) - 1,         # Max positive signed
            mask - 1,                       # Max unsigned - 1
        ]

        # Powers of 2
        for i in [1, 2, 4, 8, width // 4, width // 2, width - 1]:
            if 0 <= i < width:
                values.append(1 << i)
                if (1 << i) - 1 not in values:
                    values.append((1 << i) - 1)

        # Alternating patterns
        patterns = [0x55, 0xAA, 0x0F, 0xF0]
        for p in patterns:
            val = 0
            for j in range(0, width, 8):
                val |= p << j
            values.append(val & mask)

        return list(set(values))

    return _generate


@pytest.fixture
def shift_amounts():
    """Generate test shift amounts for various bit widths."""
    def _generate(width: int) -> list[int]:
        return [0, 1, 2, width // 2, width - 2, width - 1, width, width + 1]

    return _generate


@pytest.fixture
def symbolic_vars():
    """Create common symbolic variables for testing."""
    return {
        "x8": claripy.BVS("x8", 8),
        "y8": claripy.BVS("y8", 8),
        "x16": claripy.BVS("x16", 16),
        "y16": claripy.BVS("y16", 16),
        "x32": claripy.BVS("x32", 32),
        "y32": claripy.BVS("y32", 32),
        "x64": claripy.BVS("x64", 64),
        "y64": claripy.BVS("y64", 64),
    }


# Known unsupported operations in Rust engine
RUST_UNSUPPORTED_OPS = frozenset({
    # Floating point symbolic operations
    "Iop_SinF64", "Iop_CosF64", "Iop_TanF64", "Iop_2xm1F64",
    "Iop_AtanF64", "Iop_SinF32", "Iop_CosF32",

    # Complex float conversions with rounding
    "Iop_RoundF64toInt", "Iop_RoundF32toInt",

    # Vector floating point
    "Iop_RecipEst32Fx4", "Iop_RSqrtEst32Fx4",
})


@pytest.fixture
def is_rust_supported():
    """Check if an operation is supported by Rust engine."""
    def _check(op_name: str) -> bool:
        if op_name in RUST_UNSUPPORTED_OPS:
            return False
        return True

    return _check
