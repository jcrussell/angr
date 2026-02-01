"""
Pytest configuration for differential testing.
"""
from __future__ import annotations

import pytest
import logging

# Set up logging to reduce noise
logging.getLogger("angr").setLevel(logging.WARNING)
logging.getLogger("claripy").setLevel(logging.WARNING)
logging.getLogger("cle").setLevel(logging.WARNING)

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
        "markers", "slow: mark test as slow running (Tier 2+)"
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
def rust_engine_x86():
    """Create x86 Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    return RustVEXEngineWrapper("x86")


@pytest.fixture
def rust_engine_amd64():
    """Create AMD64 Rust engine wrapper."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    return RustVEXEngineWrapper("amd64")
