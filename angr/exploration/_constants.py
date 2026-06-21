"""Memory-layout constants shared across the Rust exploration glue.

PAGE_SIZE / PAGE_MASK are imported from the Rust extension so the Rust
side (native/angr/src/memory/page.rs) and the Python side stay in lockstep.
STACK_SIZE and MAX_OVERLAY_SECTION_SIZE live here because only Python
references them.
"""

from __future__ import annotations

from angr.rustylib.vex_engine import PAGE_MASK, PAGE_SIZE

STACK_SIZE = 0x11_0000
MAX_OVERLAY_SECTION_SIZE = 0x10000

__all__ = ["MAX_OVERLAY_SECTION_SIZE", "PAGE_MASK", "PAGE_SIZE", "STACK_SIZE"]
