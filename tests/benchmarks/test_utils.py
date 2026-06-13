"""Shared test utilities for benchmark scripts."""

from __future__ import annotations

import io


class BufferedStringIO(io.StringIO):
    """StringIO with a buffer attribute for compatibility with code that uses stdout.buffer."""

    def __init__(self):
        super().__init__()
        self._buffer = io.BytesIO()

    @property
    def buffer(self):
        return self._buffer

    def getvalue(self) -> str:
        """Get combined output from both text and binary writes."""
        text_output = super().getvalue()
        binary_output = self._buffer.getvalue()
        if binary_output:
            try:
                text_output += binary_output.decode("utf-8", errors="replace")
            except Exception:
                pass
        return text_output
