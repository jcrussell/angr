from __future__ import annotations

import logging

import angr

l = logging.getLogger(name=__name__)


class vsprintf(angr.SimProcedure):
    # pylint:disable=arguments-differ,unused-argument

    def run(self, dst_ptr, fmt, ap):
        # va_list (`ap`) is unmodeled, so, as in vprintf/vfprintf, we copy the
        # raw format string into the destination buffer (NUL-terminated)
        # without %-substitution rather than formatting garbage arguments
        # read through the sprintf core.
        strcpy = angr.SIM_PROCEDURES["libc"]["strcpy"]
        self.inline_call(strcpy, dst_ptr, fmt)
        strlen = angr.SIM_PROCEDURES["libc"]["strlen"]
        return self.inline_call(strlen, dst_ptr).ret_expr
