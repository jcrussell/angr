from __future__ import annotations

import logging

import angr

l = logging.getLogger(name=__name__)


class vprintf(angr.SimProcedure):
    # pylint:disable=arguments-differ,unused-argument

    def run(self, fmt, ap):
        # va_list (`ap`) is arch-specific (x86-64 SysV reg_save_area) and
        # unmodeled by angr, so we do NOT perform %-substitution: we write the
        # raw format string, which is exact for formats without conversions.
        stdout = self.state.posix.get_fd(1)
        if stdout is None:
            return -1

        strlen = angr.SIM_PROCEDURES["libc"]["strlen"]
        length = self.inline_call(strlen, fmt).ret_expr
        stdout.write(fmt, length)
        return length
