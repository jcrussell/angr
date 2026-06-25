from __future__ import annotations

import logging

from cle.backends.externs.simdata.io_file import io_file_data_for_arch

import angr

l = logging.getLogger(name=__name__)


class vfprintf(angr.SimProcedure):
    # pylint:disable=arguments-differ,unused-argument

    def run(self, file_ptr, fmt, ap):
        # va_list (`ap`) is unmodeled, so — like the native fprintf core and
        # the `vfprintf = fprintf` alias — we write the raw format string with
        # no %-substitution. Stream/fd resolution mirrors fprintf.
        fd_offset = io_file_data_for_arch(self.state.arch)["fd"]
        fileno = self.state.mem[file_ptr + fd_offset :].int.resolved
        simfd = self.state.posix.get_fd(fileno)
        if simfd is None:
            return -1

        strlen = angr.SIM_PROCEDURES["libc"]["strlen"]
        length = self.inline_call(strlen, fmt).ret_expr
        simfd.write(fmt, length)
        return length
