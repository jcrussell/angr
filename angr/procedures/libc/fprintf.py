from __future__ import annotations

import logging

from cle.backends.externs.simdata.io_file import io_file_data_for_arch

from angr.procedures.stubs.format_parser import FormatParser

l = logging.getLogger(name=__name__)


class fprintf(FormatParser):
    def run(self, file_ptr, fmt):  # pylint:disable=unused-argument
        fd_offset = io_file_data_for_arch(self.state.arch)["fd"]
        fileno = self.state.mem[file_ptr + fd_offset :].int.resolved
        simfd = self.state.posix.get_fd(fileno)
        if simfd is None:
            return -1

        # The format str is at index 1
        fmt_str = self._parse(fmt)
        out_str = fmt_str.replace(self.va_arg)

        simfd.write_data(out_str, out_str.size() // 8)

        return out_str.size() // 8


class __fprintf_chk(fprintf):
    # _FORTIFY_SOURCE redirect: __fprintf_chk(fp, flag, fmt, ...). The compiler
    # injects `flag`; angr ignores it and forwards to the base fprintf (matching
    # __printf_chk/__sprintf_chk and the native fortify_printf.rs wrappers). The
    # 3-arg run() signature keeps the variadic args aligned — see the glibc.json
    # __fprintf_chk prototype.
    def run(self, file_ptr, flag, fmt):  # pylint:disable=arguments-differ,unused-argument
        return super().run(file_ptr, fmt)
