from __future__ import annotations

import logging

import claripy

from angr.procedures.stubs.format_parser import FormatParser

l = logging.getLogger(name=__name__)


class sprintf(FormatParser):
    # pylint:disable=arguments-differ

    def run(self, dst_ptr, fmt):  # pylint:disable=unused-argument
        # The format str is at index 1
        fmt_str = self._parse(fmt)
        out_str = fmt_str.replace(self.va_arg)
        self.state.memory.store(dst_ptr, out_str)

        # place the terminating null byte
        self.state.memory.store(
            dst_ptr + (out_str.size() // self.arch.byte_width), claripy.BVV(0, self.arch.byte_width)
        )

        return out_str.size() // self.arch.byte_width


class __sprintf_chk(sprintf):
    # _FORTIFY_SOURCE redirect: __sprintf_chk(s, flag, slen, fmt, ...). The
    # compiler injects `flag` and the destination size `slen`; angr ignores
    # both and forwards to the base sprintf (matching __printf_chk/__snprintf_chk
    # and the native fortify_printf.rs wrappers). The 4-arg run() signature keeps
    # the variadic args aligned — see the glibc.json __sprintf_chk prototype.
    def run(self, dst_ptr, flag, slen, fmt):  # pylint:disable=arguments-differ,unused-argument
        return super().run(dst_ptr, fmt)
