from __future__ import annotations

import claripy

import angr


class rawmemchr(angr.SimProcedure):
    # pylint:disable=arguments-differ, missing-class-docstring

    def run(self, s_addr, c_int):
        # rawmemchr is memchr without a length bound: the caller guarantees the
        # byte is present (otherwise the behavior is undefined). We bound the
        # search by libc.max_buffer_size and reuse the memchr SimProcedure so
        # the scan/constraint logic lives in one place.
        n = claripy.BVV(self.state.libc.max_buffer_size, self.state.arch.bits)
        memchr = self.inline_call(angr.SIM_PROCEDURES["libc"]["memchr"], s_addr, c_int, n)
        return memchr.ret_expr
