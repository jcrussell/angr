from __future__ import annotations

import angr


class vsnprintf(angr.SimProcedure):
    # pylint:disable=arguments-differ

    def run(self, str_ptr, size, fmt, ap):  # pylint:disable=unused-argument
        # This function returns
        # Add another exit to the retn_addr that is at the top of the stack now

        if self.state.solver.eval(size) == 0:
            return 0

        self.state.memory.store(str_ptr, b"\x00")

        return 1


class __vsnprintf_chk(vsnprintf):
    # _FORTIFY_SOURCE redirect: __vsnprintf_chk(s, maxlen, flag, slen, fmt, ap).
    # The compiler injects `flag` and the destination object size `slen`; angr
    # ignores both and forwards to vsnprintf with the caller's `maxlen`, as
    # __snprintf_chk does for snprintf.
    def run(self, str_ptr, maxlen, flag, slen, fmt, ap):  # pylint:disable=arguments-differ,unused-argument
        return super().run(str_ptr, maxlen, fmt, ap)
