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
    # ignores both and forwards to the base vsnprintf with the user-supplied
    # `maxlen` as the size argument (matching __snprintf_chk/__sprintf_chk and
    # the native fortify_printf.rs wrappers). Keeps the variadic args aligned —
    # see the glibc.json __vsnprintf_chk prototype.
    def run(self, str_ptr, maxlen, flag, slen, fmt, ap):  # pylint:disable=arguments-differ,unused-argument
        return super().run(str_ptr, maxlen, fmt, ap)
