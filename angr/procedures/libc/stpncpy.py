from __future__ import annotations

import claripy

import angr


class stpncpy(angr.SimProcedure):
    """stpncpy"""

    # pylint:disable=arguments-differ

    def run(self, dst, src, n):
        strlen = angr.SIM_PROCEDURES["libc"]["strlen"]
        strncpy = angr.SIM_PROCEDURES["libc"]["strncpy"]
        src_len = self.inline_call(strlen, src).ret_expr

        # Bounded copy + NUL-pad of the n-byte window is exactly strncpy (DRY).
        self.inline_call(strncpy, dst, src, n, src_len=src_len)

        # stpncpy returns a pointer to the written NUL, i.e. dst + min(strlen(src), n);
        # when src has no NUL inside the n-byte window the result is dst + n.
        return dst + claripy.If(claripy.ULE(src_len, n), src_len, n)
