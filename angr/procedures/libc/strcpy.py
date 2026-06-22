from __future__ import annotations

import angr


class strcpy(angr.SimProcedure):
    # pylint:disable=arguments-differ

    def run(self, dst, src):
        strlen = angr.SIM_PROCEDURES["libc"]["strlen"]
        strncpy = angr.SIM_PROCEDURES["libc"]["strncpy"]
        src_len = self.inline_call(strlen, src)

        return self.inline_call(strncpy, dst, src, src_len.ret_expr + 1, src_len=src_len.ret_expr).ret_expr


class __strcpy_chk(strcpy):
    # _FORTIFY_SOURCE redirect; the trailing destlen is ignored (matches __memcpy_chk).
    def run(self, dst, src, _destlen):  # type:ignore[reportIncompatibleMethodOverride]
        return super().run(dst, src)
