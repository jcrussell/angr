from __future__ import annotations

import claripy

import angr


class strchrnul(angr.SimProcedure):
    # pylint:disable=arguments-differ, missing-class-docstring

    def run(self, s_addr, c_int):
        # strchrnul is strchr that returns a pointer to the terminating NUL
        # (rather than NULL) when the character is not found. Reuse the strlen
        # and strchr SimProcedures so the search/constraint logic stays in one
        # place.
        strlen = self.inline_call(angr.SIM_PROCEDURES["libc"]["strlen"], s_addr)
        nul_ptr = s_addr + strlen.ret_expr

        strchr = self.inline_call(angr.SIM_PROCEDURES["libc"]["strchr"], s_addr, c_int)
        a = strchr.ret_expr

        # strchr returns NULL (0) when the byte is absent; strchrnul instead
        # returns the address of the trailing NUL terminator.
        return claripy.If(a == 0, nul_ptr, a)
