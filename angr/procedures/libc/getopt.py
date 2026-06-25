from __future__ import annotations

import logging

import claripy

import angr

l = logging.getLogger(name=__name__)

# struct option { const char *name; int has_arg; int *flag; int val; };
_NO_ARGUMENT = 0
_REQUIRED_ARGUMENT = 1
_OPTIONAL_ARGUMENT = 2

_DASH = ord("-")
_COLON = ord(":")


def _eval_ptr(state, addr):
    """Load a pointer-sized word at ``addr``; return a concrete int or None if symbolic."""
    word = state.memory.load(addr, state.arch.bytes, endness=state.arch.memory_endness)
    if state.solver.symbolic(word):
        return None
    return state.solver.eval(word)


def _concrete_cstr(state, ptr, cap=4096):
    """Read a NUL-terminated string at ``ptr``. Returns bytes (no NUL) or None if any byte is symbolic."""
    out = bytearray()
    for i in range(cap):
        b = state.memory.load(ptr + i, 1)
        if state.solver.symbolic(b):
            return None
        v = state.solver.eval(b)
        if v == 0:
            return bytes(out)
        out.append(v)
    return bytes(out)


def _parse_optstring(s):
    """Return ({char: has_arg}, leading_colon) for an optstring (POSIX/glibc syntax)."""
    opts = {}
    i = 0
    # leading mode chars ('+'/'-' affect ordering, which we do not model)
    while i < len(s) and s[i] in (ord("+"), _DASH):
        i += 1
    leading_colon = i < len(s) and s[i] == _COLON
    if leading_colon:
        i += 1
    while i < len(s):
        c = s[i]
        i += 1
        nargs = _NO_ARGUMENT
        if i < len(s) and s[i] == _COLON:
            nargs = _REQUIRED_ARGUMENT
            i += 1
            if i < len(s) and s[i] == _COLON:
                nargs = _OPTIONAL_ARGUMENT
                i += 1
        opts[c] = nargs
    return opts, leading_colon


class _GetOptBase(angr.SimProcedure):
    """Shared concrete-argv getopt engine.

    Faithful for the common case where ``argv``, the option strings, and the
    relevant bytes are concrete. When anything required is symbolic we fall
    back to an unconstrained return (preserving the pre-SimProc behaviour) so
    we never branch on a symbolic option scan. POSIX non-permuting semantics:
    option scanning stops at the first non-option operand (glibc's '+' /
    POSIXLY_CORRECT mode); argv is never reordered.
    """

    # ---- global-variable plumbing (optind / optarg / optopt / opterr) ----

    def _global_addr(self, name):
        proj = self.project if self.project is not None else self.state.project
        if proj is None:
            return None
        sym = proj.loader.find_symbol(name)
        return sym.rebased_addr if sym is not None else None

    def _store_int(self, name, value):
        addr = self._global_addr(name)
        if addr is not None:
            self.state.memory.store(
                addr,
                claripy.BVV(value, 32),
                endness=self.state.arch.memory_endness,
                inspect=False,
                disable_actions=True,
            )

    def _store_ptr(self, name, value):
        addr = self._global_addr(name)
        if addr is not None:
            self.state.memory.store(
                addr,
                claripy.BVV(value, self.state.arch.bits),
                endness=self.state.arch.memory_endness,
                inspect=False,
                disable_actions=True,
            )

    def _load_cursor(self):
        """Resolve (optind, optchar) honouring a guest reset of ``optind``."""
        plugin_optind = self.state.libc.getopt_optind
        plugin_optchar = self.state.libc.getopt_optchar
        addr = self._global_addr("optind")
        if addr is not None:
            word = self.state.memory.load(addr, 4, endness=self.state.arch.memory_endness)
            if not self.state.solver.symbolic(word):
                guest = self.state.solver.eval(word)
                if guest == 0:
                    return 1, 0  # glibc: optind==0 requests a full reset
                if guest >= 1:
                    # honour a guest rescan (optind reset to a smaller index)
                    return guest, (plugin_optchar if guest == plugin_optind else 0)
        return plugin_optind, plugin_optchar

    def _save_cursor(self, optind, optchar):
        self.state.libc.getopt_optind = optind
        self.state.libc.getopt_optchar = optchar
        self._store_int("optind", optind)

    @staticmethod
    def _unconstrained():
        return claripy.BVS("getopt", 32)

    @staticmethod
    def _ret_char(c):
        return claripy.BVV(c, 32)

    @staticmethod
    def _ret_done():
        return claripy.BVV(0xFFFFFFFF, 32)  # -1

    # ---- long-option support ----

    def _handle_long(
        self,
        name_part,
        prefix_len,
        longopts_addr,
        has_longindex,
        longindex_addr,
        optind,
        argc,
        argv_ptr,
        leading_colon,
    ):
        """Match a long option (name, possibly ``name=value``). Returns a BV
        result, or None when there is no match (so getopt_long_only can fall
        back to short-option parsing)."""
        state = self.state
        ps = state.arch.bytes
        stride = 4 * ps  # name(ps) has_arg(int,aligned) flag(ptr) val(int) -> 4*ps with padding

        eq = name_part.find(b"=")
        if eq >= 0:
            name = name_part[:eq]
            inline_val = name_part[eq + 1 :]
        else:
            name = name_part
            inline_val = None

        # scan the option table for an exact match, then a unique prefix match
        ptr = longopts_addr
        exact = None
        prefixes = []
        idx = 0
        while True:
            name_ptr = _eval_ptr(state, ptr)
            if name_ptr is None:
                return self._unconstrained()
            if name_ptr == 0:
                break
            optname = _concrete_cstr(state, name_ptr)
            if optname is None:
                return self._unconstrained()
            if optname == name:
                exact = (idx, ptr)
                break
            if name and optname.startswith(name):
                prefixes.append((idx, ptr))
            idx += 1
            ptr += stride

        if exact is not None:
            match = exact
        elif len(prefixes) == 1:
            match = prefixes[0]
        elif len(prefixes) > 1:
            # ambiguous abbreviation
            self._store_int("optopt", 0)
            self._store_ptr("optarg", 0)
            self._save_cursor(optind + 1, 0)
            return self._ret_char(ord("?"))
        else:
            return None  # no match

        match_idx, match_ptr = match
        has_arg = state.memory.load(match_ptr + ps, 4, endness=state.arch.memory_endness)
        has_arg = state.solver.eval(has_arg) if not state.solver.symbolic(has_arg) else _NO_ARGUMENT
        flag_ptr = _eval_ptr(state, match_ptr + 2 * ps)
        val_word = state.memory.load(match_ptr + 3 * ps, 4, endness=state.arch.memory_endness)
        val = state.solver.eval(val_word) if not state.solver.symbolic(val_word) else 0

        if has_longindex and longindex_addr:
            state.memory.store(
                longindex_addr,
                claripy.BVV(match_idx, 32),
                endness=state.arch.memory_endness,
                inspect=False,
                disable_actions=True,
            )

        # resolve the option-argument
        new_optind = optind + 1
        optarg = 0
        if has_arg == _REQUIRED_ARGUMENT:
            if inline_val is not None:
                # optarg points just past the '=' in the original argv element
                elem_ptr = _eval_ptr(state, argv_ptr + optind * ps)
                optarg = (elem_ptr + prefix_len + (len(name_part) - len(inline_val))) if elem_ptr is not None else 0
            elif optind + 1 < argc:
                optarg = _eval_ptr(state, argv_ptr + (optind + 1) * ps) or 0
                new_optind = optind + 2
            else:
                # missing required argument
                self._store_int("optopt", val if flag_ptr == 0 else 0)
                self._store_ptr("optarg", 0)
                self._save_cursor(new_optind, 0)
                return self._ret_char(_COLON if leading_colon else ord("?"))
        elif has_arg == _OPTIONAL_ARGUMENT and inline_val is not None:
            elem_ptr = _eval_ptr(state, argv_ptr + optind * ps)
            optarg = (elem_ptr + (len(name_part) - len(inline_val))) if elem_ptr is not None else 0

        self._store_ptr("optarg", optarg)
        self._save_cursor(new_optind, 0)

        if flag_ptr:
            state.memory.store(
                flag_ptr,
                claripy.BVV(val, 32),
                endness=state.arch.memory_endness,
                inspect=False,
                disable_actions=True,
            )
            return self._ret_char(0)
        return self._ret_char(val)

    # ---- main engine ----

    def _getopt(self, argc, argv, optstring, longopts=None, longindex=None, long_only=False):
        state = self.state
        ps = state.arch.bytes

        if state.solver.symbolic(argc):
            return self._unconstrained()
        argc = state.solver.eval(argc)
        if state.solver.symbolic(argv):
            return self._unconstrained()
        argv_ptr = state.solver.eval(argv)
        if state.solver.symbolic(optstring):
            return self._unconstrained()
        optstring_bytes = _concrete_cstr(state, state.solver.eval(optstring))
        if optstring_bytes is None:
            return self._unconstrained()

        longopts_addr = None
        if longopts is not None and not state.solver.symbolic(longopts):
            longopts_addr = state.solver.eval(longopts)
            if longopts_addr == 0:
                longopts_addr = None
        longindex_addr = None
        has_longindex = longindex is not None and not state.solver.symbolic(longindex)
        if has_longindex:
            longindex_addr = state.solver.eval(longindex)
            has_longindex = longindex_addr != 0

        opts, leading_colon = _parse_optstring(optstring_bytes)
        optind, optchar = self._load_cursor()

        if optind >= argc:
            self._save_cursor(optind, 0)
            self._store_ptr("optarg", 0)
            return self._ret_done()

        elem_ptr = _eval_ptr(state, argv_ptr + optind * ps)
        if elem_ptr is None:
            return self._unconstrained()
        if elem_ptr == 0:
            self._save_cursor(optind, 0)
            return self._ret_done()
        arg = _concrete_cstr(state, elem_ptr)
        if arg is None:
            return self._unconstrained()

        if optchar == 0:
            if not arg or arg[0] != _DASH or arg == b"-":
                # non-option operand -> stop (non-permuting)
                self._save_cursor(optind, 0)
                return self._ret_done()
            if arg == b"--":
                self._save_cursor(optind + 1, 0)
                return self._ret_done()
            if longopts_addr is not None:
                long_attempt = None
                prefix_len = 0
                if arg.startswith(b"--"):
                    long_attempt = arg[2:]
                    prefix_len = 2
                elif long_only:
                    long_attempt = arg[1:]
                    prefix_len = 1
                if long_attempt is not None:
                    res = self._handle_long(
                        long_attempt,
                        prefix_len,
                        longopts_addr,
                        has_longindex,
                        longindex_addr,
                        optind,
                        argc,
                        argv_ptr,
                        leading_colon,
                    )
                    if res is not None:
                        return res
                    if arg.startswith(b"--"):
                        # unknown "--long" never falls back to short parsing
                        self._store_int("optopt", 0)
                        self._store_ptr("optarg", 0)
                        self._save_cursor(optind + 1, 0)
                        return self._ret_char(ord("?"))
                    # long_only single-dash with no match -> try short below
            optchar = 1

        c = arg[optchar]
        spec = opts.get(c)
        if spec is None or c == _COLON:
            self._store_int("optopt", c)
            self._store_ptr("optarg", 0)
            optchar += 1
            if optchar >= len(arg):
                optind += 1
                optchar = 0
            self._save_cursor(optind, optchar)
            return self._ret_char(ord("?"))

        if spec == _NO_ARGUMENT:
            self._store_ptr("optarg", 0)
            optchar += 1
            if optchar >= len(arg):
                optind += 1
                optchar = 0
            self._save_cursor(optind, optchar)
            return self._ret_char(c)

        # option takes an argument (required or optional)
        if optchar + 1 < len(arg):
            self._store_ptr("optarg", elem_ptr + optchar + 1)
            self._save_cursor(optind + 1, 0)
            return self._ret_char(c)
        if spec == _OPTIONAL_ARGUMENT:
            self._store_ptr("optarg", 0)
            self._save_cursor(optind + 1, 0)
            return self._ret_char(c)
        # required argument from the next argv element
        if optind + 1 < argc:
            next_ptr = _eval_ptr(state, argv_ptr + (optind + 1) * ps)
            if next_ptr is None:
                return self._unconstrained()
            self._store_ptr("optarg", next_ptr)
            self._save_cursor(optind + 2, 0)
            return self._ret_char(c)
        # missing required argument
        self._store_int("optopt", c)
        self._store_ptr("optarg", 0)
        self._save_cursor(optind + 1, 0)
        return self._ret_char(_COLON if leading_colon else ord("?"))


class getopt(_GetOptBase):
    # pylint:disable=arguments-differ
    def run(self, argc, argv, optstring):
        return self._getopt(argc, argv, optstring)


class getopt_long(_GetOptBase):
    # pylint:disable=arguments-differ
    def run(self, argc, argv, optstring, longopts, longindex):
        return self._getopt(argc, argv, optstring, longopts=longopts, longindex=longindex)


class getopt_long_only(_GetOptBase):
    # pylint:disable=arguments-differ
    def run(self, argc, argv, optstring, longopts, longindex):
        return self._getopt(argc, argv, optstring, longopts=longopts, longindex=longindex, long_only=True)
