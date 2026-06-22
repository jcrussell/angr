"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import pytest

import angr

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestNativeIdentitySyscalls:
    """angr-0hif.3: native ``getpid`` / ``getppid`` / ``gettid`` / ``getuid``
    / ``geteuid`` / ``getgid`` / ``getegid`` handlers must short-circuit the
    Python callback path (``syscall_python_fallback_count`` stays 0).

    The Rust unit tests in ``native/angr/src/syscalls/identity.rs`` already
    pin the return values (pid=1337, ppid=1336, uid/gid=1000); this is the
    cross-the-FFI dispatch check.
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (39, "getpid"),
            (110, "getppid"),
            (186, "gettid"),
            (102, "getuid"),
            (107, "geteuid"),
            (104, "getgid"),
            (108, "getegid"),
        ],
    )
    def test_identity_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeSetuidSetgidSyscalls:
    """angr-pqgu: native ``setuid`` / ``setgid`` handlers mirror the
    Python ``syscall_stub`` ReturnUnconstrained fallback. The Rust
    cargo unit test ``setuid_setgid_return_fresh_symbolic`` in
    ``native/angr/src/syscalls/identity.rs`` pins the symbolic-return
    invariant (fresh BV each call, width == arch().bits()); this is
    the cross-the-FFI dispatch check (``syscall_python_fallback_count``
    stays 0).
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (105, "setuid"),
            (106, "setgid"),
        ],
    )
    def test_set_id_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        state.regs.rdi = 0  # concrete uid/gid arg (ignored by handler)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeMemoryExtraSyscalls:
    """angr-0hif.4: native ``madvise`` / ``mremap`` / ``msync`` / ``mlock``
    / ``munlock`` / ``mlockall`` / ``munlockall`` handlers mirror the
    Python ``syscall_stub`` ReturnUnconstrained fallback. The Rust cargo
    unit test ``memory_extras_return_fresh_symbolic_on_all_arches`` in
    ``native/angr/src/syscalls/memory_extras.rs`` pins the symbolic-return
    invariant (fresh BV each call, width == arch().bits()); this is the
    cross-the-FFI dispatch check (``syscall_python_fallback_count`` stays
    0).
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (28, "madvise"),
            (25, "mremap"),
            (26, "msync"),
            (149, "mlock"),
            (150, "munlock"),
            (151, "mlockall"),
            (152, "munlockall"),
        ],
    )
    def test_memory_extra_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        # Concrete args (ignored by handler).
        for reg in ("rdi", "rsi", "rdx", "r10", "r8"):
            setattr(state.regs, reg, 0)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeSignalSyscalls:
    """angr-0hif.6: native ``kill`` / ``tgkill`` / ``rt_sigreturn`` /
    ``pause`` / ``alarm`` handlers. ``kill``, ``rt_sigreturn``, ``pause``,
    and ``alarm`` have no Python ``SimProcedure`` and mirror the
    ``syscall_stub`` ReturnUnconstrained fallback (fresh symbolic BV per
    call). ``tgkill`` returns concrete 0, matching
    ``procedures/linux_kernel/tgkill.py``. ``rt_sigaction`` is already
    covered by ``syscalls/sigaction.rs``; ``rt_sigprocmask`` is
    intentionally NOT native (its Python impl mutates
    ``state.posix.sigmask`` which ``RustSimState`` does not carry).

    The Rust cargo unit tests in ``native/angr/src/syscalls/signals.rs``
    pin the per-handler invariants; this is the cross-the-FFI dispatch
    check (``syscall_python_fallback_count`` stays 0 for each one).
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (62, "kill"),
            (234, "tgkill"),
            (15, "rt_sigreturn"),
            (34, "pause"),
            (37, "alarm"),
        ],
    )
    def test_signal_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        for reg in ("rdi", "rsi", "rdx", "r10", "r8"):
            setattr(state.regs, reg, 0)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeResourceLimitSyscalls:
    """angr-0hif.7: native ``getrlimit`` / ``setrlimit`` / ``prlimit64``
    handlers. ``getrlimit`` mirrors ``procedures/linux_kernel/getrlimit.py``
    — RLIMIT_STACK (resource=3) writes 8388608 + fresh symbolic to
    ``*rlim`` and returns 0; other resources return a fresh symbolic.
    ``setrlimit`` and ``prlimit64`` have no Python ``SimProcedure`` and
    mirror the ``syscall_stub`` ReturnUnconstrained fallback. Rust unit
    tests in ``native/angr/src/syscalls/rlimit.rs`` pin the per-handler
    invariants; this is the cross-FFI dispatch + integration check.
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (97, "getrlimit"),
            (160, "setrlimit"),
            (302, "prlimit64"),
        ],
    )
    def test_rlimit_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        # Concrete args. For getrlimit pass resource != 3 so we hit the
        # symbolic-return branch (no memory write needed).
        state.regs.rdi = 1  # resource (RLIMIT_FSIZE for getrlimit)
        state.regs.rsi = 0  # rlim* (unused on non-stack branch)
        state.regs.rdx = 0
        state.regs.r10 = 0
        state.regs.r8 = 0

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )

    def test_getrlimit_rlimit_stack_writes_concrete_cur(self):
        """RLIMIT_STACK branch must populate ``*rlim`` with 8388608 as
        ``rlim_cur`` (8 bytes LE), matching Python's
        ``procedures/linux_kernel/getrlimit.py``.

        We allocate a small mapped page for ``rlim``, fire the syscall
        with ``rdi=3, rsi=<page>``, and read back ``state.memory[page:8]``.
        """

        shellcode = b"\x0f\x05" + b"\x90" * 0x100
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = 97  # getrlimit
        state.regs.rdi = 3  # RLIMIT_STACK
        rlim_addr = 0x500000
        state.memory.map_region(rlim_addr, 0x1000, 0b110)  # RW
        state.regs.rsi = rlim_addr
        state.regs.rdx = 0
        state.regs.r10 = 0
        state.regs.r8 = 0

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0
        # Successor should still be reachable; load *rlim and confirm.
        all_states = list(mgr.active) + list(mgr.deadended) + list(mgr.found)
        assert all_states, "expected at least one state after getrlimit"
        s = all_states[0]
        cur = s.memory.load(rlim_addr, 8, endness="Iend_LE")
        cur_val = s.solver.eval(cur)
        assert cur_val == 8388608, f"RLIMIT_STACK rlim_cur should be 8388608, got {cur_val}"


class TestNativeSimTimeSyscalls:
    """angr-0y0v: native ``gettimeofday`` (96), ``time`` (201), and
    ``clock_gettime`` (228) handlers in
    ``native/angr/src/syscalls/sim_time.rs`` always write a fresh symbolic
    ``timeval``/``timespec`` (``USE_SYSTEM_TIMES`` is rejected, not honored
    — see ``_REJECTED_OPTION_NAMES``). Rust unit tests pin the per-handler
    semantics; this is the cross-FFI dispatch check (each takes the Rust
    fast path with no Python fallback). Args are 0 so each hits its
    no-write early return (``tv``/``ts`` null → -1; ``time`` ptr null →
    symbolic rax; ``clock_gettime`` ``which_clock``=0 is CLOCK_REALTIME).
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (96, "gettimeofday"),
            (201, "time"),
            (228, "clock_gettime"),
        ],
    )
    def test_sim_time_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeDirectorySyscalls:
    """angr-0y0v: native directory-family handlers in
    ``native/angr/src/syscalls/directory.rs``. ``getcwd`` (79) is a real
    handler (``size``=0 → ERANGE, no write); ``fchdir`` (81), ``rename``
    (82), ``mkdir`` (83), ``rmdir`` (84), and ``unlink`` (87) are
    ``stub_syscall!`` handlers that return a fresh symbolic regardless of
    args. Rust unit tests pin per-handler semantics; this is the cross-FFI
    dispatch check (Rust fast path, no Python fallback).
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (79, "getcwd"),
            (81, "fchdir"),
            (82, "rename"),
            (83, "mkdir"),
            (84, "rmdir"),
            (87, "unlink"),
        ],
    )
    def test_directory_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestRustRejectsUseSystemTimes:
    """angr-0y0v: ``USE_SYSTEM_TIMES`` is warn-once rejected (policy (b)) —
    the native sim_time handlers always return a fresh symbolic value and
    never consult the option, so we warn rather than silently diverge.
    """

    def test_use_system_times_warns_once(self):

        proj = angr.load_shellcode(b"\x0f\x05" + b"\x90" * 0x100, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000, add_options={angr.options.USE_SYSTEM_TIMES})

        with pytest.warns(UserWarning, match="USE_SYSTEM_TIMES"):
            RustExplorationManager(proj, [state])


class TestNativeReadlinkSyscall:
    """angr-wv38: native ``readlink`` returns ``-1`` for every path
    because the Rust ``FileSystem`` has no symlinks (EINVAL for known
    paths, ENOENT for unknown). The buffer is left untouched.

    Rust cargo tests in ``native/angr/src/syscalls/file_path.rs`` pin
    the per-handler semantics (unknown→-1, known→-1, empty→-1, buf
    untouched, symbolic-pathname fallback, cross-arch). This Python
    test pins cross-FFI dispatch (``syscall_python_fallback_count``
    stays 0).
    """

    def test_readlink_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 89  # readlink
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0x4000  # pathname
        state.regs.rsi = 0x5000  # buf (must not be touched)
        state.regs.rdx = 256  # bufsiz

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native readlink(89) must take the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeReadlinkatSyscall:
    """angr-wv38: native ``readlinkat`` also returns ``-1`` for every
    path. The dirfd is validated (must be concrete) and the
    absolute / ``AT_FDCWD`` / relative-with-arbitrary-dirfd branches
    exist for symmetry with ``faccessat`` / ``openat`` — but the
    result is always ``-1``.

    Rust cargo tests pin per-handler semantics (unknown→-1, known→-1,
    AT_FDCWD vs arbitrary dirfd vs relative path, empty→-1, symbolic
    dirfd / pathname fallback, cross-arch). This Python test pins
    cross-FFI dispatch (``syscall_python_fallback_count`` stays 0).
    """

    def test_readlinkat_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 267  # readlinkat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0xFFFFFFFFFFFFFF9C  # AT_FDCWD
        state.regs.rsi = 0x4000  # pathname
        state.regs.rdx = 0x5000  # buf (untouched)
        state.regs.r10 = 256  # bufsiz

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "native readlinkat(267) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestSeededSymlinkReadlink:
    """angr-m7s7y: a Python harness can seed pre-existing symlinks via the
    ``RustExplorationManager(..., symlinks=...)`` kwarg, which forwards to the
    native ``PyRustSimState.register_symlink`` / ``FileSystem::add_symlink``
    setter. With an entry registered, native ``readlink`` resolves the link
    (writes the raw target bytes, NOT NUL-terminated, returns the count)
    instead of returning ``-1``. Unseeded links keep returning ``-1`` (the
    pre-6.2 behavior — pinned by ``TestNativeReadlinkSyscall``).
    """

    def test_seeded_symlink_readlink_resolves(self):

        # syscall; jmp self — the trailing self-loop pins the post-syscall
        # state at 0x1002 so its rax (the return value) survives instead of
        # diverging into unmapped memory and getting filled.
        shellcode = b"\x0f\x05\xeb\xfe"
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        target = b"/real/target"
        state.memory.store(0x4000, b"/link\x00")
        buf_addr = 0x500000
        state.memory.map_region(buf_addr, 0x1000, 0b110)  # RW
        state.regs.rax = 89  # readlink
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0x4000  # pathname -> "/link"
        state.regs.rsi = buf_addr  # buf
        state.regs.rdx = 256  # bufsiz

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True, symlinks={"/link": target})
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"seeded readlink must stay on the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )
        all_states = list(mgr.active) + list(mgr.deadended) + list(mgr.found)
        assert all_states, "expected at least one state after readlink"
        s = all_states[0]
        ret = s.solver.eval(s.regs.rax)
        assert ret == len(target), f"readlink should return target len {len(target)}, got {ret}"
        written = s.solver.eval(s.memory.load(buf_addr, len(target)), cast_to=bytes)
        assert written == target, f"buf should hold the raw target, got {written!r}"


class TestNativeFaccessatSyscall:
    """angr-6009: native ``faccessat`` mirrors ``NativeAccessSyscall``
    with dirfd handling. Absolute paths and ``AT_FDCWD`` query
    ``FileSystem::is_path_known``; relative paths with non-AT_FDCWD
    dirfd return ``-1`` (matches ``NativeOpenatSyscall``'s policy — we
    do not model directory fds).

    Rust cargo tests in ``native/angr/src/syscalls/file_path.rs`` pin
    the per-handler semantics (unknown→-1, known→0, AT_FDCWD vs
    arbitrary dirfd, empty path→-1, symbolic fallback,
    cross-arch). This Python test pins cross-FFI dispatch
    (``syscall_python_fallback_count`` stays 0).
    """

    def test_faccessat_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 269  # faccessat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0xFFFFFFFFFFFFFF9C  # AT_FDCWD (-100 reinterpreted u64)
        state.regs.rsi = 0x4000  # pathname
        state.regs.rdx = 0  # mode = F_OK

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "native faccessat(269) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeFdAllocatingSyscalls:
    """angr-k3ol.1: native ``open`` / ``openat`` / ``close`` allocate /
    release FDs in ``RustSimState::file_system()``. They mirror the
    existing ``procedures/fileops::NativeOpen`` / ``NativeClose`` libc
    procs and do NOT mirror Python's ``state.posix.fd`` / ``state.fs``
    — same trade-off as ``dup``/``dup2``. The Rust cargo tests in
    ``native/angr/src/syscalls/file_path.rs`` pin per-handler semantics
    (fresh fd, NEG_ONE for empty/relative-without-AT_FDCWD paths, fd
    book-keeping for close); this is the cross-the-FFI dispatch check
    (``syscall_python_fallback_count`` stays 0 for open / openat /
    close on amd64).

    The companion ``stat`` (4) / ``fstat`` (5) syscalls still fall
    back to Python (need per-arch struct stat layouts +
    ``state.posix.fstat_with_result``). ``access`` (21) is covered by
    ``TestNativeAccessSyscall`` below — it queries
    ``FileSystem::known_paths``.
    """

    @pytest.mark.parametrize(
        "syscall_num,label,setup_args",
        [
            # open: rdi = path_addr, rsi = O_RDONLY, rdx = mode
            (2, "open", {"rdi": 0x4000, "rsi": 0, "rdx": 0}),
            # openat: rdi = AT_FDCWD (-100 as u32), rsi = path_addr, rdx = O_RDONLY
            (257, "openat", {"rdi": 0xFFFFFFFFFFFFFF9C, "rsi": 0x4000, "rdx": 0, "r10": 0}),
            # close: rdi = fd (use 0 = stdin which is pre-open)
            (3, "close", {"rdi": 0}),
        ],
    )
    def test_fd_alloc_syscall_dispatches_natively(self, syscall_num, label, setup_args):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/tmp/k3ol\x00")
        state.regs.rax = syscall_num
        # Zero out then apply the setup args.
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        for reg, val in setup_args.items():
            setattr(state.regs, reg, val)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeAccessSyscall:
    """angr-k3ol.2: native ``access`` looks up the path in the Rust
    ``FileSystem::known_paths`` set (populated by ``open`` / ``openat``
    or seeded by ``register_known_path``) and returns ``0`` if known,
    ``-1`` otherwise. Mirrors ``procedures/linux_kernel/access.py`` —
    the Python proc returns ``-1`` when ``state.fs.get(path)`` is
    ``None`` and ``0`` otherwise.

    Pre-populated Python ``state.fs`` entries are NOT mirrored into
    the Rust side automatically — same trade-off as the FD-allocating
    handlers in ``TestNativeFdAllocatingSyscalls``. Rust cargo tests
    in ``native/angr/src/syscalls/file_path.rs`` pin the per-handler
    semantics (unknown→-1, known→0, empty path→-1, symbolic
    fallback); this test pins cross-FFI dispatch
    (``syscall_python_fallback_count`` stays 0).
    """

    def test_access_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 21  # access
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0x4000  # pathname
        state.regs.rsi = 0  # mode = F_OK

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native access(21) must take the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeFstatSyscall:
    """angr-k3ol.3: native ``fstat`` looks up ``content_len`` via the
    Rust ``FileSystem::fd_info(fd)`` and writes a per-arch
    ``struct stat`` (AMD64 + ARM64 only) to the buffer. Mirrors
    ``procedures/linux_kernel/fstat.py::run`` semantics with two
    intentional divergences: ``st_mode`` is the concrete
    ``S_IFREG | 0o755`` instead of a fresh ``BVS`` (the Python proc
    mints one in ``state.posix.fstat_with_result``) and ``st_size``
    is concrete (``content_len`` of the fd's backing buffer).

    Unknown fd → ``-1`` (matches ``fstat_with_result``'s ``result=-1``
    branch). Rust cargo tests in
    ``native/angr/src/syscalls/file_path.rs`` pin the per-arch field
    offsets and the symbolic/unmapped fallback paths; this test pins
    cross-FFI dispatch (``syscall_python_fallback_count`` stays 0).
    """

    def test_fstat_unknown_fd_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        # statbuf at 0x4000 — page is mapped by blank_state setup or the
        # store would fault, but with an unknown fd the handler returns
        # NEG_ONE before touching memory, so we do not need to map.
        state.regs.rax = 5  # fstat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 99  # fd that was never opened
        state.regs.rsi = 0x4000  # statbuf

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native fstat(5) must take the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeStatSyscall:
    """angr-k3ol.4: native ``stat`` resolves ``pathname`` via
    ``read_path``, queries ``FileSystem::is_path_known`` (returning
    ``-1`` for empty / unknown paths) and reuses ``write_amd64_stat``
    when the path is known. AMD64 only — ARM64's asm-generic ABI
    dropped legacy ``stat``. Diverges from
    ``procedures/linux_kernel/stat.py``'s open→fstat→close in that the
    Rust path never mutates the fd table.

    Rust cargo tests in ``native/angr/src/syscalls/file_path.rs`` pin
    the per-handler semantics (unknown→-1, empty→-1, known→0,
    largest-content-len-across-fds, unsupported-arch, symbolic fd /
    statbuf fallback, unmapped buf MemoryError). This Python test pins
    cross-FFI dispatch (``syscall_python_fallback_count`` stays 0).
    """

    def test_stat_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 4  # stat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0x4000  # pathname
        state.regs.rsi = 0x5000  # statbuf (unread on the failure path)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native stat(4) must take the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeLstatSyscall:
    """angr-poao: native ``lstat`` collapses to ``stat`` semantics
    (the Rust ``FileSystem`` has no symlinks), reusing
    ``write_amd64_stat`` against ``FileSystem::content_size_for_path``.
    AMD64 only — ARM64's asm-generic ABI dropped legacy ``lstat``;
    x86 / ARM EABI / MIPS32 carry the legacy 32-bit ``struct stat``
    with no Python proc.

    Rust cargo tests in ``native/angr/src/syscalls/file_path.rs`` pin
    the per-handler semantics (unknown→-1, empty→-1, known→0,
    unsupported-arch, symbolic-pathname fallback, unmapped-buf
    MemoryError). This Python test pins cross-FFI dispatch
    (``syscall_python_fallback_count`` stays 0).
    """

    def test_lstat_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 6  # lstat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0x4000  # pathname
        state.regs.rsi = 0x5000  # statbuf (unread on the failure path)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native lstat(6) must take the Rust fast path (got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeNewfstatatSyscall:
    """angr-poao: native ``newfstatat`` adds ``openat``-style dirfd
    handling on top of ``stat`` semantics. Absolute paths and
    ``AT_FDCWD`` resolve via ``FileSystem``; relative paths with any
    other dirfd return ``-1``. ``AT_EMPTY_PATH`` is ignored (deferred
    follow-up — would dispatch to ``fstat(dirfd)``). Per-arch struct
    stat layout via ``write_amd64_stat`` / ``write_aarch64_stat``.

    Rust cargo tests pin per-handler semantics (unknown→-1, known
    AMD64/ARM64 layout, absolute path ignores dirfd, relative +
    non-AT_FDCWD→-1, empty→-1, unsupported-arch, symbolic fallbacks,
    unmapped-buf MemoryError). This Python test pins cross-FFI
    dispatch on AMD64 (``syscall_python_fallback_count`` stays 0).
    """

    def test_newfstatat_unknown_path_dispatches_natively(self):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.memory.store(0x4000, b"/no/such/path\x00")
        state.regs.rax = 262  # newfstatat
        for reg in ("rdi", "rsi", "rdx", "r10", "r8", "r9"):
            setattr(state.regs, reg, 0)
        state.regs.rdi = 0xFFFFFFFFFFFFFF9C  # AT_FDCWD
        state.regs.rsi = 0x4000  # pathname
        state.regs.rdx = 0x5000  # statbuf (unread on failure)
        state.regs.r10 = 0  # flag

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            "native newfstatat(262) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeConcurrencySyscalls:
    """angr-0hif.7: native ``futex`` / ``eventfd`` / ``eventfd2`` /
    ``epoll_create`` / ``epoll_create1`` / ``epoll_ctl`` / ``epoll_wait``
    handlers. ``futex`` mirrors ``procedures/linux_kernel/futex.py``
    (FUTEX_WAKE returns 0, else symbolic). The other six fall through
    to ``syscall_stub`` in Python and emit fresh symbolic from native.
    angr is single-threaded symex; blocking is never modeled.
    """

    @pytest.mark.parametrize(
        "syscall_num,label,futex_op",
        [
            (202, "futex_wake", 1),  # FUTEX_WAKE -> concrete 0
            (202, "futex_wait", 0),  # FUTEX_WAIT -> symbolic
            (284, "eventfd", 0),
            (290, "eventfd2", 0),
            (213, "epoll_create", 0),
            (291, "epoll_create1", 0),
            (233, "epoll_ctl", 0),
            (232, "epoll_wait", 0),
        ],
    )
    def test_concurrency_syscall_dispatches_natively(self, syscall_num, label, futex_op):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        state.regs.rdi = 0
        state.regs.rsi = futex_op
        state.regs.rdx = 0
        state.regs.r10 = 0
        state.regs.r8 = 0
        state.regs.r9 = 0

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )


class TestNativeFileDescriptorSyscalls:
    """angr-0hif.5 stub-fallthrough subset: ``fcntl`` / ``ioctl`` /
    ``pipe`` / ``pipe2`` handlers. None of these have a Python
    ``SimProcedure`` bound in ``definitions/linux_kernel.py`` —
    ``posix/fcntl.py`` is libc-side only — so the unhandled-syscall
    path in pure Python angr falls through to
    ``procedures/stubs/syscall_stub.py``. The native handlers mirror
    that with a fresh symbolic of ``arch().bits()``.

    ``dup`` / ``dup2`` / ``dup3`` are intentionally NOT native — they
    have ``posix/dup.py`` procs that mutate ``state.posix.fd``, which
    needs the FD table plumbed into ``RustSimState`` (same blocker as
    angr-k3ol). The dispatcher falls back to Python for those so the
    side-effects continue to apply.

    Rust unit tests in ``native/angr/src/syscalls/file_descriptor.rs``
    pin the per-handler invariants (correct ``name()``/arity, fresh
    symbolic on each call). This is the cross-FFI dispatch check.
    """

    @pytest.mark.parametrize(
        "syscall_num,label",
        [
            (16, "ioctl"),
            (22, "pipe"),
            (72, "fcntl"),
            (293, "pipe2"),
        ],
    )
    def test_fd_control_syscall_dispatches_natively(self, syscall_num, label):

        shellcode = b"\x0f\x05" + b"\x90" * 0x100  # syscall; nop pad
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rax = syscall_num
        for reg in ("rdi", "rsi", "rdx", "r10", "r8"):
            setattr(state.regs, reg, 0)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)

        stats = mgr._rust_mgr.stats()
        assert stats["syscall_python_fallback_count"] == 0, (
            f"native {label}({syscall_num}) must take the Rust fast path "
            f"(got fallback={stats['syscall_python_fallback_count']})"
        )
