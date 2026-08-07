Native SimProcedure and syscall coverage matrix
===============================================

The Rust engine (``use_rust_engine=True``) ships with two parallel
registries that short-circuit angr's Python dispatch for ubiquitous
calls:

* ``NativeProcedureRegistry`` (``native/angr/src/procedures/mod.rs``) —
  intercepts SimProcedure calls by **name**, e.g. ``strlen``,
  ``malloc``, ``printf``. See
  :doc:`simprocedures` (the *Native (Rust) SimProcedures* section)
  for the dispatch-priority chain and the contributor guide.
* ``NativeSyscallRegistry`` (``native/angr/src/syscalls/mod.rs``) —
  intercepts ``syscall`` instructions keyed by ``(arch_name,
  syscall_num)`` so the same handler can serve multiple ABIs.

This page is a **what's covered** index, not a contributor guide.
Cross-list it when answering *"is X supported natively, or does it
fall back to Python on the Rust engine?"*

For empirical fallback frequencies see *Bench fallback distribution*
below; the data is reproducible via the per-bench counters returned
by ``RustExplorationManager.stats``
(``simprocedure_python_fallback_count``,
``simprocedure_fallback_by_name``, ``native_proc_stats.*_by_name``,
``syscall_python_fallback_count``).

Native SimProcedure coverage
----------------------------

Every entry below is registered in ``NativeProcedureRegistry::new``
(``native/angr/src/procedures/mod.rs``). The registry is **name-keyed**
across all architectures — the calling-convention layer extracts
arguments from the right registers, so a single handler serves every
arch where the C ABI matches.

Status legend:

* **Native** — full Rust implementation; falls back to Python only on
  ``ProcedureError`` (typically a symbolic argument).
* **Native (stub)** — Rust handler returns the same constant as the
  Python proc (e.g. ``setvbuf`` always returns 0) without modeling
  side-effects beyond what angr's Python stub does.
* **Native (forced Python)** — registered to claim the name, but
  ``set_python_override`` routes execution back through the angr
  Python proc. Currently used by ``__libc_start_main`` to keep angr's
  Python init path while still claiming the name.

String and memory
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Status
     - Notes
   * - ``strlen``, ``strnlen``
     - Native
     - ``MAX_STRLEN`` cap on the symbolic-byte ITE chain.
   * - ``strcmp``, ``strncmp``, ``strcasecmp``
     - Native
     - ``declare_proc!`` concrete-args fast path.
   * - ``strcpy``, ``strncpy``, ``strdup``
     - Native
     - ``strdup`` allocates via ``NativeMalloc``.
   * - ``strcat``, ``strncat``
     - Native
     -
   * - ``strchr``, ``strrchr``, ``memchr``
     - Native
     - Includes the byte-by-byte ITE for symbolic haystacks.
   * - ``strstr``
     - Native
     -
   * - ``strpbrk``, ``strspn``, ``strcspn``
     - Native
     - Byte-set search (angr-f16h.5).
   * - ``memcpy``, ``memmove``, ``memset``, ``memcmp``
     - Native
     - ``memmove`` shares ``NativeMemcpy``'s loop with reverse-copy
       support.

Character classification (``ctype.h``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Names
     - Status
   * - ``isdigit``, ``isalpha``, ``isspace``, ``isalnum``,
       ``isupper``, ``islower``, ``isxdigit``, ``isprint``,
       ``tolower``, ``toupper``
     - Native (``declare_proc!`` ITE over the symbolic char).

String → numeric conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Names
     - Status
   * - ``strtol``, ``strtoul``, ``strtoll``, ``strtoull``,
       ``atoi``, ``atol``
     - Native — concrete buffer, base 0/8/10/16.
   * - ``strtod``
     - Native (amd64 ``xmm0`` / AArch64 ``v0`` FP-return slot; other
       arches defer to Python).

Heap
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Status
     - Notes
   * - ``malloc``, ``calloc``, ``realloc``, ``memalign``,
       ``posix_memalign``
     - Native
     - Bump allocator backed by ``state.heap`` (matches
       ``SimHeapBrk``).
   * - ``free``
     - Native (stub)
     - No-op (consistent with ``SimHeapBrk``).

Stdio output
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Status
     - Notes
   * - ``puts``, ``putchar``, ``fputs``, ``fputc``, ``putc``
     - Native
     - Resolve fd via ``state.posix.fd``.
   * - ``printf``
     - Native
     - Concrete-format fast path; symbolic format string falls back.
   * - ``fprintf``
     - Native
     - Stream variant of ``printf``: resolves ``stream->_fileno`` and
       writes the raw format string to that fd; symbolic FILE*/format
       falls back (angr-884yn).
   * - ``sprintf``, ``snprintf``
     - Native
     - Bounded buffer write.
   * - ``fwrite``, ``fflush``, ``setvbuf``, ``feof``, ``ferror``
     - Native (stub)
     - ``fflush``/``setvbuf`` always return 0; ``feof``/``ferror``
       return concrete flags (angr-70no, angr-f16h.1).

Stdio input
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Status
     - Notes
   * - ``fgets``, ``fgetc``, ``getchar``, ``getc``
     - Native
     - Mints symbolic stdin bytes; tracked in
       ``state.stdin_symbols`` for ``posix.dumps(0)`` export.
   * - ``scanf``, ``__isoc99_scanf``, ``sscanf``
     - Native
     - Concrete format string; ``%d``/``%s``/``%c``/``%x`` conversions.

File ops
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Status
     - Notes
   * - ``open``, ``close``, ``lseek``, ``dup``, ``dup2``, ``pipe``
     - Native
     - Updates angr's ``FileSystem`` fd table.
   * - ``read``, ``write``
     - Native
     - Re-enabled by angr-3tek.2; Python-side cache is replayed per
       dirty page on callback creation.
   * - ``fopen``, ``fdopen``, ``fclose``, ``fseek``, ``ftell``,
       ``rewind``
     - Native
     - Allocates ``_IO_FILE`` struct, dispatches through
       ``FILE._fileno`` (angr-karp). The heap range
       ``[0xC0000000, 0xC1000000)`` is registered as a lazy region and
       ``RustSimState::memory_store`` auto-maps lazy pages, so the write of
       ``_fileno`` into a freshly ``heap_alloc``-ed struct succeeds natively
       instead of erroring ``Unmapped`` and falling back to Python. Reads of
       never-written heap stay unmapped (fall back to Python, preserving
       symbolic-fill). Before this fix every ``fseek``/``fputc``/``fopen``/
       ``fclose`` on a fresh FILE fell back to Python (e.g. sharif7_rev50:
       175 -> 44 SimProcedure fallbacks, 0.9s -> 0.32s).

Environment
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Names
     - Status
   * - ``getenv``, ``setenv``, ``unsetenv``, ``putenv``,
       ``clearenv``
     - Native

Termination / init / misc
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Name
     - Status
     - Notes
   * - ``exit``, ``_exit``, ``exit_group``
     - Native (``no_return``)
     - Routes the state to ``STASH_DEADENDED``.
   * - ``abort``
     - Native (``no_return``)
     -
   * - ``__stack_chk_fail``
     - Native (``no_return``)
     - Top fallback name on ``defcon2016quals_baby-re`` before the
       native handler landed (14 calls/run).
   * - ``rand``, ``srand``
     - Native
     - Deterministic PRNG state in ``state.globals``.
   * - ``__libc_start_main``
     - Native (forced Python)
     - Python init path (``rust_manager._step_python_to_main``) is
       canonical; native registration claims the name so dispatch
       cannot accidentally take the Python proc twice.

What is **not** registered
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following are intentionally not in the registry; they always go
through the Python SimProcedure:

* User-placed in-binary hooks (``proj.hook(addr, MyProc())``) — the
  dispatcher's ``is_in_binary`` check bypasses the registry before
  any name lookup. See the *Dispatch priority* section of
  :doc:`simprocedures`.
* C++ runtime (``operator new``, ``operator delete``,
  ``std::operator<<``, ``std::string`` ctors / destructors) — top
  fallback group on ``csaw_wyvern`` (~25 calls/run), already at
  16.9x speedup, so a native port would not move the bench.
* App-specific CTF hooks (``my_scanf``, ``get_flag``, ``UserHook``,
  ``CallReturn``) — Python-only by construction.

Bench fallback distribution
---------------------------

Snapshot from ``bench-simprocedure-fallback-distribution`` (bd memory
``angr-twyr``, 2026-06-02). 17 benches surveyed across the fast tier;
74 total ``simprocedure_python_fallback`` calls.

**Zero-fallback benches (7/17 — fully native-covered):**
``ais3_crackme``, ``codegate_2017-angrybird``, ``csgames2018``,
``defcamp_r100``, ``flareon2015_2``, ``strcpy_find``, ``sym-write``.

**Top fallback names (aggregate):**

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Name
     - Calls
     - Where
   * - ``__stack_chk_fail``
     - 14
     - ``defcon2016quals_baby-re`` (now covered natively;
       counter is pre-native data)
   * - ``my_scanf``
     - 13
     - ``defcon2016quals_baby-re`` — app-specific user proc
   * - ``std::operator<<<...>``
     - 7
     - ``csaw_wyvern`` only — C++ stdlib
   * - ``operator new(unsigned long)``
     - 6
     - ``csaw_wyvern`` only — C++ stdlib
   * - ``UserHook``
     - 6
     - ``flareon2015_5``/``_10``, ``whitehatvn_re400``
   * - ``memmove``
     - 5
     - ``csaw_wyvern`` — all symbolic-arg fallbacks
   * - ``operator delete(void*)``
     - 5
     - ``csaw_wyvern`` only
   * - ``CallReturn``
     - 2
     - ``mma_howtouse``, ``flareon2015_10``
   * - ``strncpy``
     - 2
     - 1 symbolic, 1 other
   * - ``open``, ``strncmp``, ``get_flag`` (CTF), plus a handful
       of one-shot C++ ``std::string`` names
     - 1 each
     -

**Symbolic-arg gap (native handler tried, gave up on a symbolic
argument):** ``memmove`` (5), ``open`` (1), ``strncpy`` (1) — total 7
calls across all 17 benches. Tracked under
``native_proc_stats.symbolic_fallbacks_by_name`` for future tuning of
each handler's symbolic-input acceptance.

Implication: native SimProcedure coverage is **not** a current bench
bottleneck. The counter remains a diagnostic surface so new workloads
that hit unregistered hot procs show up immediately.

Native syscall coverage
-----------------------

Every entry below is registered in ``NativeSyscallRegistry::new``
(``native/angr/src/syscalls/mod.rs``). Handlers themselves are
arch-agnostic; the per-arch table maps Linux ABI numbers from
``<asm/unistd_*.h>`` to the same handler set.

Coverage shorthand:

* ✓ — registered on this arch.
* — — Linux ABI omits the syscall on this arch (asm-generic dropped
  legacy variants on AArch64; ``arch_prctl`` is amd64-only; etc.).
* ✗ — the ABI *does* have the syscall, but it is deliberately left
  unregistered so it falls through to Python (registering it would only
  add a dispatch hop before the same fallback).
* ⚠ — handler exists but is currently stubbed or known-incomplete
  (see *Stubbed / incomplete* below).

Per-arch matrix
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 8 8 10 10 10

   * - Syscall
     - AMD64
     - X86
     - ARM
     - ARM64
     - MIPS32
     - MIPS64
   * - ``read`` / ``write``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``open`` / ``close``
     - ✓
     - ✓
     - ✓
     - —
     - ✓
     - ✓
   * - ``openat`` / ``close``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``stat`` / ``fstat``
     - ✓
     - ✓ (LFS ``stat64`` 195 / ``fstat64`` 197)
     - ✓ (LFS ``stat64`` 195 / ``fstat64`` 197)
     - ✓ (``fstat`` only — asm-generic dropped ``stat``)
     - ✓ (LFS ``stat64`` 4213 / ``fstat64`` 4215)
     - ✗ (no MIPS64 ``struct stat`` writer)
   * - ``lstat``
     - ✓
     - ✓ (LFS ``lstat64`` 196)
     - ✓ (LFS ``lstat64`` 196)
     - —
     - ✓ (LFS ``lstat64`` 4214)
     - ✗ (no MIPS64 ``struct stat`` writer)
   * - ``newfstatat``
     - ✓
     - ✓ (``fstatat64`` 300)
     - ✓ (``fstatat64`` 327)
     - ✓
     - ✓ (``fstatat64`` 4293)
     - ✗ (no MIPS64 ``struct stat`` writer)
   * - ``access`` / ``faccessat``
     - ✓
     - ✓
     - ✓
     - ✓ (only ``faccessat``)
     - ✓
     - ✓
   * - ``readlink`` / ``readlinkat``
     - ✓
     - ✓
     - ✓
     - ✓ (only ``readlinkat``)
     - ✓
     - ✓
   * - ``mmap``
     - ✓
     - — (uses ``old_mmap`` / ``mmap2``)
     - — (uses ``old_mmap`` / ``mmap2``)
     - ✓
     - — (uses ``old_mmap`` only)
     - ✓
   * - ``old_mmap``
     - —
     - ✓
     - ✓
     - —
     - ✓
     - —
   * - ``mmap2``
     - —
     - ✓
     - ✓
     - —
     - ⚠ (number 4210; needs stack-arg traversal)
     - —
   * - ``mprotect`` / ``munmap``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``brk``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mremap`` / ``msync`` / ``madvise``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mlock`` / ``munlock`` / ``mlockall`` / ``munlockall``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``ioctl`` / ``fcntl``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fcntl64``
     - —
     - ✓
     - ✓
     - —
     - ✓
     - —
   * - ``dup`` / ``dup2`` / ``dup3``
     - ✓
     - ✓
     - ✓
     - dup, dup3 (no dup2)
     - ✓
     - ✓
   * - ``pipe`` / ``pipe2``
     - ✓
     - ✓
     - ✓
     - pipe2 only
     - ✓
     - ✓
   * - ``getcwd`` / ``chdir`` / ``fchdir``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mkdir`` / ``rmdir`` / ``unlink`` / ``rename``
     - ✓
     - ✓
     - ✓
     - — (at-only)
     - ✓
     - ✓
   * - ``mkdirat`` / ``unlinkat`` / ``renameat`` / ``renameat2``
     - ✓
     - ✓ (no ``renameat2`` on i386 table)
     - ✓ (no ``renameat2`` on ARM EABI table)
     - ✓ (no ``renameat2``)
     - ✓ (no ``renameat2``)
     - ✓ (no ``renameat2``)
   * - ``get*id`` (uid/gid/euid/egid/pid/ppid/tid)
     - ✓
     - ✓ (legacy + LFS 32-bit aliases)
     - ✓ (legacy + LFS)
     - ✓
     - ✓
     - ✓
   * - ``set*id`` (uid/gid)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``exit`` / ``exit_group``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``kill`` / ``tgkill``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``alarm`` / ``pause``
     - ✓
     - ✓
     - ✓
     - — (asm-generic omits both)
     - ✓
     - ✓
   * - ``rt_sigaction`` / ``rt_sigreturn``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``time`` / ``gettimeofday`` / ``clock_gettime``
     - ✓ (no ``time`` slot)
     - ✓
     - ✓
     - ✓ (no ``time``)
     - ✓
     - ✓ (no ``time``)
   * - ``getrlimit`` / ``setrlimit`` / ``prlimit64``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``futex``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``eventfd`` / ``eventfd2``
     - ✓
     - ✓
     - ✓
     - — / ✓ (only ``eventfd2``)
     - ✓
     - ✓
   * - ``epoll_create`` / ``epoll_create1`` / ``epoll_ctl`` /
       ``epoll_wait``
     - ✓
     - ✓
     - ✓
     - ✓ (no legacy ``epoll_create`` / ``epoll_wait``;
       ``epoll_pwait`` not yet registered)
     - ✓
     - ✓
   * - ``arch_prctl``
     - ✓
     - — (amd64 only)
     - —
     - —
     - —
     - —

Counts (current registrations):

* AMD64: 72 ``(arch, num)`` entries.
* X86: 75 entries.
* ARM: 74 entries.
* ARM64: 50 entries (asm-generic ABI drops legacy variants).
* MIPS32: 67 entries.
* MIPS64: 65 entries.

Bench fallback rate: **zero**. Across the 19 baseline benchmarks
measured 2026-06-02 (bd memory ``bench-syscall-fallback-zero-coverage``,
``angr-0wam``), ``syscall_python_fallback_count == 0`` on every bench.
The existing handler set fully covers the CTF-heavy bench corpus; new
syscall handlers cannot move bench numbers until a workload that hits
an unregistered ``(arch, num)`` arrives.

Stubbed / incomplete
~~~~~~~~~~~~~~~~~~~~

* **``mremap``** — number registered on every arch but the handler
  models only the simplest case (no full page-table coordination
  beyond stub parity). Tracked as ``angr-uahs``.
* **MIPS32 ``mmap2`` (4210)** — six register args, O32 only passes
  four in ``$a0``–``$a3``; the remaining offset/fd args live on the
  stack which ``extract_syscall_args`` does not currently traverse.
  Falls back to Python; tracked indirectly by the comment in
  ``syscalls/mod.rs``.
* **AArch64 ``epoll_pwait``** — asm-generic ABI replaces the legacy
  ``epoll_wait``; ``epoll_pwait`` (22) is not yet a registered slot.
  Small follow-up after ``angr-0hif.7``.
* **Symbolic-rax dispatch** — when ``rax`` is symbolic the dispatcher
  records the fallback under ``syscall_python_fallback_by_num`` key
  ``-1`` and hands the state to Python. The counter is at zero on the
  current bench corpus.
* **``lstat`` / ``newfstatat`` / ``faccessat`` flag args** — the
  ``angr-6009`` audit landed: all five names it covered (``readlink``,
  ``readlinkat``, ``lstat``, ``newfstatat``, ``faccessat``) are real
  native handlers in ``syscalls/file_path.rs``, registered per the
  matrix above, and no longer route through the stub-symbolic Python
  handler. Two deliberate gaps remain. ``newfstatat`` ignores its
  ``flag`` argument — ``AT_EMPTY_PATH`` (0x1000) would have to
  re-dispatch to ``NativeFstatSyscall(dirfd)``, and
  ``AT_SYMLINK_NOFOLLOW`` (0x100) would have to route through
  ``stat_lookup_nofollow`` instead of the following
  ``stat_lookup_follow`` it shares with ``stat``. ``faccessat``
  ignores ``mode``, matching angr's Python ``access``.
  (The third gap — ``lstat``/``stat`` ignoring the symlink table that
  ``readlink`` reads — was closed by ``angr-9ke6b.235``: ``lstat`` now
  reports a registered symlink as ``S_IFLNK | 0777`` sized to its raw
  target bytes, and ``stat``/``newfstatat`` walk the link to its target,
  up to ``MAX_SYMLINK_HOPS``, returning ``-1`` on a dangling link or a
  cycle. MIPS32 is a partial exception: its ``struct stat64`` writer
  emits no ``st_mode`` field at all, so there the link shows up only in
  ``st_size``.)

.. _native-documented-divergences:

Documented divergences
----------------------

Every fallback name the census
(:ref:`rust-engine-fallback-census`) observed is proven
observationally silent by a differential parity assertion in
``tests/engines/rust/test_fallback_parity.py``: the same call is run on
``RustExplorationManager`` and on the pure-Python engine, and the two
engines must agree on every observable the procedure touches (return
register, the memory it wrote, ``posix.dumps``, the fd table).

The names below are the exception. They have **no native counterpart to
diff against** — angr's Python procedure *is* the reference
implementation — so they carry a documented-divergence row here instead
of an assertion. Divergence in these rows means "the Rust engine has no
independent model", not "the two engines disagree".

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Name(s)
     - Why there is nothing to diff
   * - ``UserHook``, ``my_scanf``, ``get_flag``, ``readline_hook``,
       ``strtol_hook``
     - User-supplied Python hooks (CTF ``solve.py`` procs). Python by
       definition on both engines. ``_snapshot_orig_state`` hands any
       ``UserHook`` a full ``state.copy()``, so everything it writes is
       diffed back across the resume FFI.
   * - ``operator new(unsigned long)``, ``operator delete(void*)``
     - Resolve to angr's ``malloc`` / ``free`` procs. Parity is
       inherited from those rows: the bounced heap bump round-trips
       since ``angr-op0dn.14.1.3``, and ``free`` does not move the bump.
   * - ``std::allocator<char>``, ``std::basic_string<...>``,
       ``std::basic_ostream<...> operator<<``, ``std::string::length``
     - C++ stdlib symbols. angr models them with a Python proc (or
       ``ReturnUnconstrained``); the native registry claims no C++
       symbol, so there is no second implementation to compare.

A newly-observed fallback name must land in **one** of the two places —
a parity scenario or a row above — or
``test_fallback_parity.py::TestCensusCoverage`` fails.

See also
--------

* :doc:`simprocedures` — Python SimProcedures plus the *Native (Rust)
  SimProcedures* contributor guide and the dispatch-priority chain.
* :doc:`/advanced-topics/rust_engine` — the engine's behavioral
  contract, architecture support matrix, and pipeline counters.
* ``RustExplorationManager.stats`` (``get_solver_stats`` /
  ``get_simprocedure_stats``) — runtime introspection for fallback
  counters used to refresh this page after registry changes.
