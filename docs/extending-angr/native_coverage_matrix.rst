Native SimProcedure and syscall coverage matrix
===============================================

The Rust engine (``engine="rust"``) ships with two parallel
registries that short-circuit angr's Python dispatch for ubiquitous
calls:

* ``NativeProcedureRegistry`` (``native/angr/src/procedures/mod.rs``) —
  intercepts SimProcedure calls by **name**, e.g. ``strlen``,
  ``malloc``, ``printf``. See
  :doc:`simprocedures` (the *Native (Rust) SimProcedures* section)
  for the dispatch-priority chain and the contributor guide.
* ``NativeSyscallRegistry`` (``native/angr/src/syscalls/mod.rs``) —
  intercepts syscalls. Handlers are registered by the name angr's
  syscall library gives each syscall, and each manager resolves the
  numbers from its project's SimOS, so the same handler serves every
  ABI that names the syscall.

This page is a **what's covered** index, not a contributor guide.
Cross-list it when answering *"is X supported natively, or does it
fall back to Python on the Rust engine?"*

To see what a given workload actually bounces to Python, read the
per-bench counters on ``RustExplorationManager.stats``
(``simprocedure_python_fallback_count``,
``simprocedure_fallback_by_name``, ``native_proc_stats.*_by_name``,
``syscall_python_fallback_count``) or run
``tests/benchmarks/run_single.py <bench> --dump-counters``.

Native SimProcedure coverage
----------------------------

Every entry below is registered in ``NativeProcedureRegistry::new``
(``native/angr/src/procedures/mod.rs``). The registry is **name-keyed**
across all architectures — the calling-convention layer extracts
arguments from the right registers, so a single handler serves every
arch where the C ABI matches. Rows are grouped by function family and
list every name the registry answers to, **including aliases** (an
alias reuses the same impl — see ``NativeProcedureRegistry::register``).

The ``Native`` column is a fraction ``M / N`` rather than a status enum:
every member has a Python fallback, so the meaningful axis is how many
of the named members are on the fast path. A member marked ✗ is
registered (or deliberately not) but always takes the Python path; the
provenance column says why. ``__libc_start_main`` is registered to
claim the name but routed back to Python through
``set_python_override`` so angr's init path still runs.

.. list-table::
   :header-rows: 1
   :widths: 25 35 15 25

   * - Group
     - Members
     - Native
     - Provenance
   * - String length / copy / concat
     - ``strlen``, ``strnlen``, ``strcpy``, ``strncpy``, ``stpcpy``,
       ``stpncpy``, ``strdup``, ``strndup``, ``strxfrm``,
       ``strcat``, ``strncat``
     - 11 / 11
     - ``strlen.rs``, ``strcpy.rs``, ``strcat.rs``
   * - String / memory compare
     - ``strcmp`` (alias ``strcoll``), ``strncmp``, ``strcasecmp``,
       ``strncasecmp``, ``memcmp``
     - 5 / 5
     - ``strcmp.rs``, ``memcmp.rs``
   * - Search
     - ``strchr``, ``strchrnul``, ``strrchr``, ``memchr``,
       ``rawmemchr``, ``memrchr``, ``strstr``, ``strpbrk``,
       ``strspn``, ``strcspn``, ``strtok`` ✗
     - 10 / 11
     - ``strchr.rs``, ``strstr.rs``, ``byteset.rs`` (angr-f16h.5).
       ``strtok`` intentionally left on the Python fallback path (angr-4c65 wontfix):
       a 14-bench fallback profile shows zero
       ``strtok`` appearances, so the stateful ``state.globals`` save
       pointer + ``strtok_r`` symbolic-write plumbing is not justified.
   * - Bulk memory
     - ``memcpy``, ``memmove``, ``mempcpy``, ``memset``, ``bzero``
     - 5 / 5
     - ``memcpy.rs``, ``memset.rs``
   * - Fortify-source ``_chk`` wrappers
     - ``__memcpy_chk``, ``__memmove_chk``, ``__mempcpy_chk``,
       ``__memset_chk``, ``__strcpy_chk``, ``__strncpy_chk``,
       ``__stpcpy_chk``, ``__strcat_chk``, ``__strncat_chk``,
       ``__printf_chk``, ``__sprintf_chk``, ``__snprintf_chk``,
       ``__fprintf_chk``, ``__vsnprintf_chk``
     - 14 / 14
     - ``fortify_mem.rs``, ``fortify_str.rs``, ``fortify_printf.rs``.
       Each wrapper only drops the injected ``destlen``/``flag``/``slen``
       args and then **calls the base native procedure** — it never
       reimplements the copy/format logic. Note it deliberately does
       **not** emulate glibc's destination-size bound check: matching
       Python angr (whose ``__*_chk`` classes subclass the base and drop
       the extra args), a check would prune paths angr otherwise
       explores. See each file's module docs for the parity rationale.
   * - Character classification (ctype.h)
     - ``isdigit``, ``isalpha``, ``isspace``, ``isalnum``,
       ``isupper``, ``islower``, ``isxdigit``, ``isprint``,
       ``isascii``, ``isblank``, ``iscntrl``, ``isgraph``,
       ``ispunct``, ``tolower``, ``toupper``, ``__ctype_b_loc``,
       ``__ctype_tolower_loc``, ``__ctype_toupper_loc``
     - 18 / 18
     - ``ctype.rs`` (symbolic-aware helper pattern lives in the file;
       the ``*_loc`` accessors return the tables Python's
       ``__libc_start_main`` built — see ``CtypeLocPtrs``)
   * - String → numeric
     - ``strtol``, ``strtoul``, ``strtoll``, ``strtoull``, ``atoi``,
       ``atol``, ``strtod``
     - 7 / 7
     - ``strtol.rs``, ``strtod.rs`` (angr-f16h.3; ``strtod`` is
       amd64 xmm0 / AArch64 v0 FP-return)
   * - Heap
     - ``malloc``, ``free``, ``calloc``, ``realloc``, ``memalign``,
       ``posix_memalign``
     - 6 / 6
     - ``malloc.rs`` covers all six (bump allocator, matches
       ``SimHeapBrk``; angr-f16h.2)
   * - Input (stdin)
     - ``read``, ``fgets``, ``fgetc``, ``getchar``, ``getc``,
       ``gets``, ``scanf`` (alias ``__isoc99_scanf``), ``fscanf``
       (alias ``__isoc99_fscanf``), ``sscanf`` ✗ — plus the
       ``*_unlocked`` aliases of the ``fgets`` family
     - 8 / 9
     - ``read.rs``, ``fgets.rs``, ``scanf.rs``. ``sscanf`` is
       registered (``scanf.rs::NativeSscanf``) but its ``call`` body
       unconditionally returns ``ProcedureError::Other``, so every
       invocation pays the full Python round-trip and it does not
       count toward the native speed-up. Deliberate: only Python
       parses the concrete source region and constrains the outputs,
       and a native free-BVS mint would explore impossible paths (angr-8onrp).
   * - Output (stdout / formatted)
     - ``write``, ``puts``, ``putchar``, ``fputc`` (aliases ``putc``,
       ``*_unlocked``), ``perror``, ``printf`` (alias ``vprintf``),
       ``fprintf`` (alias ``vfprintf``), ``sprintf``, ``asprintf``,
       ``snprintf``, ``vsnprintf``, ``vsprintf``
     - 12 / 12
     - ``write.rs``, ``puts.rs``, ``perror.rs``, ``printf.rs``,
       ``sprintf.rs``
   * - C stdio ``FILE`` I/O
     - ``fopen``, ``fdopen``, ``fclose``, ``feof``, ``ferror``,
       ``fflush``, ``fputs``, ``fwrite``, ``fread``
       (``fread_unlocked``), ``ftell`` (alias ``ftello``), ``fseek``
       (alias ``fseeko``), ``rewind``, ``setvbuf``, ``setbuf``
     - 14 / 14
     - ``stdio.rs``, ``stream_ops.rs``, ``fread.rs``,
       ``fwrite.rs`` (angr-f16h.1 / angr-karp / angr-70no / angr-m674p).
       ``fopen``/``fdopen``
       heap-allocate an ``_IO_FILE``; the rest dispatch off
       ``FILE._fileno``.
   * - Low-level file ops
     - ``open``, ``close``, ``lseek``, ``dup``, ``dup2``, ``pipe``
     - 6 / 6
     - ``fileops.rs`` (fd-tracking through ``FileSystem``)
   * - Process / exit / RNG
     - ``exit``, ``_exit``, ``abort``, ``__stack_chk_fail``,
       ``rand``, ``srand``, ``system``, ``__libc_start_main``
     - 8 / 8
     - ``exit.rs``, ``rand.rs``, ``system.rs`` (models the exit
       status as an unconstrained 8-bit code; angr-ae54t.16),
       ``libc_start_main.rs``
   * - Environment / identity / args
     - ``getenv``, ``setenv``, ``putenv``, ``unsetenv``,
       ``clearenv``, ``getopt``, ``getuid``, ``geteuid``, ``getgid``,
       ``getegid``
     - 10 / 10
     - ``getenv.rs`` (with the mutators; angr-f16h.4), ``getopt.rs``,
       ``getid.rs`` (the four id getters return the constant 1000,
       mirroring ``syscalls/identity.rs``; angr-ae54t.9)
   * - Time / sleep / logging no-ops
     - ``time``, ``sleep``, ``usleep``, ``openlog``, ``closelog``,
       ``access``
     - 6 / 6
     - ``time.rs`` (forwards to
       ``syscalls::sim_time::fresh_monotonic_time``, so the libc entry
       and the syscall share one model; angr-ae54t.21), ``sleep.rs`` (angr-ae54t.15),
       ``syslog.rs`` (void no-ops; ``syslog(3)`` itself
       stays on Python; angr-ae54t.20), ``access.rs``
   * - Threads
     - ``pthread_mutex_lock``, ``pthread_mutex_unlock``,
       ``pthread_once``
     - 3 / 3
     - ``pthread.rs`` — single-path symex means locks always succeed;
       ``pthread_once`` is the worked example of the native sub-call
       (``CallAndResume``) dispatcher
   * - Byte order
     - ``htonl`` (alias ``ntohl``), ``htons`` (alias ``ntohs``)
     - 2 / 2
     - ``byteorder.rs``

Not in the table: **unconstrained stubs**. ``procedures/stub.rs``'s
``NativeReturnUnconstrained`` is the native counterpart of angr's
``ReturnUnconstrained`` SimProcedure, and it is not keyed on a fixed
libc name — the dispatch name is whatever symbol the *binary* left
unresolved (``get_flag``, some unmodelled import). Instances are
registered dynamically at setup time by
``RustExplorationManager::register_unconstrained_stubs``
(``exploration/manager_methods_procedures.rs``), which walks
``project._sim_procedures`` from the Python side and registers one
stub per display name and return width. So a given binary's native
coverage is the static table above *plus* a per-binary set of stubs.

What is **not** registered
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following always go through the Python SimProcedure:

* User-placed in-binary hooks (``proj.hook(addr, MyProc())``) — the
  dispatcher's main-object check bypasses the registry before any name
  lookup. See the *Dispatch priority* section of :doc:`simprocedures`.
* C++ runtime (``operator new``, ``operator delete``,
  ``std::operator<<``, ``std::string`` ctors / destructors) — the
  native registry claims no C++ symbol; ``operator new`` / ``delete``
  resolve to angr's ``malloc`` / ``free`` procs and inherit their
  parity.
* App-specific CTF hooks the ``solve.py`` installs (``my_scanf``,
  ``UserHook``, …) — Python by definition. An *unresolved* import such
  as ``get_flag`` is different: it gets a native ``ReturnUnconstrained``
  stub, as described above.

Maintaining the matrix
~~~~~~~~~~~~~~~~~~~~~~

When a procedure gains or loses a native path, flip the member's mark
and the ``M / N`` count in the same commit, and keep the provenance
column pointing at the file (and at the tracking issue, when the reason is not
obvious from the code). A brand-new family gets a new row. The two
``mod.rs`` registries are the source of truth; this page is the index.

Native syscall coverage
-----------------------

Every entry below is a row of ``BY_NAME`` (``native/angr/src/syscalls/mod.rs``),
keyed by the syscall's name in angr's syscall library. No syscall number
is spelled in Rust: when the manager is built, the Python wrapper hands
``set_syscall_tables`` angr's ``syscall_number_mapping`` for each ABI the
project's SimOS lists (``amd64``, ``i386``, ``arm``, ``aarch64``,
``mips-o32``, ``mips-n64``, and ``cgcabi`` for DECREE; see *CGC / DECREE*
below), and ``NativeSyscallRegistry::from_angr_tables`` resolves each
number through its name. Handlers themselves are arch-agnostic. A row
carries an explicit ABI scope only where angr's name means a different
handler on some ABI (``mmap`` is the struct-argument form on arm and
mips-o32) or must stay unregistered there (MIPS64 leaves the stat family
unregistered because ``write_stat_for_arch`` has no MIPS64
``struct stat`` writer). A SimOS without a syscall library (a plain
``SimOS``, Windows) gets no native syscalls, as Python's engine has no
syscall procedures there. On AMD64 only the ``syscall`` instruction uses
the amd64 table; ``int 0x80`` goes to Python, as ``SimLinux.syscall_abi``
decides its ABI. A symbolic syscall number, and any unregistered concrete
number, falls through ``core_outcome_handlers.rs::handle_syscall_core``
to Python's ``engines/successors.py::_resolve_syscall``.

Coverage shorthand:

* ✓ — registered on this arch.
* — — angr's table for this arch's ABI (in
  ``angr/procedures/definitions/linux_kernel.py``) gives the syscall no
  number, so there is nothing to resolve: either the Linux ABI omits it
  (asm-generic dropped legacy variants on AArch64; ``arch_prctl`` is
  amd64-only), or angr's table stops short of it (``getrandom`` outside
  AMD64 / X86).
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
   * - ``readv`` / ``writev``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``pread64`` / ``pwrite64``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``lseek``
     - ✓
     - ✓ (no ``_llseek`` 140)
     - ✓ (no ``_llseek`` 140)
     - ✓
     - ✓ (no ``_llseek`` 4140)
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
     - ✓ (number 4210; args 5–6 read from the O32 stack)
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
       ``epoll_pwait`` (22) covers both)
     - ✓
     - ✓
   * - ``uname``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``set_tid_address`` / ``set_robust_list``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``getrandom``
     - ✓
     - ✓
     - — (no ARM entry in angr's table)
     - — (no AArch64 entry)
     - — (no MIPS-O32 entry)
     - — (no MIPS-N64 entry)
   * - ``arch_prctl``
     - ✓
     - — (amd64 only)
     - —
     - —
     - —
     - —

The registry itself is authoritative for what each ABI dispatches
natively: ``RustExplorationManager._rust_mgr.native_syscall_table()``
reports ``{abi: {number: handler}}`` for a project, rather than trusting
a number written here. ``syscall_python_fallback_count`` on
``RustExplorationManager.stats`` says whether a workload reaches an
unregistered number at all; on the CTF benchmark corpus it does not, so
a new handler only pays off once a workload that needs it arrives.

CGC / DECREE
~~~~~~~~~~~~

DECREE binaries run on x86 but use their own syscall numbers, which
collide with Linux i386 (1 is ``exit`` there and ``_terminate`` here).
angr's ``SimCGC`` lists only its own ``cgcabi`` numbering
(``angr/procedures/definitions/cgc.py``), so a CGC project's registry
holds only the DECREE table and every syscall dispatches through it; the
``BY_NAME`` rows scoped to ``cgcabi`` map those names to the handlers in
``native/angr/src/syscalls/cgc.rs``. The module doc of ``cgc.rs`` is
authoritative for each handler's divergences; the Python counterparts live
under ``angr/procedures/cgc/``.

.. list-table::
   :header-rows: 1
   :widths: 8 22 70

   * - Num
     - Syscall
     - Native coverage (falls back to Python otherwise)
   * - 1
     - ``_terminate``
     - Full; deadends the state like ``exit``.
   * - 2
     - ``transmit``
     - Concrete bytes to a stdout / stderr fd; symbolic args or bytes, or
       any other fd, fall back.
   * - 3
     - ``receive``
     - Fresh symbolic stdin bytes for ``fd == 0``; symbolic args, other
       fds or oversized counts fall back.
   * - 4
     - ``fdwait``
     - ⚠ Models only the ``CGC_NON_BLOCKING_FDS`` mode (every fd ready);
       with that option unset it falls back, since Python makes each
       ready bit unconstrained.
   * - 5
     - ``allocate``
     - Bump allocator on ``cgc_allocation_base`` / ``cgc_sinkholes``;
       symbolic args, or a bump that would collide with the loader /
       flag-page region, fall back.
   * - 6
     - ``deallocate``
     - Page-aligned unmap with sinkhole bookkeeping.
   * - 7
     - ``random``
     - Fresh symbolic bytes (non-fastpath mode).

Stubbed / incomplete
~~~~~~~~~~~~~~~~~~~~

* **``mremap``** — number registered on every arch but the handler
  models only the simplest case (no full page-table coordination
  beyond stub parity).
* **``set_tid_address`` / ``set_robust_list``** — registered everywhere
  but deliberately threadless (``syscalls/startup.rs``):
  ``set_tid_address`` returns tid 1 and ignores ``tidptr`` (no
  clear-child-tid futex is maintained, matching
  ``linux_kernel/set_tid_address.py``), and ``set_robust_list`` is a
  no-op returning 0. Both are correct for the single-threaded model the
  engine executes; neither has state a multi-threaded model could read
  back.
* **``_llseek``** — the 32-bit LFS variant (140 on X86 / ARM, 4140 on
  MIPS32) is *not* registered even though ``lseek`` is: its 64-bit
  offset arrives split across two registers and is returned through a
  ``loff_t *result`` out-param, so it needs a distinct handler rather
  than a ``NativeLseekSyscall`` alias.
* **Symbolic-rax dispatch** — when ``rax`` is symbolic the dispatcher
  records the fallback under ``syscall_python_fallback_by_num`` key
  ``-1`` and hands the state to Python. The counter is at zero on the
  current bench corpus.
* **``lstat`` / ``newfstatat`` / ``faccessat`` flag args** —
  ``readlink``, ``readlinkat``, ``lstat``, ``newfstatat`` and
  ``faccessat`` are all real native handlers (``angr-6009``) in
  ``syscalls/file_path/``, registered per the matrix above, and no
  longer route through the stub-symbolic Python handler. Two deliberate gaps remain. ``newfstatat`` ignores its
  ``flag`` argument — ``AT_EMPTY_PATH`` (0x1000) would have to
  re-dispatch to ``NativeFstatSyscall(dirfd)``, and
  ``AT_SYMLINK_NOFOLLOW`` (0x100) would have to route through
  ``stat_lookup_nofollow`` instead of the following
  ``stat_lookup_follow`` it shares with ``stat``. ``faccessat``
  ignores ``mode``, matching angr's Python ``access``.
  ``lstat``/``stat`` honor the symlink table that ``readlink``
  reads (``angr-9ke6b.235``): ``lstat`` reports a registered symlink
  as ``S_IFLNK | 0777`` sized to its raw target bytes, and ``stat``/``newfstatat`` walk the link to its target,
  up to ``MAX_SYMLINK_HOPS``, returning ``-1`` on a dangling link or a
  cycle. MIPS32 is a partial exception: its ``struct stat64`` writer
  emits no ``st_mode`` field at all, so there the link shows up only in
  ``st_size``.

.. _native-documented-divergences:

Documented divergences
----------------------

Every fallback name the census
(``tests/benchmarks/fallback_census.py``) observed is proven
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
       definition on both engines. The callback state's memory and
       registers are Rust proxies, so everything a hook writes lands in
       the Rust state as it is written.
   * - ``operator new(unsigned long)``, ``operator delete(void*)``
     - Resolve to angr's ``malloc`` / ``free`` procs. Parity is
       inherited from those rows: the bounced heap bump round-trips (``angr-op0dn.14.1.3``),
       and ``free`` does not move the bump.
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
* ``RustExplorationManager.stats`` (and ``mgr._rust_mgr.get_fallback_stats()``
  for the per-address breakdown) — runtime introspection for the
  fallback counters used to check this page after registry changes.
