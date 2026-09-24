//! Native `FILE *`-based stdio primitives: fopen, fdopen, fclose, fseek,
//! ftell, rewind.
//!
//! fopen/fdopen allocate an `_IO_FILE` struct on the heap and store the fd at
//! the arch-specific `_fileno` offset; fclose/fseek/ftell/rewind dispatch off
//! it. The fd-table primitives these ultimately drive (open/close/lseek/dup/
//! dup2/pipe) live in the sibling [`fileops`](super::fileops) module.
//!
//! This module also owns the `FILE *` -> fd resolution surface every other
//! stream-facing proc family shares — [`io_file_for_arch`], [`read_fileno`],
//! [`read_fileno_or_demote_all`] and [`resolve_stream_fd_or_demote_all`] —
//! used from `stdio.rs`, `fwrite.rs`, `fread.rs`, `fgets.rs`, `printf.rs`,
//! `puts.rs` and `scanf.rs`.
//!
//! Only handles concrete arguments; symbolic arguments fall back to Python.

use super::strings::{MAX_PATH_SCAN as MAX_PATH, scan_concrete_until_null};
use super::{ProcedureError, arch_word};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

/// `_IO_FILE` size and fd offset per arch, mirroring
/// `cle.backends.externs.simdata.io_file.io_file_data_for_arch`.
/// Returns `(fd_offset, total_size)`. Single source of truth for the arch ->
/// `_fileno` offset mapping.
/// Arch names match what `Arch::name()` returns (only `"ARM"`/`"ARM64"` for the
/// ARM family — the `ARMEL`/`ARMHF`/`AARCH64` aliases never reach here).
pub(super) fn io_file_for_arch(name: &str) -> Option<(u64, u64)> {
    match name {
        "AMD64" => Some((112, 216)),
        "X86" => Some((56, 148)),
        "ARM" => Some((14, 84)),
        "ARM64" => Some((20, 152)),
        "MIPS32" => Some((56, 148)),
        "MIPS64" => Some((112, 216)),
        _ => None,
    }
}

const MAX_FOPEN_MODE_LEN: u64 = 8;

/// Read a NUL-terminated string from memory up to `max_len` bytes. Concrete
/// only: a symbolic byte (or exhausting `max_len` without a null) errors and
/// falls back to Python.
///
/// The `_strict` suffix distinguishes this from `getenv.rs`'s
/// `read_cstring_tolerant`, which returns the unterminated prefix instead of
/// erroring (angr-03vl4.47): the two used to share the bare name `read_cstring`
/// with silently opposite cap-exhaustion contracts. `fileops::read_pathname`
/// deliberately follows the strict contract for the same reason (angr-sqfj8.80).
fn read_cstring_strict(
    state: &mut RustSimState,
    addr: u64,
    max_len: u64,
    name: &str,
) -> Result<Vec<u8>, ProcedureError> {
    scan_concrete_until_null(state, addr, max_len as usize, name)
}

/// Convert an fopen-style mode string (e.g. `"r"`, `"w+b"`, `"rb+"`) to an
/// access mode plus the append bit.
/// Returns None for unrecognized modes (caller falls back to Python).
///
/// glibc semantics: the first character selects the base access mode
/// (`r`/`w`/`a`); the remaining flag characters (`+`, `b`, `t`, `c`, `e`, `m`,
/// `x`) may appear in **any order**. Only `+` upgrades the access mode to
/// read+write — the rest are buffering/sharing/exclusivity hints that do not
/// affect the FdFlags mapping. Parsing the trailing flags positionally (only
/// popping a trailing `b`/`t`) silently missed valid orderings like `"rb+"`.
///
/// The second tuple element is `O_APPEND`, which `"a"`/`"a+"` carry and
/// `"w"`/`"w+"` do not — the one thing that distinguishes the two truncating
/// modes in this model (see [`FileDescriptor::append`](crate::state::FileDescriptor::append)).
/// It is returned separately because [`FdFlags`] is the POSIX access mode
/// alone.
fn parse_fopen_mode(mode: &[u8]) -> Option<(FdFlags, bool)> {
    let first = *mode.first()?;
    // Flag characters glibc accepts after the base mode. Anything outside this
    // set is genuinely unrecognized, so defer to Python rather than guess.
    const FLAG_CHARS: &[u8] = b"+btcexm";
    if mode[1..].iter().any(|c| !FLAG_CHARS.contains(c)) {
        return None;
    }
    let read_write = mode[1..].contains(&b'+');
    let access = match (first, read_write) {
        (b'r', false) => FdFlags::ReadOnly,
        (b'r', true) => FdFlags::ReadWrite,
        (b'w', false) => FdFlags::WriteOnly,
        (b'w', true) => FdFlags::ReadWrite,
        (b'a', false) => FdFlags::WriteOnly,
        (b'a', true) => FdFlags::ReadWrite,
        _ => return None,
    };
    Some((access, first == b'a'))
}

/// Read a 32-bit fd from a FILE struct on the given arch. Returns the signed fd
/// (so -1 sentinels are preserved) and any symbolic / arch-resolution errors.
pub(crate) fn read_fileno(state: &RustSimState, file_ptr: u64) -> Result<i32, ProcedureError> {
    let arch_name = state.arch().name();
    let (fd_off, _) = io_file_for_arch(arch_name)
        .ok_or_else(|| ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}")))?;
    let bv = state.memory_load(file_ptr.wrapping_add(fd_off), 4)?;
    let raw = bv
        .as_u64()
        .ok_or_else(|| ProcedureError::SymbolicArgument("FILE._fileno".to_string()))?;
    Ok(raw as u32 as i32)
}

/// [`read_fileno`] for a **write** path: on failure, demote every bounded
/// symbolic file's content before propagating the error.
///
/// Write-intent bounces must know the target fd so they can demote its bounded
/// symbolic content before falling back (angr-0xyq2 A4). An unresolvable fd
/// (symbolic `_fileno`, unmapped FILE struct, unknown arch) could alias any
/// registered file, so it demotes everything — otherwise Python takes over the
/// write while Rust keeps serving the now-stale symbolic content. O(1) when no
/// bounded symbolic file is attached.
///
/// Every native write-family proc that dispatches off a `FILE *` funnels
/// through this (or [`resolve_stream_fd_or_demote_all`]) rather than calling
/// [`read_fileno`] directly, so the demotion protocol cannot diverge per proc.
///
/// Read-family procs do **not** use this: they have no symbolic content to
/// demote, and instead carve out stdin via
/// `fgets.rs::read_fileno_or_stdin`.
pub(crate) fn read_fileno_or_demote_all(
    state: &mut RustSimState,
    file_ptr: u64,
) -> Result<i32, ProcedureError> {
    match read_fileno(state, file_ptr) {
        Ok(fd) => Ok(fd),
        Err(e) => {
            state.file_system().demote_all_symbolic_content();
            Err(e)
        }
    }
}

/// [`read_fileno_or_demote_all`] for callers whose `FILE *` is still a
/// [`RustBV`]: a *symbolic* stream pointer is just as unresolvable as a
/// symbolic `_fileno`, so it demotes on the same terms rather than bailing out
/// of the write path with the demotion skipped.
///
/// The `resolve_stream_fd` prefix marks the `&RustBV` entry point; the
/// `read_fileno` prefix marks the already-concretized `u64` one. Both are
/// write-side — the read-side sibling is
/// `fgets.rs::read_fileno_or_stdin`.
pub(crate) fn resolve_stream_fd_or_demote_all(
    state: &mut RustSimState,
    stream: &RustBV,
) -> Result<i32, ProcedureError> {
    match super::extract_concrete_arg(stream, "file_ptr") {
        Ok(file_ptr) => read_fileno_or_demote_all(state, file_ptr),
        Err(e) => {
            state.file_system().demote_all_symbolic_content();
            Err(e)
        }
    }
}

crate::declare_proc! {
    /// Native fopen implementation.
    ///
    /// ```c
    /// FILE *fopen(const char *pathname, const char *mode);
    /// ```
    ///
    /// Opens an fd, allocates an `_IO_FILE` struct on the heap, writes the fd at the
    /// arch-specific `_fileno` offset, and returns the struct pointer. Returns 0 on
    /// unrecognized modes / unsupported arches / unmapped heap pages so the Python
    /// implementation can take over.
    ///
    /// The `"a"`/`"a+"` modes open with `O_APPEND`
    /// ([`FileDescriptor::append`](crate::state::FileDescriptor::append)), which is
    /// the only thing separating them from `"w"`/`"w+"` here: this model has no
    /// `O_TRUNC` to skip, since a freshly-opened path starts empty either way.
    /// Python `fopen.py::mode_to_flag` maps the same two modes to `O_APPEND` and
    /// `storage/file.py::SimFileDescriptor::write_data` honors it, so the engines
    /// agree on append ordering (angr-fs8kb.47).
    name = "fopen",
    struct = NativeFopen,
    args = [path_addr: concrete, mode_addr: concrete],
    call |state| {
        let path = read_cstring_strict(state, path_addr, MAX_PATH, "pathname")?;
        let mode = read_cstring_strict(state, mode_addr, MAX_FOPEN_MODE_LEN, "mode")?;
        let (flags, append) = parse_fopen_mode(&mode).ok_or_else(|| {
            ProcedureError::Other(format!(
                "unsupported fopen mode {:?}",
                String::from_utf8_lossy(&mode)
            ))
        })?;

        let arch_name = state.arch().name();
        let (fd_off, struct_size) = io_file_for_arch(arch_name).ok_or_else(|| {
            ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}"))
        })?;

        let path_str = String::from_utf8_lossy(&path).to_string();
        // `None` = the fd space is exhausted (angr-03vl4.88) — see `NativeOpen`.
        let fd = state
            .file_system()
            .open_with_append(path_str, flags, append)
            .ok_or_else(|| ProcedureError::Other("fopen: fd space exhausted".to_string()))?;

        let file_ptr = state.heap_alloc(struct_size);
        state.memory_store(
            file_ptr.wrapping_add(fd_off),
            RustBV::concrete(fd as u128, 32),
        )?;

        Ok(Some(arch_word(state, file_ptr)))
    }
}

crate::declare_proc! {
    /// Native fdopen implementation.
    ///
    /// ```c
    /// FILE *fdopen(int fd, const char *mode);
    /// ```
    ///
    /// Allocates an `_IO_FILE` struct for an already-open fd, and returns a
    /// concrete NULL for an `fd` that is not open — a **successful** native
    /// return, not a `ProcedureError`, so Python never runs (angr-6cp06.11;
    /// the doc used to claim a fallback here that the code does not perform).
    /// Symbolic fds are already excluded by the `concrete` arg spec, so
    /// `fdopen.py`'s `get_concrete_fd` concretization path is unreachable
    /// natively either way.
    ///
    /// Parity: NULL matches `fdopen.py`'s `if fd_concr not in
    /// self.state.posix.fd: return 0`, for **every** mode. The
    /// `create_file=True` that `fdopen.py::create_file` passes for the
    /// `w`/`w+`/`a`/`a+` modes reads like a divergence — Python materializing a
    /// backing file where native returns NULL — but it is unreachable from
    /// here: `SimSystemPosix::_get_concrete_fd` only opens its
    /// `/tmp/angr_implicit_N` file on the branch where `eval_one` raises
    /// `SimSolverError`, i.e. a **multi-solution symbolic** fd. A concrete fd
    /// evaluates cleanly and is returned unchanged whatever the mode, so both
    /// engines take the `not in posix.fd` path together (angr-j0dp3). Hence
    /// the mode is parsed only to reject unsupported spellings; its `FdFlags`
    /// — and its append bit — are deliberately unused. That matches
    /// `fdopen.py`'s own `# TODO: handle append and other mode subtleties`:
    /// Python re-opens nothing and never re-flags the existing descriptor, so
    /// neither does this (unlike [`NativeFopen`], which mints the fd itself).
    name = "fdopen",
    struct = NativeFdopen,
    args = [fd_raw: concrete, mode_addr: concrete],
    call |state| {
        let fd = fd_raw as u32 as i32;
        let mode = read_cstring_strict(state, mode_addr, MAX_FOPEN_MODE_LEN, "mode")?;
        parse_fopen_mode(&mode).ok_or_else(|| {
            ProcedureError::Other(format!(
                "unsupported fdopen mode {:?}",
                String::from_utf8_lossy(&mode)
            ))
        })?;

        if fd < 0 || !state.file_system_ref().is_open(fd as u32) {
            return Ok(Some(arch_word(state, 0u64)));
        }

        let arch_name = state.arch().name();
        let (fd_off, struct_size) = io_file_for_arch(arch_name).ok_or_else(|| {
            ProcedureError::Other(format!("no _IO_FILE layout for arch {arch_name}"))
        })?;

        let file_ptr = state.heap_alloc(struct_size);
        state.memory_store(
            file_ptr.wrapping_add(fd_off),
            RustBV::concrete(fd as u32 as u128, 32),
        )?;

        Ok(Some(arch_word(state, file_ptr)))
    }
}

crate::declare_proc! {
    /// Native fclose implementation.
    ///
    /// ```c
    /// int fclose(FILE *stream);
    /// ```
    ///
    /// Reads `stream->_fileno` and calls FileSystem::close. Returns 0 on success
    /// and -1 if the fd was not open.
    name = "fclose",
    struct = NativeFclose,
    args = [file_ptr: concrete],
    call |state| {
        let fd = read_fileno(state, file_ptr)?;

        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        let ret = if state.file_system().close(fd as u32) {
            0
        } else {
            -1i64 as u64
        };
        Ok(Some(arch_word(state, ret)))
    }
}

crate::declare_proc! {
    /// Native fseek implementation.
    ///
    /// ```c
    /// int fseek(FILE *stream, long offset, int whence);
    /// ```
    ///
    /// Reads `stream->_fileno` and seeks via FileSystem::seek. Returns 0 on
    /// success and -1 on failure (matching glibc).
    name = "fseek",
    struct = NativeFseek,
    args = [file_ptr: concrete, offset_raw: concrete, whence_raw: concrete],
    aliases = ["fseeko"],
    call |state| {
        let offset = offset_raw as i64;
        let whence = whence_raw as u32;
        let fd = read_fileno(state, file_ptr)?;
        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        match state.file_system().seek(fd as u32, offset, whence) {
            Some(_) => Ok(Some(arch_word(state, 0u64))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
        }
    }
}

crate::declare_proc! {
    /// Native ftell implementation.
    ///
    /// ```c
    /// long ftell(FILE *stream);
    /// ```
    ///
    /// Reads `stream->_fileno` and returns the current position via
    /// FileSystem::fd_info. Returns -1 if the fd is not tracked.
    name = "ftell",
    struct = NativeFtell,
    args = [file_ptr: concrete],
    aliases = ["ftello"],
    call |state| {
        let fd = read_fileno(state, file_ptr)?;

        if fd < 0 {
            return Ok(Some(arch_word(state, -1i64 as u64)));
        }
        match state.file_system_ref().fd_info(fd as u32) {
            Some((_, pos, _, _, _)) => Ok(Some(arch_word(state, pos))),
            None => Ok(Some(arch_word(state, -1i64 as u64))),
        }
    }
}

crate::declare_proc! {
    /// Native rewind implementation.
    ///
    /// ```c
    /// void rewind(FILE *stream);
    /// ```
    ///
    /// Equivalent to `fseek(stream, 0, SEEK_SET)` but returns void. The exploration
    /// manager handles void returns by leaving the return register untouched.
    ///
    /// A seek failure (stale or never-registered `_fileno`) is deliberately
    /// swallowed — see the `SILENT(cat-a)` note in the body; `NativeFseek` is
    /// the procedure that reports the same failure, as `-1`.
    name = "rewind",
    struct = NativeRewind,
    args = [file_ptr: concrete],
    call |state| {
        let fd = read_fileno(state, file_ptr)?;

        if fd >= 0 {
            // SEEK_SET = 0.
            // SILENT(cat-a): rewind is void — C gives it no channel to report a
            // seek failure (callers are told to check ferror), so a seek on a
            // stale or never-registered fd is a no-op by design, not a lost
            // error. `NativeFseek` reports the same failure as -1.
            //
            // The workspace's `let_underscore_must_use = "deny"` does not reach
            // this line: clippy skips macro-expanded bodies, and every proc here
            // is one `declare_proc!` expansion. Hence the comment tag rather
            // than the `#[expect(..)]` form `CancelToken::cancel_for_budget`
            // uses — an `expect` here is reported as unfulfilled.
            let _ = state.file_system().seek(fd as u32, 0, 0);
        }
        Ok(None)
    }
}

test_submod!("stream_ops_tests.rs" => stream_ops_tests);
