//! Shared `#[cfg(test)]` builders for procedure unit tests.
//!
//! Only helpers that are genuinely identical across families live here — the
//! state-construction core, plus the stdio-family FILE/symbolic-file builders
//! shared by `stdio_tests` and `fwrite_tests`. Per-family helpers with custom
//! signatures (e.g. `byteset_tests`'s `(s, set)` setup, or `fread_tests`'s
//! deliberately amd64-hardcoded `write_file_struct`) stay local to their
//! module.

use crate::memory::{PAGE_MASK, Permission};
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

/// Fresh amd64 state — the common base for every procedure test family.
pub(crate) fn amd64_state() -> RustSimState {
    RustSimState::new("amd64").unwrap()
}

/// amd64 state with each `(addr, size)` region mapped RWX. Mirrors the
/// region-mapping dance that the `setup_state` helpers used to inline.
pub(crate) fn amd64_state_with_regions(regions: &[(u64, u64)]) -> RustSimState {
    let mut state = amd64_state();
    for &(addr, size) in regions {
        state.map_memory(addr, size, Permission::RWX);
    }
    state
}

/// Build a FILE struct at `file_ptr` whose `_fileno` field holds `fd`, laid
/// out per the running arch's `_IO_FILE` offsets
/// ([`crate::procedures::fileops::io_file_for_arch`]). The page holding
/// `file_ptr` is mapped RWX first, with room for the struct plus its buffer.
///
/// Shared by every stdio-family test module, since all of those procedures
/// enter through `stream->_fileno`.
pub(crate) fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    let arch_name = state.arch().name();
    let (off, _size) =
        crate::procedures::fileops::io_file_for_arch(arch_name).expect("test arch supported");
    state.map_memory_data(file_ptr & !PAGE_MASK, &vec![0u8; 0x4000], Permission::RWX);
    let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
    // This helper is `#[cfg(test)]` but lives outside a `*_tests.rs` file, so
    // the overflow audit still scans it.
    // overflow-ok: both operands are test-chosen constants — `file_ptr` is a
    // literal from the caller, `off` a fixed per-arch struct offset.
    state.memory_store(file_ptr + off, fd_bv).unwrap();
}

/// Register `n` fully symbolic bytes as the content of `path`, then open it
/// read-only. Returns the fd. This is the angr-0xyq2 Phase 2 "bounded
/// symbolic file" shape the write-demotion tests need.
pub(crate) fn open_registered_sym_file(state: &mut RustSimState, path: &str, n: usize) -> u32 {
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..n)
            .map(|i| RustBV::symbolic(&ctx, format!("symfile_{i}"), 8))
            .collect()
    };
    state.file_system().register_file_content(path, bytes);
    state
        .file_system()
        .open(path.to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests")
}

/// Assert `err` is the `Other` fallback naming `what` as over-limit.
///
/// The procedures-side mirror of
/// [`crate::syscalls::tests_support::assert_over_limit`]; the two differ only
/// in error type, and neither family can use the other's. Deliberately matches
/// on the message and not just [`ProcedureError::Other`]: every fallback arm in
/// these procedures is an `Other`, so the weaker assertion passes even when the
/// bounce came from an unrelated check.
pub(crate) fn assert_over_limit(err: &crate::procedures::ProcedureError, what: &str) {
    assert!(
        matches!(err, crate::procedures::ProcedureError::Other(m)
                 if m.contains(what) && m.contains("exceeds limit")),
        "expected an over-limit fallback mentioning {what}, got {err:?}"
    );
}

/// Width, in bits, of a C `int` return value — what an `int`-typed native
/// procedure builds its return `RustBV` at: `fgets.rs`'s
/// `fgetc`/`getchar`/`getc`, `stdio.rs`'s
/// `fflush`/`setvbuf`/`feof`/`ferror`/`fputs`, plus `puts`, `printf`,
/// `getopt` and `posix_memalign`. Pointer/`size_t`/`ssize_t` returns use
/// [`arch_ret_bits`] instead.
pub(crate) const RET_INT_BITS: u32 = 32;

/// Width, in bits, of a pointer / `size_t` / `ssize_t` return value on
/// `state`'s arch — i.e. what [`crate::procedures::arch_word`] produces.
pub(crate) fn arch_ret_bits(state: &RustSimState) -> u32 {
    state.arch().bits()
}

/// Assert a native procedure's return value is exactly `bits` wide.
///
/// Every assertion on these returns used to be an `.as_u64()` value check, so
/// nothing in the suite could tell a correctly-typed return from one built at
/// the wrong width — the gap that let the return-register extension bug
/// (angr-fs8kb.40) go unnoticed across seven procedure families. The width is
/// observable to the guest: `write_proc_return_with_abi` zero- or
/// sign-extends a narrow value to the return register per
/// [`crate::arch::CallingConvention::return_extension`], so a return minted at
/// 64 bits where the C signature says `int` writes bits an `int`-returning
/// call can never set, with an identical `.as_u64()`.
pub(crate) fn assert_ret_width(ret: &RustBV, bits: u32, what: &str) {
    assert_eq!(
        ret.width(),
        bits,
        "{what} must return a {bits}-bit value, got {}",
        ret.width()
    );
}
