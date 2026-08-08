//! Native fortify-source `_chk` wrappers for the string family.
//!
//! glibc's `_FORTIFY_SOURCE` build mode redirects `strcpy`/`strncpy`/`strcat`/
//! `strncat`/`stpcpy` to `__*_chk` variants that take an extra trailing
//! `destlen` argument (the compiler-known size of the destination object) and
//! abort via `__chk_fail` when the write provably overflows it.
//!
//! Matching Python angr (`procedures/libc/{strcpy,strncpy,strcat,strncat,
//! stpcpy}.py`, where each `__*_chk` subclasses the base and drops `destlen`),
//! the native variants **ignore** `destlen` and forward to the base routine.
//! Emulating the runtime bound check would diverge from the Python engine and
//! prune paths angr otherwise explores, so we deliberately do not. These
//! wrappers add zero copy logic — they delegate to the existing native
//! `strcpy`/`strncpy`/`strcat`/`strncat` impls (DRY) and only drop the extra
//! arg.
//!
//! `__stpcpy_chk` forwards to the native `stpcpy` base (which returns
//! `dest + strlen(src)`, the pointer to the written NUL), dropping `destlen`
//! like the other wrappers.

use super::strcat::{NativeStrcat, NativeStrncat};
use super::strcpy::{NativeStpcpy, NativeStrcpy, NativeStrncpy};

crate::declare_proc! {
    /// `char *__strcpy_chk(char *dest, const char *src, size_t destlen)`.
    ///
    /// Forwards to native `strcpy`, dropping `destlen` (matches Python angr).
    name = "__strcpy_chk",
    struct = NativeStrcpyChk,
    args = [dest_bv: bv, src_bv: bv, _destlen: bv],
    call |state| {
        NativeStrcpy.call(state, &[dest_bv, src_bv])
    }
}

crate::declare_proc! {
    /// `char *__strncpy_chk(char *dest, const char *src, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `strncpy`, dropping `destlen` (matches Python angr).
    name = "__strncpy_chk",
    struct = NativeStrncpyChk,
    args = [dest_bv: bv, src_bv: bv, n_bv: bv, _destlen: bv],
    call |state| {
        NativeStrncpy.call(state, &[dest_bv, src_bv, n_bv])
    }
}

crate::declare_proc! {
    /// `char *__strcat_chk(char *dest, const char *src, size_t destlen)`.
    ///
    /// Forwards to native `strcat`, dropping `destlen` (matches Python angr).
    name = "__strcat_chk",
    struct = NativeStrcatChk,
    args = [dest_bv: bv, src_bv: bv, _destlen: bv],
    call |state| {
        NativeStrcat.call(state, &[dest_bv, src_bv])
    }
}

crate::declare_proc! {
    /// `char *__strncat_chk(char *dest, const char *src, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `strncat`, dropping `destlen` (matches Python angr).
    name = "__strncat_chk",
    struct = NativeStrncatChk,
    args = [dest_bv: bv, src_bv: bv, n_bv: bv, _destlen: bv],
    call |state| {
        NativeStrncat.call(state, &[dest_bv, src_bv, n_bv])
    }
}

crate::declare_proc! {
    /// `char *__stpcpy_chk(char *dest, const char *src, size_t destlen)`.
    ///
    /// Forwards to native `stpcpy` (which returns `dest + strlen(src)`),
    /// dropping `destlen` (matches Python angr).
    name = "__stpcpy_chk",
    struct = NativeStpcpyChk,
    args = [dest_bv: bv, src_bv: bv, _destlen: bv],
    call |state| {
        NativeStpcpy.call(state, &[dest_bv, src_bv])
    }
}

test_submod!("fortify_str_tests.rs" => tests);
