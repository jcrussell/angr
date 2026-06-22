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
//! `__stpcpy_chk` is the one exception: there is no native `stpcpy` base, so it
//! reuses the native `strcpy` copy logic and adjusts the return value to
//! `dest + strlen(src)` (the pointer to the written NUL), matching glibc and
//! Python's `stpcpy`.

use super::extract_concrete_arg;
use super::strcat::{NativeStrcat, NativeStrncat};
use super::strcpy::{NativeStrcpy, NativeStrncpy};
use super::strings::find_null_addr;
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

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
    /// `stpcpy` is `strcpy` that returns `dest + strlen(src)` (a pointer to the
    /// written NUL) instead of `dest`. There is no native `stpcpy` base, so we
    /// reuse the native `strcpy` copy logic and adjust the return value (DRY);
    /// `destlen` is dropped (matches Python angr).
    name = "__stpcpy_chk",
    struct = NativeStpcpyChk,
    args = [dest_bv: bv, src_bv: bv, _destlen: bv],
    call |state| {
        let src = extract_concrete_arg(&src_bv, "src")?;
        NativeStrcpy.call(state, &[dest_bv.clone(), src_bv])?;
        // strlen(src) = (address of NUL) - src.
        let src_len = find_null_addr(state, src, MAX_STRLEN, "src")?.wrapping_sub(src);
        let bits = state.arch().bits();
        let len_bv = RustBV::concrete(src_len as u128, bits);
        let ret = {
            let ctx = state.solver().borrow();
            dest_bv.add(&len_bv, &ctx)
        };
        Ok(Some(ret))
    }
}

#[cfg(test)]
#[path = "fortify_str_tests.rs"]
mod tests;
