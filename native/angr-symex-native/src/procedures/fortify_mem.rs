//! Native fortify-source `_chk` wrappers for the mem family.
//!
//! glibc's `_FORTIFY_SOURCE` build mode redirects `memcpy`/`memmove`/`memset`/
//! `mempcpy` to `__*_chk` variants that take an extra trailing `destlen`
//! argument (the compiler-known size of the destination object) and abort via
//! `__chk_fail` when the copy length provably exceeds it.
//!
//! Matching Python angr (`procedures/libc/{memcpy,memmove,memset,mempcpy}.py`),
//! the `_chk` variants **ignore** `destlen` and forward to the base routine.
//! Emulating the runtime bound check would diverge from the Python engine and
//! prune paths angr otherwise explores, so we deliberately do not. These
//! wrappers therefore add zero copy logic — they delegate to the existing
//! native `memcpy`/`memmove`/`memset`/`mempcpy` impls (DRY) and only drop the
//! extra `destlen` arg.

use super::memcpy::{NativeMemcpy, NativeMemmove, NativeMempcpy};
use super::memset::NativeMemset;

crate::declare_proc! {
    /// `void *__memcpy_chk(void *dest, const void *src, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `memcpy`, dropping `destlen` (matches Python angr).
    name = "__memcpy_chk",
    struct = NativeMemcpyChk,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv, _destlen: bv],
    call |state| {
        NativeMemcpy.call(state, &[dst_bv, src_bv, size_bv])
    }
}

crate::declare_proc! {
    /// `void *__memmove_chk(void *dest, const void *src, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `memmove`, dropping `destlen` (matches Python angr).
    name = "__memmove_chk",
    struct = NativeMemmoveChk,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv, _destlen: bv],
    call |state| {
        NativeMemmove.call(state, &[dst_bv, src_bv, size_bv])
    }
}

crate::declare_proc! {
    /// `void *__memset_chk(void *dest, int c, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `memset`, dropping `destlen` (matches Python angr).
    name = "__memset_chk",
    struct = NativeMemsetChk,
    args = [dst_bv: bv, char_bv: bv, size_bv: bv, _destlen: bv],
    call |state| {
        NativeMemset.call(state, &[dst_bv, char_bv, size_bv])
    }
}

crate::declare_proc! {
    /// `void *__mempcpy_chk(void *dest, const void *src, size_t n, size_t destlen)`.
    ///
    /// Forwards to native `mempcpy` (which returns `dest + n`), dropping
    /// `destlen` (matches Python angr).
    name = "__mempcpy_chk",
    struct = NativeMempcpyChk,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv, _destlen: bv],
    call |state| {
        NativeMempcpy.call(state, &[dst_bv, src_bv, size_bv])
    }
}

test_submod!("fortify_mem_tests.rs" => tests);
