//! Native strcpy/strncpy implementation.
//!
//! # Behavior
//!
//! - If any address is symbolic, falls back to Python
//! - If any source byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes

use super::strings::{
    scan_concrete_bounded, scan_concrete_until_null, write_concrete_bytes, write_cstr,
};
use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

crate::declare_proc! {
    /// Native strcpy implementation.
    ///
    /// ```c
    /// char *strcpy(char *dest, const char *src);
    /// ```
    ///
    /// `dest` is declared `bv` so the original pointer BV is returned
    /// verbatim; it is extracted to a concrete u64 in the body.
    name = "strcpy",
    struct = NativeStrcpy,
    args = [dest_bv: bv, src: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        // Read source string up to (but not including) the null terminator.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        // Write to destination byte-by-byte, then the null terminator.
        write_cstr(state, dest, &buf)?;

        Ok(Some(dest_bv))
    }
}

crate::declare_proc! {
    /// Native strncpy implementation.
    ///
    /// ```c
    /// char *strncpy(char *dest, const char *src, size_t n);
    /// ```
    ///
    /// `dest` is declared `bv` so the original pointer BV is returned
    /// verbatim; it is extracted to a concrete u64 in the body.
    name = "strncpy",
    struct = NativeStrncpy,
    args = [dest_bv: bv, src: concrete, n: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        if n > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(n as usize));
        }

        // Read up to n bytes from source, stopping early at null. Pre-null
        // bytes go in `buf` (null itself excluded); when null is found we
        // pad the rest of the n-byte window with zeros.
        let (mut buf, null_found) = scan_concrete_bounded(state, src, n as usize, "src")?;
        if null_found {
            buf.resize(n as usize, 0);
        }

        write_concrete_bytes(state, dest, &buf)?;

        Ok(Some(dest_bv))
    }
}

crate::declare_proc! {
    /// Native stpcpy implementation.
    ///
    /// ```c
    /// char *stpcpy(char *dest, const char *src);
    /// ```
    ///
    /// Like `strcpy`, but returns `dest + strlen(src)` (a pointer to the
    /// written NUL) instead of `dest`. Reuses the same concrete-copy logic as
    /// `strcpy` (DRY); `strlen(src)` is the length of the scanned buffer.
    name = "stpcpy",
    struct = NativeStpcpy,
    args = [dest_bv: bv, src: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        // Read source string up to (but not including) the null terminator.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        // Write to destination byte-by-byte, then the null terminator.
        write_cstr(state, dest, &buf)?;

        // Return dest + strlen(src) (pointer to the written NUL).
        let bits = state.arch().bits();
        let len_bv = RustBV::concrete(buf.len() as u128, bits);
        let ret = {
            let ctx = state.solver().borrow();
            dest_bv.add(&len_bv, &ctx)
        };
        Ok(Some(ret))
    }
}

crate::declare_proc! {
    /// Native stpncpy implementation.
    ///
    /// ```c
    /// char *stpncpy(char *dest, const char *src, size_t n);
    /// ```
    ///
    /// Like `strncpy` (bounded copy + NUL-pad of the n-byte window), but returns
    /// `dest + min(strlen(src), n)` — a pointer to the written NUL, or `dest + n`
    /// when no NUL fits inside the window — instead of `dest`. Reuses the same
    /// concrete-copy logic as `strncpy` (DRY).
    name = "stpncpy",
    struct = NativeStpncpy,
    args = [dest_bv: bv, src: concrete, n: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        if n > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(n as usize));
        }

        let (mut buf, null_found) = scan_concrete_bounded(state, src, n as usize, "src")?;
        // Return offset is the #non-null bytes copied = min(strlen(src), n),
        // captured before the NUL-pad resize grows `buf` to the full window.
        let ret_off = buf.len() as u128;
        if null_found {
            buf.resize(n as usize, 0);
        }

        write_concrete_bytes(state, dest, &buf)?;

        let bits = state.arch().bits();
        let off_bv = RustBV::concrete(ret_off, bits);
        let ret = {
            let ctx = state.solver().borrow();
            dest_bv.add(&off_bv, &ctx)
        };
        Ok(Some(ret))
    }
}

crate::declare_proc! {
    /// Native strdup implementation.
    ///
    /// ```c
    /// char *strdup(const char *s);
    /// ```
    ///
    /// Allocates a new string via heap_alloc, copies the source string
    /// (including null terminator), and returns pointer to the new string.
    name = "strdup",
    struct = NativeStrdup,
    args = [src: concrete],
    call |state| {
        // Read source string up to (but not including) the null terminator.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        // Allocate new buffer (strlen + 1 for null terminator).
        let new_addr = state.heap_alloc(buf.len() as u64 + 1);

        write_cstr(state, new_addr, &buf)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(new_addr as u128, bits)))
    }
}

crate::declare_proc! {
    /// Native strxfrm implementation.
    ///
    /// ```c
    /// size_t strxfrm(char *dest, const char *src, size_t n);
    /// ```
    ///
    /// In the C/POSIX locale (angr's default, matching
    /// `procedures/libc/strxfrm.py`) the transform degenerates to
    /// `strncpy(dest, src, n)` followed by returning `strlen(src)`. We scan
    /// the full source once for the length, then write the bounded+NUL-padded
    /// `n`-byte window exactly as `strncpy` would (reusing the same string
    /// helpers, DRY). Note the return value is the *untruncated* source
    /// length, so it can exceed `n` (snprintf-style "would-be" length).
    name = "strxfrm",
    struct = NativeStrxfrm,
    args = [dest_bv: bv, src: concrete, n: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        if n > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(n as usize));
        }

        // Full source length (strlen), excluding the NUL — this is the return.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;
        let src_len = buf.len();

        // strncpy(dest, src, n): first min(n, src_len) bytes, NUL-padded to n.
        let n_usize = n as usize;
        let mut out = buf;
        out.truncate(n_usize);
        out.resize(n_usize, 0);
        write_concrete_bytes(state, dest, &out)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(src_len as u128, bits)))
    }
}

#[cfg(test)]
#[path = "strcpy_tests.rs"]
mod tests;
