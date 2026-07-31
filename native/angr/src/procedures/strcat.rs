//! Native strcat and strncat implementations.
//!
//! Concrete string concatenation. Symbolic arguments fall back to Python.

use super::strings::{
    MAX_STRING_SCAN, find_null_addr, scan_concrete_bounded, scan_concrete_until_null, write_cstr,
};
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// strcat: append src string to dest.
    name = "strcat",
    struct = NativeStrcat,
    args = [dest: concrete, src: concrete],
    call |state| {
        let dest_end = find_null_addr(state, dest, MAX_STRING_SCAN, "dest")?;
        let buf = scan_concrete_until_null(state, src, MAX_STRING_SCAN, "src")?;

        write_cstr(state, dest_end, &buf)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(dest as u128, bits)))
    }
}

crate::declare_proc! {
    /// strncat: append at most n bytes from src to dest.
    name = "strncat",
    struct = NativeStrncat,
    args = [dest: concrete, src: concrete, n: concrete],
    call |state| {
        let dest_end = find_null_addr(state, dest, MAX_STRING_SCAN, "dest")?;
        let max_copy = n.min(MAX_STRING_SCAN as u64);

        // Copy at most `max_copy` non-null bytes from src.
        let (buf, _null_found) = scan_concrete_bounded(state, src, max_copy as usize, "src")?;

        // Always null-terminate after the copied bytes.
        write_cstr(state, dest_end, &buf)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(dest as u128, bits)))
    }
}

#[cfg(test)]
#[path = "strcat_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
