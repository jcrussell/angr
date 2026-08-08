//! Native strcat and strncat implementations.
//!
//! Concrete string concatenation. Symbolic arguments fall back to Python.

use super::strings::{
    MAX_STRING_SCAN, find_null_addr, scan_concrete_bounded, scan_concrete_until_null, write_cstr,
};
use super::{arch_word, check_max};

crate::declare_proc! {
    /// strcat: append src string to dest.
    name = "strcat",
    struct = NativeStrcat,
    args = [dest: concrete, src: concrete],
    call |state| {
        let dest_end = find_null_addr(state, dest, MAX_STRING_SCAN, "dest")?;
        let buf = scan_concrete_until_null(state, src, MAX_STRING_SCAN, "src")?;

        write_cstr(state, dest_end, &buf)?;

        Ok(Some(arch_word(state, dest)))
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
        let (buf, null_found) = scan_concrete_bounded(state, src, max_copy as usize, "src")?;

        // `n` past the scan cap is only servable when src's terminator landed
        // inside the cap — then the copy is exactly the same as for an
        // unbounded `n`. Otherwise the answer would be a silently truncated
        // 4096-byte copy, so defer to Python like the sibling strncpy does.
        if !null_found {
            check_max(n, MAX_STRING_SCAN)?;
        }

        // Always null-terminate after the copied bytes.
        write_cstr(state, dest_end, &buf)?;

        Ok(Some(arch_word(state, dest)))
    }
}

test_submod!("strcat_tests.rs" => tests);
