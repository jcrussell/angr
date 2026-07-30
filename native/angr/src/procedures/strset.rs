//! Native byte-set search procedures: strpbrk, strspn, strcspn.
//!
//! All three take `(s, set)` where `set` is a null-terminated byte set. They
//! walk `s` byte-by-byte until the null terminator and consult a precomputed
//! lookup table built from the `set` arg.
//!
//! - `strpbrk(s, accept)`: returns pointer to first byte of `s` that is in
//!   `accept`, or NULL.
//! - `strspn(s, accept)`: returns length of the initial prefix of `s`
//!   consisting entirely of bytes from `accept`.
//! - `strcspn(s, reject)`: returns length of the initial prefix of `s`
//!   consisting entirely of bytes NOT in `reject`.
//!
//! Concrete-only: any symbolic argument (s addr, set addr, or a string byte)
//! falls back to Python. The set arg is read once into a fixed 256-bit
//! lookup table.

use super::strings::{MAX_STRING_SCAN, scan_concrete_until_null};
use super::{ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_SET: usize = 256;

/// Build a 256-entry byte-set lookup table from the null-terminated string
/// at `set_addr`. Bails on symbolic bytes via the shared concrete-scan
/// helper.
fn build_byte_set(
    state: &mut RustSimState,
    set_addr: u64,
    label: &str,
) -> Result<[bool; 256], ProcedureError> {
    let set_bytes = scan_concrete_until_null(state, set_addr, MAX_SET, label)?;
    let mut table = [false; 256];
    for &b in &set_bytes {
        table[b as usize] = true;
    }
    Ok(table)
}

crate::declare_proc! {
    /// strpbrk: find first byte of `s` that is in `accept`.
    ///
    /// ```c
    /// char *strpbrk(const char *s, const char *accept);
    /// ```
    name = "strpbrk",
    struct = NativeStrpbrk,
    args = [s_addr: concrete, accept_addr: concrete],
    call |state| {
        let bits = state.arch().bits();

        let accept = build_byte_set(state, accept_addr, "accept")?;

        for i in 0..MAX_STRING_SCAN as u64 {
            let byte_addr = s_addr.wrapping_add(i);
            let byte_val = state.memory_load(byte_addr, 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{i}]"))? as u8;
            if byte == 0 {
                return Ok(Some(RustBV::concrete(0u128, bits)));
            }
            if accept[byte as usize] {
                return Ok(Some(RustBV::concrete(byte_addr as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_STRING_SCAN))
    }
}

crate::declare_proc! {
    /// strspn: length of prefix of `s` consisting entirely of bytes from `accept`.
    ///
    /// ```c
    /// size_t strspn(const char *s, const char *accept);
    /// ```
    name = "strspn",
    struct = NativeStrspn,
    args = [s_addr: concrete, accept_addr: concrete],
    call |state| {
        let bits = state.arch().bits();

        let accept = build_byte_set(state, accept_addr, "accept")?;

        for i in 0..MAX_STRING_SCAN as u64 {
            let byte_val = state.memory_load(s_addr.wrapping_add(i), 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{i}]"))? as u8;
            if byte == 0 || !accept[byte as usize] {
                return Ok(Some(RustBV::concrete(i as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_STRING_SCAN))
    }
}

crate::declare_proc! {
    /// strcspn: length of prefix of `s` consisting entirely of bytes NOT in `reject`.
    ///
    /// ```c
    /// size_t strcspn(const char *s, const char *reject);
    /// ```
    name = "strcspn",
    struct = NativeStrcspn,
    args = [s_addr: concrete, reject_addr: concrete],
    call |state| {
        let bits = state.arch().bits();

        let reject = build_byte_set(state, reject_addr, "reject")?;

        for i in 0..MAX_STRING_SCAN as u64 {
            let byte_val = state.memory_load(s_addr.wrapping_add(i), 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{i}]"))? as u8;
            if byte == 0 || reject[byte as usize] {
                return Ok(Some(RustBV::concrete(i as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_STRING_SCAN))
    }
}

#[cfg(test)]
#[path = "strset_tests.rs"]
mod strset_tests;
