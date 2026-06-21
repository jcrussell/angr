//! Native getenv/setenv/putenv implementations.
//!
//! Uses per-state environment map (`HashMap<Vec<u8>, Vec<u8>>`).
//! - getenv: looks up key, allocates heap buffer for value, returns pointer (or NULL)
//! - setenv: stores key=value in the environment map
//! - putenv: parses "KEY=VALUE" string and stores in environment map
//!
//! Concrete keys only — falls back to Python for symbolic arguments.

use super::ProcedureError;
use super::strings::{scan_concrete_bounded, write_cstr};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_STR_LEN: usize = 4096;

/// Read a null-terminated concrete string from memory (up to `MAX_STR_LEN`
/// bytes; exhausting the cap without a null is not an error). A symbolic byte
/// or out-of-bounds read propagates as an `Err` and falls back to Python.
fn read_cstring(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let (buf, _null_found) = scan_concrete_bounded(state, addr, MAX_STR_LEN, "string")?;
    Ok(buf)
}

crate::declare_proc! {
    /// getenv: look up `name`, heap-allocate the value, return a pointer (or NULL).
    ///
    /// ```c
    /// char *getenv(const char *name);
    /// ```
    name = "getenv",
    struct = NativeGetenv,
    args = [name_addr: concrete],
    call |state| {
        let key = read_cstring(state, name_addr)?;
        let bits = state.arch().bits();

        match state.getenv(&key) {
            Some(value) => {
                let value = value.to_vec(); // clone before mutable borrow
                // Allocate heap space for value + NUL
                let buf_addr = state.heap_alloc(value.len() as u64 + 1);
                // Write value bytes + NUL terminator
                write_cstr(state, buf_addr, &value)?;

                Ok(Some(RustBV::concrete(buf_addr as u128, bits)))
            }
            None => {
                // Not found — return NULL
                Ok(Some(RustBV::concrete(0, bits)))
            }
        }
    }
}

crate::declare_proc! {
    /// setenv: store `name=value` in the environment map. Returns 0 on success.
    ///
    /// ```c
    /// int setenv(const char *name, const char *value, int overwrite);
    /// ```
    name = "setenv",
    struct = NativeSetenv,
    args = [name_addr: concrete, value_addr: concrete, overwrite: concrete],
    call |state| {
        let key = read_cstring(state, name_addr)?;
        let value = read_cstring(state, value_addr)?;

        // Only set if overwrite is non-zero or key doesn't exist
        if overwrite != 0 || state.getenv(&key).is_none() {
            state.setenv(key, value);
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

crate::declare_proc! {
    /// unsetenv: remove `name` from the environment. Returns 0 on success.
    ///
    /// ```c
    /// int unsetenv(const char *name);
    /// ```
    ///
    /// Per POSIX, returns 0 if the name was absent too — only sets errno
    /// for invalid names (NULL, empty, contains '='), which we do not
    /// model here. Symbolic name → Python fallback.
    name = "unsetenv",
    struct = NativeUnsetenv,
    args = [name_addr: concrete],
    call |state| {
        let key = read_cstring(state, name_addr)?;
        state.unsetenv(&key);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

crate::declare_proc! {
    /// clearenv: remove all environment variables. Returns 0 on success.
    ///
    /// ```c
    /// int clearenv(void);
    /// ```
    name = "clearenv",
    struct = NativeClearenv,
    args = [],
    call |state| {
        state.clearenv();
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

crate::declare_proc! {
    /// putenv: parse "KEY=VALUE" and store in the environment. Returns 0.
    ///
    /// ```c
    /// int putenv(char *string);
    /// ```
    name = "putenv",
    struct = NativePutenv,
    args = [str_addr: concrete],
    call |state| {
        let s = read_cstring(state, str_addr)?;

        // Find '=' separator
        if let Some(eq_pos) = s.iter().position(|&b| b == b'=') {
            let key = s[..eq_pos].to_vec();
            let value = s[eq_pos + 1..].to_vec();
            state.setenv(key, value);
        }
        // If no '=', putenv behavior is implementation-defined — just ignore

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

#[cfg(test)]
#[path = "getenv_tests.rs"]
mod getenv_tests;
