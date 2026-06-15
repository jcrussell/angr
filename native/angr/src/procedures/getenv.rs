//! Native getenv/setenv/putenv implementations.
//!
//! Uses per-state environment map (`HashMap<Vec<u8>, Vec<u8>>`).
//! - getenv: looks up key, allocates heap buffer for value, returns pointer (or NULL)
//! - setenv: stores key=value in the environment map
//! - putenv: parses "KEY=VALUE" string and stores in environment map
//!
//! Concrete keys only — falls back to Python for symbolic arguments.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_STR_LEN: usize = 4096;

/// Read a null-terminated concrete string from memory.
fn read_cstring(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let mut buf = Vec::new();
    for i in 0..MAX_STR_LEN as u64 {
        match state.memory_load(addr.wrapping_add(i), 1) {
            Ok(bv) => {
                let byte = extract_concrete_arg(&bv, "string byte")? as u8;
                if byte == 0 {
                    break;
                }
                buf.push(byte);
            }
            Err(_) => break,
        }
    }
    Ok(buf)
}

/// Native getenv implementation.
///
/// ```c
/// char *getenv(const char *name);
/// ```
///
/// Returns a pointer to the value string, or NULL if not found.
/// The value is heap-allocated and written to memory.
pub struct NativeGetenv;

impl NativeSimProcedure for NativeGetenv {
    fn name(&self) -> &'static str {
        "getenv"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let name_addr = extract_concrete_arg(&args[0], "name")?;

        let key = read_cstring(state, name_addr)?;
        let bits = state.arch().bits();

        match state.getenv(&key) {
            Some(value) => {
                let value = value.to_vec(); // clone before mutable borrow
                // Allocate heap space for value + NUL
                let buf_addr = state.heap_alloc(value.len() as u64 + 1);
                // Write value bytes
                for (i, &byte) in value.iter().enumerate() {
                    state.memory_store(
                        buf_addr.wrapping_add(i as u64),
                        RustBV::concrete(byte as u128, 8),
                    )?;
                }
                // NUL terminator
                state.memory_store(
                    buf_addr.wrapping_add(value.len() as u64),
                    RustBV::concrete(0, 8),
                )?;

                Ok(Some(RustBV::concrete(buf_addr as u128, bits)))
            }
            None => {
                // Not found — return NULL
                Ok(Some(RustBV::concrete(0, bits)))
            }
        }
    }
}

/// Native setenv implementation.
///
/// ```c
/// int setenv(const char *name, const char *value, int overwrite);
/// ```
///
/// Returns 0 on success.
pub struct NativeSetenv;

impl NativeSimProcedure for NativeSetenv {
    fn name(&self) -> &'static str {
        "setenv"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let name_addr = extract_concrete_arg(&args[0], "name")?;
        let value_addr = extract_concrete_arg(&args[1], "value")?;
        let overwrite = extract_concrete_arg(&args[2], "overwrite")?;

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

/// Native unsetenv implementation.
///
/// ```c
/// int unsetenv(const char *name);
/// ```
///
/// Removes `name` from the environment. Returns 0 on success.
/// Per POSIX, returns 0 if the name was absent too — only sets errno
/// for invalid names (NULL, empty, contains '='), which we do not
/// model here. Symbolic name → Python fallback.
pub struct NativeUnsetenv;

impl NativeSimProcedure for NativeUnsetenv {
    fn name(&self) -> &'static str {
        "unsetenv"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let name_addr = extract_concrete_arg(&args[0], "name")?;

        let key = read_cstring(state, name_addr)?;
        state.unsetenv(&key);

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

/// Native clearenv implementation.
///
/// ```c
/// int clearenv(void);
/// ```
///
/// Removes all environment variables. Returns 0 on success.
pub struct NativeClearenv;

impl NativeSimProcedure for NativeClearenv {
    fn name(&self) -> &'static str {
        "clearenv"
    }

    fn num_args(&self) -> usize {
        0
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        state.clearenv();
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(0, bits))) // success
    }
}

/// Native putenv implementation.
///
/// ```c
/// int putenv(char *string);
/// ```
///
/// Takes "KEY=VALUE" string. Stores key and value in environment.
pub struct NativePutenv;

impl NativeSimProcedure for NativePutenv {
    fn name(&self) -> &'static str {
        "putenv"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let str_addr = extract_concrete_arg(&args[0], "string")?;

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
