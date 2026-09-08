//! Native getenv/setenv/putenv implementations.
//!
//! Uses per-state environment map (`HashMap<Vec<u8>, Vec<u8>>`).
//! - getenv: looks up key, allocates heap buffer for value, returns pointer (or NULL)
//! - setenv: stores key=value in the environment map
//! - putenv: parses "KEY=VALUE" string and stores in environment map
//!
//! Concrete keys only — falls back to Python for symbolic arguments.
//!
//! The map is **seeded** with the guest's initial `entry_state(env=...)`
//! environment by `rust_manager.py::_seed_environ_to_rust`, which walks the
//! memory-backed `envp` array angr's `simos/linux.py` builds and pushes the
//! concrete `KEY=VALUE` pairs through `RustSimState::seed_environment`
//! (angr-6cp06.12). Before that seed existed the map only ever held vars the
//! guest set at runtime, so `getenv` on a harness-supplied var returned NULL
//! where Python's `getenv` SimProcedure — which reads that same `envp` array —
//! finds it.
//!
//! Residual divergence from Python, both narrow: an env entry whose bytes are
//! symbolic is skipped by the bridge (Python's `getenv` would still return a
//! pointer to it), and the map is write-only with respect to memory — a guest
//! that walks `environ`/`__environ` itself, rather than calling `getenv`, does
//! not see a native `setenv`/`putenv`/`unsetenv`.

use super::strings::{MAX_STRING_SCAN, scan_concrete_bounded, write_cstr};
use super::{ProcedureError, arch_word};
use crate::state::RustSimState;

/// Read a null-terminated concrete string from memory (up to [`MAX_STRING_SCAN`]
/// bytes; exhausting the cap without a null is **not** an error — the prefix is
/// returned as-is). A symbolic byte or out-of-bounds read propagates as an `Err`
/// and falls back to Python.
///
/// The `_tolerant` suffix distinguishes this from `fileops.rs`'s
/// `read_cstring_strict`, which errors out on an unterminated buffer instead
/// (angr-03vl4.47): the two used to share the bare name `read_cstring` with
/// silently opposite cap-exhaustion contracts.
fn read_cstring_tolerant(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let (buf, _null_found) = scan_concrete_bounded(state, addr, MAX_STRING_SCAN, "string")?;
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
        let key = read_cstring_tolerant(state, name_addr)?;

        match state.getenv(&key) {
            Some(value) => {
                let value = value.to_vec(); // clone before mutable borrow
                // Allocate heap space for value + NUL
                let buf_addr = state.heap_alloc(value.len() as u64 + 1);
                // Write value bytes + NUL terminator
                write_cstr(state, buf_addr, &value)?;

                Ok(Some(arch_word(state, buf_addr)))
            }
            None => {
                // Not found — return NULL
                Ok(Some(arch_word(state, 0u64)))
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
        let key = read_cstring_tolerant(state, name_addr)?;
        let value = read_cstring_tolerant(state, value_addr)?;

        // Only set if overwrite is non-zero or key doesn't exist
        if overwrite != 0 || state.getenv(&key).is_none() {
            state.setenv(key, value);
        }

        Ok(Some(arch_word(state, 0u64))) // success
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
        let key = read_cstring_tolerant(state, name_addr)?;
        state.unsetenv(&key);

        Ok(Some(arch_word(state, 0u64))) // success
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
        Ok(Some(arch_word(state, 0u64))) // success
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
        let s = read_cstring_tolerant(state, str_addr)?;

        // Find '=' separator
        if let Some(eq_pos) = s.iter().position(|&b| b == b'=') {
            let key = s[..eq_pos].to_vec();
            let value = s[eq_pos + 1..].to_vec();
            state.setenv(key, value);
        }
        // If no '=', putenv behavior is implementation-defined — just ignore

        Ok(Some(arch_word(state, 0u64))) // success
    }
}

test_submod!("getenv_tests.rs" => getenv_tests);
