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
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_state() -> RustSimState {
        // Two scratch pages plus the heap region.
        crate::procedures::test_util::amd64_state_with_regions(&[
            (0x1000, 0x1000),
            (0x2000, 0x1000),
            (0xC000_0000, 0x10000),
        ])
    }

    #[test]
    fn test_getenv_not_found() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"PATH\x00", Permission::RWX);

        let result = NativeGetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();

        // Should return NULL
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_setenv_then_getenv() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"HOME\x00", Permission::RWX);
        state.map_memory_data(0x1100, b"/root\x00", Permission::RWX);

        // setenv("HOME", "/root", 1)
        let result = NativeSetenv
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x1100, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // getenv("HOME")
        let result = NativeGetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();

        let ptr = result.unwrap().as_u64().unwrap();
        assert_ne!(ptr, 0, "getenv should return non-NULL");

        // Read back the value
        let mut val = Vec::new();
        for i in 0..5u64 {
            let byte = state.memory_load(ptr + i, 1).unwrap();
            val.push(byte.as_u64().unwrap() as u8);
        }
        assert_eq!(&val, b"/root");

        // NUL terminator
        let nul = state.memory_load(ptr + 5, 1).unwrap();
        assert_eq!(nul.as_u64(), Some(0));
    }

    #[test]
    fn test_setenv_no_overwrite() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);
        state.map_memory_data(0x1100, b"val1\x00", Permission::RWX);
        state.map_memory_data(0x1200, b"val2\x00", Permission::RWX);

        // setenv("KEY", "val1", 1)
        NativeSetenv
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x1100, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .unwrap();

        // setenv("KEY", "val2", 0) — should NOT overwrite
        NativeSetenv
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x1200, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // getenv should still return "val1"
        let result = NativeGetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        let ptr = result.unwrap().as_u64().unwrap();
        let mut val = Vec::new();
        for i in 0..4u64 {
            let byte = state.memory_load(ptr + i, 1).unwrap();
            val.push(byte.as_u64().unwrap() as u8);
        }
        assert_eq!(&val, b"val1");
    }

    #[test]
    fn test_putenv() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"LANG=en_US\x00", Permission::RWX);

        let result = NativePutenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Verify via state's environment
        assert_eq!(state.getenv(b"LANG"), Some(b"en_US".as_slice()));
    }

    #[test]
    fn test_putenv_no_equals() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"NOEQUALS\x00", Permission::RWX);

        // Should succeed but not add anything
        let result = NativePutenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_getenv_symbolic_name() {
        let mut state = setup_state();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "name", 64);
        drop(ctx);

        let result = NativeGetenv.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_setenv_symbolic_value_falls_back() {
        // The current native setenv reads the value byte-by-byte from
        // memory. If the value pointer is symbolic, the address-extraction
        // returns SymbolicArgument before any memory access happens.
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);
        let sym_value_ptr = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "value_ptr", 64)
        };

        let result = NativeSetenv.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                sym_value_ptr,
                RustBV::concrete(1, 64),
            ],
        );
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_putenv_symbolic_arg_falls_back() {
        let mut state = setup_state();
        let sym = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "str_ptr", 64)
        };

        let result = NativePutenv.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_unsetenv_removes_key() {
        let mut state = setup_state();
        state.setenv(b"FOO".to_vec(), b"bar".to_vec());
        state.map_memory_data(0x1000, b"FOO\x00", Permission::RWX);

        // unsetenv("FOO") returns 0
        let result = NativeUnsetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // gone
        assert_eq!(state.getenv(b"FOO"), None);
    }

    #[test]
    fn test_unsetenv_missing_key_returns_zero() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"MISSING\x00", Permission::RWX);

        let result = NativeUnsetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_unsetenv_symbolic_name() {
        let mut state = setup_state();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "name", 64);
        drop(ctx);

        let result = NativeUnsetenv.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_unsetenv_then_getenv_returns_null() {
        let mut state = setup_state();
        state.setenv(b"KEY".to_vec(), b"value".to_vec());
        state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);

        NativeUnsetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();

        let result = NativeGetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_clearenv_removes_all() {
        let mut state = setup_state();
        state.setenv(b"A".to_vec(), b"1".to_vec());
        state.setenv(b"B".to_vec(), b"2".to_vec());

        let result = NativeClearenv.call(&mut state, &[]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        assert_eq!(state.getenv(b"A"), None);
        assert_eq!(state.getenv(b"B"), None);
        assert!(state.environment().is_empty());
    }

    #[test]
    fn test_clearenv_on_empty_state() {
        let mut state = setup_state();
        let result = NativeClearenv.call(&mut state, &[]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_setenv_unsetenv_chain_visible_to_getenv() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"VAR\x00", Permission::RWX);
        state.map_memory_data(0x1100, b"hello\x00", Permission::RWX);

        // setenv("VAR", "hello", 1)
        NativeSetenv
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x1100, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .unwrap();
        assert_eq!(state.getenv(b"VAR"), Some(b"hello".as_slice()));

        // unsetenv("VAR")
        NativeUnsetenv
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(state.getenv(b"VAR"), None);

        // setenv("VAR", "hello", 1) again — overwrite-flag irrelevant since cleared
        NativeSetenv
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x1100, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(state.getenv(b"VAR"), Some(b"hello".as_slice()));

        // clearenv() wipes everything
        NativeClearenv.call(&mut state, &[]).unwrap();
        assert!(state.environment().is_empty());
    }

    #[test]
    fn test_getenv_env_preserved_on_fork() {
        let mut state = setup_state();
        state.setenv(b"KEY".to_vec(), b"value".to_vec());

        let forked = state.fork();
        assert_eq!(forked.getenv(b"KEY"), Some(b"value".as_slice()));
    }
}
