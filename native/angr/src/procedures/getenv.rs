//! Native getenv/setenv/putenv implementations.
//!
//! Uses per-state environment map (HashMap<Vec<u8>, Vec<u8>>).
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
                    state
                        .memory_store(
                            buf_addr.wrapping_add(i as u64),
                            RustBV::concrete(byte as u128, 8),
                        )
                        ?;
                }
                // NUL terminator
                state
                    .memory_store(
                        buf_addr.wrapping_add(value.len() as u64),
                        RustBV::concrete(0, 8),
                    )
                    ?;

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
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x1000, 0x1000, Permission::RWX);
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        // Map heap region
        state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
        state
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
    fn test_getenv_env_preserved_on_fork() {
        let mut state = setup_state();
        state.setenv(b"KEY".to_vec(), b"value".to_vec());

        let forked = state.fork();
        assert_eq!(forked.getenv(b"KEY"), Some(b"value".as_slice()));
    }
}
