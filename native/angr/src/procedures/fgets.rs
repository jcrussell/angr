//! Native fgets, fgetc, and getchar implementations.
//!
//! These procedures handle stdin input by creating symbolic bytes.
//! fgets stores symbolic bytes to a buffer, fgetc/getchar return a single symbolic byte.
//!
//! # Behavior
//!
//! - For stdin: creates symbolic BVS variables representing user input
//! - fgets: stores up to (size-1) symbolic bytes + NUL terminator at buf
//! - fgetc: returns one symbolic byte zero-extended to int
//! - getchar: equivalent to fgetc(stdin)
//! - Non-stdin FILE* streams fall back to Python

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FGETS_SIZE: u64 = 4096;

/// Native fgets implementation.
///
/// ```c
/// char *fgets(char *s, int size, FILE *stream);
/// ```
///
/// Creates (size-1) symbolic bytes and a NUL terminator at the buffer.
/// Returns the buffer address on success.
/// The FILE* stream argument is ignored — all streams treated as stdin.
pub struct NativeFgets;

impl NativeSimProcedure for NativeFgets {
    fn name(&self) -> &'static str {
        "fgets"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let buf = extract_concrete_arg(&args[0], "s")?;
        let size = extract_concrete_arg(&args[1], "size")?;
        // args[2] is FILE* stream — ignored (treated as stdin)

        if size == 0 {
            // fgets with size 0 returns NULL
            let bits = state.arch().bits();
            return Ok(Some(RustBV::concrete(0, bits)));
        }

        if size > MAX_FGETS_SIZE {
            return Err(ProcedureError::Other(format!(
                "fgets size {} exceeds limit",
                size
            )));
        }

        let read_count = size - 1; // fgets reads at most size-1 bytes
        let read_id = symbol_counter("fgets");

        // Create symbolic bytes and record for stdin tracking
        let names: Vec<String> = (0..read_count)
            .map(|i| format!("stdin_fgets_{}_{}", read_id, i))
            .collect();

        let sym_bytes: Vec<RustBV> = {
            let ctx = state.solver().borrow();
            names
                .iter()
                .map(|name| RustBV::symbolic(&ctx, name, 8))
                .collect()
        };

        // Record stdin symbols for posix.dumps(0) export
        for name in &names {
            state.record_stdin_symbol(name.clone(), 8);
        }

        // Store symbolic bytes to buffer
        for (i, sym_byte) in sym_bytes.into_iter().enumerate() {
            state
                .memory_store(buf.wrapping_add(i as u64), sym_byte)
                ?;
        }

        // Store NUL terminator
        state
            .memory_store(buf.wrapping_add(read_count), RustBV::concrete(0, 8))
            ?;

        // Return buffer address
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(buf as u128, bits)))
    }
}

/// Native fgetc implementation.
///
/// ```c
/// int fgetc(FILE *stream);
/// ```
///
/// Returns one symbolic byte zero-extended to int size.
/// The FILE* stream argument is ignored — all streams treated as stdin.
pub struct NativeFgetc;

impl NativeSimProcedure for NativeFgetc {
    fn name(&self) -> &'static str {
        "fgetc"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // args[0] is FILE* stream — ignored (treated as stdin)
        let _ = &args[0];

        let read_id = symbol_counter("fgetc");
        let name = format!("stdin_fgetc_{}", read_id);
        let result = {
            let ctx = state.solver().borrow();
            let sym_byte = RustBV::symbolic(&ctx, &name, 8);
            // Zero-extend to int (32-bit, matching C int type)
            sym_byte.zero_extend(32, &ctx)
        };
        // Record for posix.dumps(0) export
        state.record_stdin_symbol(name, 8);
        Ok(Some(result))
    }
}

/// Native getchar implementation.
///
/// ```c
/// int getchar(void);
/// ```
///
/// Equivalent to fgetc(stdin). Returns one symbolic byte zero-extended to int.
pub struct NativeGetchar;

impl NativeSimProcedure for NativeGetchar {
    fn name(&self) -> &'static str {
        "getchar"
    }

    fn num_args(&self) -> usize {
        0
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let read_id = symbol_counter("getchar");
        let name = format!("stdin_getchar_{}", read_id);
        let result = {
            let ctx = state.solver().borrow();
            let sym_byte = RustBV::symbolic(&ctx, &name, 8);
            // Zero-extend to int (32-bit, matching C int type)
            sym_byte.zero_extend(32, &ctx)
        };
        // Record for posix.dumps(0) export
        state.record_stdin_symbol(name, 8);
        Ok(Some(result))
    }
}

/// Native getc implementation (alias for fgetc).
pub struct NativeGetc;

impl NativeSimProcedure for NativeGetc {
    fn name(&self) -> &'static str {
        "getc"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        NativeFgetc.call(state, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_fgets_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, Permission::RWX);

        let result = NativeFgets
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64), // buf
                    RustBV::concrete(10, 64),     // size
                    RustBV::concrete(0, 64),      // stream (stdin)
                ],
            )
            .unwrap();

        // Should return buf address
        assert_eq!(result.unwrap().as_u64(), Some(0x2000));

        // First 9 bytes should be symbolic
        for i in 0..9u64 {
            let byte = state.memory_load(0x2000 + i, 1).unwrap();
            assert!(byte.as_u64().is_none(), "byte {} should be symbolic", i);
        }

        // Byte 9 should be NUL terminator
        let nul = state.memory_load(0x2009, 1).unwrap();
        assert_eq!(nul.as_u64(), Some(0));
    }

    #[test]
    fn test_fgets_size_1() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, Permission::RWX);

        let result = NativeFgets
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64), // size=1 means only NUL
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0x2000));

        // Only NUL should be written
        let nul = state.memory_load(0x2000, 1).unwrap();
        assert_eq!(nul.as_u64(), Some(0));
    }

    #[test]
    fn test_fgets_size_0() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeFgets
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64), // size=0 returns NULL
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0)); // NULL
    }

    #[test]
    fn test_fgets_symbolic_buf() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "buf", 64);
        drop(ctx);

        let result = NativeFgets.call(
            &mut state,
            &[sym, RustBV::concrete(10, 64), RustBV::concrete(0, 64)],
        );
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_fgetc_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeFgetc
            .call(
                &mut state,
                &[RustBV::concrete(0, 64)], // stream (stdin)
            )
            .unwrap();

        let val = result.unwrap();
        // Should be symbolic (can't get concrete value)
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_getchar_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeGetchar.call(&mut state, &[]).unwrap();
        let val = result.unwrap();
        // Should be symbolic
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_getc_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        let result = NativeGetc
            .call(&mut state, &[RustBV::concrete(0, 64)])
            .unwrap();
        let val = result.unwrap();
        assert!(val.as_u64().is_none());
    }
}
