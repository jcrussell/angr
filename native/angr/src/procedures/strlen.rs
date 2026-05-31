//! Native strlen / strnlen implementations.
//!
//! strlen returns the length of a null-terminated string, not including the
//! null terminator. strnlen does the same but bounded by maxlen.
//!
//! # Behavior
//!
//! - Concrete address required (symbolic addresses fall back to Python).
//! - Concrete fast path scans byte-by-byte.
//! - When a symbolic byte is encountered, we build an ITE chain expressing
//!   the symbolic null position:
//!     result = ITE(b_i == 0, i, result_next)
//!   built right-to-left up to MAX_STRLEN (or a concrete null). The chain's
//!   initial right-most value is the upper bound (MAX_STRLEN for strlen,
//!   maxlen for strnlen). This is an approximation: if no null is ever
//!   reachable in MAX_STRLEN bytes the result will saturate at MAX_STRLEN.
//! - Maximum string length is 4096 bytes (configurable).

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Maximum string length before falling back to Python.
const MAX_STRLEN: usize = 4096;

/// Build the strlen ITE chain over collected (position, byte_8bit) loads.
/// Chain is built right-to-left so earlier null positions take precedence.
/// `default_len` is the value used past the scanned region (typically the
/// upper bound: MAX_STRLEN for strlen, maxlen for strnlen).
fn build_strlen_chain(
    bytes: &[(u64, RustBV)],
    arch_bits: u32,
    default_len: u64,
    ctx: &SymContext,
) -> RustBV {
    let zero_byte = RustBV::concrete(0u128, 8);
    let mut result = RustBV::concrete(default_len as u128, arch_bits);
    for (pos, byte) in bytes.iter().rev() {
        let is_null = byte.eq(&zero_byte, ctx);
        let len_bv = RustBV::concrete(*pos as u128, arch_bits);
        result = is_null.ite(&len_bv, &result, ctx);
    }
    result
}

/// Shared scan for strlen / strnlen. `max_scan` is the upper bound on the
/// number of positions inspected (MAX_STRLEN for strlen, min(maxlen, MAX) for
/// strnlen). Returns Ok(Some(length_bv)) on success.
fn scan_for_null(
    state: &mut RustSimState,
    addr: u64,
    max_scan: u64,
) -> Result<Option<RustBV>, ProcedureError> {
    let arch_bits = state.arch().bits();
    if max_scan == 0 {
        return Ok(Some(RustBV::concrete(0u128, arch_bits)));
    }

    let mut bytes: Vec<(u64, RustBV)> = Vec::new();
    let mut symbolic_seen = false;

    for i in 0..max_scan {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state
            .memory_load(byte_addr, 1)
            ?;

        if !symbolic_seen {
            if let Some(b) = byte_val.as_u64() {
                if (b as u8) == 0 {
                    return Ok(Some(RustBV::concrete(i as u128, arch_bits)));
                }
                continue;
            }
            symbolic_seen = true;
        }

        let stop_scan = byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false);
        bytes.push((i, byte_val));
        if stop_scan {
            break;
        }
    }

    if !symbolic_seen {
        // All concrete and no null seen in max_scan bytes. For strnlen
        // (max_scan < MAX_STRLEN) the natural answer is max_scan. For strlen
        // we ran past MAX_STRLEN — keep prior behavior of erroring.
        if max_scan >= MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(MAX_STRLEN));
        }
        return Ok(Some(RustBV::concrete(max_scan as u128, arch_bits)));
    }

    let ctx = state.solver().borrow();
    Ok(Some(build_strlen_chain(&bytes, arch_bits, max_scan, &ctx)))
}

crate::declare_proc! {
    /// Native strlen: `size_t strlen(const char *s)`.
    ///
    /// Returns the number of bytes before the first null byte.
    name = "strlen",
    struct = NativeStrlen,
    args = [addr: concrete],
    call |state| {
        scan_for_null(state, addr, MAX_STRLEN as u64)
    }
}

crate::declare_proc! {
    /// Native strnlen: `size_t strnlen(const char *s, size_t maxlen)`.
    ///
    /// Returns the lesser of the string length and maxlen.
    name = "strnlen",
    struct = NativeStrnlen,
    args = [s: concrete, maxlen: concrete],
    call |state| {
        if maxlen > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(maxlen as usize));
        }
        scan_for_null(state, s, maxlen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::procedures::NativeSimProcedure;

    #[test]
    fn test_strlen_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"test\x00", Permission::RWX);
        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(4));
    }

    #[test]
    fn test_strlen_empty() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00", Permission::RWX);
        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strlen_longer_string() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(11));
    }

    #[test]
    fn test_strnlen_within_limit() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        let proc = NativeStrnlen;
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(10, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(5));
    }

    #[test]
    fn test_strnlen_at_limit() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        let proc = NativeStrnlen;
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(5, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(5));
    }

    #[test]
    fn test_strnlen_zero_maxlen() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        let proc = NativeStrnlen;
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strnlen_empty_string() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
        let proc = NativeStrnlen;
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(10, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strlen_symbolic_addr() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
        drop(ctx);
        let proc = NativeStrlen;
        let result = proc.call(&mut state, &[sym_addr]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    /// Insert a fully-symbolic byte at `addr` (page must be pre-mapped).
    fn place_symbolic_byte(state: &mut RustSimState, addr: u64, name: &str) -> RustBV {
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, name, 8);
        drop(ctx);
        state.memory_store(addr, sym.clone()).unwrap();
        sym
    }

    #[test]
    fn test_strlen_symbolic_byte_returns_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Buffer: [a, ?, c, \0, ...]
        state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
        let _sym = place_symbolic_byte(&mut state, 0x1001, "b1");

        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none(), "expected symbolic length");
    }

    #[test]
    fn test_strlen_symbolic_byte_zero_solution() {
        // Constraining the symbolic byte to 0 should yield length 1.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "b1");

        let ctx = state.solver().borrow();
        let zero = RustBV::concrete(0u128, 8);
        let eq = sym.eq(&zero, &ctx);
        drop(ctx);

        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(1));
        assert_eq!(ctx.max(&result, false), Some(1));
    }

    #[test]
    fn test_strlen_symbolic_byte_nonzero_solution() {
        // Constraining the symbolic byte to 'b' should yield length 3.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "b1");

        let ctx = state.solver().borrow();
        let b = RustBV::concrete(b'b' as u128, 8);
        let eq = sym.eq(&b, &ctx);
        drop(ctx);

        let proc = NativeStrlen;
        let result = proc
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(3));
        assert_eq!(ctx.max(&result, false), Some(3));
    }

    #[test]
    fn test_strnlen_symbolic_byte_capped_at_maxlen() {
        // Buffer of all symbolic bytes; with maxlen=2 result is bounded by 2.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, &[1u8; 16], Permission::RWX);
        let _s0 = place_symbolic_byte(&mut state, 0x1000, "b0");
        let _s1 = place_symbolic_byte(&mut state, 0x1001, "b1");

        let proc = NativeStrnlen;
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(2, 64)],
            )
            .unwrap()
            .unwrap();
        let ctx = state.solver().borrow();
        // result must be in [0, 2].
        let min = ctx.min(&result, false).unwrap();
        let max = ctx.max(&result, false).unwrap();
        assert!(min <= 2);
        assert!(max <= 2);
    }
}
