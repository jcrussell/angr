//! Shared helpers for null-terminated string scans across libc string procedures.
//!
//! The procedures `strlen`, `strcpy`, `strncpy`, `strdup`, `strcat`, `strncat`,
//! `strstr`, `strchr`, and friends all need to walk memory byte-by-byte until a
//! null terminator. Two flavors emerge:
//!
//! 1. **Concrete-only**: bail to Python on the first symbolic byte. Used by
//!    procedures that write to memory (strcpy/strdup/strcat) or do
//!    string-search work that can't be expressed as an ITE chain
//!    (strstr). Helpers: [`scan_concrete_until_null`], [`find_null_addr`].
//!
//! 2. **Symbolic-aware**: build an ITE chain over collected bytes so the
//!    result can be symbolic. Used by strlen (and could be used by strcmp).
//!    Helpers: [`scan_for_null_symbolic`], [`build_strlen_chain`].
//!
//! Centralizing these means null-boundary edge cases live in one place. Tests
//! for the shared helpers live in this file.
//!
//! Anchor for source-of-history: angr-arf5.

use super::{ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Concrete byte-by-byte scan up to and including the null terminator.
///
/// Returns the bytes read **up to but not including** the null terminator.
/// Errors:
/// - `SymbolicArgument` if any byte is symbolic.
/// - `MaxIterations(max)` if `max` bytes are scanned without finding a null.
///
/// `addr_label` is used to format symbolic-byte error messages.
pub fn scan_concrete_until_null(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<Vec<u8>, ProcedureError> {
    let mut buf = Vec::with_capacity(256.min(max));
    for i in 0..max as u64 {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)?;
        let byte = extract_concrete_arg(&byte_val, &format!("{}[{}]", addr_label, i))? as u8;
        if byte == 0 {
            return Ok(buf);
        }
        buf.push(byte);
    }
    Err(ProcedureError::MaxIterations(max))
}

/// Bounded concrete scan. Reads up to `max` bytes, stopping at the first
/// null terminator (the null itself is **not** included in the returned
/// vec). Hitting `max` without finding a null is **not** an error — the
/// boolean `null_found` distinguishes the two cases.
///
/// Used by procedures like `strncpy` / `strncat` where `max` is a
/// caller-supplied bound and exhausting it is the natural stop, not an
/// error.
pub fn scan_concrete_bounded(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<(Vec<u8>, bool), ProcedureError> {
    let mut buf = Vec::with_capacity(max);
    for i in 0..max as u64 {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)?;
        let byte = extract_concrete_arg(&byte_val, &format!("{}[{}]", addr_label, i))? as u8;
        if byte == 0 {
            return Ok((buf, true));
        }
        buf.push(byte);
    }
    Ok((buf, false))
}

/// Concrete scan that returns the address of the first null terminator.
///
/// Used by strcat/strncat which need the null position, not the bytes.
/// Errors identical to [`scan_concrete_until_null`].
pub fn find_null_addr(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<u64, ProcedureError> {
    for i in 0..max as u64 {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state.memory_load(byte_addr, 1)?;
        let byte =
            extract_concrete_arg(&byte_val, &format!("{} at 0x{:x}", addr_label, byte_addr))? as u8;
        if byte == 0 {
            return Ok(byte_addr);
        }
    }
    Err(ProcedureError::MaxIterations(max))
}

/// Result of [`scan_for_null_symbolic`].
pub enum ScanOutcome {
    /// All scanned bytes were concrete; null terminator found at this length
    /// (or `max` was reached without finding null — caller decides whether
    /// that's an error or natural saturation).
    AllConcrete { length: u64 },
    /// At least one byte was symbolic; the chain builder must be invoked.
    /// Each entry is `(position, byte_8bit)`.
    Symbolic { bytes: Vec<(u64, RustBV)> },
}

/// Scan up to `max` bytes, collecting (position, byte) pairs once a symbolic
/// byte is seen. Stops early when a concretely-null byte is found in
/// symbolic mode (positions past null cannot affect downstream ITE chains).
///
/// In all-concrete mode, returns [`ScanOutcome::AllConcrete`] with the
/// length found (or `max` if no null was hit).
pub fn scan_for_null_symbolic(
    state: &mut RustSimState,
    addr: u64,
    max: u64,
) -> Result<ScanOutcome, ProcedureError> {
    let mut bytes: Vec<(u64, RustBV)> = Vec::new();
    let mut symbolic_seen = false;

    for i in 0..max {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)?;

        if !symbolic_seen {
            if let Some(b) = byte_val.as_u64() {
                if (b as u8) == 0 {
                    return Ok(ScanOutcome::AllConcrete { length: i });
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

    if symbolic_seen {
        Ok(ScanOutcome::Symbolic { bytes })
    } else {
        Ok(ScanOutcome::AllConcrete { length: max })
    }
}

/// Build the strlen ITE chain over collected `(position, byte_8bit)` loads.
///
/// Chain is built right-to-left so earlier null positions take precedence:
/// `result = ITE(b_i == 0, i, result_next)`.
///
/// `default_len` is the value used past the scanned region (typically the
/// upper bound: `MAX_STRLEN` for strlen, `maxlen` for strnlen).
pub fn build_strlen_chain(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_scan_concrete_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00rest", Permission::RWX);
        let buf = scan_concrete_until_null(&mut state, 0x1000, 4096, "s").unwrap();
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn test_scan_concrete_empty() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
        let buf = scan_concrete_until_null(&mut state, 0x1000, 4096, "s").unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_scan_concrete_symbolic_byte() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"ab\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "b", 8);
        drop(ctx);
        state.memory_store(0x1001, sym).unwrap();
        let res = scan_concrete_until_null(&mut state, 0x1000, 4096, "s");
        assert!(matches!(res, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_scan_concrete_max_iters() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, &[b'a'; 8], Permission::RWX);
        let res = scan_concrete_until_null(&mut state, 0x1000, 4, "s");
        assert!(matches!(res, Err(ProcedureError::MaxIterations(4))));
    }

    #[test]
    fn test_scan_concrete_bounded_null_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hi\x00xyz", Permission::RWX);
        let (buf, found) = scan_concrete_bounded(&mut state, 0x1000, 10, "s").unwrap();
        assert_eq!(buf, b"hi");
        assert!(found);
    }

    #[test]
    fn test_scan_concrete_bounded_max_no_null() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abcdef", Permission::RWX);
        let (buf, found) = scan_concrete_bounded(&mut state, 0x1000, 4, "s").unwrap();
        assert_eq!(buf, b"abcd");
        assert!(!found);
    }

    #[test]
    fn test_find_null_addr_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        assert_eq!(
            find_null_addr(&mut state, 0x1000, 4096, "p").unwrap(),
            0x1003
        );
    }

    #[test]
    fn test_find_null_addr_at_start() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
        assert_eq!(
            find_null_addr(&mut state, 0x1000, 4096, "p").unwrap(),
            0x1000
        );
    }

    #[test]
    fn test_scan_for_null_symbolic_all_concrete() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        match scan_for_null_symbolic(&mut state, 0x1000, 4096).unwrap() {
            ScanOutcome::AllConcrete { length } => assert_eq!(length, 3),
            ScanOutcome::Symbolic { .. } => panic!("expected concrete"),
        }
    }

    #[test]
    fn test_scan_for_null_symbolic_with_sym_byte() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "b1", 8);
        drop(ctx);
        state.memory_store(0x1001, sym).unwrap();
        match scan_for_null_symbolic(&mut state, 0x1000, 4096).unwrap() {
            ScanOutcome::Symbolic { bytes } => {
                // Position 0 was concrete 'a' (skipped), positions 1..3 collected.
                assert!(bytes.iter().any(|(p, _)| *p == 1));
                // Concrete '\x00' at position 3 stops the scan.
                let last = bytes.last().unwrap();
                assert_eq!(last.0, 3);
            }
            ScanOutcome::AllConcrete { .. } => panic!("expected symbolic"),
        }
    }

    #[test]
    fn test_build_strlen_chain_concrete() {
        let state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        // Bytes: [(0, 'a'), (1, '\0')]. Chain should reduce to 1.
        let bytes = vec![
            (0u64, RustBV::concrete(b'a' as u128, 8)),
            (1u64, RustBV::concrete(0u128, 8)),
        ];
        let result = build_strlen_chain(&bytes, 64, 4096, &ctx);
        assert_eq!(result.as_u64(), Some(1));
    }
}
