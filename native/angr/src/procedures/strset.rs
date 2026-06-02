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

use super::strings::scan_concrete_until_null;
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_SCAN: usize = 4096;
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

/// strpbrk: find first byte of `s` that is in `accept`.
///
/// ```c
/// char *strpbrk(const char *s, const char *accept);
/// ```
pub struct NativeStrpbrk;

impl NativeSimProcedure for NativeStrpbrk {
    fn name(&self) -> &'static str {
        "strpbrk"
    }
    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s_addr = extract_concrete_arg(&args[0], "s")?;
        let accept_addr = extract_concrete_arg(&args[1], "accept")?;
        let bits = state.arch().bits();

        let accept = build_byte_set(state, accept_addr, "accept")?;

        for i in 0..MAX_SCAN as u64 {
            let byte_addr = s_addr.wrapping_add(i);
            let byte_val = state.memory_load(byte_addr, 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{}]", i))? as u8;
            if byte == 0 {
                return Ok(Some(RustBV::concrete(0u128, bits)));
            }
            if accept[byte as usize] {
                return Ok(Some(RustBV::concrete(byte_addr as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_SCAN))
    }
}

/// strspn: length of prefix of `s` consisting entirely of bytes from `accept`.
///
/// ```c
/// size_t strspn(const char *s, const char *accept);
/// ```
pub struct NativeStrspn;

impl NativeSimProcedure for NativeStrspn {
    fn name(&self) -> &'static str {
        "strspn"
    }
    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s_addr = extract_concrete_arg(&args[0], "s")?;
        let accept_addr = extract_concrete_arg(&args[1], "accept")?;
        let bits = state.arch().bits();

        let accept = build_byte_set(state, accept_addr, "accept")?;

        for i in 0..MAX_SCAN as u64 {
            let byte_val = state.memory_load(s_addr.wrapping_add(i), 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{}]", i))? as u8;
            if byte == 0 || !accept[byte as usize] {
                return Ok(Some(RustBV::concrete(i as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_SCAN))
    }
}

/// strcspn: length of prefix of `s` consisting entirely of bytes NOT in `reject`.
///
/// ```c
/// size_t strcspn(const char *s, const char *reject);
/// ```
pub struct NativeStrcspn;

impl NativeSimProcedure for NativeStrcspn {
    fn name(&self) -> &'static str {
        "strcspn"
    }
    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s_addr = extract_concrete_arg(&args[0], "s")?;
        let reject_addr = extract_concrete_arg(&args[1], "reject")?;
        let bits = state.arch().bits();

        let reject = build_byte_set(state, reject_addr, "reject")?;

        for i in 0..MAX_SCAN as u64 {
            let byte_val = state.memory_load(s_addr.wrapping_add(i), 1)?;
            let byte = extract_concrete_arg(&byte_val, &format!("s[{}]", i))? as u8;
            if byte == 0 || reject[byte as usize] {
                return Ok(Some(RustBV::concrete(i as u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_SCAN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_state(s: &[u8], set: &[u8]) -> RustSimState {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, s, Permission::RWX);
        state.map_memory_data(0x2000, set, Permission::RWX);
        state
    }

    // --- strpbrk ---

    #[test]
    fn test_strpbrk_first_match() {
        let mut state = setup_state(b"hello world\x00", b"aeiou\x00");
        let r = NativeStrpbrk
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        // 'e' at 0x1001 is the first vowel.
        assert_eq!(r.as_u64(), Some(0x1001));
    }

    #[test]
    fn test_strpbrk_no_match() {
        let mut state = setup_state(b"xyz\x00", b"abc\x00");
        let r = NativeStrpbrk
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_strpbrk_empty_set() {
        // Empty accept set: nothing can match, return NULL.
        let mut state = setup_state(b"abc\x00", b"\x00");
        let r = NativeStrpbrk
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_strpbrk_empty_haystack() {
        let mut state = setup_state(b"\x00", b"abc\x00");
        let r = NativeStrpbrk
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_strpbrk_symbolic_addr_falls_back() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "s", 64);
        drop(ctx);
        let r = NativeStrpbrk.call(&mut state, &[sym, RustBV::concrete(0x2000, 64)]);
        assert!(matches!(r, Err(ProcedureError::SymbolicArgument(_))));
    }

    // --- strspn ---

    #[test]
    fn test_strspn_full_prefix() {
        let mut state = setup_state(b"aaabbb\x00", b"ab\x00");
        let r = NativeStrspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        // entire string consists of ab.
        assert_eq!(r.as_u64(), Some(6));
    }

    #[test]
    fn test_strspn_partial() {
        let mut state = setup_state(b"abc123\x00", b"abc\x00");
        let r = NativeStrspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(3));
    }

    #[test]
    fn test_strspn_zero() {
        let mut state = setup_state(b"xyz\x00", b"abc\x00");
        let r = NativeStrspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(0));
    }

    // --- strcspn ---

    #[test]
    fn test_strcspn_partial() {
        let mut state = setup_state(b"abc,def\x00", b",;\x00");
        let r = NativeStrcspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(3));
    }

    #[test]
    fn test_strcspn_no_reject_byte() {
        // No byte from reject in s → return strlen(s).
        let mut state = setup_state(b"hello\x00", b",;\x00");
        let r = NativeStrcspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(5));
    }

    #[test]
    fn test_strcspn_immediate_reject() {
        let mut state = setup_state(b",hello\x00", b",;\x00");
        let r = NativeStrcspn
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_strspn_symbolic_byte_falls_back() {
        // Symbolic byte in s → SymbolicArgument.
        let mut state = setup_state(b"ab\x00", b"ab\x00");
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "b1", 8);
        drop(ctx);
        state.memory_store(0x1001, sym).unwrap();
        let r = NativeStrspn.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        );
        assert!(matches!(r, Err(ProcedureError::SymbolicArgument(_))));
    }
}
