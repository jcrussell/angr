//! Native strtol/strtoul/atoi/atol implementations.
//!
//! Handles concrete string-to-integer conversion with whitespace skip,
//! sign handling, and base conversion. Symbolic strings fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{extract_concrete_arg, NativeSimProcedure, ProcedureError};

const MAX_DIGITS: usize = 64;

/// Read a concrete null-terminated string from state memory (max_len bytes).
fn read_concrete_string(state: &mut RustSimState, addr: u64, max_len: usize) -> Result<Vec<u8>, ProcedureError> {
    let mut result = Vec::new();
    for i in 0..max_len as u64 {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        let byte = extract_concrete_arg(&byte_val, &format!("memory byte at 0x{:x}", addr.wrapping_add(i)))? as u8;
        if byte == 0 { break; }
        result.push(byte);
    }
    Ok(result)
}

/// Core strtol logic: parse string with given base, return (value, bytes_consumed).
fn parse_strtol(s: &[u8], base_arg: i64) -> Result<(i64, usize), ProcedureError> {
    let mut idx = 0;

    // Skip whitespace
    while idx < s.len() && (s[idx] as char).is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= s.len() {
        return Ok((0, idx));
    }

    // Handle sign
    let negative = s[idx] == b'-';
    if s[idx] == b'-' || s[idx] == b'+' {
        idx += 1;
    }
    if idx >= s.len() {
        return Ok((0, idx));
    }

    // Determine base
    let base = if base_arg == 0 {
        if s[idx] == b'0' {
            if idx + 1 < s.len() && (s[idx + 1] == b'x' || s[idx + 1] == b'X') {
                idx += 2;
                16
            } else {
                idx += 1;
                8
            }
        } else {
            10
        }
    } else if base_arg == 16 && s[idx] == b'0' && idx + 1 < s.len() && (s[idx + 1] == b'x' || s[idx + 1] == b'X') {
        idx += 2;
        16
    } else {
        base_arg as u32
    };

    if !(2..=36).contains(&base) {
        return Ok((0, idx));
    }

    // Parse digits
    let mut value: i64 = 0;
    let mut found_digit = false;
    while idx < s.len() {
        let digit = match s[idx] {
            b'0'..=b'9' => (s[idx] - b'0') as u32,
            b'a'..=b'z' => (s[idx] - b'a' + 10) as u32,
            b'A'..=b'Z' => (s[idx] - b'A' + 10) as u32,
            _ => break,
        };
        if digit >= base {
            break;
        }
        found_digit = true;
        value = value.wrapping_mul(base as i64).wrapping_add(digit as i64);
        idx += 1;
    }

    if !found_digit {
        return Ok((0, idx));
    }

    if negative {
        value = value.wrapping_neg();
    }

    Ok((value, idx))
}

/// strtol: string to long integer.
pub struct NativeStrtol;

impl NativeSimProcedure for NativeStrtol {
    fn name(&self) -> &'static str { "strtol" }
    fn num_args(&self) -> usize { 3 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = extract_concrete_arg(&args[0], "nptr")?;
        let endptr = extract_concrete_arg(&args[1], "endptr")?;
        let base = extract_concrete_arg(&args[2], "base")? as i64;

        let s = read_concrete_string(state, addr, MAX_DIGITS)?;
        let (value, consumed) = parse_strtol(&s, base)?;

        // Write endptr if non-NULL
        if endptr != 0 {
            let end_addr = addr.wrapping_add(consumed as u64);
            let ptr_bits = state.arch().bits();
            state.memory_store(endptr, RustBV::concrete(end_addr as u128, ptr_bits))
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(value as u128, bits)))
    }
}

/// strtoul: string to unsigned long.
pub struct NativeStrtoul;

impl NativeSimProcedure for NativeStrtoul {
    fn name(&self) -> &'static str { "strtoul" }
    fn num_args(&self) -> usize { 3 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Same implementation — unsigned semantics handled by bit interpretation
        NativeStrtol.call(state, args)
    }
}

/// atoi: string to integer (base 10).
pub struct NativeAtoi;

impl NativeSimProcedure for NativeAtoi {
    fn name(&self) -> &'static str { "atoi" }
    fn num_args(&self) -> usize { 1 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = extract_concrete_arg(&args[0], "nptr")?;

        let s = read_concrete_string(state, addr, MAX_DIGITS)?;
        let (value, _) = parse_strtol(&s, 10)?;

        let bits = state.arch().bits();
        // atoi returns int (32-bit), but we return arch-width for ABI compat
        Ok(Some(RustBV::concrete(value as u128, bits)))
    }
}

/// atol: string to long (base 10).
pub struct NativeAtol;

impl NativeSimProcedure for NativeAtol {
    fn name(&self) -> &'static str { "atol" }
    fn num_args(&self) -> usize { 1 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = extract_concrete_arg(&args[0], "nptr")?;

        let s = read_concrete_string(state, addr, MAX_DIGITS)?;
        let (value, _) = parse_strtol(&s, 10)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(value as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_string(state: &mut RustSimState, addr: u64, s: &[u8]) {
        let mut data = s.to_vec();
        data.push(0); // null terminate
        state.map_memory_data(addr, &data, Permission::RWX);
    }

    #[test]
    fn test_parse_basic() {
        assert_eq!(parse_strtol(b"123", 10).unwrap(), (123, 3));
        assert_eq!(parse_strtol(b"-42", 10).unwrap(), (-42, 3));
        assert_eq!(parse_strtol(b"  +99", 10).unwrap(), (99, 5));
    }

    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_strtol(b"0xff", 0).unwrap(), (255, 4));
        assert_eq!(parse_strtol(b"0xFF", 16).unwrap(), (255, 4));
        assert_eq!(parse_strtol(b"ff", 16).unwrap(), (255, 2));
    }

    #[test]
    fn test_parse_octal() {
        assert_eq!(parse_strtol(b"077", 0).unwrap(), (63, 3));
        assert_eq!(parse_strtol(b"77", 8).unwrap(), (63, 2));
    }

    #[test]
    fn test_atoi() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"42");

        let p = NativeAtoi;
        let result = p.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(42));
    }

    #[test]
    fn test_atoi_negative() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"-123");

        let p = NativeAtoi;
        let result = p.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap().unwrap();
        // -123 as u64
        assert_eq!(result.as_u64(), Some((-123i64) as u64));
    }

    #[test]
    fn test_atoi_whitespace() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"  \t 56");

        let p = NativeAtoi;
        let result = p.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(56));
    }
}
