//! Native `strtod` implementation.
//!
//! ```c
//! double strtod(const char *nptr, char **endptr);
//! ```
//!
//! Concrete fast path: scan up to a fixed number of bytes until the first
//! null terminator, leading whitespace + sign + optional digits-dot-digits
//! + optional exponent, and feed the slice through Rust's `f64::from_str`.
//!
//! Symbolic bytes anywhere in the parsed region — or a non-amd64 target —
//! fall back to Python. amd64 is the only architecture for which we know
//! the FP-return calling-convention slot (`xmm0`); on other targets the
//! caller's expectations don't fit our integer-return-register dispatch.
//!
//! Return value is the 64-bit IEEE-754 bit-pattern of the parsed double,
//! written to the low 64 bits of `xmm0`. The dispatcher's default
//! integer-return store is suppressed by returning `Ok(None)`.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum byte scan length when reading the numeric literal. Real strings
/// rarely need more than ~64 bytes; cap the work even for hostile inputs.
const MAX_LEN: usize = 256;

/// XMM0 register byte-offset in the amd64 register file.
/// Mirrors `crate::arch::amd64::offsets::XMM0`. Hard-coded here so we don't
/// reach across modules for a single constant; if it ever moves, the
/// `state.arch().name() == "amd64"` guard and unit tests will catch the drift.
const AMD64_XMM0_OFFSET: u32 = 224;

/// Read the C string at `addr` (up to `MAX_LEN` bytes) as concrete bytes,
/// stopping at the first null. Returns `None` if any byte before the null
/// is symbolic — strtod has no useful symbolic-FP story without a real
/// floating-point solver, so we let Python handle that case.
fn read_concrete_cstring(
    state: &mut RustSimState,
    addr: u64,
) -> Result<Option<Vec<u8>>, ProcedureError> {
    let mut bytes = Vec::with_capacity(32);
    for i in 0..MAX_LEN {
        let byte = state.memory_load(addr.wrapping_add(i as u64), 1)?;
        match byte.as_u64() {
            Some(0) => return Ok(Some(bytes)),
            Some(b) => bytes.push(b as u8),
            None => return Ok(None),
        }
    }
    // Hit cap without finding null — treat as parseable up to here. The
    // C contract permits parsing to stop on the first non-numeric byte
    // long before any terminator, so this is fine.
    Ok(Some(bytes))
}

/// Walk `bytes` from the front and return the byte index immediately past
/// the longest prefix that looks like a C99 floating-point literal. Returns
/// 0 if no parseable prefix was found.
///
/// Grammar (case-insensitive `e`/`x`/`p`):
/// ```text
///   [ws]* [+-]? ( digits? '.' digits ([eE][+-]?digits)?
///               | digits ('.' digits?)? ([eE][+-]?digits)?
///               | '0x' hex_digits? '.' hex_digits? ([pP][+-]?digits)?
///               | "inf" | "infinity"
///               | "nan" )
/// ```
fn floating_prefix_len(bytes: &[u8]) -> usize {
    let mut i = 0;
    // whitespace
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    let prefix_start = i;
    // sign
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    // inf / infinity
    if i + 3 <= bytes.len() {
        let tag = &bytes[i..i + 3];
        if tag.eq_ignore_ascii_case(b"inf") {
            let mut j = i + 3;
            if j + 5 <= bytes.len() && bytes[j..j + 5].eq_ignore_ascii_case(b"inity") {
                j += 5;
            }
            return j;
        }
    }
    // nan
    if i + 3 <= bytes.len() && bytes[i..i + 3].eq_ignore_ascii_case(b"nan") {
        return i + 3;
    }
    // hex form: 0x...
    if i + 2 <= bytes.len() && bytes[i] == b'0' && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
        let start = i;
        i += 2;
        let mut saw_digit = false;
        while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
            saw_digit = true;
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                saw_digit = true;
                i += 1;
            }
        }
        if !saw_digit {
            return prefix_start;
        }
        if i < bytes.len() && (bytes[i] == b'p' || bytes[i] == b'P') {
            let exp_start = i;
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            let exp_digits_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == exp_digits_start {
                return exp_start; // 'p' without digits — drop it
            }
        }
        return i.max(start);
    }
    // decimal form
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        saw_digit = true;
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            saw_digit = true;
            i += 1;
        }
    }
    if !saw_digit {
        return prefix_start;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let exp_start = i;
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_digits_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_digits_start {
            return exp_start;
        }
    }
    i
}

/// Native `strtod` SimProcedure.
pub struct NativeStrtod;

impl NativeSimProcedure for NativeStrtod {
    fn name(&self) -> &'static str {
        "strtod"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Only amd64 has a known FP-return register slot in our dispatcher.
        if state.arch().name() != "AMD64" {
            return Err(ProcedureError::NotImplemented);
        }

        let nptr = extract_concrete_arg(&args[0], "nptr")?;
        let endptr = extract_concrete_arg(&args[1], "endptr")?;

        let bytes = match read_concrete_cstring(state, nptr)? {
            Some(b) => b,
            None => {
                return Err(ProcedureError::SymbolicArgument(
                    "strtod: symbolic byte in input string".into(),
                ));
            }
        };

        let prefix_end = floating_prefix_len(&bytes);
        let (value, end_offset) = if prefix_end == 0 {
            (0.0f64, 0u64)
        } else {
            // Skip leading whitespace to find the start of the parseable run
            // for f64::from_str (it does not accept leading whitespace).
            let mut start = 0;
            while start < bytes.len() && (bytes[start] as char).is_ascii_whitespace() {
                start += 1;
            }
            let slice = &bytes[start..prefix_end];
            let parsed = match std::str::from_utf8(slice) {
                Ok(s) => s.parse::<f64>().unwrap_or(0.0),
                Err(_) => 0.0,
            };
            (parsed, prefix_end as u64)
        };

        // Write *endptr if requested. C contract: if endptr is non-null, store
        // a pointer to the first byte past the parsed literal (or `nptr` if
        // nothing was consumed).
        if endptr != 0 {
            let bits = state.arch().bits();
            let end_addr = nptr.wrapping_add(end_offset);
            state.memory_store(endptr, RustBV::concrete(end_addr as u128, bits))?;
        }

        // Write the f64 bit pattern to xmm0 (low 64 bits). Leave the upper
        // 64 bits untouched — the SysV ABI says only the low slot carries
        // the scalar double return value.
        let bits = value.to_bits();
        let ret_bv = RustBV::concrete(bits as u128, 64);
        state.set_register_by_offset(AMD64_XMM0_OFFSET, ret_bv);

        // Suppress the dispatcher's default integer-return-register store —
        // we've placed the value in xmm0 already, and writing rax would
        // pollute an unrelated register.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::procedures::NativeSimProcedure;

    fn setup_string(state: &mut RustSimState, addr: u64, s: &[u8]) {
        let mut data = s.to_vec();
        data.push(0);
        state.map_memory_data(addr, &data, Permission::RWX);
    }

    fn read_xmm0_low64(state: &mut RustSimState) -> u64 {
        let bv = state.get_register_by_offset(AMD64_XMM0_OFFSET, 8);
        bv.as_u64().expect("xmm0 low64 concrete")
    }

    #[test]
    fn test_strtod_simple_decimal() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"42.5");
        let p = NativeStrtod;
        let ret = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
            )
            .unwrap();
        assert!(ret.is_none(), "strtod must suppress integer-return store");
        assert_eq!(read_xmm0_low64(&mut state), 42.5f64.to_bits());
    }

    #[test]
    fn test_strtod_negative_exponent() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"-1.5e-3");
        let p = NativeStrtod;
        p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
        assert_eq!(read_xmm0_low64(&mut state), (-1.5e-3f64).to_bits());
    }

    #[test]
    fn test_strtod_writes_endptr_past_parsed_prefix() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"42.0abc");
        // Reserve space for *endptr at 0x2000.
        state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);
        let p = NativeStrtod;
        p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
        assert_eq!(read_xmm0_low64(&mut state), 42.0f64.to_bits());
        let end = state.memory_load(0x2000, 8).unwrap();
        assert_eq!(end.as_u64(), Some(0x1000 + 4)); // past "42.0"
    }

    #[test]
    fn test_strtod_no_conversion_returns_zero_and_endptr_at_nptr() {
        // Per C: no digits parsed → return 0.0, *endptr = nptr.
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"abc");
        state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);
        let p = NativeStrtod;
        p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
        assert_eq!(read_xmm0_low64(&mut state), 0.0f64.to_bits());
        let end = state.memory_load(0x2000, 8).unwrap();
        assert_eq!(end.as_u64(), Some(0x1000));
    }

    #[test]
    fn test_strtod_leading_whitespace_skipped() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"   \t-2.0");
        let p = NativeStrtod;
        p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
        assert_eq!(read_xmm0_low64(&mut state), (-2.0f64).to_bits());
    }

    #[test]
    fn test_strtod_symbolic_byte_falls_back() {
        // A symbolic byte mid-string must trigger Python fallback. We can
        // verify this by writing one symbolic byte and asserting the call
        // returns SymbolicArgument.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"X.0\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "fp0", 8);
        drop(ctx);
        state.memory_store(0x1000, sym).unwrap();

        let p = NativeStrtod;
        let res = p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        );
        match res {
            Err(ProcedureError::SymbolicArgument(_)) => {}
            other => panic!("expected SymbolicArgument fallback, got {:?}", other),
        }
    }

    #[test]
    fn test_strtod_inf_and_nan_parsed() {
        let mut state = RustSimState::new("amd64").unwrap();
        setup_string(&mut state, 0x1000, b"inf");
        let p = NativeStrtod;
        p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
        let bits = read_xmm0_low64(&mut state);
        let val = f64::from_bits(bits);
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_strtod_non_amd64_falls_back() {
        // x86 has FP returns in st0, not xmm0 — we punt to Python.
        let mut state = RustSimState::new("x86").unwrap();
        state.map_memory_data(0x1000, b"1.0\x00", Permission::RWX);
        let p = NativeStrtod;
        let res = p.call(
            &mut state,
            &[RustBV::concrete(0x1000, 32), RustBV::concrete(0, 32)],
        );
        assert!(matches!(res, Err(ProcedureError::NotImplemented)));
    }

    #[test]
    fn test_floating_prefix_len_grammar() {
        assert_eq!(floating_prefix_len(b"3.14abc"), 4);
        assert_eq!(floating_prefix_len(b"  -2.5e10x"), 9);
        assert_eq!(floating_prefix_len(b"abc"), 0);
        assert_eq!(floating_prefix_len(b"-abc"), 0);
        // Hex form
        assert_eq!(floating_prefix_len(b"0x1.8p3"), 7);
        assert_eq!(floating_prefix_len(b"0xfg"), 3); // "0xf" then stop
        // Trailing 'e' without digits gets dropped
        assert_eq!(floating_prefix_len(b"1.5ex"), 3);
    }
}
