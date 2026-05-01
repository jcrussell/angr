//! Native character classification functions (ctype.h).
//!
//! isdigit, isalpha, isspace, isalnum, isupper, islower, isxdigit, isprint,
//! tolower, toupper.
//!
//! Each takes a single int argument and returns 0 or non-zero.
//! Symbolic arguments fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{extract_concrete_arg, NativeSimProcedure, ProcedureError};

fn get_concrete_arg(args: &[RustBV]) -> Result<u8, ProcedureError> {
    Ok(extract_concrete_arg(&args[0], "c")? as u8)
}

fn bool_result(state: &RustSimState, v: bool) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    Ok(Some(RustBV::concrete(if v { 1 } else { 0 }, bits)))
}

macro_rules! ctype_proc {
    ($name:ident, $func_name:expr, $check:expr) => {
        pub struct $name;

        impl NativeSimProcedure for $name {
            fn name(&self) -> &'static str { $func_name }
            fn num_args(&self) -> usize { 1 }

            fn call(
                &self,
                state: &mut RustSimState,
                args: &[RustBV],
            ) -> Result<Option<RustBV>, ProcedureError> {
                let c = get_concrete_arg(args)?;
                let result: fn(u8) -> bool = $check;
                bool_result(state, result(c))
            }
        }
    };
}

ctype_proc!(NativeIsDigit, "isdigit", |c: u8| c.is_ascii_digit());
ctype_proc!(NativeIsAlpha, "isalpha", |c: u8| c.is_ascii_alphabetic());
ctype_proc!(NativeIsSpace, "isspace", |c: u8| c.is_ascii_whitespace());
ctype_proc!(NativeIsAlnum, "isalnum", |c: u8| c.is_ascii_alphanumeric());
ctype_proc!(NativeIsUpper, "isupper", |c: u8| c.is_ascii_uppercase());
ctype_proc!(NativeIsLower, "islower", |c: u8| c.is_ascii_lowercase());
ctype_proc!(NativeIsXdigit, "isxdigit", |c: u8| c.is_ascii_hexdigit());
ctype_proc!(NativeIsPrint, "isprint", |c: u8| (0x20..=0x7e).contains(&c));

/// tolower: convert uppercase to lowercase.
pub struct NativeToLower;

impl NativeSimProcedure for NativeToLower {
    fn name(&self) -> &'static str { "tolower" }
    fn num_args(&self) -> usize { 1 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let c = get_concrete_arg(args)?;
        let result = if c.is_ascii_uppercase() { c + 32 } else { c };
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(result as u128, bits)))
    }
}

/// toupper: convert lowercase to uppercase.
pub struct NativeToUpper;

impl NativeSimProcedure for NativeToUpper {
    fn name(&self) -> &'static str { "toupper" }
    fn num_args(&self) -> usize { 1 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let c = get_concrete_arg(args)?;
        let result = if c.is_ascii_lowercase() { c - 32 } else { c };
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(result as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> RustSimState {
        RustSimState::new("amd64").unwrap()
    }

    fn call_with(proc: &dyn NativeSimProcedure, state: &mut RustSimState, c: u8) -> u64 {
        let args = [RustBV::concrete(c as u128, 64)];
        proc.call(state, &args).unwrap().unwrap().as_u64().unwrap()
    }

    #[test]
    fn test_isdigit() {
        let mut s = make_state();
        let p = NativeIsDigit;
        assert_eq!(call_with(&p, &mut s, b'0'), 1);
        assert_eq!(call_with(&p, &mut s, b'9'), 1);
        assert_eq!(call_with(&p, &mut s, b'a'), 0);
    }

    #[test]
    fn test_isalpha() {
        let mut s = make_state();
        let p = NativeIsAlpha;
        assert_eq!(call_with(&p, &mut s, b'A'), 1);
        assert_eq!(call_with(&p, &mut s, b'z'), 1);
        assert_eq!(call_with(&p, &mut s, b'5'), 0);
    }

    #[test]
    fn test_isspace() {
        let mut s = make_state();
        let p = NativeIsSpace;
        assert_eq!(call_with(&p, &mut s, b' '), 1);
        assert_eq!(call_with(&p, &mut s, b'\t'), 1);
        assert_eq!(call_with(&p, &mut s, b'\n'), 1);
        assert_eq!(call_with(&p, &mut s, b'a'), 0);
    }

    #[test]
    fn test_tolower() {
        let mut s = make_state();
        let p = NativeToLower;
        assert_eq!(call_with(&p, &mut s, b'A'), b'a' as u64);
        assert_eq!(call_with(&p, &mut s, b'z'), b'z' as u64);
        assert_eq!(call_with(&p, &mut s, b'5'), b'5' as u64);
    }

    #[test]
    fn test_toupper() {
        let mut s = make_state();
        let p = NativeToUpper;
        assert_eq!(call_with(&p, &mut s, b'a'), b'A' as u64);
        assert_eq!(call_with(&p, &mut s, b'Z'), b'Z' as u64);
        assert_eq!(call_with(&p, &mut s, b'5'), b'5' as u64);
    }

    #[test]
    fn test_symbolic_arg_fallback() {
        let mut s = make_state();
        let ctx = s.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        drop(ctx);
        let p = NativeIsDigit;
        assert!(matches!(p.call(&mut s, &[sym]), Err(ProcedureError::SymbolicArgument(_))));
    }
}
