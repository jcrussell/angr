//! Native strcmp/strncmp/strcasecmp implementations.
//!
//! strcmp compares two null-terminated strings lexicographically.
//! strncmp compares at most n characters.
//! strcasecmp compares case-insensitively.
//!
//! # Behavior
//!
//! - Concrete addresses are required (symbolic addresses fall back to Python).
//! - Concrete fast path scans byte-by-byte and short-circuits on the first
//!   mismatch or null terminator (mirroring libc behavior).
//! - When the scan encounters a symbolic byte (or for strncmp when n is
//!   symbolic — currently unsupported), we switch to building a 32-bit ITE
//!   chain expressing the byte-wise diff:
//!     result = ITE(c1_i != c2_i, sext(c1_i) - sext(c2_i),
//!                  ITE(c1_i == 0, 0, result_next))   -- strcmp/strncmp
//!     result = ITE(c1_i != c2_i, sext(c1_i) - sext(c2_i), result_next) -- memcmp
//! - Maximum compare length is 4096 bytes (configurable).

use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};
use super::ProcedureError;

/// Maximum string length before falling back to Python.
pub(super) const MAX_STRCMP_LEN: usize = 4096;

/// Per-position case-folding option used by strcasecmp.
fn case_fold_byte(byte: &RustBV, ctx: &SymContext) -> RustBV {
    // result = ITE(byte in [A, Z], byte + 32, byte) over 8 bits
    let lo = RustBV::concrete(b'A' as u128, 8);
    let hi = RustBV::concrete(b'Z' as u128, 8);
    let in_range = byte.uge(&lo, ctx).and(&byte.ule(&hi, ctx), ctx);
    let delta = RustBV::concrete(32u128, 8);
    let lowered = byte.add(&delta, ctx);
    in_range.ite(&lowered, byte, ctx)
}

/// Build the ITE chain from a list of (c1_i, c2_i) pairs (each 8-bit BVs).
fn build_diff_chain(
    pairs: &[(RustBV, RustBV)],
    stop_at_null: bool,
    case_insensitive: bool,
    ctx: &SymContext,
) -> RustBV {
    let zero32 = RustBV::concrete(0u128, 32);
    let zero8 = RustBV::concrete(0u128, 8);
    let mut result = zero32.clone();
    for (c1, c2) in pairs.iter().rev() {
        let (lhs, rhs) = if case_insensitive {
            (case_fold_byte(c1, ctx), case_fold_byte(c2, ctx))
        } else {
            (c1.clone(), c2.clone())
        };
        let diff = lhs.zero_extend(32, ctx).sub(&rhs.zero_extend(32, ctx), ctx);
        let mismatch = lhs.ne(&rhs, ctx);
        if stop_at_null {
            // result = ITE(c1 != c2, diff, ITE(c1 == 0, 0, result_next))
            let null_cond = c1.eq(&zero8, ctx);
            let inner = null_cond.ite(&zero32, &result, ctx);
            result = mismatch.ite(&diff, &inner, ctx);
        } else {
            // memcmp: result = ITE(c1 != c2, diff, result_next)
            result = mismatch.ite(&diff, &result, ctx);
        }
    }
    result
}

/// Shared scan body for strcmp / strncmp / strcasecmp / memcmp.
pub(super) fn compare_bytes(
    state: &mut RustSimState,
    s1_addr: u64,
    s2_addr: u64,
    max_len: u64,
    stop_at_null: bool,
    case_insensitive: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    if max_len == 0 {
        return Ok(Some(RustBV::zero(32)));
    }
    if max_len > MAX_STRCMP_LEN as u64 {
        return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
    }

    let mut pairs: Vec<(RustBV, RustBV)> = Vec::new();
    let mut symbolic_seen = false;

    for i in 0..max_len {
        let c1_addr = s1_addr.wrapping_add(i);
        let c2_addr = s2_addr.wrapping_add(i);
        let c1_val = state.memory_load(c1_addr, 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        let c2_val = state.memory_load(c2_addr, 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

        if !symbolic_seen {
            match (c1_val.as_u64(), c2_val.as_u64()) {
                (Some(b1), Some(b2)) => {
                    let mut a = b1 as u8;
                    let mut b = b2 as u8;
                    if case_insensitive {
                        if (b'A'..=b'Z').contains(&a) { a += 32; }
                        if (b'A'..=b'Z').contains(&b) { b += 32; }
                    }
                    if a != b {
                        let diff = (a as i32) - (b as i32);
                        return Ok(Some(RustBV::concrete(diff as u128, 32)));
                    }
                    if stop_at_null && a == 0 {
                        return Ok(Some(RustBV::zero(32)));
                    }
                    continue;
                }
                _ => {
                    symbolic_seen = true;
                }
            }
        }

        // Symbolic-mode collection. Stop scanning when c1 is concretely null
        // (positions past the null cannot affect the result for strcmp).
        let stop_scan = stop_at_null
            && c1_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false);
        pairs.push((c1_val, c2_val));
        if stop_scan {
            break;
        }
    }

    if !symbolic_seen {
        // Walked through max_len with all-concrete bytes and no
        // mismatch / null hit. For strcmp/strncmp this means we ran out
        // of room — error out (matches the prior MaxIterations behavior).
        // For memcmp, equal-up-to-limit is the natural "0" return.
        if stop_at_null && max_len >= MAX_STRCMP_LEN as u64 {
            return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
        }
        return Ok(Some(RustBV::zero(32)));
    }

    let ctx = state.solver().borrow();
    Ok(Some(build_diff_chain(&pairs, stop_at_null, case_insensitive, &ctx)))
}

crate::declare_proc! {
    /// Native strcmp: `int strcmp(const char *s1, const char *s2)`.
    ///
    /// Returns < 0, 0, or > 0 per lexicographic comparison.
    name = "strcmp",
    struct = NativeStrcmp,
    args = [s1: concrete, s2: concrete],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64,
                      /*stop_at_null=*/true, /*case_insensitive=*/false)
    }
}

crate::declare_proc! {
    /// Native strncmp: `int strncmp(const char *s1, const char *s2, size_t n)`.
    ///
    /// Like strcmp, but compares at most `n` characters.
    name = "strncmp",
    struct = NativeStrncmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        let max_len = n.min(MAX_STRCMP_LEN as u64);
        compare_bytes(state, s1, s2, max_len,
                      /*stop_at_null=*/true, /*case_insensitive=*/false)
    }
}

crate::declare_proc! {
    /// Native strcasecmp (case-insensitive strcmp).
    name = "strcasecmp",
    struct = NativeStrcasecmp,
    args = [s1: concrete, s2: concrete],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64,
                      /*stop_at_null=*/true, /*case_insensitive=*/true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::procedures::NativeSimProcedure;

    #[test]
    fn test_strcmp_equal() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strcmp_less_than() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"abd\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // 'c' - 'd' = -1
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_strcmp_greater_than() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"abd\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"abc\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // 'd' - 'c' = 1
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val > 0);
    }

    #[test]
    fn test_strcmp_prefix() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello world\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // '\0' - ' ' < 0
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_strncmp_limit() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello1\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello2\x00", Permission::RWX);

        let proc = NativeStrncmp;

        // Compare only first 5 chars (should be equal)
        let result = proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5, 64),
            ],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Compare 6 chars (should differ)
        let result = proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(6, 64),
            ],
        ).unwrap();

        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val != 0);
    }

    #[test]
    fn test_strcasecmp() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"Hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

        let proc = NativeStrcasecmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    // ---------- Symbolic-byte tests ----------

    /// Insert a fully-symbolic byte at `addr`. The page must already be
    /// mapped; this overwrites the byte without disturbing the rest.
    fn place_symbolic_byte(state: &mut RustSimState, addr: u64, name: &str) -> RustBV {
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, name, 8);
        drop(ctx);
        state.memory_store(addr, sym.clone()).unwrap();
        sym
    }

    #[test]
    fn test_strcmp_symbolic_byte_returns_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Concrete s2 = "ab\0", s1 = [a, ?, \0]. Symbolic byte at offset 1.
        state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
        state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
        let _sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap().unwrap();
        assert_eq!(result.width(), 32);
        assert!(result.as_u64().is_none(), "expected symbolic result");
    }

    #[test]
    fn test_strcmp_symbolic_byte_solver_evaluation_equal() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
        state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");
        // Constrain symbolic byte to 'b' so strings are equal -> 0.
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(b'b' as u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap().unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0));
        assert_eq!(ctx.max(&result, false), Some(0));
    }

    #[test]
    fn test_strcmp_symbolic_byte_solver_evaluation_mismatch() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"ab\x00", Permission::RWX);
        state.map_memory_data(0x1000, b"a\x00\x00", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");
        // Constrain symbolic byte to 'c' so 'c' - 'b' = 1.
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(b'c' as u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap().unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        // The 32-bit result equals 1 (signed).
        assert_eq!(ctx.min(&result, false), Some(1));
        assert_eq!(ctx.max(&result, false), Some(1));
    }

    #[test]
    fn test_strncmp_symbolic_byte_within_limit_equal() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"abx", Permission::RWX);
        state.map_memory_data(0x1000, b"a\x00x", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

        // Force sym == 'b'; with n=3, expecting 0 (since byte 2 is 'x' both).
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(b'b' as u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let proc = NativeStrncmp;
        let result = proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        ).unwrap().unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0));
        assert_eq!(ctx.max(&result, false), Some(0));
    }
}
