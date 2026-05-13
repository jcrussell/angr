//! Native memcmp implementation.
//!
//! memcmp compares n bytes of two memory regions.
//!
//! # Behavior
//!
//! - Concrete addresses are required (symbolic addresses fall back to Python).
//! - Concrete `n` is required (symbolic n falls back).
//! - Concrete bytes scan with short-circuit on first mismatch (matches libc).
//! - Symbolic bytes produce a 32-bit ITE chain via `compare_bytes` shared
//!   with strcmp/strncmp (with stop_at_null=false).

use super::ProcedureError;
use super::strcmp::{MAX_STRCMP_LEN, compare_bytes};
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// Native memcmp: `int memcmp(const void *s1, const void *s2, size_t n)`.
    ///
    /// Returns < 0, 0, or > 0 per byte-wise comparison.
    name = "memcmp",
    struct = NativeMemcmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        if n == 0 {
            return Ok(Some(RustBV::zero(32)));
        }
        if n > MAX_STRCMP_LEN as u64 {
            return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
        }
        compare_bytes(state, s1, s2, n,
                      /*stop_at_null=*/false, /*case_insensitive=*/false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::procedures::NativeSimProcedure;
    use crate::state::RustSimState;

    #[test]
    fn test_memcmp_equal() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello", Permission::RWX);
        state.map_memory_data(0x2000, b"hello", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_memcmp_less_than() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03", Permission::RWX);
        state.map_memory_data(0x2000, b"\x01\x02\x04", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_memcmp_greater_than() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x04", Permission::RWX);
        state.map_memory_data(0x2000, b"\x01\x02\x03", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val > 0);
    }

    #[test]
    fn test_memcmp_zero_length() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc", Permission::RWX);
        state.map_memory_data(0x2000, b"xyz", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_memcmp_partial() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello1", Permission::RWX);
        state.map_memory_data(0x2000, b"hello2", Permission::RWX);

        let proc = NativeMemcmp;
        // Compare only first 5 bytes (should be equal)
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Compare 6 bytes (should differ)
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(6, 64),
                ],
            )
            .unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val != 0);
    }

    #[test]
    fn test_memcmp_with_nulls() {
        let mut state = RustSimState::new("amd64").unwrap();
        // memcmp does NOT stop at null bytes (unlike strcmp)
        state.map_memory_data(0x1000, b"ab\x00cd", Permission::RWX);
        state.map_memory_data(0x2000, b"ab\x00ce", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0); // 'd' < 'e'
    }

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
    fn test_memcmp_symbolic_byte_returns_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"\x01\x02\x03", Permission::RWX);
        state.map_memory_data(0x1000, b"\x01\x00\x03", Permission::RWX);
        let _sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.width(), 32);
        assert!(result.as_u64().is_none());
    }

    #[test]
    fn test_memcmp_symbolic_byte_solver_evaluation() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"\x01\x02\x03", Permission::RWX);
        state.map_memory_data(0x1000, b"\x01\x00\x03", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

        // Constrain sym == 4 -> 4 - 2 = 2.
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(4u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(2));
        assert_eq!(ctx.max(&result, false), Some(2));
    }

    #[test]
    fn test_memcmp_symbolic_byte_equal_solution() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x2000, b"\x01\x02\x03", Permission::RWX);
        state.map_memory_data(0x1000, b"\x01\x00\x03", Permission::RWX);
        let sym = place_symbolic_byte(&mut state, 0x1001, "s1_1");

        // Constrain sym == 2 -> result == 0.
        let ctx = state.solver().borrow();
        let target = RustBV::concrete(2u128, 8);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);

        let proc = NativeMemcmp;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64),
                ],
            )
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0));
        assert_eq!(ctx.max(&result, false), Some(0));
    }
}
