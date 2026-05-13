//! Native strchr and memchr implementations.
//!
//! strchr: find first occurrence of a character in a null-terminated string.
//! memchr: find first occurrence of a byte in a memory region.
//!
//! Both procedures support symbolic target bytes and symbolic memory bytes.
//! Concrete inputs take a fast scan path; if the target is symbolic OR a
//! loaded memory byte is symbolic, we switch to building an ITE chain over
//! up to MAX_SCAN positions: result = ITE(byte_i == target, addr+i, ...).
//! Concrete addresses are still required (symbolic addresses fall back to
//! Python via SymbolicArgument).

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

const MAX_SCAN: usize = 4096;

/// Build the symbolic ITE chain for strchr/memchr starting at `start_i`.
///
/// `byte_loads`: list of (byte_addr, byte_val_8bit) already collected. The
/// chain is built backward from the last load so that earlier matches take
/// precedence over later ones (per C strchr/memchr semantics).
///
/// `stop_at_null` controls the strchr-only ITE arm: if a byte is the null
/// terminator and didn't match the target, the result is NULL and no
/// further (later) match is considered.
fn build_ite_chain(
    byte_loads: &[(u64, RustBV)],
    target_byte: &RustBV,
    arch_bits: u32,
    stop_at_null: bool,
    ctx: &SymContext,
) -> RustBV {
    let null_addr = RustBV::concrete(0u128, arch_bits);
    let zero_byte = RustBV::concrete(0u128, 8);
    let mut result = null_addr.clone();
    for (byte_addr, byte_val) in byte_loads.iter().rev() {
        let addr_bv = RustBV::concrete(*byte_addr as u128, arch_bits);
        let match_cond = byte_val.eq(target_byte, ctx);
        if stop_at_null {
            // result = ITE(byte == target, addr, ITE(byte == 0, NULL, result_next))
            let null_cond = byte_val.eq(&zero_byte, ctx);
            let inner = null_cond.ite(&null_addr, &result, ctx);
            result = match_cond.ite(&addr_bv, &inner, ctx);
        } else {
            // result = ITE(byte == target, addr, result_next)
            result = match_cond.ite(&addr_bv, &result, ctx);
        }
    }
    result
}

/// Shared symbolic-aware scan body for strchr/memchr.
///
/// `max_scan` is the upper bound on positions to inspect (MAX_SCAN for strchr,
/// min(n, MAX_SCAN) for memchr).
/// `stop_at_null` is true for strchr (stops at concrete null terminators).
fn scan_for_byte(
    state: &mut RustSimState,
    addr: u64,
    target_arg: &RustBV,
    max_scan: u64,
    stop_at_null: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let arch_bits = state.arch().bits();

    // Concrete fast path: target is concrete. Walk byte by byte with the
    // existing short-circuiting semantics. If a symbolic byte is encountered
    // mid-scan we hand the rest off to the ITE-chain path.
    if let Some(target) = target_arg.as_u64() {
        let target_byte = target as u8;
        let mut byte_loads: Vec<(u64, RustBV)> = Vec::new();
        let mut symbolic_seen = false;
        for i in 0..max_scan {
            let byte_addr = addr.wrapping_add(i);
            let byte_val = state
                .memory_load(byte_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            if !symbolic_seen {
                if let Some(b) = byte_val.as_u64() {
                    let byte = b as u8;
                    if byte == target_byte {
                        return Ok(Some(RustBV::concrete(byte_addr as u128, arch_bits)));
                    }
                    if stop_at_null && byte == 0 {
                        // Null terminator hit before target — strchr returns NULL.
                        // (Note: target == 0 is captured by the byte == target_byte
                        // branch above.)
                        return Ok(Some(RustBV::concrete(0u128, arch_bits)));
                    }
                    // Concrete non-match — no contribution to result, skip.
                    continue;
                }
                symbolic_seen = true;
            }
            // Symbolic-mode: collect the load. For strchr, stop scanning once
            // we hit a concrete null since later positions are unreachable.
            let stop_scan =
                stop_at_null && byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false);
            byte_loads.push((byte_addr, byte_val));
            if stop_scan {
                break;
            }
        }

        if !symbolic_seen {
            // All concrete, scanned to end without match.
            // strchr/memchr both return NULL if not found.
            // (For strchr this only happens if the buffer has no null in MAX_SCAN.)
            if stop_at_null {
                return Err(ProcedureError::MaxIterations(max_scan as usize));
            }
            return Ok(Some(RustBV::concrete(0u128, arch_bits)));
        }

        let ctx = state.solver().borrow();
        let target_byte_bv = RustBV::concrete(target_byte as u128, 8);
        let chain = build_ite_chain(&byte_loads, &target_byte_bv, arch_bits, stop_at_null, &ctx);
        return Ok(Some(chain));
    }

    // Symbolic target: collect bytes up to max_scan (or first concrete null
    // for strchr) and build a full ITE chain.
    let mut byte_loads: Vec<(u64, RustBV)> = Vec::new();
    for i in 0..max_scan {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state
            .memory_load(byte_addr, 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        let stop_scan = stop_at_null && byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false);
        byte_loads.push((byte_addr, byte_val));
        if stop_scan {
            break;
        }
    }

    let ctx = state.solver().borrow();
    let target_byte_bv = target_arg.extract(7, 0, &ctx);
    let chain = build_ite_chain(&byte_loads, &target_byte_bv, arch_bits, stop_at_null, &ctx);
    Ok(Some(chain))
}

/// strchr: find character in string.
///
/// ```c
/// char *strchr(const char *s, int c);
/// ```
///
/// Returns pointer to first occurrence of c in s, or NULL if not found.
pub struct NativeStrchr;

impl NativeSimProcedure for NativeStrchr {
    fn name(&self) -> &'static str {
        "strchr"
    }
    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = extract_concrete_arg(&args[0], "s")?;
        scan_for_byte(
            state,
            addr,
            &args[1],
            MAX_SCAN as u64,
            /*stop_at_null=*/ true,
        )
    }
}

/// memchr: find byte in memory region.
///
/// ```c
/// void *memchr(const void *s, int c, size_t n);
/// ```
///
/// Returns pointer to first occurrence of c in the first n bytes of s, or NULL.
pub struct NativeMemchr;

impl NativeSimProcedure for NativeMemchr {
    fn name(&self) -> &'static str {
        "memchr"
    }
    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = extract_concrete_arg(&args[0], "s")?;
        // n must be concrete to bound the scan. Symbolic n could be handled by
        // taking the maximum solution, but that would require a solver call we
        // currently push onto the Python fallback path.
        let n = extract_concrete_arg(&args[2], "n")?;
        let scan_len = n.min(MAX_SCAN as u64);
        scan_for_byte(
            state, addr, &args[1], scan_len, /*stop_at_null=*/ false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strchr_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let p = NativeStrchr;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(b'l' as u128, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1002));
    }

    #[test]
    fn test_strchr_not_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let p = NativeStrchr;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(b'z' as u128, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0)); // NULL
    }

    #[test]
    fn test_strchr_finds_null() {
        // C standard: strchr(s, '\0') returns pointer to the null terminator.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);
        let p = NativeStrchr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0u128, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1002));
    }

    #[test]
    fn test_memchr_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

        let p = NativeMemchr;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(3, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1002));
    }

    #[test]
    fn test_memchr_not_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

        let p = NativeMemchr;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0xFF, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0)); // NULL
    }

    #[test]
    fn test_strchr_symbolic_target_returns_symbolic() {
        // Symbolic target should produce a symbolic result, not a fallback error.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        drop(ctx);
        let p = NativeStrchr;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
            .unwrap()
            .unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none(), "expected symbolic, got concrete");
    }

    #[test]
    fn test_strchr_symbolic_target_solver_evaluation() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        // Constrain c == 'b' so the only match is at 0x1001.
        let target = RustBV::concrete(b'b' as u128, 64);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);
        let p = NativeStrchr;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0x1001));
        assert_eq!(ctx.max(&result, false), Some(0x1001));
    }

    #[test]
    fn test_strchr_symbolic_target_null_match() {
        // Constraining target to 0 should yield the address of the null byte.
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        let zero = RustBV::concrete(0u128, 64);
        let eq = sym.eq(&zero, &ctx);
        drop(ctx);
        let p = NativeStrchr;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0x1003));
        assert_eq!(ctx.max(&result, false), Some(0x1003));
    }

    #[test]
    fn test_strchr_symbolic_target_no_match_returns_null() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        // Constrain c == 'z' (not in the string and not null) -> NULL result.
        let target = RustBV::concrete(b'z' as u128, 64);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);
        let p = NativeStrchr;
        let result = p
            .call(&mut state, &[RustBV::concrete(0x1000, 64), sym])
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0));
        assert_eq!(ctx.max(&result, false), Some(0));
    }

    #[test]
    fn test_memchr_symbolic_target_solver_evaluation() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        let target = RustBV::concrete(3u128, 64);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);
        let p = NativeMemchr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), sym, RustBV::concrete(4, 64)],
            )
            .unwrap()
            .unwrap();
        state.add_constraint(eq);
        let ctx = state.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(0x1002));
        assert_eq!(ctx.max(&result, false), Some(0x1002));
    }
}
