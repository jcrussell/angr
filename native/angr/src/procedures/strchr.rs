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

use super::ProcedureError;
use super::strings::{ConcreteStep, ScanResult, scan_concrete_then_collect};
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
///
/// `nul_returns_addr` (strchrnul) makes the null-terminator arm yield the
/// address of the terminator instead of NULL, so an absent target returns a
/// pointer to the trailing NUL rather than NULL.
fn build_ite_chain(
    byte_loads: &[(u64, RustBV)],
    target_byte: &RustBV,
    arch_bits: u32,
    stop_at_null: bool,
    nul_returns_addr: bool,
    ctx: &SymContext,
) -> RustBV {
    let null_addr = RustBV::concrete(0u128, arch_bits);
    let zero_byte = RustBV::concrete(0u128, 8);
    let mut result = null_addr.clone();
    for (byte_addr, byte_val) in byte_loads.iter().rev() {
        let addr_bv = RustBV::concrete(*byte_addr as u128, arch_bits);
        let match_cond = byte_val.eq(target_byte, ctx);
        if stop_at_null {
            // result = ITE(byte == target, addr, ITE(byte == 0, <nul>, result_next))
            // where <nul> is the terminator address (strchrnul) or NULL (strchr).
            let null_cond = byte_val.eq(&zero_byte, ctx);
            let nul_result = if nul_returns_addr {
                &addr_bv
            } else {
                &null_addr
            };
            let inner = null_cond.ite(nul_result, &result, ctx);
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
    nul_returns_addr: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let arch_bits = state.arch().bits();

    // `collect_stop` for both target flavors: in symbolic-collect mode, stop
    // (for strchr) once a concrete null is seen since later positions are
    // unreachable.
    let collect_stop = |byte_val: &RustBV| {
        stop_at_null && byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false)
    };

    // Concrete fast path: target is concrete. Walk byte by byte with the
    // existing short-circuiting semantics. If a symbolic byte is encountered
    // mid-scan we hand the rest off to the ITE-chain path.
    if let Some(target) = target_arg.as_u64() {
        let target_byte = target as u8;
        let result = scan_concrete_then_collect(
            state,
            max_scan,
            |st, i| Ok(st.memory_load(addr.wrapping_add(i), 1)?),
            |byte_val, i| {
                let byte_addr = addr.wrapping_add(i);
                match byte_val.as_u64() {
                    Some(b) => {
                        let byte = b as u8;
                        if byte == target_byte {
                            ConcreteStep::Stop(RustBV::concrete(byte_addr as u128, arch_bits))
                        } else if stop_at_null && byte == 0 {
                            // Null terminator hit before target — strchr returns
                            // NULL; strchrnul returns the terminator address.
                            // (target == 0 is captured by the match above.)
                            let nul_addr = if nul_returns_addr { byte_addr } else { 0 };
                            ConcreteStep::Stop(RustBV::concrete(nul_addr as u128, arch_bits))
                        } else {
                            // Concrete non-match — no contribution, skip.
                            ConcreteStep::Continue
                        }
                    }
                    None => ConcreteStep::BeginCollect,
                }
            },
            collect_stop,
        )?;

        return Ok(Some(match result {
            ScanResult::Stopped(addr_bv) => addr_bv,
            ScanResult::Exhausted => {
                // All concrete, scanned to end without match. strchr/memchr both
                // return NULL if not found (for strchr this only happens if the
                // buffer has no null in MAX_SCAN).
                if stop_at_null {
                    return Err(ProcedureError::MaxIterations(max_scan as usize));
                }
                RustBV::concrete(0u128, arch_bits)
            }
            ScanResult::Collected(byte_loads) => {
                let byte_loads: Vec<(u64, RustBV)> = byte_loads
                    .into_iter()
                    .map(|(i, bv)| (addr.wrapping_add(i), bv))
                    .collect();
                let ctx = state.solver().borrow();
                let target_byte_bv = RustBV::concrete(target_byte as u128, 8);
                build_ite_chain(
                    &byte_loads,
                    &target_byte_bv,
                    arch_bits,
                    stop_at_null,
                    nul_returns_addr,
                    &ctx,
                )
            }
        }));
    }

    // Symbolic target: collect bytes up to max_scan (or first concrete null
    // for strchr) and build a full ITE chain.
    let result = scan_concrete_then_collect(
        state,
        max_scan,
        |st, i| Ok(st.memory_load(addr.wrapping_add(i), 1)?),
        |_byte_val, _i| ConcreteStep::<RustBV>::BeginCollect,
        collect_stop,
    )?;
    let byte_loads: Vec<(u64, RustBV)> = match result {
        ScanResult::Collected(v) => v
            .into_iter()
            .map(|(i, bv)| (addr.wrapping_add(i), bv))
            .collect(),
        ScanResult::Exhausted => Vec::new(), // max_scan == 0
        ScanResult::Stopped(_) => unreachable!("symbolic target never stops in concrete mode"),
    };

    let ctx = state.solver().borrow();
    let target_byte_bv = target_arg.extract(7, 0, &ctx);
    let chain = build_ite_chain(
        &byte_loads,
        &target_byte_bv,
        arch_bits,
        stop_at_null,
        nul_returns_addr,
        &ctx,
    );
    Ok(Some(chain))
}

crate::declare_proc! {
    /// strchr: find character in string.
    ///
    /// ```c
    /// char *strchr(const char *s, int c);
    /// ```
    ///
    /// Returns pointer to first occurrence of c in s, or NULL if not found.
    name = "strchr",
    struct = NativeStrchr,
    args = [addr: concrete, c: bv],
    call |state| {
        scan_for_byte(
            state,
            addr,
            &c,
            MAX_SCAN as u64,
            /*stop_at_null=*/ true,
            /*nul_returns_addr=*/ false,
        )
    }
}

crate::declare_proc! {
    /// strchrnul: like strchr, but return a pointer to the terminating NUL
    /// (rather than NULL) when the character is not found.
    ///
    /// ```c
    /// char *strchrnul(const char *s, int c);
    /// ```
    name = "strchrnul",
    struct = NativeStrchrnul,
    args = [addr: concrete, c: bv],
    call |state| {
        scan_for_byte(
            state,
            addr,
            &c,
            MAX_SCAN as u64,
            /*stop_at_null=*/ true,
            /*nul_returns_addr=*/ true,
        )
    }
}

crate::declare_proc! {
    /// strrchr: find last occurrence of a character in a null-terminated string.
    ///
    /// ```c
    /// char *strrchr(const char *s, int c);
    /// ```
    ///
    /// Returns pointer to the last occurrence of c in s, or NULL if not found.
    /// Special case (C standard): `strrchr(s, '\0')` returns a pointer to the
    /// trailing null terminator.
    ///
    /// # Symbolic handling
    ///
    /// Concrete address required. The target byte and individual string bytes
    /// may be symbolic — we build an ITE chain over collected loads in the
    /// *forward* direction so later matches override earlier ones (opposite of
    /// strchr, which builds backward so earlier matches win).
    name = "strrchr",
    struct = NativeStrrchr,
    args = [addr: concrete, c: bv],
    call |state| {
        scan_for_byte_last(state, addr, &c, MAX_SCAN as u64, /*stop_at_null=*/ true)
    }
}

/// Scan for the LAST match of `target_arg` at `addr`. Concrete fast path
/// tracks the last hit; on the first symbolic byte we collect remaining loads
/// and build a forward ITE chain.
///
/// `stop_at_null` (strrchr) treats a concrete null as end-of-string: the scan
/// stops there and `max_scan` exhaustion is an error (no terminator found, so
/// we defer to Python). When false (memrchr) the full `max_scan` window is
/// scanned regardless of null bytes, and exhaustion returns the last match (or
/// NULL).
fn scan_for_byte_last(
    state: &mut RustSimState,
    addr: u64,
    target_arg: &RustBV,
    max_scan: u64,
    stop_at_null: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let arch_bits = state.arch().bits();
    let null_addr = RustBV::concrete(0u128, arch_bits);

    // Stop collecting once a concrete null is seen (strrchr end-of-string only).
    let collect_stop = |byte_val: &RustBV| {
        stop_at_null && byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false)
    };

    if let Some(target) = target_arg.as_u64() {
        let target_byte = target as u8;
        let mut last_match: Option<u64> = None;
        let result = scan_concrete_then_collect(
            state,
            max_scan,
            |st, i| Ok(st.memory_load(addr.wrapping_add(i), 1)?),
            |byte_val, i| {
                let byte_addr = addr.wrapping_add(i);
                match byte_val.as_u64() {
                    Some(b) => {
                        let byte = b as u8;
                        if byte == target_byte {
                            last_match = Some(byte_addr);
                        }
                        if stop_at_null && byte == 0 {
                            // End of string in concrete mode — return last match
                            // (or NULL if none).
                            let result_addr = last_match.unwrap_or(0);
                            ConcreteStep::Stop(RustBV::concrete(result_addr as u128, arch_bits))
                        } else {
                            ConcreteStep::Continue
                        }
                    }
                    // Seed the chain with the best concrete match so far.
                    None => ConcreteStep::BeginCollect,
                }
            },
            collect_stop,
        )?;

        return Ok(Some(match result {
            ScanResult::Stopped(addr_bv) => addr_bv,
            ScanResult::Exhausted => {
                // strrchr: ran past max without hitting a null terminator —
                // defer to Python. memrchr: the n-window is exhausted, so the
                // last concrete match (or NULL) is the answer.
                if stop_at_null {
                    return Err(ProcedureError::MaxIterations(max_scan as usize));
                }
                RustBV::concrete(last_match.unwrap_or(0) as u128, arch_bits)
            }
            ScanResult::Collected(byte_loads) => {
                let byte_loads: Vec<(u64, RustBV)> = byte_loads
                    .into_iter()
                    .map(|(i, bv)| (addr.wrapping_add(i), bv))
                    .collect();
                let ctx = state.solver().borrow();
                let target_byte_bv = RustBV::concrete(target_byte as u128, 8);
                // Build forward so later matches override earlier ones.
                let seed = match last_match {
                    Some(a) => RustBV::concrete(a as u128, arch_bits),
                    None => null_addr,
                };
                build_ite_chain_forward(&byte_loads, &target_byte_bv, arch_bits, &seed, &ctx)
            }
        }));
    }

    // Symbolic target: collect bytes up to (and including) concrete null.
    let result = scan_concrete_then_collect(
        state,
        max_scan,
        |st, i| Ok(st.memory_load(addr.wrapping_add(i), 1)?),
        |_byte_val, _i| ConcreteStep::<RustBV>::BeginCollect,
        collect_stop,
    )?;
    let byte_loads: Vec<(u64, RustBV)> = match result {
        ScanResult::Collected(v) => v
            .into_iter()
            .map(|(i, bv)| (addr.wrapping_add(i), bv))
            .collect(),
        ScanResult::Exhausted => Vec::new(), // max_scan == 0
        ScanResult::Stopped(_) => unreachable!("symbolic target never stops in concrete mode"),
    };

    let ctx = state.solver().borrow();
    let target_byte_bv = target_arg.extract(7, 0, &ctx);
    let chain = build_ite_chain_forward(&byte_loads, &target_byte_bv, arch_bits, &null_addr, &ctx);
    Ok(Some(chain))
}

/// Forward variant of [`build_ite_chain`]: later matches override earlier
/// ones. Used by strrchr so the final hit wins over the first hit.
///
/// `seed` is the starting result (concrete-match address or NULL); ITE arms
/// rewrite it with the current address when `byte == target`.
fn build_ite_chain_forward(
    byte_loads: &[(u64, RustBV)],
    target_byte: &RustBV,
    arch_bits: u32,
    seed: &RustBV,
    ctx: &SymContext,
) -> RustBV {
    let zero_byte = RustBV::concrete(0u128, 8);
    let mut result = seed.clone();
    for (byte_addr, byte_val) in byte_loads.iter() {
        let addr_bv = RustBV::concrete(*byte_addr as u128, arch_bits);
        // Past a null terminator the position is unreachable; treat it as
        // "no match" so the chain doesn't promote bytes beyond end-of-string.
        let is_null = byte_val.eq(&zero_byte, ctx);
        let match_cond = byte_val.eq(target_byte, ctx);
        // If null AND target == 0, we still want the null address to be the
        // result (C standard: strrchr(s, '\0') → pointer to null terminator).
        // The match_cond above already covers target == 0 at the null byte,
        // so just respect match_cond and ignore later positions when null.
        // Implementation detail: we stop collecting loads after the first
        // concrete null in scan_for_byte_last, so positions past that point
        // are never on the chain.
        let _ = is_null;
        result = match_cond.ite(&addr_bv, &result, ctx);
    }
    result
}

crate::declare_proc! {
    /// memchr: find byte in memory region.
    ///
    /// ```c
    /// void *memchr(const void *s, int c, size_t n);
    /// ```
    ///
    /// Returns pointer to first occurrence of c in the first n bytes of s, or NULL.
    ///
    /// `n` must be concrete to bound the scan. Symbolic n could be handled by
    /// taking the maximum solution, but that would require a solver call we
    /// currently push onto the Python fallback path.
    name = "memchr",
    struct = NativeMemchr,
    args = [addr: concrete, c: bv, n: concrete],
    call |state| {
        let scan_len = n.min(MAX_SCAN as u64);
        scan_for_byte(
            state, addr, &c, scan_len, /*stop_at_null=*/ false,
            /*nul_returns_addr=*/ false,
        )
    }
}

crate::declare_proc! {
    /// rawmemchr: like memchr, but without a length bound — the caller
    /// guarantees the byte is present (UB otherwise).
    ///
    /// ```c
    /// void *rawmemchr(const void *s, int c);
    /// ```
    ///
    /// We scan up to MAX_SCAN bytes; if the byte is not found we return NULL
    /// (the UB case), matching the non-found behavior of the bounded memchr.
    name = "rawmemchr",
    struct = NativeRawmemchr,
    args = [addr: concrete, c: bv],
    call |state| {
        scan_for_byte(
            state, addr, &c, MAX_SCAN as u64, /*stop_at_null=*/ false,
            /*nul_returns_addr=*/ false,
        )
    }
}

crate::declare_proc! {
    /// memrchr: find the LAST occurrence of a byte in the first n bytes.
    ///
    /// ```c
    /// void *memrchr(const void *s, int c, size_t n);
    /// ```
    ///
    /// Returns pointer to the last occurrence of c in the first n bytes of s,
    /// or NULL if not found. `n` must be concrete to bound the scan.
    name = "memrchr",
    struct = NativeMemrchr,
    args = [addr: concrete, c: bv, n: concrete],
    call |state| {
        let scan_len = n.min(MAX_SCAN as u64);
        scan_for_byte_last(state, addr, &c, scan_len, /*stop_at_null=*/ false)
    }
}

#[cfg(test)]
#[path = "strchr_tests.rs"]
mod strchr_tests;
