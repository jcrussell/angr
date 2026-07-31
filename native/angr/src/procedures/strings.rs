//! Shared helpers for null-terminated string scans across libc string procedures.
//!
//! The procedures `strlen`, `strcpy`, `strncpy`, `strdup`, `strcat`, `strncat`,
//! `strstr`, `strchr`, and friends all need to walk memory byte-by-byte until a
//! null terminator. Two flavors emerge:
//!
//! 1. **Concrete-only**: bail to Python on the first symbolic byte. Used by
//!    procedures that write to memory (strcpy/strdup/strcat) or do
//!    string-search / comparison work that can't be expressed as an ITE
//!    chain (strstr, strcmp, strchr).
//!    Helpers: [`scan_concrete_until_null`], [`find_null_addr`],
//!    [`scan_concrete_then_collect`] (collect-as-you-scan for strcmp/strchr).
//!
//! 2. **Symbolic-aware**: build an ITE chain over collected bytes so the
//!    result can be symbolic. Used by strlen, strcpy, and the file-read
//!    scan helpers.
//!    Helpers: [`scan_for_null_symbolic`], [`build_strlen_chain`].
//!
//! Centralizing these means null-boundary edge cases live in one place. Tests
//! for the shared helpers live in this file.
//!
//! Anchor for source-of-history: angr-arf5.

use super::{ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Shared upper bound on a concrete null-terminated string scan for the
/// str-family procedures (strcpy/strncpy/strcat/strncat, strstr, strset,
/// memset-of-string, …). Hitting this without a null terminator bails to
/// Python via [`ProcedureError::MaxIterations`].
///
/// One home so a future security-driven reduction is applied everywhere at
/// once rather than silently missing a copy that kept its own local `4096`
/// (angr-myzjx.7). `strcmp`/`memcmp` share their own equivalent
/// (`strcmp::MAX_STRCMP_LEN`); the printf/scanf family share
/// [`super::format_common::MAX_FORMAT_LEN`].
pub(crate) const MAX_STRING_SCAN: usize = 4096;

/// Concrete byte-by-byte scan up to and including the null terminator.
///
/// Returns the bytes read **up to but not including** the null terminator.
/// Errors:
/// - `SymbolicArgument` if any byte is symbolic.
/// - `MaxIterations(max)` if `max` bytes are scanned without finding a null.
///
/// `addr_label` is used to format symbolic-byte error messages.
pub(crate) fn scan_concrete_until_null(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<Vec<u8>, ProcedureError> {
    let mut buf = Vec::with_capacity(256.min(max));
    for i in 0..max as u64 {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)?;
        let byte = extract_concrete_arg(&byte_val, &format!("{addr_label}[{i}]"))? as u8;
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
pub(crate) fn scan_concrete_bounded(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<(Vec<u8>, bool), ProcedureError> {
    let mut buf = Vec::with_capacity(max);
    for i in 0..max as u64 {
        let byte_val = state.memory_load(addr.wrapping_add(i), 1)?;
        let byte = extract_concrete_arg(&byte_val, &format!("{addr_label}[{i}]"))? as u8;
        if byte == 0 {
            return Ok((buf, true));
        }
        buf.push(byte);
    }
    Ok((buf, false))
}

/// Best-effort concrete scan that never errors. Reads up to `max` bytes,
/// stopping (and returning whatever was collected so far) at the first null
/// terminator, the first symbolic byte, the first failed `memory_load`, or
/// `max` — whichever comes first. The null itself is not included.
///
/// Used by consumers like `puts` that print whatever concrete prefix is
/// available and have no need to distinguish the stop reasons (so, unlike
/// [`scan_concrete_bounded`], a symbolic byte or out-of-bounds read is a quiet
/// stop, not a `SymbolicArgument` / load error).
pub(crate) fn scan_concrete_lossy(state: &mut RustSimState, addr: u64, max: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    for i in 0..max as u64 {
        match state.memory_load(addr.wrapping_add(i), 1) {
            Ok(bv) => match bv.as_u64() {
                Some(0) | None => break,
                Some(b) => buf.push(b as u8),
            },
            Err(_) => break,
        }
    }
    buf
}

/// Concrete scan that returns the address of the first null terminator.
///
/// Used by strcat/strncat which need the null position, not the bytes.
/// Errors identical to [`scan_concrete_until_null`].
pub(crate) fn find_null_addr(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
) -> Result<u64, ProcedureError> {
    for i in 0..max as u64 {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state.memory_load(byte_addr, 1)?;
        let byte =
            extract_concrete_arg(&byte_val, &format!("{addr_label} at 0x{byte_addr:x}"))? as u8;
        if byte == 0 {
            return Ok(byte_addr);
        }
    }
    Err(ProcedureError::MaxIterations(max))
}

/// Concrete byte-by-byte scan at `addr`, consulting `decide` for each byte
/// (the null terminator included) until it yields `Some(result)`.
///
/// Factors out the shared boilerplate of the concrete *search* procedures —
/// strstr's haystack scan and strpbrk/strspn/strcspn — which all walk
/// `0..max`, load one byte, bail to Python on a symbolic byte via
/// [`extract_concrete_arg`], and error with [`ProcedureError::MaxIterations`]
/// if `max` bytes pass without a decision. Only the per-byte predicate
/// differs.
///
/// `decide` receives `(state, byte, i, byte_addr)`. Threading `state` in lets
/// predicates that must peek ahead (strstr's needle match) do their own
/// loads; the set-membership predicates (strpbrk/strspn/strcspn) ignore it.
/// The null terminator is passed to `decide` like any other byte, so each
/// procedure encodes its own end-of-string result (strstr → NULL, strspn →
/// the run length, …).
pub(crate) fn scan_concrete_predicate<R>(
    state: &mut RustSimState,
    addr: u64,
    max: usize,
    addr_label: &str,
    mut decide: impl FnMut(&mut RustSimState, u8, u64, u64) -> Result<Option<R>, ProcedureError>,
) -> Result<R, ProcedureError> {
    for i in 0..max as u64 {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state.memory_load(byte_addr, 1)?;
        let byte = extract_concrete_arg(&byte_val, &format!("{addr_label}[{i}]"))? as u8;
        if let Some(r) = decide(state, byte, i, byte_addr)? {
            return Ok(r);
        }
    }
    Err(ProcedureError::MaxIterations(max))
}

/// Write a slice of concrete bytes into memory, one 8-bit store per byte.
///
/// This is the write-side counterpart to the read/scan helpers above. The
/// per-byte `memory_store(addr + i, RustBV::concrete(byte, 8))` loop is the
/// single most-duplicated idiom across the writing procedures (strcpy,
/// strcat, getenv, sprintf, strncpy, snprintf, fread, read). Centralizing it
/// keeps the store boundary in one place.
pub(crate) fn write_concrete_bytes(
    state: &mut RustSimState,
    addr: u64,
    bytes: &[u8],
) -> Result<(), ProcedureError> {
    for (i, &byte) in bytes.iter().enumerate() {
        state.memory_store(
            addr.wrapping_add(i as u64),
            RustBV::concrete(byte as u128, 8),
        )?;
    }
    Ok(())
}

/// Write a vector of byte-wide (possibly symbolic) `RustBV`s into memory,
/// one 8-bit store per entry — the symbolic sibling of
/// [`write_concrete_bytes`]. Used by the `content_sym` serve paths
/// (read/fread/readv/pread64, angr-0xyq2 Phase 2) and the symbolic-byte
/// minting loops (stdin reads, fgets). Returns the raw
/// [`MemoryError`](crate::memory::MemoryError) so both procedure
/// (`ProcedureError`) and syscall (`SyscallError`) callers can `?` it
/// through their `#[from]` conversions.
pub(crate) fn write_bv_bytes(
    state: &mut RustSimState,
    addr: u64,
    bytes: Vec<RustBV>,
) -> Result<(), crate::memory::MemoryError> {
    for (i, b) in bytes.into_iter().enumerate() {
        state.memory_store(addr.wrapping_add(i as u64), b)?;
    }
    Ok(())
}

/// Write `bytes` followed by a trailing NUL terminator at `addr + bytes.len()`.
///
/// Used by the C-string-producing procedures (strcpy, strdup, strcat,
/// strncat, getenv, sprintf, snprintf) that always null-terminate. For
/// truncated/bounded writers that place the terminator at a caller-chosen
/// offset, call [`write_concrete_bytes`] and store the NUL separately.
pub(crate) fn write_cstr(
    state: &mut RustSimState,
    addr: u64,
    bytes: &[u8],
) -> Result<(), ProcedureError> {
    write_concrete_bytes(state, addr, bytes)?;
    state.memory_store(
        addr.wrapping_add(bytes.len() as u64),
        RustBV::concrete(0u128, 8),
    )?;
    Ok(())
}

/// One decision of the concrete fast path inside [`scan_concrete_then_collect`].
pub(crate) enum ConcreteStep<R> {
    /// Byte was concrete and contributes nothing to the result; advance.
    Continue,
    /// Byte was concrete and terminates the scan with result `R`.
    Stop(R),
    /// A symbolic byte was seen; switch to collecting loads for an ITE chain.
    BeginCollect,
}

/// Outcome of [`scan_concrete_then_collect`].
pub(crate) enum ScanResult<R, B> {
    /// The concrete fast path returned early with `R`.
    Stopped(R),
    /// The loop ran to `max` entirely in concrete mode without stopping.
    Exhausted,
    /// A symbolic byte was seen; here are the collected `(position, load)`
    /// entries. The symbolic byte that triggered collection is included, and
    /// collection halts after the first load for which `collect_stop` is true
    /// (typically a concrete null terminator, past which positions cannot
    /// affect the ITE chain).
    Collected(Vec<(u64, B)>),
}

/// Number of *symbolic* positions a scan may collect before it stops looking.
///
/// Mirrors `state.libc.buf_symbolic_bytes` (`angr/state_plugins/libc.py`),
/// which angr's Python string procedures pass to `memory.find` as
/// `max_symbolic_bytes`: `smart_find_mixin._find_iter_items` stops generating
/// cases once that many symbolic characters have been seen. Without the cap a
/// fully-symbolic buffer yields an ITE chain (and a `null_exists_constraint`
/// disjunction) thousands of terms wide, and every one of those terms keeps a
/// downstream branch feasible — the fork-storm the Python bound exists to
/// prevent (angr-gorvf.5).
///
/// Concrete positions are free, exactly as in Python: only symbolic ones draw
/// down the budget.
pub(crate) const MAX_SYMBOLIC_SCAN_BYTES: usize = 60;

/// Generic skeleton shared by the symbolic-aware string scans
/// (strlen/strchr/strrchr/strcmp/memchr/memcmp). It walks positions `0..max`,
/// loading per-position data of type `B` via `load`. While still in the
/// concrete fast path it consults `concrete` for each load; once a symbolic
/// byte forces [`ConcreteStep::BeginCollect`] it collects every subsequent
/// load (plus the triggering one) into a vec, stopping after the first load
/// for which `collect_stop` returns true — or once
/// [`MAX_SYMBOLIC_SCAN_BYTES`] symbolic positions have been collected.
///
/// Centralizing the loop keeps the null-boundary edge cases — the most
/// soundness-sensitive part of these procedures — in one tested place.
pub(crate) fn scan_concrete_then_collect<B, R>(
    state: &mut RustSimState,
    max: u64,
    mut load: impl FnMut(&mut RustSimState, u64) -> Result<B, ProcedureError>,
    mut concrete: impl FnMut(&B, u64) -> ConcreteStep<R>,
    mut collect_stop: impl FnMut(&B) -> bool,
) -> Result<ScanResult<R, B>, ProcedureError> {
    let mut collected: Vec<(u64, B)> = Vec::new();
    let mut symbolic_seen = false;
    let mut symbolic_budget = MAX_SYMBOLIC_SCAN_BYTES;
    for i in 0..max {
        let loaded = load(state, i)?;
        // `concrete` doubles as the symbolic predicate: it answers
        // `BeginCollect` exactly for the loads this scan considers symbolic.
        let step = concrete(&loaded, i);
        let is_symbolic = matches!(step, ConcreteStep::BeginCollect);
        if !symbolic_seen {
            match step {
                ConcreteStep::Continue => continue,
                ConcreteStep::Stop(r) => return Ok(ScanResult::Stopped(r)),
                ConcreteStep::BeginCollect => symbolic_seen = true,
            }
        }
        if is_symbolic {
            match symbolic_budget.checked_sub(1) {
                Some(rest) => symbolic_budget = rest,
                // Budget spent: stop before collecting this position, matching
                // `_find_iter_items`' early `return`.
                None => break,
            }
        }
        let stop = collect_stop(&loaded);
        collected.push((i, loaded));
        if stop {
            break;
        }
    }
    if symbolic_seen {
        Ok(ScanResult::Collected(collected))
    } else {
        Ok(ScanResult::Exhausted)
    }
}

/// Result of [`scan_for_null_symbolic`].
pub(crate) enum ScanOutcome {
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
pub(crate) fn scan_for_null_symbolic(
    state: &mut RustSimState,
    addr: u64,
    max: u64,
) -> Result<ScanOutcome, ProcedureError> {
    let result = scan_concrete_then_collect(
        state,
        max,
        |st, i| Ok(st.memory_load(addr.wrapping_add(i), 1)?),
        |byte_val, i| match byte_val.as_u64() {
            Some(b) if (b as u8) == 0 => ConcreteStep::Stop(i),
            Some(_) => ConcreteStep::Continue,
            None => ConcreteStep::BeginCollect,
        },
        |byte_val| byte_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false),
    )?;
    Ok(match result {
        ScanResult::Stopped(length) => ScanOutcome::AllConcrete { length },
        ScanResult::Exhausted => ScanOutcome::AllConcrete { length: max },
        ScanResult::Collected(bytes) => ScanOutcome::Symbolic { bytes },
    })
}

/// Build the strlen ITE chain over collected `(position, byte_8bit)` loads.
///
/// Chain is built right-to-left so earlier null positions take precedence:
/// `result = ITE(b_i == 0, i, result_next)`.
///
/// `default_len` is the value used past the scanned region (typically the
/// upper bound: `MAX_STRLEN` for strlen, `maxlen` for strnlen).
pub(crate) fn build_strlen_chain(
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

/// Constraint asserting that a null terminator exists somewhere in the
/// scanned window, given the `(position, byte)` pairs collected by
/// [`scan_for_null_symbolic`] (or the s1 side of a strcmp scan).
///
/// This is the pruning step angr's Python string procedures get for free from
/// `state.memory.find`, which asserts that the searched-for byte is actually
/// present within `max_str_len` (`procedures/libc/strlen.py`). Without it the
/// native ITE chains leave "the string has no terminator in the window"
/// satisfiable, so a fully-symbolic buffer keeps every downstream comparison
/// branch feasible and exploration fork-storms where Python deadends
/// (angr-sgcye).
///
/// Returns `None` — i.e. no constraint needed — when the window already
/// contains a concretely-null byte (the assertion would be trivially true) or
/// when no collected byte is symbolic (the assertion would be trivially false
/// and would wrongly kill the state).
///
/// **Only sound when the window is the procedure's own MAX bound**, not a
/// caller-supplied one: `strncmp(a, b, 4)` may legitimately compare four
/// non-null bytes, so bounded variants (strnlen/strncmp with `n < MAX`) must
/// not apply this.
pub(crate) fn null_exists_constraint(bytes: &[(u64, RustBV)], ctx: &SymContext) -> Option<RustBV> {
    let zero_byte = RustBV::concrete(0u128, 8);
    let mut disjunction: Option<RustBV> = None;
    for (_, byte) in bytes {
        match byte.as_u64() {
            // Concrete null in the window: a terminator provably exists.
            Some(0) => return None,
            // Concrete non-null: a false disjunct, contributes nothing.
            Some(_) => continue,
            None => {
                let is_null = byte.eq(&zero_byte, ctx);
                disjunction = Some(match disjunction {
                    Some(d) => d.or(&is_null, ctx),
                    None => is_null,
                });
            }
        }
    }
    disjunction
}

#[cfg(test)]
#[path = "strings_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
