//! Free `RustBV` <-> bytes conversion helpers shared by the statement and
//! expression paths.
//!
//! Nothing here touches [`VEXInterpreter`] state — these are pure functions
//! over a bitvector, kept out of `expressions.rs`/`statements.rs` so both can
//! reach them without one depending on the other. Beyond the two byte
//! conversions, the file holds [`extract_ite_targets`] (harvest the concrete
//! leaves of a nested ITE jump target), [`build_balanced_ite`] (the inverse:
//! fold a value set back into a depth-balanced ITE), and
//! [`reject_symbolic_byte_store`], the guard that turns the lossy
//! symbolic-value byte-store fallback into a loud error.

use super::*;

pub(super) fn extract_ite_targets(bv: &RustBV, max_targets: usize) -> Option<Vec<u64>> {
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<&RustBV> = vec![bv];

    while let Some(current) = stack.pop() {
        if targets.len() > max_targets {
            return None; // Too many targets
        }

        match current {
            RustBV::Concrete { value, .. } => {
                let addr = *value as u64;
                if seen.insert(addr) {
                    targets.push(addr);
                }
            }
            RustBV::Constrained { value, .. } => {
                let addr = *value as u64;
                if seen.insert(addr) {
                    targets.push(addr);
                }
            }
            RustBV::Expression {
                op: BVOp::Ite,
                operands,
                ..
            } if operands.len() == 3 => {
                // ITE: operands[0] = condition, operands[1] = true_val, operands[2] = false_val
                stack.push(&operands[1]); // true branch
                stack.push(&operands[2]); // false branch
            }
            _ => {
                // Non-ITE symbolic expression — can't extract targets
                return None;
            }
        }
    }

    if targets.is_empty() {
        None
    } else {
        Some(targets)
    }
}

/// Convert a RustBV to bytes (little-endian).
///
/// Shared with the callback-dispatch store path
/// (`callbacks::dispatch::store_bytes_via_callback`), which used to carry a
/// byte-identical private copy that inherited every bug fixed here
/// (angr-03vl4.4).
pub(crate) fn bv_to_bytes(bv: &RustBV) -> Vec<u8> {
    let width = bv.width();
    let num_bytes = width.div_ceil(8) as usize;

    if let Some(value) = bv.as_u128() {
        let mut bytes = vec![0u8; num_bytes];
        // `take(WIDE_PAYLOAD_BYTES)`: a `Concrete` wider than 128 bits stores
        // only its low 128 bits (see `RustBV::Concrete`'s `width` doc and the
        // `width.min(128)` branches in `value_ops.rs::shl_into`), so every byte
        // from index 16 up reads as 0 — which the `vec![0u8; _]` already gives.
        // Shifting a `u128` by >= 128 would abort under debug assertions and
        // wrap in release, aliasing byte 16 onto byte 0 (angr-03vl4.4).
        for (i, byte) in bytes.iter_mut().take(WIDE_PAYLOAD_BYTES).enumerate() {
            *byte = (value >> (i * 8)) as u8;
        }
        bytes
    } else {
        // For symbolic values, return zeros (the callback will handle it)
        vec![0u8; num_bytes]
    }
}

/// Number of little-endian bytes a `RustBV::Concrete`'s `u128` payload can
/// hold. Widths above `WIDE_PAYLOAD_BYTES * 8` are representable (AVX/YMM
/// scale, up to 256) but keep only their low bits — see `RustBV::Concrete`.
const WIDE_PAYLOAD_BYTES: usize = 16;

/// Reject a byte-level store of a *symbolic* value (angr-9ke6b.19).
///
/// Several store paths fall back to [`bv_to_bytes`] + the byte-level
/// `memory_store` callback when `PythonCallbacks::has_memory_store_symbolic_value`
/// is false. `bv_to_bytes` yields all-zero bytes for a symbolic expression, so
/// that fallback would silently overwrite memory with 0 instead of storing the
/// value — a wrong answer, not a degraded one. Concrete values *are* exact
/// under `bv_to_bytes`, so only the symbolic case is an error.
///
/// Call this immediately before any `bv_to_bytes`-based fallback that a
/// `has_memory_store_symbolic_value()` guard would otherwise have skipped, so
/// the loud error the `avoid-silent-no-op-callback-fallbacks` invariant
/// promises fires even if a future caller drops the guard.
/// `PythonCallbacks::call_memory_store_symbolic_value` enforces the same rule
/// on its own fallback path.
pub(super) fn reject_symbolic_byte_store(
    value: &RustBV,
    addr: u64,
    site: &str,
) -> Result<(), CbExecutionError> {
    if value.is_symbolic() {
        return Err(CbExecutionError::Unsupported(format!(
            "{site}: symbolic store to 0x{addr:x} needs the \
             memory_store_symbolic_value callback, which is not set \
             (refusing to zero-fill memory)"
        )));
    }
    Ok(())
}

/// Convert bytes (little-endian) to a RustBV.
///
/// The mirror image of [`bv_to_bytes`], and it inherits the same
/// `WIDE_PAYLOAD_BYTES` ceiling: a `RustBV::Concrete` of `width > 128` keeps
/// only its low 128 bits, so bytes at index 16 and up cannot be represented.
pub(super) fn bytes_to_bv(bytes: &[u8], width: u32) -> RustBV {
    let width_bytes = (width as usize).div_ceil(8).min(bytes.len());
    let mut value: u128 = 0;
    // `min(WIDE_PAYLOAD_BYTES)`: `<< (i * 8)` with `i >= 16` would abort under
    // debug assertions and wrap in release, OR-ing byte 16 over byte 0 — the
    // mirror image of the `bv_to_bytes` overflow (angr-03vl4.31).
    for (i, &byte) in bytes
        .iter()
        .take(width_bytes.min(WIDE_PAYLOAD_BYTES))
        .enumerate()
    {
        value |= (byte as u128) << (i * 8);
    }
    // SILENT(cat-c): the high bytes of a >128-bit load have nowhere to go in
    // the `u128` payload, so they are dropped and the value silently reads back
    // as zero above bit 127. Losing *nonzero* data that way is a wrong answer
    // rather than a degraded one, so say so.
    if bytes[..width_bytes]
        .iter()
        .skip(WIDE_PAYLOAD_BYTES)
        .any(|&b| b != 0)
    {
        log::warn!(
            "bytes_to_bv: dropping nonzero bytes above bit 127 of a {width}-bit value; \
             RustBV::Concrete stores only its low 128 bits"
        );
    }
    RustBV::concrete(value, width)
}

/// Build a balanced ITE tree from a list of (condition, value) pairs.
///
/// Build a linear ITE chain: ITE(c1, v1, ITE(c2, v2, ... ITE(cn, vn, default)))
/// Returns the value for the first matching condition, or a default value.
pub(super) fn build_balanced_ite(
    pairs: &[(RustBV, RustBV)],
    default_value: RustBV,
    ctx: &SymContext,
) -> RustBV {
    // Build right-to-left: innermost ITE is the last pair
    let mut result = default_value;
    for (cond, val) in pairs.iter().rev() {
        result = cond.ite(val, &result, ctx);
    }
    result
}

test_submod!("bv_utils_tests.rs" => tests);
