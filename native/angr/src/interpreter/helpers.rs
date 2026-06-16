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
pub(super) fn bv_to_bytes(bv: &RustBV) -> Vec<u8> {
    let width = bv.width();
    let num_bytes = width.div_ceil(8) as usize;

    if let Some(value) = bv.as_u128() {
        let mut bytes = vec![0u8; num_bytes];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> (i * 8)) as u8;
        }
        bytes
    } else {
        // For symbolic values, return zeros (the callback will handle it)
        vec![0u8; num_bytes]
    }
}

/// Convert bytes (little-endian) to a RustBV.
pub(super) fn bytes_to_bv(bytes: &[u8], width: u32) -> RustBV {
    let mut value: u128 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i * 8 >= width as usize {
            break;
        }
        value |= (byte as u128) << (i * 8);
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

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
