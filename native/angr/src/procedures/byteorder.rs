//! Native byte-order conversion (arpa/inet.h): htonl, htons, ntohl, ntohs.
//!
//! These convert between host and network (big-endian) byte order. On a
//! little-endian host they byte-swap the low N bytes of the argument and
//! zero-extend the result back to the arch word width; on a big-endian host
//! they are the identity (host order already equals network order).
//!
//! `ntohl`/`ntohs` perform the identical operation — byte-order conversion is
//! symmetric — so they are registered as aliases of the `hton*` impls. Python
//! ships neither `ntoh*` proc, so the aliases are pure parity bonus.
//!
//! Mirrors `angr/procedures/libc/htonl.py` and `htons.py` verbatim
//! (`to_convert[N-1:0].reversed.zero_extend(...)` on `Iend_LE`, else identity).

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Shared host<->network conversion: on a little-endian host, byte-reverse the
/// low `low_bits` of `arg` and zero-extend back to the arch width; on a
/// big-endian host, return `arg` unchanged (matches the Python procedures).
///
/// `reverse`/`extract`/`zero_extend` each fold concrete operands internally, so
/// this single path covers both the concrete and symbolic cases.
fn host_network_swap(
    state: &RustSimState,
    arg: &RustBV,
    low_bits: u32,
) -> Result<Option<RustBV>, ProcedureError> {
    if !state.is_little_endian() {
        return Ok(Some(arg.clone()));
    }
    let bits = state.arch().bits();
    let ctx = state.solver().borrow();
    let low = arg.extract(low_bits - 1, 0, &ctx);
    let swapped = low.reverse(&ctx);
    Ok(Some(swapped.zero_extend(bits, &ctx)))
}

crate::declare_proc! {
    /// `uint32_t htonl(uint32_t hostlong)` — host to network long.
    /// `ntohl` is aliased (identical byte-order swap).
    name = "htonl",
    struct = NativeHtonl,
    args = [v: bv],
    aliases = ["ntohl"],
    call |state| {
        host_network_swap(state, &v, 32)
    }
}

crate::declare_proc! {
    /// `uint16_t htons(uint16_t hostshort)` — host to network short.
    /// `ntohs` is aliased (identical byte-order swap).
    name = "htons",
    struct = NativeHtons,
    args = [v: bv],
    aliases = ["ntohs"],
    call |state| {
        host_network_swap(state, &v, 16)
    }
}

test_submod!("byteorder_tests.rs" => tests);
