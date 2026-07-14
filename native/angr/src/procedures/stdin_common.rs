//! Shared harness-seeded-stdin plumbing for the native stdin readers.
//!
//! When a Python harness fills `posix.stdin.content` before exploration, those
//! bytes are attached to fd 0 as bounded symbolic content at seed time
//! (`RustExplorationManager::_seed_stdin_to_rust` -> `FileSystem::set_fd_content_sym`;
//! see the `stdin-seed-binding-architecture` memory). Every native reader of
//! fd 0 — `read`, `fgets`/`fgetc`/`getchar`/`gets`, `scanf`, CGC `receive` —
//! must consume that seed in read order, otherwise a harness that seeds stdin
//! and whose target reads it through, say, `fgets` gets fresh unconstrained
//! symbols and a zero prefix in `posix.dumps(0)` (angr-ptf54).
//!
//! [`mint_stdin_bytes`] is the one place that does it.

use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Mint one fresh 8-bit leaf symbol per name, consuming any harness-seeded
/// fd-0 bytes and binding the leaves to them.
///
/// The buffer the caller stores must hold plain 8-bit *leaf* symbols: the
/// Python-bounce memory round-trip preserves leaf identity but not a composite
/// `Extract` over the harness's wide BVS, which silently drops branch
/// constraints on a bounced path (see `avoid-storing-extract-asts-in-guest-buffers`).
/// So the seed is tied in with an `eq` constraint rather than stored directly —
/// the path condition still references the harness's own symbol, so an exported
/// found state can `solver.eval` it.
///
/// Bytes with a seeded counterpart are deliberately *not* `record_stdin_symbol`'d:
/// they are already present in the Python stdin stream (and now solve to the same
/// value), so recording them would make `_inject_rust_stdin` append a duplicate
/// copy after the harness's own chunk. Reads past the end of the seed mint and
/// record fresh symbols as before.
///
/// Advances fd 0's position by the number of seeded bytes actually available,
/// mirroring Python, where the same `SimFileStream` read consumes them.
pub fn mint_stdin_bytes(state: &mut RustSimState, names: &[String]) -> Vec<RustBV> {
    let seeded: Vec<RustBV> = state
        .file_system()
        .read_sym(0, names.len())
        .unwrap_or_default();

    let (sym_bytes, bindings): (Vec<RustBV>, Vec<RustBV>) = {
        let ctx = state.solver().borrow();
        let sym_bytes: Vec<RustBV> = names
            .iter()
            .map(|name| RustBV::symbolic(&ctx, name, 8))
            .collect();
        let bindings = seeded
            .iter()
            .zip(&sym_bytes)
            .map(|(seed, sym)| sym.eq(seed, &ctx))
            .collect();
        (sym_bytes, bindings)
    };
    for c in bindings {
        state.add_constraint(c);
    }

    for name in names.iter().skip(seeded.len()) {
        state.record_stdin_symbol(name.clone(), 8);
    }

    sym_bytes
}

/// True while fd 0 still holds harness-seeded bytes the guest has not read.
///
/// Position-vs-length peek only — no read, no CoW. A native stdin reader that
/// cannot map the seed byte-for-byte onto what it mints must defer to Python
/// rather than mint an unconstrained value and leave the seed unconsumed: the
/// only such reader is `scanf`'s numeric conversion, which models a decimal/hex
/// parse of a digit run instead of copying bytes (angr-ggb66).
pub fn stdin_seed_unconsumed(state: &RustSimState) -> bool {
    let fs = state.file_system_ref();
    let Some(content) = fs.fd_content_sym(0) else {
        return false;
    };
    let pos = fs.fd_info(0).map_or(0, |info| info.1);
    (pos as usize) < content.len()
}
