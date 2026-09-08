//! Lazy Multi cells: the `MultiPayload` data structure and the whole
//! install → collapse → cache → coalesce lifecycle around it.
//!
//! A symbolic-address store whose concretization yields a *candidate set*
//! (see `AddressConcretizer::write_range_applies`) installs one lazy Multi
//! cell per covered byte — a disjunction of `(addr == cand) -> byte` guarded
//! alternatives — instead of eagerly folding an ITE into `symbolic_objects`.
//! Loads must then reconstruct that disjunction, and a later concrete store
//! must invalidate it. These tests cover both halves plus the caches that sit
//! between them.
//!
//! This module holds only the fixtures every submodule shares; the tests are
//! split by lifecycle stage, each submodule still the regression home for the
//! delivery phase that built it:
//!
//! - `payload` — **Phase 1.1** (angr-me3z): `MultiPayload`/`MultiAlternative`
//!   themselves and their sidecar storage (`multi_objects`, `multi_bitmap`)
//!   round-trips; and **angr-6cp06.63**: installing a cell over a byte a
//!   wider `symbolic_objects` entry already covers must retire that container
//!   into per-byte lanes (`retire_symbolic_object_at`).
//! - `collapse` — **Phase 1.2** (angr-n082): collapse of installed cells on
//!   the lazy load path (`load_concrete_lazy_inner` →
//!   `assemble_load_with_multi`), plus **angr-9ke6b.96**: the non-lazy
//!   `load_concrete` path (behind `RustSimState::memory_load`) must dispatch
//!   to `assemble_load_with_multi` as well, since
//!   `install_multi_for_candidates` never sets the page's symbolic bit.
//! - `install` — **Phase 1.3** (angr-aija): the store-side helpers
//!   `store_concrete_multi` / `store_symbolic_unified_multi`; and **Phase 2**
//!   (angr-qh5u): end-to-end install via `install_multi_for_candidates`,
//!   lazy-region safety, and Multi/concrete interaction.
//! - `cache` — **Phase 3** (angr-j0n4): the per-load collapse cache; **Phase
//!   4.1** (angr-mmdh.1): the wider-load collapse cache; and **angr-1tes**: a
//!   concrete overwrite must clear the `multi_objects` entry, not just the
//!   page bitmap bit, so the stale alternative cannot resurface.
//! - `coalesce` — **Phase 4.2** (angr-mmdh.2): `flush_multi_cells` run
//!   coalescing, including its stop at the address-space wrap and the
//!   `cond_fingerprint` run key.
//!
//! Merge-time Multi semantics live in the sibling `merge_multi` module, not
//! here.

use super::super::*;

mod cache;
mod coalesce;
mod collapse;
mod install;
mod payload;

/// A concretizer with `SYMBOLIC_WRITE_ADDRESSES` on, so the Range write
/// strategy is part of the chain and a symbolic-address store concretizes to
/// a candidate *set*. With the option off (the default) Python — and, since
/// angr-9ke6b.194, Rust — uses a Max-only chain for unannotated addresses,
/// which yields a single address and installs no Multi cells. See
/// `AddressConcretizer::write_range_applies`.
fn multi_write_concretizer() -> AddressConcretizer {
    AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    }
}

/// Build a Multi alternative `(addr == cand) -> byte(value)` for tests.
fn make_alt(ctx: &SymContext, addr_var: &RustBV, cand: u64, value: u8) -> MultiAlternative {
    let cand_const = RustBV::concrete(cand as u128, addr_var.width());
    let cond = addr_var.eq(&cand_const, ctx);
    let val = RustBV::concrete(value as u128, 8);
    MultiAlternative::new(cond, val)
}
