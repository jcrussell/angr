//! Unit tests for `SymbolicMemory` and friends.
//!
//! Split into per-feature submodules to keep individual files manageable.
//! Each submodule lives next door to the production code via `super::super`.
//!
//! - `basic`: core ops (concrete load/store, endianness, fork, map_data,
//!   fast-path symbolic, permission/check-executable).
//! - `symbolic`: symbolic stores/loads, partial overlap, wide-store paths,
//!   cross-page concretization, fork isolation, counters/metrics.
//! - `multi`: `MultiPayload` data structure and Phase 1–4.2 multi-cell
//!   collapse/coalesce behavior.
//! - `ite_dedup`: ITE deduplication on loads + address-disjunction hoisting.

mod basic;
mod ite_dedup;
mod multi;
mod symbolic;
