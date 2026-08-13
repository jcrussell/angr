//! Unit tests for [`RustSimState`](super::RustSimState).
//!
//! Split per concern to mirror the `state/` source directory's own
//! decomposition, so a change to (say) `state/migration.rs` has an obvious
//! test sibling. Each submodule reaches the state module's private items via
//! `use super::super::*`, exactly as the former single `state_tests.rs` did
//! through `use super::*`.
//!
//! - `helpers`: fixtures shared by more than one submodule.
//! - `basics`: construction, fork identity/lineage, and the per-state config
//!   knobs' defaults + fork isolation.
//! - `history`: `set_detailed_history` / `set_max_history` cap behaviour.
//! - `memory`: state-level load/store, CoW fork isolation, unflushed Multi
//!   cells, and `apply_changes` write chunking.
//! - `filesystem_state`: the filesystem behaviour that needs a whole
//!   `RustSimState` — fork isolation and the `write_stdout`/`fd_buffer`
//!   wrappers. The bare-`FileSystem` CRUD/fd-table surface is
//!   `state/filesystem_tests.rs`'s, not this module's (angr-03vl4.56).
//! - `filesystem_symbolic`: `register_file_content` and the `content_sym`
//!   sharing / serde / length / `read_sym*` contracts.
//! - `filesystem_demote`: demotion of symbolic content to concrete and the
//!   write choke point that refuses rather than dropping it.
//! - `inspection`: the `state.inspect` event ring.
//! - `solver_gate`: the `survives_sat_prune` exploration prune gate and its
//!   undecided-query (Z3 timeout) contract.
//! - `snapshot`: `to_snapshot`/`from_snapshot` round-trip and the serialized
//!   envelope's error paths.
//! - `migration_translate`: `translate_state` across two `SymContext`s.
//! - `migration_snapshot`: cross-context migration via snapshot, including
//!   across a real OS thread.
//! - `merge_scalars`: merge rules for the scalar/watermark fields.
//! - `merge_heap`: merge rules for the heap and CGC allocator state.
//! - `merge_multi_state`: merges decided by comparing more than two branches.
//! - `merge_config`: merge rules for the config-like maps/sets and their
//!   removal tombstones.
//! - `merge_property`: census property test over every `RustSimState` field
//!   carrying `#[merge_policy = "..."]` — the mechanical-policy fields
//!   against their generated `merge_field_<name>` method, the hand-written
//!   ones against a specific expected value reasoned from
//!   `fork.rs::RustSimState::merge`. Additive to (not a replacement for) the
//!   deeper per-family coverage the other `merge_*` modules provide.
//! - `heap`: the heap and CGC allocators themselves (not their merge).
//! - `export`: `export_full`'s dump shape.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69), which
//! angr-c2cv had in turn pulled out of `state.rs`'s in-file `mod tests`.

mod basics;
mod export;
mod filesystem_demote;
mod filesystem_state;
mod filesystem_symbolic;
mod heap;
mod helpers;
mod history;
mod inspection;
mod memory;
mod merge_config;
mod merge_heap;
mod merge_multi_state;
mod merge_property;
mod merge_scalars;
mod migration_snapshot;
mod migration_translate;
mod snapshot;
mod solver_gate;
