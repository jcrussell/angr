//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//!
//! # File index
//!
//! The file count here is much larger than the concept count: two types
//! (`RustBV` and `SymContext`) dominate the directory, and each is split across
//! several files purely to keep every file under a <2000-line cap
//! (angr-a2br.2, angr-7hwz). Those splits are *mechanical* —
//! one file per `impl` block or per slice of one — so a filename does not
//! reliably tell you which concept lives inside. Use this table instead of
//! guessing. Files marked **(z3)** hold the Z3-backed solving surface.
//!
//! ## `RustBV` — the bitvector value
//!
//! The value type and its ops live in `angr_bv` (angr-mx00p.116.7), which
//! names no solver; the items are re-exported here so `crate::symbolic::RustBV`
//! paths are unchanged. What stays here is the solver-facing half:
//!
//! | File | Holds |
//! |------|-------|
//! | `to_z3.rs` | Z3 lowering — the `ToZ3` trait (`to_z3_ast*` / `to_z3_bool*`), fallible (a `LowerError`), through `z3_lower` and its per-thread cache; and `admit_clarirs`, the trust-boundary constructor that refuses a term with no Z3 form. **(z3)** |
//! | `z3_lower.rs` | Lowering a clarirs `AstRef` into the thread-local Z3 context (`lower` / `lower_bv` / `lower_bool`), with the per-thread node cache and `LowerError`. `claripy_bridge::lower` is its PyO3 face. **(z3)** |
//! | `bv_codec.rs` | Codec between concrete values and Z3 BV constants (`context.rs` slice 5). **(z3)** |
//!
//! ## `SymContext` — the solver context
//!
//! `context.rs` keeps the struct, its construction, and the `&self` methods no
//! slice claimed; every other `&self` `impl` block became its own slice file.
//!
//! | File | Holds |
//! |------|-------|
//! | `context.rs` | The `SymContext` struct, its Z3 construction (`new` / `with_timeout` / `new_mock` / `Default` / `Clone`), `DEFAULT_SOLVER_TIMEOUT_MS`, and the `#[path]` wiring for `merge_instrument.rs` + `context_tests/`. Plus the unsliced `&self` surface: the lazily-materializing `solver()` accessor and the assumed-constraint export log (`assumed_constraints_push`, `assumed_local_len`, `truncate_assumed_local`, `get_assumed_constraints`, `assumed_constraint_count`, `export_z3_assertion_ptrs`). |
//! | `context_snapshot.rs` | `SymContextSnapshot` — the serde capture of a context's path constraints, plus its `serde(default)` helper. Data only; `to_snapshot` / `restore_from_snapshot` live in `snapshot_fork_ops.rs`. |
//! | `local_constraints.rs` | `LocalConstraints` — the post-fork local additions (`assumed`, `z3_assertions`, `non_bv_assertions`, `dedup_set`) that share `SymContext`'s one hot-path Mutex. |
//! | `merge_instrument.rs` | `#[cfg(test)]` guarded-assertion counter `SymContext::merge` bumps and `context_tests_merge_prefix` reads. Declared from `context.rs` via `#[path]`, so its module path is still `context::merge_instrument`. |
//! | `bv_id_ops.rs` | Slice 8: unique-id allocation (`next_id`), the symbolic-BV factories, `num_constraints`, and the `SymbolIdRebase` watermark helpers. |
//! | `constraint_ops.rs` | Slice 9: constraint mutation — `add_constraint*`, `add_bv_constraint`, `assume_true`. **(z3)** |
//! | `solving_ops.rs` | Slice 7: the read path — `is_sat`, branch feasibility, `eval*` / `eval_upto*`, extrema queries. |
//! | `transaction_ops.rs` | Slice 10: solver scoping (`push` / `pop` / `try_pop`), timeout accessors, SAT-cache primer. |
//! | `snapshot_fork_ops.rs` | Slice 11: lifecycle — `fork`, `merge`, `to_snapshot` / `restore_from_snapshot`, and the `freeze_into_shared` helper those three share. |
//! | `lineage_ops.rs` | Slice 6: the accessors over the shared-lineage cells (`lineage`, `scope_path`, savepoints). Not purely lineage-scoped: `fold_sharing_walk` reads the assumed-constraint log, and the savepoint push/pop pair records/truncates `local_constraints` — see the module doc. |
//! | `solver_build.rs` | Slice 2: Z3 `Solver` construction and the per-check timing / sampling wrappers. Free functions, no `&self`. **(z3)** |
//! | `lineage.rs` | The shared-lineage solver itself (`SharedLineageSolver`) — one `z3::Solver` shared by a descendant set, with push/pop scope tracking. **(z3)** |
//!
//! ## Identity and trust-boundary guards
//!
//! `symbol_kind.rs` and `width_guards.rs` moved to `angr_bv`.
//!
//! ## Instrumentation and analysis
//!
//! | File | Holds |
//! |------|-------|
//! | `stats.rs` | The process-global counters behind `get_solver_stats()`. Engine-wide despite the name, not solver-only. |
//! | `query_class.rs` | Structural classification of the checks that actually reach Z3, for the cheap-pre-solver-tier spike. **(z3)** |
//! | `sharing.rs` | `ConstraintSharingWalk` — measures `Arc` sharing across a population of constraint DAGs. |
//!
//! ## Tests
//!
//! Test files are not declared in this file. Each `*_tests.rs` sibling is pulled
//! in by its own parent via `#[path]` (so it can reach the parent's private
//! items), and `SymContext`'s tests live in the `context_tests/` subdirectory,
//! themed one file per area and wired from `context.rs`.

mod bv_codec;
mod bv_id_ops;
pub use bv_id_ops::{
    SymbolIdRebase, reserve_symbol_id, symbol_id_rebase_offset, symbol_id_watermark,
};
mod constraint_ops;
mod context;
mod context_snapshot;
pub mod lineage;
mod lineage_ops;
mod local_constraints;
mod query_class;
mod sharing;
mod snapshot_fork_ops;
mod solver_build;
mod solving_ops;
mod stats;
mod to_z3;
mod transaction_ops;
pub(crate) mod z3_lower;

pub(crate) use angr_bv::names;
pub use angr_bv::stats::{
    record_bvop_concat, record_bvop_extract, record_bvop_reverse, record_zext_cmp_collapse,
    record_zext_cmp_trivial_decide,
};
pub use angr_bv::symbol_kind;
pub use angr_bv::{
    BVOp, FloatOpKind, FloatPrec, MAX_BV_WIDTH, MAX_CONCRETE_CHUNK, MAX_CONCRETE_LOAD_BYTES,
    SymBV as RustBV, SymBVData as RustBVData, SymbolKind, UnsupportedRoundingMode, VexRm,
    check_bv_width, check_concrete_load_size, check_extract_bounds, load_concrete_bytes_chunked,
    store_concrete_bytes_chunked, u128_le_byte, u128_to_le_bytes,
};
pub use context::{DEFAULT_SOLVER_TIMEOUT_MS, SymContext};
pub use context_snapshot::SymContextSnapshot;
pub use sharing::{ConstraintSharingStats, ConstraintSharingWalk};
#[cfg(test)]
pub(crate) use solver_build::stop_enumeration_after_for_test;
pub use solving_ops::Enumeration;
pub use stats::{
    VexOpFamily, get_solver_stats, record_concretize_disjunction, record_concretize_read,
    record_concretize_write, record_mem_ite_depth, record_mem_lazy_page_fault, record_mem_load,
    record_mem_load_symbolic_addr, record_mem_store, record_mem_store_symbolic_addr,
    record_symfile_read_native, record_symfile_write_demotion, record_vex_binop, record_vex_qop,
    record_vex_triop, record_vex_unop, reset_solver_stats,
};
pub(crate) use to_z3::lowered_or_warn;
pub use to_z3::{ToZ3, admit_clarirs};
pub use z3_lower::LowerError;
// Used by the claripy export.
pub(crate) use angr_bv::{Node, bool_inside};
