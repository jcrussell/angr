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
//! | File | Holds |
//! |------|-------|
//! | `value.rs` | The `RustBV` enum itself (`Concrete` / `Ast`) plus `BVOp`, `FloatPrec`, `FloatOpKind`, the VEX rounding-mode decoder `VexRm`, and the constructor / accessor `impl` block. |
//! | `value_ops.rs` | Slice .2: arithmetic, bitwise, shift/rotate, comparison and structural (`extract` / `concat` / `reverse` / extend) ops, and the construction-time canonicalization rules they run. |
//! | `value_clarirs.rs` | `RustBV::to_clarirs`: the clarirs term a value denotes, with the Bool and bitcount encodings its Z3 lowering uses. |
//! | `value_shape.rs` | `RustBV::node`: the op and operands of a compound value, read off a clarirs `Ast` term in `BVOp` vocabulary, plus `leaf_var`. |
//! | `value_z3.rs` | Z3 lowering — `to_z3_ast*` / `to_z3_bool*`, fallible (a `LowerError`), through `z3_lower` with the per-value memo. **(z3)** |
//! | `z3_lower.rs` | Lowering a clarirs `AstRef` into the thread-local Z3 context (`lower` / `lower_bv` / `lower_bool`), with the per-thread node cache and `LowerError`. `claripy_bridge::lower` is its PyO3 face. **(z3)** |
//! | `bv_concrete.rs` | Z3-independent concrete folds over raw `u128`. Deliberately Z3-free so concrete paths still build with no z3. |
//! | `bv_codec.rs` | Codec between concrete values and Z3 BV constants (`context.rs` slice 5). **(z3)** |
//! | `bv_chunk.rs` | The `<= 16`-byte chunking loop every memory entry point needs to move a `&[u8]` through `RustBV::Concrete`. |
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
//! | `names.rs` | The fresh-name counters, keyed by caller, and the watermarks a snapshot carries so a restore never re-mints a restored name. |
//! | `lineage.rs` | The shared-lineage solver itself (`SharedLineageSolver`) — one `z3::Solver` shared by a descendant set, with push/pop scope tracking. **(z3)** |
//!
//! ## Identity and trust-boundary guards
//!
//! | File | Holds |
//! |------|-------|
//! | `symbol_kind.rs` | `SymbolKind` and the `!bool!` name tag that keeps a `BoolS` leaf and a same-named 1-bit `BVS` two variables. |
//! | `width_guards.rs` | `MAX_BV_WIDTH` / `check_bv_width` / `check_extract_bounds` — the bound checks the claripy import trust boundary (`claripy_bridge::import`) runs before an externally-chosen width reaches a `RustBV` constructor. |
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

mod bv_chunk;
mod bv_codec;
mod bv_concrete;
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
pub(crate) mod names;
mod query_class;
mod sharing;
mod snapshot_fork_ops;
mod solver_build;
mod solving_ops;
mod stats;
pub mod symbol_kind;
mod transaction_ops;
mod value;
mod value_clarirs;
mod value_ops;
mod value_shape;
mod value_z3;
mod width_guards;
pub(crate) mod z3_lower;

pub use bv_chunk::{
    MAX_CONCRETE_CHUNK, MAX_CONCRETE_LOAD_BYTES, check_concrete_load_size,
    load_concrete_bytes_chunked, store_concrete_bytes_chunked, u128_le_byte, u128_to_le_bytes,
};
pub use context::{DEFAULT_SOLVER_TIMEOUT_MS, SymContext};
pub use context_snapshot::SymContextSnapshot;
pub use sharing::{ConstraintSharingStats, ConstraintSharingWalk};
#[cfg(test)]
pub(crate) use solver_build::stop_enumeration_after_for_test;
pub use solving_ops::Enumeration;
pub use stats::{
    VexOpFamily, get_solver_stats, record_bvop_concat, record_bvop_extract, record_bvop_reverse,
    record_concretize_disjunction, record_concretize_read, record_concretize_write,
    record_mem_ite_depth, record_mem_lazy_page_fault, record_mem_load,
    record_mem_load_symbolic_addr, record_mem_store, record_mem_store_symbolic_addr,
    record_symfile_read_native, record_symfile_write_demotion, record_vex_binop, record_vex_qop,
    record_vex_triop, record_vex_unop, record_zext_cmp_collapse, record_zext_cmp_trivial_decide,
    reset_solver_stats,
};
pub use symbol_kind::SymbolKind;
pub use value::{BVOp, FloatOpKind, FloatPrec, RustBV, UnsupportedRoundingMode, VexRm};
// Used by the claripy export.
pub(crate) use value_clarirs::bool_inside;
pub(crate) use value_shape::Node;
pub(crate) use value_z3::lowered_or_warn;
pub use width_guards::{MAX_BV_WIDTH, check_bv_width, check_extract_bounds};
pub use z3_lower::LowerError;
