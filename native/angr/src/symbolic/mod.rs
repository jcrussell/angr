//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//! - `RustBVHandle`: Python-facing opaque handle for bypassing claripy
//! - `RustSymbolTable`: Registry mapping handles to RustBV values
//! - `SymbolicIdentityRegistry`: Preserves symbolic identity across Python<->Rust
//!
//! # File index
//!
//! The file count here is much larger than the concept count: two types
//! (`RustBV` and `SymContext`) dominate the directory, and each is split across
//! several files purely to keep every file under the <2000-line cap the
//! `angr-a2br.2` / `angr-7hwz` splits adopted. Those splits are *mechanical* —
//! one file per `impl` block or per slice of one — so a filename does not
//! reliably tell you which concept lives inside. Use this table instead of
//! guessing. Files marked **(z3)** are `#[cfg(feature = "vex-engine-z3")]` and
//! vanish from a `--no-default-features` build.
//!
//! ## `RustBV` — the bitvector value
//!
//! | File | Holds |
//! |------|-------|
//! | `value.rs` | The `RustBV` enum itself (`Concrete` / `Symbolic` / `Expression`) plus `BVOp`, `FloatPrec`, `FloatOpKind`, and the constructor / accessor `impl` block. |
//! | `value_ops.rs` | Slice .2: arithmetic, bitwise, shift/rotate, comparison and structural (`extract` / `concat` / `reverse` / extend) ops, and the construction-time canonicalization rules they run. |
//! | `value_z3.rs` | Slice .2: Z3 AST construction — `to_z3_ast*` / `to_z3_bool*`, the memoized integer builder, and the FP builders. **(z3)** |
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
//! | `context.rs` | The `SymContext` struct and `SymContextSnapshot`, `new` / `with_timeout` / `new_mock` / `Default` / `Clone`, `DEFAULT_SOLVER_TIMEOUT_MS`, and the `#[path]` wiring for `context_tests/`. Plus the unsliced `&self` surface: the lazily-materializing `solver()` accessor, the assumed-constraint export log (`assumed_constraints_push`, `assumed_local_len`, `truncate_assumed_local`, `get_assumed_constraints`, `assumed_constraint_count`, `export_z3_assertion_ptrs`), the **non-z3** mock twins of `assume_true` / `assume_false` / `check_branch_feasibility` (their z3 halves live in `constraint_ops.rs` / `solving_ops.rs`), the `LocalConstraints` type, and the `freeze_into_shared` helper. |
//! | `bv_id_ops.rs` | Slice 8: unique-id allocation (`next_id`), the symbolic-BV factories, `num_constraints`, and the `SymbolIdRebase` watermark helpers. |
//! | `constraint_ops.rs` | Slice 9: constraint mutation — `add_constraint*`, `add_bv_constraint`, `assume_true`. **(z3)** |
//! | `solving_ops.rs` | Slice 7: the read path — `is_sat`, branch feasibility, `eval*` / `eval_upto*`, extrema queries. |
//! | `transaction_ops.rs` | Slice 10: solver scoping (`push` / `pop` / `try_pop`), timeout accessors, SAT-cache primer. |
//! | `snapshot_fork_ops.rs` | Slice 11: lifecycle — `fork`, `merge`, `to_snapshot` / `restore_from_snapshot`. |
//! | `lineage_ops.rs` | Slice 6: the accessors over the shared-lineage cells (`lineage`, `scope_path`, savepoints). |
//! | `solver_build.rs` | Slice 2: Z3 `Solver` construction and the per-check timing / sampling wrappers. Free functions, no `&self`. **(z3)** |
//! | `parse.rs` | Slice 4: parsers turning Z3's hex / binary / decimal numeral strings into concrete values. **(z3)** |
//! | `lineage.rs` | The shared-lineage solver itself (`SharedLineageSolver`) — one `z3::Solver` shared by a descendant set, with push/pop scope tracking. **(z3)** |
//!
//! ## Python-facing handles and identity
//!
//! | File | Holds |
//! |------|-------|
//! | `handle.rs` | `RustBVHandle` — the opaque id Python holds instead of round-tripping a claripy AST. |
//! | `table.rs` | `RustSymbolTable` — the handle-id to `RustBV` store behind those handles, plus `BinaryOpError`. |
//! | `registry.rs` | `SymbolicIdentityRegistry` — maps symbols back to their originating Python objects so a claripy -> RustBV -> claripy round trip returns the same AST. |
//! | `z3_ast_ptr.rs` | `Z3AstPtr` — typed, refcounted wrapper over a raw `Z3_ast` at the claripy FFI boundary. **(z3)** |
//! | `width_guards.rs` | `MAX_BV_WIDTH` / `check_bv_width` / `check_extract_bounds` — the bound checks both external trust boundaries (`claripy_bridge::import`, `solver::handle_api`) run before an externally-chosen width reaches a `RustBV` constructor. |
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
#[cfg(feature = "vex-engine-z3")]
mod bv_codec;
mod bv_concrete;
mod bv_id_ops;
pub use bv_id_ops::{
    SymbolIdRebase, reserve_symbol_id, symbol_id_rebase_offset, symbol_id_watermark,
};
#[cfg(feature = "vex-engine-z3")]
mod constraint_ops;
mod context;
mod handle;
#[cfg(feature = "vex-engine-z3")]
pub mod lineage;
mod lineage_ops;
#[cfg(feature = "vex-engine-z3")]
mod parse;
/// Dev-only re-export of the pure Z3-numeral-string decoders for cargo-fuzz
/// targets (angr-qwyti.9). These parse untrusted-shaped `&str` into concrete
/// values with no `SymContext` / Z3 handle, so they fuzz hermetically. Gated
/// behind `fuzzing` so the normal build keeps them `pub(super)`.
#[cfg(all(feature = "vex-engine-z3", feature = "fuzzing"))]
pub mod fuzz_exports {
    // Thin `pub` wrappers over the `pub(super)` numeral-string decoders so the
    // cargo-fuzz targets can call them without widening the parsers' own
    // visibility in a normal build.
    pub fn parse_wide_hex_low128(s: &str) -> Option<u128> {
        super::parse::parse_wide_hex_low128(s)
    }
    pub fn parse_wide_binary_low128(s: &str) -> Option<u128> {
        super::parse::parse_wide_binary_low128(s)
    }
    pub fn parse_hex_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
        super::parse::parse_hex_to_bytes(s, width)
    }
    pub fn parse_binary_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
        super::parse::parse_binary_to_bytes(s, width)
    }
    pub fn parse_decimal_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
        super::parse::parse_decimal_to_bytes(s, width)
    }
}
#[cfg(feature = "vex-engine-z3")]
mod query_class;
pub mod registry;
mod sharing;
mod snapshot_fork_ops;
#[cfg(feature = "vex-engine-z3")]
mod solver_build;
mod solving_ops;
mod stats;
mod table;
mod transaction_ops;
mod value;
mod value_ops;
#[cfg(feature = "vex-engine-z3")]
mod value_z3;
mod width_guards;
#[cfg(feature = "vex-engine-z3")]
mod z3_ast_ptr;

pub use bv_chunk::{
    MAX_CONCRETE_CHUNK, MAX_CONCRETE_LOAD_BYTES, check_concrete_load_size,
    load_concrete_bytes_chunked, store_concrete_bytes_chunked, u128_le_byte, u128_to_le_bytes,
};
pub use context::{DEFAULT_SOLVER_TIMEOUT_MS, SymContext, SymContextSnapshot};
pub use handle::RustBVHandle;
#[cfg(test)]
pub use registry::clear_global_registry;
pub use registry::{SymbolInfo, SymbolKind, SymbolicIdentityRegistry, global_registry};
pub use sharing::{ConstraintSharingStats, ConstraintSharingWalk};
pub use solving_ops::Enumeration;
pub use stats::{
    VexOpFamily, get_solver_stats, record_bvop_concat, record_bvop_extract, record_bvop_reverse,
    record_claripy_ast_cache, record_concretize_disjunction, record_concretize_read,
    record_concretize_write, record_export_sound_clz, record_export_unconstrained_clz,
    record_export_unconstrained_fp, record_mem_ite_depth, record_mem_lazy_page_fault,
    record_mem_load, record_mem_load_symbolic_addr, record_mem_store,
    record_mem_store_symbolic_addr, record_symfile_read_native, record_symfile_write_demotion,
    record_vex_binop, record_vex_qop, record_vex_triop, record_vex_unop, record_zext_cmp_collapse,
    record_zext_cmp_trivial_decide, reset_solver_stats,
};
pub use table::{BinaryOpError, RustSymbolTable};
pub use value::{BVOp, FloatOpKind, FloatPrec, RustBV};
// angr-0jh0j.9: `claripy_bridge::export`'s concrete-operand fold for
// `BVOp::Clz`/`BVOp::Ctz` shares these with `RustBV::{clz,ctz}_into` rather
// than keeping its own copy of the width adjustment. `claripy_bridge` is the
// re-export's only consumer and is itself `#[cfg(feature = "vex-engine")]`, so
// the gate must match or the no-default-features build warns unused-imports
// (angr-9hkr6).
#[cfg(feature = "vex-engine")]
pub(crate) use value_ops::{concrete_clz, concrete_ctz};
pub use width_guards::{MAX_BV_WIDTH, check_bv_width, check_extract_bounds};
#[cfg(feature = "vex-engine-z3")]
pub use z3_ast_ptr::Z3AstPtr;
