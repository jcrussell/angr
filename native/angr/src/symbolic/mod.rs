//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//! - `RustBVHandle`: Python-facing opaque handle for bypassing claripy
//! - `RustSymbolTable`: Registry mapping handles to RustBV values
//! - `SymbolicIdentityRegistry`: Preserves symbolic identity across Python<->Rust

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
#[cfg(feature = "vex-engine-z3")]
mod z3_ast_ptr;

pub use context::{DEFAULT_SOLVER_TIMEOUT_MS, SymContext, SymContextSnapshot};
pub use handle::RustBVHandle;
#[cfg(test)]
pub use registry::clear_global_registry;
pub use registry::{SymbolInfo, SymbolKind, SymbolicIdentityRegistry, global_registry};
pub use sharing::{ConstraintSharingStats, ConstraintSharingWalk};
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
pub use value::{BVOp, BitWidth, FloatOpKind, FloatPrec, RustBV};
#[cfg(feature = "vex-engine-z3")]
pub use z3_ast_ptr::Z3AstPtr;
