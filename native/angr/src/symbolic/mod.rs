//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//! - `RustBVHandle`: Python-facing opaque handle for bypassing claripy
//! - `RustSymbolTable`: Registry mapping handles to RustBV values
//! - `SymbolicIdentityRegistry`: Preserves symbolic identity across Python<->Rust

mod context;
mod handle;
#[cfg(feature = "vex-engine-z3")]
pub mod lineage;
pub mod registry;
mod sharing;
#[cfg(feature = "vex-engine-z3")]
mod solver_build;
mod stats;
mod table;
mod value;
#[cfg(feature = "vex-engine-z3")]
mod z3_ast_ptr;

pub use context::{ConstraintSyncError, DEFAULT_SOLVER_TIMEOUT_MS, SymContext, SymContextSnapshot};
pub use handle::RustBVHandle;
pub use registry::{SymbolInfo, SymbolicIdentityRegistry, clear_global_registry, global_registry};
pub use sharing::{ConstraintSharingStats, ConstraintSharingWalk};
pub use stats::{
    VexOpFamily, get_solver_stats, record_bvop_concat, record_bvop_extract, record_bvop_reverse,
    record_concretize_disjunction, record_concretize_read, record_concretize_write,
    record_export_sound_clz, record_export_unconstrained_clz, record_export_unconstrained_fp,
    record_mem_ite_depth, record_mem_lazy_page_fault, record_mem_load,
    record_mem_load_symbolic_addr, record_mem_store, record_mem_store_symbolic_addr,
    record_vex_binop, record_vex_qop, record_vex_triop, record_vex_unop, record_zext_cmp_collapse,
    record_zext_cmp_trivial_decide, reset_solver_stats,
};
pub use table::RustSymbolTable;
pub use value::{BVOp, BitWidth, FloatOpKind, FloatPrec, RustBV, Signedness};
#[cfg(feature = "vex-engine-z3")]
pub use z3_ast_ptr::Z3AstPtr;
