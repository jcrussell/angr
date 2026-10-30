//! The bitvector value layer: [`RustBV`], a concrete `u128` or a clarirs
//! term, and the ops over it. Nothing here names a solver or the engine's
//! state: Z3 lowering is `symbolic::ToZ3`, admission at the trust boundaries
//! is `symbolic::admit_clarirs`, and the ops take no `SymContext`
//! (angr-mx00p.116.6, angr-mx00p.116.7).
//!
//! **Import rule.** Files here reach each other through `super::` only and
//! name no `crate::` path, no crate-root macro and no `z3` / `clarirs_z3`
//! item; external crates (`clarirs_core`, `serde`, `num_bigint`, `log`) are
//! fine. That keeps the directory liftable into its own crate as is.
//!
//! # File index
//!
//! | File | Holds |
//! |------|-------|
//! | `value.rs` | The `RustBV` enum itself (`Concrete` / `Ast`) plus `BVOp`, `FloatPrec`, `FloatOpKind`, the VEX rounding-mode decoder `VexRm`, the serde shadow `RustBVData`, and the constructor / accessor `impl` block (`concrete`, `symbol`, `fresh`, `from_clarirs`). |
//! | `value_ops.rs` | Arithmetic, bitwise, shift/rotate, comparison and structural (`extract` / `concat` / `reverse` / extend) ops, and the construction-time canonicalization rules they run. |
//! | `value_clarirs.rs` | `RustBV::to_clarirs`: the clarirs term a value denotes, with the Bool and bitcount encodings. |
//! | `value_shape.rs` | `RustBV::node`: the op and operands of a compound value, read off a clarirs `Ast` term in `BVOp` vocabulary, plus `leaf_var`. |
//! | `bv_concrete.rs` | Concrete folds over raw `u128`. |
//! | `bv_chunk.rs` | The `<= 16`-byte chunking loop every memory entry point needs to move a `&[u8]` through `RustBV::Concrete`. |
//! | `width_guards.rs` | `MAX_BV_WIDTH` / `check_bv_width` / `check_extract_bounds` — the bound checks the claripy import trust boundary (`claripy_bridge::import`) runs before an externally-chosen width reaches a `RustBV` constructor. |
//! | `symbol_kind.rs` | `SymbolKind` and the `!bool!` name tag that keeps a `BoolS` leaf and a same-named 1-bit `BVS` two variables. |
//! | `names.rs` | The fresh-name counters, keyed by caller, and the watermarks a snapshot carries so a restore never re-mints a restored name. |
//! | `stats.rs` | The value layer's counters (AST-construction shapes, the zext-compare fold, commutative canonicalization), read by `symbolic::stats`. |
//!
//! Test files are pulled in by their parent via `#[path]`. Tests that lower
//! or solve live in `symbolic` (the `*_z3_tests.rs` files wired from
//! `symbolic/to_z3.rs`).

mod bv_chunk;
mod bv_concrete;
pub mod names;
pub mod stats;
pub mod symbol_kind;
mod value;
mod value_clarirs;
mod value_ops;
mod value_shape;
mod width_guards;

pub use bv_chunk::{
    MAX_CONCRETE_CHUNK, MAX_CONCRETE_LOAD_BYTES, check_concrete_load_size,
    load_concrete_bytes_chunked, store_concrete_bytes_chunked, u128_le_byte, u128_to_le_bytes,
};
pub use symbol_kind::{SymbolKind, strip_bool_symbol_name};
pub use value::{BVOp, FloatOpKind, FloatPrec, RustBV, RustBVData, UnsupportedRoundingMode, VexRm};
pub use value_clarirs::{bit_of, bool_inside, bool_of, build_term};
pub use value_ops::{concrete_clz, concrete_ctz};
pub use value_shape::Node;
pub use width_guards::{MAX_BV_WIDTH, check_bv_width, check_extract_bounds};
