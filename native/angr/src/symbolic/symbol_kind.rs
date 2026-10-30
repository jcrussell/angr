//! The sort of a named leaf, carried in its Rust-side name.
//!
//! `RustBV` has no Bool sort: a claripy `BoolS` leaf is a width-1 value whose
//! term is built from its *name* (`RustBV::symbol`), so the name carries
//! the sort. [`SymbolKind::rust_symbol_name`] tags a Bool leaf's name and
//! `strip_bool_symbol_name` reads the tag back, so `BVS("x", 1)` and
//! `BoolS("x")` stay two variables (angr-9ke6b.38, angr-9ke6b.223).

use std::borrow::Cow;

/// Claripy-side sort of a named leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// A claripy `BVS` bitvector leaf.
    BitVector,
    /// A claripy `BoolS` boolean leaf.
    Bool,
}

/// Prefix that namespaces the Rust-side name of a claripy `Bool` leaf.
///
/// Chosen to contain `!`, which claripy's own auto-renamer never emits (it
/// appends `_<counter>_<width>`), so a mangled name can only collide with a
/// caller that passes `explicit_name=True` and a literal `!bool!` prefix.
const BOOL_NAME_PREFIX: &str = "!bool!";

impl SymbolKind {
    /// Map a claripy symbol name to the name Rust uses for it.
    ///
    /// The name has to carry the sort, or `BVS("x", 1, explicit_name=True)`
    /// and `BoolS("x", explicit_name=True)` collapse to one variable, and
    /// constraining one silently constrains the other (angr-9ke6b.223).
    /// `BitVector` is the identity, so every bitvector name is unchanged.
    pub fn rust_symbol_name(self, name: &str) -> Cow<'_, str> {
        match self {
            SymbolKind::BitVector => Cow::Borrowed(name),
            SymbolKind::Bool => Cow::Owned(format!("{BOOL_NAME_PREFIX}{name}")),
        }
    }
}

/// Recover the claripy name from a Bool leaf's Rust name, or `None` for a
/// bitvector leaf.
///
/// Inverse of [`SymbolKind::rust_symbol_name`], and the reason the tag lives
/// in the name rather than in a `RustBV` field: `RustBV::symbol` builds a
/// leaf from its name alone — from the constructors and from the legacy
/// snapshot record — so the sort survives a snapshot round-trip for free. The
/// returned name is claripy's own, so the leaf is the very `BoolS` Python
/// holds.
pub(crate) fn strip_bool_symbol_name(name: &str) -> Option<&str> {
    name.strip_prefix(BOOL_NAME_PREFIX)
}

test_submod!("symbol_kind_tests.rs" => tests);
