//! An `inspect_dispatch!` entry must reach its callback slot through the
//! `self.with_slot(..)` placeholder — a body that never calls it is rejected
//! rather than silently compiling into a dispatch that fires nothing.
//!
//! Unlike its two sibling function-like macros, this one expands to a plain
//! inherent impl, so the case needs no `pyo3` in scope (see the module header
//! of `tests/compile_fail.rs`).
struct Holder;

angr_macros::inspect_dispatch! {
    Holder =>

    fn fork(&self) -> Result<(), ()> {
        Ok(())
    }
}

fn main() {}
