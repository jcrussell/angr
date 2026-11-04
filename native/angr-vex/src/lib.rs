//! The VEX layer of angr's native engine: VEX IR types, the pyvex opcode
//! map, the op semantics over `angr_bv::SymBV`, the clean calls (flag
//! calculations) and the native dirty helpers (OUT port access; see
//! `dirty.rs`'s header for why CPUID/RDTSCP, RDTSC and IN are deliberately not
//! among them). The lifting interface is the [`VEXLifter`] trait; the libVEX
//! implementation lives in the engine shell (`libvex_lifter`).
//!
//! **Dependencies.** `angr-bv`, `clarirs-core` and the plain-data crates
//! only: no `z3`, `clarirs-z3` or `pyo3` (check with
//! `cargo tree -p angr-vex -e normal`). `clarirs-z3` is a dev-dependency: the
//! op tests solve through it (`ops/test_helpers.rs`).
//!
//! The engine (`native/angr`) re-exports this crate as `vex`, so
//! `crate::vex::` and `rustylib::vex::` paths name these items
//! (angr-mx00p.51.10). It also owns the crate-root macros [`test_submod!`]
//! and [`silent_default!`], which every engine crate invokes unqualified.
//!
//! # File index
//!
//! | File | Holds |
//! |------|-------|
//! | `ir/` | The VEX IR: statements, expressions, types, `IROp`, the guest arch descriptor, and the pyvex IRSB deserialization. |
//! | `lifter.rs` | The `VEXLifter` trait and `LiftError`. |
//! | `opcode_map.rs` | pyvex name -> `IROp` / `IRType` / endness / jumpkind parsing; the float and vector op tables are `opcode_map_float.rs` and `opcode_map_vector.rs`. |
//! | `ops/` | `VEXOps`: the op semantics, by family (`int_arith`, `float_arith`, `conversions`, `vec_*`, ...), plus `classify.rs` (`VexOpFamily`) and `error.rs` (`OpError`). |
//! | `ccall/` | The clean calls: per-arch condition-code and flag helpers. |
//! | `dirty.rs` | The native dirty helpers (`DirtyHelperDispatch`). |
//! | `transcendentals.rs` | The x87 transcendental ops. |
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![warn(unreachable_pub)]

#[doc(hidden)]
pub use log as __log;

/// Declare a test-only submodule that lives in a sibling file.
///
/// Every `#[cfg(test)] mod tests;` in this crate points at an out-of-line
/// `*_tests.rs` (or a `*_tests/` directory of scoped test files) and has to
/// re-`allow` the two test-only clippy lints, because each test module's own
/// `deny` overrides the crate-wide `cfg_attr(test, allow(..))` at the top of
/// the crate root. That is four attributes of identical boilerplate per site,
/// repeated at every out-of-line test module (angr-12jjk.16); this macro is
/// the single place the justification wording lives.
///
/// ```ignore
/// test_submod!("stash_tests.rs" => tests);        // explicit `#[path]`
/// test_submod!(tests_core);                       // default `tests_core.rs`
/// test_submod!("context_tests/constraints.rs" => context_tests_constraints);
/// test_submod!(tests_int_arith);
/// ```
///
/// Defined before the crate's `mod` declarations so legacy textual macro scope
/// reaches every module; invoke it unqualified. The engine crate pulls it in
/// with `#[macro_use] extern crate angr_vex;`, so the same holds there.
///
/// It is the convention for **every** out-of-line test submodule, not just the
/// ones where the `allow` is load-bearing. The payload only bites under a host
/// module's own `#![deny(clippy::unwrap_used, clippy::expect_used)]` (for
/// example `exploration::shadow_probe`), so the hand-written spelling compiles fine
/// everywhere else — which is exactly why a partial sweep keeps looking
/// complete (angr-c7xno.48, angr-03vl4.8, angr-03vl4.89). The invariant a
/// future audit checks is therefore a simple one:
/// outside this macro's own definition, `native/angr/src/**` and
/// `native/angr-vex/src/**` contain no
/// `#[cfg(test)] #[path = ...] mod ...;`.
#[macro_export]
macro_rules! test_submod {
    ($file:literal => $name:ident) => {
        #[cfg(test)]
        #[path = $file]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
    ($name:ident) => {
        #[cfg(test)]
        #[allow(
            clippy::unwrap_used,
            clippy::expect_used,
            reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
        )]
        mod $name;
    };
}

/// Take a fallible value's success payload, or log the failure and fall back to
/// `$default`.
///
/// A site that discards an error and continues with a degraded result should
/// say so in the log, and one that risks a wrong answer should log at `warn`.
/// Writing that out by hand is four lines of `unwrap_or_else` boilerplate, so
/// the tempting silent form (`.unwrap_or(0)`, `let _ = ...`) keeps winning
/// (angr-91vj9.6). This macro makes the correct form the shorter one: the
/// category picks the log level, and the invocation itself is greppable.
///
/// `cat_b` logs at `debug` (fallback with loss), `cat_c` at `warn`
/// (wrong-answer risk). There is deliberately no `cat_a` arm: expected control
/// flow needs no log, so it stays a plain `match`/`unwrap_or` with a comment.
/// An unrecognized category is a compile error rather than a silent downgrade.
///
/// ```ignore
/// // Option<T>: `None` is the failure, and there is nothing to `Display`.
/// silent_default!(cat_c, maybe_symbolic_addr, 0, "address unavailable ({context})");
///
/// // Result<T, E>: `|err|` names a binding for the error, in scope for the
/// // message. Not mentioning it is an unused-variable error under the crate's
/// // `-D warnings`, so the cause cannot be dropped. (The binder must be
/// // written by the caller rather than injected by the macro: macro_rules
/// // hygiene would put a macro-defined `err` out of the message's scope.)
/// silent_default!(cat_b, parse(s), Default::default(), |err| "bad input: {err}");
/// ```
///
/// Defined before the crate's `mod` declarations so legacy textual macro scope
/// reaches every module; invoke it unqualified. The engine crate pulls it in
/// with `#[macro_use] extern crate angr_vex;`, so the same holds there.
#[macro_export]
macro_rules! silent_default {
    // `Result<T, E>` form. Listed first: its `|err|` prefix would otherwise be
    // swallowed by the `$($msg:tt)+` of the `Option` arm below.
    ($cat:ident, $fallible:expr, $default:expr, |$err:ident| $($msg:tt)+) => {
        match $fallible {
            Ok(value) => value,
            Err($err) => {
                $crate::silent_default!(@log $cat, $($msg)+);
                $default
            }
        }
    };
    // `Option<T>` form.
    ($cat:ident, $fallible:expr, $default:expr, $($msg:tt)+) => {
        match $fallible {
            Some(value) => value,
            None => {
                $crate::silent_default!(@log $cat, $($msg)+);
                $default
            }
        }
    };
    // Internal: the category selects the log level. Keep in sync with the
    // cat-a/b/c definitions in the `tools/audit_silent_fallback.py` docstring.
    (@log cat_b, $($msg:tt)+) => { $crate::__log::debug!($($msg)+) };
    (@log cat_c, $($msg:tt)+) => { $crate::__log::warn!($($msg)+) };
}

test_submod!("silent_default_tests.rs" => silent_default_tests);

pub mod ccall;
pub mod dirty;
pub mod ir;
mod lifter;
pub mod opcode_map;
pub(crate) mod opcode_map_float;
pub(crate) mod opcode_map_vector;
pub mod ops;
pub mod transcendentals;

/// pyvex `_lift`'s `max_bytes` default (`pyvex/lifting/libvex.py`): the largest
/// byte window any single-block lift consumes.
///
/// Shared so every lift path agrees on one budget. libVEX stops at the block
/// boundary, so trailing bytes beyond the block are simply unused — reading up
/// to this many is always safe, and reading *fewer* is what goes wrong: the SMC
/// dirty-bytes probe in `VEXInterpreter::get_or_lift_block` used to cap at one
/// page, so a block needing fresh post-store bytes past offset 4096 would have
/// been lifted from stale image bytes (angr-sqfj8.67).
pub const VEX_MAX_BYTES: usize = 5000;

pub use dirty::{DirtyHelperDispatch, DirtyHelperResult};
pub use ir::*;
pub use lifter::{LiftError, VEXLifter};
pub use opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type};
pub use ops::{OpError, VEXOps, VexOpFamily, iropclass};
