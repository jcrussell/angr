//! Native SimProcedures and syscall handlers for angr's Rust engine:
//! [`procedures`] (the `NativeSimProcedure` trait, its registry and the libc
//! families) and [`syscalls`] (the `NativeSyscall` trait, its registry and
//! the kernel families). Each falls back to Python through `ProcedureError` /
//! `SyscallError`. The two depend on each other, so they share one crate.
//!
//! **Dependencies.** `angr-symex-core` (its `state`, `memory`, `symbolic`
//! and `arch`), `angr-vex` and the plain-data crates: no `pyo3` (check with
//! `cargo tree -p angr-symex-native -e normal`).
//!
//! The engine (`native/angr`) re-exports [`procedures`] and [`syscalls`], so
//! `crate::procedures::` / `crate::syscalls::` name these items there
//! (angr-mx00p.51.17).
//!
//! # File index
//!
//! | File | Holds |
//! |------|-------|
//! | `procedures/` | The native SimProcedures, their registry and the shared format-string, memory and string helpers; `procedures/mod.rs` documents the design. |
//! | `syscalls/` | The native syscall handlers, their registry and the shared argument helpers; `syscalls/mod.rs` documents the design. |
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![warn(unreachable_pub)]

// `test_submod!` and `silent_default!` are `#[macro_export]`ed by angr_vex
// (their docs live in native/angr-vex/src/lib.rs); `#[macro_use]` ahead of
// the `mod` declarations keeps every invocation unqualified.
#[macro_use]
extern crate angr_vex;

// Keep the `crate::arch::` / `crate::memory::` / `crate::state::` /
// `crate::symbolic::` / `crate::vex::` spellings the modules (and the
// `declare_proc!` expansion's `$crate::` paths) were written against.
use angr_symex_core::{arch, memory, state, symbolic};
use angr_vex as vex;

#[cfg(test)]
use angr_symex_core::test_boundary_values;

pub mod procedures;
pub mod syscalls;
