//! Raw FFI declarations for the libVEX seam (non-default `libvex-ffi` feature).
//!
//! These bindings are **generated** by `build.rs::generate_pyvex_ffi_bindings`
//! (bindgen) from the vendored cffi cdef `native/angr/vendor/pyvex_ffi.h`,
//! which is itself extracted verbatim from `pyvex.vex_ffi.ffi_str` (see
//! `tools/regen-pyvex-ffi-header.py`). Binding against pyvex's own cdef is the
//! parity guarantee: the `VEXLiftResult` / `IRSB` / `IRStmt` / `IRExpr` struct
//! ABI stays in lock step with the `libpyvex.so` we link, so a native lift is
//! byte-for-byte what the Python callback path produces today.
//!
//! This module is *only* the FFI surface — no marshalling. The
//! `NativeLibVEXLifter` that turns a `VEXLiftResult` into `vex::ir::IRSB`
//! lands in a later increment (bd angr-3s5js.3). See
//! docs/advanced-topics/rust_libvex_ffi.rst for the full plan.
//!
//! Safety notes (all inherited from libVEX, see the doc's "Open risks"):
//! - `vex_lift` returns a pointer into libVEX's temporary arena; it is
//!   clobbered on the next `vex_lift` call, so callers must marshal out
//!   immediately.
//! - libVEX global state is not re-entrant; a parallel backend needs a
//!   per-worker lift lock or a single lifter thread.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/pyvex_ffi_bindings.rs"));
