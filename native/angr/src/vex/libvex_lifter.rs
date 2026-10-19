//! Native libVEX lifter.
//!
//! `NativeLibVEXLifter` implements the [`VEXLifter`] trait by calling the
//! `vex_lift` shim in the `libpyvex` the running pyvex already loaded (the
//! *same* object pyvex calls through cffi) and marshalling the returned C
//! `VEXLiftResult->irsb` into a Rust [`IRSB`]. The symbol and the C struct
//! layouts come from upstream's [`crate::ailment::vex_ffi`], which resolves
//! `vex_lift` at runtime (`init_symbols`), so the extension has no link-time
//! dependency on `libpyvex`. See `docs/advanced-topics/rust_libvex_ffi.rst`.
//!
//! A lifter is built from the Python `archinfo.Arch` (`from_py_arch`): the
//! guest tag comes from `arch.vex_arch` and the `VexArchInfo` from
//! `arch.vex_archinfo`, the same inputs pyvex's `LibVEXLifter._lift` passes,
//! so endianness, hwcaps and `x86_cr0` follow the project's arch instead of
//! being guessed from the Rust [`VexArch`]. [`LiftSettings`] carries the
//! engine's opt level, `strict_block_end`, `cross_insn_opt` and
//! self-modifying-code setting, so a lift matches what the Python callback's
//! `project.factory.block(addr)` would produce.
//!
//! The marshalling mirrors `pyvex_reader.rs` node-for-node — same `vex::ir`
//! target shape — but reads the raw C structs instead of pyvex objects. Op tags go through `vexop::op_name` and the other
//! enum tags through the name tables below, so the existing string parsers in
//! `opcode_map.rs` are reused verbatim (DRY).
//!
//! Safety invariants (see the doc's "Open risks"):
//! - `vex_lift` returns a pointer into libVEX's temporary arena; it is clobbered
//!   on the next `vex_lift` call. We marshal the whole `IRSB` into owned Rust
//!   types *before* releasing the lift lock, so nothing outlives the arena.
//! - libVEX global state (`vex_control`, the arena) is not re-entrant. The
//!   process-wide lock behind `lift_lock` serializes every lift the engine
//!   makes, native or through the Python callback.
//!
//! **Panic policy (angr-9ke6b.212):** the guest bytes handed to
//! [`NativeLibVEXLifter::lift_at`] are untrusted and every marshalling failure
//! returns a [`LiftError`], which the caller turns into a Python-lift fallback.
//!
//! The `deny` below follows the crate [Panic policy](crate#panic-policy)
//! (angr-qwyti.11).
#![deny(clippy::unwrap_used, clippy::expect_used)]
// Every marshalling helper below is an `unsafe fn` whose entire body walks the
// libVEX arena. Wrapping each individual deref in its own `unsafe {}` block adds
// only rightward drift here — the whole module is a single unsafe domain gated
// by `LIFT_LOCK` and the top-level SAFETY contract in `lift`. Scope the
// edition-2024 lint to this module rather than sprinkling per-op blocks.
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, c_int};
use std::ptr;

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::ir::{
    DirtyFx, Endness, IRCallee, IRConst, IRDirty, IRExpr, IRLoadGOp, IRRegArray, IRSB, IRStmt,
    IRType, JumpKind, MBusEvent, TypeEnv, VexArch,
};
use super::lifter::{LiftError, VEXLifter};
use super::opcode_map::{parse_jumpkind, parse_opcode, parse_type_or_log};
use crate::ailment::vex_ffi as ffi;
use crate::ailment::vexop;

/// VEX's `IRTemp_INVALID` sentinel (an absent temp, e.g. the `oldHi` slot of a
/// single-width CAS or an untyped Dirty result).
const IRTEMP_INVALID: u32 = 0xFFFF_FFFF;

/// pyvex `_lift` default (`pyvex/lifting/libvex.py`). Its `max_bytes`
/// companion is the crate-shared `super::VEX_MAX_BYTES`.
const VEX_MAX_INSTRUCTIONS: u32 = 99;

/// `VexRegisterUpdates::VexRegUpdUnwindregsAtMemAccess`, the `px_control`
/// pyvex passes when `cross_insn_opt` is on (angr's default).
const VEX_REG_UPD_UNWINDREGS_AT_MEM_ACCESS: u32 = 0x702;

/// `VexRegisterUpdates::VexRegUpdLdAllregsAtEachInsn`, the `px_control`
/// pyvex passes when `cross_insn_opt` is off.
const VEX_REG_UPD_LD_ALLREGS_AT_EACH_INSN: u32 = 0x705;

// Enum tags `ailment::vex_ffi` does not carry, from `pyvex/pyvex/vex_ffi.py`.
// `test_native_lift_parity.py` checks every one against pyvex's cffi library.
const IJK_BASE: u32 = 0x1A00;
const IFX_READ: u32 = 0x1B01;
const IFX_WRITE: u32 = 0x1B02;
const IFX_MODIFY: u32 = 0x1B03;
const ILGOP_IDENT_V128: u32 = 0x1D01;
const ILGOP_IDENT64: u32 = 0x1D02;
const ILGOP_IDENT32: u32 = 0x1D03;
const ILGOP_IDENT16: u32 = 0x1D04;
const ILGOP_IDENT8: u32 = 0x1D05;
const ILGOP_16UTO32: u32 = 0x1D06;
const ILGOP_16STO32: u32 = 0x1D07;
const ILGOP_8UTO32: u32 = 0x1D08;
const ILGOP_8STO32: u32 = 0x1D09;

/// `IRJumpKind` names, indexed by `tag - IJK_BASE` (`Ijk_INVALID` first).
const IJK_NAMES: &[&str] = &[
    "Ijk_INVALID",
    "Ijk_Boring",
    "Ijk_Call",
    "Ijk_Ret",
    "Ijk_ClientReq",
    "Ijk_Yield",
    "Ijk_EmWarn",
    "Ijk_EmFail",
    "Ijk_NoDecode",
    "Ijk_MapFail",
    "Ijk_InvalICache",
    "Ijk_FlushDCache",
    "Ijk_NoRedir",
    "Ijk_SigILL",
    "Ijk_SigTRAP",
    "Ijk_SigSEGV",
    "Ijk_SigBUS",
    "Ijk_SigFPE",
    "Ijk_SigFPE_IntDiv",
    "Ijk_SigFPE_IntOvf",
    "Ijk_Privileged",
    "Ijk_Sys_syscall",
    "Ijk_Sys_int",
    "Ijk_Sys_int32",
    "Ijk_Sys_int128",
    "Ijk_Sys_int129",
    "Ijk_Sys_int130",
    "Ijk_Sys_int145",
    "Ijk_Sys_int210",
    "Ijk_Sys_sysenter",
    "Ijk_Extension",
];

/// `IRType` names, indexed by `tag - ffi::ITY_INVALID` (`Ity_INVALID` first).
const ITY_NAMES: &[&str] = &[
    "Ity_INVALID",
    "Ity_I1",
    "Ity_I8",
    "Ity_I16",
    "Ity_I32",
    "Ity_I64",
    "Ity_I128",
    "Ity_F16",
    "Ity_F32",
    "Ity_F64",
    "Ity_D32",
    "Ity_D64",
    "Ity_D128",
    "Ity_F128",
    "Ity_V128",
    "Ity_V256",
    "Ity_V512",
];

/// Name of an enum tag in a table that starts at `base`.
fn tag_name(table: &'static [&'static str], base: u32, tag: u32) -> Option<&'static str> {
    let idx = tag.checked_sub(base)?;
    table.get(usize::try_from(idx).ok()?).copied()
}

/// Serializes every libVEX lift the engine makes — libVEX global state is not
/// re-entrant. Taken through [`lift_lock`] only.
static LIFT_LOCK: ReentrantMutex<()> = parking_lot::const_reentrant_mutex(());

/// Take the process-wide libVEX lift lock (angr-mx00p.45).
///
/// Both of the engine's lift paths hold it: [`NativeLibVEXLifter`]'s FFI
/// call, and the Python `lift_block` callback in
/// `VEXInterpreter::get_or_lift_block`, whose pyvex lift enters the same
/// libVEX with the GIL released inside cffi. The lock order is lift lock,
/// then GIL: this never *waits* for the lock while holding the GIL, so a
/// thread that holds the lock may block on the GIL (the callback does)
/// without deadlocking against a thread that holds the GIL and wants the
/// lock. Reentrant so a callback that lifts again on the same thread does
/// not self-deadlock. pyvex's own `pyvex.lift` callers outside the engine
/// still serialize only on pyvex's Python lock.
pub(crate) fn lift_lock() -> ReentrantMutexGuard<'static, ()> {
    if let Some(guard) = LIFT_LOCK.try_lock() {
        return guard;
    }
    // Contended: wait with the GIL released. `try_attach` reuses the GIL when
    // this thread holds it (and otherwise takes it, harmless while no lift
    // lock is held); it returns `None` when there is no interpreter at all.
    // The guard is not `Send`, so it cannot leave `detach`'s closure: forget
    // it there and rebuild it below.
    let waited = Python::try_attach(|py| py.detach(|| std::mem::forget(LIFT_LOCK.lock())));
    match waited {
        // SAFETY: `detach` runs its closure on this thread, so the forgotten
        // guard left this thread holding exactly one acquisition, which the
        // rebuilt guard now owns.
        Some(()) => unsafe { LIFT_LOCK.make_guard_unchecked() },
        None => LIFT_LOCK.lock(),
    }
}

/// The `VexArchInfo` fields pyvex fills from `arch.vex_archinfo`, kept as
/// plain data so the lifter is `Send` (the C struct holds a raw cache pointer).
/// Mirrors upstream's `convert_vex::build_archinfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArchInfo {
    hwcaps: u32,
    endness: c_int,
    ppc_icache_line_sz_b: c_int,
    ppc_dcbz_sz_b: u32,
    ppc_scv_supported: u8,
    ppc_dcbzl_sz_b: u32,
    arm64_d_min_line_lg2_sz_b: u32,
    arm64_i_min_line_lg2_sz_b: u32,
    arm64_cache_block_size: u8,
    arm64_requires_fallback_llsc: u8,
    x86_cr0: u32,
}

impl ArchInfo {
    fn to_ffi(self) -> ffi::VexArchInfo {
        ffi::VexArchInfo {
            hwcaps: self.hwcaps,
            endness: self.endness,
            hwcache_info: ffi::VexCacheInfo {
                num_levels: 0,
                num_caches: 0,
                caches: ptr::null_mut(),
                icaches_maintain_coherence: 1,
            },
            ppc_icache_line_sz_b: self.ppc_icache_line_sz_b,
            ppc_dcbz_sz_b: self.ppc_dcbz_sz_b,
            ppc_scv_supported: self.ppc_scv_supported,
            ppc_dcbzl_sz_b: self.ppc_dcbzl_sz_b,
            arm64_d_min_line_lg2_sz_b: self.arm64_d_min_line_lg2_sz_b,
            arm64_i_min_line_lg2_sz_b: self.arm64_i_min_line_lg2_sz_b,
            arm64_cache_block_size: self.arm64_cache_block_size,
            arm64_requires_fallback_llsc: self.arm64_requires_fallback_llsc,
            x86_cr0: self.x86_cr0,
        }
    }
}

/// libVEX `VexArch` guest tag for an `archinfo` `vex_arch` name (mirrors
/// upstream's `convert_vex::vex_arch_int`, which is private).
fn vex_arch_tag(name: &str) -> Option<u32> {
    Some(match name {
        "VexArchX86" => 0x401,
        "VexArchAMD64" => 0x402,
        "VexArchARM" => 0x403,
        "VexArchARM64" => 0x404,
        "VexArchPPC32" => 0x405,
        "VexArchPPC64" => 0x406,
        "VexArchS390X" => 0x407,
        "VexArchMIPS32" => 0x408,
        "VexArchMIPS64" => 0x409,
        "VexArchRISCV64" => 0x40B,
        _ => return None,
    })
}

/// Native libVEX lifter backed by `libpyvex`'s `vex_lift` shim, configured
/// for one guest arch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeLibVEXLifter {
    guest: u32,
    info: ArchInfo,
    settings: LiftSettings,
}

/// The lift settings angr's `VEXLifter.lift_vex` resolves for a stateless
/// `project.factory.block(addr)` — what the Python `lift_block` callback
/// lifts with — so a native lift passes libVEX the same configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiftSettings {
    /// The engine's `_default_opt_level`, used when no override applies.
    pub default_opt_level: i32,
    /// The engine's `default_strict_block_end`.
    pub strict_block_end: bool,
    /// `cross_insn_opt`; on for a stateless lift.
    pub cross_insn_opt: bool,
    /// The engine's `selfmodifying_code`, which forces opt level 0.
    pub selfmodifying_code: bool,
}

impl Default for LiftSettings {
    /// angr's `VEXLifter` defaults.
    fn default() -> Self {
        LiftSettings {
            default_opt_level: 1,
            strict_block_end: false,
            cross_insn_opt: true,
            selfmodifying_code: false,
        }
    }
}

impl NativeLibVEXLifter {
    /// Build a lifter from a Python `archinfo.Arch`.
    ///
    /// Resolves `vex_lift` in the loaded `libpyvex` (`vex_ffi::init_symbols`)
    /// and raises `RuntimeError` when it cannot be found, `ValueError` for an
    /// arch libVEX has no guest for.
    pub fn from_py_arch(arch: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = arch.py();
        // pyvex must be imported for init_symbols to find libpyvex; importing
        // it also runs libVEX's one-time `vex_init`.
        py.import("pyvex")?;
        ffi::init_symbols(py);
        if ffi::vex_lift_fn().is_none() {
            return Err(PyRuntimeError::new_err(
                "native lift: vex_lift was not found in the loaded libpyvex",
            ));
        }
        let name: String = arch.getattr("vex_arch")?.extract()?;
        let guest = vex_arch_tag(&name).ok_or_else(|| {
            PyValueError::new_err(format!("native lift: no libVEX guest for {name}"))
        })?;
        let vai = arch.getattr("vex_archinfo")?;
        // A key archinfo leaves out is zero, as pyvex's cffi struct
        // initialization makes it; any other lookup error propagates.
        let geti = |k: &str| -> PyResult<i64> {
            match vai.get_item(k) {
                Ok(v) => Ok(v.extract::<Option<i64>>()?.unwrap_or(0)),
                Err(e) if e.is_instance_of::<PyKeyError>(py) => Ok(0),
                Err(e) => Err(e),
            }
        };
        let info = ArchInfo {
            hwcaps: geti("hwcaps")? as u32,
            endness: geti("endness")? as c_int,
            ppc_icache_line_sz_b: geti("ppc_icache_line_szB")? as c_int,
            ppc_dcbz_sz_b: geti("ppc_dcbz_szB")? as u32,
            ppc_scv_supported: geti("ppc_scv_supported")? as u8,
            ppc_dcbzl_sz_b: geti("ppc_dcbzl_szB")? as u32,
            arm64_d_min_line_lg2_sz_b: geti("arm64_dMinLine_lg2_szB")? as u32,
            arm64_i_min_line_lg2_sz_b: geti("arm64_iMinLine_lg2_szB")? as u32,
            arm64_cache_block_size: geti("arm64_cache_block_size")? as u8,
            arm64_requires_fallback_llsc: geti("arm64_requires_fallback_LLSC")? as u8,
            x86_cr0: geti("x86_cr0")? as u32,
        };
        Ok(NativeLibVEXLifter {
            guest,
            info,
            settings: LiftSettings::default(),
        })
    }

    /// This lifter with `settings` in place of angr's defaults.
    pub fn with_settings(self, settings: LiftSettings) -> Self {
        NativeLibVEXLifter { settings, ..self }
    }

    /// Lift at `opt_level` (`None` takes the default), forced to 0 under
    /// self-modifying code as angr's lifter does.
    pub fn lift_at(
        &self,
        bytes: &[u8],
        addr: u64,
        arch: VexArch,
        opt_level: Option<i32>,
    ) -> Result<IRSB, LiftError> {
        // `from_py_arch` refuses to build a lifter without the symbol, so a
        // miss here means it was never resolved in this process.
        let Some(vex_lift) = ffi::vex_lift_fn() else {
            return Err(LiftError::LiftFailed {
                addr,
                reason: "vex_lift is not resolved".to_string(),
            });
        };
        if bytes.is_empty() {
            return Err(LiftError::LiftFailed {
                addr,
                reason: "empty byte slice".to_string(),
            });
        }

        // Cap max_bytes to the slice length so libVEX never reads past `bytes`.
        let max_bytes = bytes.len().min(super::VEX_MAX_BYTES) as u32;
        let settings = self.settings;
        let opt_level = if settings.selfmodifying_code {
            0
        } else {
            opt_level.unwrap_or(settings.default_opt_level)
        };
        let px_control = if settings.cross_insn_opt {
            VEX_REG_UPD_UNWINDREGS_AT_MEM_ACCESS
        } else {
            VEX_REG_UPD_LD_ALLREGS_AT_EACH_INSN
        };

        let _guard = lift_lock();
        // SAFETY: single-threaded through LIFT_LOCK; we marshal the whole IRSB
        // into owned Rust types before the guard drops, so nothing escapes the
        // arena. `bytes` outlives the call. vex_lift does not write through the
        // insn_start pointer.
        unsafe {
            let result = vex_lift(
                self.guest,
                self.info.to_ffi(),
                bytes.as_ptr(),
                addr,
                VEX_MAX_INSTRUCTIONS,
                max_bytes,
                opt_level,
                0, // traceflags
                1, // allow_arch_optimizations
                c_int::from(settings.strict_block_end),
                0, // collect_data_refs
                0, // load_from_ro_regions
                0, // const_prop
                px_control,
                0, // lookback_amount
            );
            if result.is_null() {
                return Err(LiftError::LiftFailed {
                    addr,
                    reason: "vex_lift returned NULL".to_string(),
                });
            }
            let irsb_ptr = (*result).irsb;
            if irsb_ptr.is_null() || (*result).size == 0 {
                return Err(LiftError::LiftFailed {
                    addr,
                    reason: format!(
                        "vex_lift decoded no instructions @ 0x{addr:x} (size={})",
                        (*result).size
                    ),
                });
            }
            Ok(marshal_irsb(irsb_ptr, addr, arch))
        }
    }
}

impl VEXLifter for NativeLibVEXLifter {
    fn lift(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError> {
        self.lift_at(bytes, addr, arch, None)
    }
}

// =============================================================================
// Marshalling: C VEXLiftResult->irsb  ->  Rust vex::ir::IRSB
// =============================================================================
//
// Every fn below is `unsafe` — it dereferences arena pointers valid only while
// LIFT_LOCK is held. Structural target shape matches pyvex_reader::read_*.

/// Endness tag -> Rust `Endness`.
///
/// There is no `Endness` sentinel to carry an unknown tag the way `c_op`
/// does, so the raw tag goes to the log instead.
fn c_endness(end: u32) -> Endness {
    match end {
        ffi::IEND_LE => Endness::Little,
        ffi::IEND_BE => Endness::Big,
        // SILENT(cat-c): defaulting to LE on an unknown tag can silently
        // byte-swap a BE load/store — warn so the drift is traceable.
        _ => {
            log::warn!("Unknown libVEX IREndness tag {end:#x}; assuming Iend_LE");
            Endness::Little
        }
    }
}

/// Jumpkind tag -> Rust `JumpKind` (via its pyvex `Ijk_*` name).
///
/// As with [`c_endness`], an unknown tag has no sentinel variant to preserve
/// it, so it is logged.
fn c_jumpkind(jk: u32) -> JumpKind {
    match tag_name(IJK_NAMES, IJK_BASE, jk) {
        Some(name) => parse_jumpkind(name),
        // SILENT(cat-c): defaulting to Boring turns a call/ret/syscall exit
        // into a plain jump — warn so the drift is traceable.
        None => {
            log::warn!("Unknown libVEX IRJumpKind tag {jk:#x}; assuming Ijk_Boring");
            JumpKind::Boring
        }
    }
}

/// Type tag -> Rust `IRType` (via its pyvex `Ity_*` name), defaulting to
/// `I64` on an unknown tag (as `pyvex_reader`'s `parse_type_or_log` does).
///
/// Both miss paths (no name for the tag, and a name [`parse_type_or_log`]
/// does not know) are logged for the same reason as [`c_endness`]; the latter
/// shares its fallback with the pyvex marshalling path in `pyvex_reader` so
/// both stay equally observable (angr-c7xno.90).
fn c_type_parse(ty: u32) -> IRType {
    match tag_name(ITY_NAMES, ffi::ITY_INVALID, ty) {
        Some(name) => parse_type_or_log(name, "libVEX IRType"),
        // SILENT(cat-c): defaulting to I64 mis-sizes the value at every use
        // site — warn so the drift is traceable.
        None => {
            log::warn!("Unknown libVEX IRType tag {ty:#x}; assuming Ity_I64");
            IRType::I64
        }
    }
}

/// Convert a C `IROp` tag to a Rust `IROp` via its pyvex name.
fn c_op(op: u32) -> super::ir::IROp {
    match vexop::op_name(op) {
        Some(name) => parse_opcode(name),
        // Preserve the numeric tag so an unmapped op is still traceable.
        None => parse_opcode(&format!("Iop_UNKNOWN_{op}")),
    }
}

/// Expand VEX's restricted 16-bit `Ico_V128` pattern into the full 128-bit value.
///
/// libVEX stores a V128 const as one bit per byte lane: bit `i` set means byte
/// `i` (counting from the least significant) is `0xff`, clear means `0x00`.
/// pyvex expands the same way in `IRConst.V128._from_c`, so parity with the
/// serialized pyvex path requires expanding here rather than passing the raw
/// pattern through.
fn expand_v128(pattern: u16) -> u128 {
    let mut value: u128 = 0;
    for i in 0..16 {
        if (pattern >> i) & 1 == 1 {
            value |= 0xffu128 << (8 * i);
        }
    }
    value
}

/// Expand VEX's restricted 32-bit `Ico_V256` pattern into four 64-bit limbs.
///
/// Same one-bit-per-byte-lane encoding as [`expand_v128`], over 32 byte lanes.
/// Limbs are ascending: `limbs[0]` holds bits 0..63.
fn expand_v256(pattern: u32) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for i in 0..32 {
        if (pattern >> i) & 1 == 1 {
            limbs[i / 8] |= 0xffu64 << (8 * (i % 8));
        }
    }
    limbs
}

/// Marshal a C `IRConst`.
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `c` must be a non-null `IRConst*` into the live
/// libVEX arena (see the module-level arena contract). The pointee is only read
/// and the `Ico` union is projected by `tag`, never retained past return.
unsafe fn marshal_const(c: *const ffi::IRConst) -> IRConst {
    let ico = &(*c).ico;
    match (*c).tag {
        ffi::ICO_U1 => IRConst::U1(ico.u1 != 0),
        ffi::ICO_U8 => IRConst::U8(ico.u8_),
        ffi::ICO_U16 => IRConst::U16(ico.u16_),
        ffi::ICO_U32 => IRConst::U32(ico.u32_),
        ffi::ICO_U64 => IRConst::U64(ico.u64_),
        // A restricted constant in the same one-bit-per-byte encoding as
        // Ico_V128 (libvex_ir.h).
        ffi::ICO_U128 => IRConst::U128(expand_v128(ico.u128)),
        ffi::ICO_F32 => IRConst::F32(ico.f32_),
        ffi::ICO_F32I => IRConst::F32(f32::from_bits(ico.f32i)),
        ffi::ICO_F64 => IRConst::F64(ico.f64_),
        ffi::ICO_F64I => IRConst::F64(f64::from_bits(ico.f64i)),
        ffi::ICO_V128 => IRConst::V128(expand_v128(ico.v128)),
        ffi::ICO_V256 => IRConst::V256(expand_v256(ico.v256)),
        ffi::ICO_V512 => {
            // SILENT(cat-c): the engine has no 512-bit vector type, so an
            // AVX-512 constant cannot be represented; the zero stand-in is
            // wrong for any nonzero pattern. Only EVEX-encoded code reaches
            // this (angr-mx00p.5).
            log::warn!("libVEX Ico_V512 constant is unsupported; assuming Ico_U64(0)");
            IRConst::U64(0)
        }
        tag => {
            // SILENT(cat-c): defaulting to U64(0) on an unrecognized libVEX
            // Ico_* tag silently corrupts every constant of that kind — warn
            // so the drift is traceable.
            log::warn!("Unknown libVEX IRConstTag {tag:#x}; assuming Ico_U64(0)");
            IRConst::U64(0)
        }
    }
}

/// Read `dst` (an `IRConst*`) as a u64 (for `Exit`).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `c` must be a non-null `IRConst*` into the live
/// libVEX arena. Delegates the deref to [`marshal_const`]; retains nothing.
unsafe fn const_to_u64(c: *const ffi::IRConst) -> u64 {
    match marshal_const(c) {
        IRConst::U1(v) => v as u64,
        IRConst::U8(v) => v as u64,
        IRConst::U16(v) => v as u64,
        IRConst::U32(v) => v as u64,
        IRConst::U64(v) => v,
        IRConst::U128(v) => v as u64,
        IRConst::F32(v) => v.to_bits() as u64,
        IRConst::F64(v) => v.to_bits(),
        IRConst::V128(v) => v as u64,
        IRConst::V256(v) => v[0],
    }
}

/// Marshal a C `IRCallee` (CCall/Dirty helper reference).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `cee` must be a non-null `IRCallee*` into the
/// live libVEX arena. Its `name` field, if non-null, must be a NUL-terminated C
/// string valid for the read; it is copied into an owned `String` before return.
unsafe fn marshal_callee(cee: *const ffi::IRCallee) -> IRCallee {
    let name = if (*cee).name.is_null() {
        String::new()
    } else {
        CStr::from_ptr((*cee).name).to_string_lossy().into_owned()
    };
    IRCallee {
        name,
        // The C `IRCallee.addr` is the *host* address of the helper (e.g. the
        // in-process pointer to `amd64g_calculate_condition`), which is
        // non-deterministic across loads and unused by the Rust interpreter — it
        // dispatches CCalls by `name`. `pyvex_reader::read_callee` reads 0 too,
        // so a block is identical whichever path lifted it (parity test).
        addr: 0,
        mcx_mask: (*cee).mcx_mask,
    }
}

/// Marshal a C `IRRegArray` (rotating-register-window descriptor).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `r` must be a non-null `IRRegArray*` into the
/// live libVEX arena. All fields are read by value; nothing is retained.
unsafe fn marshal_reg_array(r: *const ffi::IRRegArray) -> IRRegArray {
    IRRegArray {
        base: (*r).base as u32,
        elemTy: c_type_parse((*r).elem_ty),
        nElems: (*r).n_elems as u32,
    }
}

/// Collect a NULL-terminated `IRExpr**` vector into owned `IRExpr`s.
///
/// # Safety
/// Caller must hold `LIFT_LOCK`. `args` may be null (returns empty). If non-null
/// it must point at a NULL-terminated `IRExpr*` array in the live libVEX arena;
/// each element is walked with [`marshal_expr`]. Nothing is retained.
unsafe fn marshal_expr_vec(mut args: *mut *mut ffi::IRExpr) -> Vec<IRExpr> {
    let mut out = Vec::new();
    if args.is_null() {
        return out;
    }
    while !(*args).is_null() {
        out.push(marshal_expr(*args));
        args = args.add(1);
    }
    out
}

/// Marshal a C `IRExpr` (recursively via its tagged `Iex` union).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `e` must be a non-null `IRExpr*` into the live
/// libVEX arena, with every child pointer reachable from its `Iex` union
/// likewise valid. Recurses over the whole expr tree into owned types; retains
/// no arena pointer past return.
unsafe fn marshal_expr(e: *const ffi::IRExpr) -> IRExpr {
    let iex = &(*e).iex;
    match (*e).tag {
        ffi::IEX_CONST => IRExpr::Const(marshal_const(iex.con.con)),
        ffi::IEX_RDTMP => IRExpr::RdTmp(iex.rdtmp.tmp),
        ffi::IEX_GET => IRExpr::Get {
            offset: iex.get.offset as u32,
            ty: c_type_parse(iex.get.ty),
        },
        ffi::IEX_GETI => IRExpr::GetI {
            descr: marshal_reg_array(iex.geti.descr),
            ix: Box::new(marshal_expr(iex.geti.ix)),
            bias: iex.geti.bias as u32,
        },
        ffi::IEX_LOAD => IRExpr::Load {
            addr: Box::new(marshal_expr(iex.load.addr)),
            ty: c_type_parse(iex.load.ty),
            endness: c_endness(iex.load.end),
        },
        ffi::IEX_UNOP => IRExpr::Unop {
            op: c_op(iex.unop.op),
            arg: Box::new(marshal_expr(iex.unop.arg)),
        },
        ffi::IEX_BINOP => IRExpr::Binop {
            op: c_op(iex.binop.op),
            left: Box::new(marshal_expr(iex.binop.arg1)),
            right: Box::new(marshal_expr(iex.binop.arg2)),
        },
        ffi::IEX_TRIOP => {
            let d = iex.triop.details;
            IRExpr::Triop {
                op: c_op((*d).op),
                arg1: Box::new(marshal_expr((*d).arg1)),
                arg2: Box::new(marshal_expr((*d).arg2)),
                arg3: Box::new(marshal_expr((*d).arg3)),
            }
        }
        ffi::IEX_QOP => {
            let d = iex.qop.details;
            IRExpr::Qop {
                op: c_op((*d).op),
                arg1: Box::new(marshal_expr((*d).arg1)),
                arg2: Box::new(marshal_expr((*d).arg2)),
                arg3: Box::new(marshal_expr((*d).arg3)),
                arg4: Box::new(marshal_expr((*d).arg4)),
            }
        }
        ffi::IEX_ITE => IRExpr::ITE {
            cond: Box::new(marshal_expr(iex.ite.cond)),
            iftrue: Box::new(marshal_expr(iex.ite.iftrue)),
            iffalse: Box::new(marshal_expr(iex.ite.iffalse)),
        },
        ffi::IEX_CCALL => IRExpr::CCall {
            cee: marshal_callee(iex.ccall.cee),
            retty: c_type_parse(iex.ccall.retty),
            args: marshal_expr_vec(iex.ccall.args),
        },
        ffi::IEX_VECRET => IRExpr::VECRET,
        ffi::IEX_GSPTR => IRExpr::GSPTR,
        // Binder and any future tag: no Rust equivalent; surface as a 0 const.
        _ => IRExpr::Const(IRConst::U64(0)),
    }
}

/// `IRTemp_INVALID` -> None, else Some.
fn opt_temp(t: u32) -> Option<u32> {
    if t == IRTEMP_INVALID { None } else { Some(t) }
}

/// Null pointer -> None, else Some(boxed marshalled expr).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`. `e` may be null (returns `None`). If non-null it
/// must be a valid `IRExpr*` into the live libVEX arena, delegated to
/// [`marshal_expr`]; nothing is retained.
unsafe fn opt_expr(e: *mut ffi::IRExpr) -> Option<Box<IRExpr>> {
    if e.is_null() {
        None
    } else {
        Some(Box::new(marshal_expr(e)))
    }
}

fn loadg_op(cvt: u32) -> IRLoadGOp {
    match cvt {
        ILGOP_IDENT_V128 | ILGOP_IDENT64 | ILGOP_IDENT32 | ILGOP_IDENT16 | ILGOP_IDENT8 => {
            IRLoadGOp::Identity
        }
        ILGOP_8UTO32 => IRLoadGOp::WidenZ { src_bits: 8 },
        ILGOP_8STO32 => IRLoadGOp::WidenS { src_bits: 8 },
        ILGOP_16UTO32 => IRLoadGOp::WidenZ { src_bits: 16 },
        ILGOP_16STO32 => IRLoadGOp::WidenS { src_bits: 16 },
        _ => IRLoadGOp::Unknown,
    }
}

fn dirty_fx(fx: u32) -> DirtyFx {
    match fx {
        IFX_READ => DirtyFx::Read,
        IFX_WRITE => DirtyFx::Write,
        IFX_MODIFY => DirtyFx::Modify,
        _ => DirtyFx::None,
    }
}

fn mbe_event(_ev: u32) -> MBusEvent {
    // SILENT(cat-a): the payload is inert. VEX only emits Imbe_Fence /
    // Imbe_CancelReservation, and `Interpreter`'s `IRStmt::MBE` arm discards
    // the event anyway (`statements.rs`, `IRStmt::MBE(_) => Continue`) — this
    // engine treats every barrier as a full fence. The pyvex sibling
    // `pyvex_reader::parse_mbe_event` keeps the SFence/LFence/MFence
    // discriminants only because the string form carries them for free; both
    // collapse every unrecognized tag to `Fence`. Logging here would warn
    // about a value nothing reads.
    MBusEvent::Fence
}

/// Marshal a C `IRStmt` (via its tagged `Ist` union), including the boxed
/// `details` sub-structs of PutI/StoreG/LoadG/CAS/Dirty.
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `s` must be a non-null `IRStmt*` into the live
/// libVEX arena, with every child pointer reachable from its `Ist` union (and
/// the `details` sub-structs) likewise valid. Marshals into owned types; retains
/// no arena pointer past return.
unsafe fn marshal_stmt(s: *const ffi::IRStmt) -> IRStmt {
    let ist = &(*s).ist;
    match (*s).tag {
        ffi::IST_IMARK => IRStmt::IMark {
            addr: ist.imark.addr,
            len: ist.imark.len,
            delta: ist.imark.delta,
        },
        ffi::IST_WRTMP => IRStmt::WrTmp {
            tmp: ist.wrtmp.tmp,
            data: marshal_expr(ist.wrtmp.data),
        },
        ffi::IST_PUT => IRStmt::Put {
            offset: ist.put.offset as u32,
            data: marshal_expr(ist.put.data),
        },
        ffi::IST_STORE => IRStmt::Store {
            addr: marshal_expr(ist.store.addr),
            data: marshal_expr(ist.store.data),
            endness: c_endness(ist.store.end),
        },
        ffi::IST_EXIT => IRStmt::Exit {
            guard: marshal_expr(ist.exit.guard),
            dst: const_to_u64(ist.exit.dst),
            jk: c_jumpkind(ist.exit.jk),
            offsIP: ist.exit.offs_ip as u32,
        },
        ffi::IST_NOOP => IRStmt::NoOp,
        ffi::IST_ABIHINT => IRStmt::AbiHint {
            base: Box::new(marshal_expr(ist.abi_hint.base)),
            len: ist.abi_hint.len as u32,
            nia: Box::new(marshal_expr(ist.abi_hint.nia)),
        },
        ffi::IST_PUTI => {
            let d = ist.puti.details;
            IRStmt::PutI {
                descr: marshal_reg_array((*d).descr),
                ix: Box::new(marshal_expr((*d).ix)),
                bias: (*d).bias as u32,
                data: Box::new(marshal_expr((*d).data)),
            }
        }
        ffi::IST_STOREG => {
            let d = ist.storeg.details;
            IRStmt::StoreG {
                addr: Box::new(marshal_expr((*d).addr)),
                data: Box::new(marshal_expr((*d).data)),
                guard: Box::new(marshal_expr((*d).guard)),
                endness: c_endness((*d).end),
            }
        }
        ffi::IST_LOADG => {
            let d = ist.loadg.details;
            IRStmt::LoadG {
                dst: (*d).dst,
                addr: Box::new(marshal_expr((*d).addr)),
                alt: Box::new(marshal_expr((*d).alt)),
                guard: Box::new(marshal_expr((*d).guard)),
                cvt: loadg_op((*d).cvt),
                endness: c_endness((*d).end),
            }
        }
        ffi::IST_CAS => {
            let d = ist.cas.details;
            IRStmt::CAS {
                old_hi: opt_temp((*d).old_hi),
                old_lo: (*d).old_lo,
                addr: Box::new(marshal_expr((*d).addr)),
                expdHi: opt_expr((*d).expd_hi),
                expdLo: Box::new(marshal_expr((*d).expd_lo)),
                dataHi: opt_expr((*d).data_hi),
                dataLo: Box::new(marshal_expr((*d).data_lo)),
                endness: c_endness((*d).end),
            }
        }
        ffi::IST_LLSC => IRStmt::LLSC {
            storedata: opt_expr(ist.llsc.storedata),
            result: ist.llsc.result,
            addr: Box::new(marshal_expr(ist.llsc.addr)),
            endness: c_endness(ist.llsc.end),
        },
        ffi::IST_MBE => IRStmt::MBE(mbe_event(ist.mbe.event)),
        ffi::IST_DIRTY => {
            let d = ist.dirty.details;
            IRStmt::Dirty(IRDirty {
                cee: marshal_callee((*d).cee),
                guard: opt_expr((*d).guard),
                tmp: opt_temp((*d).tmp),
                mFx: dirty_fx((*d).m_fx),
                mAddr: opt_expr((*d).m_addr),
                mSize: (*d).m_size as u32,
                nFxState: (*d).n_fx_state as u32,
                args: marshal_expr_vec((*d).args),
            })
        }
        _ => IRStmt::NoOp,
    }
}

/// Marshal a C `IRTypeEnv` (the block's temp type table).
///
/// # Safety
/// Caller must hold `LIFT_LOCK`. `tyenv` may be null (returns empty). If non-null
/// it must be a valid `IRTypeEnv*` into the live libVEX arena whose `types`
/// array holds at least `types_used` entries; each is read by value.
unsafe fn marshal_tyenv(tyenv: *const ffi::IRTypeEnv) -> TypeEnv {
    let mut types = Vec::new();
    if !tyenv.is_null() {
        let used = (*tyenv).types_used.max(0) as usize;
        let base = (*tyenv).types;
        for i in 0..used {
            types.push(c_type_parse(*base.add(i)));
        }
    }
    TypeEnv { types }
}

/// Marshal the whole C `IRSB` into an owned Rust [`IRSB`] — the top-level entry
/// called by `lift` while `LIFT_LOCK` is held.
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `irsb` must be a non-null `IRSB*` into the live
/// libVEX arena whose `stmts` array holds at least `stmts_used` entries and
/// whose `next`/`tyenv` pointers are valid. Walks the entire block into owned
/// types before returning, so nothing survives the arena's next clobber.
unsafe fn marshal_irsb(irsb: *const ffi::IRSB, addr: u64, arch: VexArch) -> IRSB {
    let mut statements = Vec::new();
    let used = (*irsb).stmts_used.max(0) as usize;
    let stmts = (*irsb).stmts;
    for i in 0..used {
        let stmt_ptr = *stmts.add(i);
        if !stmt_ptr.is_null() {
            statements.push(marshal_stmt(stmt_ptr));
        }
    }

    IRSB {
        addr,
        arch,
        statements,
        next: marshal_expr((*irsb).next),
        jumpkind: c_jumpkind((*irsb).jumpkind),
        offsIP: (*irsb).offs_ip as u32,
        tyenv: marshal_tyenv((*irsb).tyenv),
    }
}

/// Whether the lift lock is free, for `engine::lift_lock_is_held_for_test`.
pub(crate) fn lift_lock_is_free_for_test() -> bool {
    LIFT_LOCK.try_lock().is_some()
}

/// The fork-local enum tables, as `(name, tag)` pairs, for
/// `test_native_lift_parity.py` to check against pyvex's cffi library.
pub(crate) fn enum_tables_for_test() -> Vec<(&'static str, u32)> {
    let mut out: Vec<(&'static str, u32)> = Vec::new();
    for (table, base) in [(IJK_NAMES, IJK_BASE), (ITY_NAMES, ffi::ITY_INVALID)] {
        for (i, name) in table.iter().enumerate() {
            out.push((name, base + i as u32)); // overflow-ok: a table index, a few dozen at most
        }
    }
    out.extend([
        ("Iend_LE", ffi::IEND_LE),
        ("Iend_BE", ffi::IEND_BE),
        ("Ifx_Read", IFX_READ),
        ("Ifx_Write", IFX_WRITE),
        ("Ifx_Modify", IFX_MODIFY),
        ("ILGop_IdentV128", ILGOP_IDENT_V128),
        ("ILGop_Ident64", ILGOP_IDENT64),
        ("ILGop_Ident32", ILGOP_IDENT32),
        ("ILGop_Ident16", ILGOP_IDENT16),
        ("ILGop_Ident8", ILGOP_IDENT8),
        ("ILGop_16Uto32", ILGOP_16UTO32),
        ("ILGop_16Sto32", ILGOP_16STO32),
        ("ILGop_8Uto32", ILGOP_8UTO32),
        ("ILGop_8Sto32", ILGOP_8STO32),
        (
            "VexRegUpdUnwindregsAtMemAccess",
            VEX_REG_UPD_UNWINDREGS_AT_MEM_ACCESS,
        ),
        (
            "VexRegUpdLdAllregsAtEachInsn",
            VEX_REG_UPD_LD_ALLREGS_AT_EACH_INSN,
        ),
        ("Ico_U1", ffi::ICO_U1),
        ("Ico_U8", ffi::ICO_U8),
        ("Ico_U16", ffi::ICO_U16),
        ("Ico_U32", ffi::ICO_U32),
        ("Ico_U64", ffi::ICO_U64),
        ("Ico_U128", ffi::ICO_U128),
        ("Ico_F32", ffi::ICO_F32),
        ("Ico_F32i", ffi::ICO_F32I),
        ("Ico_F64", ffi::ICO_F64),
        ("Ico_F64i", ffi::ICO_F64I),
        ("Ico_V128", ffi::ICO_V128),
        ("Ico_V256", ffi::ICO_V256),
        ("Ico_V512", ffi::ICO_V512),
    ]);
    for name in [
        "VexArchX86",
        "VexArchAMD64",
        "VexArchARM",
        "VexArchARM64",
        "VexArchPPC32",
        "VexArchPPC64",
        "VexArchS390X",
        "VexArchMIPS32",
        "VexArchMIPS64",
        "VexArchRISCV64",
    ] {
        if let Some(tag) = vex_arch_tag(name) {
            out.push((name, tag));
        }
    }
    out
}

/// `(struct, size)` and `(struct.field, offset)` for the `ailment::vex_ffi`
/// mirrors this lifter reads, for `test_native_lift_parity.py` to check
/// against pyvex's cffi `sizeof` / `offsetof`.
pub(crate) fn layouts_for_test() -> Vec<(&'static str, usize)> {
    use std::mem::{offset_of, size_of};
    vec![
        ("IRSB", size_of::<ffi::IRSB>()),
        ("IRSB.next", offset_of!(ffi::IRSB, next)),
        ("IRSB.jumpkind", offset_of!(ffi::IRSB, jumpkind)),
        ("IRSB.offsIP", offset_of!(ffi::IRSB, offs_ip)),
        ("IRTypeEnv", size_of::<ffi::IRTypeEnv>()),
        ("IRStmt", size_of::<ffi::IRStmt>()),
        ("IRExpr", size_of::<ffi::IRExpr>()),
        ("IRConst", size_of::<ffi::IRConst>()),
        ("IRCallee", size_of::<ffi::IRCallee>()),
        ("IRCallee.mcx_mask", offset_of!(ffi::IRCallee, mcx_mask)),
        ("IRRegArray", size_of::<ffi::IRRegArray>()),
        ("IRTriop", size_of::<ffi::IRTriop>()),
        ("IRQop", size_of::<ffi::IRQop>()),
        ("IRCAS", size_of::<ffi::IRCAS>()),
        ("IRPutI", size_of::<ffi::IRPutI>()),
        ("IRStoreG", size_of::<ffi::IRStoreG>()),
        ("IRLoadG", size_of::<ffi::IRLoadG>()),
        ("IRDirty", size_of::<ffi::IRDirty>()),
        ("IRDirty.nFxState", offset_of!(ffi::IRDirty, n_fx_state)),
        ("VexArchInfo", size_of::<ffi::VexArchInfo>()),
        ("VexArchInfo.x86_cr0", offset_of!(ffi::VexArchInfo, x86_cr0)),
        ("VexCacheInfo", size_of::<ffi::VexCacheInfo>()),
    ]
}

test_submod!("libvex_lifter_tests.rs" => tests);
