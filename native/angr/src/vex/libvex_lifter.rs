//! Native libVEX lifter (`libvex-ffi` feature, AMD64 only for Stage-1).
//!
//! `NativeLibVEXLifter` implements the [`VEXLifter`] trait by calling the
//! `vex_lift` shim exported from `libpyvex.so` (the *same* object pyvex loads
//! via cffi) and marshalling the returned C `VEXLiftResult->irsb` into a Rust
//! [`IRSB`]. Binding pyvex's exact-config shim is the parity guarantee: the
//! lifted IR is byte-for-byte what the Python callback path produces today (see
//! `docs/advanced-topics/rust_libvex_ffi.rst`).
//!
//! The marshalling mirrors `pyvex_bridge.rs::convert_*` node-for-node — same
//! `vex::ir` target shape — but reads the raw C structs (via the bindgen
//! `libvex_ffi` bindings) instead of a serde-JSON round-trip. Enum tags are
//! bridged back to pyvex-style names with the generated `enum_names` tables so
//! the existing string parsers in `opcode_map.rs` can be reused verbatim (DRY).
//!
//! Safety invariants (see the doc's "Open risks"):
//! - `vex_lift` returns a pointer into libVEX's temporary arena; it is clobbered
//!   on the next `vex_lift` call. We marshal the whole `IRSB` into owned Rust
//!   types *before* releasing the lift lock, so nothing outlives the arena.
//! - libVEX global state (`vex_control`, the arena) is not re-entrant. A single
//!   process-wide `LIFT_LOCK` mutex serializes every lift.

// Every marshalling helper below is an `unsafe fn` whose entire body walks the
// libVEX arena. Wrapping each individual deref in its own `unsafe {}` block adds
// only rightward drift here — the whole module is a single unsafe domain gated
// by `LIFT_LOCK` and the top-level SAFETY contract in `lift`. Scope the
// edition-2024 lint to this module rather than sprinkling per-op blocks.
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::c_uchar;
use std::ptr;
use std::sync::{Mutex, Once};

use super::ir::{
    DirtyFx, Endness, IRCallee, IRConst, IRDirty, IRExpr, IRLoadGOp, IRRegArray, IRSB, IRStmt,
    IRType, JumpKind, MBusEvent, TypeEnv, VexArch,
};
use super::libvex_ffi as ffi;
use super::libvex_ffi::enum_names;
use super::lifter::{LiftError, VEXLifter};
use super::opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type};

/// VEX's `IRTemp_INVALID` sentinel (an absent temp, e.g. the `oldHi` slot of a
/// single-width CAS or an untyped Dirty result).
const IRTEMP_INVALID: u32 = 0xFFFF_FFFF;

/// pyvex `_lift` defaults (disassembled from `pyvex/lifting/libvex.py`).
const VEX_MAX_INSTRUCTIONS: u32 = 99;
const VEX_MAX_BYTES: u32 = 5000;

/// Serializes every `vex_lift` call — libVEX global state is not re-entrant.
static LIFT_LOCK: Mutex<()> = Mutex::new(());
static VEX_INIT: Once = Once::new();

/// Native libVEX lifter backed by `libpyvex.so`'s `vex_lift` shim.
///
/// AMD64 only for Stage-1; other arches return [`LiftError::InvalidArch`] so the
/// caller falls back to the pyvex-callback path.
#[derive(Debug, Default)]
pub struct NativeLibVEXLifter;

impl NativeLibVEXLifter {
    pub fn new() -> Self {
        NativeLibVEXLifter
    }
}

/// Build the AMD64 `VexArchInfo` pyvex passes to `vex_lift` (from
/// `archinfo.ArchAMD64().vex_archinfo`): baseline hwcaps, little-endian, empty
/// cache info, `x86_cr0 = 0xFFFFFFFF`.
fn amd64_archinfo() -> ffi::VexArchInfo {
    ffi::VexArchInfo {
        hwcaps: 0,
        endness: ffi::VexEndness::VexEndnessLE,
        hwcache_info: ffi::VexCacheInfo {
            num_levels: 0,
            num_caches: 0,
            caches: ptr::null_mut(),
            icaches_maintain_coherence: 1,
        },
        ppc_icache_line_szB: 0,
        ppc_dcbz_szB: 0,
        ppc_dcbzl_szB: 0,
        arm64_dMinLine_lg2_szB: 0,
        arm64_iMinLine_lg2_szB: 0,
        x86_cr0: 0xFFFF_FFFF,
    }
}

impl VEXLifter for NativeLibVEXLifter {
    fn lift(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError> {
        if arch != VexArch::AMD64 {
            return Err(LiftError::InvalidArch(format!(
                "NativeLibVEXLifter supports AMD64 only (Stage-1), got {arch:?}"
            )));
        }
        if bytes.is_empty() {
            return Err(LiftError::LiftFailed {
                addr,
                reason: "empty byte slice".to_string(),
            });
        }

        // vex_init is idempotent (guarded by libVEX's `vex_initdone`), but we
        // only need it once per process.
        VEX_INIT.call_once(|| unsafe {
            ffi::vex_init();
        });

        // Cap max_bytes to the slice length so libVEX never reads past `bytes`.
        let max_bytes = (bytes.len() as u32).min(VEX_MAX_BYTES);

        let _guard = LIFT_LOCK.lock().expect("libVEX lift lock poisoned");
        // SAFETY: single-threaded through LIFT_LOCK; we marshal the whole IRSB
        // into owned Rust types before the guard drops, so nothing escapes the
        // arena. `bytes` outlives the call. vex_lift does not write through the
        // insn_start pointer.
        unsafe {
            let result = ffi::vex_lift(
                ffi::VexArch::VexArchAMD64,
                amd64_archinfo(),
                bytes.as_ptr() as *mut c_uchar,
                addr,
                VEX_MAX_INSTRUCTIONS,
                max_bytes,
                1, // opt_level
                0, // traceflags
                1, // allow_arch_optimizations
                0, // strict_block_end
                0, // collect_data_refs
                0, // load_from_ro_regions
                0, // const_prop
                ffi::VexRegisterUpdates::VexRegUpdUnwindregsAtMemAccess,
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
            Ok(marshal_irsb(irsb_ptr, addr))
        }
    }
}

// =============================================================================
// Marshalling: C VEXLiftResult->irsb  ->  Rust vex::ir::IRSB
// =============================================================================
//
// Every fn below is `unsafe` — it dereferences arena pointers valid only while
// LIFT_LOCK is held. Structural target shape matches pyvex_bridge::convert_*.

/// Endness discriminant -> Rust `Endness` (via pyvex `Iend_*` name).
fn c_endness(end: ffi::IREndness) -> Endness {
    match enum_names::iend_name(end.0) {
        Some(name) => parse_endness(name),
        None => Endness::Little,
    }
}

/// Jumpkind discriminant -> Rust `JumpKind` (via pyvex `Ijk_*` name).
fn c_jumpkind(jk: ffi::IRJumpKind) -> JumpKind {
    match enum_names::ijk_name(jk.0) {
        Some(name) => parse_jumpkind(name),
        None => JumpKind::Boring,
    }
}

/// Type discriminant -> Rust `IRType` (via pyvex `Ity_*` name), defaulting to
/// `I64` on an unknown tag (mirrors `pyvex_bridge`'s `unwrap_or(IRType::I64)`).
fn c_type_parse(ty: ffi::IRType) -> IRType {
    match enum_names::irtype_name(ty.0) {
        Some(name) => parse_type(name).unwrap_or(IRType::I64),
        None => IRType::I64,
    }
}

/// Convert a C `IROp` discriminant to a Rust `IROp` via its pyvex name.
///
/// # Safety
/// Dereferences no pointers — `op` is a by-value discriminant newtype, so the
/// caller has no aliasing/lifetime obligation. Marked `unsafe` only to keep the
/// marshalling helpers in a single unsafe domain (see the module note).
unsafe fn c_op(op: ffi::IROp) -> super::ir::IROp {
    match enum_names::irop_name(op.0) {
        Some(name) => parse_opcode(name),
        // Preserve the numeric tag so an unmapped op is still traceable.
        None => parse_opcode(&format!("Iop_UNKNOWN_{}", op.0)),
    }
}

/// Marshal a C `IRConst`.
///
/// # Safety
/// Caller must hold `LIFT_LOCK`; `c` must be a non-null `IRConst*` into the live
/// libVEX arena (see the module-level arena contract). The pointee is only read
/// and the `Ico` union is projected by `tag`, never retained past return.
unsafe fn marshal_const(c: *const ffi::IRConst) -> IRConst {
    let tag = (*c).tag.0;
    let ico = &(*c).Ico;
    if tag == ffi::IRConstTag::Ico_U1.0 {
        IRConst::U1(ico.U1 != 0)
    } else if tag == ffi::IRConstTag::Ico_U8.0 {
        IRConst::U8(ico.U8)
    } else if tag == ffi::IRConstTag::Ico_U16.0 {
        IRConst::U16(ico.U16)
    } else if tag == ffi::IRConstTag::Ico_U32.0 {
        IRConst::U32(ico.U32)
    } else if tag == ffi::IRConstTag::Ico_U64.0 {
        IRConst::U64(ico.U64)
    } else if tag == ffi::IRConstTag::Ico_F32.0 {
        IRConst::F32(ico.F32)
    } else if tag == ffi::IRConstTag::Ico_F32i.0 {
        IRConst::F32(f32::from_bits(ico.F32i))
    } else if tag == ffi::IRConstTag::Ico_F64.0 {
        IRConst::F64(ico.F64)
    } else if tag == ffi::IRConstTag::Ico_F64i.0 {
        IRConst::F64(f64::from_bits(ico.F64i))
    } else if tag == ffi::IRConstTag::Ico_V128.0 {
        // VEX encodes a V128 const as a 16-bit "one bit per byte" pattern.
        // Stage-1 stores the raw pattern zero-extended; the .4 parity harness
        // owns exact V128/V256 const expansion.
        IRConst::V128(ico.V128 as u128)
    } else if tag == ffi::IRConstTag::Ico_V256.0 {
        IRConst::V256([ico.V256 as u64, 0, 0, 0])
    } else {
        IRConst::U64(0)
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
        // dispatches CCalls by `name`. The pyvex serializer hardcodes `addr: 0`
        // (`rust_irsb_serializer._serialize_cee`), so normalize to 0 here to stay
        // a byte-for-byte drop-in for the pyvex-serialized path (corpus parity gate).
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
        elemTy: c_type_parse((*r).elemTy),
        nElems: (*r).nElems as u32,
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
    let tag = (*e).tag.0;
    let iex = &(*e).Iex;
    if tag == ffi::IRExprTag::Iex_Const.0 {
        IRExpr::Const(marshal_const(iex.Const.con))
    } else if tag == ffi::IRExprTag::Iex_RdTmp.0 {
        IRExpr::RdTmp(iex.RdTmp.tmp)
    } else if tag == ffi::IRExprTag::Iex_Get.0 {
        IRExpr::Get {
            offset: iex.Get.offset as u32,
            ty: c_type_parse(iex.Get.ty),
        }
    } else if tag == ffi::IRExprTag::Iex_GetI.0 {
        IRExpr::GetI {
            descr: marshal_reg_array(iex.GetI.descr),
            ix: Box::new(marshal_expr(iex.GetI.ix)),
            bias: iex.GetI.bias as u32,
        }
    } else if tag == ffi::IRExprTag::Iex_Load.0 {
        IRExpr::Load {
            addr: Box::new(marshal_expr(iex.Load.addr)),
            ty: c_type_parse(iex.Load.ty),
            endness: c_endness(iex.Load.end),
        }
    } else if tag == ffi::IRExprTag::Iex_Unop.0 {
        IRExpr::Unop {
            op: c_op(iex.Unop.op),
            arg: Box::new(marshal_expr(iex.Unop.arg)),
        }
    } else if tag == ffi::IRExprTag::Iex_Binop.0 {
        IRExpr::Binop {
            op: c_op(iex.Binop.op),
            left: Box::new(marshal_expr(iex.Binop.arg1)),
            right: Box::new(marshal_expr(iex.Binop.arg2)),
        }
    } else if tag == ffi::IRExprTag::Iex_Triop.0 {
        let d = iex.Triop.details;
        IRExpr::Triop {
            op: c_op((*d).op),
            arg1: Box::new(marshal_expr((*d).arg1)),
            arg2: Box::new(marshal_expr((*d).arg2)),
            arg3: Box::new(marshal_expr((*d).arg3)),
        }
    } else if tag == ffi::IRExprTag::Iex_Qop.0 {
        let d = iex.Qop.details;
        IRExpr::Qop {
            op: c_op((*d).op),
            arg1: Box::new(marshal_expr((*d).arg1)),
            arg2: Box::new(marshal_expr((*d).arg2)),
            arg3: Box::new(marshal_expr((*d).arg3)),
            arg4: Box::new(marshal_expr((*d).arg4)),
        }
    } else if tag == ffi::IRExprTag::Iex_ITE.0 {
        IRExpr::ITE {
            cond: Box::new(marshal_expr(iex.ITE.cond)),
            iftrue: Box::new(marshal_expr(iex.ITE.iftrue)),
            iffalse: Box::new(marshal_expr(iex.ITE.iffalse)),
        }
    } else if tag == ffi::IRExprTag::Iex_CCall.0 {
        IRExpr::CCall {
            cee: marshal_callee(iex.CCall.cee),
            retty: c_type_parse(iex.CCall.retty),
            args: marshal_expr_vec(iex.CCall.args),
        }
    } else if tag == ffi::IRExprTag::Iex_VECRET.0 {
        IRExpr::VECRET
    } else if tag == ffi::IRExprTag::Iex_GSPTR.0 {
        IRExpr::GSPTR
    } else {
        // Binder and any future tag: no Rust equivalent; surface as a 0 const.
        IRExpr::Const(IRConst::U64(0))
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

fn loadg_op(cvt: ffi::IRLoadGOp) -> IRLoadGOp {
    let v = cvt.0;
    if v == ffi::IRLoadGOp::ILGop_IdentV128.0
        || v == ffi::IRLoadGOp::ILGop_Ident64.0
        || v == ffi::IRLoadGOp::ILGop_Ident32.0
    {
        IRLoadGOp::Identity
    } else if v == ffi::IRLoadGOp::ILGop_8Uto32.0 {
        IRLoadGOp::WidenZ { src_bits: 8 }
    } else if v == ffi::IRLoadGOp::ILGop_8Sto32.0 {
        IRLoadGOp::WidenS { src_bits: 8 }
    } else if v == ffi::IRLoadGOp::ILGop_16Uto32.0 {
        IRLoadGOp::WidenZ { src_bits: 16 }
    } else if v == ffi::IRLoadGOp::ILGop_16Sto32.0 {
        IRLoadGOp::WidenS { src_bits: 16 }
    } else {
        IRLoadGOp::Unknown
    }
}

fn dirty_fx(fx: ffi::IREffect) -> DirtyFx {
    let v = fx.0;
    if v == ffi::IREffect::Ifx_Read.0 {
        DirtyFx::Read
    } else if v == ffi::IREffect::Ifx_Write.0 {
        DirtyFx::Write
    } else if v == ffi::IREffect::Ifx_Modify.0 {
        DirtyFx::Modify
    } else {
        DirtyFx::None
    }
}

fn mbe_event(ev: ffi::IRMBusEvent) -> MBusEvent {
    // VEX only emits Imbe_Fence / Imbe_CancelReservation; both map to Fence
    // here (the Rust engine treats every barrier as a full fence).
    let _ = ev;
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
    let tag = (*s).tag.0;
    let ist = &(*s).Ist;
    if tag == ffi::IRStmtTag::Ist_IMark.0 {
        IRStmt::IMark {
            addr: ist.IMark.addr,
            len: ist.IMark.len,
            delta: ist.IMark.delta,
        }
    } else if tag == ffi::IRStmtTag::Ist_WrTmp.0 {
        IRStmt::WrTmp {
            tmp: ist.WrTmp.tmp,
            data: marshal_expr(ist.WrTmp.data),
        }
    } else if tag == ffi::IRStmtTag::Ist_Put.0 {
        IRStmt::Put {
            offset: ist.Put.offset as u32,
            data: marshal_expr(ist.Put.data),
        }
    } else if tag == ffi::IRStmtTag::Ist_Store.0 {
        IRStmt::Store {
            addr: marshal_expr(ist.Store.addr),
            data: marshal_expr(ist.Store.data),
            endness: c_endness(ist.Store.end),
        }
    } else if tag == ffi::IRStmtTag::Ist_Exit.0 {
        IRStmt::Exit {
            guard: marshal_expr(ist.Exit.guard),
            dst: const_to_u64(ist.Exit.dst),
            jk: c_jumpkind(ist.Exit.jk),
            offsIP: ist.Exit.offsIP as u32,
        }
    } else if tag == ffi::IRStmtTag::Ist_NoOp.0 {
        IRStmt::NoOp
    } else if tag == ffi::IRStmtTag::Ist_AbiHint.0 {
        IRStmt::AbiHint {
            base: Box::new(marshal_expr(ist.AbiHint.base)),
            len: ist.AbiHint.len as u32,
            nia: Box::new(marshal_expr(ist.AbiHint.nia)),
        }
    } else if tag == ffi::IRStmtTag::Ist_PutI.0 {
        let d = ist.PutI.details;
        IRStmt::PutI {
            descr: marshal_reg_array((*d).descr),
            ix: Box::new(marshal_expr((*d).ix)),
            bias: (*d).bias as u32,
            data: Box::new(marshal_expr((*d).data)),
        }
    } else if tag == ffi::IRStmtTag::Ist_StoreG.0 {
        let d = ist.StoreG.details;
        IRStmt::StoreG {
            addr: Box::new(marshal_expr((*d).addr)),
            data: Box::new(marshal_expr((*d).data)),
            guard: Box::new(marshal_expr((*d).guard)),
            endness: c_endness((*d).end),
        }
    } else if tag == ffi::IRStmtTag::Ist_LoadG.0 {
        let d = ist.LoadG.details;
        IRStmt::LoadG {
            dst: (*d).dst,
            addr: Box::new(marshal_expr((*d).addr)),
            alt: Box::new(marshal_expr((*d).alt)),
            guard: Box::new(marshal_expr((*d).guard)),
            cvt: loadg_op((*d).cvt),
            endness: c_endness((*d).end),
        }
    } else if tag == ffi::IRStmtTag::Ist_CAS.0 {
        let d = ist.CAS.details;
        IRStmt::CAS {
            old_hi: opt_temp((*d).oldHi),
            old_lo: (*d).oldLo,
            addr: Box::new(marshal_expr((*d).addr)),
            expdHi: opt_expr((*d).expdHi),
            expdLo: Box::new(marshal_expr((*d).expdLo)),
            dataHi: opt_expr((*d).dataHi),
            dataLo: Box::new(marshal_expr((*d).dataLo)),
            endness: c_endness((*d).end),
        }
    } else if tag == ffi::IRStmtTag::Ist_LLSC.0 {
        IRStmt::LLSC {
            storedata: opt_expr(ist.LLSC.storedata),
            result: ist.LLSC.result,
            addr: Box::new(marshal_expr(ist.LLSC.addr)),
            endness: c_endness(ist.LLSC.end),
        }
    } else if tag == ffi::IRStmtTag::Ist_MBE.0 {
        IRStmt::MBE(mbe_event(ist.MBE.event))
    } else if tag == ffi::IRStmtTag::Ist_Dirty.0 {
        let d = ist.Dirty.details;
        IRStmt::Dirty(IRDirty {
            cee: marshal_callee((*d).cee),
            guard: opt_expr((*d).guard),
            tmp: opt_temp((*d).tmp),
            mFx: dirty_fx((*d).mFx),
            mAddr: opt_expr((*d).mAddr),
            mSize: (*d).mSize as u32,
            nFxState: (*d).nFxState as u32,
            args: marshal_expr_vec((*d).args),
        })
    } else {
        IRStmt::NoOp
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
unsafe fn marshal_irsb(irsb: *const ffi::IRSB, addr: u64) -> IRSB {
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
        arch: VexArch::AMD64,
        statements,
        next: marshal_expr((*irsb).next),
        jumpkind: c_jumpkind((*irsb).jumpkind),
        offsIP: (*irsb).offsIP as u32,
        tyenv: marshal_tyenv((*irsb).tyenv),
    }
}

#[cfg(test)]
#[path = "libvex_lifter_tests.rs"]
mod tests;

// Corpus IRSB parity gate (native vs pyvex-serialized) — the real Stage-1 gate.
#[cfg(test)]
#[path = "libvex_corpus_tests.rs"]
mod corpus_tests;
