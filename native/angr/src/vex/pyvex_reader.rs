//! Read a live pyvex `IRSB` object into a Rust [`IRSB`].
//!
//! The Python `lift_block` callback returns `project.factory.block(addr).vex`
//! itself, and this module walks it over PyO3 with the GIL held: statements
//! and expressions dispatch on their pyvex class name, and enum-valued fields
//! (ops, types, endness, jumpkinds) are pyvex's strings, parsed by the same
//! `opcode_map.rs` parsers the native lifter uses. It replaced a JSON
//! round-trip (angr-mx00p.33) whose serializer and serde schema drifted apart
//! on Python-built statements (angr-mx00p.44).
//!
//! Field conventions match [`super::libvex_lifter`], so a block reads the same
//! whichever path lifted it (`test_native_lift_parity.py`): an `IRTemp_INVALID`
//! temp is `None`, `IRCallee.addr` is 0, and a `None` that libVEX would have
//! written as 0 (a Spotter-built `Dirty`'s `mSize` / `nFxState`) reads as 0.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::ir::{
    DirtyFx, IRCallee, IRConst, IRDirty, IRExpr, IRLoadGOp, IRRegArray, IRSB, IRStmt, IRType,
    MBusEvent, TypeEnv, VexArch,
};
use super::opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type_or_log};

/// VEX's `IRTemp_INVALID` sentinel, as pyvex leaves it in `CAS.oldHi` and
/// `Dirty.tmp`.
const IRTEMP_INVALID: u32 = 0xFFFF_FFFF;

/// Read a pyvex `IRSB` into a Rust [`IRSB`]. Errors name the pyvex class or
/// field that did not read.
pub fn read_irsb(irsb: &Bound<'_, PyAny>) -> PyResult<IRSB> {
    let arch_name: String = irsb.getattr("arch")?.getattr("name")?.extract()?;
    let arch = parse_arch(&arch_name)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported IRSB arch {arch_name:?}")))?;
    let statements = irsb
        .getattr("statements")?
        .try_iter()?
        .map(|s| read_stmt(&s?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(IRSB {
        addr: irsb.getattr("addr")?.extract()?,
        arch,
        statements,
        next: read_expr(&irsb.getattr("next")?)?,
        jumpkind: parse_jumpkind(&string(irsb, "jumpkind")?),
        offsIP: irsb.getattr("offsIP")?.extract()?,
        tyenv: read_tyenv(&irsb.getattr("tyenv")?)?,
    })
}

/// pyvex `arch.name` → [`VexArch`].
pub(crate) fn parse_arch(name: &str) -> Option<VexArch> {
    Some(match name.to_lowercase().as_str() {
        "x86" => VexArch::X86,
        "amd64" | "x86_64" | "x64" => VexArch::AMD64,
        "arm" | "armel" | "armhf" | "armcortexm" => VexArch::ARM,
        "arm64" | "aarch64" => VexArch::ARM64,
        "mips32" | "mips" => VexArch::MIPS32,
        "mips64" => VexArch::MIPS64,
        "ppc32" | "powerpc" => VexArch::PPC32,
        "ppc64" | "powerpc64" => VexArch::PPC64,
        "s390x" => VexArch::S390X,
        _ => return None,
    })
}

fn class_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name()?.extract()
}

/// `str(obj.<field>)`: pyvex exposes enum fields as `Ity_*` / `Iend_*` / ...
/// strings.
fn string(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<String> {
    Ok(obj.getattr(field)?.str()?.to_string())
}

fn expr_field(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<IRExpr> {
    read_expr(&obj.getattr(field)?)
}

fn boxed(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Box<IRExpr>> {
    expr_field(obj, field).map(Box::new)
}

/// An expression field that may be `None`.
fn opt_boxed(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Option<Box<IRExpr>>> {
    let value = obj.getattr(field)?;
    if value.is_none() {
        Ok(None)
    } else {
        read_expr(&value).map(|e| Some(Box::new(e)))
    }
}

/// An `IRTemp` field that may be `None` or `IRTemp_INVALID`.
fn opt_temp(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Option<u32>> {
    let value: Option<u32> = obj.getattr(field)?.extract()?;
    Ok(value.filter(|t| *t != IRTEMP_INVALID))
}

/// An integer field libVEX always writes but a Python-built statement may
/// leave `None`.
fn int_or_zero(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<u32> {
    Ok(obj.getattr(field)?.extract::<Option<u32>>()?.unwrap_or(0))
}

fn args(obj: &Bound<'_, PyAny>) -> PyResult<Vec<IRExpr>> {
    obj.getattr("args")?
        .try_iter()?
        .map(|a| read_expr(&a?))
        .collect()
}

fn read_tyenv(tyenv: &Bound<'_, PyAny>) -> PyResult<TypeEnv> {
    if tyenv.is_none() {
        return Ok(TypeEnv { types: Vec::new() });
    }
    let types = tyenv
        .getattr("types")?
        .try_iter()?
        .map(|t| {
            Ok(parse_type_or_log(
                &t?.str()?.to_string(),
                "IRSB tyenv entry",
            ))
        })
        .collect::<PyResult<Vec<IRType>>>()?;
    Ok(TypeEnv { types })
}

fn read_reg_array(descr: &Bound<'_, PyAny>) -> PyResult<IRRegArray> {
    Ok(IRRegArray {
        base: descr.getattr("base")?.extract()?,
        elemTy: parse_type_or_log(&string(descr, "elemTy")?, "IRRegArray.elemTy"),
        nElems: descr.getattr("nElems")?.extract()?,
    })
}

fn read_callee(cee: &Bound<'_, PyAny>) -> PyResult<IRCallee> {
    Ok(IRCallee {
        name: cee.getattr("name")?.extract()?,
        // The helper's host address is meaningless here; CCalls dispatch by
        // name. Same normalization as the native lifter.
        addr: 0,
        mcx_mask: cee
            .getattr("mcx_mask")?
            .extract::<Option<u32>>()?
            .unwrap_or(0),
    })
}

/// Read a pyvex `IRConst` (`U1` ... `V256`).
fn read_const(con: &Bound<'_, PyAny>) -> PyResult<IRConst> {
    let value = con.getattr("value")?;
    Ok(match class_name(con)?.as_str() {
        // C `Bool` is an unsigned char, so a Python-built U1 carries an int.
        "U1" => IRConst::U1(value.is_truthy()?),
        "U8" => IRConst::U8(value.extract()?),
        "U16" => IRConst::U16(value.extract()?),
        "U32" => IRConst::U32(value.extract()?),
        "U64" => IRConst::U64(value.extract()?),
        "U128" => IRConst::U128(value.extract()?),
        "F32" => IRConst::F32(value.extract()?),
        "F32i" => IRConst::F32(f32::from_bits(value.extract()?)),
        "F64" => IRConst::F64(value.extract()?),
        "F64i" => IRConst::F64(f64::from_bits(value.extract()?)),
        "V128" => IRConst::V128(value.extract()?),
        "V256" => {
            let bytes = value.call_method1("to_bytes", (32, "little"))?;
            let bytes = bytes.cast::<PyBytes>()?.as_bytes();
            let mut limbs = [0u64; 4];
            let (words, _) = bytes.as_chunks::<8>();
            for (limb, word) in limbs.iter_mut().zip(words) {
                *limb = u64::from_le_bytes(*word);
            }
            IRConst::V256(limbs)
        }
        "V512" => {
            // SILENT(cat-c): the engine has no 512-bit vector type, so an
            // AVX-512 constant cannot be represented; the zero stand-in is
            // wrong for any nonzero pattern. Mirrors the native lifter's
            // `marshal_const` (angr-mx00p.5).
            log::warn!("pyvex V512 constant is unsupported; assuming Ico_U64(0)");
            IRConst::U64(0)
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pyvex constant class {other:?}"
            )));
        }
    })
}

/// The low 64 bits of a constant, as an `Exit` destination.
fn const_to_u64(con: &IRConst) -> u64 {
    match *con {
        IRConst::U1(v) => v as u64,
        IRConst::U8(v) => v as u64,
        IRConst::U16(v) => v as u64,
        IRConst::U32(v) => v as u64,
        IRConst::U64(v) => v,
        IRConst::U128(v) | IRConst::V128(v) => v as u64,
        IRConst::F32(v) => v.to_bits() as u64,
        IRConst::F64(v) => v.to_bits(),
        IRConst::V256(v) => v[0],
    }
}

/// Read a pyvex `IRExpr`.
pub(crate) fn read_expr(e: &Bound<'_, PyAny>) -> PyResult<IRExpr> {
    Ok(match class_name(e)?.as_str() {
        "Const" => IRExpr::Const(read_const(&e.getattr("con")?)?),
        "RdTmp" => IRExpr::RdTmp(e.getattr("tmp")?.extract()?),
        "Get" => IRExpr::Get {
            offset: e.getattr("offset")?.extract()?,
            ty: parse_type_or_log(&string(e, "ty")?, "IRExpr::Get.ty"),
        },
        "GetI" => IRExpr::GetI {
            descr: read_reg_array(&e.getattr("descr")?)?,
            ix: boxed(e, "ix")?,
            bias: e.getattr("bias")?.extract::<i64>()? as u32,
        },
        "Load" => IRExpr::Load {
            addr: boxed(e, "addr")?,
            ty: parse_type_or_log(&string(e, "ty")?, "IRExpr::Load.ty"),
            endness: parse_endness(&string(e, "end")?),
        },
        kind @ ("Unop" | "Binop" | "Triop" | "Qop") => {
            let op = parse_opcode(&string(e, "op")?);
            let mut a = args(e)?.into_iter().map(Box::new);
            let arity = match kind {
                "Unop" => 1,
                "Binop" => 2,
                "Triop" => 3,
                _ => 4,
            };
            let mut next = || {
                a.next().ok_or_else(|| {
                    PyValueError::new_err(format!("pyvex {kind} has fewer than {arity} args"))
                })
            };
            match kind {
                "Unop" => IRExpr::Unop { op, arg: next()? },
                "Binop" => IRExpr::Binop {
                    op,
                    left: next()?,
                    right: next()?,
                },
                "Triop" => IRExpr::Triop {
                    op,
                    arg1: next()?,
                    arg2: next()?,
                    arg3: next()?,
                },
                _ => IRExpr::Qop {
                    op,
                    arg1: next()?,
                    arg2: next()?,
                    arg3: next()?,
                    arg4: next()?,
                },
            }
        }
        "ITE" => IRExpr::ITE {
            cond: boxed(e, "cond")?,
            iftrue: boxed(e, "iftrue")?,
            iffalse: boxed(e, "iffalse")?,
        },
        "CCall" => IRExpr::CCall {
            cee: read_callee(&e.getattr("cee")?)?,
            retty: parse_type_or_log(&string(e, "retty")?, "IRExpr::CCall.retty"),
            args: args(e)?,
        },
        "VECRET" => IRExpr::VECRET,
        "GSPTR" => IRExpr::GSPTR,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pyvex expression class {other:?}"
            )));
        }
    })
}

/// pyvex `LoadG.cvt` → [`IRLoadGOp`].
pub(crate) fn parse_loadg_op(cvt: &str) -> IRLoadGOp {
    match cvt {
        "ILGop_IdentV128" | "ILGop_Ident64" | "ILGop_Ident32" | "ILGop_Ident16"
        | "ILGop_Ident8" => IRLoadGOp::Identity,
        // VEX defines only the *to32 widening ops; the *to64 forms are accepted
        // defensively in case a future lifter emits them.
        "ILGop_8Uto32" => IRLoadGOp::WidenZ { src_bits: 8 },
        "ILGop_8Sto32" => IRLoadGOp::WidenS { src_bits: 8 },
        "ILGop_16Uto32" | "ILGop_16Uto64" => IRLoadGOp::WidenZ { src_bits: 16 },
        "ILGop_16Sto32" | "ILGop_16Sto64" => IRLoadGOp::WidenS { src_bits: 16 },
        "ILGop_32Uto64" => IRLoadGOp::WidenZ { src_bits: 32 },
        "ILGop_32Sto64" => IRLoadGOp::WidenS { src_bits: 32 },
        _ => IRLoadGOp::Unknown,
    }
}

/// pyvex `MBE.event` → [`MBusEvent`].
pub(crate) fn parse_mbe_event(event: &str) -> MBusEvent {
    match event {
        "Imbe_Fence" => MBusEvent::Fence,
        "Imbe_SFence" => MBusEvent::SFence,
        "Imbe_LFence" => MBusEvent::LFence,
        "Imbe_MFence" => MBusEvent::MFence,
        // SILENT(cat-a): the payload is inert. `Interpreter`'s `IRStmt::MBE`
        // arm discards it (`statements.rs`, `IRStmt::MBE(_) => Continue`), and
        // the FFI sibling `libvex_lifter::mbe_event` collapses *every*
        // discriminant to `Fence` unconditionally — this engine treats each
        // barrier as a full fence. Widening the fallback to a log would warn
        // about a value nothing reads.
        _ => MBusEvent::Fence,
    }
}

/// pyvex `Dirty.mFx` → [`DirtyFx`].
pub(crate) fn parse_dirty_fx(fx: &str) -> DirtyFx {
    let parsed = match fx {
        "Ifx_None" => Some(DirtyFx::None),
        "Ifx_Read" => Some(DirtyFx::Read),
        "Ifx_Write" => Some(DirtyFx::Write),
        "Ifx_Modify" => Some(DirtyFx::Modify),
        _ => None,
    };
    // cat-b rather than cat-c only because `IRDirty::mFx` is write-only today
    // (see its field comment in `vex::ir::descriptors`): claiming a helper has
    // no memory effect loses information but cannot yet produce a wrong
    // answer. Promote to cat-c when memory-effect-aware code invalidation
    // starts reading it.
    silent_default!(
        cat_b,
        parsed,
        DirtyFx::None,
        "Unknown pyvex dirty-call effect string {fx:?}; assuming Ifx_None — \
         the helper's memory effect is lost"
    )
}

/// Read a pyvex `IRStmt`.
fn read_stmt(s: &Bound<'_, PyAny>) -> PyResult<IRStmt> {
    Ok(match class_name(s)?.as_str() {
        "NoOp" => IRStmt::NoOp,
        "IMark" => IRStmt::IMark {
            addr: s.getattr("addr")?.extract()?,
            len: s.getattr("len")?.extract()?,
            delta: s.getattr("delta")?.extract()?,
        },
        "AbiHint" => IRStmt::AbiHint {
            base: boxed(s, "base")?,
            len: s.getattr("len")?.extract()?,
            nia: boxed(s, "nia")?,
        },
        "Put" => IRStmt::Put {
            offset: s.getattr("offset")?.extract()?,
            data: expr_field(s, "data")?,
        },
        "PutI" => IRStmt::PutI {
            descr: read_reg_array(&s.getattr("descr")?)?,
            ix: boxed(s, "ix")?,
            bias: s.getattr("bias")?.extract::<i64>()? as u32,
            data: boxed(s, "data")?,
        },
        "WrTmp" => IRStmt::WrTmp {
            tmp: s.getattr("tmp")?.extract()?,
            data: expr_field(s, "data")?,
        },
        "Store" => IRStmt::Store {
            addr: expr_field(s, "addr")?,
            data: expr_field(s, "data")?,
            endness: parse_endness(&string(s, "end")?),
        },
        "StoreG" => IRStmt::StoreG {
            addr: boxed(s, "addr")?,
            data: boxed(s, "data")?,
            guard: boxed(s, "guard")?,
            endness: parse_endness(&string(s, "end")?),
        },
        "LoadG" => IRStmt::LoadG {
            dst: s.getattr("dst")?.extract()?,
            addr: boxed(s, "addr")?,
            alt: boxed(s, "alt")?,
            guard: boxed(s, "guard")?,
            cvt: parse_loadg_op(&string(s, "cvt")?),
            endness: parse_endness(&string(s, "end")?),
        },
        "CAS" => IRStmt::CAS {
            old_hi: opt_temp(s, "oldHi")?,
            old_lo: s.getattr("oldLo")?.extract()?,
            addr: boxed(s, "addr")?,
            expdHi: opt_boxed(s, "expdHi")?,
            expdLo: boxed(s, "expdLo")?,
            dataHi: opt_boxed(s, "dataHi")?,
            dataLo: boxed(s, "dataLo")?,
            endness: parse_endness(&string(s, "end")?),
        },
        "LLSC" => IRStmt::LLSC {
            storedata: opt_boxed(s, "storedata")?,
            result: s.getattr("result")?.extract()?,
            addr: boxed(s, "addr")?,
            endness: parse_endness(&string(s, "end")?),
        },
        "MBE" => IRStmt::MBE(parse_mbe_event(&string(s, "event")?)),
        "Dirty" => {
            let m_fx = s.getattr("mFx")?;
            IRStmt::Dirty(IRDirty {
                cee: read_callee(&s.getattr("cee")?)?,
                guard: opt_boxed(s, "guard")?,
                tmp: opt_temp(s, "tmp")?,
                mFx: if m_fx.is_none() {
                    DirtyFx::None
                } else {
                    parse_dirty_fx(&m_fx.str()?.to_string())
                },
                mAddr: opt_boxed(s, "mAddr")?,
                mSize: int_or_zero(s, "mSize")?,
                nFxState: int_or_zero(s, "nFxState")?,
                args: args(s)?,
            })
        }
        "Exit" => IRStmt::Exit {
            guard: expr_field(s, "guard")?,
            dst: const_to_u64(&read_const(&s.getattr("dst")?)?),
            jk: parse_jumpkind(&string(s, "jk")?),
            offsIP: s.getattr("offsIP")?.extract()?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pyvex statement class {other:?}"
            )));
        }
    })
}

test_submod!("pyvex_reader_tests.rs" => tests);
