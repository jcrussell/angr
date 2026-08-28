//! Bridge for deserializing pyvex IRSB JSON into Rust IRSB.
//!
//! This module provides functionality to convert pyvex's JSON-serialized IRSB
//! representation into Rust's native IRSB types.

use serde::Deserialize;

use super::ir::{
    DirtyFx, IRCallee, IRConst, IRDirty, IRExpr, IRLoadGOp, IRRegArray, IRSB, IRStmt, IRType,
    MBusEvent, TypeEnv, VexArch,
};
use super::opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type_or_log};

/// Error type for IRSB deserialization.
///
/// Every variant here is reachable, and the split follows who does the
/// checking. `JsonError` covers every shape/field failure, because the
/// `PyVexIRSB` / `PyVexStmt` / `PyVexExpr` tree is decoded by
/// `#[derive(Deserialize)]` rather than a hand-rolled walker — a missing field
/// or a wrong JSON type is serde's error, not one we construct, so no
/// hand-rolled missing-field/invalid-type variant belongs here. `InvalidArch`
/// is raised by `parse_arch`, validation that happens after serde has handed
/// back a plain `String`.
#[derive(Debug, thiserror::Error)]
pub enum DeserializeError {
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Invalid architecture: {0}")]
    InvalidArch(String),
}

/// JSON representation of a pyvex IRSB.
#[derive(Debug, Deserialize)]
pub struct PyVexIRSB {
    pub addr: u64,
    pub arch: String,
    pub statements: Vec<PyVexStmt>,
    pub next: PyVexExpr,
    pub jumpkind: String,
    #[serde(rename = "offsIP")]
    pub offs_ip: u32,
    pub tyenv: PyVexTypeEnv,
}

/// JSON representation of pyvex type environment.
#[derive(Debug, Deserialize)]
pub struct PyVexTypeEnv {
    pub types: Vec<String>,
}

/// JSON representation of a pyvex statement.
#[derive(Debug, Deserialize)]
#[serde(tag = "tag")]
pub enum PyVexStmt {
    #[serde(rename = "Ist_NoOp")]
    NoOp,

    #[serde(rename = "Ist_IMark")]
    IMark { addr: u64, len: u32, delta: u8 },

    #[serde(rename = "Ist_AbiHint")]
    AbiHint {
        base: Box<PyVexExpr>,
        len: u32,
        nia: Box<PyVexExpr>,
    },

    #[serde(rename = "Ist_Put")]
    Put { offset: u32, data: PyVexExpr },

    #[serde(rename = "Ist_PutI")]
    PutI {
        descr: PyVexRegArray,
        ix: Box<PyVexExpr>,
        bias: u32,
        data: Box<PyVexExpr>,
    },

    #[serde(rename = "Ist_WrTmp")]
    WrTmp { tmp: u32, data: PyVexExpr },

    #[serde(rename = "Ist_Store")]
    Store {
        addr: PyVexExpr,
        data: PyVexExpr,
        end: String,
    },

    #[serde(rename = "Ist_StoreG")]
    StoreG {
        addr: Box<PyVexExpr>,
        data: Box<PyVexExpr>,
        guard: Box<PyVexExpr>,
        end: String,
    },

    #[serde(rename = "Ist_LoadG")]
    LoadG {
        dst: u32,
        addr: Box<PyVexExpr>,
        alt: Box<PyVexExpr>,
        guard: Box<PyVexExpr>,
        cvt: String,
        end: String,
    },

    #[serde(rename = "Ist_CAS")]
    CAS {
        #[serde(rename = "oldHi")]
        old_hi: Option<u32>,
        #[serde(rename = "oldLo")]
        old_lo: u32,
        addr: Box<PyVexExpr>,
        #[serde(rename = "expdHi")]
        expd_hi: Option<Box<PyVexExpr>>,
        #[serde(rename = "expdLo")]
        expd_lo: Box<PyVexExpr>,
        #[serde(rename = "dataHi")]
        data_hi: Option<Box<PyVexExpr>>,
        #[serde(rename = "dataLo")]
        data_lo: Box<PyVexExpr>,
        end: String,
    },

    #[serde(rename = "Ist_LLSC")]
    LLSC {
        storedata: Option<Box<PyVexExpr>>,
        result: u32,
        addr: Box<PyVexExpr>,
        end: String,
    },

    #[serde(rename = "Ist_MBE")]
    MBE { event: String },

    #[serde(rename = "Ist_Dirty")]
    Dirty {
        cee: PyVexCallee,
        guard: Option<Box<PyVexExpr>>,
        tmp: Option<u32>,
        #[serde(rename = "mFx")]
        m_fx: String,
        #[serde(rename = "mAddr")]
        m_addr: Option<Box<PyVexExpr>>,
        #[serde(rename = "mSize")]
        m_size: u32,
        #[serde(rename = "nFxState")]
        n_fx_state: u32,
        args: Vec<PyVexExpr>,
    },

    #[serde(rename = "Ist_Exit")]
    Exit {
        guard: PyVexExpr,
        dst: PyVexConst,
        jk: String,
        #[serde(rename = "offsIP")]
        offs_ip: u32,
    },
}

/// JSON representation of a pyvex expression.
#[derive(Debug, Deserialize)]
#[serde(tag = "tag")]
pub enum PyVexExpr {
    #[serde(rename = "Iex_Const")]
    Const { con: PyVexConst },

    #[serde(rename = "Iex_RdTmp")]
    RdTmp { tmp: u32 },

    #[serde(rename = "Iex_Get")]
    Get { offset: u32, ty: String },

    #[serde(rename = "Iex_GetI")]
    GetI {
        descr: PyVexRegArray,
        ix: Box<PyVexExpr>,
        bias: u32,
    },

    #[serde(rename = "Iex_Load")]
    Load {
        addr: Box<PyVexExpr>,
        ty: String,
        end: String,
    },

    #[serde(rename = "Iex_Unop")]
    Unop { op: String, arg: Box<PyVexExpr> },

    #[serde(rename = "Iex_Binop")]
    Binop {
        op: String,
        args: (Box<PyVexExpr>, Box<PyVexExpr>),
    },

    #[serde(rename = "Iex_Triop")]
    Triop {
        op: String,
        args: (Box<PyVexExpr>, Box<PyVexExpr>, Box<PyVexExpr>),
    },

    #[serde(rename = "Iex_Qop")]
    Qop {
        op: String,
        args: (
            Box<PyVexExpr>,
            Box<PyVexExpr>,
            Box<PyVexExpr>,
            Box<PyVexExpr>,
        ),
    },

    #[serde(rename = "Iex_ITE")]
    ITE {
        cond: Box<PyVexExpr>,
        iftrue: Box<PyVexExpr>,
        iffalse: Box<PyVexExpr>,
    },

    #[serde(rename = "Iex_CCall")]
    CCall {
        cee: PyVexCallee,
        retty: String,
        args: Vec<PyVexExpr>,
    },

    #[serde(rename = "Iex_VECRET")]
    VECRET,

    #[serde(rename = "Iex_GSPTR")]
    GSPTR,
}

/// JSON representation of a pyvex constant.
#[derive(Debug, Deserialize)]
#[serde(tag = "tag")]
pub enum PyVexConst {
    #[serde(rename = "Ico_U1")]
    U1 { value: bool },

    #[serde(rename = "Ico_U8")]
    U8 { value: u8 },

    #[serde(rename = "Ico_U16")]
    U16 { value: u16 },

    #[serde(rename = "Ico_U32")]
    U32 { value: u32 },

    #[serde(rename = "Ico_U64")]
    U64 { value: u64 },

    #[serde(rename = "Ico_U128")]
    U128 { low: u64, high: u64 },

    #[serde(rename = "Ico_F32")]
    F32 { value: f32 },

    #[serde(rename = "Ico_F32i")]
    F32i { value: u32 },

    #[serde(rename = "Ico_F64")]
    F64 { value: f64 },

    #[serde(rename = "Ico_F64i")]
    F64i { value: u64 },

    #[serde(rename = "Ico_V128")]
    V128 { low: u64, high: u64 },

    #[serde(rename = "Ico_V256")]
    V256 { value: [u64; 4] },
}

/// JSON representation of a pyvex register array descriptor.
#[derive(Debug, Deserialize)]
pub struct PyVexRegArray {
    pub base: u32,
    #[serde(rename = "elemTy")]
    pub elem_ty: String,
    #[serde(rename = "nElems")]
    pub n_elems: u32,
}

/// JSON representation of a pyvex callee.
#[derive(Debug, Deserialize)]
pub struct PyVexCallee {
    pub name: String,
    pub addr: u64,
    #[serde(rename = "mcx_mask")]
    pub mcx_mask: u32,
}

/// Convert a pyvex architecture string to VexArch.
fn parse_arch(arch_str: &str) -> Result<VexArch, DeserializeError> {
    match arch_str.to_lowercase().as_str() {
        "x86" | "vexarchx86" => Ok(VexArch::X86),
        "amd64" | "x86_64" | "x64" | "vexarchamd64" => Ok(VexArch::AMD64),
        "arm" | "armel" | "armhf" | "vexarcharm" => Ok(VexArch::ARM),
        "arm64" | "aarch64" | "vexarcharm64" => Ok(VexArch::ARM64),
        "mips32" | "mips" | "vexarchmips32" => Ok(VexArch::MIPS32),
        "mips64" | "vexarchmips64" => Ok(VexArch::MIPS64),
        "ppc32" | "powerpc" | "vexarchppc32" => Ok(VexArch::PPC32),
        "ppc64" | "powerpc64" | "vexarchppc64" => Ok(VexArch::PPC64),
        "s390x" | "vexarchs390x" => Ok(VexArch::S390X),
        _ => Err(DeserializeError::InvalidArch(arch_str.to_string())),
    }
}

/// Convert a pyvex constant to Rust IRConst.
fn convert_const(c: &PyVexConst) -> IRConst {
    match c {
        PyVexConst::U1 { value } => IRConst::U1(*value),
        PyVexConst::U8 { value } => IRConst::U8(*value),
        PyVexConst::U16 { value } => IRConst::U16(*value),
        PyVexConst::U32 { value } => IRConst::U32(*value),
        PyVexConst::U64 { value } => IRConst::U64(*value),
        PyVexConst::U128 { low, high } => {
            // Reconstruct u128 from low/high u64 pair
            let value = (*low as u128) | ((*high as u128) << 64);
            IRConst::U128(value)
        }
        PyVexConst::F32 { value } => IRConst::F32(*value),
        PyVexConst::F32i { value } => IRConst::F32(f32::from_bits(*value)),
        PyVexConst::F64 { value } => IRConst::F64(*value),
        PyVexConst::F64i { value } => IRConst::F64(f64::from_bits(*value)),
        PyVexConst::V128 { low, high } => {
            // Reconstruct u128 from low/high u64 pair
            let value = (*low as u128) | ((*high as u128) << 64);
            IRConst::V128(value)
        }
        PyVexConst::V256 { value } => IRConst::V256(*value),
    }
}

/// Convert a pyvex constant to u64 (for Exit dst).
fn const_to_u64(c: &PyVexConst) -> u64 {
    match c {
        PyVexConst::U1 { value } => *value as u64,
        PyVexConst::U8 { value } => *value as u64,
        PyVexConst::U16 { value } => *value as u64,
        PyVexConst::U32 { value } => *value as u64,
        PyVexConst::U64 { value } => *value,
        PyVexConst::U128 { low, .. } => *low, // Take low 64 bits
        PyVexConst::F32 { value } => value.to_bits() as u64,
        PyVexConst::F32i { value } => *value as u64,
        PyVexConst::F64 { value } => value.to_bits(),
        PyVexConst::F64i { value } => *value,
        PyVexConst::V128 { low, .. } => *low, // Take low 64 bits
        PyVexConst::V256 { value } => value[0],
    }
}

/// Convert a pyvex expression to Rust IRExpr.
fn convert_expr(e: &PyVexExpr) -> IRExpr {
    match e {
        PyVexExpr::Const { con } => IRExpr::Const(convert_const(con)),

        PyVexExpr::RdTmp { tmp } => IRExpr::RdTmp(*tmp),

        PyVexExpr::Get { offset, ty } => IRExpr::Get {
            offset: *offset,
            ty: parse_type_or_log(ty, "IRExpr::Get.ty"),
        },

        PyVexExpr::GetI { descr, ix, bias } => IRExpr::GetI {
            descr: convert_reg_array(descr),
            ix: Box::new(convert_expr(ix)),
            bias: *bias,
        },

        PyVexExpr::Load { addr, ty, end } => IRExpr::Load {
            addr: Box::new(convert_expr(addr)),
            ty: parse_type_or_log(ty, "IRExpr::Load.ty"),
            endness: parse_endness(end),
        },

        PyVexExpr::Unop { op, arg } => IRExpr::Unop {
            op: parse_opcode(op),
            arg: Box::new(convert_expr(arg)),
        },

        PyVexExpr::Binop { op, args } => IRExpr::Binop {
            op: parse_opcode(op),
            left: Box::new(convert_expr(&args.0)),
            right: Box::new(convert_expr(&args.1)),
        },

        PyVexExpr::Triop { op, args } => IRExpr::Triop {
            op: parse_opcode(op),
            arg1: Box::new(convert_expr(&args.0)),
            arg2: Box::new(convert_expr(&args.1)),
            arg3: Box::new(convert_expr(&args.2)),
        },

        PyVexExpr::Qop { op, args } => IRExpr::Qop {
            op: parse_opcode(op),
            arg1: Box::new(convert_expr(&args.0)),
            arg2: Box::new(convert_expr(&args.1)),
            arg3: Box::new(convert_expr(&args.2)),
            arg4: Box::new(convert_expr(&args.3)),
        },

        PyVexExpr::ITE {
            cond,
            iftrue,
            iffalse,
        } => IRExpr::ITE {
            cond: Box::new(convert_expr(cond)),
            iftrue: Box::new(convert_expr(iftrue)),
            iffalse: Box::new(convert_expr(iffalse)),
        },

        PyVexExpr::CCall { cee, retty, args } => IRExpr::CCall {
            cee: convert_callee(cee),
            retty: parse_type_or_log(retty, "IRExpr::CCall.retty"),
            args: args.iter().map(convert_expr).collect(),
        },

        PyVexExpr::VECRET => IRExpr::VECRET,
        PyVexExpr::GSPTR => IRExpr::GSPTR,
    }
}

/// Convert a pyvex register array to Rust IRRegArray.
fn convert_reg_array(r: &PyVexRegArray) -> IRRegArray {
    IRRegArray {
        base: r.base,
        elemTy: parse_type_or_log(&r.elem_ty, "IRRegArray.elemTy"),
        nElems: r.n_elems,
    }
}

/// Convert a pyvex callee to Rust IRCallee.
fn convert_callee(c: &PyVexCallee) -> IRCallee {
    IRCallee {
        name: c.name.clone(),
        addr: c.addr,
        mcx_mask: c.mcx_mask,
    }
}

/// Convert LoadG cvt string to IRLoadGOp.
fn parse_loadg_op(cvt: &str) -> IRLoadGOp {
    match cvt {
        "ILGop_IdentV128" | "ILGop_Ident64" | "ILGop_Ident32" => IRLoadGOp::Identity,
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

/// Convert MBE event string to MBusEvent.
fn parse_mbe_event(event: &str) -> MBusEvent {
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

/// Convert dirty Fx string to DirtyFx.
fn parse_dirty_fx(fx: &str) -> DirtyFx {
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

/// Convert a pyvex statement to Rust IRStmt.
fn convert_stmt(s: &PyVexStmt) -> IRStmt {
    match s {
        PyVexStmt::NoOp => IRStmt::NoOp,

        PyVexStmt::IMark { addr, len, delta } => IRStmt::IMark {
            addr: *addr,
            len: *len,
            delta: *delta,
        },

        PyVexStmt::AbiHint { base, len, nia } => IRStmt::AbiHint {
            base: Box::new(convert_expr(base)),
            len: *len,
            nia: Box::new(convert_expr(nia)),
        },

        PyVexStmt::Put { offset, data } => IRStmt::Put {
            offset: *offset,
            data: convert_expr(data),
        },

        PyVexStmt::PutI {
            descr,
            ix,
            bias,
            data,
        } => IRStmt::PutI {
            descr: convert_reg_array(descr),
            ix: Box::new(convert_expr(ix)),
            bias: *bias,
            data: Box::new(convert_expr(data)),
        },

        PyVexStmt::WrTmp { tmp, data } => IRStmt::WrTmp {
            tmp: *tmp,
            data: convert_expr(data),
        },

        PyVexStmt::Store { addr, data, end } => IRStmt::Store {
            addr: convert_expr(addr),
            data: convert_expr(data),
            endness: parse_endness(end),
        },

        PyVexStmt::StoreG {
            addr,
            data,
            guard,
            end,
        } => IRStmt::StoreG {
            addr: Box::new(convert_expr(addr)),
            data: Box::new(convert_expr(data)),
            guard: Box::new(convert_expr(guard)),
            endness: parse_endness(end),
        },

        PyVexStmt::LoadG {
            dst,
            addr,
            alt,
            guard,
            cvt,
            end,
        } => IRStmt::LoadG {
            dst: *dst,
            addr: Box::new(convert_expr(addr)),
            alt: Box::new(convert_expr(alt)),
            guard: Box::new(convert_expr(guard)),
            cvt: parse_loadg_op(cvt),
            endness: parse_endness(end),
        },

        PyVexStmt::CAS {
            old_hi,
            old_lo,
            addr,
            expd_hi,
            expd_lo,
            data_hi,
            data_lo,
            end,
        } => IRStmt::CAS {
            old_hi: *old_hi,
            old_lo: *old_lo,
            addr: Box::new(convert_expr(addr)),
            expdHi: expd_hi.as_ref().map(|e| Box::new(convert_expr(e))),
            expdLo: Box::new(convert_expr(expd_lo)),
            dataHi: data_hi.as_ref().map(|e| Box::new(convert_expr(e))),
            dataLo: Box::new(convert_expr(data_lo)),
            endness: parse_endness(end),
        },

        PyVexStmt::LLSC {
            storedata,
            result,
            addr,
            end,
        } => IRStmt::LLSC {
            storedata: storedata.as_ref().map(|e| Box::new(convert_expr(e))),
            result: *result,
            addr: Box::new(convert_expr(addr)),
            endness: parse_endness(end),
        },

        PyVexStmt::MBE { event } => IRStmt::MBE(parse_mbe_event(event)),

        PyVexStmt::Dirty {
            cee,
            guard,
            tmp,
            m_fx,
            m_addr,
            m_size,
            n_fx_state,
            args,
        } => IRStmt::Dirty(IRDirty {
            cee: convert_callee(cee),
            guard: guard.as_ref().map(|g| Box::new(convert_expr(g))),
            tmp: *tmp,
            mFx: parse_dirty_fx(m_fx),
            mAddr: m_addr.as_ref().map(|a| Box::new(convert_expr(a))),
            mSize: *m_size,
            nFxState: *n_fx_state,
            args: args.iter().map(convert_expr).collect(),
        }),

        PyVexStmt::Exit {
            guard,
            dst,
            jk,
            offs_ip,
        } => IRStmt::Exit {
            guard: convert_expr(guard),
            dst: const_to_u64(dst),
            jk: parse_jumpkind(jk),
            offsIP: *offs_ip,
        },
    }
}

/// Convert type environment from pyvex format.
fn convert_tyenv(tyenv: &PyVexTypeEnv) -> TypeEnv {
    let types: Vec<IRType> = tyenv
        .types
        .iter()
        .map(|t| parse_type_or_log(t, "IRSB tyenv entry"))
        .collect();
    TypeEnv { types }
}

/// Deserialize a pyvex IRSB from JSON.
pub fn deserialize_irsb(json: &str) -> Result<IRSB, DeserializeError> {
    let pyvex: PyVexIRSB = serde_json::from_str(json)?;
    convert_pyvex_irsb(&pyvex)
}

/// Convert a PyVexIRSB to Rust IRSB.
pub fn convert_pyvex_irsb(pyvex: &PyVexIRSB) -> Result<IRSB, DeserializeError> {
    let arch = parse_arch(&pyvex.arch)?;

    let statements: Vec<IRStmt> = pyvex.statements.iter().map(convert_stmt).collect();

    let next = convert_expr(&pyvex.next);
    let jumpkind = parse_jumpkind(&pyvex.jumpkind);
    let tyenv = convert_tyenv(&pyvex.tyenv);

    Ok(IRSB {
        addr: pyvex.addr,
        arch,
        statements,
        next,
        jumpkind,
        offsIP: pyvex.offs_ip,
        tyenv,
    })
}

test_submod!("pyvex_bridge_tests.rs" => tests);
