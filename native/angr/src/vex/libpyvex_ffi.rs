//! FFI bindings to libpyvex for native VEX lifting.
//!
//! This module provides safe Rust wrappers around the libpyvex C library,
//! enabling native VEX lifting without Python callbacks.
//!
//! This module is only available when the `native-lift` feature is enabled.

#![cfg(feature = "native-lift")]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::{CStr, c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void};
use std::ptr;
use std::sync::Once;

use super::ir::{
    DirtyFx, Endness, IRCallee, IRConst, IRDirty, IRExpr, IRLoadGOp, IROp, IRRegArray, IRSB,
    IRStmt, IRType, JumpKind, MBusEvent, TypeEnv, VexArch,
};
use super::opcode_map;

// ============================================================================
// C Types from libvex_ir.h and pyvex.h
// ============================================================================

/// libVEX arch enum - matches VexArch in libvex.h
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CVexArch {
    Invalid = 0x400,
    X86,
    AMD64,
    ARM,
    ARM64,
    PPC32,
    PPC64,
    S390X,
    MIPS32,
    MIPS64,
    TILEGX,
    RISCV64,
}

impl From<VexArch> for CVexArch {
    fn from(arch: VexArch) -> Self {
        match arch {
            VexArch::X86 => CVexArch::X86,
            VexArch::AMD64 => CVexArch::AMD64,
            VexArch::ARM => CVexArch::ARM,
            VexArch::ARM64 => CVexArch::ARM64,
            VexArch::PPC32 => CVexArch::PPC32,
            VexArch::PPC64 => CVexArch::PPC64,
            VexArch::S390X => CVexArch::S390X,
            VexArch::MIPS32 => CVexArch::MIPS32,
            VexArch::MIPS64 => CVexArch::MIPS64,
        }
    }
}

impl TryFrom<CVexArch> for VexArch {
    type Error = &'static str;

    fn try_from(arch: CVexArch) -> Result<Self, Self::Error> {
        match arch {
            CVexArch::X86 => Ok(VexArch::X86),
            CVexArch::AMD64 => Ok(VexArch::AMD64),
            CVexArch::ARM => Ok(VexArch::ARM),
            CVexArch::ARM64 => Ok(VexArch::ARM64),
            CVexArch::PPC32 => Ok(VexArch::PPC32),
            CVexArch::PPC64 => Ok(VexArch::PPC64),
            CVexArch::S390X => Ok(VexArch::S390X),
            CVexArch::MIPS32 => Ok(VexArch::MIPS32),
            CVexArch::MIPS64 => Ok(VexArch::MIPS64),
            _ => Err("unsupported architecture"),
        }
    }
}

/// VexArchInfo structure - contains architecture details
#[repr(C)]
#[derive(Clone)]
pub struct CVexArchInfo {
    pub hwcaps: c_uint,
    pub endness: c_int, // VexEndness enum
    // Additional fields exist but we only need these for basic lifting
    pub _padding: [u8; 128], // Reserve space for other fields
}

impl Default for CVexArchInfo {
    fn default() -> Self {
        let mut info = CVexArchInfo {
            hwcaps: 0,
            endness: 0x601, // VexEndnessLE
            _padding: [0; 128],
        };
        // For AMD64, enable common capabilities
        // SSE3 | CX16 | LZCNT | AVX | RDTSCP | BMI | AVX2
        info.hwcaps = (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11);
        info
    }
}

impl CVexArchInfo {
    pub fn for_arch(arch: VexArch) -> Self {
        let mut info = Self::default();
        match arch {
            VexArch::AMD64 => {
                info.endness = 0x601; // LE
                info.hwcaps =
                    (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11);
            }
            VexArch::X86 => {
                info.endness = 0x601; // LE
                info.hwcaps = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
            }
            VexArch::ARM | VexArch::ARM64 => {
                info.endness = 0x601; // LE (ARM can be either, default to LE)
                info.hwcaps = 0;
            }
            VexArch::MIPS32
            | VexArch::MIPS64
            | VexArch::PPC32
            | VexArch::PPC64
            | VexArch::S390X => {
                info.endness = 0x602; // BE
                info.hwcaps = 0;
            }
        }
        info
    }
}

/// VexRegisterUpdates enum
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CVexRegisterUpdates {
    Invalid = 0x700,
    SpAtMemAccess,
    UnwindregsAtMemAccess,
    AllregsAtMemAccess,
    AllregsAtEachInsn,
    LdAllregsAtEachInsn,
}

// ============================================================================
// libvex IR C types
// ============================================================================

/// IR type tag
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIRType {
    Invalid = 0x1100,
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
    F16,
    F32,
    F64,
    D32,
    D64,
    D128,
    F128,
    V128,
    V256,
}

impl From<CIRType> for IRType {
    fn from(ty: CIRType) -> Self {
        match ty {
            CIRType::I1 => IRType::I1,
            CIRType::I8 => IRType::I8,
            CIRType::I16 => IRType::I16,
            CIRType::I32 => IRType::I32,
            CIRType::I64 => IRType::I64,
            CIRType::I128 => IRType::I128,
            CIRType::F16 => IRType::F16,
            CIRType::F32 => IRType::F32,
            CIRType::F64 => IRType::F64,
            CIRType::V128 => IRType::V128,
            CIRType::V256 => IRType::V256,
            _ => IRType::I64, // Default fallback
        }
    }
}

/// IREndness enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIREndness {
    Invalid = 0x1200,
    LE,
    BE,
}

impl From<CIREndness> for Endness {
    fn from(e: CIREndness) -> Self {
        match e {
            CIREndness::LE => Endness::Little,
            CIREndness::BE => Endness::Big,
            _ => Endness::Little,
        }
    }
}

/// IR constant tag
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CIRConstTag {
    Ico_U1 = 0x1300,
    Ico_U8,
    Ico_U16,
    Ico_U32,
    Ico_U64,
    Ico_F32,
    Ico_F32i,
    Ico_F64,
    Ico_F64i,
    Ico_V128,
    Ico_V256,
}

/// IR constant union
#[repr(C)]
pub union CIRConstUnion {
    pub u1: c_int, // Bool stored as int
    pub u8_: c_uchar,
    pub u16_: u16,
    pub u32_: u32,
    pub u64_: u64,
    pub f32_: f32,
    pub f32i: u32,
    pub f64_: f64,
    pub f64i: u64,
    pub v128: u16, // V128 stored as 16-bit selector
    pub v256: u32, // V256 stored as 32-bit selector
}

/// IR constant structure
#[repr(C)]
pub struct CIRConst {
    pub tag: CIRConstTag,
    pub con: CIRConstUnion,
}

/// IRJumpKind enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIRJumpKind {
    Invalid = 0x1A00,
    Boring,
    Call,
    Ret,
    ClientReq,
    Yield,
    EmWarn,
    EmFail,
    NoDecode,
    MapFail,
    InvalICache,
    FlushDCache,
    NoRedir,
    SigILL,
    SigTRAP,
    SigSEGV,
    SigBUS,
    SigFPE,
    SigFPE_IntDiv,
    SigFPE_IntOvf,
    Sys_syscall,
    Sys_int32,
    Sys_int128,
    Sys_int129,
    Sys_int130,
    Sys_int145,
    Sys_int210,
    Sys_sysenter,
}

impl From<CIRJumpKind> for JumpKind {
    fn from(jk: CIRJumpKind) -> Self {
        match jk {
            CIRJumpKind::Boring => JumpKind::Boring,
            CIRJumpKind::Call => JumpKind::Call,
            CIRJumpKind::Ret => JumpKind::Ret,
            CIRJumpKind::Sys_syscall => JumpKind::Sys_syscall,
            CIRJumpKind::Sys_int128 => JumpKind::Sys_int128,
            CIRJumpKind::Sys_int129 => JumpKind::Sys_int129,
            CIRJumpKind::Sys_int130 => JumpKind::Sys_int130,
            CIRJumpKind::Sys_int145 => JumpKind::Sys_int145,
            CIRJumpKind::Sys_int210 => JumpKind::Sys_int210,
            CIRJumpKind::Sys_sysenter => JumpKind::Sys_sysenter,
            CIRJumpKind::ClientReq => JumpKind::ClientReq,
            CIRJumpKind::Yield => JumpKind::Yield,
            CIRJumpKind::EmWarn => JumpKind::EmWarn,
            CIRJumpKind::EmFail => JumpKind::EmFail,
            CIRJumpKind::NoDecode => JumpKind::NoDecode,
            CIRJumpKind::MapFail => JumpKind::MapFail,
            CIRJumpKind::InvalICache => JumpKind::InvalICache,
            CIRJumpKind::FlushDCache => JumpKind::FlushDCache,
            _ => JumpKind::Boring,
        }
    }
}

/// IRStmtTag enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIRStmtTag {
    NoOp = 0x1E00,
    IMark,
    AbiHint,
    Put,
    PutI,
    WrTmp,
    Store,
    LoadG,
    StoreG,
    CAS,
    LLSC,
    Dirty,
    MBE,
    Exit,
}

/// IRExprTag enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIRExprTag {
    Binder = 0x1900,
    Get,
    GetI,
    RdTmp,
    Qop,
    Triop,
    Binop,
    Unop,
    Load,
    Const,
    ITE,
    CCall,
    VECRET,
    GSPTR,
}

// Forward declarations for recursive structures
#[repr(C)]
pub struct CIRExpr {
    pub tag: CIRExprTag,
    pub data: CIRExprData,
}

#[repr(C)]
pub union CIRExprData {
    pub binder: c_int,
    pub get: CIRExprGet,
    pub geti: CIRExprGetI,
    pub rdtmp: CIRExprRdTmp,
    pub qop: CIRExprQop,
    pub triop: CIRExprTriop,
    pub binop: CIRExprBinop,
    pub unop: CIRExprUnop,
    pub load: CIRExprLoad,
    pub konst: CIRExprConst,
    pub ite: CIRExprITE,
    pub ccall: CIRExprCCall,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprGet {
    pub offset: c_int,
    pub ty: CIRType,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprGetI {
    pub descr: *mut CIRRegArray,
    pub ix: *mut CIRExpr,
    pub bias: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprRdTmp {
    pub tmp: u32, // IRTemp
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprQop {
    pub details: *mut CIRQop,
}

#[repr(C)]
pub struct CIRQop {
    pub op: c_uint, // IROp
    pub arg1: *mut CIRExpr,
    pub arg2: *mut CIRExpr,
    pub arg3: *mut CIRExpr,
    pub arg4: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprTriop {
    pub details: *mut CIRTriop,
}

#[repr(C)]
pub struct CIRTriop {
    pub op: c_uint, // IROp
    pub arg1: *mut CIRExpr,
    pub arg2: *mut CIRExpr,
    pub arg3: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprBinop {
    pub op: c_uint, // IROp
    pub arg1: *mut CIRExpr,
    pub arg2: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprUnop {
    pub op: c_uint, // IROp
    pub arg: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprLoad {
    pub end: CIREndness,
    pub ty: CIRType,
    pub addr: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprConst {
    pub con: *mut CIRConst,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprITE {
    pub cond: *mut CIRExpr,
    pub iftrue: *mut CIRExpr,
    pub iffalse: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRExprCCall {
    pub cee: *mut CIRCallee,
    pub retty: CIRType,
    pub args: *mut *mut CIRExpr, // NULL-terminated array
}

#[repr(C)]
pub struct CIRCallee {
    pub name: *const c_char,
    pub addr: *mut c_void,
    pub regparms: c_uint,
    pub mcx_mask: c_uint,
}

#[repr(C)]
pub struct CIRRegArray {
    pub base: c_int,
    pub elemTy: CIRType,
    pub nElems: c_int,
}

/// IRStmt structure
#[repr(C)]
pub struct CIRStmt {
    pub tag: CIRStmtTag,
    pub data: CIRStmtData,
}

#[repr(C)]
pub union CIRStmtData {
    pub noop: CIRStmtNoOp,
    pub imark: CIRStmtIMark,
    pub abihint: CIRStmtAbiHint,
    pub put: CIRStmtPut,
    pub puti: CIRStmtPutI,
    pub wrtmp: CIRStmtWrTmp,
    pub store: CIRStmtStore,
    pub loadg: CIRStmtLoadG,
    pub storeg: CIRStmtStoreG,
    pub cas: CIRStmtCAS,
    pub llsc: CIRStmtLLSC,
    pub dirty: CIRStmtDirty,
    pub mbe: CIRStmtMBE,
    pub exit: CIRStmtExit,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtNoOp {
    pub dummy: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtIMark {
    pub addr: u64, // Addr
    pub len: c_uint,
    pub delta: c_uchar,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtAbiHint {
    pub base: *mut CIRExpr,
    pub len: c_int,
    pub nia: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtPut {
    pub offset: c_int,
    pub data: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtPutI {
    pub details: *mut CIRPutI,
}

#[repr(C)]
pub struct CIRPutI {
    pub descr: *mut CIRRegArray,
    pub ix: *mut CIRExpr,
    pub bias: c_int,
    pub data: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtWrTmp {
    pub tmp: u32, // IRTemp
    pub data: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtStore {
    pub end: CIREndness,
    pub addr: *mut CIRExpr,
    pub data: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtLoadG {
    pub details: *mut CIRLoadG,
}

#[repr(C)]
pub struct CIRLoadG {
    pub end: CIREndness,
    pub cvt: c_uint, // IRLoadGOp
    pub dst: u32,    // IRTemp
    pub addr: *mut CIRExpr,
    pub alt: *mut CIRExpr,
    pub guard: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtStoreG {
    pub details: *mut CIRStoreG,
}

#[repr(C)]
pub struct CIRStoreG {
    pub end: CIREndness,
    pub addr: *mut CIRExpr,
    pub data: *mut CIRExpr,
    pub guard: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtCAS {
    pub details: *mut CIRCAS,
}

#[repr(C)]
pub struct CIRCAS {
    pub oldHi: u32, // IRTemp, 0xFFFFFFFF if unused
    pub oldLo: u32,
    pub end: CIREndness,
    pub addr: *mut CIRExpr,
    pub expdHi: *mut CIRExpr, // NULL if single-element
    pub expdLo: *mut CIRExpr,
    pub dataHi: *mut CIRExpr,
    pub dataLo: *mut CIRExpr,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtLLSC {
    pub end: CIREndness,
    pub result: u32, // IRTemp
    pub addr: *mut CIRExpr,
    pub storedata: *mut CIRExpr, // NULL for LL
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtDirty {
    pub details: *mut CIRDirty,
}

#[repr(C)]
pub struct CIRDirty {
    pub cee: *mut CIRCallee,
    pub guard: *mut CIRExpr,
    pub args: *mut *mut CIRExpr, // NULL-terminated
    pub tmp: u32,                // IRTemp, 0xFFFFFFFF if no return
    pub mFx: c_uint,             // IREffect
    pub mAddr: *mut CIRExpr,
    pub mSize: c_int,
    pub nFxState: c_int,
    // fxState array follows but we don't need it
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtMBE {
    pub event: c_uint, // IRMBusEvent
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CIRStmtExit {
    pub guard: *mut CIRExpr,
    pub dst: *mut CIRConst,
    pub jk: CIRJumpKind,
    pub offsIP: c_int,
}

/// IRTypeEnv structure
#[repr(C)]
pub struct CIRTypeEnv {
    pub types: *mut CIRType,
    pub types_size: c_int,
    pub types_used: c_int,
}

/// IRSB structure - the main block
#[repr(C)]
pub struct CIRSB {
    pub tyenv: *mut CIRTypeEnv,
    pub stmts: *mut *mut CIRStmt,
    pub stmts_size: c_int,
    pub stmts_used: c_int,
    pub next: *mut CIRExpr,
    pub jumpkind: CIRJumpKind,
    pub offsIP: c_int,
}

/// VEXLiftResult structure from pyvex.h
#[repr(C)]
pub struct CVEXLiftResult {
    pub irsb: *mut CIRSB,
    pub size: c_int,
    pub is_noop_block: c_int, // Bool
    pub exit_count: c_int,
    pub exits: [CExitInfo; 400],
    pub is_default_exit_constant: c_int,
    pub default_exit: u64,
    pub insts: c_int,
    pub inst_addrs: [u64; 200],
    pub data_ref_count: c_int,
    pub data_refs: [CDataRef; 2000],
    pub const_val_count: c_int,
    pub const_vals: [CConstVal; 1000],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CExitInfo {
    pub stmt_idx: c_int,
    pub ins_addr: u64,
    pub stmt: *mut CIRStmt,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CDataRef {
    pub data_addr: u64,
    pub size: c_int,
    pub data_type: c_uint,
    pub stmt_idx: c_int,
    pub ins_addr: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CConstVal {
    pub tmp: c_int,
    pub stmt_idx: c_int,
    pub value: c_ulonglong,
}

// ============================================================================
// FFI Function declarations
// ============================================================================

#[link(name = "pyvex")]
extern "C" {
    /// Initialize VEX - must be called before vex_lift
    pub fn vex_init() -> c_int;

    /// Lift bytes to VEX IR
    pub fn vex_lift(
        guest: CVexArch,
        archinfo: CVexArchInfo,
        insn_start: *const c_uchar,
        insn_addr: c_ulonglong,
        max_insns: c_uint,
        max_bytes: c_uint,
        opt_level: c_int,
        traceflags: c_int,
        allow_arch_optimizations: c_int,
        strict_block_end: c_int,
        collect_data_refs: c_int,
        load_from_ro_regions: c_int,
        const_prop: c_int,
        px_control: CVexRegisterUpdates,
        lookback_amount: c_uint,
    ) -> *mut CVEXLiftResult;

    /// Register a readonly region for const propagation
    pub fn register_readonly_region(
        start: c_ulonglong,
        size: c_ulonglong,
        content: *const c_uchar,
    ) -> c_int;

    /// Clear all registered readonly regions
    pub fn deregister_all_readonly_regions();
}

// ============================================================================
// Safe Rust wrapper
// ============================================================================

static VEX_INIT: Once = Once::new();
static mut VEX_INITIALIZED: bool = false;

/// Error type for native lifting
#[derive(Debug, Clone, thiserror::Error)]
pub enum NativeLiftError {
    /// VEX initialization failed
    #[error("VEX initialization failed")]
    InitFailed,
    /// Lifting failed (invalid code, unsupported instruction, etc.)
    #[error("lifting failed: {0}")]
    LiftFailed(String),
    /// Null pointer returned
    #[error("null result from vex_lift")]
    NullResult,
    /// Invalid IR structure
    #[error("invalid IR: {0}")]
    InvalidIR(String),
}

/// Initialize the VEX library (called automatically on first lift)
pub fn init_vex() -> Result<(), NativeLiftError> {
    VEX_INIT.call_once(|| {
        let result = unsafe { vex_init() };
        // vex_init returns 1 on success (not 0)
        unsafe {
            VEX_INITIALIZED = result == 1;
        }
    });

    if unsafe { VEX_INITIALIZED } {
        Ok(())
    } else {
        Err(NativeLiftError::InitFailed)
    }
}

/// Check if VEX is initialized
pub fn is_vex_initialized() -> bool {
    unsafe { VEX_INITIALIZED }
}

/// Native VEX lifting - lift bytes to IRSB
///
/// # Safety
/// The bytes slice must be valid for the duration of the call.
pub fn lift_native(
    bytes: &[u8],
    addr: u64,
    arch: VexArch,
    max_insns: u32,
    max_bytes: u32,
    opt_level: i32,
) -> Result<IRSB, NativeLiftError> {
    // Initialize VEX if needed
    init_vex()?;

    let c_arch = CVexArch::from(arch);
    let arch_info = CVexArchInfo::for_arch(arch);

    let result = unsafe {
        vex_lift(
            c_arch,
            arch_info,
            bytes.as_ptr(),
            addr,
            max_insns,
            max_bytes,
            opt_level,                               // opt_level
            0,                                       // traceflags
            1,                                       // allow_arch_optimizations
            0,                                       // strict_block_end
            0,                                       // collect_data_refs
            0,                                       // load_from_ro_regions
            0,                                       // const_prop
            CVexRegisterUpdates::AllregsAtMemAccess, // px_control
            0,                                       // lookback_amount
        )
    };

    if result.is_null() {
        return Err(NativeLiftError::NullResult);
    }

    // Convert C IRSB to Rust IRSB
    let c_result = unsafe { &*result };
    if c_result.irsb.is_null() {
        return Err(NativeLiftError::NullResult);
    }

    convert_irsb(unsafe { &*c_result.irsb }, addr, arch)
}

/// Convert a C IRSB to a Rust IRSB
fn convert_irsb(c_irsb: &CIRSB, addr: u64, arch: VexArch) -> Result<IRSB, NativeLiftError> {
    let mut irsb = IRSB::new(addr, arch);

    // Convert type environment
    if !c_irsb.tyenv.is_null() {
        let tyenv = unsafe { &*c_irsb.tyenv };
        irsb.tyenv = TypeEnv {
            types: (0..tyenv.types_used)
                .map(|i| {
                    let ty = unsafe { *tyenv.types.add(i as usize) };
                    IRType::from(ty)
                })
                .collect(),
        };
    }

    // Convert statements
    for i in 0..c_irsb.stmts_used {
        let stmt_ptr = unsafe { *c_irsb.stmts.add(i as usize) };
        if !stmt_ptr.is_null() {
            let c_stmt = unsafe { &*stmt_ptr };
            if let Some(stmt) = convert_stmt(c_stmt)? {
                irsb.statements.push(stmt);
            }
        }
    }

    // Convert next expression
    if !c_irsb.next.is_null() {
        irsb.next = convert_expr(unsafe { &*c_irsb.next })?;
    }

    // Convert jumpkind and offsIP
    irsb.jumpkind = JumpKind::from(c_irsb.jumpkind);
    irsb.offsIP = c_irsb.offsIP as u32;

    Ok(irsb)
}

/// Convert a C IRStmt to a Rust IRStmt
fn convert_stmt(c_stmt: &CIRStmt) -> Result<Option<IRStmt>, NativeLiftError> {
    match c_stmt.tag {
        CIRStmtTag::NoOp => Ok(Some(IRStmt::NoOp)),

        CIRStmtTag::IMark => {
            let imark = unsafe { c_stmt.data.imark };
            Ok(Some(IRStmt::IMark {
                addr: imark.addr,
                len: imark.len,
                delta: imark.delta,
            }))
        }

        CIRStmtTag::AbiHint => {
            let hint = unsafe { c_stmt.data.abihint };
            Ok(Some(IRStmt::AbiHint {
                base: Box::new(convert_expr(unsafe { &*hint.base })?),
                len: hint.len as u32,
                nia: Box::new(convert_expr(unsafe { &*hint.nia })?),
            }))
        }

        CIRStmtTag::Put => {
            let put = unsafe { c_stmt.data.put };
            Ok(Some(IRStmt::Put {
                offset: put.offset as u32,
                data: convert_expr(unsafe { &*put.data })?,
            }))
        }

        CIRStmtTag::PutI => {
            let puti = unsafe { c_stmt.data.puti };
            let details = unsafe { &*puti.details };
            let descr = unsafe { &*details.descr };
            Ok(Some(IRStmt::PutI {
                descr: IRRegArray {
                    base: descr.base as u32,
                    elemTy: IRType::from(descr.elemTy),
                    nElems: descr.nElems as u32,
                },
                ix: Box::new(convert_expr(unsafe { &*details.ix })?),
                bias: details.bias as u32,
                data: Box::new(convert_expr(unsafe { &*details.data })?),
            }))
        }

        CIRStmtTag::WrTmp => {
            let wrtmp = unsafe { c_stmt.data.wrtmp };
            Ok(Some(IRStmt::WrTmp {
                tmp: wrtmp.tmp,
                data: convert_expr(unsafe { &*wrtmp.data })?,
            }))
        }

        CIRStmtTag::Store => {
            let store = unsafe { c_stmt.data.store };
            Ok(Some(IRStmt::Store {
                addr: convert_expr(unsafe { &*store.addr })?,
                data: convert_expr(unsafe { &*store.data })?,
                endness: Endness::from(store.end),
            }))
        }

        CIRStmtTag::LoadG => {
            let loadg = unsafe { c_stmt.data.loadg };
            let details = unsafe { &*loadg.details };
            let cvt = match details.cvt {
                0x1500 => IRLoadGOp::WidenS,   // ILGop_IdentV128
                0x1501 => IRLoadGOp::Identity, // ILGop_Ident64
                0x1502 => IRLoadGOp::Identity, // ILGop_Ident32
                0x1503 => IRLoadGOp::WidenS,   // ILGop_16Uto32
                0x1504 => IRLoadGOp::WidenS,   // ILGop_16Sto32
                0x1505 => IRLoadGOp::WidenZ,   // ILGop_8Uto32
                0x1506 => IRLoadGOp::WidenS,   // ILGop_8Sto32
                _ => IRLoadGOp::Identity,
            };
            Ok(Some(IRStmt::LoadG {
                dst: details.dst,
                addr: Box::new(convert_expr(unsafe { &*details.addr })?),
                alt: Box::new(convert_expr(unsafe { &*details.alt })?),
                guard: Box::new(convert_expr(unsafe { &*details.guard })?),
                cvt,
                endness: Endness::from(details.end),
            }))
        }

        CIRStmtTag::StoreG => {
            let storeg = unsafe { c_stmt.data.storeg };
            let details = unsafe { &*storeg.details };
            Ok(Some(IRStmt::StoreG {
                addr: Box::new(convert_expr(unsafe { &*details.addr })?),
                data: Box::new(convert_expr(unsafe { &*details.data })?),
                guard: Box::new(convert_expr(unsafe { &*details.guard })?),
                endness: Endness::from(details.end),
            }))
        }

        CIRStmtTag::CAS => {
            let cas = unsafe { c_stmt.data.cas };
            let details = unsafe { &*cas.details };
            let old_hi = if details.oldHi == 0xFFFFFFFF {
                None
            } else {
                Some(details.oldHi)
            };
            let expd_hi = if details.expdHi.is_null() {
                None
            } else {
                Some(Box::new(convert_expr(unsafe { &*details.expdHi })?))
            };
            let data_hi = if details.dataHi.is_null() {
                None
            } else {
                Some(Box::new(convert_expr(unsafe { &*details.dataHi })?))
            };
            Ok(Some(IRStmt::CAS {
                old_hi,
                old_lo: details.oldLo,
                addr: Box::new(convert_expr(unsafe { &*details.addr })?),
                expdHi: expd_hi,
                expdLo: Box::new(convert_expr(unsafe { &*details.expdLo })?),
                dataHi: data_hi,
                dataLo: Box::new(convert_expr(unsafe { &*details.dataLo })?),
                endness: Endness::from(details.end),
            }))
        }

        CIRStmtTag::LLSC => {
            let llsc = unsafe { c_stmt.data.llsc };
            let storedata = if llsc.storedata.is_null() {
                None
            } else {
                Some(Box::new(convert_expr(unsafe { &*llsc.storedata })?))
            };
            Ok(Some(IRStmt::LLSC {
                storedata,
                result: llsc.result,
                addr: Box::new(convert_expr(unsafe { &*llsc.addr })?),
                endness: Endness::from(llsc.end),
            }))
        }

        CIRStmtTag::Dirty => {
            let dirty = unsafe { c_stmt.data.dirty };
            let details = unsafe { &*dirty.details };
            let cee = unsafe { &*details.cee };
            let name = if cee.name.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(cee.name) }
                    .to_string_lossy()
                    .into_owned()
            };
            let guard = if details.guard.is_null() {
                None
            } else {
                Some(Box::new(convert_expr(unsafe { &*details.guard })?))
            };
            let tmp = if details.tmp == 0xFFFFFFFF {
                None
            } else {
                Some(details.tmp)
            };
            let mAddr = if details.mAddr.is_null() {
                None
            } else {
                Some(Box::new(convert_expr(unsafe { &*details.mAddr })?))
            };

            // Convert args array
            let mut args = Vec::new();
            if !details.args.is_null() {
                let mut i = 0;
                loop {
                    let arg_ptr = unsafe { *details.args.add(i) };
                    if arg_ptr.is_null() {
                        break;
                    }
                    args.push(convert_expr(unsafe { &*arg_ptr })?);
                    i += 1;
                }
            }

            let mFx = match details.mFx {
                0 => DirtyFx::None,
                1 => DirtyFx::Read,
                2 => DirtyFx::Write,
                3 => DirtyFx::Modify,
                _ => DirtyFx::None,
            };

            Ok(Some(IRStmt::Dirty(IRDirty {
                cee: IRCallee {
                    name,
                    addr: cee.addr as u64,
                    mcx_mask: cee.mcx_mask,
                },
                guard,
                tmp,
                mFx,
                mAddr,
                mSize: details.mSize as u32,
                nFxState: details.nFxState as u32,
                args,
            })))
        }

        CIRStmtTag::MBE => {
            let mbe = unsafe { c_stmt.data.mbe };
            let event = match mbe.event {
                0x1D00 => MBusEvent::Fence,
                0x1D01 => MBusEvent::SFence,
                0x1D02 => MBusEvent::LFence,
                0x1D03 => MBusEvent::MFence,
                _ => MBusEvent::Fence,
            };
            Ok(Some(IRStmt::MBE(event)))
        }

        CIRStmtTag::Exit => {
            let exit = unsafe { c_stmt.data.exit };
            let dst_const = convert_const(unsafe { &*exit.dst })?;
            let dst = dst_const.as_u128() as u64;
            Ok(Some(IRStmt::Exit {
                guard: convert_expr(unsafe { &*exit.guard })?,
                dst,
                jk: JumpKind::from(exit.jk),
                offsIP: exit.offsIP as u32,
            }))
        }
    }
}

/// Convert a C IRExpr to a Rust IRExpr
fn convert_expr(c_expr: &CIRExpr) -> Result<IRExpr, NativeLiftError> {
    match c_expr.tag {
        CIRExprTag::Const => {
            let konst = unsafe { c_expr.data.konst };
            Ok(IRExpr::Const(convert_const(unsafe { &*konst.con })?))
        }

        CIRExprTag::RdTmp => {
            let rdtmp = unsafe { c_expr.data.rdtmp };
            Ok(IRExpr::RdTmp(rdtmp.tmp))
        }

        CIRExprTag::Get => {
            let get = unsafe { c_expr.data.get };
            Ok(IRExpr::Get {
                offset: get.offset as u32,
                ty: IRType::from(get.ty),
            })
        }

        CIRExprTag::GetI => {
            let geti = unsafe { c_expr.data.geti };
            let descr = unsafe { &*geti.descr };
            Ok(IRExpr::GetI {
                descr: IRRegArray {
                    base: descr.base as u32,
                    elemTy: IRType::from(descr.elemTy),
                    nElems: descr.nElems as u32,
                },
                ix: Box::new(convert_expr(unsafe { &*geti.ix })?),
                bias: geti.bias as u32,
            })
        }

        CIRExprTag::Load => {
            let load = unsafe { c_expr.data.load };
            Ok(IRExpr::Load {
                addr: Box::new(convert_expr(unsafe { &*load.addr })?),
                ty: IRType::from(load.ty),
                endness: Endness::from(load.end),
            })
        }

        CIRExprTag::Unop => {
            let unop = unsafe { c_expr.data.unop };
            let op = convert_op(unop.op)?;
            Ok(IRExpr::Unop {
                op,
                arg: Box::new(convert_expr(unsafe { &*unop.arg })?),
            })
        }

        CIRExprTag::Binop => {
            let binop = unsafe { c_expr.data.binop };
            let op = convert_op(binop.op)?;
            Ok(IRExpr::Binop {
                op,
                left: Box::new(convert_expr(unsafe { &*binop.arg1 })?),
                right: Box::new(convert_expr(unsafe { &*binop.arg2 })?),
            })
        }

        CIRExprTag::Triop => {
            let triop = unsafe { c_expr.data.triop };
            let details = unsafe { &*triop.details };
            let op = convert_op(details.op)?;
            Ok(IRExpr::Triop {
                op,
                arg1: Box::new(convert_expr(unsafe { &*details.arg1 })?),
                arg2: Box::new(convert_expr(unsafe { &*details.arg2 })?),
                arg3: Box::new(convert_expr(unsafe { &*details.arg3 })?),
            })
        }

        CIRExprTag::Qop => {
            let qop = unsafe { c_expr.data.qop };
            let details = unsafe { &*qop.details };
            let op = convert_op(details.op)?;
            Ok(IRExpr::Qop {
                op,
                arg1: Box::new(convert_expr(unsafe { &*details.arg1 })?),
                arg2: Box::new(convert_expr(unsafe { &*details.arg2 })?),
                arg3: Box::new(convert_expr(unsafe { &*details.arg3 })?),
                arg4: Box::new(convert_expr(unsafe { &*details.arg4 })?),
            })
        }

        CIRExprTag::ITE => {
            let ite = unsafe { c_expr.data.ite };
            Ok(IRExpr::ITE {
                cond: Box::new(convert_expr(unsafe { &*ite.cond })?),
                iftrue: Box::new(convert_expr(unsafe { &*ite.iftrue })?),
                iffalse: Box::new(convert_expr(unsafe { &*ite.iffalse })?),
            })
        }

        CIRExprTag::CCall => {
            let ccall = unsafe { c_expr.data.ccall };
            let cee = unsafe { &*ccall.cee };
            let name = if cee.name.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(cee.name) }
                    .to_string_lossy()
                    .into_owned()
            };

            // Convert args array
            let mut args = Vec::new();
            if !ccall.args.is_null() {
                let mut i = 0;
                loop {
                    let arg_ptr = unsafe { *ccall.args.add(i) };
                    if arg_ptr.is_null() {
                        break;
                    }
                    args.push(convert_expr(unsafe { &*arg_ptr })?);
                    i += 1;
                }
            }

            Ok(IRExpr::CCall {
                cee: IRCallee {
                    name,
                    addr: cee.addr as u64,
                    mcx_mask: cee.mcx_mask,
                },
                retty: IRType::from(ccall.retty),
                args,
            })
        }

        CIRExprTag::VECRET => Ok(IRExpr::VECRET),
        CIRExprTag::GSPTR => Ok(IRExpr::GSPTR),

        CIRExprTag::Binder => {
            // Binder expressions shouldn't appear in final IR
            Err(NativeLiftError::InvalidIR("Binder expression".to_string()))
        }
    }
}

/// Convert a C IRConst to a Rust IRConst
fn convert_const(c_const: &CIRConst) -> Result<IRConst, NativeLiftError> {
    match c_const.tag {
        CIRConstTag::Ico_U1 => Ok(IRConst::U1(unsafe { c_const.con.u1 != 0 })),
        CIRConstTag::Ico_U8 => Ok(IRConst::U8(unsafe { c_const.con.u8_ })),
        CIRConstTag::Ico_U16 => Ok(IRConst::U16(unsafe { c_const.con.u16_ })),
        CIRConstTag::Ico_U32 => Ok(IRConst::U32(unsafe { c_const.con.u32_ })),
        CIRConstTag::Ico_U64 => Ok(IRConst::U64(unsafe { c_const.con.u64_ })),
        CIRConstTag::Ico_F32 => Ok(IRConst::F32(unsafe { c_const.con.f32_ })),
        CIRConstTag::Ico_F32i => Ok(IRConst::F32(f32::from_bits(unsafe { c_const.con.f32i }))),
        CIRConstTag::Ico_F64 => Ok(IRConst::F64(unsafe { c_const.con.f64_ })),
        CIRConstTag::Ico_F64i => Ok(IRConst::F64(f64::from_bits(unsafe { c_const.con.f64i }))),
        CIRConstTag::Ico_V128 => {
            // V128 in libvex is stored as a 16-bit selector
            let selector = unsafe { c_const.con.v128 } as u128;
            // Expand the selector to 128 bits
            let mut result: u128 = 0;
            for i in 0..16 {
                if selector & (1 << i) != 0 {
                    result |= 0xFF << (i * 8);
                }
            }
            Ok(IRConst::V128(result))
        }
        CIRConstTag::Ico_V256 => {
            // V256 is stored as 32-bit selector
            let selector = unsafe { c_const.con.v256 } as u64;
            Ok(IRConst::V256([selector, 0, 0, 0]))
        }
    }
}

/// Convert a C IROp code to a Rust IROp
fn convert_op(op_code: c_uint) -> Result<IROp, NativeLiftError> {
    // Use the existing opcode_map module for conversion
    match opcode_map::parse_opcode_from_u32(op_code) {
        Some(op) => Ok(op),
        None => {
            // Return a Raw op for unrecognized opcodes
            Ok(IROp::Raw(op_code))
        }
    }
}

/// Register binary regions as readonly for constant propagation
pub fn register_binary_region(addr: u64, bytes: &[u8]) -> bool {
    if !is_vex_initialized() {
        return false;
    }
    unsafe { register_readonly_region(addr, bytes.len() as u64, bytes.as_ptr()) != 0 }
}

/// Clear all registered binary regions
pub fn clear_binary_regions() {
    if is_vex_initialized() {
        unsafe {
            deregister_all_readonly_regions();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_conversion() {
        assert_eq!(CVexArch::from(VexArch::AMD64), CVexArch::AMD64);
        assert_eq!(CVexArch::from(VexArch::X86), CVexArch::X86);
        assert_eq!(CVexArch::from(VexArch::ARM), CVexArch::ARM);
    }

    #[test]
    fn test_type_conversion() {
        assert_eq!(IRType::from(CIRType::I32), IRType::I32);
        assert_eq!(IRType::from(CIRType::I64), IRType::I64);
        assert_eq!(IRType::from(CIRType::V128), IRType::V128);
    }
}
