use super::*;

/// Guarded load operation.
///
/// The widening variants carry the source (in-memory) width in bits parsed
/// from the VEX `cvt` string (e.g. `ILGop_16Sto32` => `WidenS { src_bits: 16 }`).
/// This is required to load the correct number of bytes: a 32-bit destination
/// can be fed by either an 8-bit or a 16-bit memory load, and the destination
/// type alone cannot disambiguate them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IRLoadGOp {
    /// Sign-extend a `src_bits`-wide memory load to the destination width.
    WidenS { src_bits: u32 },
    /// Zero-extend a `src_bits`-wide memory load to the destination width.
    WidenZ { src_bits: u32 },
    /// No conversion: the memory load width equals the destination width.
    Identity,
    /// Unrecognized `cvt` string; surfaces an `InvalidIR` error at execution
    /// rather than silently defaulting to `Identity`.
    Unknown,
}

/// Register array descriptor.
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct IRRegArray {
    pub base: u32,
    pub elemTy: IRType,
    pub nElems: u32,
}

/// Clean helper callee info.
#[derive(Debug, Clone)]
pub struct IRCallee {
    pub name: String,
    pub addr: u64,
    pub mcx_mask: u32,
}

/// Dirty call info.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct IRDirty {
    pub cee: IRCallee,
    pub guard: Option<Box<IRExpr>>,
    pub tmp: Option<u32>,
    // mFx/mAddr/mSize/nFxState are populated by both lifter marshal paths
    // (pyvex_bridge/libvex_lifter) but currently write-only: interpreter's
    // dirty-call handling reads only cee.name/guard/tmp/args. Retained for
    // VEX-ABI parity and future memory-effect-aware code invalidation —
    // see angr-36vvn.9.
    pub mFx: DirtyFx,
    pub mAddr: Option<Box<IRExpr>>,
    pub mSize: u32,
    pub nFxState: u32,
    pub args: Vec<IRExpr>,
}

/// Dirty call side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyFx {
    None,
    Read,
    Write,
    Modify,
}
