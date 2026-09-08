//! Architecture-level IR enums: `VexArch` (the lifted guest ISA, with its
//! `pointer_size` / `endness` accessors), `Endness`, `JumpKind` (how a block
//! leaves, plus the `is_syscall` / `is_trap` / `is_call` / `is_ret`
//! classifiers and `ijk_name` for the libVEX `Ijk_*` spelling), and
//! `MBusEvent`. These describe the block's *context* and exit, as opposed to
//! its contents (`ast`), its operand types (`types`) or its operations
//! (`ops_def`).

/// VEX architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum VexArch {
    X86,
    AMD64,
    ARM,
    ARM64,
    MIPS32,
    MIPS64,
    PPC32,
    PPC64,
    S390X,
}

impl VexArch {
    /// Get the pointer size for this architecture in bits.
    pub fn pointer_size(&self) -> u32 {
        match self {
            VexArch::X86 | VexArch::ARM | VexArch::MIPS32 | VexArch::PPC32 => 32,
            VexArch::AMD64 | VexArch::ARM64 | VexArch::MIPS64 | VexArch::PPC64 | VexArch::S390X => {
                64
            }
        }
    }

    /// Get the endianness.
    pub fn endness(&self) -> Endness {
        match self {
            VexArch::X86 | VexArch::AMD64 | VexArch::ARM | VexArch::ARM64 => Endness::Little,
            VexArch::MIPS32
            | VexArch::MIPS64
            | VexArch::PPC32
            | VexArch::PPC64
            | VexArch::S390X => Endness::Big,
        }
    }
}

/// Jump kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum JumpKind {
    /// Normal jump (fallthrough or unconditional).
    Boring,
    /// Function call.
    Call,
    /// Function return.
    Ret,
    /// System call.
    Sys_syscall,
    Sys_int128,
    Sys_int129,
    Sys_int130,
    Sys_int145,
    Sys_int210,
    Sys_sysenter,
    /// `int N` for an N with no dedicated `Ijk_Sys_intNNN` tag.
    Sys_int,
    /// `int 0x20`.
    Sys_int32,
    /// Client request (Valgrind).
    ClientReq,
    /// Yield (threading).
    Yield,
    /// Emit warning.
    EmWarn,
    /// Emulation failure: libVEX could not emulate the instruction.
    EmFail,
    /// Instruction decoding failed — libVEX could not decode the bytes.
    NoDecode,
    /// Address translation (guest page map) failed.
    MapFail,
    /// Invalid instruction.
    InvalICache,
    /// Flush dcache.
    FlushDCache,
    /// Flush dcache line.
    FlushDCacheLine,
    /// Vectorized exit.
    ExtV128,
    /// Extended exit.
    Extension,
    /// Jump without redirection (Valgrind translation-table hint).
    NoRedir,
    /// Guest hit an illegal instruction (e.g. `ud2`).
    SigILL,
    /// Guest hit a breakpoint/trap instruction (e.g. `int3`).
    SigTRAP,
    /// Guest took a segmentation fault.
    SigSEGV,
    /// Guest took a bus error.
    SigBUS,
    /// Guest took a floating-point exception.
    SigFPE,
    /// Guest divided by zero.
    SigFPE_IntDiv,
    /// Guest integer operation overflowed.
    SigFPE_IntOvf,
    /// Guest executed a privileged instruction from user mode.
    Privileged,
}

impl JumpKind {
    /// Check if this is a syscall.
    pub fn is_syscall(&self) -> bool {
        matches!(
            self,
            JumpKind::Sys_syscall
                | JumpKind::Sys_int128
                | JumpKind::Sys_int129
                | JumpKind::Sys_int130
                | JumpKind::Sys_int145
                | JumpKind::Sys_int210
                | JumpKind::Sys_sysenter
                | JumpKind::Sys_int
                | JumpKind::Sys_int32
        )
    }

    /// Check if this exit delivers a guest trap/signal rather than transferring
    /// control normally.
    ///
    /// Python treats these as unexecutable: `engines/failure.py`'s
    /// `SimEngineFailure::process_successors` raises `AngrExitError` for any
    /// `Ijk_Sig*` parent jumpkind, so such a successor always ends up in the
    /// errored stash. Before angr-sqfj8.111 `parse_jumpkind` had no variants
    /// for them at all and folded every one into `Boring`, so a native `int3`
    /// or `ud2` continued executing as an ordinary fallthrough — a silently
    /// wrong answer with no diagnostic trail.
    ///
    /// `Ijk_Privileged` joins them: the guest took a privilege fault, and
    /// continuing straight-line is equally wrong. `Ijk_NoRedir` does not — it
    /// is a Valgrind translation hint on an otherwise ordinary jump.
    ///
    /// `Ijk_EmFail` and `Ijk_MapFail` join them too (angr-6cp06.70): the
    /// Python check quoted above is `jumpkind in ("Ijk_EmFail", "Ijk_MapFail")
    /// or jumpkind.startswith("Ijk_Sig")`, so both raise `AngrExitError`
    /// exactly as every `Ijk_Sig*` kind does. `Ijk_EmWarn` is the survivable
    /// sibling of `EmFail` and stays an ordinary jump, matching Python.
    ///
    /// `Ijk_NoDecode` is deliberately absent: Python catches it one layer
    /// earlier, at lift time, and only for the self-targeting shape — see
    /// `interpreter::execution::is_undecodable_block`.
    pub fn is_trap(&self) -> bool {
        matches!(
            self,
            JumpKind::EmFail
                | JumpKind::MapFail
                | JumpKind::SigILL
                | JumpKind::SigTRAP
                | JumpKind::SigSEGV
                | JumpKind::SigBUS
                | JumpKind::SigFPE
                | JumpKind::SigFPE_IntDiv
                | JumpKind::SigFPE_IntOvf
                | JumpKind::Privileged
        )
    }

    /// Check if this is a function call.
    pub fn is_call(&self) -> bool {
        matches!(self, JumpKind::Call)
    }

    /// Check if this is a function return.
    pub fn is_ret(&self) -> bool {
        matches!(self, JumpKind::Ret)
    }

    /// Return the VEX `Ijk_*` tag name for this jumpkind. Used by the
    /// state.inspect exit dispatcher to match angr Python's exit_jumpkind
    /// attribute convention.
    pub fn ijk_name(&self) -> &'static str {
        match self {
            JumpKind::Boring => "Ijk_Boring",
            JumpKind::Call => "Ijk_Call",
            JumpKind::Ret => "Ijk_Ret",
            JumpKind::Sys_syscall => "Ijk_Sys_syscall",
            JumpKind::Sys_int128 => "Ijk_Sys_int128",
            JumpKind::Sys_int129 => "Ijk_Sys_int129",
            JumpKind::Sys_int130 => "Ijk_Sys_int130",
            JumpKind::Sys_int145 => "Ijk_Sys_int145",
            JumpKind::Sys_int210 => "Ijk_Sys_int210",
            JumpKind::Sys_sysenter => "Ijk_Sys_sysenter",
            JumpKind::Sys_int => "Ijk_Sys_int",
            JumpKind::Sys_int32 => "Ijk_Sys_int32",
            JumpKind::ClientReq => "Ijk_ClientReq",
            JumpKind::Yield => "Ijk_Yield",
            JumpKind::EmWarn => "Ijk_EmWarn",
            JumpKind::EmFail => "Ijk_EmFail",
            JumpKind::NoDecode => "Ijk_NoDecode",
            JumpKind::MapFail => "Ijk_MapFail",
            JumpKind::InvalICache => "Ijk_InvalICache",
            JumpKind::FlushDCache => "Ijk_FlushDCache",
            JumpKind::FlushDCacheLine => "Ijk_FlushDCacheLine",
            JumpKind::ExtV128 => "Ijk_ExtV128",
            JumpKind::Extension => "Ijk_Extension",
            JumpKind::NoRedir => "Ijk_NoRedir",
            JumpKind::SigILL => "Ijk_SigILL",
            JumpKind::SigTRAP => "Ijk_SigTRAP",
            JumpKind::SigSEGV => "Ijk_SigSEGV",
            JumpKind::SigBUS => "Ijk_SigBUS",
            JumpKind::SigFPE => "Ijk_SigFPE",
            JumpKind::SigFPE_IntDiv => "Ijk_SigFPE_IntDiv",
            JumpKind::SigFPE_IntOvf => "Ijk_SigFPE_IntOvf",
            JumpKind::Privileged => "Ijk_Privileged",
        }
    }
}

/// Endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Endness {
    Little,
    Big,
}

/// Memory bus event.
///
/// Real VEX only emits `Imbe_Fence`/`Imbe_CancelReservation`, so
/// `SFence`/`LFence`/`MFence` are unreached in practice on either lifting
/// path — and the interpreter discards the MBE payload entirely, so the
/// distinction is moot even if one did arrive. The two lifters nonetheless
/// differ: `libvex_lifter::mbe_event` collapses *every* C-tag to `Fence`,
/// while `pyvex_bridge::parse_mbe_event` preserves the
/// `Imbe_SFence`/`Imbe_LFence`/`Imbe_MFence` spellings and only defaults
/// unrecognized ones to `Fence`. Retained for VEX-ABI parity and future
/// fence-granularity honoring, see angr-36vvn.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MBusEvent {
    Fence,
    SFence,
    LFence,
    MFence,
}
