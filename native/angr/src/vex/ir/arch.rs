
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
    /// Client request (Valgrind).
    ClientReq,
    /// Yield (threading).
    Yield,
    /// Emit warning.
    EmWarn,
    /// Emit fail.
    EmFail,
    /// No redirect.
    NoDecode,
    /// Map fail.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MBusEvent {
    Fence,
    SFence,
    LFence,
    MFence,
}
