//! Architecture definitions and register files.
//!
//! This module provides:
//! - `Arch` trait for architecture abstraction
//! - Register file implementations for each architecture
//! - Register name mappings

mod amd64;
mod arm;
mod arm64;
mod mips;
mod x86;

pub use amd64::AMD64;
pub use arm::ARM;
pub use arm64::ARM64;
pub use mips::{MIPS32, MIPS64};
pub use x86::X86;

use crate::symbolic::RustBV;
use crate::vex::VexArch;

/// Architecture trait.
///
/// Provides architecture-specific information and register access.
pub trait Arch: Send + Sync {
    /// Get the VEX architecture enum.
    fn vex_arch(&self) -> VexArch;

    /// Get the architecture name.
    fn name(&self) -> &'static str;

    /// Get the pointer size in bits.
    fn bits(&self) -> u32;

    /// Get the pointer size in bytes.
    fn bytes(&self) -> u32 {
        self.bits() / 8
    }

    /// Get the size of the guest state in bytes.
    fn state_size(&self) -> usize;

    /// Get the offset of the instruction pointer register.
    fn ip_offset(&self) -> u32;

    /// Get the offset of the stack pointer register.
    fn sp_offset(&self) -> u32;

    /// Get the offset of the base pointer register (if any).
    fn bp_offset(&self) -> Option<u32>;

    /// Get the offset of a register by name.
    fn register_offset(&self, name: &str) -> Option<u32>;

    /// Get the size of a register by name (in bytes).
    fn register_size(&self, name: &str) -> Option<u32>;

    /// Get the name of a register by offset.
    fn register_name(&self, offset: u32) -> Option<&'static str>;

    /// Get all register names.
    fn register_names(&self) -> &[&'static str];

    /// Get the calling convention argument registers.
    fn argument_registers(&self) -> &[u32];

    /// Get the return value register offset.
    fn return_register(&self) -> u32;

    /// Check if the architecture is little-endian.
    fn is_little_endian(&self) -> bool;
}

/// Register file for storing register values.
pub struct RegisterFile<'ctx> {
    /// Raw storage (byte-addressable).
    data: Vec<u8>,
    /// Symbolic overlays (offset -> value).
    symbolic: std::collections::HashMap<u32, RustBV<'ctx>>,
    /// Architecture information.
    arch: Box<dyn Arch>,
}

impl<'ctx> RegisterFile<'ctx> {
    /// Create a new register file for the given architecture.
    pub fn new(arch: Box<dyn Arch>) -> Self {
        let size = arch.state_size();
        RegisterFile {
            data: vec![0; size],
            symbolic: std::collections::HashMap::new(),
            arch,
        }
    }

    /// Get the architecture.
    pub fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Read a register value by offset and size.
    pub fn get(&self, offset: u32, size: u32, ctx: &'ctx crate::symbolic::SymContext<'ctx>) -> RustBV<'ctx> {
        // Check for symbolic value first
        if let Some(sym) = self.symbolic.get(&offset) {
            if sym.width() == size * 8 {
                return sym.clone();
            }
        }

        // Read concrete value
        let start = offset as usize;
        let end = start + size as usize;

        if end > self.data.len() {
            return RustBV::zero(size * 8);
        }

        let mut value: u128 = 0;
        for (i, &byte) in self.data[start..end].iter().enumerate() {
            value |= (byte as u128) << (i * 8);
        }

        RustBV::concrete(value, size * 8)
    }

    /// Write a register value by offset.
    pub fn put(&mut self, offset: u32, value: RustBV<'ctx>) {
        let size = value.width() / 8;

        // If symbolic, store in symbolic map
        if value.is_symbolic() {
            self.symbolic.insert(offset, value);
            return;
        }

        // Store concrete value
        if let Some(v) = value.as_u128() {
            let start = offset as usize;
            let end = start + size as usize;

            if end <= self.data.len() {
                for i in 0..size as usize {
                    self.data[start + i] = (v >> (i * 8)) as u8;
                }
                // Clear any symbolic overlay
                self.symbolic.remove(&offset);
            }
        }
    }

    /// Read a register by name.
    pub fn get_reg(&self, name: &str, ctx: &'ctx crate::symbolic::SymContext<'ctx>) -> Option<RustBV<'ctx>> {
        let offset = self.arch.register_offset(name)?;
        let size = self.arch.register_size(name)?;
        Some(self.get(offset, size, ctx))
    }

    /// Write a register by name.
    pub fn put_reg(&mut self, name: &str, value: RustBV<'ctx>) -> bool {
        if let (Some(offset), Some(size)) = (
            self.arch.register_offset(name),
            self.arch.register_size(name),
        ) {
            if value.width() == size * 8 {
                self.put(offset, value);
                return true;
            }
        }
        false
    }

    /// Get the instruction pointer.
    pub fn get_ip(&self, ctx: &'ctx crate::symbolic::SymContext<'ctx>) -> RustBV<'ctx> {
        let offset = self.arch.ip_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Set the instruction pointer.
    pub fn set_ip(&mut self, value: RustBV<'ctx>) {
        let offset = self.arch.ip_offset();
        self.put(offset, value);
    }

    /// Get the stack pointer.
    pub fn get_sp(&self, ctx: &'ctx crate::symbolic::SymContext<'ctx>) -> RustBV<'ctx> {
        let offset = self.arch.sp_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Set the stack pointer.
    pub fn set_sp(&mut self, value: RustBV<'ctx>) {
        let offset = self.arch.sp_offset();
        self.put(offset, value);
    }

    /// Copy concrete register values from a byte slice.
    ///
    /// This is used to initialize the register file from external state.
    pub fn copy_from_bytes(&mut self, bytes: &[u8]) {
        let len = std::cmp::min(bytes.len(), self.data.len());
        self.data[..len].copy_from_slice(&bytes[..len]);
        // Clear symbolic overlays since we're replacing with concrete values
        self.symbolic.clear();
    }

    /// Copy concrete register values to a byte slice.
    ///
    /// This is used to extract the register state after execution.
    /// Note: symbolic values are converted to their concrete value (0 if unknown).
    pub fn copy_to_bytes(&self, bytes: &mut [u8]) {
        let len = std::cmp::min(bytes.len(), self.data.len());
        bytes[..len].copy_from_slice(&self.data[..len]);
    }

    /// Fork the register file for path splitting.
    pub fn fork(&self) -> RegisterFile<'ctx> {
        RegisterFile {
            data: self.data.clone(),
            symbolic: self.symbolic.clone(),
            arch: self.arch.clone(),
        }
    }
}

impl Clone for Box<dyn Arch> {
    fn clone(&self) -> Self {
        // This is a bit of a hack - we rely on architectures being singletons
        match self.vex_arch() {
            VexArch::X86 => Box::new(X86),
            VexArch::AMD64 => Box::new(AMD64),
            VexArch::ARM => Box::new(ARM),
            VexArch::ARM64 => Box::new(ARM64),
            VexArch::MIPS32 => Box::new(MIPS32),
            VexArch::MIPS64 => Box::new(MIPS64),
            _ => Box::new(AMD64), // Default to AMD64 for unsupported archs
        }
    }
}

/// Create an architecture by name.
pub fn arch_from_name(name: &str) -> Option<Box<dyn Arch>> {
    match name.to_lowercase().as_str() {
        // x86/AMD64
        "x86" | "i386" | "i486" | "i586" | "i686" => Some(Box::new(X86)),
        "amd64" | "x86_64" | "x64" => Some(Box::new(AMD64)),
        // ARM
        "arm" | "armel" | "armhf" | "armv7" | "armv7l" => Some(Box::new(ARM)),
        "arm64" | "aarch64" | "armv8" => Some(Box::new(ARM64)),
        // MIPS
        "mips" | "mips32" | "mipsel" | "mipsle" => Some(Box::new(MIPS32)),
        "mips64" | "mips64el" | "mips64le" => Some(Box::new(MIPS64)),
        _ => None,
    }
}

/// Create an architecture from VexArch.
pub fn arch_from_vex(arch: VexArch) -> Box<dyn Arch> {
    match arch {
        VexArch::X86 => Box::new(X86),
        VexArch::AMD64 => Box::new(AMD64),
        VexArch::ARM => Box::new(ARM),
        VexArch::ARM64 => Box::new(ARM64),
        VexArch::MIPS32 => Box::new(MIPS32),
        VexArch::MIPS64 => Box::new(MIPS64),
        _ => Box::new(AMD64), // Default for unsupported archs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::SymContext;

    #[test]
    fn test_register_file_concrete() {
        let ctx = SymContext::new_mock();

        let mut regs = RegisterFile::new(Box::new(AMD64));

        // Write RAX
        regs.put_reg("rax", RustBV::concrete(0x1234567890ABCDEF, 64));

        // Read back
        let rax = regs.get_reg("rax", &ctx).unwrap();
        assert_eq!(rax.as_u64(), Some(0x1234567890ABCDEF));

        // Read sub-register (AL is the low byte of RAX)
        // RAX is at offset 16 in VEX AMD64 guest state
        let al = regs.get(16, 1, &ctx);
        assert_eq!(al.as_u64(), Some(0xEF));
    }

    #[test]
    fn test_register_file_symbolic() {
        let ctx = SymContext::new_mock();

        let mut regs = RegisterFile::new(Box::new(AMD64));

        // Write a symbolic value
        let sym = RustBV::symbolic(&ctx, "rax", 64);
        regs.put_reg("rax", sym.clone());

        // Read back
        let rax = regs.get_reg("rax", &ctx).unwrap();
        assert!(rax.is_symbolic());
    }
}
