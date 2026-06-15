//! Architecture definitions and register files.
//!
//! This module provides:
//! - `Arch` trait for architecture abstraction
//! - Register file implementations for each architecture
//! - Register name mappings
//! - Calling convention implementations for argument extraction

mod amd64;
mod arm;
mod arm64;
pub mod calling_conventions;
mod mips;
mod x86;

pub use amd64::AMD64;
pub use arm::ARM;
pub use arm64::ARM64;
pub use calling_conventions::{
    AArch64CC, ARMEABI, CallingConvention, Cdecl, ExtractionError, SystemVAMD64,
    default_cc_for_arch,
};
pub use mips::{MIPS32, MIPS64};
pub use x86::X86;

use crate::symbolic::RustBV;
use crate::vex::VexArch;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

    /// Get the offset of the register holding the syscall number.
    ///
    /// The size always equals `bytes()` (pointer width), so callers
    /// only need the offset. Returns `None` for archs without a
    /// well-defined syscall convention here.
    fn syscall_num_offset(&self) -> Option<u32> {
        None
    }

    /// Check if the architecture is little-endian.
    fn is_little_endian(&self) -> bool;
}

/// Register table entry: `(name, byte offset, size in bytes)`.
///
/// Used by per-arch `CANONICAL` and `ALIASES` const slices that drive
/// `register_offset` / `register_size` / `register_name` via the
/// `lookup_register_*` helpers below. Each register is listed once per
/// (name, size) combination instead of being duplicated across three
/// separate match statements.
pub(crate) type RegEntry = (&'static str, u32, u32);

/// Look up a register's byte offset by name in the canonical or alias table.
///
/// Names are matched case-insensitively. The canonical table is searched
/// first; aliases (sub-register names sharing an offset) are searched as a
/// fallback.
pub(crate) fn lookup_register_offset(
    name: &str,
    canonical: &[RegEntry],
    aliases: &[RegEntry],
) -> Option<u32> {
    let lc = name.to_lowercase();
    canonical
        .iter()
        .chain(aliases.iter())
        .find(|(n, _, _)| *n == lc.as_str())
        .map(|(_, off, _)| *off)
}

/// Look up a register's size in bytes by name in the canonical or alias table.
pub(crate) fn lookup_register_size(
    name: &str,
    canonical: &[RegEntry],
    aliases: &[RegEntry],
) -> Option<u32> {
    let lc = name.to_lowercase();
    canonical
        .iter()
        .chain(aliases.iter())
        .find(|(n, _, _)| *n == lc.as_str())
        .map(|(_, _, sz)| *sz)
}

/// Look up a canonical register name by byte offset.
///
/// Only the canonical table is searched — aliases (e.g. "eax" sharing
/// offset 16 with "rax") never appear here.
pub(crate) fn lookup_register_name(offset: u32, canonical: &[RegEntry]) -> Option<&'static str> {
    canonical
        .iter()
        .find(|(_, off, _)| *off == offset)
        .map(|(n, _, _)| *n)
}

/// Generate the four name/offset/size trait methods of `Arch` from a
/// per-arch `(canonical, aliases, register_names)` triple. Place inside
/// `impl Arch for $Arch { ... }` blocks alongside the other arch fields.
macro_rules! impl_arch_registers {
    ($canonical:expr, $aliases:expr, $names:expr) => {
        fn register_offset(&self, name: &str) -> Option<u32> {
            super::lookup_register_offset(name, $canonical, $aliases)
        }

        fn register_size(&self, name: &str) -> Option<u32> {
            super::lookup_register_size(name, $canonical, $aliases)
        }

        fn register_name(&self, offset: u32) -> Option<&'static str> {
            super::lookup_register_name(offset, $canonical)
        }

        fn register_names(&self) -> &[&'static str] {
            $names
        }
    };
}
pub(crate) use impl_arch_registers;

/// Register file for storing register values.
///
/// ## Serialization
///
/// Implements `Serialize`/`Deserialize` via the [`RegisterFileData`]
/// shadow type — the `FxHashMap<u32, RustBV>` collapses to a
/// `BTreeMap<u32, RustBV>` for deterministic ordering on the wire, and
/// the architecture is recorded by its `arch.name()` string and rebuilt
/// via [`arch_from_name`]. Symbolic overlay AST caches are rebuilt
/// inside the active thread-local Z3 context per the [`RustBV`] serde
/// shape (see `snapshot-rustbv-shadow-type-pattern` bd memory).
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "RegisterFileData", into = "RegisterFileData")]
pub struct RegisterFile {
    /// Raw storage (byte-addressable).
    data: Vec<u8>,
    /// Symbolic overlays (offset -> value).
    symbolic: FxHashMap<u32, RustBV>,
    /// Architecture information.
    arch: Box<dyn Arch>,
}

/// Serde shadow form for [`RegisterFile`].
///
/// The architecture is represented by its `name()` string. Unknown
/// architecture names round-trip through [`arch_from_name`]'s AMD64
/// fallback (matches the existing `Box<dyn Arch>` Clone behavior at
/// arch/mod.rs:638). `data` is right-padded / truncated to the rebuilt
/// architecture's `state_size()` so a snapshot taken at one arch is
/// structurally usable even if reloaded under a different one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterFileData {
    pub data: Vec<u8>,
    pub symbolic: BTreeMap<u32, RustBV>,
    pub arch_name: String,
}

impl From<RegisterFile> for RegisterFileData {
    fn from(rf: RegisterFile) -> Self {
        let arch_name = rf.arch.name().to_string();
        RegisterFileData {
            data: rf.data,
            symbolic: rf.symbolic.into_iter().collect(),
            arch_name,
        }
    }
}

impl From<RegisterFileData> for RegisterFile {
    fn from(d: RegisterFileData) -> Self {
        let arch = arch_from_name(&d.arch_name).unwrap_or_else(|| Box::new(AMD64));
        let size = arch.state_size();
        let mut data = vec![0u8; size];
        let copy_len = d.data.len().min(size);
        data[..copy_len].copy_from_slice(&d.data[..copy_len]);
        let symbolic: FxHashMap<u32, RustBV> = d.symbolic.into_iter().collect();
        RegisterFile {
            data,
            symbolic,
            arch,
        }
    }
}

impl RegisterFile {
    /// Create a new register file for the given architecture.
    pub fn new(arch: Box<dyn Arch>) -> Self {
        let size = arch.state_size();
        RegisterFile {
            data: vec![0; size],
            symbolic: FxHashMap::default(),
            arch,
        }
    }

    /// Get the architecture.
    pub fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Read a register value by offset and size.
    pub fn get(&self, offset: u32, size: u32, ctx: &crate::symbolic::SymContext) -> RustBV {
        // Check for symbolic value at this exact offset
        if let Some(sym) = self.symbolic.get(&offset) {
            if sym.width() == size * 8 {
                return sym.clone();
            }
            // Partial read: extract low bytes from wider symbolic value
            if sym.width() > size * 8 {
                return sym.extract(size * 8 - 1, 0, ctx);
            }
            // Wider read of narrower symbolic: e.g., reading ecx (32-bit) when
            // cx (16-bit) was written symbolically. Compose the symbolic low part
            // with the concrete high bytes from the data array.
            if sym.width() < size * 8 {
                let sym_bytes = sym.width() / 8;
                let remaining_offset = offset + sym_bytes;
                let remaining_bytes = size - sym_bytes;
                let remaining_start = remaining_offset as usize;
                let remaining_end = remaining_start + remaining_bytes as usize;

                if remaining_end <= self.data.len() {
                    // Check if the upper portion also has a symbolic value
                    let upper = if let Some(upper_sym) = self.symbolic.get(&remaining_offset) {
                        if upper_sym.width() == remaining_bytes * 8 {
                            upper_sym.clone()
                        } else {
                            // Read concrete upper bytes
                            let mut v: u128 = 0;
                            for (i, &byte) in
                                self.data[remaining_start..remaining_end].iter().enumerate()
                            {
                                v |= (byte as u128) << (i * 8);
                            }
                            RustBV::concrete(v, remaining_bytes * 8)
                        }
                    } else {
                        // Read concrete upper bytes
                        let mut v: u128 = 0;
                        for (i, &byte) in
                            self.data[remaining_start..remaining_end].iter().enumerate()
                        {
                            v |= (byte as u128) << (i * 8);
                        }
                        RustBV::concrete(v, remaining_bytes * 8)
                    };

                    // Compose: upper (MSB) concat sym (LSB) — little-endian layout
                    return upper.concat(sym, ctx);
                }
            }
        }

        // Check if this offset is within a wider symbolic register
        // E.g., reading al (offset=16, size=1) when rax (offset=16, size=8) is symbolic
        // Already handled above. But also check for sub-register reads at higher offsets
        // E.g., reading ah (offset=17, size=1) when rax (offset=16, size=8) is symbolic
        for (&sym_offset, sym_val) in &self.symbolic {
            if sym_offset < offset && offset + size <= sym_offset + sym_val.width() / 8 {
                let bit_lo = (offset - sym_offset) * 8;
                let bit_hi = bit_lo + size * 8 - 1;
                return sym_val.extract(bit_hi, bit_lo, ctx);
            }
        }

        // Check if any symbolic sub-register falls within our read range
        // E.g., reading eax (offset=8, size=4) when al (offset=8, size=1) is symbolic
        // or reading eax when ah (offset=9, size=1) is symbolic
        for (&sym_offset, sym_val) in &self.symbolic {
            let sym_size = sym_val.width() / 8;
            if sym_offset >= offset && sym_offset + sym_size <= offset + size {
                // This symbolic sub-register is contained within our read range
                // Build the result by composing symbolic and concrete parts
                let mut parts: Vec<RustBV> = Vec::new();
                let mut pos = offset;
                while pos < offset + size {
                    if let Some(sub_sym) = self.symbolic.get(&pos) {
                        let sub_size = sub_sym.width() / 8;
                        if pos + sub_size <= offset + size {
                            parts.push(sub_sym.clone());
                            pos += sub_size;
                            continue;
                        }
                    }
                    // Concrete byte
                    let idx = pos as usize;
                    if idx < self.data.len() {
                        parts.push(RustBV::concrete(self.data[idx] as u128, 8));
                    } else {
                        parts.push(RustBV::zero(8));
                    }
                    pos += 1;
                }
                // Compose parts: in little-endian, lower offset = LSB
                // Concat builds MSB first, so we reverse
                if !parts.is_empty() {
                    let mut result = parts.pop().expect(
                        "parts vec is non-empty because at least one sub-register was found",
                    );
                    while let Some(part) = parts.pop() {
                        result = result.concat(&part, ctx);
                    }
                    return result;
                }
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
    pub fn put(&mut self, offset: u32, value: RustBV) {
        let size = value.width() / 8;
        let write_bits = value.width();

        // Check if this write is to a SUB-REGISTER of a wider symbolic value.
        // E.g., writing cl (8-bit at offset 12) when ecx (32-bit at offset 12)
        // is symbolic. We must compose the new value with the remaining symbolic
        // bits to preserve them.
        if let Some(wider_sym) = self.symbolic.get(&offset).cloned()
            && wider_sym.width() > write_bits
        {
            // Writing to the LOW portion of a wider symbolic
            let upper = wider_sym.extract_no_ctx(wider_sym.width() - 1, write_bits);
            let composed = upper.concat_no_ctx(&value);
            self.symbolic.insert(offset, composed);
            // Also update concrete data for the written portion if concrete
            if let Some(v) = value.as_u128() {
                let start = offset as usize;
                for i in 0..size as usize {
                    if start + i < self.data.len() {
                        self.data[start + i] = (v >> (i * 8)) as u8;
                    }
                }
            }
            return;
        }
        // Also check if writing to the middle/upper portion of a wider symbolic.
        // E.g., writing ch (8-bit at offset 13) when ecx (32-bit at offset 12) is symbolic.
        for (&sym_offset, sym_val) in &self.symbolic {
            let sym_size = sym_val.width() / 8;
            if sym_offset < offset && offset + size <= sym_offset + sym_size {
                // Our write is fully contained within a wider symbolic at a lower offset
                let sym_val = sym_val.clone();
                let bit_lo = (offset - sym_offset) * 8;
                let bit_hi = bit_lo + write_bits;
                let sym_bits = sym_val.width();

                let mut parts: Vec<RustBV> = Vec::new();
                // Upper portion (if any)
                if bit_hi < sym_bits {
                    parts.push(sym_val.extract_no_ctx(sym_bits - 1, bit_hi));
                }
                // The written value
                parts.push(value.clone());
                // Lower portion (if any)
                if bit_lo > 0 {
                    parts.push(sym_val.extract_no_ctx(bit_lo - 1, 0));
                }

                // Compose: concat all parts (MSB first)
                let mut composed = parts[0].clone();
                for part in &parts[1..] {
                    composed = composed.concat_no_ctx(part);
                }
                self.symbolic.insert(sym_offset, composed);
                // Update concrete data for the written portion if concrete
                if let Some(v) = value.as_u128() {
                    let start = offset as usize;
                    for i in 0..size as usize {
                        if start + i < self.data.len() {
                            self.data[start + i] = (v >> (i * 8)) as u8;
                        }
                    }
                }
                return;
            }
        }

        // If symbolic, store in symbolic map
        if value.is_symbolic() {
            // Clean up any narrower symbolic overlays within our range
            let overlapping: Vec<u32> = self
                .symbolic
                .keys()
                .filter(|&&k| k >= offset && k < offset + size && k != offset)
                .copied()
                .collect();
            for k in overlapping {
                self.symbolic.remove(&k);
            }
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
                // Clear any symbolic overlay at this offset
                self.symbolic.remove(&offset);
                // Also remove any narrower symbolic overlays within our range
                let overlapping: Vec<u32> = self
                    .symbolic
                    .keys()
                    .filter(|&&k| k >= offset && k < offset + size)
                    .copied()
                    .collect();
                for k in overlapping {
                    self.symbolic.remove(&k);
                }
            }
        }
    }

    /// Read a register by name.
    pub fn get_reg(&self, name: &str, ctx: &crate::symbolic::SymContext) -> Option<RustBV> {
        let offset = self.arch.register_offset(name)?;
        let size = self.arch.register_size(name)?;
        Some(self.get(offset, size, ctx))
    }

    /// Write a register by name.
    pub fn put_reg(&mut self, name: &str, value: RustBV) -> bool {
        if let (Some(offset), Some(size)) = (
            self.arch.register_offset(name),
            self.arch.register_size(name),
        ) && value.width() == size * 8
        {
            self.put(offset, value);
            return true;
        }
        false
    }

    /// Get the instruction pointer.
    pub fn get_ip(&self, ctx: &crate::symbolic::SymContext) -> RustBV {
        let offset = self.arch.ip_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Set the instruction pointer.
    pub fn set_ip(&mut self, value: RustBV) {
        let offset = self.arch.ip_offset();
        self.put(offset, value);
    }

    /// Get the stack pointer.
    pub fn get_sp(&self, ctx: &crate::symbolic::SymContext) -> RustBV {
        let offset = self.arch.sp_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Get the stack pointer as a concrete u64 value (for fast checks).
    pub fn get_sp_value(&self) -> Option<u64> {
        let offset = self.arch.sp_offset() as usize;
        let size = self.arch.bytes() as usize;
        if offset + size > self.data.len() {
            return None;
        }
        let mut value: u64 = 0;
        for i in 0..size.min(8) {
            value |= (self.data[offset + i] as u64) << (i * 8);
        }
        Some(value)
    }

    /// Set the stack pointer.
    pub fn set_sp(&mut self, value: RustBV) {
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
    pub fn fork(&self) -> RegisterFile {
        RegisterFile {
            data: self.data.clone(),
            symbolic: self.symbolic.clone(),
            arch: self.arch.clone(),
        }
    }

    /// Merge another register file into this one using a merge condition.
    ///
    /// For each register offset, if the values differ between `self` and `other`,
    /// the result is `ITE(merge_cond_other, other_val, self_val)`.
    ///
    /// `merge_cond_other` is the 1-bit condition for `other`'s path being active.
    ///
    /// Returns true if any register was actually merged (values differed).
    pub fn merge(
        &mut self,
        other: &RegisterFile,
        merge_cond_other: &crate::symbolic::RustBV,
        ctx: &crate::symbolic::SymContext,
    ) -> bool {
        use crate::symbolic::RustBV;

        let mut merged = false;

        // Collect all symbolic offsets from both register files
        let mut all_offsets: std::collections::HashSet<u32> =
            self.symbolic.keys().copied().collect();
        all_offsets.extend(other.symbolic.keys());

        // Merge symbolic registers
        for &offset in &all_offsets {
            let self_val = self.symbolic.get(&offset);
            let other_val = other.symbolic.get(&offset);

            match (self_val, other_val) {
                (Some(sv), Some(ov)) => {
                    if sv.width() == ov.width() {
                        // Both symbolic at same width — ITE merge
                        let ite_val = merge_cond_other.ite(ov, sv, ctx);
                        self.symbolic.insert(offset, ite_val);
                        merged = true;
                    }
                    // Width mismatch: keep self's value (edge case)
                }
                (Some(_sv), None) => {
                    // self is symbolic, other is concrete — read other's concrete
                    let size = _sv.width() / 8;
                    let start = offset as usize;
                    let end = start + size as usize;
                    if end <= other.data.len() {
                        let mut v: u128 = 0;
                        for (i, &byte) in other.data[start..end].iter().enumerate() {
                            v |= (byte as u128) << (i * 8);
                        }
                        let other_concrete = RustBV::concrete(v, _sv.width());
                        let ite_val = merge_cond_other.ite(&other_concrete, _sv, ctx);
                        self.symbolic.insert(offset, ite_val);
                        merged = true;
                    }
                }
                (None, Some(ov)) => {
                    // self is concrete, other is symbolic — read self's concrete
                    let size = ov.width() / 8;
                    let start = offset as usize;
                    let end = start + size as usize;
                    if end <= self.data.len() {
                        let mut v: u128 = 0;
                        for (i, &byte) in self.data[start..end].iter().enumerate() {
                            v |= (byte as u128) << (i * 8);
                        }
                        let self_concrete = RustBV::concrete(v, ov.width());
                        let ite_val = merge_cond_other.ite(ov, &self_concrete, ctx);
                        self.symbolic.insert(offset, ite_val);
                        merged = true;
                    }
                }
                (None, None) => {
                    // Both concrete — handled below in data comparison
                }
            }
        }

        // Check concrete data for differences at non-symbolic offsets.
        // We iterate register-sized chunks. For simplicity, use the arch's
        // native register width (e.g. 8 bytes for amd64).
        let reg_bytes = (self.arch.bits() / 8) as usize;
        let len = self.data.len().min(other.data.len());
        let mut off = 0;
        while off + reg_bytes <= len {
            let u32_off = off as u32;
            // Skip offsets that are already handled by symbolic merge
            if !all_offsets.contains(&u32_off) {
                let self_slice = &self.data[off..off + reg_bytes];
                let other_slice = &other.data[off..off + reg_bytes];
                if self_slice != other_slice {
                    // Concrete values differ — create ITE
                    let width = (reg_bytes * 8) as u32;
                    let mut sv: u128 = 0;
                    for (i, &byte) in self_slice.iter().enumerate() {
                        sv |= (byte as u128) << (i * 8);
                    }
                    let mut ov: u128 = 0;
                    for (i, &byte) in other_slice.iter().enumerate() {
                        ov |= (byte as u128) << (i * 8);
                    }
                    let self_bv = RustBV::concrete(sv, width);
                    let other_bv = RustBV::concrete(ov, width);
                    let ite_val = merge_cond_other.ite(&other_bv, &self_bv, ctx);
                    self.symbolic.insert(u32_off, ite_val);
                    merged = true;
                }
            }
            off += reg_bytes;
        }

        merged
    }
}

impl Clone for Box<dyn Arch> {
    fn clone(&self) -> Self {
        // This is a bit of a hack - we rely on architectures being singletons.
        // The match is exhaustive: every `dyn Arch` implementor is one of the
        // six supported singletons, so PPC32/PPC64/S390X can never be the
        // dynamic type here. We still panic loudly rather than silently
        // mis-cloning to AMD64 should a new VexArch implementor ever appear.
        match self.vex_arch() {
            VexArch::X86 => Box::new(X86),
            VexArch::AMD64 => Box::new(AMD64),
            VexArch::ARM => Box::new(ARM),
            VexArch::ARM64 => Box::new(ARM64),
            VexArch::MIPS32 => Box::new(MIPS32),
            VexArch::MIPS64 => Box::new(MIPS64),
            other @ (VexArch::PPC32 | VexArch::PPC64 | VexArch::S390X) => {
                panic!("{}", unsupported_arch_msg(other))
            }
        }
    }
}

/// Panic/error message for a VexArch the Rust engine does not implement.
///
/// The Rust engine supports six arches (X86, AMD64, ARM, ARM64, MIPS32,
/// MIPS64). For PPC32/PPC64/S390X — which Python angr supports — callers
/// should fall back to the Python engine. `RustExplorationManager`
/// construction rejects these loudly via `arch_from_name`; this message
/// covers the downstream snapshot-restore / interpreter-fork paths that
/// take a `VexArch` directly.
fn unsupported_arch_msg(arch: VexArch) -> String {
    format!(
        "Rust engine does not support {arch:?}; supported arches are X86, AMD64, \
         ARM, ARM64, MIPS32, MIPS64. Use the Python engine for this architecture."
    )
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
        other @ (VexArch::PPC32 | VexArch::PPC64 | VexArch::S390X) => {
            panic!("{}", unsupported_arch_msg(other))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::SymContext;

    #[test]
    fn test_arch_from_vex_supported() {
        // The six supported arches resolve to a singleton with the matching VexArch.
        for vex in [
            VexArch::X86,
            VexArch::AMD64,
            VexArch::ARM,
            VexArch::ARM64,
            VexArch::MIPS32,
            VexArch::MIPS64,
        ] {
            assert_eq!(arch_from_vex(vex).vex_arch(), vex);
        }
    }

    #[test]
    #[should_panic(expected = "Rust engine does not support")]
    fn test_arch_from_vex_ppc32_panics() {
        let _ = arch_from_vex(VexArch::PPC32);
    }

    #[test]
    #[should_panic(expected = "Rust engine does not support")]
    fn test_arch_from_vex_ppc64_panics() {
        let _ = arch_from_vex(VexArch::PPC64);
    }

    #[test]
    #[should_panic(expected = "Rust engine does not support")]
    fn test_arch_from_vex_s390x_panics() {
        let _ = arch_from_vex(VexArch::S390X);
    }

    #[test]
    fn test_clone_box_dyn_arch_roundtrips() {
        // Cloning a boxed supported arch preserves its identity (no AMD64 fallback).
        for vex in [
            VexArch::X86,
            VexArch::AMD64,
            VexArch::ARM,
            VexArch::ARM64,
            VexArch::MIPS32,
            VexArch::MIPS64,
        ] {
            let boxed = arch_from_vex(vex);
            assert_eq!(boxed.clone().vex_arch(), vex);
        }
    }

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

    /// Concrete-only round-trip: write a few registers, serialize, restore,
    /// and verify the same values come back. Uses mock SymContext so the
    /// test does not require Z3.
    #[test]
    fn serde_roundtrip_register_file_concrete() {
        let ctx = SymContext::new_mock();
        let mut regs = RegisterFile::new(Box::new(AMD64));
        regs.put_reg("rax", RustBV::concrete(0x1234567890ABCDEF, 64));
        regs.put_reg("rsp", RustBV::concrete(0x7fff_ffff_0000_1000, 64));

        let s = serde_json::to_string(&regs).expect("serialize");
        let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");

        assert_eq!(restored.arch().name(), "AMD64");
        assert_eq!(
            restored.get_reg("rax", &ctx).unwrap().as_u64(),
            Some(0x1234567890ABCDEF)
        );
        assert_eq!(
            restored.get_reg("rsp", &ctx).unwrap().as_u64(),
            Some(0x7fff_ffff_0000_1000)
        );
    }

    /// Symbolic round-trip: a register holding a `RustBV::Symbolic` is
    /// reconstructed with a fresh Z3 AST under the active thread-local
    /// context, and the rebuilt value is still recognized as symbolic
    /// with the same `id`, `width`, and `name`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn serde_roundtrip_register_file_symbolic() {
        use z3::{Config, Context, with_z3_context};

        let ctx = SymContext::new();
        let mut regs = RegisterFile::new(Box::new(AMD64));
        let sym = RustBV::symbolic(&ctx, "rdi_sym", 64);
        let (orig_id, orig_width) = match &sym {
            RustBV::Symbolic { id, width, .. } => (*id, *width),
            _ => panic!("expected symbolic"),
        };
        regs.put_reg("rdi", sym);

        let s = serde_json::to_string(&regs).expect("serialize");

        // Deserialize inside a fresh Z3 context to prove the AST cache
        // rebuilds correctly across context boundaries (the production
        // snapshot/restore path).
        let cfg = Config::new();
        let new_ctx = Context::new(&cfg);
        let (sym_id, sym_width, sym_name, is_symbolic) =
            with_z3_context(&new_ctx, || -> (u64, u32, String, bool) {
                let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");
                let ctx2 = SymContext::new();
                let bv = restored.get_reg("rdi", &ctx2).unwrap();
                match &bv {
                    RustBV::Symbolic {
                        id, width, name, ..
                    } => (*id, *width, name.to_string(), true),
                    _ => (0, 0, String::new(), false),
                }
            });

        assert!(is_symbolic, "restored register must still be Symbolic");
        assert_eq!(sym_id, orig_id);
        assert_eq!(sym_width, orig_width);
        assert_eq!(sym_name, "rdi_sym");
    }

    /// Unknown arch names fall back to AMD64 (matches `Box<dyn Arch>::clone`
    /// at arch/mod.rs:638), so a corrupted snapshot still loads into a
    /// usable register file. Tests the fallback path explicitly.
    #[test]
    fn serde_unknown_arch_name_falls_back_to_amd64() {
        let bad = RegisterFileData {
            data: vec![0u8; AMD64.state_size()],
            symbolic: BTreeMap::new(),
            arch_name: "not_a_real_arch".to_string(),
        };
        let s = serde_json::to_string(&bad).expect("serialize");
        let restored: RegisterFile = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(restored.arch().name(), "AMD64");
    }
}
