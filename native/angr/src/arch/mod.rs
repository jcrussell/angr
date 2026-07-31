//! Architecture definitions and register files.
//!
//! This module provides:
//! - `Arch` trait for architecture abstraction
//! - Register file implementations for each architecture
//! - Register name mappings
//! - Calling convention implementations for argument extraction
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** register
//! offsets and sizes reaching this module come from VEX/guest state, so it
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. There are no
//! `unwrap`/`expect` sites left: the sub-register composition in
//! `RegisterFile::read` folds its `parts` vec with `if let Some(mut result) =
//! parts.pop()`, which also subsumes the old non-empty pre-check.
#![deny(clippy::unwrap_used, clippy::expect_used)]

// `pub(crate)` rather than private: the per-arch `offsets` modules are the
// single documented source of truth for VEX guest-state register offsets
// (provenance comments cite `VEX/pub/libvex_guest_*.h`), and
// `calling_conventions` references those consts by name instead of restating
// the numbers as bare literals.
pub(crate) mod amd64;
pub(crate) mod arm;
pub(crate) mod arm64;
pub(crate) mod calling_conventions;
pub(crate) mod mips;
pub(crate) mod x86;

pub(crate) use amd64::AMD64;
pub(crate) use arm::ARM;
pub(crate) use arm64::ARM64;
pub(crate) use calling_conventions::{
    AArch64CC, ARMEABI, CallingConvention, Cdecl, ExtractionError, MipsN64, MipsO32, SystemVAMD64,
    cc_for_arch, default_cc_for_arch,
};
pub(crate) use mips::{MIPS32, MIPS64};
pub(crate) use x86::X86;

use crate::symbolic::RustBV;
use crate::vex::VexArch;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

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
    /// Raw storage (byte-addressable). `Arc`-wrapped so `fork`/`Clone`
    /// share the buffer O(1); every `&mut self` write site routes through
    /// `Arc::make_mut` for copy-on-write isolation (mirrors the CoW
    /// discipline of `SymbolicMemory::fork`).
    data: Arc<Vec<u8>>,
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
            data: Arc::unwrap_or_clone(rf.data),
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
            data: Arc::new(data),
            symbolic,
            arch,
        }
    }
}

impl RegisterFile {
    /// Create a new register file for the given architecture.
    pub(crate) fn new(arch: Box<dyn Arch>) -> Self {
        let size = arch.state_size();
        RegisterFile {
            data: Arc::new(vec![0; size]),
            symbolic: FxHashMap::default(),
            arch,
        }
    }

    /// Get the architecture.
    pub(crate) fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Read a register value by offset and size.
    pub(crate) fn get(&self, offset: u32, size: u32, ctx: &crate::symbolic::SymContext) -> RustBV {
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
                if let Some(mut result) = parts.pop() {
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

    /// Read an architectural register (at `offset`, full arch byte-width) to a
    /// concrete `u64`, honoring the symbolic overlay. Returns `None` when the
    /// register is symbolic / not representable as a u64. Shared by the
    /// syscall-num and stack-pointer reads in the interpreter. Note this is
    /// distinct from `get_sp_value`, which reads raw concrete bytes and ignores
    /// the symbolic overlay.
    pub(crate) fn get_offset_u64(
        &self,
        offset: u32,
        ctx: &crate::symbolic::SymContext,
    ) -> Option<u64> {
        self.get(offset, self.arch.bytes(), ctx).as_u64()
    }

    /// Write a register value by offset.
    pub(crate) fn put(&mut self, offset: u32, value: RustBV) {
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
                let len = self.data.len();
                let data = Arc::make_mut(&mut self.data);
                for i in 0..size as usize {
                    if start + i < len {
                        data[start + i] = (v >> (i * 8)) as u8;
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
                    let len = self.data.len();
                    let data = Arc::make_mut(&mut self.data);
                    for i in 0..size as usize {
                        if start + i < len {
                            data[start + i] = (v >> (i * 8)) as u8;
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
                let data = Arc::make_mut(&mut self.data);
                for i in 0..size as usize {
                    data[start + i] = (v >> (i * 8)) as u8;
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
    pub(crate) fn get_reg(&self, name: &str, ctx: &crate::symbolic::SymContext) -> Option<RustBV> {
        let offset = self.arch.register_offset(name)?;
        let size = self.arch.register_size(name)?;
        Some(self.get(offset, size, ctx))
    }

    /// Write a register by name.
    pub(crate) fn put_reg(&mut self, name: &str, value: RustBV) -> bool {
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
    pub(crate) fn get_ip(&self, ctx: &crate::symbolic::SymContext) -> RustBV {
        let offset = self.arch.ip_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Set the instruction pointer.
    pub(crate) fn set_ip(&mut self, value: RustBV) {
        let offset = self.arch.ip_offset();
        self.put(offset, value);
    }

    /// Get the stack pointer.
    pub(crate) fn get_sp(&self, ctx: &crate::symbolic::SymContext) -> RustBV {
        let offset = self.arch.sp_offset();
        let size = self.arch.bytes();
        self.get(offset, size, ctx)
    }

    /// Get the stack pointer as a concrete u64 value (for fast checks).
    pub(crate) fn get_sp_value(&self) -> Option<u64> {
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
    pub(crate) fn set_sp(&mut self, value: RustBV) {
        let offset = self.arch.sp_offset();
        self.put(offset, value);
    }

    /// Copy concrete register values from a byte slice.
    ///
    /// This is used to initialize the register file from external state.
    pub(crate) fn copy_from_bytes(&mut self, bytes: &[u8]) {
        let len = std::cmp::min(bytes.len(), self.data.len());
        Arc::make_mut(&mut self.data)[..len].copy_from_slice(&bytes[..len]);
        // Clear symbolic overlays since we're replacing with concrete values
        self.symbolic.clear();
    }

    /// Copy concrete register values to a byte slice.
    ///
    /// This is used to extract the register state after execution.
    /// Note: symbolic values are converted to their concrete value (0 if unknown).
    pub(crate) fn copy_to_bytes(&self, bytes: &mut [u8]) {
        let len = std::cmp::min(bytes.len(), self.data.len());
        bytes[..len].copy_from_slice(&self.data[..len]);
    }

    /// Fork the register file for path splitting.
    pub(crate) fn fork(&self) -> RegisterFile {
        RegisterFile {
            // O(1) Arc refcount bump; the buffer is copied lazily on the
            // first write to either parent or child via `Arc::make_mut`.
            data: Arc::clone(&self.data),
            symbolic: self.symbolic.clone(),
            arch: self.arch.clone(),
        }
    }

    /// Cross-context twin of [`Self::fork`] (angr-ahypj): copy this register
    /// file, deep-translating every symbolic overlay BV into `target_ctx`.
    /// The concrete `data` buffer and architecture are context-independent and
    /// shared/cloned verbatim.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn translate_into(&self, target_ctx: &z3::Context) -> RegisterFile {
        RegisterFile {
            data: Arc::clone(&self.data),
            symbolic: self
                .symbolic
                .iter()
                .map(|(&off, bv)| (off, bv.translate_into(target_ctx)))
                .collect(),
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
    pub(crate) fn merge(
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
        // The dispatch is exhaustive: every `dyn Arch` implementor is one of
        // the six supported singletons, so PPC32/PPC64/S390X can never be the
        // dynamic type here. We delegate to `arch_from_vex` so the singleton
        // lookup table lives in exactly one place; it panics loudly rather
        // than silently mis-cloning should a new VexArch implementor appear.
        arch_from_vex(self.vex_arch())
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

/// One row of [`ALL_ARCHES`]: everything the engine knows about a supported
/// architecture, keyed by the names a caller may spell it with.
pub(crate) struct ArchDesc {
    /// Canonical name; equals `Arch::name()` of the arch `make_arch` builds.
    pub name: &'static str,
    /// Accepted spellings other than `name`. Matched case-insensitively,
    /// same as `name` itself.
    pub aliases: &'static [&'static str],
    pub vex: VexArch,
    /// Constructor for the arch singleton. A `&'static dyn Arch` field would
    /// read more directly, but `arch_from_name` / `arch_from_vex` hand out
    /// owned `Box<dyn Arch>` and `dyn Arch` is not `Clone` on its own.
    pub make_arch: fn() -> Box<dyn Arch>,
    pub make_cc: fn() -> Box<dyn CallingConvention>,
}

/// The single registry of supported architectures.
///
/// Before this table the alias lists were restated in four places
/// (`arch_from_name`'s match arms, a per-CC `ARCH_ALIASES` const,
/// `cc_for_arch`'s if/else chain, and a hand-written list in
/// `calling_conventions_tests`), which let them drift: `mips64be` was
/// registered on the MIPS N64 calling convention but never accepted by
/// `arch_from_name`. It is **not** listed here — MIPS64 big-endian states are
/// built as `mips64` plus `Iend_BE`, so a separate arch name would be a second
/// spelling for a distinction we do not model. Widening `arch_from_name` to
/// accept it would be a behavior change, not a refactor.
///
/// Every `name`/`aliases` entry must resolve through both `make_arch` and
/// `make_cc`; `calling_conventions_tests::test_arch_from_name_names_all_have_a_cc`
/// enforces that, and `test_arch_aliases_disjoint` enforces that no spelling
/// belongs to two rows.
pub(crate) const ALL_ARCHES: &[ArchDesc] = &[
    ArchDesc {
        name: "X86",
        aliases: &["x86", "i386", "i486", "i586", "i686"],
        vex: VexArch::X86,
        make_arch: || Box::new(X86),
        make_cc: || Box::new(Cdecl),
    },
    ArchDesc {
        name: "AMD64",
        aliases: &["amd64", "x86_64", "x64"],
        vex: VexArch::AMD64,
        make_arch: || Box::new(AMD64),
        make_cc: || Box::new(SystemVAMD64),
    },
    ArchDesc {
        name: "ARM",
        aliases: &["arm", "armel", "armhf", "armv7", "armv7l"],
        vex: VexArch::ARM,
        make_arch: || Box::new(ARM),
        make_cc: || Box::new(ARMEABI),
    },
    ArchDesc {
        name: "ARM64",
        aliases: &["arm64", "aarch64", "armv8"],
        vex: VexArch::ARM64,
        make_arch: || Box::new(ARM64),
        make_cc: || Box::new(AArch64CC),
    },
    ArchDesc {
        name: "MIPS32",
        aliases: &["mips", "mips32", "mipsel", "mipsle"],
        vex: VexArch::MIPS32,
        make_arch: || Box::new(MIPS32),
        make_cc: || Box::new(MipsO32),
    },
    ArchDesc {
        name: "MIPS64",
        aliases: &["mips64", "mips64el", "mips64le"],
        vex: VexArch::MIPS64,
        make_arch: || Box::new(MIPS64),
        make_cc: || Box::new(MipsN64),
    },
];

/// Look up the [`ArchDesc`] whose canonical name or alias list contains
/// `name`. Matching is case-insensitive.
pub(crate) fn arch_desc_from_name(name: &str) -> Option<&'static ArchDesc> {
    ALL_ARCHES.iter().find(|d| {
        d.name.eq_ignore_ascii_case(name) || d.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    })
}

/// Create an architecture by name.
pub(crate) fn arch_from_name(name: &str) -> Option<Box<dyn Arch>> {
    arch_desc_from_name(name).map(|d| (d.make_arch)())
}

/// Create an architecture from VexArch.
pub(crate) fn arch_from_vex(arch: VexArch) -> Box<dyn Arch> {
    match ALL_ARCHES.iter().find(|d| d.vex == arch) {
        Some(d) => (d.make_arch)(),
        // PPC32/PPC64/S390X are the only VexArch values with no row.
        None => panic!("{}", unsupported_arch_msg(arch)),
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
