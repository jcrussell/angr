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
    AArch64, ARMEABI, CallingConvention, Cdecl, ExtractionError, MAX_EXTRACT_ARGS, MipsN64,
    MipsO32, SystemVAMD64, cc_for_arch, default_cc_for_arch,
};
pub(crate) use mips::{MIPS32, MIPS64};
pub(crate) use x86::X86;

use crate::symbolic::{MAX_CONCRETE_CHUNK, RustBV, u128_le_byte};
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
    ///
    /// **Deliberately has no non-test caller** (bead angr-sqfj8.3), unlike its
    /// `ip_offset` / `sp_offset` siblings: SP is read by `RegisterFile::get_sp`
    /// / `get_sp_value`, `CallingConvention::get_return_addr` and the
    /// SimProcedure/prefetch paths, and IP by `RegisterFile::get_ip` /
    /// `set_ip`, but nothing in the engine needs the frame pointer — we never
    /// walk or synthesize a frame, and code that wants BP by name gets it from
    /// `register_offset("bp")` / `register_offset("fp")`. The Python side's
    /// `arch.bp_offset`
    /// (`slicer.py`, `blade.py`, the reaching-definitions analyses) is
    /// archinfo's, not this one.
    ///
    /// Kept rather than deleted because it is the pinned half of
    /// `mod_tests::test_all_arches_report_expected_special_register_offsets`,
    /// which asserts each arch's answer against a literal in that file's
    /// `ArchExpect` table — so a guest-state layout edit that shifts EBP/RBP/
    /// R11/X29/R30 fails there, the same integrity-sweep role `register_name`
    /// plays. Deleting it would silently drop that arm.
    ///
    /// The `Option` is future-proofing, not a live case: all six arches return
    /// `Some`. It exists for a frame-pointer-less target (a pure-RISC ABI that
    /// spends the register as a GPR), and a caller must therefore treat `None`
    /// as "this arch has no frame pointer", never as "unknown".
    ///
    /// As with `register_name`, rustc's `dead_code` lint does not flag this
    /// (trait methods on a used trait are exempt), so the "unused" reading has
    /// to be re-derived by hand each audit — hence this note.
    fn bp_offset(&self) -> Option<u32>;

    /// Get the offset of a register by name.
    fn register_offset(&self, name: &str) -> Option<u32>;

    /// Get the size of a register by name (in bytes).
    fn register_size(&self, name: &str) -> Option<u32>;

    /// Get the name of a register by offset — the reverse of
    /// `register_offset`, resolved against `CANONICAL` only (never `ALIASES`).
    ///
    /// **Deliberately has no non-test caller** (bead angr-9ke6b.8). Every
    /// production path into the register file is name-keyed
    /// (`register_offset` / `register_size` / `register_names`), so nothing in
    /// the engine ever needs to go offset -> name. What this method exists for
    /// is to be the assertion primitive of the arch-table integrity sweeps,
    /// which is a load-bearing role, not dead weight — deleting it deletes
    /// these gates:
    ///
    /// - `mod_tests::register_names_all_resolve_and_fit_in_u128`
    /// - `mod_tests::exported_register_names_reverse_resolve` (angr-9ke6b.9:
    ///   catches an exported name that was left in `ALIASES`)
    /// - `mod_tests::vex_bookkeeping_fields_resolve_on_every_arch`
    /// - `calling_conventions_tests::test_cc_arg_registers_resolve_to_expected_register_names`
    ///   and `..::test_link_register_set_on_link_register_abis`, which turn a
    ///   raw CC offset back into a name to compare against the ABI tables
    ///
    /// Note that rustc's `dead_code` lint does not flag it (trait methods on a
    /// used trait are exempt), so the "unused" reading has to be re-derived by
    /// hand each audit — hence this note. Contrast
    /// `CallingConvention::endness`, which angr-9ke6b.218 item 7 *deleted*
    /// rather than kept: that one returned a hardcoded answer that would be
    /// wrong for its first real caller, whereas this one is a correct lookup
    /// with real callers that happen to all be tests.
    fn register_name(&self, offset: u32) -> Option<&'static str>;

    /// Get all register names that cross the Python boundary.
    ///
    /// This drives `register_names_for_arch` (the Python-side
    /// `_supported_register_names` filter) and `RustSimState::export_full`'s
    /// `named_registers`, so it is the set of registers that round-trip
    /// through `state.regs.*`. It is a SUBSET of the per-arch `CANONICAL`
    /// table: entries wider than 16 bytes must be omitted, because the
    /// named-register channel carries each value as a `u128`
    /// (`ExplorationStateSnapshot::get_registers_named`) and
    /// `RegisterFile::get`'s concrete read composes bytes into a `u128`.
    /// Today `fpreg` (64 B on x86/AMD64) is the only such register.
    fn register_names(&self) -> &[&'static str];

    /// The arch's whole `CANONICAL` register table — the superset
    /// `register_names` is drawn from.
    ///
    /// Like `register_name`, this exists to be the assertion primitive of the
    /// arch-table integrity sweeps rather than to serve a production caller:
    /// `mod_tests::every_narrow_canonical_register_is_exported` walks it to
    /// check the *converse* of the `register_names` doc above — every entry
    /// that fits the u128 named-register channel has to be exported, or the
    /// corresponding `state.regs.*` write is silently dropped on the Python
    /// side. Only checking the forward direction is how X86's missing
    /// `sseround` survived the angr-9ke6b.6 sweep (angr-sqfj8.1/.6).
    fn canonical_registers(&self) -> &[RegEntry];

    /// Get the offset of the register holding the syscall number.
    ///
    /// The size always equals `bytes()` (pointer width), so callers
    /// only need the offset. Returns `None` for archs without a
    /// well-defined syscall convention here.
    fn syscall_num_offset(&self) -> Option<u32> {
        None
    }

    /// Check if the architecture is little-endian.
    ///
    /// **Not authoritative for byte order.** Every impl hardcodes `true`,
    /// including the two either-endian families (ARM, MIPS) where the choice
    /// is a per-binary build flag rather than a property of the arch. The real
    /// per-state byte order is `RustSimState::is_little_endian`, seeded from
    /// the `little_endian` override in `with_solver_endian`; read that instead
    /// when the answer has to be correct for a big-endian target.
    ///
    /// (angr-9ke6b.218 item 7: `CallingConvention::endness` used to duplicate
    /// this same hardcoded answer and was deleted rather than kept, because a
    /// dead accessor that returns `Little` unconditionally hands its first
    /// real caller a wrong answer on exactly those two families.)
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
/// Names are matched ASCII-case-insensitively (every table entry is an
/// ASCII-lowercase register name). The canonical table is searched first;
/// aliases (sub-register names sharing an offset) are searched as a fallback.
pub(crate) fn lookup_register_offset(
    name: &str,
    canonical: &[RegEntry],
    aliases: &[RegEntry],
) -> Option<u32> {
    canonical
        .iter()
        .chain(aliases.iter())
        .find(|(n, _, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, off, _)| *off)
}

/// Look up a register's size in bytes by name in the canonical or alias table.
///
/// Matching follows `lookup_register_offset`: ASCII-case-insensitive, canonical
/// table before aliases.
pub(crate) fn lookup_register_size(
    name: &str,
    canonical: &[RegEntry],
    aliases: &[RegEntry],
) -> Option<u32> {
    canonical
        .iter()
        .chain(aliases.iter())
        .find(|(n, _, _)| n.eq_ignore_ascii_case(name))
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

/// Generate the five name/offset/size trait methods of `Arch` from a
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

        fn canonical_registers(&self) -> &[super::RegEntry] {
            $canonical
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
/// via [`arch_from_name`]. Deserialization is fallible
/// (`#[serde(try_from = ...)]`): an unrecognized `arch_name` is rejected with
/// [`UnknownRegisterFileArch`], see [`RegisterFileData`]. Symbolic overlay AST caches are rebuilt
/// inside the active thread-local Z3 context per the [`RustBV`] serde
/// shape (see `snapshot-rustbv-shadow-type-pattern` bd memory).
///
/// ## Merge coverage (angr-91vj9.13)
///
/// `RustSimState` labels `registers` `#[merge_policy = "delegate"]`, which ends
/// the top-level derive's guarantee at this struct's boundary — and, exactly as
/// with [`crate::memory::SymbolicMemory`], the "field-added-without-a-merge-line"
/// bug family recurred one level deeper (angr-c7xno.1: [`Self::merge`]'s
/// concrete-diff chunk scan shadowed an already-merged unaligned sub-register
/// overlay). Deriving [`angr_macros::MergePolicy`] here extends the guarantee:
/// a new field does not compile until it declares how [`Self::merge`] treats it.
///
/// [`Self::merge`] mutates `self` in place rather than building a new struct, so
/// "self wins" is spelled `in_place_self` — there is no assignment for a
/// generated `merge_field_<name>()` to replace.
#[derive(Clone, Serialize, Deserialize, angr_macros::MergePolicy)]
#[serde(try_from = "RegisterFileData", into = "RegisterFileData")]
pub struct RegisterFile {
    /// Raw storage (byte-addressable). `Arc`-wrapped so `fork`/`Clone`
    /// share the buffer O(1); every `&mut self` write site routes through
    /// `Arc::make_mut` for copy-on-write isolation (mirrors the CoW
    /// discipline of `SymbolicMemory::fork`).
    ///
    /// Never assigned by [`Self::merge`]: the buffer stays `self`'s and each
    /// byte range where the two branches' backing bytes differ is lifted into a
    /// `symbolic` ITE overlay by the concrete-diff chunk scan, which shadows the
    /// stale bytes underneath. Jointly computed with `symbolic` for that reason.
    #[merge_policy = "joint"]
    data: Arc<Vec<u8>>,
    /// Symbolic overlays (offset -> value).
    ///
    /// The whole body of [`Self::merge`] targets this map — per-offset ITEs,
    /// the widening arm, and the subsumption pass that drops overlays a wide
    /// merge now covers. Reads `data` (see above), hence `joint`.
    #[merge_policy = "joint"]
    symbolic: FxHashMap<u32, RustBV>,
    /// Architecture information.
    ///
    /// Merge-invariant: merge arms are fork siblings, and no API reassigns a
    /// live register file's architecture.
    #[merge_policy = "in_place_self"]
    arch: Box<dyn Arch>,
    /// Big-endian register-storage adapter: byte offset -> `(base, size)` of the
    /// widest canonical register covering it, or `(0, 0)` where no canonical
    /// register does. `None` on a little-endian register file, where
    /// [`Self::mirror_offset`] is the identity and this costs nothing.
    ///
    /// See [`Self::mirror_offset`] for what it is for. `Arc`-shared so
    /// [`Self::fork`] stays O(1); immutable after construction.
    ///
    /// Merge-invariant: derived purely from `arch` + the state's endianness,
    /// both of which merge arms share as fork siblings.
    #[merge_policy = "in_place_self"]
    containment: Option<Arc<Vec<(u32, u32)>>>,
}

/// Serde shadow form for [`RegisterFile`].
///
/// The architecture is represented by its `name()` string and rebuilt via
/// [`arch_from_name`]. An unrecognized name is **rejected** with
/// [`UnknownRegisterFileArch`] rather than silently substituted with AMD64
/// (angr-c7xno.3): `data` is a raw register-byte image, so reinterpreting one
/// arch's layout as another's makes every subsequent register read
/// wrong-but-plausible. Rejecting matches how the rest of this subsystem
/// handles an unknown arch name — `RustSimState::from_snapshot` returns
/// `SnapshotError::UnknownArch`, and [`arch_from_vex`] panics (which
/// `impl Clone for Box<dyn Arch>` inherits, since commit 91b9f82ca replaced its
/// own AMD64 fallback with that panic).
///
/// `data` is right-padded / truncated to the rebuilt architecture's
/// `state_size()` so a snapshot taken at one arch is structurally usable even
/// if reloaded under a different, *recognized* one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterFileData {
    pub data: Vec<u8>,
    pub symbolic: BTreeMap<u32, RustBV>,
    pub arch_name: String,
    /// Whether the register file was created for a big-endian target — the
    /// `containment` adapter is rebuilt from this plus `arch_name` rather than
    /// serialized. Defaults to `false` so a snapshot written before this field
    /// existed still loads (every such snapshot was little-endian: the
    /// big-endian path did not exist yet).
    #[serde(default)]
    pub register_be: bool,
}

impl From<RegisterFile> for RegisterFileData {
    fn from(rf: RegisterFile) -> Self {
        let arch_name = rf.arch.name().to_string();
        let register_be = rf.containment.is_some();
        RegisterFileData {
            data: Arc::unwrap_or_clone(rf.data),
            symbolic: rf.symbolic.into_iter().collect(),
            arch_name,
            register_be,
        }
    }
}

/// Deserializing a [`RegisterFile`] whose recorded `arch_name` matches no
/// architecture this build knows about. The sole failure mode of
/// `TryFrom<RegisterFileData> for RegisterFile`; surfaces as a plain serde
/// error (e.g. through `RustExplorationManager::load_snapshot_bytes`).
#[derive(Debug, thiserror::Error)]
#[error("unknown architecture in serialized register file: {name}")]
pub struct UnknownRegisterFileArch {
    pub name: String,
}

impl TryFrom<RegisterFileData> for RegisterFile {
    type Error = UnknownRegisterFileArch;

    fn try_from(d: RegisterFileData) -> Result<Self, Self::Error> {
        let arch =
            arch_from_name(&d.arch_name).ok_or(UnknownRegisterFileArch { name: d.arch_name })?;
        let size = arch.state_size();
        let mut data = vec![0u8; size];
        let copy_len = d.data.len().min(size);
        data[..copy_len].copy_from_slice(&d.data[..copy_len]);
        let symbolic: FxHashMap<u32, RustBV> = d.symbolic.into_iter().collect();
        let containment = (d.register_be).then(|| build_containment(arch.as_ref()));
        Ok(RegisterFile {
            data: Arc::new(data),
            symbolic,
            arch,
            containment,
        })
    }
}

/// Compose little-endian `bytes` (lowest index = LSB) into a `u128`.
///
/// Callers must pass at most [`MAX_CONCRETE_CHUNK`] bytes: a longer slice
/// shifts past the width of a `u128`, which panics under `overflow-checks` and
/// otherwise wraps the shift amount mod 128, folding the excess bytes back over
/// the low ones. The merge scan slices `arch.bits() / 8`, at most 8 bytes on
/// every supported arch (angr-sqfj8.7); `RegisterFile::get` used to be a
/// counterexample — `fpreg` is 64 bytes wide — and now routes any register past
/// the cap through `compose_range` instead (angr-0jh0j.1).
fn le_bytes_to_u128(bytes: &[u8]) -> u128 {
    debug_assert!(
        bytes.len() <= MAX_CONCRETE_CHUNK,
        "le_bytes_to_u128 takes at most {MAX_CONCRETE_CHUNK} bytes, got {}",
        bytes.len()
    );
    let mut value: u128 = 0;
    for (i, &byte) in bytes.iter().take(MAX_CONCRETE_CHUNK).enumerate() {
        value |= (byte as u128) << (i * 8);
    }
    value
}

/// A stack-pointer or return-address read that may be symbolic/unavailable.
///
/// Deliberately carries no `Default`, `From<u64>`, or `Deref<Target = u64>`
/// impl: a bare `.unwrap_or(0)` on the inner value is exactly the
/// silently-wrong-answer bug this type exists to make uncompilable — the
/// same `Option<u64>`-returning shape reintroduced it three times
/// (angr-sqfj8.62, angr-sqfj8.63, angr-c7xno.29). Callers that need the two
/// cases explicitly should use [`Self::concrete`]; callers that want the
/// logged fallback should use [`Self::or_log`].
#[must_use]
pub(crate) struct AddrOrSymbolic(Option<u64>);

impl AddrOrSymbolic {
    /// The concrete address, or `None` when symbolic/unavailable.
    pub(crate) fn concrete(&self) -> Option<u64> {
        self.0
    }

    /// [`Self::concrete`], substituting `0` and logging when the answer is
    /// symbolic/unavailable, instead of silently carrying a
    /// plausible-looking-but-wrong address forward.
    ///
    /// `accessor` names the method that produced `self` (e.g.
    /// `"get_stack_pointer"`) and `context` identifies the call site, so the
    /// warning names both what collapsed to 0 and where — a `&'static str`
    /// literal, not built with `format!`, so the common concrete-value path
    /// pays no allocation; the interpolation only happens inside
    /// `silent_default!`'s already-lazy None arm.
    // SILENT(cat-c): a symbolic/unavailable address collapsing to the
    // literal 0 is a wrong-answer risk downstream (e.g. exported verbatim as
    // `CallStackEntry.stack_ptr`) — every caller must go through this single
    // logged fallback rather than a bare `.unwrap_or(0)`.
    pub(crate) fn or_log(&self, accessor: &str, context: &str) -> u64 {
        silent_default!(
            cat_c,
            self.0,
            0,
            "{accessor}() returned None ({context}); using 0 — likely a symbolic or \
             unavailable address collapsed to a wrong value"
        )
    }
}

impl From<Option<u64>> for AddrOrSymbolic {
    fn from(value: Option<u64>) -> Self {
        Self(value)
    }
}

/// Build the byte-offset -> widest-covering-canonical-register table backing
/// [`RegisterFile::mirror_offset`].
///
/// A byte no canonical register covers keeps the `(0, 0)` sentinel; a real
/// register at base 0 always has a nonzero size, so `size == 0` is an
/// unambiguous "not covered". Widest wins, because a nested canonical entry
/// (x86's `eax` inside `rax`, both at offset 16) is a *view* of the wider
/// register's bytes, and mirroring has to happen relative to the storage unit
/// the whole register occupies.
fn build_containment(arch: &dyn Arch) -> Arc<Vec<(u32, u32)>> {
    let mut table = vec![(0u32, 0u32); arch.state_size()];
    for &(_, base, size) in arch.canonical_registers() {
        for byte in base..base.saturating_add(size) {
            if let Some(slot) = table.get_mut(byte as usize)
                && slot.1 < size
            {
                *slot = (base, size);
            }
        }
    }
    Arc::new(table)
}

impl RegisterFile {
    /// Create a new little-endian register file for the given architecture.
    ///
    /// Equivalent to [`Self::new_with_endian`] with `is_le = true`, and a
    /// test-only convenience: every production caller — state construction and
    /// `VEXInterpreter::with_config_endian` — has a target endianness to supply
    /// and must go through [`Self::new_with_endian`] so a big-endian target
    /// gets its containment table (angr-21cz6).
    #[cfg(test)]
    pub(crate) fn new(arch: Box<dyn Arch>) -> Self {
        Self::new_with_endian(arch, true)
    }

    /// Create a new register file for the given architecture and target byte
    /// order.
    ///
    /// `is_le` is the *state's* endianness (the `little_endian` override
    /// threaded through `RustSimState::with_solver_endian`), not
    /// `Arch::is_little_endian`. On a big-endian target it turns on the
    /// [`Self::mirror_offset`] adapter; see there for why.
    pub(crate) fn new_with_endian(arch: Box<dyn Arch>, is_le: bool) -> Self {
        let size = arch.state_size();
        let containment = (!is_le).then(|| build_containment(arch.as_ref()));
        RegisterFile {
            data: Arc::new(vec![0; size]),
            symbolic: FxHashMap::default(),
            arch,
            containment,
        }
    }

    /// Map a VEX register offset into this file's little-endian storage,
    /// mirroring it inside its containing register on a big-endian target.
    ///
    /// angr's Python `SimState` stores the register file as a flat byte array
    /// in `arch.register_endness` — `Iend_BE` for MIPS32/MIPS64/ARMEB — so on
    /// those targets a register's MSB sits at its *lowest* byte offset. This
    /// file's storage is unconditionally little-endian instead, which agrees
    /// with Python for any access that covers a whole named register (the two
    /// engines exchange registers as integers, converted with
    /// `register_endness` on the Python side — see `rust_state_sync.py`) but
    /// *disagrees* for a VEX Get/Put narrower than the register containing it.
    /// MIPS `mov.d $f0, $f2` is the canonical case: VEX models FR=0, so it
    /// lifts to an F32 copy of the `fN_lo` sub-field at each register's base
    /// offset, and the two engines then pick opposite 32-bit halves
    /// (angr-fuhmm).
    ///
    /// Rather than making every byte<->value composition in this struct (and
    /// its merge scan, its snapshot image, `get_sp_value`, the symbolic overlay
    /// bit arithmetic) endianness-aware, the storage stays little-endian and
    /// the *offset* is mirrored at the door: an access of `size` bytes at
    /// `offset`, inside a canonical register spanning `[base, base + reg_size)`,
    /// reads the storage at `base + reg_size - (offset - base) - size`. The map
    /// is an involution, so overlay keys written through [`Self::put`] and read
    /// back through [`Self::get`] stay consistent, and a whole-register access
    /// (`offset == base`, `size == reg_size`) is the identity.
    ///
    /// Identity — i.e. today's behaviour — for a little-endian file, for an
    /// offset no canonical register covers (the VEX bookkeeping tail), and for
    /// a range that straddles two registers, which is meaningless under either
    /// model.
    fn mirror_offset(&self, offset: u32, size: u32) -> u32 {
        let Some(table) = self.containment.as_ref() else {
            return offset;
        };
        let Some(&(base, reg_size)) = table.get(offset as usize) else {
            return offset;
        };
        // Not covered by any canonical register.
        if reg_size == 0 || size == 0 {
            return offset;
        }
        // overflow-ok: `offset`/`size` are VEX guest-state coordinates bounded
        // by `state_size()` (a few KiB) and a register width; the checked forms
        // below refuse rather than wrap anyway.
        let Some(end) = offset.checked_add(size) else {
            return offset;
        };
        // Straddles the end of the containing register (or runs past it).
        if end > base.saturating_add(reg_size) {
            return offset;
        }
        base + reg_size - (offset - base) - size
    }

    /// Get the architecture.
    pub(crate) fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Compose the byte range `[offset, offset + size)` out of this file's
    /// symbolic overlays and its concrete backing bytes.
    ///
    /// Walks the range low-to-high: an overlay starting at a position
    /// contributes its bytes (truncated to what still fits in the range), and
    /// every position no overlay starts at contributes its concrete byte, or a
    /// zero byte once the range runs past the end of `data`. Little-endian, so
    /// the lowest offset is the LSB of the result. Callers therefore need no
    /// bounds check of their own.
    ///
    /// Shared by both of [`Self::get`]'s composition paths so that a read wider
    /// than the overlay at its own offset sees *all* the overlays inside the
    /// range, not just one that happens to fill the remainder exactly
    /// (angr-49v03).
    fn compose_range(&self, offset: u32, size: u32, ctx: &crate::symbolic::SymContext) -> RustBV {
        let end = offset + size;
        let mut parts: Vec<RustBV> = Vec::new();
        let mut pos = offset;
        while pos < end {
            if let Some(sub_sym) = self.symbolic.get(&pos) {
                let sub_size = sub_sym.width() / 8;
                if sub_size > 0 {
                    if pos + sub_size <= end {
                        parts.push(sub_sym.clone());
                        pos += sub_size;
                        continue;
                    }
                    // Overlay runs past the end of the read — take the low bytes
                    // that do fit rather than falling back to stale concrete.
                    parts.push(sub_sym.extract((end - pos) * 8 - 1, 0, ctx));
                    break;
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
        // Compose parts: in little-endian, lower offset = LSB. Concat builds
        // MSB first, so we consume the vector back-to-front.
        let Some(mut result) = parts.pop() else {
            return RustBV::zero(size * 8);
        };
        while let Some(part) = parts.pop() {
            result = result.concat(&part, ctx);
        }
        result
    }

    /// Read a register value by offset and size.
    pub(crate) fn get(&self, offset: u32, size: u32, ctx: &crate::symbolic::SymContext) -> RustBV {
        let offset = self.mirror_offset(offset, size);
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
            // with whatever covers the remaining bytes — which may be several
            // narrower overlays, not just one that exactly fills the remainder
            // (angr-49v03), falling back to the concrete backing bytes.
            //
            // Deliberately duplicates the contained-overlay loop below: an
            // overlay at exactly `offset` narrower than the read always also
            // satisfies that loop's `sym_offset >= offset && sym_offset +
            // sym_size <= offset + size` test and reaches the same
            // `compose_range(offset, size, ctx)` call, so this arm is a pure
            // O(1) short-circuit past two O(overlays) scans of `symbolic`. It
            // earns its keep because x86 sub-registers alias a *shared* offset
            // (rax/eax/ax/al are all offset 16), making "read wider than the
            // overlay written at the same offset" the common case rather than
            // an edge one. Keeping it also pins the current precedence: the
            // wider-symbolic loop below never gets to preempt a read that has
            // its own overlay. `put` keeps overlays non-overlapping, so today
            // no entry could preempt it anyway.
            if sym.width() < size * 8 {
                return self.compose_range(offset, size, ctx);
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
        // or reading eax when ah (offset=9, size=1) is symbolic.
        // The `sym_offset == offset` half of this is what the same-offset arm at
        // the top of this function short-circuits; see the comment there.
        for (&sym_offset, sym_val) in &self.symbolic {
            let sym_size = sym_val.width() / 8;
            if sym_offset >= offset && sym_offset + sym_size <= offset + size {
                // This symbolic sub-register is contained within our read range
                return self.compose_range(offset, size, ctx);
            }
        }

        // Read concrete value
        let start = offset as usize;
        let end = start + size as usize;

        if end > self.data.len() {
            return RustBV::zero(size * 8);
        }

        // A `Concrete` is backed by a `u128`, so a register wider than
        // `MAX_CONCRETE_CHUNK` has no concrete representation at all: composing
        // its bytes here would shift past bit 127 (panic under
        // `overflow-checks`, silent 16-byte-cycle garbage otherwise) and the
        // result could not hold the upper bytes even if it did not. x86/amd64's
        // `fpreg` is 64 bytes and reachable by name through `get_reg`, which is
        // how this went uncaught while the bulk `register_names()` export
        // excluded fpreg for the same width reason (angr-0jh0j.1). Compose
        // byte-wise instead: `concat` declines to fold past 128 bits, so the
        // value stays exact as a `Concat` tree — and `as_u128()` then reports
        // `None`, giving name-keyed Python callers the clean "cannot read
        // register fpreg" they already expect.
        if size as usize > MAX_CONCRETE_CHUNK {
            return self.compose_range(offset, size, ctx);
        }

        RustBV::concrete(le_bytes_to_u128(&self.data[start..end]), size * 8)
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

    /// Mirror a concrete `value` into the concrete backing bytes at `offset`,
    /// little-endian, skipping any byte that falls past the end of `data`.
    ///
    /// No-op when `value` is not representable as a `u128` (i.e. symbolic or
    /// wider than 128 bits) — the symbolic overlay is the authority in that
    /// case and the stale concrete bytes are never consulted for it.
    ///
    /// Shared by both of [`Self::put`]'s sub-register-of-a-wider-symbolic
    /// paths (angr-sqfj8.8). Deliberately *not* used by `put`'s plain concrete
    /// store, which bounds-checks the whole range up front and skips the write
    /// entirely when it does not fit, rather than truncating per byte.
    fn mirror_concrete_bytes(&mut self, offset: u32, value: &RustBV) {
        let Some(v) = value.as_u128() else {
            // SILENT(cat-a): symbolic writes keep their value in the overlay;
            // the concrete bytes for that span are shadowed and unread.
            return;
        };
        let size = (value.width() / 8) as usize;
        let start = offset as usize;
        let len = self.data.len();
        let data = Arc::make_mut(&mut self.data);
        for i in 0..size {
            if start + i < len {
                data[start + i] = u128_le_byte(v, i);
            }
        }
    }

    /// Write a register value by offset.
    pub(crate) fn put(&mut self, offset: u32, value: RustBV) {
        let size = value.width() / 8;
        let write_bits = value.width();
        let offset = self.mirror_offset(offset, size);

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
            self.mirror_concrete_bytes(offset, &value);
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
                self.mirror_concrete_bytes(offset, &value);
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

        // Store concrete value.
        //
        // `size` may exceed `MAX_CONCRETE_CHUNK` — `set_register("fpreg", ..)`
        // builds a 512-bit `Concrete` from a `u128` — so the byte extraction
        // goes through `u128_le_byte`, which yields the implicit zeros above the
        // payload instead of shift-wrapping the low 16 bytes back over the
        // remaining 48 (angr-0jh0j.2). Those zeros are the value being written,
        // not a truncation: a `Concrete` wider than 128 bits *is* its low bits
        // zero-extended.
        if let Some(v) = value.as_u128() {
            let start = offset as usize;
            let end = start + size as usize;

            if end <= self.data.len() {
                let data = Arc::make_mut(&mut self.data);
                for i in 0..size as usize {
                    data[start + i] = u128_le_byte(v, i);
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
    ///
    /// A register that currently holds a symbolic value reads back as **zero**,
    /// never as the concrete bytes it happened to hold before the symbolic write
    /// (angr-9ke6b.5). `put`'s symbolic branches leave `self.data` alone — or,
    /// for a partially-concrete sub-register write into a wider symbolic, update
    /// only the written bytes — so without this pass the flat buffer would hand
    /// a caller a stale pre-symbolic value it has no way to distinguish from a
    /// live concrete one. The whole span of a symbolic entry is zeroed, including
    /// any concrete sub-register bytes composed into it: the register as a whole
    /// is not concretely representable, and the symbolic value itself travels
    /// separately (`get_symbolic_register_names` on the export snapshot).
    pub(crate) fn copy_to_bytes(&self, bytes: &mut [u8]) {
        let len = std::cmp::min(bytes.len(), self.data.len());
        bytes[..len].copy_from_slice(&self.data[..len]);
        for (&offset, sym_val) in &self.symbolic {
            let start = std::cmp::min(offset as usize, len);
            let end = std::cmp::min(start + (sym_val.width() / 8) as usize, len);
            bytes[start..end].fill(0);
        }
    }

    /// Fork the register file for path splitting.
    pub(crate) fn fork(&self) -> RegisterFile {
        RegisterFile {
            // O(1) Arc refcount bump; the buffer is copied lazily on the
            // first write to either parent or child via `Arc::make_mut`.
            data: Arc::clone(&self.data),
            symbolic: self.symbolic.clone(),
            arch: self.arch.clone(),
            containment: self.containment.clone(),
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
            containment: self.containment.clone(),
        }
    }

    /// Merge another register file into this one using a merge condition.
    ///
    /// For each register offset, if the values differ between `self` and `other`,
    /// the result is `ITE(merge_cond_other, other_val, self_val)`.
    ///
    /// `merge_cond_other` is the 1-bit condition for `other`'s path being active.
    ///
    /// When the two paths hold symbolic values of *different* widths at the same
    /// offset — one wrote `eax` (32-bit) where the other wrote `rax` (64-bit) —
    /// both sides are first widened to the larger width via [`Self::get`], which
    /// composes each file's own overlay with its concrete backing bytes, and the
    /// ITE is built at that common width (angr-9ke6b.13). Dropping `other`'s
    /// value instead, as this used to, made the merged state behave as if only
    /// `self`'s path could reach the register: an unsound merge.
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

        // Merged values are staged here and applied only after the loop: the
        // width-mismatch arm reads `self` through `get`, which composes
        // neighbouring overlays, so mutating `self.symbolic` mid-loop would make
        // the result depend on `all_offsets`' (unordered) iteration order.
        let mut updates: Vec<(u32, RustBV)> = Vec::new();
        // Spans (offset, byte width) that a widening merge now covers whole;
        // overlays strictly inside them are subsumed and must be dropped, the
        // same cleanup `put` does when a wide symbolic write lands.
        let mut widened: Vec<(u32, u32)> = Vec::new();

        // Merge symbolic registers
        for &offset in &all_offsets {
            let self_val = self.symbolic.get(&offset);
            let other_val = other.symbolic.get(&offset);

            match (self_val, other_val) {
                (Some(sv), Some(ov)) => {
                    if sv.width() == ov.width() {
                        // Both symbolic at same width — ITE merge
                        updates.push((offset, merge_cond_other.ite(ov, sv, ctx)));
                    } else {
                        // Width mismatch — widen both sides, then ITE. `get` at
                        // the wider size returns the narrower side composed with
                        // that file's own concrete high bytes, so neither path's
                        // reachable value is lost.
                        let width = sv.width().max(ov.width());
                        let size = width / 8;
                        let end = offset as usize + size as usize;
                        if width % 8 == 0
                            && size > 0
                            && end <= self.data.len()
                            && end <= other.data.len()
                        {
                            let self_full = self.get(offset, size, ctx);
                            let other_full = other.get(offset, size, ctx);
                            updates
                                .push((offset, merge_cond_other.ite(&other_full, &self_full, ctx)));
                            widened.push((offset, size));
                        } else {
                            // SILENT(cat-c): the wider value runs past the guest
                            // state buffer (or is not byte-sized), so there is no
                            // common width to ITE at; `self`'s value is kept and
                            // `other`'s branch is lost. Unreachable for any real
                            // VEX guest-state offset — a register never straddles
                            // the end of the buffer — hence loud rather than fixed.
                            log::warn!(
                                "RegisterFile::merge: cannot widen offset {offset} ({} vs {} bits, \
                                 self_len={}, other_len={}); keeping self's value and dropping \
                                 other's — merged state may be unsound",
                                sv.width(),
                                ov.width(),
                                self.data.len(),
                                other.data.len()
                            );
                        }
                    }
                }
                (Some(sv), None) => {
                    // self is symbolic, other has no overlay at this exact
                    // offset — read other through `get`, not raw concrete bytes:
                    // a *wider* overlay of other's may still cover this span
                    // (e.g. self wrote `ah`, other wrote all of `rax`), and
                    // reading the backing array would silently substitute stale
                    // concrete for it (angr-49v03).
                    let size = sv.width() / 8;
                    if size > 0 && offset as usize + size as usize <= other.data.len() {
                        let other_full = other.get(offset, size, ctx);
                        updates.push((offset, merge_cond_other.ite(&other_full, sv, ctx)));
                    }
                }
                (None, Some(ov)) => {
                    // self has no overlay at this exact offset, other is
                    // symbolic — same reasoning as above, mirrored.
                    let size = ov.width() / 8;
                    if size > 0 && offset as usize + size as usize <= self.data.len() {
                        let self_full = self.get(offset, size, ctx);
                        updates.push((offset, merge_cond_other.ite(ov, &self_full, ctx)));
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
        //
        // The final chunk is short when `state_size()` is not a multiple of the
        // register width: amd64's guest state is 1060 bytes against 8-byte
        // chunks, so bytes 1056..1060 — the tail of archinfo's segment-selector
        // block — used to fall off the end of the scan and a concrete
        // divergence there was silently dropped in `self`'s favour
        // (angr-91vj9.13, same shape as angr-c7xno.1 one chunk over). Clamping
        // the chunk to what is left covers the whole buffer.
        //
        // Runs *before* `updates` is applied so the `get` calls below still see
        // each file's own pre-merge overlay; staging into the same vectors keeps
        // the "compose, then subsume the inner overlays" contract in one place.
        let reg_bytes = (self.arch.bits() / 8) as usize;
        let len = self.data.len().min(other.data.len());
        let mut off = 0;
        while off < len {
            let reg_bytes = reg_bytes.min(len - off);
            let u32_off = off as u32;
            // Skip offsets that are already handled by symbolic merge
            if !all_offsets.contains(&u32_off) {
                let self_slice = &self.data[off..off + reg_bytes];
                let other_slice = &other.data[off..off + reg_bytes];
                if self_slice != other_slice {
                    // An overlay may cover part of this chunk without living at
                    // its aligned start: x86/amd64 register the high-byte
                    // aliases ah/ch/dh/bh at `GPR_offset + 1` (see `ALIASES` in
                    // arch/amd64.rs and arch/x86.rs), so `all_offsets` holds
                    // unaligned keys. Comparing raw backing bytes there would
                    // insert a concrete-only ITE at the aligned offset that
                    // shadows the already-merged sub-register on the next
                    // full-width read (angr-c7xno.1). Compose both sides
                    // through `get` instead, exactly as the width-mismatch arm
                    // above does.
                    let inner_symbolic =
                        (off + 1..off + reg_bytes).any(|k| all_offsets.contains(&(k as u32)));
                    if inner_symbolic {
                        let size = reg_bytes as u32;
                        let self_full = self.get(u32_off, size, ctx);
                        let other_full = other.get(u32_off, size, ctx);
                        updates.push((u32_off, merge_cond_other.ite(&other_full, &self_full, ctx)));
                        widened.push((u32_off, size));
                    } else {
                        // Concrete values differ — create ITE
                        let width = (reg_bytes * 8) as u32;
                        let self_bv = RustBV::concrete(le_bytes_to_u128(self_slice), width);
                        let other_bv = RustBV::concrete(le_bytes_to_u128(other_slice), width);
                        updates.push((u32_off, merge_cond_other.ite(&other_bv, &self_bv, ctx)));
                    }
                }
            }
            off += reg_bytes;
        }

        for (offset, val) in updates {
            self.symbolic.insert(offset, val);
            merged = true;
        }
        // Drop overlays strictly inside a widened span. This subsumes rather
        // than loses them: both sides of the wide ITE were built with `get`,
        // which composes every overlay inside the span (angr-49v03), so an
        // inner offset merged earlier in this same loop is already represented
        // in the wide value at `offset`.
        for (offset, size) in widened {
            self.symbolic
                .retain(|&k, _| !(k > offset && k < offset + size));
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
        make_cc: || Box::new(AArch64),
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

test_submod!("mod_tests.rs" => tests);
