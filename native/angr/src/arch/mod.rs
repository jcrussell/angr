//! Architecture definitions and register files.
//!
//! This module provides:
//! - `Arch` trait for architecture abstraction
//! - The [`RegisterFile`] struct, its serde shadow [`RegisterFileData`] and
//!   the register-name lookup helpers; its get/put/merge implementation lives
//!   in the `register_file` sibling
//! - [`AddrOrSymbolic`], the SP / return-address newtype
//! - The supported-architecture table, in the `registry` sibling
//! - Calling convention implementations for argument extraction, in
//!   `calling_conventions`
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** register
//! offsets and sizes reaching this module come from VEX/guest state, so it
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]` — which, being
//! an inner attribute, covers the sibling modules declared below too. There are
//! no `unwrap`/`expect` sites left: the sub-register composition in
//! `RegisterFile::compose_range` folds its `parts` vec with
//! `if let Some(mut result) = parts.pop()`, which also subsumes the old
//! non-empty pre-check.
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

// Sibling implementation modules split out of this file (angr-5mnx3.1).
//
// `register_file` is private — it only adds inherent methods to a type this
// file already exports, so it has no surface of its own. `registry` is
// `pub(crate)` because `ArchDesc` / `ALL_ARCHES` are read only by tests;
// re-exporting those two here would be an unused import in the plain lib
// build. Its three lookup helpers are used crate-wide and are re-exported, so
// `crate::arch::arch_from_name` and friends keep their existing paths.
mod register_file;
pub(crate) mod registry;

pub(crate) use registry::{arch_desc_from_name, arch_from_name, arch_from_vex};

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
    /// `registry_tests::test_all_arches_report_expected_special_register_offsets`,
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
    /// - `registry_tests::register_names_all_resolve_and_fit_in_u128`
    /// - `registry_tests::exported_register_names_reverse_resolve` (angr-9ke6b.9:
    ///   catches an exported name that was left in `ALIASES`)
    /// - `registry_tests::vex_bookkeeping_fields_resolve_on_every_arch`
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
    /// `registry_tests::every_narrow_canonical_register_is_exported` walks it to
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
