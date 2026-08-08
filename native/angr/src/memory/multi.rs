//! Multi-cell byte storage for lazy symbolic memory (Phase 1 of angr-czph).
//!
//! When a symbolic-address store concretizes to multiple candidate addresses,
//! the existing eager path builds an `ITE(addr == cand_i, new, current)` chain
//! per candidate cell and writes it back. Each subsequent overlapping store
//! nests another ITE layer — the bottleneck behind `sym-write` (memory
//! `symwrite-eager-vs-lazy-memory`).
//!
//! `MultiPayload` is the lazy alternative: instead of folding into an ITE at
//! store time, each candidate cell records its alternatives as a flat list of
//! `(cond, value)` pairs. The collapse to an ITE happens at load time, scoped
//! to only the bytes the load actually touches.
//!
//! **Panic policy (angr-9ke6b.212):** the two `expect` shapes here are both
//! caller-side invariants of the *coalescing* walk, not properties of guest
//! data: `collapse_run`'s page re-lookup is proved by the `coalesce` guard that
//! built the run, and `build_wider_value`'s first-byte `next()` is proved by
//! the non-empty run length. `build_wider_value`'s `debug_assert!` documents
//! that contract but compiles out in release, so the `expect` is the release
//! guard and is deliberately kept loud — a zero-length run would otherwise
//! silently synthesize a wrong-width value.
//!
//! The same reasoning promotes `MultiAlternative::new`'s width-8 check to an
//! always-on `assert_eq!` (angr-sqfj8.77): it guards the identical
//! wrong-width-value failure mode from the producer side, and the constructor
//! has no error channel to report through.
//!
//! **Enforcement (angr-qwyti.11):** the parent [`memory`](super) module's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` reaches this file; it is
//! restated below so the guarantee is visible when reading this file alone.
//!
//! This module defines the data structures only. Load-time collapse lives in
//! `memory/load.rs` (Phase 1.2, bead angr-n082); store helpers that emit
//! Multi cells live in `memory/store.rs` (Phase 1.3, bead angr-aija).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;

use super::{Address, MemoryPage, Permission, SymbolicMemory};
use crate::symbolic::{RustBV, SymContext, record_mem_ite_depth};
use crate::vex::Endness;

/// Phase 4.2 (angr-mmdh.2): max number of consecutive Multi bytes that
/// `flush_multi_cells` will coalesce into a single wider `symbolic_objects`
/// entry. Capped at 16 bytes (128 bits) so the BV concat fast path in
/// `RustBV::concat_into` stays within the `u128` it uses to short-circuit
/// concrete operands; widths beyond 120 bits + 8 would silently truncate.
const COALESCE_MAX_RUN: usize = 16;

/// One alternative inside a `MultiPayload`. `cond` is a boolean BV (width 1)
/// that selects this alternative; `value` is the byte stored when `cond`
/// holds.
///
/// Invariants the producers must uphold (enforced by callers, not by this
/// type) — see `MultiPayload` doc comment.
#[derive(Debug, Clone)]
pub struct MultiAlternative {
    /// Boolean BV (width 1) under which `value` is the chosen byte.
    pub cond: RustBV,
    /// Byte value (width 8) for this alternative.
    pub value: RustBV,
}

impl MultiAlternative {
    /// # Panics
    ///
    /// Panics if `value` is not width 8. This is an `assert!`, not a
    /// `debug_assert!`, on purpose (angr-sqfj8.77): the workspace release
    /// profile leaves `debug-assertions` off, and this constructor has no
    /// error channel to report through, so a debug-only check would let a
    /// wrong-width byte reach `collapse`/`build_wider_value` and silently
    /// synthesize a wrong-width value in the shipped `.so`. The check is a
    /// `u32` field read — see the module `Panic policy` header.
    pub fn new(cond: RustBV, value: RustBV) -> Self {
        assert_eq!(
            value.width(),
            8,
            "MultiAlternative::value must be a single byte (width 8)"
        );
        MultiAlternative { cond, value }
    }
}

/// Memoized result of collapsing a `MultiPayload` to a single BV. The
/// collapse is keyed on the page's concrete byte at the time of computation
/// because that byte is the final `else` leaf of the right-fold ITE chain.
/// If a later concrete store mutates the underlying page byte without
/// clearing the Multi marker, the cache must be rebuilt — that is what the
/// `default_byte` field detects.
#[derive(Debug, Clone)]
struct CachedCollapse {
    default_byte: u8,
    bv: RustBV,
}

/// A set of alternatives for a single byte cell.
///
/// # Invariants (caller-enforced)
///
/// * Conditions are pairwise distinct concretized address equalities
///   (e.g. `addr == 0x1000`, `addr == 0x1004`) so that under any model
///   exactly one alternative's `cond` evaluates to true. Producers in
///   `memory/store.rs` construct these from `ConcretizationResult::Multiple`
///   /`Strided` results.
/// * All `value` BVs have width 8 (one byte). Wider values must be split
///   per byte before populating the payload.
///
/// # Counter contract
///
/// Per memory `invariant-mem-ite-depth-counter`, code that inserts a payload
/// via `SymbolicMemory::set_multi_alternatives` MUST call
/// `crate::symbolic::record_mem_ite_depth(payload.len() as u32)` so the Phase
/// 0 baseline comparison stays direct. The public setter handles this
/// automatically; private mutation paths must not bypass it.
///
/// # Collapse cache (Phase 3, bead angr-j0n4)
///
/// `cached_collapse` memoizes the right-folded ITE BV produced by
/// `collapse`. Reads (e.g. `assemble_load_with_multi`) reuse the cached BV
/// when the page's concrete default byte is unchanged. Any append-mutation
/// (`push`) invalidates the cache; `set_multi_alternatives` always installs
/// a fresh payload (cache starts as `None`) so the merge path in
/// `install_multi_for_candidates` is automatically safe.
#[derive(Debug, Default)]
pub struct MultiPayload {
    alternatives: Vec<MultiAlternative>,
    cached_collapse: RefCell<Option<CachedCollapse>>,
}

impl Clone for MultiPayload {
    fn clone(&self) -> Self {
        MultiPayload {
            alternatives: self.alternatives.clone(),
            // BVs are refcounted on the Z3 side, so cloning the cache is
            // cheap. The fork path benefits from carrying it forward.
            cached_collapse: RefCell::new(self.cached_collapse.borrow().clone()),
        }
    }
}

impl MultiPayload {
    /// Build a payload from a list of alternatives. Caller is responsible for
    /// the invariants documented on the type.
    pub fn from_alternatives(alternatives: Vec<MultiAlternative>) -> Self {
        MultiPayload {
            alternatives,
            cached_collapse: RefCell::new(None),
        }
    }

    /// Number of alternatives in this cell.
    pub fn len(&self) -> usize {
        self.alternatives.len()
    }

    /// True if the payload holds no alternatives. An empty payload should not
    /// be stored in `multi_objects`; callers should drop or replace the cell.
    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Read-only view of the alternatives, in insertion order. Load-time
    /// collapse iterates this list to build the ITE.
    pub fn alternatives(&self) -> &[MultiAlternative] {
        &self.alternatives
    }

    /// Append one alternative to the payload. Invalidates the collapse cache.
    ///
    /// This is the lazy-store primitive: emitting a `Multi` cell from a
    /// symbolic-address store appends the candidate's `(addr == cand, value)`
    /// pair without rebuilding any ITE. The collapse cost is paid at load
    /// time instead.
    pub fn push(&mut self, alt: MultiAlternative) {
        self.alternatives.push(alt);
        self.cached_collapse.get_mut().take();
    }

    /// Right-fold the alternatives into an ITE BV with `default_byte` as the
    /// final `else` leaf. Reuses the memoized BV when the cached default byte
    /// matches; otherwise recomputes and stores the result.
    ///
    /// Returns an 8-bit `RustBV`. The caller must ensure all alternative
    /// `value` BVs are 8 bits (the type invariant).
    pub fn collapse(&self, default_byte: u8, ctx: &SymContext) -> RustBV {
        if let Some(cached) = self.cached_collapse.borrow().as_ref()
            && cached.default_byte == default_byte
        {
            return cached.bv.clone();
        }
        let default = RustBV::concrete(default_byte as u128, 8);
        let mut acc = default;
        for alt in self.alternatives.iter().rev() {
            acc = alt.cond.ite(&alt.value, &acc, ctx);
        }
        *self.cached_collapse.borrow_mut() = Some(CachedCollapse {
            default_byte,
            bv: acc.clone(),
        });
        acc
    }

    /// True if the collapse cache currently holds a value. Test-only helper.
    #[cfg(test)]
    pub(crate) fn has_cached_collapse(&self) -> bool {
        self.cached_collapse.borrow().is_some()
    }

    /// Deep-translate every alternative `(cond, value)` BV into `target_ctx`
    /// (angr-ahypj). Drops the collapse cache so it rebuilds against the
    /// target context on first load — the cached BV lives in the source
    /// context and must never leak across the boundary.
    ///
    /// See [`crate::symbolic::RustBV::translate_into`] for the cross-context
    /// `Z3_translate` primitive this composes over.
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> MultiPayload {
        let alternatives = self
            .alternatives
            .iter()
            .map(|alt| {
                MultiAlternative::new(
                    alt.cond.translate_into(target_ctx),
                    alt.value.translate_into(target_ctx),
                )
            })
            .collect();
        MultiPayload {
            alternatives,
            cached_collapse: RefCell::new(None),
        }
    }
}

/// Compile-time proof that [`SymbolicMemory::flush_multi_cells`] has run.
///
/// Only [`SymbolicMemory::flush_multi_cells`] can construct one (the tuple
/// field is private to this module), so any API that requires a `&MultiFlushed`
/// argument cannot be called without the caller having flushed first — turning
/// "a read path forgot to flush Multi cells before looking at
/// `symbolic_objects`" (the angr-9ke6b.96/.83/.101 and angr-sqfj8.26/.27/.52/.71
/// bug family) from a discipline lapse into a compile error. See the plan at
/// `/home/ubuntu/.claude/plans/review-the-last-two-effervescent-starlight.md`.
#[derive(Debug)]
pub struct MultiFlushed(());

impl SymbolicMemory {
    /// Install lazy alternatives at a single byte address.
    ///
    /// Marks the byte's page as Multi (via `MemoryPage::mark_multi`), clears
    /// any conflicting plain-Symbolic state at the same address, and stores
    /// the payload in `multi_objects`. Auto-maps the containing page if it
    /// is not yet present, matching the behavior of `import_symbolic_value`
    /// so callers do not have to pre-map stack regions.
    ///
    /// Per memory `invariant-mem-ite-depth-counter`, this records the
    /// alternative count via `crate::symbolic::record_mem_ite_depth` so the
    /// Phase 0 baseline comparison reflects every Multi insertion.
    ///
    /// A payload with zero alternatives clears the cell instead of
    /// installing an empty entry.
    pub fn set_multi_alternatives(&mut self, addr: impl Into<Address>, payload: MultiPayload) {
        let addr = addr.into();
        if payload.is_empty() {
            self.clear_multi_at(addr);
            return;
        }

        let depth = payload.len() as u32;

        // Auto-map the page if missing. Matches import_symbolic_value's
        // policy so callers (test rigs, future SimProcedure wiring) do not
        // need to pre-map stack regions.
        let page_num = addr.page_num();
        let offset = addr.page_offset();
        let page_addr = page_num << 12;
        let page = self
            .pages
            .entry(page_num)
            .or_insert_with(|| MemoryPage::new(page_addr, Permission::RW));

        // Clear any prior plain-Symbolic state at this byte: a Multi cell
        // supersedes single-symbolic. The page bitmap and symbolic_objects
        // sidecar must stay in sync.
        page.clear_multi(offset); // no-op if not currently Multi
        page.mark_multi(offset);

        self.symbolic_objects.remove(&addr);
        self.symbolic_spans.remove(&addr);

        self.multi_objects.insert(addr, payload);
        // Phase 4.1: bump per-byte version so the wider-load cache notices
        // this installation. Must run regardless of whether a prior
        // payload existed at this address.
        self.bump_multi_version(addr);
        // Counter contract: callers can't bypass this — this is the only
        // public path that installs a Multi cell.
        record_mem_ite_depth(depth);
    }

    /// Read-only access to the lazy alternatives at a byte address, if any.
    pub fn get_multi_alternatives(&self, addr: impl Into<Address>) -> Option<&MultiPayload> {
        self.multi_objects.get(&addr.into())
    }

    /// Remove lazy alternatives at a byte address and clear the page bit.
    /// Safe to call on a byte that is not currently Multi (no-op).
    pub fn clear_multi_at(&mut self, addr: impl Into<Address>) {
        let addr = addr.into();
        let had_payload = self.multi_objects.remove(&addr).is_some();
        let page_num = addr.page_num();
        let offset = addr.page_offset();
        // angr-fgqco: probe read-only before taking `&mut`. `pages` is an
        // `im::OrdMap` shared with forked states, so `get_mut` copies the tree
        // path down to the target page even when `clear_multi` would be a
        // no-op. `store_symbolic` calls this for *every byte* of every
        // symbolic store as soon as a single Multi cell exists anywhere in
        // memory, so the no-op case has to stay free of page-map mutation.
        if self
            .pages
            .get(&page_num)
            .is_some_and(|page| page.is_multi(offset))
            && let Some(page) = self.pages.get_mut(&page_num)
        {
            page.clear_multi(offset);
        }
        // Phase 4.1: bump version only when a payload was actually present
        // so the no-op case skips the version map too.
        if had_payload {
            self.bump_multi_version(addr);
        }
    }

    /// Count of byte addresses currently carrying lazy Multi alternatives.
    /// Used by tests and by future profiling to track the Phase 1 footprint.
    pub fn multi_cell_count(&self) -> usize {
        self.multi_objects.len()
    }

    /// Materialize every Multi cell into a `symbolic_objects` entry so the
    /// state export pipeline (`_sync_rust_symbolic_objects_to_state` in
    /// `rust_state_export.py`) sees the lazy alternatives.
    ///
    /// Phase 2 (angr-qh5u): flips the default symbolic-address store to
    /// install Multi cells. Multi bytes are tracked in a parallel
    /// `multi_bitmap`, not `symbolic_bitmap`, so `symbolic_offsets()`
    /// does not list them. Without this flush the page exporter writes
    /// the underlying concrete `data[]` byte (usually zero), losing the
    /// alternative values entirely.
    ///
    /// # Phase 4.2 (angr-mmdh.2): run coalescing
    ///
    /// Per the `phase41-bottleneck` memory, emitting one
    /// `symbolic_objects` entry per Multi byte forces `state_api`
    /// `_get_state_symbolic_z3_asts` to build N independent Z3 ASTs
    /// (one per byte) and the Python exporter to call
    /// `state.memory.store()` N times. A symbolic-address store of an
    /// N-byte value resolved to K candidates writes N×K alternatives
    /// across N adjacent byte addresses — each byte's payload shares the
    /// same per-candidate `cond` `RustBV` (created once per call in
    /// `install_multi_for_candidates` and `.clone()`d into every byte).
    ///
    /// This flush scans `multi_objects` in address order and groups runs
    /// of adjacent bytes whose payloads share identical cond fingerprints.
    /// For each run of length R (capped by `COALESCE_MAX_RUN`):
    ///   * Wider per-alternative values are built by `concat`-ing the
    ///     per-byte `value` BVs endian-correctly (matching the
    ///     `extract_byte_lane` convention used by load paths).
    ///   * A wider concrete default is built the same way from the
    ///     pages' current bytes.
    ///   * A single right-folded ITE BV of width 8·R is written to
    ///     `symbolic_objects[run_start]`, and `symbolic_spans` is
    ///     populated for the interior bytes so a per-byte load still
    ///     finds the wider object.
    ///
    /// Singletons (run length 1) and runs whose pages aren't all mapped
    /// fall back to the per-byte path, byte-identical to the pre-Phase-4.2
    /// behaviour.
    ///
    /// The wider symbolic_object is observable-equivalent to an eager
    /// `state.memory.store(run_start, value, endness=...)` of the wider
    /// ITE — every load path (`assemble_load_with_multi`,
    /// `try_byte_merge_load`, `load_concrete`) already handles wider
    /// objects via `extract_byte_lane` / `symbolic_spans`.
    #[allow(
        clippy::expect_used,
        reason = "the `coalesce` guard immediately above proves every page in `entries[i..j]` is present (`(i..j).all(|k| self.pages.contains_key(..))`), and this loop re-looks-up exactly those same entries — an index-after-check, not a guest-controlled lookup"
    )]
    pub fn flush_multi_cells(&mut self, ctx: &crate::symbolic::SymContext) -> MultiFlushed {
        if self.multi_objects.is_empty() {
            return MultiFlushed(());
        }

        let multi = std::mem::take(&mut self.multi_objects);
        let endness = self.endness;

        // Sort entries by address so adjacent-byte runs surface in a
        // single linear scan.
        let mut entries: Vec<(Address, MultiPayload)> = multi.into_iter().collect();
        entries.sort_by_key(|(addr, _)| *addr);

        let mut i = 0;
        while i < entries.len() {
            let start_addr = entries[i].0;
            let start_fp = payload_cond_fingerprint(&entries[i].1);

            // Greedy extension: same alt count + same alt-cond identity
            // order across consecutive byte addresses. Bytes whose conds
            // differ (e.g. installed by a separate store, or merged with
            // an extra alternative) terminate the run.
            let mut j = i + 1;
            while j < entries.len()
                && j - i < COALESCE_MAX_RUN
                && entries[j].0 == entries[j - 1].0 + 1
                && payload_cond_fingerprint(&entries[j].1) == start_fp
            {
                j += 1;
            }
            let run_len = j - i;

            // Need every byte's page mapped to read the wider concrete
            // default. If any page is unmapped we fall back to per-byte
            // (matches the pre-Phase-4.2 drop-on-unmap behaviour).
            let coalesce = run_len >= 2 && {
                (i..j).all(|k| self.pages.contains_key(&entries[k].0.page_num()))
            };

            if coalesce {
                // Collect per-byte concrete defaults.
                let mut concrete_bytes: Vec<u8> = Vec::with_capacity(run_len);
                for entry in &entries[i..j] {
                    let byte_addr = entry.0;
                    let page_num = byte_addr.page_num();
                    let offset = byte_addr.page_offset();
                    let page = self
                        .pages
                        .get(&page_num)
                        .expect("page presence verified by `coalesce` guard");
                    concrete_bytes
                        .push(page.load_concrete(offset, 1).first().copied().unwrap_or(0));
                }

                // Build wider default (right-fold ITE's else leaf).
                let default_bvs: Vec<RustBV> = concrete_bytes
                    .iter()
                    .map(|b| RustBV::concrete(*b as u128, 8))
                    .collect();
                let wider_default = build_wider_value(&default_bvs, endness, ctx);

                // Right-fold per-alt wider values into the final ITE BV.
                let alt_count = entries[i].1.alternatives().len();
                let mut acc = wider_default;
                for c in (0..alt_count).rev() {
                    let cond = entries[i].1.alternatives()[c].cond.clone();
                    let mut byte_vals: Vec<RustBV> = Vec::with_capacity(run_len);
                    for entry in &entries[i..j] {
                        byte_vals.push(entry.1.alternatives()[c].value.clone());
                    }
                    let wider_val = build_wider_value(&byte_vals, endness, ctx);
                    acc = cond.ite(&wider_val, &acc, ctx);
                }

                // Apply state mutations: bump versions, drop Multi bits,
                // set Symbolic bits, dirty pages, insert wider object,
                // record reverse-span entries for interior bytes.
                for entry in &entries[i..j] {
                    let byte_addr = entry.0;
                    self.bump_multi_version(byte_addr);
                    let page_num = byte_addr.page_num();
                    let offset = byte_addr.page_offset();
                    if let Some(page) = self.pages.get_mut(&page_num) {
                        page.clear_multi(offset);
                        page.mark_symbolic(offset, 1);
                    }
                    self.dirty_pages.insert(page_num);
                }
                self.symbolic_objects.insert(start_addr, acc);
                let width_bits = (run_len * 8) as u32;
                for b in 1..run_len {
                    self.symbolic_spans
                        .insert(start_addr + b as u64, (start_addr, width_bits));
                }

                i = j;
                continue;
            }

            // Fall-back per-byte path: singleton runs and runs with any
            // unmapped page. Byte-identical to the pre-Phase-4.2 flush.
            for (byte_addr, payload) in &entries[i..j] {
                self.bump_multi_version(*byte_addr);
                let page_num = byte_addr.page_num();
                let offset = byte_addr.page_offset();
                let concrete_byte: u8 = match self.pages.get(&page_num) {
                    Some(page) => page.load_concrete(offset, 1).first().copied().unwrap_or(0),
                    None => continue,
                };
                let acc = payload.collapse(concrete_byte, ctx);
                if let Some(page) = self.pages.get_mut(&page_num) {
                    page.clear_multi(offset);
                    page.mark_symbolic(offset, 1);
                }
                self.symbolic_objects.insert(*byte_addr, acc);
                self.dirty_pages.insert(page_num);
            }
            i = j;
        }

        MultiFlushed(())
    }
}

/// Phase 4.2 (angr-mmdh.2): identity-based fingerprint of a `MultiAlternative`'s
/// cond. Two alternatives produced by the same `install_multi_for_candidates`
/// call share an `Arc<[RustBV]>` operands pointer (the cond is built once per
/// candidate and `.clone()`d into every byte); bytes whose payload conds match
/// in count and per-position fingerprint originated from the same store path
/// and can be safely coalesced into a single wider `symbolic_objects` entry.
fn cond_fingerprint(bv: &RustBV) -> u64 {
    match bv {
        RustBV::Expression { operands, .. } => {
            // Within one `flush_multi_cells` call the entries Vec keeps every
            // Arc live, so ptr reuse via free/realloc can't happen — pointer
            // identity is a stable cross-byte comparison key.
            std::sync::Arc::as_ptr(operands) as *const () as usize as u64
        }
        RustBV::Concrete { value, width } => {
            // Mix in a salt distinct from the other variants so a Concrete 0
            // can't alias a Symbolic id 0.
            (*value as u64)
                .wrapping_mul(0x100000001b3)
                .wrapping_add(*width as u64)
                ^ 0x1
        }
        RustBV::Symbolic { id, .. } => *id ^ 0x2,
        RustBV::Constrained { id, .. } => *id ^ 0x3,
    }
}

/// Cond fingerprint for a whole `MultiPayload`. Used by `flush_multi_cells` to
/// decide whether a run of adjacent bytes can be coalesced.
fn payload_cond_fingerprint(payload: &MultiPayload) -> Vec<u64> {
    payload
        .alternatives()
        .iter()
        .map(|a| cond_fingerprint(&a.cond))
        .collect()
}

/// Concat per-byte `RustBV`s into a wider value matching the memory's
/// endianness convention (the same one `extract_byte_lane` decodes):
///   * `Little`: byte 0 (lowest addr) is the LSB, so byte `N-1` is concat'd
///     into the high bits first.
///   * `Big`: byte 0 (lowest addr) is the MSB, so byte 0 is concat'd into
///     the high bits first.
///
/// Caller guarantees `bytes` is non-empty and each element is width 8.
#[allow(
    clippy::expect_used,
    reason = "first-element `next()` on a non-empty slice: both call sites are inside `flush_multi_cells`'s coalesce branch, which only runs for `run_len >= 2`. The `debug_assert!` below states the contract but compiles out in release, so this expect is the release-mode guard and is kept loud on purpose — see the module Panic policy header"
)]
fn build_wider_value(bytes: &[RustBV], endness: Endness, ctx: &SymContext) -> RustBV {
    debug_assert!(
        !bytes.is_empty(),
        "build_wider_value requires non-empty input"
    );
    match endness {
        Endness::Little => {
            let mut iter = bytes.iter().rev();
            let mut acc = iter.next().expect("non-empty").clone();
            for b in iter {
                acc = acc.concat(b, ctx);
            }
            acc
        }
        Endness::Big => {
            let mut iter = bytes.iter();
            let mut acc = iter.next().expect("non-empty").clone();
            for b in iter {
                acc = acc.concat(b, ctx);
            }
            acc
        }
    }
}
