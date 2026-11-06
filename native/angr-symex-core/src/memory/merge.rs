//! State merging for `SymbolicMemory`.
//!
//! Extracted from `memory/mod.rs` (angr-03vl4.39), matching the file-splitting
//! pattern `load`/`store`/`multi`/`symbolic_objects` already follow. Holds
//! [`SymbolicMemory::merge`] and the helpers only it uses.
//!
//! The merge contract: for every byte that differs between the two arms the
//! merged value becomes `ITE(merge_cond_other, other_byte, self_byte)`, with
//! `Multi` cells collapsed to their ITE BV first so a Multi byte and a plain
//! symbolic byte feed the same expression shape (angr-op0dn.11.2.2). Deferred
//! (pending) symbolic stores cannot be collapsed that way, so they are instead
//! *guarded* by the arm's condition and carried forward unmaterialized.
//!
//! A page only one arm holds is walked against a zero stand-in for the other
//! arm (angr-vmi2l.187), lazy region or not: a page the engine materializes
//! without host bytes is the zero page (`auto_map_zero_page`, and the
//! store-side fresh map outside every region), and the manager's
//! `fetch_host_pages_for_merge` fetches every host-servable page into the
//! absent arm before the merge. Nothing is adopted from either arm unguarded.

use rustc_hash::FxHashMap;

use super::multi::{MultiAlternative, MultiPayload};
use super::page::{PAGE_SIZE, PageIndex};
use super::{Address, MemoryPage, PendingWrite, SymMap, SymbolicMemory};
use crate::symbolic::RustBV;
use angr_vex::Endness;

impl SymbolicMemory {
    /// The page an arm that never mapped `base_addr` reads: the zero page
    /// `auto_map_zero_page` would map, uninitialized outside
    /// ZERO_FILL_UNCONSTRAINED_MEMORY.
    fn fresh_stand_in(&self, base_addr: u64, permissions: super::Permission) -> MemoryPage {
        if self.zero_fill_unconstrained {
            MemoryPage::new(base_addr, permissions)
        } else {
            MemoryPage::new_uninit(base_addr, permissions)
        }
    }

    /// Merge another memory into this one using a merge condition.
    ///
    /// For each byte that differs between `self` and `other`, the merged
    /// value is `ITE(merge_cond_other, other_byte, self_byte)`.
    ///
    /// Returns true if any memory was actually merged (values differed).
    #[allow(
        clippy::expect_used,
        reason = "`s_multi`/`o_multi` are `page.has_multi() && page.is_multi(i)` bitmap reads taken from the same byte index a few lines above, and that bit is set only alongside the `multi_objects` entry for the byte's address — so both lookups are proved by the guard that gates this branch, and neither depends on guest input"
    )]
    pub fn merge(&mut self, other: &SymbolicMemory, merge_cond_other: &RustBV) -> bool {
        let mut merged = false;

        // Collect all page numbers from both memories
        let self_pages: std::collections::HashSet<u64> = self.pages.keys().copied().collect();
        let other_pages: std::collections::HashSet<u64> = other.pages.keys().copied().collect();
        let all_pages: std::collections::HashSet<u64> =
            self_pages.union(&other_pages).copied().collect();

        // Collect merge operations first to avoid borrow conflicts
        // (page_num, addr, ite_val, self_side_was_multi)
        let mut merge_ops: Vec<(u64, Address, RustBV, bool)> = Vec::new();
        let mut pages_to_add: Vec<(u64, MemoryPage)> = Vec::new();
        // Multi-cell unions (M3-2b, angr-op0dn.11.2.2): a byte that is Multi
        // on *both* arms merges to a single lazy Multi cell whose alternatives
        // union the arms under merge-condition guards, staying lazy rather
        // than collapsing to a `symbolic_objects` ITE. Collected here and
        // installed after the borrow of `self.pages` is released.
        let mut multi_ops: Vec<(Address, MultiPayload)> = Vec::new();

        for &page_num in &all_pages {
            let self_page = self.pages.get(&page_num);
            let other_page = other.pages.get(&page_num);

            let base_addr = Address(PageIndex::from_raw(page_num).base_addr());
            // A page only one arm holds is not "unchanged" on the other arm:
            // there it is unmaterialized, and reads as the zero page the
            // engine maps for a fresh lazy page (`auto_map_zero_page`, and the
            // zero page `map_fresh_pages_for_store` maps for a store outside
            // every region). Walk it against that zero stand-in so each byte
            // the holding arm changed is guarded by its own condition instead
            // of leaking onto the other arm's paths (angr-vmi2l.187). A page
            // the host can serve is fetched into the absent arm before the
            // merge (`_merge_states`), so only host-declined pages get here.
            let blank;
            let (sp, op) = match (self_page, other_page) {
                (Some(sp), Some(op)) => {
                    // CoW fast path (angr-op0dn.11.2.1, productionizes S5a): a
                    // structurally-shared page with no symbolic/Multi overlay is
                    // provably byte-identical, so it contributes nothing to the
                    // merge. Skip it without materializing PAGE_SIZE bytes,
                    // making merge cost proportional to divergent pages rather
                    // than all shared pages. This is a strict subset of the
                    // value-equality early-out below, so it is semantics-neutral.
                    if sp.is_shared_identical(op) {
                        continue;
                    }
                    (sp, op)
                }
                (Some(sp), None) => {
                    blank = self.fresh_stand_in(base_addr.raw(), sp.permissions());
                    (sp, &blank)
                }
                (None, Some(op)) => {
                    // `self` takes the page so the merged state keeps it
                    // mapped; the walk below fills in every byte `other`
                    // changed from zero.
                    blank = self.fresh_stand_in(base_addr.raw(), op.permissions());
                    pages_to_add.push((page_num, blank.clone()));
                    (&blank, op)
                }
                (None, None) => continue,
            };
            #[cfg(test)]
            super::tests::merge_instrument::note_page_walked();
            // Compare concrete data. The Multi
            // guards mirror the plain-symbolic ones: a page carrying
            // Multi cells can never take the concrete-equality
            // early-out because Multi divergence lives in
            // `multi_objects`, invisible to a `data[]` compare
            // (angr-op0dn.11.2.2).
            let s_data = sp.load_concrete(0, PAGE_SIZE as u16);
            let o_data = op.load_concrete(0, PAGE_SIZE as u16);

            if s_data == o_data
                && !sp.has_symbolic()
                && !op.has_symbolic()
                && !sp.has_multi()
                && !op.has_multi()
            {
                continue;
            }

            for i in 0..PAGE_SIZE as usize {
                let s_byte = s_data[i];
                let o_byte = o_data[i];

                let s_sym = sp.has_symbolic() && sp.is_symbolic(i as u16);
                let o_sym = op.has_symbolic() && op.is_symbolic(i as u16);
                let s_multi = sp.has_multi() && sp.is_multi(i as u16);
                let o_multi = op.has_multi() && op.is_multi(i as u16);

                // A byte contributes nothing only when it is plain
                // concrete on both arms AND the bytes are equal.
                if !s_sym && !o_sym && !s_multi && !o_multi && s_byte == o_byte {
                    continue;
                }

                // overflow-ok: `Add<u64> for Address` is `wrapping_add`
                // and `i` is a page offset, so `i < PAGE_SIZE`.
                let addr = base_addr + i as u64;

                // Both-Multi lazy union (design-doc merge rule): keep
                // the result a Multi cell, guarding each arm's
                // alternatives so exactly one arm's chain is live per
                // model. Neither chain is exhaustive (each alternative
                // is `addr == cand_i`), so a model where no candidate
                // of the live arm matches falls through to that arm's
                // page byte. The merged page keeps self's byte as the
                // collapse else-leaf, so other's differing byte rides
                // as the innermost alternative under `merge_cond_other`
                // (angr-vmi2l.188).
                if s_multi && o_multi {
                    let not_cond = merge_cond_other.not();
                    let sp_pl = self
                        .multi_objects
                        .get(&addr)
                        .expect("s_multi implies a payload");
                    let op_pl = other
                        .multi_objects
                        .get(&addr)
                        .expect("o_multi implies a payload");
                    let mut alts = Vec::with_capacity(sp_pl.len() + op_pl.len() + 1);
                    if s_byte != o_byte {
                        alts.push(MultiAlternative::new(
                            merge_cond_other.clone(),
                            RustBV::concrete(u128::from(o_byte), 8),
                        ));
                    }
                    for alt in sp_pl.alternatives() {
                        alts.push(MultiAlternative::new(
                            not_cond.and(&alt.cond),
                            alt.value.clone(),
                        ));
                    }
                    for alt in op_pl.alternatives() {
                        alts.push(MultiAlternative::new(
                            merge_cond_other.and(&alt.cond),
                            alt.value.clone(),
                        ));
                    }
                    multi_ops.push((addr, MultiPayload::from_alternatives(alts)));
                    continue;
                }

                // Otherwise collapse any Multi side to a BV and ITE it
                // against the other side (concrete / plain-symbolic).
                let self_val = Self::merge_byte_value(
                    &self.symbolic_objects,
                    &self.symbolic_spans,
                    &self.multi_objects,
                    addr,
                    s_byte,
                    s_sym,
                    s_multi,
                    self.endness,
                );
                let other_val = Self::merge_byte_value(
                    &other.symbolic_objects,
                    &other.symbolic_spans,
                    &other.multi_objects,
                    addr,
                    o_byte,
                    o_sym,
                    o_multi,
                    other.endness,
                );

                let ite_val = merge_cond_other.ite(&other_val, &self_val);
                merge_ops.push((page_num, addr, ite_val, s_multi));
            }
        }

        // Install the zero stand-ins for other-only pages first, so the
        // byte apply below marks their merged bytes symbolic.
        for (page_num, page) in pages_to_add {
            self.pages.insert(page_num, page);
            merged = true;
        }

        // Apply collected merge operations
        for (page_num, addr, ite_val, s_was_multi) in merge_ops {
            // A byte reaching `merge_ops` merged to a plain-Symbolic ITE, so a
            // Multi cell `self` still holds here has been *superseded* — the
            // both-Multi union branch `continue`d above, so this is exactly
            // the `s_multi && !o_multi` collapse. Leaving the payload and its
            // bitmap bit behind made the freshly-merged value unreachable:
            // `load_concrete_common`'s `range_has_multi` check dispatches
            // Multi-marked bytes to the Multi path before `symbolic_objects`
            // is ever consulted, and `flush_multi_cells` would later overwrite
            // the merged entry with the stale single-arm payload
            // (angr-0jh0j.31). Gated on the flag rather than calling the
            // no-op-safe `clear_multi_at` unconditionally so the common
            // Multi-free merge keeps its per-byte cost at zero map lookups.
            if s_was_multi {
                self.clear_multi_at(addr);
            }
            // Split any wider object covering `addr` into per-byte lanes before
            // the 8-bit ITE replaces this byte, so no stale wide entry or span
            // outlives the walk. Every byte of such an object sits on a walked
            // page, so its other lanes are replaced in turn.
            self.retire_symbolic_overlapping(addr, 1);
            self.symbolic_objects.insert(addr, ite_val);
            self.symbolic_spans.insert(addr, (addr, 8));
            let offset_in_page = addr.page_offset();
            if let Some(page) = self.pages.get_mut(&page_num) {
                page.mark_symbolic(offset_in_page, 1);
            }
            merged = true;
        }

        // Install both-Multi unions. `set_multi_alternatives` clears any
        // conflicting plain-Symbolic state at the byte and marks the page
        // Multi, keeping the bitmaps and sidecars in sync. Applied before the
        // symbolic-object merge below so a stray `other` entry can't shadow a
        // freshly-installed Multi cell.
        // A byte stays uninitialized only where both arms left it so. One
        // uninit on a single arm reads 0 there, which the walk compared as
        // such: it keeps that 0, its reading before angr-vmi2l.325, rather
        // than an ITE against a filler only a later load would mint.
        for page_num in &all_pages {
            if let (Some(sp), Some(op)) = (self.pages.get_mut(page_num), other.pages.get(page_num))
            {
                sp.intersect_uninit(op);
            }
        }

        for (addr, payload) in multi_ops {
            self.set_multi_alternatives(addr, payload);
            merged = true;
        }

        // Merge pending writes with merge-condition guards
        // (M3-2b; angr-op0dn.11.2.2). A deferred symbolic store is arm-specific, so a
        // blind `extend` would let `other`'s writes fire on `self`'s paths (and
        // leaves `self`'s own writes firing on `other`'s paths). Guard each
        // arm's writes so a write only materializes on the path that issued it:
        // `self`'s writes under `!merge_cond_other`, `other`'s under
        // `merge_cond_other`. Guarding composes with any pre-existing
        // conditional-store condition via `And`.
        if !self.pending_writes.is_empty() {
            let not_cond = merge_cond_other.not();
            let guarded: Vec<PendingWrite> = self
                .pending_writes
                .iter()
                .map(|pw| Self::guard_pending_write(pw, &not_cond))
                .collect();
            self.pending_writes = guarded;
            merged = true;
        }
        if !other.pending_writes.is_empty() {
            self.pending_writes.extend(
                other
                    .pending_writes
                    .iter()
                    .map(|pw| Self::guard_pending_write(pw, merge_cond_other)),
            );
            merged = true;
        }

        // Accumulating sidecars (angr-91vj9.3). None of these are keyed off
        // the page bitmaps, so the byte walk above cannot reconstruct them —
        // each grows monotonically over a branch's life, which makes a union
        // the only treatment consistent with its meaning:
        //
        // * `dirty_pages`: a page `other` wrote — including an other-only page
        //   walked against the zero stand-in above — is dirty in the merged state too.
        //   Dropping it strands the write: `_replay_rust_dirty_pages` never
        //   pushes the page to the Python mirror, which then reads stale bytes.
        // * `lazy_regions`: `RustSimState::add_memory_lazy_region` runs
        //   mid-branch from the callback memory proxy, so `other` can hold a
        //   region `self` lacks; dropping it turns a later store there into a
        //   spurious `Unmapped` instead of an auto-map.
        // * `imported_addrs`: `store_symbolic_from_python` likewise runs
        //   mid-branch. An entry only suppresses re-export while the live
        //   object *is* the recorded value, so a union is safe in both
        //   directions: a byte the merge rewrote (an ITE) no longer matches
        //   and is exported (angr-ccp2w).
        // * `fillers`: each arm's minted fillers stay registered variables of
        //   the merged state, as Python's `temporal_tracked_variables` union
        //   (angr-vmi2l.321); a shared ancestor's filler appears once.
        //
        // They deliberately do not flip `merged`, which reports whether any
        // stored *value* differed between the arms.
        self.dirty_pages.extend(other.dirty_pages.iter().copied());
        for region in &other.lazy_regions {
            if !self.lazy_regions.contains(region) {
                self.lazy_regions.push(*region);
            }
        }
        for (addr, bv) in &other.imported_addrs {
            self.imported_addrs
                .entry(*addr)
                .or_insert_with(|| bv.clone());
        }
        for filler in &other.fillers {
            if !self.fillers.contains(filler) {
                self.fillers.push_back(filler.clone());
            }
        }

        merged
    }

    /// Extract a single byte's merge value: collapse a Multi cell to its ITE
    /// BV, read a plain-Symbolic byte from `symbolic_objects`, or fall back to
    /// the concrete page byte. Shared by the two arms of the byte-merge loop so
    /// Multi and plain-symbolic bytes both feed the same `ITE(cond, other,
    /// self)` (angr-op0dn.11.2.2). Takes the side tables by reference so it can
    /// serve either `self` or `other` without a borrow conflict.
    ///
    /// The result is always **8 bits wide** — the caller ITEs it against the
    /// other arm's byte, and `RustBV::ite_into` only `debug_assert`s the two
    /// widths agree, so returning a wider object here would build a
    /// malformed-width `Ite` in a release build (angr-0jh0j.30).
    #[allow(
        clippy::too_many_arguments,
        reason = "a pure per-byte extractor over \
        two arms' side tables; bundling the six by-value byte descriptors into a \
        struct would add a type whose only purpose is to be destructured one line later"
    )]
    fn merge_byte_value(
        symbolic_objects: &SymMap<Address, RustBV>,
        symbolic_spans: &SymMap<Address, (Address, u32)>,
        multi_objects: &FxHashMap<Address, MultiPayload>,
        addr: Address,
        concrete_byte: u8,
        is_sym: bool,
        is_multi: bool,
        endness: Endness,
    ) -> RustBV {
        if is_multi {
            silent_default!(
                cat_c,
                multi_objects.get(&addr).map(|p| p.collapse(concrete_byte)),
                RustBV::concrete(u128::from(concrete_byte), 8),
                "merge: byte {addr:?} is marked Multi by the page bitmap but multi_objects \
                 holds no payload for it; merging the concrete placeholder {concrete_byte:#04x} \
                 instead"
            )
        } else if is_sym {
            silent_default!(
                cat_c,
                Self::symbolic_byte_lane(symbolic_objects, symbolic_spans, addr, endness),
                RustBV::concrete(u128::from(concrete_byte), 8),
                "merge: byte {addr:?} is marked symbolic by the page bitmap but no \
                 symbolic_objects/symbolic_spans entry resolves it; merging the concrete \
                 placeholder {concrete_byte:#04x} instead"
            )
        } else {
            RustBV::concrete(u128::from(concrete_byte), 8)
        }
    }

    /// Resolve the 8-bit lane a page-bitmap-symbolic byte carries, mirroring
    /// `try_byte_merge_load`'s two-step lookup: `symbolic_objects[addr]` names
    /// an object *based* at this byte (take lane 0), otherwise the
    /// `symbolic_spans` reverse index names the wider object this byte lies
    /// inside (take the lane at its offset).
    ///
    /// Consulting only `symbolic_objects` — which is keyed at an object's base
    /// address alone — silently dropped every interior byte of a wide symbolic
    /// value onto the page's concrete placeholder, and handed the base byte the
    /// whole wide object instead of its lane (angr-0jh0j.30).
    ///
    /// Returns `None` when the sidecars disagree with the bitmap (missing
    /// entry, stale span width, out-of-range offset) so the caller can log and
    /// fall back.
    fn symbolic_byte_lane(
        symbolic_objects: &SymMap<Address, RustBV>,
        symbolic_spans: &SymMap<Address, (Address, u32)>,
        addr: Address,
        endness: Endness,
    ) -> Option<RustBV> {
        if let Some(sym) = symbolic_objects.get(&addr) {
            return Self::extract_byte_lane(sym, 0, endness);
        }
        let &(base_addr, base_width) = symbolic_spans.get(&addr)?;
        let sym = symbolic_objects.get(&base_addr)?;
        if sym.width() != base_width {
            return None;
        }
        let offset = u32::try_from(addr.raw().wrapping_sub(base_addr.raw())).ok()?;
        Self::extract_byte_lane(sym, offset, endness)
    }

    /// Guard a deferred symbolic store with a merge condition, composing with
    /// any pre-existing conditional-store condition via `And`
    /// (angr-op0dn.11.2.2). The returned write only materializes when `guard`
    /// holds, so an arm-specific pending write stays scoped to its own path
    /// after a merge.
    fn guard_pending_write(pw: &PendingWrite, guard: &RustBV) -> PendingWrite {
        let condition = Some(match &pw.condition {
            Some(c) => guard.and(c),
            None => guard.clone(),
        });
        PendingWrite {
            addr: pw.addr.clone(),
            value: pw.value.clone(),
            size: pw.size,
            condition,
        }
    }
}
