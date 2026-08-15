//! Load operations for `SymbolicMemory`.
//!
//! Extracted from `memory/mod.rs` (angr-0lre). Holds the load_*/apply_pending_writes_*
//! family in a single file. Multiple `impl SymbolicMemory` blocks across files are
//! fine — Rust permits inherent impls to be split.
//!
//! # Variant matrix (angr-9ke6b.102)
//!
//! The `load_*` family is large and the `_lazy` / `_unified` suffixes do
//! **not** consistently signal behavior. (There used to be a
//! `load_concrete_automap` that notably did *not* auto-map — the name was
//! historical; angr-sqfj8.75 removed it as a byte-identical duplicate of
//! `load_concrete_lazy`.) Every row below is classified on four axes:
//!
//! * **Perm** — does the path run `check_perms_range(.., Permission::R)`?
//! * **Unmapped** — `Unmapped` hard error, or `UnmappedPageInRegion` for a page
//!   inside a registered lazy region (the interpreter's cue to fetch + retry)?
//! * **Auto-map** — does it materialize a missing page? **No load path does.**
//!   Speculative zero pages diverge from Python's backer data (see
//!   `SymbolicMemory::prepare_addresses_for_ite`).
//! * **Multi** — does it honor `Multi` cells? Every path that reaches
//!   `SymbolicMemory::load_concrete_common` does, via the `range_has_multi`
//!   dispatch it performs ahead of all other fast paths.
//!
//! | Entry point | Address | Perm | Unmapped | Auto-map | Multi |
//! |---|---|---|---|---|---|
//! | [`SymbolicMemory::load`] | symbolic (`ctx.eval` + pin fallback) | R | `Unmapped` | no | yes |
//! | [`SymbolicMemory::load_concrete`] | concrete | R | `Unmapped` | no | yes |
//! | [`SymbolicMemory::load_concrete_lazy`] | concrete | R | `UnmappedPageInRegion` | no | yes |
//! | `SymbolicMemory::load_concrete_lazy_inner` (`pub(super)`) | concrete | R | `UnmappedPageInRegion` | no | yes |
//! | [`SymbolicMemory::load_concrete_or_unconstrained`] | concrete | R, but **result discarded** | *swallowed* → fresh `unc_mem_*` BVS | no | yes |
//! | [`SymbolicMemory::load_symbolic`] | symbolic (concretizer) | R | `UnmappedPageInRegion` | no | yes |
//! | [`SymbolicMemory::load_symbolic_unified`] | symbolic (concretizer) | R (leaf-dependent) | leaf-dependent | no (filters instead) | yes |
//!
//! Notes that the axes alone don't carry:
//!
//! * Only `load_concrete` bumps the `mem_load` volume counter — per
//!   `invariant-mem-counter-two-paths` the `_lazy` paths must not, since they
//!   are reached from ITE-tree leaves and `store_concrete`'s read-modify-write.
//! * The `Unmapped` cells are the `lazy=false` contract of
//!   `SymbolicMemory::unmapped_page_error`, and it has one documented leak: a
//!   load that dispatches to `assemble_load_with_multi` reports a lazy-region
//!   miss as `UnmappedPageInRegion` regardless of the flag, so `load` /
//!   `load_concrete` can surface it too when a `Multi` cell is in range.
//! * `load_concrete_or_unconstrained` is infallible **by design** (ITE-leaf
//!   filler): it converts *every* error — including `MemoryError::Permission` —
//!   into an unconstrained value, so it is the one row where a permission
//!   check exists but cannot reject the load. Do not use it on a guest-visible
//!   load path.
//! * `load_symbolic` (eager) builds ITE leaves with the error-propagating
//!   `load_concrete_lazy`; `load_symbolic_unified` builds them with
//!   `load_concrete_or_unconstrained` after `prepare_addresses_for_ite` has
//!   dropped candidates whose page is unmapped. Hence the leaf-dependent cells.
//! * Both symbolic entry points short-circuit to
//!   `SymbolicMemory::unconstrained_read_value` under AVOID_MULTIVALUED_READS,
//!   before any page is touched (no perm check, no Multi consult).
//!
//! **Adding a variant:** place its body behind `load_concrete_common` rather
//! than re-deriving the fast paths — that is what keeps the Multi dispatch and
//! the angr-jvjf / angr-3zhl partial-overwrite guards on every path (the
//! angr-9ke6b.96 bug was exactly a Multi gate living on one copy only). Then
//! add a row here. The store-side matrix lives in the `memory::store` module
//! docs.
//!
//! **Panic policy (angr-9ke6b.212):** loads run on guest-supplied addresses, so
//! nothing here may panic on address shape. The concretization dispatch reports
//! an unresolvable address as `MemoryError::SymbolicAddress` (via
//! [`ConcretizationResult::as_symbolic_address_error`]), which the caller turns
//! into a Python-memory-model fallback.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`, so a new panic on an
//! untrusted address needs a reviewed, reasoned `#[allow]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::symbolic::{
    RustBV, SymContext, record_mem_lazy_page_fault, record_mem_load, record_mem_load_symbolic_addr,
};
use crate::vex::Endness;

use super::page::{PAGE_SIZE, Permission};
use super::{Address, MemoryError, SymbolicMemory, end_page_inclusive};

impl SymbolicMemory {
    /// angr-jvjf partial-overwrite guard (shared by `load_concrete` and
    /// `load_concrete_lazy_inner`). `store_concrete` clears the page bitmap
    /// for overwritten bytes but leaves `symbolic_objects` / `symbolic_spans`
    /// claims intact when those belong to a wider sym based at a different
    /// address — the wider-sym fast paths would silently return that stale
    /// claim. When a wider sym claim exists for `[addr, addr+size)` but not
    /// every byte is bitmap-symbolic, route through `assemble_load_with_multi`
    /// (which honors the page bitmap per byte). Returns `Some(result)` when
    /// the guard fires, `None` to let the caller continue its fast paths.
    pub(super) fn try_partial_overwrite_load(
        &self,
        addr: Address,
        size: u32,
        ctx: &SymContext,
    ) -> Option<Result<RustBV, MemoryError>> {
        // A range straddling the top of the space probes page 0, matching the
        // guest's own pointer wraparound and what the store side recorded.
        // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`).
        let has_wider_sym_claim = self.symbolic_objects.contains_key(&addr)
            || (0..size as u64).any(|i| self.symbolic_spans.contains_key(&(addr + i)));
        if has_wider_sym_claim && !self.bytes_all_marked_symbolic(addr, size) {
            let start_page = addr.page_num();
            let end_page = match end_page_inclusive(addr.raw(), size as u64) {
                Ok(p) => p,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = self.check_perms_range(start_page, end_page, Permission::R) {
                return Some(Err(e));
            }
            return Some(self.assemble_load_with_multi(addr, size, ctx));
        }
        None
    }

    /// Mint the unconstrained value returned by the AVOID_MULTIVALUED_READS
    /// short-circuit. Returns a zero BV when `zero_fill_unconstrained` is set
    /// (mirrors Python's `_default_value` honoring `ZERO_FILL_UNCONSTRAINED_MEMORY`),
    /// otherwise a fresh BVS named `symbolic_read_unconstrained_N` — same
    /// name stem Python uses in `AddressConcretizationMixin.load` so
    /// post-hoc state-export readers
    /// can identify the origin. The N suffix is sourced from a process-wide
    /// atomic to keep Z3 symbol names unique across SymbolicMemory instances
    /// without an `&mut self` borrow (this hook fires on a read-only path).
    pub(crate) fn unconstrained_read_value(&self, size: u32, ctx: &SymContext) -> RustBV {
        use std::sync::atomic::{AtomicU64, Ordering};
        static UNC_READ_ID: AtomicU64 = AtomicU64::new(0);
        if self.zero_fill_unconstrained {
            RustBV::concrete(0, size * 8)
        } else {
            let id = UNC_READ_ID.fetch_add(1, Ordering::Relaxed);
            RustBV::symbolic(ctx, format!("symbolic_read_unconstrained_{id}"), size * 8)
        }
    }

    /// Load bytes from memory as a RustBV.
    pub fn load(&self, addr: RustBV, size: u32, ctx: &SymContext) -> Result<RustBV, MemoryError> {
        // angr-9ke6b.229: the symbolic-addr subcounter bump used to live here,
        // where it was dead — this wrapper has no production caller (`.228`).
        // It now lives on `load_symbolic` / `load_symbolic_unified`, the two
        // concretizer entry points production actually uses.
        //
        // For symbolic addresses, we need to concretize or fork
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => {
                // Try to evaluate the address
                match ctx.eval(&addr) {
                    Some(a) => {
                        // Pin the arbitrarily-chosen load address on the path so
                        // a later eval() of inputs cannot yield a path-infeasible
                        // solution (angr-mv08h, parity with Python's load pin).
                        crate::concretize::pin_fallback_addr(ctx, &addr, a as u64);
                        a as u64
                    }
                    None => {
                        return Err(MemoryError::SymbolicAddress {
                            description: "could not resolve address".to_string(),
                        });
                    }
                }
            }
        };

        // angr-9ke6b.228: the lazy-page-fault bump lives on the producers
        // (`unmapped_page_error`, `assemble_load_with_multi`), not here — a
        // wrapper-level bump would double-count the Multi-cell path and miss
        // every production caller, which enters below this entry point.
        self.load_concrete(Address(concrete_addr), size, ctx)
    }

    /// Load from a concrete address.
    ///
    /// Thin wrapper over `SymbolicMemory::load_concrete_common` — the only
    /// difference from `SymbolicMemory::load_concrete_lazy_inner` is that an
    /// unmapped page is always a hard `Unmapped` here (no lazy-region fetch
    /// hint), plus the `record_mem_load` counter bump. Per
    /// `invariant-mem-counter-two-paths` the volume counter lives on this entry
    /// point only: the `_lazy` path is reached from ITE-tree leaves and
    /// `store_concrete`'s read-modify-write, which must not inflate it.
    pub fn load_concrete(
        &self,
        addr: impl Into<Address>,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let addr = addr.into();
        record_mem_load(size as u64);
        self.load_concrete_common(addr, size, ctx, false)
    }

    /// Shared body of [`SymbolicMemory::load_concrete`] and
    /// [`SymbolicMemory::load_concrete_lazy_inner`] (angr-9ke6b.97).
    ///
    /// These were ~200-line near-duplicates that drifted apart repeatedly —
    /// Multi-cell dispatch, the angr-3zhl partial-overlap merge, the wider-sym
    /// fast paths and the endianness-aware `containing_wider_sym` extract each
    /// landed in one copy only. `lazy` now carries the single real difference:
    /// whether an unmapped page inside a lazy region is reported as
    /// `UnmappedPageInRegion` (so the interpreter can fetch it) or as a plain
    /// `Unmapped` error.
    fn load_concrete_common(
        &self,
        addr: Address,
        size: u32,
        ctx: &SymContext,
        lazy: bool,
    ) -> Result<RustBV, MemoryError> {
        // angr-9ke6b.99: reject a zero-size load before any of the fast paths
        // below. `_pending_memory_load` forwards a Python-supplied size here
        // unchecked, and `size == 0` breaks two of them: the wider-symbolic
        // extract computes `size * 8 - 1` (u32 underflow, no overflow-checks in
        // release), and `end_page_inclusive` would have to compute
        // `addr + 0 - 1`. There is no valid 0-width BV to return either way.
        if size == 0 {
            return Err(MemoryError::ZeroSize { addr: addr.raw() });
        }
        // angr-9ke6b.96: Multi cells supersede plain Symbolic (memory
        // `invariant-multi-vs-symbolic-cell-states`), and the page
        // `symbolic_bitmap` is NOT set for a Multi byte — so none of the
        // symbolic fast paths below would fire and the load would fall
        // through to `bytes_to_bv` over the stale concrete placeholder
        // byte, silently returning the wrong value. Dispatch to the
        // per-byte assembler `assemble_load_with_multi` ahead of every
        // other path. Sharing this body between the eager and lazy entry
        // points (angr-9ke6b.97) is what keeps the gate on both — it was
        // originally added to `load_concrete_lazy_inner` only, leaving
        // `RustSimState::memory_load` (the `memory_load` pymethod used by
        // every SimProcedure) and `_pending_memory_load` silently wrong.
        if self.range_has_multi(addr, size) {
            let start_page = addr.page_num();
            let end_page = end_page_inclusive(addr.raw(), size as u64)?;
            self.check_perms_range(start_page, end_page, Permission::R)?;
            return self.assemble_load_with_multi(addr, size, ctx);
        }
        // angr-3zhl: a later partial store that begins inside [addr+1, addr+size)
        // overwrites trailing bytes of an earlier wider object at `addr`. The
        // exact-address and span fast paths below would return the stale wider
        // value, ignoring the overwrite. Detect by scanning for any
        // symbolic_objects entry whose start lies strictly inside our range,
        // and fall back to a per-byte merge (which uses both indices).
        // angr-kdyfx: both per-byte sidecar probes here (this scan and
        // `try_partial_overwrite_load` below) are provably no-ops when the
        // symbolic sidecars are empty — the common concrete-region case. An
        // empty `symbolic_objects` makes every `contains_key` false, so the
        // `.any` is false; an empty pair makes `has_wider_sym_claim` false, so
        // the partial-overwrite guard returns `None`. Gate both behind an
        // `is_empty()` fast-path, mirroring `load_concrete_lazy_inner`'s
        // `multi_objects.is_empty()` guard, to skip up to `size` empty-table
        // `contains_key` probes per concrete load.
        // Matching the guest's own pointer wraparound.
        // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`).
        let has_inner_overlap = !self.symbolic_objects.is_empty()
            && (1..size as u64).any(|i| self.symbolic_objects.contains_key(&(addr + i)));
        if has_inner_overlap {
            if let Some(merged) = self.try_byte_merge_load(addr, size, ctx) {
                return Ok(merged);
            }
            // Byte-merge fell through (e.g. unmapped/non-symbolic byte gap);
            // continue to the page-scan path below.
        } else {
            // angr-jvjf: partial-overwrite guard. `store_concrete` clears
            // the page bitmap for the overwritten bytes but leaves
            // `symbolic_objects` / `symbolic_spans` claims intact when
            // those claims belong to a wider sym based at a different
            // address. The wider-sym fast paths below would silently
            // return that stale claim. Detect by combining the claim
            // signal with a per-byte bitmap scan; on mismatch, route
            // through `assemble_load_with_multi` which uses the bitmap
            // per-byte (Symbolic / Spanned / Concrete).
            if (!self.symbolic_objects.is_empty() || !self.symbolic_spans.is_empty())
                && let Some(r) = self.try_partial_overwrite_load(addr, size, ctx)
            {
                return r;
            }
            // Check for stored symbolic object at exact address first
            if let Some(sym) = self.symbolic_objects.get(&addr) {
                if sym.width() == size * 8 {
                    return Ok(sym.clone());
                }
                // Partial read from a wider symbolic object
                // e.g., reading 1 byte from a 128-byte BVS
                if sym.width() > size * 8 {
                    let total_bits = sym.width();
                    // Endianness determines which bits correspond to byte 0.
                    // BE: byte 0 = MSB → hi = total_bits-1, lo = total_bits-size*8.
                    // LE: byte 0 = LSB → hi = size*8-1, lo = 0.
                    // The `sym.width() > size * 8` guard above bounds both
                    // `- size * 8` terms, and `size == 0` was rejected at the top
                    // of this fn, so `size * 8 - 1` cannot underflow either.
                    let (hi, lo) = match self.endness {
                        // overflow-ok: bounded by the `sym.width()` guard above.
                        Endness::Big => (total_bits - 1, total_bits - size * 8),
                        Endness::Little => (size * 8 - 1, 0),
                    };
                    return Ok(sym.extract(hi, lo, ctx));
                }
            }
            // Check if this address falls WITHIN a wider symbolic object
            // stored at a lower address (e.g., reading byte 5 of a 128-byte BVS)
            // Uses the symbolic_spans reverse index for O(1) lookup.
            if let Some(&(base_addr, _width_bits)) = self.symbolic_spans.get(&addr)
                && let Some(sym) = self.symbolic_objects.get(&base_addr)
            {
                let sym_bytes = sym.width() / 8;
                if addr.range_in(size as u64, base_addr, sym_bytes as u64) {
                    let total_bits = sym.width();
                    // `Sub<Address> for Address` is `wrapping_sub`, and the
                    // `range_in` guard above makes the distance a real in-region
                    // offset. overflow-ok: `off + size <= sym_bytes`.
                    let off_bits = (addr - base_addr) as u32 * 8;
                    // BE: bytes [off, off+size) of the wide BV occupy bits
                    //     [total-1-off_bits : total-off_bits-size*8].
                    // LE: same byte range occupies bits
                    //     [off_bits+size*8-1 : off_bits].
                    // The same `range_in` guard gives `off_bits + size * 8 <=
                    // total_bits`, with `size >= 1`.
                    let (hi, lo) = match self.endness {
                        // overflow-ok: bounded by the `range_in` guard above.
                        Endness::Big => {
                            (total_bits - off_bits - 1, total_bits - off_bits - size * 8)
                        }
                        // overflow-ok: bounded by the `range_in` guard above.
                        Endness::Little => (off_bits + size * 8 - 1, off_bits),
                    };
                    return Ok(sym.extract(hi, lo, ctx));
                }
            }
        }

        let start_page = addr.page_num();
        let end_page = end_page_inclusive(addr.raw(), size as u64)?;

        self.check_perms_range(start_page, end_page, Permission::R)?;

        let mut bytes;
        let mut has_symbolic = false;

        if start_page == end_page {
            // Fast path: entire load within a single page (common case)
            let page = match self.pages.get(&start_page) {
                Some(p) => p,
                None => {
                    return Err(self.unmapped_page_error(
                        start_page,
                        lazy,
                        MemoryError::Unmapped {
                            addr: start_page << 12,
                            size: PAGE_SIZE,
                        },
                    ));
                }
            };
            let offset = addr.page_offset();
            bytes = page.load_concrete(offset, size as u16);
            // Check symbolic markers
            for i in 0..size as u16 {
                // overflow-ok: this is the `start_page == end_page` arm, so the
                // whole load fits in one page — `offset + i < PAGE_SIZE`.
                if page.is_symbolic(offset + i) {
                    has_symbolic = true;
                    break;
                }
            }
        } else {
            // Slow path: load spans multiple pages
            bytes = Vec::with_capacity(size as usize);
            for i in 0..size {
                // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`),
                // matching the guest's own pointer wraparound.
                let byte_addr = addr + i as u64;
                let page_num = byte_addr.page_num();
                let offset = byte_addr.page_offset();
                if let Some(page) = self.pages.get(&page_num) {
                    if page.is_symbolic(offset) {
                        has_symbolic = true;
                    }
                    let byte = page.load_concrete(offset, 1);
                    bytes.push(byte.first().copied().unwrap_or(0));
                } else {
                    return Err(self.unmapped_page_error(
                        page_num,
                        lazy,
                        MemoryError::Unmapped {
                            addr: byte_addr.raw(),
                            size: 1,
                        },
                    ));
                }
            }
        }

        if has_symbolic {
            // Return stored symbolic object if available at exact address+width
            if let Some(sym) = self.symbolic_objects.get(&addr)
                && sym.width() == size * 8
            {
                return Ok(sym.clone());
            }
            // Try to reconstruct from per-byte symbolic objects
            // by concatenating individual byte-level entries
            let mut parts: Vec<RustBV> = Vec::new();
            let mut all_found = true;
            for i in 0..size {
                // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`),
                // matching the guest's own pointer wraparound.
                let byte_addr = addr + i as u64;
                if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                    if sym.width() == 8 {
                        parts.push(sym.clone());
                    } else if sym.width() > 8 {
                        // Extract the relevant byte
                        parts.push(sym.extract(7, 0, ctx));
                    } else {
                        all_found = false;
                        break;
                    }
                } else {
                    all_found = false;
                    break;
                }
            }
            if all_found && !parts.is_empty() {
                // Concatenate bytes: first byte is at lowest address.
                // LE: byte 0 = LSB → reverse so parts[N-1] is high.
                // BE: byte 0 = MSB → already high-to-low.
                // angr-kg58: balanced fold gives an O(log N) AST instead
                // of O(N) left-skewed chain.
                return Ok(concat_bytes_endian(parts, self.endness, ctx));
            }
            // angr-uwtj: spans-first containment lookup. O(1) via the
            // reverse-span index; falls back to a linear scan over
            // symbolic_objects when spans is stale/missing.
            if let Some((sym_addr, sym_val)) = self.containing_wider_sym(addr, size) {
                let total_bits = sym_val.width();
                // `Sub<Address> for Address` is `wrapping_sub`; every
                // `containing_wider_sym` return path is gated on `range_in`, so
                // overflow-ok: the distance is a real in-region byte offset with
                // `off_bits + size * 8 <= total_bits`.
                let off_bits = (addr - sym_addr) as u32 * 8;
                // Mirror of fast-path angr-v1q2 fix: the wide BV's byte
                // layout depends on memory endianness.
                // BE: bytes [off, off+size) occupy bits
                //     [total-1-off_bits : total-off_bits-size*8].
                // LE: same byte range occupies bits [off_bits+size*8-1 : off_bits].
                let (hi, lo) = match self.endness {
                    // overflow-ok: bounded by `containing_wider_sym`'s `range_in`.
                    Endness::Big => (total_bits - off_bits - 1, total_bits - off_bits - size * 8),
                    // overflow-ok: bounded by `containing_wider_sym`'s `range_in`.
                    Endness::Little => (off_bits + size * 8 - 1, off_bits),
                };
                return Ok(sym_val.extract(hi, lo, ctx));
            }
            // Cannot reconstruct - return error for Python fallback
            return Err(MemoryError::SymbolicAddress {
                description: "symbolic bytes not fully tracked".to_string(),
            });
        }

        // angr-24pv4.3: endianness-aware concrete byte packing, including the
        // >16-byte wide-load case that cannot fit a u128. See `bytes_to_bv`.
        Ok(bytes_to_bv(&bytes, size, self.endness, ctx))
    }

    /// Classify a load that hit an unmapped page (angr-9ke6b.97).
    ///
    /// The lazy entry points report a miss inside a registered lazy region as
    /// `UnmappedPageInRegion` so the interpreter can fetch the page from Python
    /// and retry; every other case (and every eager `load_concrete`) keeps the
    /// caller-supplied hard `Unmapped` error, whose shape differs between the
    /// single-page fast path (whole page) and the cross-page walk (one byte).
    ///
    /// Bumps `record_mem_lazy_page_fault` on the lazy classification
    /// (angr-9ke6b.228) — this is one of the counter's two load-side
    /// producers, the other being `assemble_load_with_multi`.
    fn unmapped_page_error(&self, page_num: u64, lazy: bool, fallback: MemoryError) -> MemoryError {
        if lazy && self.is_in_lazy_region(page_num) {
            record_mem_lazy_page_fault();
            MemoryError::UnmappedPageInRegion {
                page_addr: page_num << 12,
            }
        } else {
            fallback
        }
    }

    /// Load from a symbolic address with concretization support.
    ///
    /// This method handles symbolic addresses by:
    /// 1. Trying to concretize the address to a single value (fast path)
    /// 2. Building a balanced ITE tree for strided access patterns (efficient)
    /// 3. Building an ITE chain for multiple possible addresses
    /// 4. Returning an error if the address range is too large
    ///
    /// For unmapped pages in lazy regions, returns `UnmappedPageInRegion` so
    /// the interpreter can fetch the page on-demand.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer configuration
    ///
    /// # Returns
    /// The loaded value as a RustBV, or a MemoryError if loading fails.
    pub fn load_symbolic(
        &self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<RustBV, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.load_concrete_lazy(Address(concrete_addr), size, ctx);
        }
        // Past the fast path the address is genuinely symbolic (angr-9ke6b.229).
        record_mem_load_symbolic_addr();

        // AVOID_MULTIVALUED_READS: bypass concretization and return an
        // unconstrained value. Mirrors Python's `_default_value(...)` branch
        // in `address_concretization_mixin.AddressConcretizationMixin.load`.
        if concretizer.should_avoid_multivalued_read(&addr) {
            return Ok(self.unconstrained_read_value(size, ctx));
        }

        // Try to concretize the address (read mode: falls back to Any single solution)
        let base_value = match concretizer.concretize_read(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_lazy(Address(concrete_addr), size, ctx)?
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => self.load_strided_balanced(&addr, base, stride, count, size, ctx)?,
            ConcretizationResult::Multiple(addrs) => {
                self.build_balanced_ite_load(&addr, &addrs, size, ctx)?
            }
            other => {
                return Err(other.as_symbolic_address_error());
            }
        };

        // Apply any pending writes that might overlap this symbolic load
        Ok(self.apply_pending_writes_symbolic(&addr, size, base_value, ctx))
    }

    /// Load from a concrete address, returning an unconstrained symbolic value if unmapped.
    ///
    /// This is used as a fallback in ITE construction when an address cannot be mapped.
    /// Instead of failing, we return a fresh symbolic value representing unknown memory.
    ///
    /// # Arguments
    /// * `addr` - The address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    ///
    /// # Returns
    /// The loaded value, or a fresh unconstrained symbolic value if unmapped.
    pub fn load_concrete_or_unconstrained(
        &self,
        addr: impl Into<Address>,
        size: u32,
        ctx: &SymContext,
    ) -> RustBV {
        let addr = addr.into();
        match self.load_concrete_lazy(addr, size, ctx) {
            Ok(value) => value,
            Err(_) => {
                if self.zero_fill_unconstrained {
                    RustBV::concrete(0, size * 8)
                } else {
                    // angr-03vl4.41: the name must be unique *process-wide*, not
                    // just within one ITE-tree build. `RustBV::from_parts` lowers
                    // the name straight to `z3::ast::BV::new_const`, which interns
                    // by name and ignores the Rust-side `.id`, so two unconstrained
                    // reads that stringify the same would literally alias in the
                    // solver. `SymContext::new_bv` suffixes the globally-monotonic
                    // `next_id()`, which no per-call counter can collide with.
                    ctx.new_bv(&format!("unc_mem_{:x}", addr.raw()), size * 8)
                }
            }
        }
    }

    /// Unified symbolic load that handles all concretization results in Rust.
    ///
    /// This method replaces the Python fallback for symbolic memory loads.
    /// It handles all cases:
    /// - Single address: direct load
    /// - Multiple addresses: build balanced ITE tree with auto-mapping
    /// - Strided access: build balanced ITE tree
    /// - Too large range: return unconstrained symbolic value
    /// - Failed concretization: return error
    ///
    /// The key improvement is that unmapped pages in lazy regions are auto-mapped
    /// before ITE construction, and truly unmapped addresses use unconstrained
    /// symbolic values instead of failing.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to load from
    /// * `size` - Number of bytes to load
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer
    ///
    /// # Returns
    /// The loaded value as a RustBV, or an error if loading fails.
    pub fn load_symbolic_unified(
        &mut self,
        addr: RustBV,
        size: u32,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<RustBV, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.load_concrete_lazy(Address(concrete_addr), size, ctx);
        }
        // Past the fast path the address is genuinely symbolic (angr-9ke6b.229).
        record_mem_load_symbolic_addr();

        // AVOID_MULTIVALUED_READS: bypass concretization and return an
        // unconstrained value.
        if concretizer.should_avoid_multivalued_read(&addr) {
            return Ok(self.unconstrained_read_value(size, ctx));
        }

        // Try to concretize the address (read mode: falls back to Any single solution)
        let base_value = match concretizer.concretize_read(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.load_concrete_lazy(Address(concrete_addr), size, ctx)?
            }
            ConcretizationResult::Multiple(addrs) => {
                let ready_addrs = self.prepare_addresses_for_ite(&addrs, size);

                if ready_addrs.is_empty() {
                    return Ok(RustBV::symbolic(
                        ctx,
                        format!("mem_all_unmapped_{size}"),
                        size * 8,
                    ));
                }

                let addr_clone = addr.clone();
                let value =
                    self.build_balanced_ite_load_after_prep(&addr_clone, &ready_addrs, size, ctx)?;
                // angr-62li: hoist the addr-domain disjunction over the
                // post-prep set (addresses that actually had data; matches
                // the set Z3 ITE-loads against, and is the soundest
                // restriction we can communicate without re-running the
                // concretizer over unmapped pages).
                Self::assert_address_disjunction(&addr, &ready_addrs, ctx);
                value
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                self.prepare_strided_region(base, stride, count, size);
                self.load_strided_balanced(&addr, base, stride, count, size, ctx)?
            }
            // Return error so caller can fall back to Python's memory model,
            // which handles large symbolic address ranges natively.
            other => {
                return Err(other.as_symbolic_address_error());
            }
        };

        // Apply any pending writes that might overlap this symbolic load
        Ok(self.apply_pending_writes_symbolic(&addr, size, base_value, ctx))
    }

    /// Apply pending writes that might overlap a concrete load address.
    /// Returns the value with ITE chains for any matching pending writes.
    pub(super) fn apply_pending_writes_concrete(
        &self,
        _addr: Address,
        _size: u32,
        base_value: RustBV,
        _ctx: &SymContext,
    ) -> RustBV {
        // Pending writes overlay is disabled during execution.
        // Stores go through the eager concretize+ITE path.
        // Pending writes are only used for deferred flushing on export.
        base_value
    }

    /// Apply pending writes that might overlap a symbolic load address.
    /// Returns the value with ITE chains for any matching pending writes.
    pub(super) fn apply_pending_writes_symbolic(
        &self,
        _addr: &RustBV,
        _size: u32,
        base_value: RustBV,
        _ctx: &SymContext,
    ) -> RustBV {
        // Pending writes overlay is disabled during execution.
        // See apply_pending_writes_concrete for rationale.
        base_value
    }

    /// Load from a concrete address, returning UnmappedPageInRegion for lazy regions.
    ///
    /// This is similar to load_concrete but distinguishes between:
    /// - Unmapped page in a lazy region (can be fetched)
    /// - Totally unmapped memory (error)
    pub fn load_concrete_lazy(
        &self,
        addr: impl Into<Address>,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let addr = addr.into();
        let base = self.load_concrete_lazy_inner(addr, size, ctx)?;
        Ok(self.apply_pending_writes_concrete(addr, size, base, ctx))
    }

    /// Internal implementation of `load_concrete_lazy`.
    ///
    /// Identical to [`SymbolicMemory::load_concrete`] except that an unmapped
    /// page inside a registered lazy region surfaces as `UnmappedPageInRegion`
    /// (so the interpreter can fetch it) rather than a hard `Unmapped`, and
    /// that it does not bump the `mem_load` volume counter. See
    /// [`SymbolicMemory::load_concrete_common`].
    pub(super) fn load_concrete_lazy_inner(
        &self,
        addr: Address,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        self.load_concrete_common(addr, size, ctx, true)
    }

    /// True when any byte in `[addr, addr + size)` carries a Multi cell.
    ///
    /// Shared gate for `assemble_load_with_multi` dispatch: every load path
    /// must consult this *before* the `symbolic_objects` / `symbolic_spans`
    /// fast paths, because `install_multi_for_candidates` never sets the
    /// page's symbolic bit (see `invariant-multi-vs-symbolic-cell-states`).
    /// The `is_empty()` guard keeps the common all-concrete load free of up
    /// to `size` empty-map probes.
    pub(super) fn range_has_multi(&self, addr: Address, size: u32) -> bool {
        // Matching the guest's own pointer wraparound.
        // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`).
        !self.multi_objects.is_empty()
            && (0..size as u64).any(|i| self.multi_objects.contains_key(&(addr + i)))
    }

    /// Per-byte reconstruction for loads that touch at least one Multi cell
    /// (Phase 1.2 of angr-czph, bead angr-n082). For each byte in
    /// `[addr, addr + size)`:
    ///   * Multi byte: right-fold the payload's `(cond, value)` pairs into
    ///     an ITE chain whose final `else` is the page's concrete byte.
    ///     Calls `crate::symbolic::record_mem_ite_depth(payload.len())`
    ///     per memory `invariant-mem-ite-depth-counter`.
    ///   * Plain Symbolic byte: extract from `symbolic_objects` /
    ///     `symbolic_spans` mirroring `try_byte_merge_load`.
    ///   * Concrete byte: read the page byte into an 8-bit `RustBV`.
    ///
    /// The per-byte parts are concatenated endianness-correctly to match
    /// `try_byte_merge_load`. The caller is responsible for permission
    /// checks; this helper only handles unmapped pages by returning the
    /// usual `MemoryError`.
    pub(super) fn assemble_load_with_multi(
        &self,
        addr: Address,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        // Phase 4.1 (angr-mmdh.1): wider-load collapse cache. When a load
        // spans the same Multi/Concrete byte mix as a previous load and
        // none of those bytes have changed (per-byte versions for Multi,
        // concrete byte values for Concrete), reuse the assembled BV
        // directly — skipping the N page lookups + N collapse + N concat
        // sequence that dominates gate-on cost for sym-write. Loads
        // touching plain Symbolic bytes are not cached.
        //
        // The cache is bypassed for size=1 because there is no concat to
        // skip — the cost is exactly one `MultiPayload::collapse` call,
        // which Phase 3 already memoizes.
        let fingerprint = if size > 1 {
            self.compute_wider_load_fingerprint(addr, size)
        } else {
            None
        };
        if let Some(fp) = &fingerprint
            && let Some(cached) = self.wider_load_cache.borrow().get(&(addr, size))
            && cached.byte_fingerprints == *fp
        {
            // invariant-mem-ite-depth-counter: replay the same
            // count the per-byte miss path would have recorded.
            if cached.total_ite_depth > 0 {
                crate::symbolic::record_mem_ite_depth(cached.total_ite_depth);
            }
            return Ok(cached.bv.clone());
        }

        let mut total_ite_depth: u32 = 0;
        let mut byte_parts: Vec<RustBV> = Vec::with_capacity(size as usize);
        for i in 0..size {
            // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`),
            // matching the guest's own pointer wraparound.
            let byte_addr = addr + i as u64;
            let page_num = byte_addr.page_num();
            let offset = byte_addr.page_offset();
            let page = match self.pages.get(&page_num) {
                Some(p) => p,
                None => {
                    if self.is_in_lazy_region(page_num) {
                        // Second load-side producer of the counter
                        // (angr-9ke6b.228); see `unmapped_page_error`.
                        record_mem_lazy_page_fault();
                        return Err(MemoryError::UnmappedPageInRegion {
                            page_addr: page_num << 12,
                        });
                    } else {
                        return Err(MemoryError::Unmapped {
                            addr: byte_addr.raw(),
                            size: 1,
                        });
                    }
                }
            };

            let part = if let Some(payload) = self.multi_objects.get(&byte_addr) {
                // Phase 3 (angr-j0n4): MultiPayload::collapse memoizes the
                // right-folded ITE keyed on the page's concrete default byte.
                // Cache hits skip the alt-by-alt Z3 ITE build, eliminating
                // the per-load cost that gated Phase 2.
                let concrete_byte = page.load_concrete(offset, 1).first().copied().unwrap_or(0);
                let depth = payload.len() as u32;
                total_ite_depth = total_ite_depth.saturating_add(depth);
                crate::symbolic::record_mem_ite_depth(depth);
                payload.collapse(concrete_byte, ctx)
            } else if page.is_symbolic(offset) {
                if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                    Self::extract_byte_lane(sym, 0, self.endness, ctx).ok_or_else(|| {
                        MemoryError::SymbolicAddress {
                            description: "symbolic byte lane out of range".to_string(),
                        }
                    })?
                } else if let Some(&(base_addr, base_width)) = self.symbolic_spans.get(&byte_addr) {
                    let sym = self.symbolic_objects.get(&base_addr).ok_or(
                        MemoryError::SymbolicAddress {
                            description: "stale symbolic span".to_string(),
                        },
                    )?;
                    if sym.width() != base_width {
                        return Err(MemoryError::SymbolicAddress {
                            description: "symbolic span width mismatch".to_string(),
                        });
                    }
                    // `extract_byte_lane` refuses any offset (including a wrapped
                    // one from a stale span) that overruns `sym`, so:
                    // overflow-ok: `Sub<Address> for Address` is `wrapping_sub`.
                    let off_in_sym = (byte_addr - base_addr) as u32;
                    Self::extract_byte_lane(sym, off_in_sym, self.endness, ctx).ok_or_else(
                        || MemoryError::SymbolicAddress {
                            description: "symbolic span byte offset out of range".to_string(),
                        },
                    )?
                } else {
                    return Err(MemoryError::SymbolicAddress {
                        description: "symbolic byte without tracked object".to_string(),
                    });
                }
            } else {
                let concrete_byte = page.load_concrete(offset, 1);
                RustBV::concrete(concrete_byte.first().copied().unwrap_or(0) as u128, 8)
            };
            byte_parts.push(part);
        }

        // Concatenate per endianness with a balanced tree (angr-kg58):
        //   LE: byte[0] is the LSB → reverse so high byte is first.
        //   BE: byte[0] is the MSB → already high-to-low.
        let result = concat_bytes_endian(byte_parts, self.endness, ctx);

        // Phase 4.1: cache the assembled BV when the load is fully covered
        // by Multi + Concrete bytes (fingerprint is Some). The hit path
        // above already returned for size==1, so insertion here is only
        // for size>1.
        if let Some(fp) = fingerprint {
            self.insert_wider_load_cache(
                (addr, size),
                crate::memory::CachedWiderLoad {
                    byte_fingerprints: fp,
                    total_ite_depth,
                    bv: result.clone(),
                },
            );
        }

        Ok(result)
    }

    /// Reconstruct a load by per-byte lookup against `symbolic_objects` and
    /// `symbolic_spans`, producing the correct interleaving when an earlier
    /// wider symbolic value was partially overwritten by a later store
    /// (angr-3zhl).
    ///
    /// For each byte in `[addr, addr + size)`:
    /// - if `symbolic_objects[byte_addr]` is set, this byte is the start
    ///   of a stored symbolic value — its byte-0 lane is used;
    /// - otherwise `symbolic_spans[byte_addr]` is consulted, and the
    ///   referenced wider value's byte at the matching offset is used;
    /// - any other case (concrete bytes, gaps, stale-span mismatch)
    ///   short-circuits to `None` so the caller can fall through to the
    ///   page-scan path.
    ///
    /// Returns the byte-merged bitvector, or `None` if a fully-symbolic
    /// reconstruction was not possible.
    /// angr-uwtj: find a wider symbolic object that fully contains
    /// `[addr, addr+size)`. Consults the O(1) `symbolic_spans` reverse
    /// index first; falls back to a linear scan over `symbolic_objects`
    /// when spans is stale or missing.
    pub(super) fn containing_wider_sym(
        &self,
        addr: Address,
        size: u32,
    ) -> Option<(Address, &RustBV)> {
        // O(1) path 1: addr lies strictly inside a wider sym (spans
        // populates offsets 1..sym_bytes; offset 0 is handled below).
        if let Some(&(base_addr, _base_width)) = self.symbolic_spans.get(&addr)
            && let Some(sym) = self.symbolic_objects.get(&base_addr)
        {
            let sym_size = sym.width() / 8;
            if addr.range_in(size as u64, base_addr, sym_size as u64) {
                return Some((base_addr, sym));
            }
        }
        // O(1) path 2: addr IS the base of a wider sym. Caller's earlier
        // exact-width branch already returns on `sym.width() == size*8`;
        // this catches partial reads where the sym is wider.
        if let Some(sym) = self.symbolic_objects.get(&addr) {
            let sym_size = sym.width() / 8;
            if sym.width() > size * 8 && size as u64 <= sym_size as u64 {
                return Some((addr, sym));
            }
        }
        // Fallback linear scan — preserves correctness when spans is
        // stale (e.g. test setups that bypass `import_symbolic_value`).
        for (&sym_addr, sym_val) in &self.symbolic_objects {
            let sym_size = sym_val.width() / 8;
            if addr.range_in(size as u64, sym_addr, sym_size as u64) {
                return Some((sym_addr, sym_val));
            }
        }
        None
    }

    fn try_byte_merge_load(&self, addr: Address, size: u32, ctx: &SymContext) -> Option<RustBV> {
        let mut byte_parts: Vec<RustBV> = Vec::with_capacity(size as usize);
        for i in 0..size {
            // overflow-ok: `Address + u64` is `wrapping_add` (see `address.rs`),
            // matching the guest's own pointer wraparound.
            let byte_addr = addr + i as u64;
            let part = if let Some(sym) = self.symbolic_objects.get(&byte_addr) {
                Self::extract_byte_lane(sym, 0, self.endness, ctx)?
            } else if let Some(&(base_addr, base_width)) = self.symbolic_spans.get(&byte_addr) {
                let sym = self.symbolic_objects.get(&base_addr)?;
                if sym.width() != base_width {
                    return None;
                }
                // `extract_byte_lane` refuses any offset (including a wrapped one
                // from a stale span) that overruns `sym`, so:
                // overflow-ok: `Sub<Address> for Address` is `wrapping_sub`.
                let offset = (byte_addr - base_addr) as u32;
                Self::extract_byte_lane(sym, offset, self.endness, ctx)?
            } else {
                return None;
            };
            byte_parts.push(part);
        }
        // Concatenate per endianness with a balanced tree (angr-kg58):
        //   LE: byte[0] is the LSB → reverse so high byte is first.
        //   BE: byte[0] is the MSB → already high-to-low.
        Some(concat_bytes_endian(byte_parts, self.endness, ctx))
    }

    /// Extract the byte at `byte_offset` from a wider symbolic value,
    /// honouring memory endianness:
    ///   LE: byte 0 is the LSB → bits `[off*8+7 : off*8]`.
    ///   BE: byte 0 is the MSB → bits `[total-off*8-1 : total-off*8-8]`.
    /// Returns `None` if the offset is out of range.
    pub(super) fn extract_byte_lane(
        sym: &RustBV,
        byte_offset: u32,
        endness: Endness,
        ctx: &SymContext,
    ) -> Option<RustBV> {
        let total_bits = sym.width();
        // angr-xloth.4: `byte_offset` reaches here as a *wrapping* `Address`
        // difference (`assemble_load_with_multi` / `try_byte_merge_load` both
        // form `(byte_addr - base_addr) as u32`), so a stale `symbolic_spans`
        // entry whose base sits above the byte yields a near-`u32::MAX` offset.
        // Spelled `(byte_offset + 1) * 8` that wrapped to a small value with
        // release overflow-checks off, passing the bounds check below and
        // handing the BE arm an underflowed `total_bits - byte_offset * 8 - 1`.
        // Refuse: an offset with no representable bit position is out of range.
        let end_bit = byte_offset.checked_add(1)?.checked_mul(8)?;
        if end_bit > total_bits {
            return None;
        }
        Some(match endness {
            Endness::Little => sym.extract(byte_offset * 8 + 7, byte_offset * 8, ctx),
            // overflow-ok: `end_bit <= total_bits` above bounds both terms.
            Endness::Big => sym.extract(
                total_bits - byte_offset * 8 - 1,
                // overflow-ok: `end_bit <= total_bits` above bounds this too.
                total_bits - byte_offset * 8 - 8,
                ctx,
            ),
        })
    }
}

/// Concatenate per-byte `parts` (with `parts[0]` at the lowest address)
/// into a single RustBV honoring `endness`, using a balanced O(log N)
/// fold (angr-kg58). LE: byte 0 is the LSB, so reverse to high-bits-first
/// before folding; BE: byte 0 is the MSB, already high-to-low. `parts`
/// must be non-empty (`concat_balanced` asserts this).
pub(super) fn concat_bytes_endian(
    mut parts: Vec<RustBV>,
    endness: Endness,
    ctx: &SymContext,
) -> RustBV {
    if matches!(endness, Endness::Little) {
        parts.reverse();
    }
    RustBV::concat_balanced(&parts, ctx)
}

/// Pack concrete `bytes` (with `bytes[0]` at the lowest address) into a
/// RustBV of `size` bytes honoring `endness`.
///
/// A `Concrete` RustBV stores its value in a u128 (16 bytes). A wider
/// concrete load cannot be packed into one — a u128 shift would wrap
/// (mod 128) and OR high bytes back over the low bytes, silently
/// corrupting the value (e.g. a 32-byte AVX load). Those are assembled as
/// a balanced `Concat` of per-byte concretes instead; `<=16` bytes fold
/// into a u128 directly.
pub(super) fn bytes_to_bv(bytes: &[u8], size: u32, endness: Endness, ctx: &SymContext) -> RustBV {
    if size as usize > 16 {
        let parts: Vec<RustBV> = bytes
            .iter()
            .map(|&b| RustBV::concrete(b as u128, 8))
            .collect();
        return concat_bytes_endian(parts, endness, ctx);
    }
    let value = match endness {
        Endness::Little => {
            let mut v: u128 = 0;
            for (i, &byte) in bytes.iter().enumerate() {
                v |= (byte as u128) << (i * 8);
            }
            v
        }
        Endness::Big => {
            let mut v: u128 = 0;
            for &byte in bytes {
                v = (v << 8) | (byte as u128);
            }
            v
        }
    };
    RustBV::concrete(value, size * 8)
}
