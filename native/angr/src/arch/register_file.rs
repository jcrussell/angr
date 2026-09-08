//! [`RegisterFile`] read / write / merge implementation.
//!
//! Sibling of `arch/mod.rs`, which keeps the [`RegisterFile`] struct itself,
//! its serde shadow [`RegisterFileData`], the [`Arch`] trait and the
//! per-offset lookup helpers. Extracted in angr-5mnx3.1: `get`/`put`/`merge`
//! carry the densest regression history in `arch/` and sat interleaved with
//! the unrelated architecture registry (now `registry.rs`).

use super::*;

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
    pub(super) fn mirror_offset(&self, offset: u32, size: u32) -> u32 {
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
    pub(super) fn compose_range(
        &self,
        offset: u32,
        size: u32,
        ctx: &crate::symbolic::SymContext,
    ) -> RustBV {
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

    /// Read a register value by *raw VEX* offset and size.
    ///
    /// Mirrors the offset into storage coordinates (see
    /// [`Self::mirror_offset`]) and delegates to [`Self::get_storage`].
    pub(crate) fn get(&self, offset: u32, size: u32, ctx: &crate::symbolic::SymContext) -> RustBV {
        self.get_storage(self.mirror_offset(offset, size), size, ctx)
    }

    /// [`Self::get`] in *storage* coordinates: reads the byte range
    /// `[offset, offset + size)` of this file's unconditionally little-endian
    /// storage without applying [`Self::mirror_offset`] first.
    ///
    /// A caller that already holds a storage key — i.e. a key of
    /// `self.symbolic`, which [`Self::put`] wrote *after* mirroring — must come
    /// through here rather than [`Self::get`]. [`Self::merge`] builds its
    /// offset set out of both files' `symbolic` keys and out of `data` indices,
    /// all of which are storage coordinates; re-mirroring one of those on a
    /// big-endian file lands on the *other* half of the containing register, so
    /// the merge composes and ITEs the wrong sub-field (angr-6cp06.74). The
    /// involution property does not rescue it: the doubly-mirrored value is
    /// used as a real lookup key and bit-range reference against actual storage
    /// entries, not merely recomputed and discarded.
    pub(super) fn get_storage(
        &self,
        offset: u32,
        size: u32,
        ctx: &crate::symbolic::SymContext,
    ) -> RustBV {
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
    pub(super) fn mirror_concrete_bytes(&mut self, offset: u32, value: &RustBV) {
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
    /// Every offset this function handles — the `symbolic` keys of both files
    /// and the `data` chunk indices — is a *storage* coordinate, so both files
    /// are read through [`Self::get_storage`], never [`Self::get`]. On a
    /// big-endian file [`Self::get`] would mirror an already-mirrored key a
    /// second time and compose the wrong half of the containing register
    /// (angr-6cp06.74).
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
                            let self_full = self.get_storage(offset, size, ctx);
                            let other_full = other.get_storage(offset, size, ctx);
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
                        let other_full = other.get_storage(offset, size, ctx);
                        updates.push((offset, merge_cond_other.ite(&other_full, sv, ctx)));
                    }
                }
                (None, Some(ov)) => {
                    // self has no overlay at this exact offset, other is
                    // symbolic — same reasoning as above, mirrored.
                    let size = ov.width() / 8;
                    if size > 0 && offset as usize + size as usize <= self.data.len() {
                        let self_full = self.get_storage(offset, size, ctx);
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
                        let self_full = self.get_storage(u32_off, size, ctx);
                        let other_full = other.get_storage(u32_off, size, ctx);
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

test_submod!("register_file_tests.rs" => tests);
