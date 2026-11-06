//! The stepping state as a `state.inspect` BP action sees it mid-step.
//!
//! While a block runs, the state's own register file and memory are stale
//! (the interpreter commits them when the block ends), and the run loop owns
//! the state, so a host cannot answer a BP action's `state.regs` /
//! `state.memory` read from the state. [`InspectView`] is an owned snapshot of
//! the interpreter's live copies instead, handed to the host around each
//! inspect fire while [`EngineHost::inspect_wants_view`] holds
//! (angr-mx00p.132.53). Owned, not borrowed: the host keeps it for the length
//! of a Python call, which a borrow of the interpreter cannot outlive safely.

use super::*;

/// An owned snapshot of the stepping state's live values at an inspect fire.
///
/// Every field is cheap to take: [`RegisterFile`], [`SymbolicMemory`] and the
/// [`SymContext`] fork copy-on-write, and the store buffers hold only the writes Rust memory
/// refused (empty on the common path).
pub struct InspectView {
    block_addr: u64,
    insn_addr: u64,
    registers: RegisterFile,
    memory: Option<SymbolicMemory>,
    pending_stores: PendingStoreBuffer,
    pending_symbolic_stores: FxHashMap<u64, RustBV>,
    flushed_stores: FxHashMap<u64, Vec<u8>>,
    flushed_symbolic_stores: FxHashMap<u64, RustBV>,
    solver: SymContext,
}

impl InspectView {
    /// Address of the block being executed (`SimState.addr` during a
    /// `HeavyVEXMixin` breakpoint).
    pub fn block_addr(&self) -> u64 {
        self.block_addr
    }

    /// Address of the instruction being executed (the last `IMark`).
    pub fn insn_addr(&self) -> u64 {
        self.insn_addr
    }

    /// The register at `offset`, `size` bytes wide, as the block has left it.
    pub fn register(&self, offset: u32, size: u32) -> RustBV {
        self.registers.get(offset, size)
    }

    /// The `size`-byte value at `addr`, in the order `VEXInterpreter::load_layered`
    /// reads it: Rust memory, then the pending and flushed store buffers.
    ///
    /// `None` when none of them serves it whole — an unmapped page, a page
    /// the host serves, or a store buffer entry that only partly covers the
    /// range. The caller must fail loudly then rather than read the state's
    /// stale memory.
    pub fn load(&self, addr: u64, size: u32) -> Option<RustBV> {
        if let Some(memory) = &self.memory
            && let Ok(value) = memory.load_concrete(addr, size)
        {
            return Some(value);
        }
        let bytes = size as usize;
        let bits = size * 8;
        if let Some(value) = symbolic_store_exact(&self.pending_symbolic_stores, addr, bits) {
            return Some(value);
        }
        if let Some(data) = self.pending_stores.try_load_assembled(addr, bytes) {
            return Some(bytes_to_bv(&data, bits));
        }
        if let Some(value) = symbolic_store_exact(&self.flushed_symbolic_stores, addr, bits) {
            return Some(value);
        }
        self.flushed_stores
            .get(&addr)
            .filter(|data| bytes <= data.len())
            .map(|data| bytes_to_bv(&data[..bytes], bits))
    }
}

impl InspectView {
    /// The stepping state's solver, for a host that queries it or builds a
    /// value against it.
    pub fn solver(&self) -> &SymContext {
        &self.solver
    }

    /// The concrete bytes from `addr` on, at most `max`, stopping at the
    /// first byte [`Self::load`] cannot serve concretely: the code a
    /// `state.block()` lift reads on a self-modifying-code project.
    pub fn concrete_bytes(&self, addr: u64, max: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(max);
        for i in 0..max {
            // overflow-ok: a code read near the top of the address space wraps, as the lifter's fetch does
            let Some(byte) = self
                .load(addr.wrapping_add(i as u64), 1)
                .and_then(|v| v.as_u64())
            else {
                break;
            };
            out.push(byte as u8);
        }
        out
    }
}

/// A symbolic store recorded at exactly `addr`, cut to its low `bits` as
/// `VEXInterpreter::symbolic_store_load` does. An overlapping store that does
/// not start at `addr` is not served.
fn symbolic_store_exact(map: &FxHashMap<u64, RustBV>, addr: u64, bits: u32) -> Option<RustBV> {
    let value = map.get(&addr)?;
    match value.width().cmp(&bits) {
        std::cmp::Ordering::Equal => Some(value.clone()),
        std::cmp::Ordering::Greater => Some(value.extract(bits - 1, 0)),
        std::cmp::Ordering::Less => None,
    }
}

impl<'a> VEXInterpreter<'a> {
    /// Snapshot the live state for an inspect fire. See [`InspectView`].
    pub(super) fn inspect_view(&self) -> InspectView {
        InspectView {
            block_addr: self.pc,
            insn_addr: self.current_insn_addr,
            registers: self.registers.clone(),
            memory: self
                .rust_memory
                .as_ref()
                .filter(|_| self.use_rust_memory)
                .map(SymbolicMemory::fork),
            pending_stores: self.pending_stores.clone(),
            pending_symbolic_stores: self.pending_symbolic_stores.clone(),
            flushed_stores: self.all_flushed_stores.clone(),
            flushed_symbolic_stores: self.all_flushed_symbolic_stores.clone(),
            solver: self.ctx.fork(),
        }
    }

    /// Run `fire` (one `callbacks.inspect_*` call) with the live state
    /// published to the host, when the host asked for it.
    pub(super) fn fire_inspect<R>(
        &self,
        callbacks: &dyn EngineHost,
        fire: impl FnOnce() -> R,
    ) -> R {
        if !callbacks.inspect_wants_view() {
            return fire();
        }
        callbacks.publish_inspect_view(Some(self.inspect_view()));
        let result = fire();
        callbacks.publish_inspect_view(None);
        result
    }
}
