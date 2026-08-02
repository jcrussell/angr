//! SimProcedure registry plus argument / return-address extraction.
use super::*;

/// Information about a registered SimProcedure.
///
/// Deliberately carries no `no_return` flag (angr-9ke6b.218 item 2). Python's
/// registration tuple has one, but it is consumed one layer up — the run loop
/// reads it out of `StepContext::simprocedures` when it dispatches natively
/// (`run_loop_single`'s hook block). The interpreter only needs enough to
/// build `RunResult::SimProcedure`, which has no `no_return` member either:
/// terminal handling happens at the dispatch site, where the serial and core
/// paths deliberately disagree on the source of the flag (see
/// `dispatch_native_proc`'s doc comment). Re-adding it here and threading it
/// into `RunResult` would silently pick one of those two answers.
#[derive(Clone, Debug)]
pub(crate) struct SimProcedureInfo {
    /// Name of the SimProcedure (e.g., "strlen", "malloc").
    pub name: String,
    /// Number of arguments to extract.
    pub num_args: usize,
}

impl<'a> VEXInterpreter<'a> {
    /// Register a SimProcedure at an address.
    ///
    /// This allows the interpreter to pre-extract arguments when the hook is hit,
    /// reducing Python callback overhead.
    pub(crate) fn register_simprocedure(&mut self, addr: u64, name: String, num_args: usize) {
        Arc::make_mut(&mut self.hook_addrs).insert(addr);
        Arc::make_mut(&mut self.simprocedure_registry)
            .insert(addr, SimProcedureInfo { name, num_args });
    }

    /// Get the return address for a function call.
    ///
    /// Probes the in-flight store buffers before the memory fallback, since on
    /// a stack-return ABI the `call` instruction pushes the return address via
    /// VEX stores that are still buffered when the interpreter detects the
    /// SimProcedure hook.
    ///
    /// The probe mirrors `load_concrete_addr`'s precedence exactly — pending
    /// symbolic, pending concrete, flushed symbolic, flushed concrete — so a
    /// return address that landed in a *symbolic* buffer (a hook-written or
    /// otherwise unusual call sequence) is not shadowed by whatever stale
    /// concrete bytes `rust_memory` still holds at `[sp]` (angr-9ke6b.87).
    /// Symbolic stores push no placeholder bytes into the concrete buffers
    /// (angr-ofyh), so without these two checks such a store is invisible here.
    ///
    /// Only stack-return ABIs get the probe: on a link-register ABI
    /// (`pops_return_addr() == false`) `[sp]` holds no return address at all,
    /// so peeking there would hand back an unrelated caller local instead of
    /// deferring to the convention's link-register read.
    pub(crate) fn get_return_addr(&self) -> Option<u64> {
        let cc_fallback = || {
            self.calling_convention.get_return_addr(
                &self.registers,
                self.rust_memory.as_ref(),
                self.ctx,
            )
        };
        if !self.calling_convention.pops_return_addr() {
            return cc_fallback();
        }

        let ptr_size = self.calling_convention.pointer_size();
        let size = ptr_size as usize;
        let sp = self
            .registers
            .get(self.registers.arch().sp_offset(), ptr_size, self.ctx);
        let sp_val = sp.as_u64()?;

        // Buffered stores keep raw little-endian bytes; the low `min(size, 8)`
        // of them are the pointer.
        let from_le = |data: &[u8]| {
            let mut bytes = [0u8; 8];
            let len = std::cmp::min(size, 8);
            bytes[..len].copy_from_slice(&data[..len]);
            u64::from_le_bytes(bytes)
        };

        // Pending symbolic stores (most recent writes, same block).
        if let Some(sym_val) = self.symbolic_store_load(&self.pending_symbolic_stores, sp_val, size)
        {
            // SILENT(cat-a): a genuinely symbolic return address has no `u64`
            // answer, and `None` is exactly what the convention's own
            // `get_return_addr` yields for a symbolic `[sp]`. Falling through
            // to `rust_memory` instead would answer with the stale pre-call
            // bytes — the wrong-answer case this probe exists to avoid.
            return sym_val.as_u64();
        }

        // Pending concrete stores.
        if let Some(data) = self.pending_stores.try_load_exact(sp_val, size) {
            return Some(from_le(data));
        }

        // Flushed symbolic stores (cross-block within same step).
        if let Some(sym_val) =
            self.symbolic_store_load(&self.all_flushed_symbolic_stores, sp_val, size)
        {
            // SILENT(cat-a): see the pending-symbolic branch above.
            return sym_val.as_u64();
        }

        // Flushed concrete stores (cross-block within same step).
        if let Some(data) = self.all_flushed_stores.get(&sp_val)
            && data.len() >= size
        {
            return Some(from_le(data));
        }

        cc_fallback()
    }
}

#[cfg(test)]
#[path = "simprocedures_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod simprocedures_tests;
