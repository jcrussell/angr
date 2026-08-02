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
    /// Checks pending_stores and all_flushed_stores first, since the call
    /// instruction pushes the return address via VEX stores before the
    /// interpreter detects the SimProcedure hook.
    pub(crate) fn get_return_addr(&self) -> Option<u64> {
        let ptr_size = self.calling_convention.pointer_size();
        let sp = self
            .registers
            .get(self.registers.arch().sp_offset(), ptr_size, self.ctx);
        let sp_val = sp.as_u64()?;

        // Check pending_stores first (most recent writes, same block)
        if let Some(data) = self
            .pending_stores
            .try_load_exact(sp_val, ptr_size as usize)
        {
            let mut bytes = [0u8; 8];
            let len = std::cmp::min(ptr_size as usize, 8);
            bytes[..len].copy_from_slice(&data[..len]);
            return Some(u64::from_le_bytes(bytes));
        }

        // Check all_flushed_stores (cross-block within same step)
        if let Some(data) = self.all_flushed_stores.get(&sp_val)
            && data.len() >= ptr_size as usize
        {
            let mut bytes = [0u8; 8];
            let len = std::cmp::min(ptr_size as usize, 8);
            bytes[..len].copy_from_slice(&data[..len]);
            return Some(u64::from_le_bytes(bytes));
        }

        // Fall back to rust_memory
        self.calling_convention.get_return_addr(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
        )
    }
}
