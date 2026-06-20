//! SimProcedure registry plus argument / return-address extraction.
use super::*;

/// Information about a registered SimProcedure.
#[derive(Clone, Debug)]
pub struct SimProcedureInfo {
    /// Name of the SimProcedure (e.g., "strlen", "malloc").
    pub name: String,
    /// Number of arguments to extract.
    pub num_args: usize,
    /// Whether this is a no-return procedure (e.g., "exit", "abort").
    pub no_return: bool,
}

impl<'a> VEXInterpreter<'a> {
    /// Register a SimProcedure at an address.
    ///
    /// This allows the interpreter to pre-extract arguments when the hook is hit,
    /// reducing Python callback overhead.
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        Arc::make_mut(&mut self.hook_addrs).insert(addr);
        Arc::make_mut(&mut self.simprocedure_registry).insert(
            addr,
            SimProcedureInfo {
                name,
                num_args,
                no_return,
            },
        );
    }

    /// Register multiple SimProcedures at once.
    ///
    /// Each tuple is (address, name, num_args, no_return).
    pub fn register_simprocedures(&mut self, procs: &[(u64, String, usize, bool)]) {
        let hooks = Arc::make_mut(&mut self.hook_addrs);
        for (addr, _, _, _) in procs {
            hooks.insert(*addr);
        }
        let registry = Arc::make_mut(&mut self.simprocedure_registry);
        for (addr, name, num_args, no_return) in procs {
            registry.insert(
                *addr,
                SimProcedureInfo {
                    name: name.clone(),
                    num_args: *num_args,
                    no_return: *no_return,
                },
            );
        }
    }

    /// Get SimProcedure info for an address, if registered.
    pub fn get_simprocedure_info(&self, addr: u64) -> Option<&SimProcedureInfo> {
        self.simprocedure_registry.get(&addr)
    }

    /// Clear all SimProcedure registrations.
    pub fn clear_simprocedures(&mut self) {
        Arc::make_mut(&mut self.simprocedure_registry).clear();
    }

    /// Extract arguments for a SimProcedure call.
    ///
    /// Uses the calling convention to extract arguments from registers and
    /// stack. Returns the structured `ExtractionError` from the trait so
    /// callers see *why* extraction failed (stack unmapped vs symbolic SP
    /// vs missing memory view) instead of receiving silently-fabricated
    /// placeholders.
    pub fn extract_simprocedure_args(
        &self,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::calling_conventions::ExtractionError> {
        self.calling_convention.extract_args(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
            num_args,
        )
    }

    /// Get the return address for a function call.
    ///
    /// Checks pending_stores and all_flushed_stores first, since the call
    /// instruction pushes the return address via VEX stores before the
    /// interpreter detects the SimProcedure hook.
    pub fn get_return_addr(&self) -> Option<u64> {
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
