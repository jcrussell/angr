//! Register-access family for `RustSimState`.
//!
//! Name- and offset-keyed register getters/setters, the IP/SP accessors, the
//! bulk raw-bytes bridge used for sync to/from Python, and the `RegisterFile`
//! reference/replace accessors. Split out of `mod.rs` per the god-object
//! decomposition (angr-0mqkc.5); mirrors the `construction.rs` / `fork.rs` /
//! `snapshot.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Get a register by name.
    pub fn get_register(&self, name: &str) -> Option<RustBV> {
        let ctx = self.solver.borrow();
        self.registers.get_reg(name, &ctx)
    }

    /// Set a register by name.
    pub fn set_register(&mut self, name: &str, value: RustBV) -> bool {
        // angr-4rq7: writing the IP register (e.g. via the RustRegisterProxy
        // write-through gate, which routes `state.regs.ip = target` to
        // set_state_register_symbolic_ast -> set_register("rip", ...)) must
        // keep self.pc in sync, exactly as set_ip/set_pc do. Without this,
        // the register file holds the real address but self.pc stays stale
        // (often 0 on a freshly-forked state): get_state_pc_by_id() and the
        // next block fetch then read 0x0 ("Lift error at 0x0") while
        // get_register("rip") reads correctly — the divergence behind the
        // register-proxy gate corruption.
        let is_ip = self
            .arch
            .register_offset(name)
            .map(|off| off == self.arch.ip_offset())
            .unwrap_or(false);
        let ok = self.registers.put_reg(name, value);
        if ok
            && is_ip
            && let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64()
        {
            self.pc = v;
        }
        ok
    }

    /// Get a register by offset.
    pub fn get_register_by_offset(&self, offset: u32, size: u32) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get(offset, size, &ctx)
    }

    /// Set a register by offset.
    pub fn set_register_by_offset(&mut self, offset: u32, value: RustBV) {
        self.registers.put(offset, value);
    }

    /// Get the instruction pointer register.
    pub fn get_ip(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_ip(&ctx)
    }

    /// Set the instruction pointer register.
    pub fn set_ip(&mut self, value: RustBV) {
        self.registers.set_ip(value);
        if let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64() {
            self.pc = v;
        }
    }

    /// Get the stack pointer register.
    pub fn get_sp(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_sp(&ctx)
    }

    /// Set the stack pointer register.
    pub fn set_sp(&mut self, value: RustBV) {
        self.registers.set_sp(value);
    }

    /// Get all register bytes (for bulk sync to Python).
    pub fn get_registers_raw(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.arch.state_size()];
        self.registers.copy_to_bytes(&mut bytes);
        bytes
    }

    /// Set all register bytes (for bulk sync from Python).
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.registers.copy_from_bytes(bytes);
    }

    /// Get a reference to the register file.
    pub fn registers(&self) -> &RegisterFile {
        &self.registers
    }

    /// Replace the register file (including symbolic entries).
    pub fn set_registers(&mut self, registers: RegisterFile) {
        self.registers = registers;
    }
}
