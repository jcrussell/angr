//! The interpreter's view of the engine that runs it.
//!
//! [`EngineHost`] is every service a block's execution asks of the outside
//! world: memory the interpreter does not hold, a lift it cannot do natively,
//! a dirty helper it does not model, and `state.inspect` breakpoints. The
//! PyO3 shell implements it for `PythonCallbacks` (`callbacks/host_impl.rs`),
//! where Python objects are built and converted; no Python type crosses this
//! boundary, so `interpreter/` holds no PyO3 code (angr-mx00p.75).
//!
//! Errors are the host's message as a `String`: the interpreter only wraps
//! them in `CbExecutionError::Callback`, never inspects them.
//!
//! The value types on the two sides of this boundary live in the interpreter
//! too: [`LiftBlockOutcome`] here, `InspectBit` in `inspect_bits`, the
//! run-loop results in `run_result` and the execution knobs in `config`.

use super::InspectBit;
use crate::symbolic::{RustBV, SymContext};

/// What the host's lift produced ([`EngineHost::lift_block`]).
#[derive(Debug)]
pub(crate) enum LiftBlockOutcome {
    /// No lifter can produce a block here (no bytes, a lifter error).
    Unliftable,
    /// The lifted block.
    Lifted(crate::vex::ir::IRSB),
    /// The host's lifter returned something it could not read as a block.
    Malformed(String),
}

/// One load the host served.
pub(crate) struct HostLoad {
    /// The bytes, little-endian, when the value is concrete.
    pub(crate) bytes: Vec<u8>,
    /// Whether the value is symbolic.
    pub(crate) symbolic: bool,
    /// The symbolic value, when the host had one the interpreter can use.
    /// `symbolic` with no `value` means "symbolic, but mint a fresh symbol".
    pub(crate) value: Option<RustBV>,
}

/// A page the host served: its bytes, its permission bits, and whether it is
/// backed by a symbolic region.
pub(crate) type HostPage = (Vec<u8>, u8, bool);

/// Services the interpreter asks of whoever runs it.
pub(crate) trait EngineHost {
    // ---- Capability probes -------------------------------------------------

    fn has_memory_store_symbolic_value(&self) -> bool;
    fn has_memory_store_symbolic_full(&self) -> bool;
    fn has_memory_load_symbolic_full(&self) -> bool;
    fn has_fetch_page(&self) -> bool;
    fn has_dirty_call(&self) -> bool;
    /// Whether the host can serve `page_addr` at all.
    fn can_serve_page(&self, page_addr: u64) -> bool;
    /// Whether the host holds `page_addr` already.
    fn has_page(&self, page_addr: u64) -> bool;
    /// Whether a `state.inspect` breakpoint is registered for `event`. Called
    /// per statement and per expression, so it must stay a cheap read.
    fn inspect_event_enabled(&self, event: InspectBit) -> bool;

    // ---- Memory ------------------------------------------------------------

    fn memory_load(&self, ctx: &SymContext, addr: u64, size: u32) -> Result<HostLoad, String>;
    fn memory_load_batch(
        &self,
        ctx: &SymContext,
        loads: &[(u64, u32)],
    ) -> Result<Vec<HostLoad>, String>;
    /// A load at a symbolic address, resolved by the host's memory model.
    /// `None` when the host's value does not convert to a `RustBV`.
    fn memory_load_symbolic_full(
        &self,
        ctx: &SymContext,
        addr: &RustBV,
        size: u32,
    ) -> Result<Option<RustBV>, String>;
    fn memory_store(&self, addr: u64, data: &[u8]) -> Result<(), String>;
    fn memory_store_batch(&self, stores: &[(u64, Vec<u8>)]) -> Result<(), String>;
    fn memory_store_symbolic(
        &self,
        addrs: &[u64],
        data: &RustBV,
        addr_ast: &RustBV,
    ) -> Result<(), String>;
    fn memory_store_symbolic_value(&self, addr: u64, value: &RustBV) -> Result<(), String>;
    fn memory_store_symbolic_full(&self, addr: &RustBV, data: &RustBV) -> Result<(), String>;
    fn fetch_page(&self, page_addr: u64) -> Result<HostPage, String>;
    fn batch_fetch_pages(&self, page_addrs: &[u64]) -> Result<Vec<HostPage>, String>;

    // ---- Code --------------------------------------------------------------

    fn lift_block(
        &self,
        addr: u64,
        opt_level: Option<i32>,
        dirty_bytes: Option<&[u8]>,
    ) -> Result<LiftBlockOutcome, String>;
    /// Run a dirty helper the interpreter does not model. Returns the result
    /// bytes and whether the result is symbolic.
    fn dirty_call(
        &self,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> Result<(Vec<u8>, bool), String>;

    // ---- state.inspect breakpoints -----------------------------------------
    //
    // The interpreter checks `inspect_event_enabled` before calling these. A
    // breakpoint is observation-only: the host logs and drops its failures.
    // The two that can inject a value return the override, converted with
    // `ctx`; the caller still checks its width.

    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per breakpoint attr"
    )]
    fn inspect_mem_read(
        &self,
        ctx: &SymContext,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value: &RustBV,
        endness: &str,
    ) -> Option<RustBV>;
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per breakpoint attr"
    )]
    fn inspect_mem_write(
        &self,
        ctx: &SymContext,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value: &RustBV,
        endness: &str,
    ) -> Option<RustBV>;
    fn inspect_reg_read(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV);
    fn inspect_reg_write(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV);
    fn inspect_tmp_read(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV);
    fn inspect_tmp_write(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV);
    fn inspect_expr(&self, state_id: i64, when: &str, value: &RustBV);
    fn inspect_exit(&self, state_id: i64, when: &str, target: u64, jumpkind: &str, guard: &RustBV);
    fn inspect_address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr: &RustBV,
        result: Option<Vec<u64>>,
    );
    fn inspect_symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size_bits: u32,
        value: &RustBV,
    );
    fn inspect_instruction(&self, state_id: i64, when: &str, addr: u64);
    fn inspect_irsb(&self, state_id: i64, when: &str, addr: u64);
    fn inspect_statement(&self, state_id: i64, when: &str, stmt_idx: u32);
    fn inspect_call(&self, state_id: i64, when: &str, function_address: u64);
    fn inspect_return(&self, state_id: i64, when: &str, function_address: u64);
    fn inspect_vex_lift(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: Option<u32>,
        bytes: Option<&[u8]>,
    );
}
