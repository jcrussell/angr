//! [`TestHost`], the plain-Rust [`EngineHost`] the interpreter's tests run
//! against, so they need no Python interpreter (angr-mx00p.75).
//!
//! `TestHost::default()` behaves as the PyO3 shell's `PythonCallbacks::new()`
//! does with no callback set: every capability probe is off, the page probes
//! fail open (no page snapshot installed), every service the interpreter
//! calls errors with `"<slot> callback not set"`, and the `state.inspect`
//! hooks are disabled no-ops. The few knobs below switch on the services a
//! test needs and record what the interpreter asked for, in place of the
//! Python closures those tests used to register.

use std::cell::RefCell;

use super::{EngineHost, HostLoad, HostPage, InspectBit, LiftBlockOutcome};
use crate::symbolic::{RustBV, SymContext};

/// How [`TestHost`] answers `memory_store_symbolic_value`.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolicValueStore {
    /// No such service: the probe is off and a call errors.
    #[default]
    Unset,
    /// The service exists and absorbs the store, recording it, as the
    /// production callback does when `state.memory` is Rust memory.
    Accept,
    /// The service exists and fails every store.
    Fail,
}

/// A host with no Python behind it. See the module doc for the defaults.
#[derive(Default)]
pub(crate) struct TestHost {
    /// Serve `memory_load` / `memory_load_batch` with concrete zero bytes
    /// instead of erroring.
    pub(crate) serve_zero_loads: bool,
    /// Report a symbolic-address load service (`has_memory_load_symbolic_full`)
    /// and answer every such load with `Ok(None)`, the "value did not
    /// convert" reply.
    pub(crate) serve_symbolic_full_loads: bool,
    /// Every `(addr, size)` the interpreter loaded through the host, batch
    /// entries included.
    pub(crate) loads: RefCell<Vec<(u64, u32)>>,
    /// The size of every symbolic-address load the interpreter handed over.
    pub(crate) symbolic_full_loads: RefCell<Vec<u32>>,
    /// Record concrete `memory_store`s (batch entries included) instead of
    /// erroring.
    pub(crate) accept_stores: bool,
    /// Every concrete store the host accepted.
    pub(crate) stores: RefCell<Vec<(u64, Vec<u8>)>>,
    /// The `memory_store_symbolic_value` service.
    pub(crate) symbolic_value_store: SymbolicValueStore,
    /// Every symbolic-value store the host accepted.
    pub(crate) symbolic_value_stores: RefCell<Vec<(u64, RustBV)>>,
}

fn not_set<T>(slot: &str) -> Result<T, String> {
    Err(format!("{slot} callback not set"))
}

impl TestHost {
    fn zero_load(&self, addr: u64, size: u32) -> Result<HostLoad, String> {
        if !self.serve_zero_loads {
            return not_set("memory_load");
        }
        self.loads.borrow_mut().push((addr, size));
        Ok(HostLoad {
            bytes: vec![0; size as usize],
            symbolic: false,
            value: None,
        })
    }
}

impl EngineHost for TestHost {
    fn has_memory_store_symbolic_value(&self) -> bool {
        self.symbolic_value_store != SymbolicValueStore::Unset
    }

    fn has_memory_store_symbolic_full(&self) -> bool {
        false
    }

    fn has_memory_load_symbolic_full(&self) -> bool {
        self.serve_symbolic_full_loads
    }

    fn has_fetch_page(&self) -> bool {
        false
    }

    fn has_dirty_call(&self) -> bool {
        false
    }

    fn can_serve_page(&self, _page_addr: u64) -> bool {
        true
    }

    fn has_page(&self, _page_addr: u64) -> bool {
        true
    }

    fn inspect_event_enabled(&self, _event: InspectBit) -> bool {
        false
    }

    fn memory_load(&self, _ctx: &SymContext, addr: u64, size: u32) -> Result<HostLoad, String> {
        self.zero_load(addr, size)
    }

    /// Per-entry `memory_load`, as `PythonCallbacks` does with no batch
    /// callback set.
    fn memory_load_batch(
        &self,
        _ctx: &SymContext,
        loads: &[(u64, u32)],
    ) -> Result<Vec<HostLoad>, String> {
        loads
            .iter()
            .map(|&(addr, size)| self.zero_load(addr, size))
            .collect()
    }

    fn memory_load_symbolic_full(
        &self,
        _ctx: &SymContext,
        _addr: &RustBV,
        size: u32,
    ) -> Result<Option<RustBV>, String> {
        if !self.serve_symbolic_full_loads {
            return not_set("memory_load_symbolic_full");
        }
        self.symbolic_full_loads.borrow_mut().push(size);
        Ok(None)
    }

    fn memory_store(&self, addr: u64, data: &[u8]) -> Result<(), String> {
        if !self.accept_stores {
            return not_set("memory_store");
        }
        self.stores.borrow_mut().push((addr, data.to_vec()));
        Ok(())
    }

    /// Per-entry `memory_store`, as `PythonCallbacks` does with no batch
    /// callback set.
    fn memory_store_batch(&self, stores: &[(u64, Vec<u8>)]) -> Result<(), String> {
        stores
            .iter()
            .try_for_each(|(addr, data)| self.memory_store(*addr, data))
    }

    fn memory_store_symbolic(
        &self,
        _addrs: &[u64],
        _data: &RustBV,
        _addr_ast: &RustBV,
    ) -> Result<(), String> {
        not_set("memory_store_symbolic")
    }

    fn memory_store_symbolic_value(&self, addr: u64, value: &RustBV) -> Result<(), String> {
        match self.symbolic_value_store {
            SymbolicValueStore::Unset => not_set("memory_store_symbolic_value"),
            SymbolicValueStore::Accept => {
                self.symbolic_value_stores
                    .borrow_mut()
                    .push((addr, value.clone()));
                Ok(())
            }
            SymbolicValueStore::Fail => Err("symbolic store unavailable".to_string()),
        }
    }

    fn memory_store_symbolic_full(&self, _addr: &RustBV, _data: &RustBV) -> Result<(), String> {
        not_set("memory_store_symbolic_full")
    }

    fn fetch_page(&self, _page_addr: u64) -> Result<HostPage, String> {
        not_set("fetch_page")
    }

    fn batch_fetch_pages(&self, _page_addrs: &[u64]) -> Result<Vec<HostPage>, String> {
        not_set("batch_fetch_pages")
    }

    fn lift_block(
        &self,
        _addr: u64,
        _opt_level: Option<i32>,
        _dirty_bytes: Option<&[u8]>,
    ) -> Result<LiftBlockOutcome, String> {
        not_set("lift_block")
    }

    fn dirty_call(
        &self,
        _name: &str,
        _args: &[u64],
        _ret_ty_bits: u32,
    ) -> Result<(Vec<u8>, bool), String> {
        not_set("dirty_call")
    }

    fn inspect_mem_read(
        &self,
        _ctx: &SymContext,
        _state_id: i64,
        _when: &str,
        _addr: u64,
        _size: u32,
        _value: &RustBV,
        _endness: &str,
    ) -> Option<RustBV> {
        None
    }

    fn inspect_mem_write(
        &self,
        _ctx: &SymContext,
        _state_id: i64,
        _when: &str,
        _addr: u64,
        _size: u32,
        _value: &RustBV,
        _endness: &str,
    ) -> Option<RustBV> {
        None
    }

    fn inspect_reg_read(&self, _: i64, _: &str, _: u32, _: u32, _: &RustBV) {}
    fn inspect_reg_write(&self, _: i64, _: &str, _: u32, _: u32, _: &RustBV) {}
    fn inspect_tmp_read(&self, _: i64, _: &str, _: u32, _: &RustBV) {}
    fn inspect_tmp_write(&self, _: i64, _: &str, _: u32, _: &RustBV) {}
    fn inspect_expr(&self, _: i64, _: &str, _: &RustBV) {}
    fn inspect_exit(&self, _: i64, _: &str, _: u64, _: &str, _: &RustBV) {}
    fn inspect_address_concretization(
        &self,
        _: i64,
        _: &str,
        _: &str,
        _: &RustBV,
        _: Option<Vec<u64>>,
    ) {
    }
    fn inspect_symbolic_variable(&self, _: i64, _: &str, _: &str, _: u32, _: &RustBV) {}
    fn inspect_instruction(&self, _: i64, _: &str, _: u64) {}
    fn inspect_irsb(&self, _: i64, _: &str, _: u64) {}
    fn inspect_statement(&self, _: i64, _: &str, _: u32) {}
    fn inspect_call(&self, _: i64, _: &str, _: u64) {}
    fn inspect_return(&self, _: i64, _: &str, _: u64) {}
    fn inspect_vex_lift(&self, _: i64, _: &str, _: u64, _: Option<u32>, _: Option<&[u8]>) {}
}
