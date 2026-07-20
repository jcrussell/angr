//! History and call-stack tracking for `RustSimState`.
//!
//! Basic-block visit history (plain and detailed), plus the call-stack
//! push/pop/replace accessors used on `Ijk_Call` / `Ijk_Ret`. Both honor
//! `max_history` FIFO eviction. Split out of `mod.rs` per the god-object
//! decomposition (angr-0mqkc.5); mirrors the `registers.rs` / `solver.rs` /
//! `options.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Get the history (basic block addresses visited).
    ///
    /// Backed by a `VecDeque` so cap eviction is O(1); callers that need a
    /// contiguous slice or `Vec` should `.iter()` / `.range(..)` over it.
    pub fn history(&self) -> &VecDeque<u64> {
        &self.history
    }

    /// Add an address to history.
    pub fn add_to_history(&mut self, addr: u64) {
        if self.track_history {
            self.history.push_back(addr);
            if self.max_history > 0 && self.history.len() > self.max_history {
                self.history.pop_front();
            }
        }
    }

    /// Get the detailed execution history.
    pub fn detailed_history(&self) -> &VecDeque<HistoryEntry> {
        &self.detailed_history
    }

    /// Add a detailed history entry.
    pub fn add_history_entry(&mut self, addr: u64, jumpkind: u8, jump_target: u64) {
        if self.track_history {
            self.detailed_history.push_back(HistoryEntry {
                addr,
                jumpkind,
                jump_target,
            });
            if self.max_history > 0 && self.detailed_history.len() > self.max_history {
                self.detailed_history.pop_front();
            }
        }
    }

    /// Replace the detailed history (used when restoring from interpreter).
    /// Honors `max_history` — if the incoming buffer is larger than the cap,
    /// only the most-recent `max_history` entries are kept (FIFO eviction).
    pub fn set_detailed_history(&mut self, mut history: Vec<HistoryEntry>) {
        if self.max_history > 0 && history.len() > self.max_history {
            let drop = history.len() - self.max_history;
            history.drain(0..drop);
        }
        // `Vec -> VecDeque` reuses the allocation (O(1)).
        self.detailed_history = history.into();
    }

    // =========================================================================
    // Call Stack Tracking
    // =========================================================================

    /// Get the current call stack.
    pub fn call_stack(&self) -> &[CallStackEntry] {
        &self.call_stack
    }

    /// Get the call stack depth.
    pub fn call_stack_depth(&self) -> usize {
        self.call_stack.len()
    }

    /// Push a call onto the call stack (on Ijk_Call).
    pub fn push_call(
        &mut self,
        call_site_addr: u64,
        callee_addr: u64,
        return_addr: u64,
        stack_ptr: u64,
    ) {
        self.call_stack.push(CallStackEntry {
            call_site_addr,
            callee_addr,
            return_addr,
            stack_ptr,
        });
    }

    /// Pop a call from the call stack (on Ijk_Ret).
    /// Returns the popped entry, or None if the stack is empty.
    pub fn pop_call(&mut self) -> Option<CallStackEntry> {
        self.call_stack.pop()
    }

    /// Get the current function address (top of call stack), if any.
    pub fn current_function_addr(&self) -> Option<u64> {
        self.call_stack.last().map(|e| e.callee_addr)
    }

    /// Replace the call stack (used when restoring from interpreter).
    pub fn set_call_stack(&mut self, call_stack: Vec<CallStackEntry>) {
        self.call_stack = call_stack;
    }
}
