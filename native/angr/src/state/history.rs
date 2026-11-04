//! History and call-stack tracking for `RustSimState`.
//!
//! Basic-block visit history (plain and detailed), plus the call-stack
//! push/pop/replace accessors used on `Ijk_Call` / `Ijk_Ret`. Both honor
//! `max_history` FIFO eviction. Split out of `mod.rs` per the god-object
//! decomposition (angr-0mqkc.5); mirrors the `registers.rs` / `solver_access.rs` /
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
    ///
    /// Counts the block in `block_count` even when history tracking is off,
    /// as Python's `history.block_count` does.
    pub fn add_to_history(&mut self, addr: u64) {
        self.block_count = self.block_count.wrapping_add(1);
        if self.track_history {
            self.history.push_back(addr);
            if self.max_history > 0
                && self.history.len() > self.max_history
                && let Some(evicted) = self.history.pop_front()
            {
                self.record_evicted_visit(evicted);
            }
        }
    }

    /// Count `n` blocks in `block_count` without touching `history`: the
    /// blocks a step ran beyond the one its `add_to_history` counted, or a
    /// procedure step, which Python's `history.block_count` counts too.
    pub fn count_blocks(&mut self, n: u64) {
        self.block_count = self.block_count.wrapping_add(n);
    }

    /// Blocks this path has executed, not capped by `max_history`.
    pub fn block_count(&self) -> u64 {
        self.block_count
    }

    /// Per-address visit counts of the entries `max_history` evicted from
    /// `history`. An address's full visit count is this plus its
    /// occurrences in `history`.
    pub fn evicted_visits(&self) -> &HashMap<u64, u32> {
        &self.evicted_visits
    }

    /// Fold one evicted `history` entry into `evicted_visits`.
    pub(super) fn record_evicted_visit(&mut self, addr: u64) {
        let count = Arc::make_mut(&mut self.evicted_visits)
            .entry(addr)
            .or_insert(0);
        *count = count.saturating_add(1);
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

    /// Return from a hooked procedure to `return_addr`.
    ///
    /// A procedure (native or bounced to Python) returns without executing a
    /// VEX `Ijk_Ret`, so the interpreter's return path never pops the frame
    /// its call pushed. Unwind it here the same way: through the frame that
    /// expects `return_addr`, or not at all when none does (angr-mx00p.91).
    /// The return is recorded as an `Ijk_Ret` history entry from the hook
    /// address, as angr's SimProcedure step records it, so an export reads
    /// the jumpkind that led to `return_addr` (angr-mx00p.89).
    pub fn return_from_procedure(&mut self, return_addr: u64) {
        let hook_addr = self.pc();
        self.add_history_entry(hook_addr, HistoryEntry::JK_RET, return_addr);
        self.count_blocks(1);
        if let Some(i) = self
            .call_stack
            .iter()
            .rposition(|f| f.return_addr == return_addr)
        {
            self.call_stack.truncate(i);
        }
        self.set_pc(return_addr);
    }

    /// Replace the call stack (used when restoring from interpreter).
    pub fn set_call_stack(&mut self, call_stack: Vec<CallStackEntry>) {
        self.call_stack = call_stack;
    }
}
