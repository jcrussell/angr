//! Hooks and per-state symbolic-memory metadata for `RustSimState`.
//!
//! The plain hook-address set (`add_hook` / `remove_hook` / `is_hooked` /
//! `clear_hooks`) plus the per-state claripy-AST metadata maps: the
//! hook-symbolic-memory table, the addr-to-AST table, and the symbolic-pages
//! map, with their insert/read/replace/clear accessors. Split out of `mod.rs`
//! per the god-object decomposition (angr-0mqkc.5); mirrors the
//! `registers.rs` / `history.rs` / `options.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    // =========================================================================
    // Hooks
    // =========================================================================

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hooks).insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hooks).remove(&addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hooks.contains(&addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        // Avoid CoW clone if already empty.
        if !self.hooks.is_empty() {
            Arc::make_mut(&mut self.hooks).clear();
        }
    }

    // =========================================================================
    // Per-state metadata (claripy AST refs) — see field docs above.
    // =========================================================================

    /// Insert/replace a hook-symbolic-memory entry.
    pub fn set_hook_symbolic_memory(&mut self, addr: u64, ast: Py<PyAny>, size: u32) {
        self.hook_symbolic_memory
            .insert(addr, (SharedPyAst::new(ast), size));
    }

    /// Insert/replace an addr-to-AST entry.
    pub fn set_addr_to_ast(&mut self, addr: u64, ast: Py<PyAny>, size: u32) {
        self.addr_to_ast.insert(addr, (SharedPyAst::new(ast), size));
    }

    /// Read-only access to the symbolic-pages map.
    pub fn symbolic_pages(&self) -> &HashMap<u64, SharedPyAst> {
        &self.symbolic_pages
    }

    /// Read-only access to the hook-symbolic-memory map.
    pub fn hook_symbolic_memory(&self) -> &HashMap<u64, (SharedPyAst, u32)> {
        &self.hook_symbolic_memory
    }

    /// Read-only access to the addr-to-AST map.
    pub fn addr_to_ast(&self) -> &HashMap<u64, (SharedPyAst, u32)> {
        &self.addr_to_ast
    }

    /// Replace the entire symbolic-pages map (used by full-page recovery flow).
    /// Takes bare `Py` handles from the caller and wraps them — the `Arc` is an
    /// internal storage detail (see [`SharedPyAst`]), not part of this API.
    pub fn replace_symbolic_pages(&mut self, pages: HashMap<u64, Py<PyAny>>) {
        self.symbolic_pages = pages
            .into_iter()
            .map(|(addr, ast)| (addr, SharedPyAst::new(ast)))
            .collect();
    }

    /// Drop all per-state metadata (called when a state is no longer needed).
    pub fn clear_state_metadata(&mut self) {
        self.symbolic_pages.clear();
        self.hook_symbolic_memory.clear();
        self.addr_to_ast.clear();
    }
}
