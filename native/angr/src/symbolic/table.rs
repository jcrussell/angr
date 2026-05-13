//! Symbol table for managing RustBV values by handle ID.
//!
//! The RustSymbolTable stores RustBV values and provides access via
//! opaque handle IDs. This allows Python to hold lightweight handles
//! while the actual symbolic values (including Z3 ASTs) live in Rust.
//!
//! Thread-safety: The table uses RwLock for concurrent read access
//! and exclusive write access. Since Z3 types are not Send, the table
//! itself should not be shared across threads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::handle::RustBVHandle;
use super::{RustBV, SymContext};

/// Global symbol table for storing RustBV values.
///
/// Each solver context has its own symbol table, allowing different
/// exploration paths to have independent symbolic state.
pub struct RustSymbolTable {
    /// Map from handle ID to RustBV value.
    symbols: RwLock<HashMap<u64, RustBV>>,
    /// Counter for generating unique handle IDs.
    next_id: AtomicU64,
}

impl RustSymbolTable {
    /// Create a new empty symbol table.
    pub fn new() -> Self {
        RustSymbolTable {
            symbols: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    /// Generate the next unique handle ID.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the current number of entries in the table.
    pub fn len(&self) -> usize {
        self.symbols.read().len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.read().is_empty()
    }

    // =========================================================================
    // Value Creation
    // =========================================================================

    /// Create a new symbolic bitvector and return a handle.
    pub fn create_symbolic(&self, ctx: &SymContext, name: &str, width: u32) -> RustBVHandle {
        let id = self.next_id();
        let bv = RustBV::symbolic(ctx, name, width);
        self.symbols.write().insert(id, bv);
        RustBVHandle::new_symbolic(id, width)
    }

    /// Create a new concrete bitvector and return a handle.
    pub fn create_concrete(&self, value: u128, width: u32) -> RustBVHandle {
        let id = self.next_id();
        let bv = RustBV::concrete(value, width);
        self.symbols.write().insert(id, bv);
        RustBVHandle::new_concrete(id, value, width)
    }

    /// Insert an existing RustBV and return a handle.
    pub fn insert(&self, bv: RustBV) -> RustBVHandle {
        let id = self.next_id();
        let width = bv.width();
        let concrete = bv.as_u128();
        self.symbols.write().insert(id, bv);

        if let Some(v) = concrete {
            RustBVHandle::new_concrete(id, v, width)
        } else {
            RustBVHandle::new_symbolic(id, width)
        }
    }

    // =========================================================================
    // Value Access
    // =========================================================================

    /// Get a RustBV by handle ID.
    pub fn get(&self, id: u64) -> Option<RustBV> {
        self.symbols.read().get(&id).cloned()
    }

    /// Get a reference to a RustBV by handle ID (for read-only access).
    /// Note: This acquires a read lock; caller should not hold it long.
    pub fn with_value<F, R>(&self, id: u64, f: F) -> Option<R>
    where
        F: FnOnce(&RustBV) -> R,
    {
        let symbols = self.symbols.read();
        symbols.get(&id).map(f)
    }

    /// Remove a value from the table.
    pub fn remove(&self, id: u64) -> Option<RustBV> {
        self.symbols.write().remove(&id)
    }

    /// Clear all values from the table.
    pub fn clear(&self) {
        self.symbols.write().clear();
    }

    // =========================================================================
    // Arithmetic Operations (return new handles)
    // =========================================================================

    /// Add two values and return a handle to the result.
    pub fn op_add(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.add(b, ctx);
        drop(symbols); // Release read lock before acquiring write lock
        Some(self.insert(result))
    }

    /// Subtract two values and return a handle to the result.
    pub fn op_sub(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.sub(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Multiply two values and return a handle to the result.
    pub fn op_mul(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.mul(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned division of two values.
    pub fn op_udiv(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.udiv(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed division of two values.
    pub fn op_sdiv(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.sdiv(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned remainder of two values.
    pub fn op_urem(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.urem(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed remainder of two values.
    pub fn op_srem(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.srem(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Negation of a value.
    pub fn op_neg(&self, a_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.neg(ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    /// Bitwise AND of two values.
    pub fn op_and(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.and(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Bitwise OR of two values.
    pub fn op_or(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.or(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Bitwise XOR of two values.
    pub fn op_xor(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.xor(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Bitwise NOT of a value.
    pub fn op_not(&self, a_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.not(ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    // =========================================================================
    // Shift Operations
    // =========================================================================

    /// Left shift.
    pub fn op_shl(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.shl(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Logical right shift.
    pub fn op_lshr(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.lshr(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Arithmetic right shift.
    pub fn op_ashr(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.ashr(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Rotate left.
    pub fn op_rotl(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.rotl(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Rotate right.
    pub fn op_rotr(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.rotr(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    // =========================================================================
    // Comparison Operations (return 1-bit handles)
    // =========================================================================

    /// Equality comparison.
    pub fn op_eq(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.eq(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Inequality comparison.
    pub fn op_ne(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.ne(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned less than.
    pub fn op_ult(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.ult(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned less than or equal.
    pub fn op_ule(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.ule(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned greater than.
    pub fn op_ugt(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.ugt(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Unsigned greater than or equal.
    pub fn op_uge(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.uge(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed less than.
    pub fn op_slt(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.slt(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed less than or equal.
    pub fn op_sle(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.sle(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed greater than.
    pub fn op_sgt(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.sgt(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Signed greater than or equal.
    pub fn op_sge(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.sge(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    // =========================================================================
    // Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn op_zero_extend(
        &self,
        a_id: u64,
        to_width: u32,
        ctx: &SymContext,
    ) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.zero_extend(to_width, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Sign-extend to a wider width.
    pub fn op_sign_extend(
        &self,
        a_id: u64,
        to_width: u32,
        ctx: &SymContext,
    ) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.sign_extend(to_width, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Truncate to a narrower width.
    pub fn op_truncate(&self, a_id: u64, to_width: u32, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.truncate(to_width, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Extract bits [high:low] (inclusive).
    pub fn op_extract(
        &self,
        a_id: u64,
        high: u32,
        low: u32,
        ctx: &SymContext,
    ) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let result = a.extract(high, low, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// Concatenate two values (a becomes high bits).
    pub fn op_concat(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let a = symbols.get(&a_id)?;
        let b = symbols.get(&b_id)?;
        let result = a.concat(b, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    /// If-then-else.
    pub fn op_ite(
        &self,
        cond_id: u64,
        then_id: u64,
        else_id: u64,
        ctx: &SymContext,
    ) -> Option<RustBVHandle> {
        let symbols = self.symbols.read();
        let cond = symbols.get(&cond_id)?;
        let then_val = symbols.get(&then_id)?;
        let else_val = symbols.get(&else_id)?;
        let result = cond.ite(then_val, else_val, ctx);
        drop(symbols);
        Some(self.insert(result))
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the symbol table, creating a copy with the same values.
    pub fn fork(&self) -> Self {
        let symbols = self.symbols.read().clone();
        let next_id = self.next_id.load(Ordering::SeqCst);
        RustSymbolTable {
            symbols: RwLock::new(symbols),
            next_id: AtomicU64::new(next_id),
        }
    }
}

impl Default for RustSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RustSymbolTable {
    fn clone(&self) -> Self {
        self.fork()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_table_creation() {
        let table = RustSymbolTable::new();
        assert!(table.is_empty());
    }

    #[test]
    fn test_create_concrete() {
        let table = RustSymbolTable::new();
        let handle = table.create_concrete(42, 32);
        assert_eq!(handle.width(), 32);
        assert!(handle.is_concrete());
        assert_eq!(handle.concrete_value(), Some(42));
    }

    #[test]
    fn test_create_symbolic() {
        let table = RustSymbolTable::new();
        let ctx = SymContext::new_mock();
        let handle = table.create_symbolic(&ctx, "x", 32);
        assert_eq!(handle.width(), 32);
        assert!(!handle.is_concrete());
    }

    #[test]
    fn test_operations() {
        let table = RustSymbolTable::new();
        let ctx = SymContext::new_mock();

        let h1 = table.create_concrete(10, 32);
        let h2 = table.create_concrete(32, 32);

        let h3 = table.op_add(h1.id(), h2.id(), &ctx).unwrap();
        assert_eq!(h3.width(), 32);
        assert!(h3.is_concrete());
        assert_eq!(h3.concrete_value(), Some(42));
    }

    #[test]
    fn test_fork() {
        let table = RustSymbolTable::new();
        let h1 = table.create_concrete(42, 32);

        let forked = table.fork();
        assert_eq!(forked.len(), 1);
        assert!(forked.get(h1.id()).is_some());
    }
}
