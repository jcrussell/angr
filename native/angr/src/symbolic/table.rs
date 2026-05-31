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

/// Generate `op_$name(a_id, b_id, ctx)` methods that delegate to
/// `RustBV::$method(b, ctx)` after a single read-lock fetch.
///
/// See the `Op method generators` section in `impl RustSymbolTable` for
/// the rationale and how to add a new entry.
macro_rules! op_binary {
    ( $( ($name:ident, $method:ident, $doc:expr) ),+ $(,)? ) => {
        $(
            #[doc = $doc]
            pub fn $name(&self, a_id: u64, b_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
                let symbols = self.symbols.read();
                let a = symbols.get(&a_id)?;
                let b = symbols.get(&b_id)?;
                let result = a.$method(b, ctx);
                drop(symbols); // Release read lock before acquiring write lock
                Some(self.insert(result))
            }
        )+
    };
}

/// Same as [`op_binary!`] but for single-operand methods on `RustBV`.
macro_rules! op_unary {
    ( $( ($name:ident, $method:ident, $doc:expr) ),+ $(,)? ) => {
        $(
            #[doc = $doc]
            pub fn $name(&self, a_id: u64, ctx: &SymContext) -> Option<RustBVHandle> {
                let symbols = self.symbols.read();
                let a = symbols.get(&a_id)?;
                let result = a.$method(ctx);
                drop(symbols);
                Some(self.insert(result))
            }
        )+
    };
}

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
    // Op method generators
    // =========================================================================
    //
    // The 27 single-operand and two-operand op_* methods all share the same
    // shape: read-lock the table, fetch operand(s), call a same-named RustBV
    // method, drop the read lock, then insert the result. These two macros
    // generate them from a table of `(method_name, RustBV-method)` rows.
    //
    // To add a new operation, append one row to the appropriate macro call
    // below and implement the matching RustBV method.

    op_binary! {
        // Arithmetic
        (op_add,  add,  "Add two values and return a handle to the result."),
        (op_sub,  sub,  "Subtract two values and return a handle to the result."),
        (op_mul,  mul,  "Multiply two values and return a handle to the result."),
        (op_udiv, udiv, "Unsigned division of two values."),
        (op_sdiv, sdiv, "Signed division of two values."),
        (op_urem, urem, "Unsigned remainder of two values."),
        (op_srem, srem, "Signed remainder of two values."),
        // Bitwise
        (op_and,  and,  "Bitwise AND of two values."),
        (op_or,   or,   "Bitwise OR of two values."),
        (op_xor,  xor,  "Bitwise XOR of two values."),
        // Shifts
        (op_shl,  shl,  "Left shift."),
        (op_lshr, lshr, "Logical right shift."),
        (op_ashr, ashr, "Arithmetic right shift."),
        (op_rotl, rotl, "Rotate left."),
        (op_rotr, rotr, "Rotate right."),
        // Comparisons (return 1-bit handles)
        (op_eq,   eq,   "Equality comparison."),
        (op_ne,   ne,   "Inequality comparison."),
        (op_ult,  ult,  "Unsigned less than."),
        (op_ule,  ule,  "Unsigned less than or equal."),
        (op_ugt,  ugt,  "Unsigned greater than."),
        (op_uge,  uge,  "Unsigned greater than or equal."),
        (op_slt,  slt,  "Signed less than."),
        (op_sle,  sle,  "Signed less than or equal."),
        (op_sgt,  sgt,  "Signed greater than."),
        (op_sge,  sge,  "Signed greater than or equal."),
    }

    op_unary! {
        (op_neg, neg, "Negation of a value."),
        (op_not, not, "Bitwise NOT of a value."),
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
        assert_eq!(handle.concrete(), Some(42));
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
