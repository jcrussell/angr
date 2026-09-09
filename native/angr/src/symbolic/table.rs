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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use super::handle::RustBVHandle;
use super::{RustBV, SymContext};

/// Failure modes for a two-operand op on the symbol table.
///
/// Python-agnostic on purpose: this layer must stay pyo3-free (see the module
/// docs). `solver.rs` owns the `From<BinaryOpError> for PyErr` conversion that
/// turns each variant into a `PyValueError` at the Python boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpError {
    /// One or both handle ids are absent from the table.
    MissingHandle { a_id: u64, b_id: u64 },
    /// Operands have different bit-widths. Every binary op (arith, bitwise,
    /// shift, comparison) requires equal widths — the shift amount is resized
    /// to the value width before it reaches this layer, so shifts are no
    /// exception. Rejecting here prevents handing mismatched sorts to Z3 (a
    /// process abort under `panic=abort`) or silently folding to a constant.
    WidthMismatch { lhs: u32, rhs: u32 },
}

/// Generate `op_$name(a_id, b_id, ctx)` methods that delegate to
/// `RustBV::$method(b, ctx)` after a single read-lock fetch.
///
/// See the `Op method generators` section in `impl RustSymbolTable` for
/// the rationale and how to add a new entry.
macro_rules! op_binary {
    ( $( ($name:ident, $method:ident, $doc:expr) ),+ $(,)? ) => {
        $(
            #[doc = $doc]
            pub fn $name(
                &self,
                a_id: u64,
                b_id: u64,
                ctx: &SymContext,
            ) -> Result<RustBVHandle, BinaryOpError> {
                let symbols = self.symbols.read();
                let (a, b) = match (symbols.get(&a_id), symbols.get(&b_id)) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(BinaryOpError::MissingHandle { a_id, b_id }),
                };
                // Reject mismatched widths at the Python boundary; the RustBV
                // method only `debug_assert`s equal width, so a release build
                // would otherwise hand mismatched sorts to Z3.
                if a.width() != b.width() {
                    return Err(BinaryOpError::WidthMismatch {
                        lhs: a.width(),
                        rhs: b.width(),
                    });
                }
                let result = a.$method(b, ctx);
                drop(symbols); // Release read lock before acquiring write lock
                Ok(self.insert(result))
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
    ///
    /// `Arc`-wrapped so [`RustSymbolTable::fork`] is O(1): the child shares the
    /// parent's map until either side writes, at which point `write_symbols`'
    /// [`Arc::make_mut`] clones it exactly once. Same copy-on-write shape as
    /// `SymContext::symbol_table` (angr-0dgj), which this deliberately mirrors.
    ///
    /// This is one of the crate's `!Send` copy-on-write handles ([`RustBV`]
    /// holds Z3 ASTs), so it is built via [`crate::arc_shared`] rather than a
    /// local `#[allow(clippy::arc_with_non_send_sync)]` — see that helper's
    /// docs for why the suppression is sound.
    symbols: RwLock<Arc<HashMap<u64, RustBV>>>,
    /// Counter for generating unique handle IDs.
    next_id: AtomicU64,
}

impl RustSymbolTable {
    /// Create a new empty symbol table.
    pub fn new() -> Self {
        RustSymbolTable {
            symbols: RwLock::new(crate::arc_shared(HashMap::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// Run `f` against a mutable view of the map, un-sharing it from any fork
    /// sibling first.
    ///
    /// Every mutating method goes through here so the copy-on-write unshare
    /// cannot be forgotten at one call site — writing through the `RwLock`
    /// directly would mutate a map a forked table still points at.
    fn write_symbols<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut HashMap<u64, RustBV>) -> R,
    {
        let mut guard = self.symbols.write();
        f(Arc::make_mut(&mut guard))
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
        self.write_symbols(|symbols| symbols.insert(id, bv));
        RustBVHandle::new_symbolic(id, width)
    }

    /// Create a new concrete bitvector and return a handle.
    pub fn create_concrete(&self, value: u128, width: u32) -> RustBVHandle {
        let id = self.next_id();
        let bv = RustBV::concrete(value, width);
        self.write_symbols(|symbols| symbols.insert(id, bv));
        RustBVHandle::new_concrete(id, value, width)
    }

    /// Insert an existing RustBV and return a handle.
    pub fn insert(&self, bv: RustBV) -> RustBVHandle {
        let id = self.next_id();
        let width = bv.width();
        let concrete = bv.as_u128();
        self.write_symbols(|symbols| symbols.insert(id, bv));

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
        self.write_symbols(|symbols| symbols.remove(&id))
    }

    /// Clear all values from the table.
    pub fn clear(&self) {
        self.write_symbols(HashMap::clear);
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

    /// Extract bits \[high:low\] (inclusive).
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

    /// Fork the symbol table, creating a copy-on-write child with the same
    /// values.
    ///
    /// O(1): the child shares the parent's map behind an [`Arc`]. The `HashMap`
    /// clone is deferred to the first mutation on *either* side (see
    /// `write_symbols`) and never happens at all for the common
    /// fork-then-read-only case. Handle ids stay valid across the fork because
    /// `next_id` continues from the parent's counter, so neither side reissues
    /// an id the other already bound.
    ///
    /// That guarantee covers only ids inherited *through* a fork. Because the
    /// child's counter merely continues from the parent's rather than
    /// partitioning the id space, two *sibling* children of the same parent
    /// mint post-fork ids independently and can bind the same numeric id to
    /// two unrelated values — they are separate id namespaces, pinned by
    /// `table_tests.rs::test_fork_child_does_not_reissue_parent_ids`. The
    /// Python-visible consequence is documented on [`crate::symbolic::RustBVHandle`],
    /// whose `__eq__`/`__hash__` key on the raw id.
    ///
    /// This is the *only* way to copy a table: `RustSymbolTable` deliberately
    /// does not implement [`Clone`]. A `Clone` forwarding to `fork` existed and
    /// had zero call sites, so nothing would have caught it drifting out of
    /// sync with the real copy semantics. Leaving it unimplemented means a
    /// container that grows a `#[derive(Clone)]` fails to compile instead of
    /// silently picking up a second, unexercised copy path (angr-0jh0j.57).
    pub fn fork(&self) -> Self {
        let symbols = Arc::clone(&self.symbols.read());
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

test_submod!("table_tests.rs" => tests);
