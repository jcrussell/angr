//! Z3 lowering for [`RustBV`]: the [`ToZ3`] trait (`to_z3_ast` /
//! `to_z3_ast_cached`, `to_z3_bool` / `to_z3_bool_cached`), and
//! [`admit_clarirs`], the trust-boundary constructor that refuses a term
//! with no Z3 form. A value is a `Concrete` (a Z3 numeral) or a clarirs
//! term, lowered through `symbolic::z3_lower`, whose per-thread cache is the
//! only lowering cache. Lowering can fail (a term deeper than
//! `z3_lower::MAX_LOWER_DEPTH`, angr-mx00p.116.2), so each conversion
//! returns a [`LowerError`] for its query to fail on.
//!
//! The value layer itself (`value.rs`, `value_ops.rs`) never names Z3:
//! [`RustBV::from_clarirs`] only encodes the sort, and lowering lives here
//! (angr-mx00p.116.5).
//!
//! The `deny` below follows the crate [Panic policy](crate#panic-policy)
//! (angr-qwyti.11).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use clarirs_core::prelude::{AstRef, ClarirsError};

use super::z3_lower::LowerError;
use crate::bv::RustBV;

/// Lowering a value into the engine's thread-local Z3 context. Implemented
/// for [`RustBV`]; a trait so the value type stays free of Z3 and the
/// solver-facing surface is opt-in by `use crate::symbolic::ToZ3`.
pub trait ToZ3 {
    /// Convert this value to a Z3 bitvector, lowering a clarirs term on
    /// first use.
    fn to_z3_ast(&self) -> Result<z3::ast::BV, LowerError>;

    /// [`Self::to_z3_ast`] with a per-call cache keyed by value address, for
    /// callers lowering many values in one pass. A clarirs term's lowering is
    /// also kept in `z3_lower`'s per-thread cache.
    fn to_z3_ast_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> Result<z3::ast::BV, LowerError>;

    /// Convert a 1-bit value to a native Z3 Bool, avoiding ITE wrapping.
    ///
    /// For comparison ops (Eq, Ne, Ult, etc.), produces the native Z3 Bool
    /// directly instead of going through ITE(cmp, BV(1,1), BV(0,1)) then
    /// `._eq(BV(1,1))`. Saves 3 Z3 AST nodes per comparison constraint.
    ///
    /// # Panics (debug builds)
    ///
    /// The 1-bit precondition is enforced here rather than left to callers:
    /// a wider BV must be narrowed first (`bv.ne(&zero)`, as
    /// `assume_true_ast` and `add_constraint_from_ast` do). See
    /// `to_z3_bool_cached` for why a multi-bit input would be silently
    /// mis-converted instead of rejected.
    fn to_z3_bool(&self) -> Result<z3::ast::Bool, LowerError>;

    /// Cached worker behind `to_z3_bool` — same 1-bit precondition.
    ///
    /// # Panics (debug builds)
    ///
    /// Asserts `self.width() == 1`. Without it a multi-bit input is not an
    /// error but a *wrong answer*: the Bool view reads a bit, and a wider
    /// value has no single bit to read. Matches the `debug_assert_eq!` width
    /// convention the binary ops in `value_ops.rs` use.
    fn to_z3_bool_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> Result<z3::ast::Bool, LowerError>;
}

impl ToZ3 for RustBV {
    fn to_z3_ast(&self) -> Result<z3::ast::BV, LowerError> {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_ast_cached(&mut cache)
    }

    fn to_z3_ast_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> Result<z3::ast::BV, LowerError> {
        let cache_key = self as *const RustBV as usize;
        if let Some(cached) = cache.get(&cache_key) {
            super::stats::record_z3_ast_cache_hit();
            return Ok(cached.clone());
        }
        super::stats::record_z3_ast_cache_miss();
        let result = match self {
            RustBV::Concrete { value, width } => super::bv_codec::make_bv_const(*value, *width),
            RustBV::Ast { ast, .. } => super::z3_lower::lower_bv(ast)?,
        };
        cache.insert(cache_key, result.clone());
        Ok(result)
    }

    fn to_z3_bool(&self) -> Result<z3::ast::Bool, LowerError> {
        debug_assert_eq!(
            self.width(),
            1,
            "to_z3_bool expects a 1-bit RustBV; narrow with ne(0) first"
        );
        super::stats::record_z3_ast_build();
        super::stats::record_z3_bool_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_bool_cached(&mut cache)
    }

    fn to_z3_bool_cached(
        &self,
        _cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> Result<z3::ast::Bool, LowerError> {
        debug_assert_eq!(
            self.width(),
            1,
            "to_z3_bool_cached expects a 1-bit RustBV; narrow with ne(0) first"
        );
        match self {
            RustBV::Concrete { value, .. } => Ok(z3::ast::Bool::from_bool(*value != 0)),
            // A clarirs term: the Bool its bit stands for, lowered natively,
            // so `ite(cmp, 1, 0)` reaches Z3 as `cmp`.
            RustBV::Ast { ast, .. } => lower_clarirs_bool(ast),
        }
    }
}

/// Lower the Bool a 1-bit [`RustBV::Ast`] term stands for: the `c` of the
/// `ite(c, 1, 0)` encoding, or `bit != 0`.
fn lower_clarirs_bool(ast: &AstRef<'static>) -> Result<z3::ast::Bool, LowerError> {
    let cond = crate::bv::bool_of(ast)
        .map_err(|err| LowerError::InvalidArgs(format!("no Bool view: {err}")))?;
    super::z3_lower::lower_bool(&cond)
}

/// [`RustBV::from_clarirs`] for a term arriving from outside the engine (a
/// claripy AST, a ccall result, a solver-side import): the same sort
/// encoding, then the term is lowered, which both warms `z3_lower`'s cache
/// and refuses what has no Z3 form (strings nested in a bitvector, the VSA
/// ops, a term deeper than `z3_lower::MAX_LOWER_DEPTH`) at the boundary.
/// Terms the ops build later over the result can still outgrow that depth,
/// which is why [`ToZ3::to_z3_ast`] returns a `Result` (angr-mx00p.116.2,
/// angr-mx00p.116.5).
pub fn admit_clarirs(ast: AstRef<'static>) -> Result<RustBV, ClarirsError> {
    let value = RustBV::from_clarirs(ast)?;
    if let RustBV::Ast { ast, .. } = &value {
        super::z3_lower::lower_bv(ast)
            .map_err(|e| ClarirsError::UnsupportedOperation(e.to_string()))?;
    }
    Ok(value)
}

// Direct coverage for the float ops (angr-c7xno.74), built as the engine
// builds them and lowered through clarirs. `z3` gating because every test
// drives `SymContext::eval`.
test_submod!("value_z3_fp_tests.rs" => value_z3_fp_tests);

// The solver-facing halves of the `crate::bv` test files: each lowers or
// solves, so it stays on this side of the value layer (angr-mx00p.116.7).
test_submod!("value_simplify_z3_tests.rs" => value_simplify_z3_tests);
test_submod!("value_zext_cmp_z3_tests.rs" => value_zext_cmp_z3_tests);
test_submod!("value_serde_z3_tests.rs" => value_serde_z3_tests);
test_submod!("value_ast_z3_tests.rs" => value_ast_z3_tests);
test_submod!("value_concrete_arm_z3_tests.rs" => value_concrete_arm_z3_tests);
test_submod!("value_ops_z3_property_tests.rs" => value_ops_z3_property_tests);
test_submod!("names_z3_tests.rs" => names_z3_tests);

/// `lowered`'s Z3 term, or `None` after a warn naming the `query` that must
/// go without it. A term with no Z3 form (one deeper than
/// `z3_lower::MAX_LOWER_DEPTH`) fails the query it was lowered for, the way a
/// Z3 Unknown does, instead of aborting the process (angr-mx00p.116.2).
pub(crate) fn lowered_or_warn<T>(lowered: Result<T, LowerError>, query: &str) -> Option<T> {
    silent_default!(
        cat_c,
        lowered.map(Some),
        None,
        |err| "{query}: the term has no Z3 form ({err}); the query fails"
    )
}
