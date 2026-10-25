//! Z3 lowering for [`RustBV`]: `to_z3_ast` / `to_z3_ast_cached`,
//! `to_z3_bool` / `to_z3_bool_cached`. A value is a
//! `Concrete` (a Z3 numeral) or a clarirs term, lowered through
//! `symbolic::z3_lower` and memoized on the value. Lowering can fail (a term
//! deeper than `z3_lower::MAX_LOWER_DEPTH`, angr-mx00p.116.2), so each
//! conversion returns a [`LowerError`] for its query to fail on.
//!
//! The `deny` below follows the crate [Panic policy](crate#panic-policy)
//! (angr-qwyti.11).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::value::RustBV;
use super::z3_lower::LowerError;

impl RustBV {
    // =========================================================================
    // Z3 Integration (when feature is enabled)
    // =========================================================================

    /// Convert this RustBV to a Z3 AST, lowering a clarirs term on first use.
    pub fn to_z3_ast(&self) -> Result<z3::ast::BV, LowerError> {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_ast_cached(&mut cache)
    }

    /// [`Self::to_z3_ast`] with a per-call cache keyed by value address, for
    /// callers lowering many values in one pass. A clarirs term's lowering is
    /// also kept on the value (`memo`) and in `z3_lower`'s per-thread cache.
    pub fn to_z3_ast_cached(
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
            RustBV::Ast { ast, memo, .. } => {
                use z3::ast::Ast;
                let reusable = memo.borrow().as_ref().and_then(|bv| {
                    (bv.get_ctx().get_z3_context() == z3::Context::thread_local().get_z3_context())
                        .then(|| bv.clone())
                });
                if let Some(bv) = reusable {
                    super::stats::record_z3_ast_memo_hit();
                    bv
                } else {
                    let built = super::z3_lower::lower_bv(ast)?;
                    *memo.borrow_mut() = Some(built.clone());
                    built
                }
            }
        };
        cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Convert a 1-bit RustBV to a native Z3 Bool, avoiding ITE wrapping.
    ///
    /// For comparison ops (Eq, Ne, Ult, etc.), produces the native Z3 Bool
    /// directly instead of going through ITE(cmp, BV(1,1), BV(0,1)) then
    /// `._eq(BV(1,1))`. Saves 3 Z3 AST nodes per comparison constraint.
    ///
    /// # Panics (debug builds)
    ///
    /// The 1-bit precondition is enforced here rather than left to callers:
    /// a wider BV must be narrowed first (`bv.ne(&zero, ctx)`, as
    /// `assume_true_ast` and `add_constraint_from_ast` do). See
    /// `to_z3_bool_cached` for why a multi-bit input would be silently
    /// mis-converted instead of rejected.
    pub fn to_z3_bool(&self) -> Result<z3::ast::Bool, LowerError> {
        debug_assert_eq!(
            self.width(),
            1,
            "to_z3_bool expects a 1-bit RustBV; narrow with ne(0) first"
        );
        super::stats::record_z3_ast_build();
        // Bracket the conversion with the node counters so the Bool path's
        // built-vs-memo-reused split is attributable (angr-op0dn.9.6): only
        // the nodes still being *built* here are what a Bool memo (M1.d)
        // (angr-op0dn.9.4) could remove.
        let (miss0, memo0) = super::stats::ast_node_counters();
        let mut cache = std::collections::HashMap::new();
        let result = self.to_z3_bool_cached(&mut cache);
        let (miss1, memo1) = super::stats::ast_node_counters();
        let reused = memo1.saturating_sub(memo0);
        let built = (miss1.saturating_sub(miss0)).saturating_sub(reused);
        super::stats::record_z3_bool_build(built, reused);
        result
    }

    /// Cached worker behind `to_z3_bool` — same 1-bit precondition.
    ///
    /// # Panics (debug builds)
    ///
    /// Asserts `self.width() == 1`. Without it a multi-bit input is not an
    /// error but a *wrong answer*: the Bool view reads a bit, and a wider
    /// value has no single bit to read. Matches the `debug_assert_eq!` width
    /// convention the binary ops in `value_ops.rs` use.
    pub fn to_z3_bool_cached(
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
            RustBV::Ast { ast, .. } => Self::lower_clarirs_bool(ast),
        }
    }

    /// Lower the Bool a 1-bit [`RustBV::Ast`] term stands for: the `c` of the
    /// `ite(c, 1, 0)` encoding, or `bit != 0`.
    fn lower_clarirs_bool(
        ast: &clarirs_core::prelude::AstRef<'static>,
    ) -> Result<z3::ast::Bool, LowerError> {
        let cond = super::value_clarirs::bool_of(ast)
            .map_err(|err| LowerError::InvalidArgs(format!("no Bool view: {err}")))?;
        super::z3_lower::lower_bool(&cond)
    }
}

// Direct coverage for the float ops (angr-c7xno.74), built as the engine
// builds them and lowered through clarirs. `z3` gating because every test
// drives `SymContext::eval`.
test_submod!("value_z3_fp_tests.rs" => value_z3_fp_tests);

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
