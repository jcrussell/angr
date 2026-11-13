//! [`Z3Scratch`]: a bare Z3 solver for tests that check a property of Z3
//! terms they built by hand.
//!
//! A [`SymContext`](super::SymContext) takes constraints as clarirs terms
//! only, so a test comparing an engine value against a reference written in
//! Z3's own API asserts on one of these instead. The method names are the
//! context's, so such a test reads the same.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "test-only helper: a failed precondition should fail the test"
)]

/// A solver that takes raw Z3 Bools.
pub struct Z3Scratch(z3::Solver);

impl Default for Z3Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Z3Scratch {
    /// An empty solver in the thread's Z3 context.
    pub fn new() -> Self {
        Self(z3::Solver::new())
    }

    /// Assert `constraint`.
    pub fn add_constraint(&self, constraint: z3::ast::Bool) {
        self.0.assert(&constraint);
    }

    /// Open a scope.
    pub fn push(&self) {
        self.0.push();
    }

    /// Close the innermost scope.
    pub fn pop(&self) {
        self.0.pop(1);
    }

    /// Whether `bv == value` is consistent with the assertions.
    pub fn solution(&self, bv: &super::RustBV, value: u128) -> bool {
        use super::ToZ3;
        let bv_z3 = bv.to_z3_ast().expect("the value has a Z3 form");
        let value = super::bv_codec::make_bv_const(value, bv.width());
        self.push();
        self.add_constraint(bv_z3.eq(&value));
        let sat = self.is_sat();
        self.pop();
        sat
    }

    /// Whether the assertions are satisfiable. Panics if Z3 cannot decide,
    /// so a test never reads an `Unknown` as either answer.
    pub fn is_sat(&self) -> bool {
        // satresult-exempt: test helper on a bare solver; all three arms
        // are told apart and `Unknown` panics.
        match self.0.check() {
            z3::SatResult::Sat => true,    // satresult-exempt: as above
            z3::SatResult::Unsat => false, // satresult-exempt: as above
            z3::SatResult::Unknown => panic!("Z3 returned Unknown"),
        }
    }
}
