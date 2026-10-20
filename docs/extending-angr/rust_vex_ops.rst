Adding a New VEX Operation (Rust Engine)
========================================

This page is the contributor walkthrough for extending the Rust engine's
VEX layer with a new operation. It complements
``docs/advanced-topics/rust_engine.rst`` (which covers the engine's
*runtime* surface) by describing the *lifting + execution* pipeline you
hit when you teach the engine a new VEX opcode.

Most of the engine's perf headroom over the Python engine comes from
keeping IR execution inside Rust. When pyvex hands the engine an
``Iop_*`` string it has never seen, that hot path collapses into a
``log::warn!("Unmapped VEX operation: {}")`` and an
``IROp::Unmapped(name)``, which dispatch turns into a hard
``RustUnsupportedVexOpError``. Adding a new op closes that gap.

Pipeline overview
-----------------

A single VEX op flows through the files below. Keep this picture in your
head as you read the worked examples below:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Role
   * - ``native/angr/src/vex/ir/ops_def.rs``
     - Defines the ``IROp`` enum (the engine's internal opcode set,
       parameterized by ``IRType`` width, which lives in
       ``native/angr/src/vex/ir/types.rs``). Both are re-exported via
       ``native/angr/src/vex/ir/mod.rs``. A new op gets a variant here.
   * - ``native/angr/src/vex/opcode_map.rs``
     - Translates pyvex's string opcodes (``"Iop_Add32"``) into
       ``IROp`` variants in ``parse_opcode``, which consults one
       ``parse_*`` sub-router per opcode family. This file holds the
       dispatcher, the width-family ``*_arms!`` macros, the integer
       families (``parse_arithmetic`` / ``parse_bitwise`` /
       ``parse_shift`` / ``parse_comparison`` / ``parse_conversion``)
       and ``parse_special``. Both lifter paths —
       pyvex strings and the native libVEX FFI — funnel through this
       one string-based ``parse_opcode``; there is no separate numeric
       mapping.
   * - ``native/angr/src/vex/opcode_map_float.rs``
     - Sibling module holding the FP sub-routers ``parse_float`` and
       ``parse_transcendental`` (angr-fs8kb.20).
   * - ``native/angr/src/vex/opcode_map_vector.rs``
     - Sibling module holding the SIMD sub-routers ``parse_vector``,
       ``parse_vreverse`` and ``parse_neon_unimplemented`` (angr-fs8kb.20).
       ``parse_vector`` is by far the largest of the
       family, which is what motivated the split.
   * - ``native/angr/src/vex/ops/mod.rs``
     - Implements the op. ``VEXOps::unop`` / ``binop`` / ``qop`` and the
       rounding-mode-aware ``VEXOps::unop_with_rm`` / ``binop_with_rm``
       / ``qop_with_rm`` dispatch on the ``IROp`` variant and produce a
       ``RustBV``. There is no ``triop`` entry point: an IR Triop is
       ``(rm, a, b)``, so ``eval_triop`` routes it through
       ``binop_with_rm``, which honors the rounding mode and otherwise
       delegates to ``binop``. A Qop is ``(rm, a, b, c)`` and gets the
       same treatment: ``eval_qop`` routes through ``qop_with_rm``,
       which delegates to ``qop`` only for a concrete
       round-to-nearest-ties-to-even rm. Concrete fast paths live next
       to their Z3 symbolic fallbacks.
   * - ``native/angr/src/interpreter/expressions_arith.rs``
     - The interpreter site that calls ``VEXOps::*`` (``eval_unop`` /
       ``eval_binop`` / ``eval_triop`` / ``eval_qop`` / ``eval_ccall``,
       plus the ``vex_op_fallback`` unsupported-op path). The dispatch
       entry point that routes to them stays in
       ``expressions.rs``, and the ``IRExpr::Load`` ladder you edit for
       *non-op* IR features (see examples 3 and 4) lives in
       ``expressions_loads.rs``.

The parameterization (``IROp::Add(IRType::I32)`` rather than 200
separate variants ``Iop_Add8`` / ``Iop_Add16`` / …) is deliberate; a
new arithmetic-shaped op should follow that pattern unless its
semantics differ per-width.

The ``parse_*`` family pattern
------------------------------

``parse_opcode`` is a chain of sub-routers each returning
``Option<IROp>``. Each sub-router groups one shape of opcode:

.. code-block:: rust

   pub fn parse_opcode(op_str: &str) -> IROp {
       if let Some(op) = parse_arithmetic(op_str)         { return op; }
       if let Some(op) = parse_bitwise(op_str)            { return op; }
       if let Some(op) = parse_shift(op_str)              { return op; }
       if let Some(op) = parse_comparison(op_str)         { return op; }
       if let Some(op) = parse_conversion(op_str)         { return op; }
       if let Some(op) = parse_float(op_str)              { return op; }
       if let Some(op) = parse_transcendental(op_str)     { return op; }
       if let Some(op) = parse_vector(op_str)             { return op; }
       if let Some(op) = parse_vreverse(op_str)           { return op; }
       if let Some(op) = parse_special(op_str)            { return op; }
       if let Some(op) = parse_neon_unimplemented(op_str) { return op; }
       log::warn!("Unmapped VEX operation: {}", op_str);
       IROp::Unmapped(intern_unmapped_op(op_str))
   }

The ``IROp::Unmapped(name)`` sentinel (angr-tkbr.2) keeps the original pyvex
opcode string so dispatch can surface ``RustUnsupportedVexOpError``
with a useful name instead of returning a fresh-symbolic value.

When you add a new op, the right sub-router is whichever one already
holds its conceptual siblings. Don't introduce a new sub-router unless
none of the existing eleven fit.

Worked example 1 — arithmetic: ``Add``
--------------------------------------

The minimum-friction case: a width-parameterized binary op whose
implementation already exists on ``RustBV``.

**ir/ops_def.rs** — the enum variant. ``Add`` lives in the ``IROp`` enum at
``native/angr/src/vex/ir/ops_def.rs`` (search ``pub enum IROp`` then the
``// Arithmetic`` group):

.. code-block:: rust

   pub enum IROp {
       // Arithmetic (parameterized by width)
       Add(IRType),
       Sub(IRType),
       Mul(IRType),
       // …
   }

**opcode_map.rs** — the string-to-variant entries. ``parse_arithmetic``
uses the ``tuple_arms!`` macro to expand one declaration into match
arms for every width pyvex actually emits:

.. code-block:: rust

   fn parse_arithmetic(op_str: &str) -> Option<IROp> {
       tuple_arms!(op_str; "Iop_Add" => Add
                   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
       tuple_arms!(op_str; "Iop_Sub" => Sub
                   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
       // …
       None
   }

The macro expands to exactly the literal ``match`` arms it would
otherwise enumerate by hand (``"Iop_Add8" => Some(IROp::Add(IRType::I8))``,
etc.); use it whenever the new op follows the ``Iop_<base><width>``
naming convention.

There is no second, numeric table to keep in sync. The native libVEX
FFI path converts its ``IROp`` discriminant back to the pyvex name
first (``vex/libvex_lifter.rs::c_op`` → upstream's ``ailment::vexop::op_name``)
and then calls the same ``parse_opcode``, so a string arm added once covers
both lifters. An opcode libVEX knows but ``op_name`` doesn't is
routed as ``"Iop_UNKNOWN_<n>"``, which keeps the numeric tag visible in
the ``Unmapped`` name.

**ops/mod.rs** — the implementation. The dispatch arm inside
``VEXOps::binop`` delegates to a ``RustBV`` method via the
``width_binop!`` macro:

.. code-block:: rust

   pub fn binop(op: IROp, left: RustBV, right: RustBV, ctx: &SymContext)
       -> Result<RustBV, OpError>
   {
       match op {
           IROp::Add(ty) => width_binop!(left, right, ty, add_into, ctx),
           // …
       }
   }

The macro asserts ``width(left) == width(right) == ty.bits()`` and
delegates to ``left.add_into(right, ctx)``. Symbolic vs. concrete is
the ``RustBV`` method's responsibility — the dispatch layer doesn't
care. That's why so many arithmetic arms are one-liners.

**Test.** A representative unit test, from ``ops/tests_core.rs`` — the
per-family test sibling for the core integer/logic ops (the ``tests_*.rs``
files are declared at the bottom of ``ops/mod.rs`` via
``test_submod!``; angr-9hleg):

.. code-block:: rust

   #[test]
   fn test_add_op() {
       let ctx = SymContext::new_mock();
       let a = RustBV::concrete(5, 32);
       let b = RustBV::concrete(3, 32);
       let result = VEXOps::binop(IROp::Add(IRType::I32), a, b, &ctx).unwrap();
       assert_eq!(result.as_u64(), Some(8));
   }

Worked example 2 — widening multiply: ``MullU`` / ``MullS``
-----------------------------------------------------------

Use this shape when an op needs custom logic the ``width_binop!``
macro can't express. Widening multiply is the canonical example: the
result is double the operand width.

**vex/ir/ops_def.rs** — ``MullU(IRType)`` / ``MullS(IRType)`` in the same
arithmetic group of ``IROp``.

**opcode_map.rs** — entries inside ``parse_arithmetic``:

.. code-block:: rust

   "Iop_MullS8"  => Some(IROp::MullS(IRType::I8)),
   "Iop_MullS16" => Some(IROp::MullS(IRType::I16)),
   // …
   "Iop_MullU64" => Some(IROp::MullU(IRType::I64)),

**ops/mod.rs** — the dispatch arm delegates to a *named helper* instead of
the macro:

.. code-block:: rust

   IROp::MullU(ty) => Self::widening_mul(left, right, ty, false, ctx),
   IROp::MullS(ty) => Self::widening_mul(left, right, ty, true, ctx),

**ops/int_arith.rs** — the helper itself. It is a ``pub(super)`` method on
``impl VEXOps`` in a child module of ``ops``, so the dispatch arm above
still reaches it as ``Self::widening_mul``:

.. code-block:: rust

   pub(super) fn widening_mul(left: RustBV, right: RustBV, ty: IRType,
                              signed: bool, ctx: &SymContext)
       -> Result<RustBV, OpError>
   {
       let in_width = ty.bits();
       let out_width = in_width * 2;

       // Extend both operands
       let left_ext = left.extend_into(out_width, signed, ctx);
       let right_ext = right.extend_into(out_width, signed, ctx);

       Ok(left_ext.mul_into(right_ext, ctx))
   }

Note the single ``extend_into(width, signed, ctx)`` call rather than an
``if signed { sign_extend_into } else { zero_extend_into }`` split: the
signedness is already a parameter here, so branching on it just to pick
between two ``RustBV`` methods duplicates a choice ``extend_into``
already makes.

Things to take away:

* Helpers belong on ``impl VEXOps`` — either in ``ops/mod.rs`` or, for a
  whole family, in a child module of ``ops`` declared with ``mod`` (as
  ``int_arith`` is), with the methods ``pub(super)`` so the dispatch
  arms can still call them. Keep them ``#[inline]`` and avoid taking
  ``&mut`` state — VEX ops are pure transforms over ``RustBV`` plus the
  solver context.
* When the helper needs both a concrete and a symbolic path (typical
  for FP and vector ops, see ``FloatLaneOp`` in ``ops/lane_traits.rs``),
  use the trait/struct-pair pattern: each implementation
  supplies *both* branches so the compiler stops you from forgetting
  one.
* The unit test — in ``ops/tests_int_arith.rs``, the sibling matching
  the source module — demonstrates both width and overflow behavior:

  .. code-block:: rust

     #[test]
     fn test_mul_widening() {
         let ctx = SymContext::new_mock();
         let a = RustBV::concrete(0xFFFFFFFF, 32);
         let b = RustBV::concrete(0xFFFFFFFF, 32);
         let result = VEXOps::binop(IROp::MullU(IRType::I32), a, b, &ctx).unwrap();
         assert_eq!(result.width(), 64);
         assert_eq!(result.as_u128(), Some(0xFFFFFFFE00000001));
     }

Worked example 3 — memory read: ``IRExpr::Load``
------------------------------------------------

``Load`` is not an ``IROp`` — it's an ``IRExpr`` variant in
``vex/ir/ast.rs``, and it lives in the **interpreter** layer rather than
``ops/mod.rs``. This is a frequent place contributors go looking in the
wrong file. The reason is that loads need access to the state's memory
plane, which ``VEXOps`` deliberately does not have (its inputs are
``RustBV`` plus a solver context — nothing more).

The dispatch site is in
``native/angr/src/interpreter/expressions.rs`` (search
``IRExpr::Load {``); the arm itself only delegates to the
``eval_load`` helper in ``expressions_loads.rs``, which does the work:

.. code-block:: rust

   IRExpr::Load { addr, ty, endness } => {
       self.eval_load(callbacks, addr, *ty, *endness, tyenv)
   }

   // … in `eval_load`:
   let addr_val = self.eval_expr_with_callbacks(callbacks, addr, tyenv)?;
   let size = ty.bytes() as usize;
   // … profiling counter increment …
   // hand off to the symbolic memory plane on `self.state`

If you're adding a *new* memory-access shape (e.g. a guarded load,
load-linked, gather), the corresponding ``IRStmt`` / ``IRExpr``
variant goes into ``vex/ir/ast.rs``, the lifter wiring goes into
``callbacks/pyvex_reader.rs`` (pyvex-object side) and ``vex/libvex_lifter.rs``
(native side), and the *execution* goes into ``interpreter`` — not
``ops/mod.rs``. The split is durable: pure value-to-value transforms are
in ``ops/mod.rs``; anything that touches memory, registers, temps,
constraints, or call frames goes through ``interpreter``.

Worked example 4 — memory write: ``IRStmt::Store``
--------------------------------------------------

Same story as ``Load`` but on the statement side (search
``IRStmt::Store {`` in ``vex/ir/ast.rs``):

.. code-block:: rust

   pub enum IRStmt {
       // …
       Store { addr: IRExpr, data: IRExpr, endness: Endness },
       // …
   }

The dispatch site is in
``native/angr/src/interpreter/statements.rs`` (search
``IRStmt::Store {``):

.. code-block:: rust

   IRStmt::Store { addr, data, endness } => {
       let addr_val = self.eval_expr_with_callbacks(callbacks, addr, &irsb.tyenv)?;
       let mut data_val = self.eval_expr_with_callbacks(callbacks, data, &irsb.tyenv)?;
       let data_size = data_val.width().div_ceil(8) as usize;
       // … profiling, `mem_write` inspect dispatch, callback, store …
   }

A new store-shaped statement (CAS, LL/SC, guarded store) goes the same
way: variant in ``vex/ir/ast.rs``, wiring in the lifter, execution in
``interpreter/statements.rs``. If the op is *also* width-parameterized
(e.g., the CAS payload is an ``IROp::Add``), the body still calls
``VEXOps::binop`` — which is exactly how the two layers compose.

Worked example 5 — adding a brand-new IROp end-to-end
-----------------------------------------------------

Putting the pieces together. Suppose pyvex starts emitting
``Iop_PopCnt8`` (a hypothetical 8-bit population count) and the engine
doesn't know it yet. The end-to-end recipe:

1. **Pick the family.** ``PopCount`` already exists in ``IROp`` —
   so this is just a new width, not a new variant. If your op has no
   conceptual sibling, add a new variant.

2. **Add the enum variant (if new).** Append to ``IROp`` in
   ``vex/ir/ops_def.rs``. Group it with its semantic neighbors and pick the
   parameterization (``IRType`` for width, or a struct field for
   things like ``Extract { from, to, low_bit }``).

3. **Map the string opcode.** Find the right ``parse_*`` sub-router — in
   ``opcode_map.rs`` for the integer and special families, in
   ``opcode_map_float.rs`` or ``opcode_map_vector.rs`` for the FP and
   SIMD ones — and add the ``"Iop_FooN" => Some(IROp::Foo(...))``
   entries for every width pyvex actually emits. If you don't know
   which widths are real, run pyvex against a sample binary and grep
   the lifted IRSB.

4. **Nothing to do for the FFI lifter.** The libVEX FFI path maps its
   numeric opcode back to the pyvex name before calling
   ``parse_opcode`` (``vex/libvex_lifter.rs::c_op``), so step 3 already
   covers it. The discriminant-to-name table is upstream's
   ``ailment::vexop_names::IROP_NAMES``.

5. **Implement.** Add the dispatch arm to ``VEXOps::unop`` /
   ``binop`` / ``qop`` in ``ops/mod.rs`` — or to
   ``unop_with_rm`` / ``binop_with_rm`` / ``qop_with_rm`` when the op
   consumes a VEX rounding mode (every IR Triop does, and so does an
   FMA-shaped Qop; see the *Pipeline overview* note above). Wiring a
   new rm-consuming Qop into ``qop`` alone silently computes
   round-to-nearest-ties-to-even for every guest rounding mode — the
   exact regression ``qop_with_rm`` was added to fix (angr-03vl4.30),
   and what ``vex/ops/tests_rounding_mode_sweep.rs`` exists to catch.
   Prefer the ``width_unop!`` / ``width_binop!`` macros for shapes where
   the op is a one-liner on ``RustBV``; promote to a named helper when the
   logic doesn't fit on a single line. If both a concrete and a
   symbolic path are needed, follow the ``FloatLaneOp`` trait pattern
   so the compiler enforces parity.

6. **Test.** Add one happy-path test plus an edge case (overflow,
   width=0, signed/unsigned boundary, symbolic fallback — whatever
   applies). Put it in the ``ops/tests_<family>.rs`` sibling that
   matches the source module you touched (``tests_core.rs``,
   ``tests_int_arith.rs``, ``tests_vec_shift.rs``, …), declared from
   ``ops/mod.rs`` with ``test_submod!(...)`` — use the
   ``test_submod!(z3 ...)`` form when the file needs a real Z3 context.
   Shared SIMD lane-test helpers live in ``ops/test_helpers.rs``. There
   is still no separate test crate for VEX ops: every ``tests_*.rs`` is
   a child module of ``ops``, so ``use super::*`` reaches its private
   items.

7. **Build and run.** ``cargo check --manifest-path
   native/angr/Cargo.toml --release`` for the fast loop;
   ``pip install -e . --no-build-isolation --no-deps`` to refresh the
   ``.so`` (or ``make rebuild``); then
   ``python -m pytest tests/engines/rust/`` to
   confirm nothing downstream regressed.

If at any point the lifter wiring is involved (e.g. you're adding an op
whose pyvex name is *not* a simple ``Iop_<name><width>``), look at
``callbacks/pyvex_reader.rs`` and ``vex/libvex_lifter.rs``: the former reads the
``IRSB`` the Python ``lift_block`` callback returns, the latter marshals the
one native lifting gets from C-side libVEX (struct mirrors in upstream's
``ailment/vex_ffi.rs``).

Symbolic universality tests
---------------------------

Concrete unit tests cover one input; a
*universality* test proves the symbolic encoding correct for all inputs by
asserting ``got != reference`` is UNSAT (``ctx.push()``,
``ctx.add_constraint(got.to_z3_ast().eq(reference.to_z3_ast()).not())``,
``assert!(!ctx.is_sat())``, ``ctx.pop()``). Pick the reference by this ladder:

1. **Python parity.** If ``angr/engines/vex/claripy/irop.py`` defines
   ``_op_Iop_X`` / ``_op_generic_X``, rebuild that exact pattern from
   ``RustBV::extract`` + ``VEXOps::concat_le_elements``. Claripy's ``arg[127:96]``
   maps to ``extract(127, 96, &ctx)``; ``concat_le_elements`` index 0 is the
   LSB, so push slices LSB to MSB.
2. **Sibling-op identity** (preferred when the op is algebraically related to
   ops already covered). Assert an identity tying it to two or more existing
   dispatch arms, e.g. ``VMull_lane == concat(VMulHi_lane, VMul_lane)``
   (``test_vec_mulhi_symbolic_matches_mull_split``). The reference shares no
   index arithmetic with the code under test. Look for hi/lo splits,
   widen-then-narrow round trips, add/sub inverses.
3. **Spec replay.** Otherwise inline a reference built from the same ``RustBV``
   primitives. It catches wrong caps, swapped then/else, off-by-one extracts and
   sign confusion, but not a bug in the spec itself.
4. **Opposite-numbering ISA replay.** When a vendor manual specifies the op in
   a different bit/byte numbering (PPC: bit 0 = MSB), write the reference in the
   manual's numbering so the two derivations share no index arithmetic
   (``vgbbd_be_reference``). If the permutation is an involution (transposes,
   full reversals), the numbering choice provably cannot change the result.

For IR rewrite rules in ``symbolic::value`` with both a construction-time and a
Z3-emission hook, pin every symbolic to concrete values and compare
``ctx.eval()`` results; structural tree assertions miss byte-order bugs.

Always mutation-check identity and replay tests: temporarily break the op and
confirm the test fails, since an identity that also holds for the buggy version
is vacuous.

What *not* to do
----------------

* Don't add a new sub-router to ``parse_opcode`` for a single op. If
  it doesn't fit any of the eleven, the op probably belongs in
  ``parse_special``.
* Don't reach for ``IROp::Unmapped`` (the "unmapped sentinel") in
  production code paths. ``Unmapped`` exists to surface
  ``RustUnsupportedVexOpError`` when ``parse_opcode`` can't match;
  treating it as an escape hatch hides bugs from contributors who
  actually want to map the op. ``IROp::Raw(u32)`` is not an escape
  hatch either: it is reserved for the x87 / FRECPX transcendentals
  routed by ``parse_transcendental``, whose ``u32`` is an internal tag
  keyed to the ``IOP_*`` consts in ``transcendentals.rs``.
* Don't bypass the width parameterization just to ship faster.
  ``IROp::Foo32`` / ``IROp::Foo64`` separately is exactly the
  proliferation libvex pays for; the Rust engine's terseness is
  earned by *not* doing that.
* Don't put memory or register access in ``ops/mod.rs``. The split between
  ``ops/mod.rs`` (pure ``RustBV`` transforms) and ``interpreter`` (state
  access) is the engine's most useful internal boundary; preserving it
  keeps the test surface small (``SymContext::new_mock()`` is enough
  to test any pure op).

Unsupported op coverage matrix
------------------------------

At-a-glance status for op families that have historically been
placeholders. *Implemented* means a dispatch arm exists in
``native/angr/src/vex/ops/mod.rs`` and the opcode parses to a concrete
``IROp`` variant (not ``IROp::NeonUnimplemented`` or
``IROp::Unmapped``). *Placeholder* means the opcode parses but
dispatch returns ``OpError::UnsupportedNeon``, which the engine
surfaces as ``RustUnsupportedVexOpError``. *Stubbed-symbolic* means
dispatch returns a fresh-symbolic value of the expected width — used
for ops whose semantics are too expensive or under-specified to
model (e.g. ``URECPE`` / ``URSQRTE`` per-lane reciprocal estimates).

A fourth, quieter status — *parse-succeeds / dispatch-fabricates
(BYPASS)* — is documented in its own subsection at the end of this
matrix. It covers opcodes that parse to a concrete ``IROp`` variant
but have no dispatch arm. By default a symbolic operand routes the block
to Python and a concrete one surfaces ``RustUnsupportedVexOpError``;
fabricating a fresh symbolic happens only under the opt-in
``ANGR_RUST_FABRICATE_UNSUPPORTED_IROP`` escape hatch. The known
families (``Pclmul*`` / ``Crc32C``) route to Python for
concrete and symbolic operands alike.

Source of truth: ``native/angr/src/vex/opcode_map_vector.rs``
(``parse_neon_unimplemented`` is the remaining placeholder list) and
``native/angr/src/vex/ops/mod.rs`` (the dispatch arms). Update the
matching row in the same change that adds or graduates an op; the
provenance column names the ``IROp`` variant that implements each family.

NEON op families
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Op family
     - Status
     - Provenance
   * - Saturating add / sub (``Iop_QAdd*`` / ``Iop_QSub*``)
     - Implemented
     - ``IROp::VQAdd`` / ``IROp::VQSub`` (angr-tukg.1)
   * - Pairwise integer (``Iop_PwAdd*`` / ``Iop_PwAddL*`` /
       ``Iop_PwMin*`` / ``Iop_PwMax*``)
     - Implemented
     - ``IROp::VPwAdd`` / ``VPwAddL`` / ``VPwMin`` / ``VPwMax`` (angr-tukg.2)
   * - Pairwise FP (``Iop_PwAdd32Fx2``)
     - Implemented
     - ``IROp::VFPwAdd`` via ``parse_float`` (angr-cudgw.6) — the last
       family to graduate out of ``parse_neon_unimplemented``
   * - Rounding halving add (``Iop_Avg{N}{S/U}x{M}``)
     - Implemented
     - ``IROp::VAvg`` (angr-tukg.3)
   * - Byte / halfword / word / bit reversal within lane
       (``Iop_Reverse{N}sIn{M}_x{K}``)
     - Implemented
     - ``IROp::VReverse`` via ``parse_vreverse`` (angr-tukg.4)
   * - FP reciprocal estimates (``Iop_RecipEst{,S}*`` /
       ``Iop_RecipStep*`` / ``Iop_RSqrtEst{,S}*`` /
       ``Iop_RSqrtStep*``)
     - Implemented
     - ``IROp::VFRecipEst{,S}`` / ``VFRecipStep`` /
       ``VFRSqrtEst{,S}`` / ``VFRSqrtStep`` (angr-iyon)
   * - Integer reciprocal estimates (``Iop_RecipEst32Ux{2,4}`` —
       URECPE; ``Iop_RSqrtEst32Ux{2,4}`` — URSQRTE)
     - Stubbed-symbolic
     - ``IROp::VIRecipEst`` / ``IROp::VIRSqrtEst`` — fresh-symbolic
       per lane (angr-tukg.5)
   * - Polynomial multiply (``Iop_PolynomialMul8x{8,16}`` /
       ``Iop_PolynomialMull8x8``)
     - Implemented
     - ``IROp::VPolynomialMul`` (angr-tukg.6)
   * - Per-lane bitcount (``Iop_Cnt8x{8,16}`` /
       ``Iop_Clz{N}x{M}`` / ``Iop_Cls{N}x{M}``)
     - Implemented
     - ``IROp::VCnt`` / ``VClz`` / ``VCls`` (angr-tukg.6)
   * - High half of the widening multiply
       (``Iop_MulHi{8,16,32}{U,S}x{4,8,16}``)
     - Implemented
     - ``IROp::VMulHi`` (angr-0jh0j.61). The ARM doubling variants
       ``Iop_QDMulHi*`` / ``Iop_QRDMulHi*`` share the infix but are a
       different op and remain ``Unmapped``
   * - Vector shift by vector (``Iop_Shl/Shr/Sar/Sal{N}x{M}``)
     - Implemented
     - ``IROp::VShl`` / ``VShr`` / ``VSar`` — ``Sal`` aliased to
       ``VShl`` (angr-tukg.7)
   * - NEON saturating shift-left by vector
       (``Iop_QShl{N}x{M}`` / ``Iop_QSal{N}x{M}``)
     - Implemented
     - ``IROp::VQShlSat`` (angr-tukg.8)
   * - NEON saturating shift-by-immediate (``Iop_QShlN*``)
     - Placeholder
     - No enum variant or dispatch arm; parses to ``IROp::Unmapped``
       (listed in ``KNOWN_UNMAPPED_GROUPS`` in ``opcode_map_tests.rs``)
   * - Integer lane compare (``Iop_CmpEQ{N}x{M}`` /
       ``Iop_CmpGT{N}{S,U}x{M}``)
     - Implemented
     - ``IROp::VCmpEQ`` / ``IROp::VCmpGT { signed }`` via the
       ``ICmpEq`` / ``ICmpGt`` ``IntLaneOp`` impls. The unsigned
       ``CmpGT`` half (NEON ``VCGT.U*``) is mapped too (angr-9ke6b.160). libVEX defines no ``Iop_CmpGT64Ux1``, so the
       D-reg 64-bit lane is absent by design.

``Iop_QShlN*`` is the only NEON family in this table without a
handler. ``parse_neon_unimplemented`` claims nothing at all and is kept
only as the scaffold point for the next NEON gap. No tracked benchmark
drives a symbolic path through ``QShlN``.

x87 transcendental ops
^^^^^^^^^^^^^^^^^^^^^^

These ops have no named ``IROp`` variant. ``parse_transcendental``
(the sub-router right after ``parse_float``) maps the ten opcode names
to ``IROp::Raw(tag)``, and the ``IROp::Raw`` arms of ``VEXOps::binop``
(via its private ``binop_misc`` helper) and ``VEXOps::binop_with_rm``
dispatch that into the concrete-only libm fast path in
``native/angr/src/vex/transcendentals.rs`` plus a symbolic
concretization fallback (sample-pin-replace). The ``tag`` values are
the libvex ``Iop_*`` discriminants, but since ``parse_transcendental``
is the sole producer of ``IROp::Raw`` they are now just internal
tokens — they only have to agree with the ``IOP_*`` consts in
``transcendentals.rs``, not with libvex's numbering.

.. note::

   These ops reach ``parse_transcendental`` only through the string
   router, which the libVEX FFI lifter also goes through; there is no
   numeric-opcode route to them (``angr-h0ur``, ``angr-9ke6b.233``).

Opcode names below are libvex (matches what pyvex's numeric-to-string
table emits).

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Op family
     - Status
     - Provenance
   * - Trig (``Iop_SinF64``, ``Iop_CosF64``, ``Iop_TanF64``,
       ``Iop_AtanF64``)
     - Concrete + concretize-and-pin via ``Raw``
     - ``transcendentals.rs`` libm path (angr-i5lj.2)
   * - Log / exp (``Iop_Yl2xF64``, ``Iop_Yl2xp1F64``,
       ``Iop_2xm1F64``, ``Iop_ScaleF64``)
     - Concrete + concretize-and-pin via ``Raw``
     - ``transcendentals.rs`` libm path (angr-i5lj.1)
   * - ARM AArch64 FRECPX (``Iop_RecpExpF64``, ``Iop_RecpExpF32``)
     - Concrete via ``Raw`` (closed form, no concretize fallback)
     - ``transcendentals.rs`` exponent-only closed form

The x87 transcendentals only matter on workloads that drive a
symbolic path through them. ``securityfest_fairlight`` and
``ekopartyctf2016_sokohashv2`` both hit them in the original binary
but the test drivers hook them out at the Python layer, so the
matrix above does not yet block any tracked benchmark.

FP and decimal conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Wide / decimal / fixed-point FP conversions that pyvex emits for
x86 SSE / AVX, PowerPC DFP, and ARM SIMD scalar conversions but
that ``parse_float`` / ``parse_conversion`` do not yet handle.
All route to ``IROp::Unmapped(name)`` →
``RustUnsupportedVexOpError`` on dispatch.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Op family
     - Status
     - Provenance
   * - 128-bit FP / decimal (``Iop_F128toD32``,
       ``Iop_F128toI128S``)
     - Placeholder
     - No parse arm in ``parse_float`` / ``parse_conversion``
   * - SSE4.1 round-to-int (``Iop_RoundF32x4_RM`` /
       ``_RN`` / ``_RP`` / ``_RZ``)
     - Placeholder
     - No parse arm in ``parse_vector``
   * - PowerPC DFP significance round
       (``Iop_SignificanceRoundD64`` / ``D128``)
     - Placeholder
     - No parse arm in ``parse_float``
   * - FP↔fixed conversions (``Iop_F32ToFixed32S*`` /
       ``Iop_Fixed32SToF32x*``)
     - Placeholder
     - No parse arm in ``parse_conversion``

These are reachable from PowerPC, x86 SSE4.1, and ARM SIMD code.
No tracked benchmark drives a symbolic path through them.

Crypto and polynomial multiply
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARM AES / SHA / NEON polynomial-multiply-accumulate ops. Same
provenance as the FP-conversion family — no parse arm, so they
route to ``IROp::Unmapped(name)``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Op family
     - Status
     - Provenance
   * - Polynomial multiply-accumulate
       (``Iop_PolynomialMulAdd{8x16,16x8,32x4,64x2}``)
     - Placeholder
     - No parse arm in ``parse_vector`` — distinct from
       ``Iop_PolynomialMul*`` which IS implemented as
       ``IROp::VPolynomialMul`` (angr-tukg.6)
   * - AES round (``Iop_CipherV128``, ``Iop_NCipherV128``)
     - Placeholder
     - No parse arm in ``parse_special``
   * - SHA-2 round (``Iop_SHA256``, ``Iop_SHA512``)
     - Placeholder
     - No parse arm in ``parse_special``

ARM crypto extension binaries (firmware, AArch64 TLS/IPSec code)
would hit these. None of the current corpus does — when one
lands, prefer routing AES / SHA through SimProcedure hooks over
adding native dispatch arms (the per-round implementations are
large and Z3-hostile).

Parse-succeeds / dispatch-fabricates (silent BYPASS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The families above are *parse-time* gaps: the opcode never reaches a
concrete ``IROp`` variant, so dispatch surfaces
``RustUnsupportedVexOpError`` (loud) or ``IROp::NeonUnimplemented``.
This subsection covers a categorically different and quieter gap —
opcodes that **do** parse to a concrete ``IROp`` variant but have **no
dispatch arm** in ``ops/mod.rs``. They fall through the per-family
sub-router to its catch-all (``OpError::NotUnary`` /
``NotBinary`` / ``NotQuaternary`` — there is no ``NotTernary``,
matching the absence of a ``triop`` entry point), and the
``eval_unop`` / ``eval_binop`` / ``eval_triop`` / ``eval_qop``
dispatchers in ``interpreter/expressions_arith.rs`` hand the error to
``VEXInterpreter::vex_op_fallback``, which splits on operand
concreteness:

- **Concrete args** → the typed ``OpError`` propagates and surfaces as
  ``RustUnsupportedVexOpError`` (no silently-wrong value; angr-sa3j).
- **Symbolic args** → by default the block is routed to Python's VEX
  engine via ``CbExecutionError::NeedPythonFallback`` (angr-oyzvj).
  Python is the reference implementation for the float / vector-float
  conversions that land here, so the result keeps its input→output
  relationship.

Both arms bump ``python_vex_op_fallback_count`` and the per-arm
``python_vex_{unop,binop,triop,qop}_fallback_count`` in
``ExecutionStats`` (``mgr.get_execution_stats()``).

**The opt-in fabricate escape hatch.** Setting the environment variable
``ANGR_RUST_FABRICATE_UNSUPPORTED_IROP`` to any non-empty value other
than ``0`` (read once per process; see ``fabricate_unsupported_irop`` in
``interpreter/expressions_arith.rs``) restores the older behaviour for
the symbolic arm: a fresh ``RustBV::symbolic("unsup_unop_<pc>" /
"unsup_binop_<pc>" / …)`` of the result width stands in for the real
value. This is the **BYPASS**: it loses the input→output relationship
entirely, so any downstream condition over the value explores both
branches unconstrained. The same gate covers the condition-flag arm of
``VEXInterpreter::eval_ccall`` and the no-handler-anywhere arm of
``VEXInterpreter::handle_dirty_call``. Every fabrication bumps
``vex_bypass_fabricate_count``, which therefore reads zero unless the
variable is set. The name is deliberately ``FABRICATE_*`` rather than
``BYPASS_*``: the ``BYPASS_UNSUPPORTED_IROP`` SimOption means the
opposite (route to Python's resilience mixin).

The families below are matched before ``vex_op_fallback``:
``eval_binop`` recognises them (``is_dispatch_fabricate_family``) and
routes the block to Python's VEX engine via
``CbExecutionError::NeedPythonFallback`` (reason ``DISPATCH_FABRICATE_REASON``)
for *both* concrete and symbolic args (angr-s6miz). Because they route
before the fabricate gate, they never increment
``vex_bypass_fabricate_count``, even with the escape hatch set.

These opcodes parse (``opcode_map.rs``) but have no arm in dispatch
(``ops/mod.rs``); they are the families the Python-fallback route covers:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Op family
     - Classification
     - Provenance / fix path
   * - x86 carry-less multiply
       (``Iop_PclmulLQLQ`` / ``HQHQ`` / ``LQHQ`` / ``HQLQ``)
     - **must-fallback**
     - Parse to ``IROp::Pclmul*`` (search ``"Iop_PclmulLQLQ"`` in
       ``opcode_map.rs``);
       ``iropclass`` files them under ``Arith`` but ``binop``'s Arith
       arm does not list them, so they reach ``binop_misc`` →
       ``NotBinary``. Deterministic GF(2) product — must compute or
       fall back, never fabricate.
   * - x86 SSE4.2 CRC32
       (``Iop_Crc32C``)
     - **must-fallback**
     - Parses to ``IROp::Crc32C`` (search ``"Iop_Crc32C"`` in
       ``opcode_map.rs``); same
       ``Arith``-classified-but-undispatched path as ``Pclmul*``.
       Deterministic checksum.

Both are **must-fallback** (deterministic functions of their
inputs) — none qualify as *fabricate-ok*. The only legitimately
*fabricate-ok* ops are the under-specified estimates already handled
as **Stubbed-symbolic** by deliberate policy (``URECPE`` / ``URSQRTE``
/ FP ``RecipEst`` — fresh-symbolic per lane is the correct model, and
angr Python does the same). Every other concrete ``IROp`` variant
either has a dispatch arm (``Raw`` included — it routes into
``transcendentals.rs``) or is a sentinel (``NeonUnimplemented`` /
``Unmapped``), so these families are the whole
parse-succeeds / no-dispatch-arm surface.

No tracked benchmark drives a *symbolic* path through them (x86 crypto
code is rare in the CTF corpus and usually hooked at the Python layer).

The lane permutes (``Iop_Perm8x{8,16}``, ``Iop_Perm32x{4,8}`` and the
``Iop_PermOrZero8x{8,16}`` form libVEX lifts x86 ``PSHUFB`` to) left this
list for a native arm, ``VEXOps::vec_perm``, which follows Python's
``_op_generic_Perm`` (angr-mx00p.6). ``Iop_Perm8x16x2`` (two-table AArch64
``TBL``) stays unmapped: it is a triop, and ``_op_generic_Perm`` is
two-argument.

Special expressions: VECRET / GSPTR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``IRExpr::VECRET`` (vector-call return-value holder) and
``IRExpr::GSPTR`` (guest-state pointer used by helper functions)
are emitted by VEX only on dirty-helper call edges. The Rust
interpreter routes both to a Python VEX fallback via
``CbExecutionError::NeedPythonFallback`` rather than implementing
native handlers
(the ``IRExpr::VECRET | IRExpr::GSPTR`` arm of
``native/angr/src/interpreter/expressions.rs::eval_expr_with_callbacks_inner``
— the public ``eval_expr_with_callbacks`` wrapper only adds profiling and
``inspect`` dispatch before delegating to it).
The error reason carries the shared ``VECRET_GSPTR_REASON`` marker
from ``native/angr/src/interpreter/execution_error.rs``; the manager scans
fallback reasons in ``exploration::run_loop_single::callback_event`` and bumps
``vecret_gsptr_fallback_count`` so future regressions are
visible via ``mgr.stats`` / ``mgr._rust_mgr.get_fallback_stats()``.

On the benchmark corpus the counter reads zero (``angr-2iow``): the path
is reachable in principle but inert under those workloads, so it stays
a documented fallback rather than a native handler. Watch the counter on any new
SIMD-call-heavy or kernel-helper bench (e.g. NEON intrinsic
code, ARM TLS helpers) — non-zero readings re-open this work.

Catching new placeholders
^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter ``RustUnsupportedVexOpError("Iop_<name>", arch)``
on a new workload:

1. ``grep "Iop_<name>"`` under ``native/angr/src/vex/`` to confirm
   it has no parse arm. (If it does, the missing piece is a dispatch
   arm in ``ops/mod.rs`` — see the "Pipeline overview" section above.)
2. If it has no parse arm, decide which sub-router it belongs in
   (``parse_float`` in ``opcode_map_float.rs``, ``parse_vector`` in
   ``opcode_map_vector.rs``, the integer families in ``opcode_map.rs``)
   and add it there.
3. Add a row to the matrix above with status ``Placeholder`` (or
   ``Implemented`` plus the implementing ``IROp`` variant, if you add the
   handler in the same change).
