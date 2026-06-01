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
``log::warn!("Unmapped VEX operation: {}")`` and an ``IROp::Raw(0)``,
which downstream code treats as a hard error. Adding a new op closes
that gap.

Pipeline overview
-----------------

A single VEX op flows through four files. Keep this picture in your
head as you read the worked examples below:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Role
   * - ``native/angr/src/vex/ir.rs``
     - Defines the ``IROp`` enum (the engine's internal opcode set,
       parameterized by ``IRType`` width). A new op gets a variant
       here.
   * - ``native/angr/src/vex/opcode_map.rs``
     - Translates pyvex's string opcodes (``"Iop_Add32"``) into
       ``IROp`` variants in ``parse_opcode`` and its
       ``parse_arithmetic`` / ``parse_bitwise`` / ``parse_shift`` /
       ``parse_comparison`` / ``parse_conversion`` / ``parse_float`` /
       ``parse_vector`` / ``parse_special`` sub-routers. Also has a
       numeric variant ``parse_opcode_from_u32`` for the native FFI
       path.
   * - ``native/angr/src/vex/ops.rs``
     - Implements the op. ``VEXOps::unop`` / ``binop`` / ``triop`` /
       ``qop`` dispatch on the ``IROp`` variant and produce a
       ``RustBV``. Concrete fast paths live next to their Z3 symbolic
       fallbacks.
   * - ``native/angr/src/interpreter/expressions.rs``
     - The interpreter site that calls ``VEXOps::*``. You only edit
       this file for *non-op* IR features (``IRExpr::Load``,
       ``IRStmt::Store``, etc. — see examples 3 and 4).

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
       if let Some(op) = parse_arithmetic(op_str) { return op; }
       if let Some(op) = parse_bitwise(op_str)    { return op; }
       if let Some(op) = parse_shift(op_str)      { return op; }
       if let Some(op) = parse_comparison(op_str) { return op; }
       if let Some(op) = parse_conversion(op_str) { return op; }
       if let Some(op) = parse_float(op_str)      { return op; }
       if let Some(op) = parse_vector(op_str)     { return op; }
       if let Some(op) = parse_special(op_str)    { return op; }
       if let Some(op) = parse_neon_unimplemented(op_str) { return op; }
       log::warn!("Unmapped VEX operation: {}", op_str);
       IROp::Raw(0)
   }

When you add a new op, the right sub-router is whichever one already
holds its conceptual siblings. Don't introduce a new sub-router unless
none of the existing nine fit.

Worked example 1 — arithmetic: ``Add``
--------------------------------------

The minimum-friction case: a width-parameterized binary op whose
implementation already exists on ``RustBV``.

**ir.rs** — the enum variant. ``Add`` lives at
``native/angr/src/vex/ir.rs:520``:

.. code-block:: rust

   pub enum IROp {
       // Arithmetic (parameterized by width)
       Add(IRType),
       Sub(IRType),
       Mul(IRType),
       // …
   }

**opcode_map.rs** — the string-to-variant entries. ``parse_arithmetic``
(``opcode_map.rs:47``) has four lines per width:

.. code-block:: rust

   fn parse_arithmetic(op_str: &str) -> Option<IROp> {
       match op_str {
           "Iop_Add8"  => Some(IROp::Add(IRType::I8)),
           "Iop_Add16" => Some(IROp::Add(IRType::I16)),
           "Iop_Add32" => Some(IROp::Add(IRType::I32)),
           "Iop_Add64" => Some(IROp::Add(IRType::I64)),
           // …
       }
   }

**opcode_map.rs (FFI)** — the numeric mapping. ``parse_opcode_from_u32``
mirrors the libvex enum order (``Iop_INVALID = 0x1400`` + offset). ``Add``
sits at ``0x1401`` (I8) through ``0x1404`` (I64). If your new op already
has an upstream libvex enum value, slot it in here too.

**ops.rs** — the implementation. The dispatch arm
(``ops.rs:394``) delegates to a ``RustBV`` method via the
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

**Test.** A representative unit test from ``ops.rs:2950``:

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

**ir.rs** — ``MullU(IRType)`` / ``MullS(IRType)`` (``ir.rs:523-524``).

**opcode_map.rs** — entries inside ``parse_arithmetic``:

.. code-block:: rust

   "Iop_MullS8"  => Some(IROp::MullS(IRType::I8)),
   "Iop_MullS16" => Some(IROp::MullS(IRType::I16)),
   // …
   "Iop_MullU64" => Some(IROp::MullU(IRType::I64)),

**ops.rs** — the dispatch arm delegates to a *named helper* instead of
the macro:

.. code-block:: rust

   IROp::MullU(ty) => Self::widening_mul(left, right, ty, false, ctx),
   IROp::MullS(ty) => Self::widening_mul(left, right, ty, true,  ctx),

   fn widening_mul(left: RustBV, right: RustBV, ty: IRType, signed: bool,
                   ctx: &SymContext) -> Result<RustBV, OpError> {
       let in_width = ty.bits();
       let out_width = in_width * 2;
       let (left_ext, right_ext) = if signed {
           ( left.sign_extend_into(out_width, ctx),
             right.sign_extend_into(out_width, ctx) )
       } else {
           ( left.zero_extend_into(out_width, ctx),
             right.zero_extend_into(out_width, ctx) )
       };
       Ok(left_ext.mul_into(right_ext, ctx))
   }

Things to take away:

* Helpers belong on ``impl VEXOps`` (or as free functions in the same
  file). Keep them ``#[inline]`` and avoid taking ``&mut`` state — VEX
  ops are pure transforms over ``RustBV`` plus the solver context.
* When the helper needs both a concrete and a symbolic path (typical
  for FP and vector ops, see ``FloatLaneOp`` in ``ops.rs:43``), use
  the trait/struct-pair pattern: each implementation supplies *both*
  branches so the compiler stops you from forgetting one.
* The unit test demonstrates both width and overflow behavior
  (``ops.rs:2961``):

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

``Load`` is not an ``IROp`` — it's an ``IRExpr`` variant
(``ir.rs:270``), and it lives in the **interpreter** layer rather than
``ops.rs``. This is a frequent place contributors go looking in the
wrong file. The reason is that loads need access to the state's memory
plane, which ``VEXOps`` deliberately does not have (its inputs are
``RustBV`` plus a solver context — nothing more).

The dispatch site is in
``native/angr/src/interpreter/expressions.rs:49``:

.. code-block:: rust

   IRExpr::Load { addr, ty, .. } => {
       let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, tyenv)?;
       let size = ty.bytes() as usize;
       // … profiling counter increment …
       // hand off to the symbolic memory plane on `self.state`
   }

If you're adding a *new* memory-access shape (e.g. a guarded load,
load-linked, gather), the corresponding ``IRStmt`` / ``IRExpr``
variant goes into ``ir.rs``, the lifter wiring goes into
``vex/pyvex_bridge.rs`` (string side) and ``vex/libpyvex_ffi.rs``
(native side), and the *execution* goes into ``interpreter`` — not
``ops.rs``. The split is durable: pure value-to-value transforms are
in ``ops.rs``; anything that touches memory, registers, temps,
constraints, or call frames goes through ``interpreter``.

Worked example 4 — memory write: ``IRStmt::Store``
--------------------------------------------------

Same story as ``Load`` but on the statement side
(``ir.rs:175-183``):

.. code-block:: rust

   pub enum IRStmt {
       // …
       Store { addr: IRExpr, data: IRExpr, endness: Endness },
       // …
   }

The dispatch site is ``interpreter/statements.rs:54``:

.. code-block:: rust

   IRStmt::Store { addr, data, .. } => {
       let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
       let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
       let data_size = ((data_val.width() + 7) / 8) as usize;
       // … profiling, callback, store …
   }

A new store-shaped statement (CAS, LL/SC, guarded store) goes the same
way: variant in ``ir.rs``, wiring in the lifter, execution in
``interpreter/statements.rs``. If the op is *also* width-parameterized
(e.g., the CAS payload is an ``IROp::Add``), the body still calls
``VEXOps::binop`` — which is exactly how the two layers compose.

Worked example 5 — adding a brand-new IROp end-to-end
-----------------------------------------------------

Putting the pieces together. Suppose pyvex starts emitting
``Iop_PopCnt8`` (a hypothetical 8-bit population count) and the engine
doesn't know it yet. The end-to-end recipe:

1. **Pick the family.** ``PopCount`` already exists at ``ir.rs:588`` —
   so this is just a new width, not a new variant. If your op has no
   conceptual sibling, add a new variant.

2. **Add the enum variant (if new).** Append to ``IROp`` in
   ``ir.rs``. Group it with its semantic neighbors and pick the
   parameterization (``IRType`` for width, or a struct field for
   things like ``Extract { from, to, low_bit }``).

3. **Map the string opcode.** Find the right ``parse_*`` sub-router in
   ``opcode_map.rs`` and add the ``"Iop_FooN" => Some(IROp::Foo(...))``
   entries for every width pyvex actually emits. If you don't know
   which widths are real, run pyvex against a sample binary and grep
   the lifted IRSB.

4. **Map the numeric opcode (if pyvex emits it).** Add the
   corresponding numeric arms to ``parse_opcode_from_u32`` using the
   libvex enum offsets from ``libvex_ir.h``. Stay in numeric order so
   the table remains scannable.

5. **Implement.** Add the dispatch arm to ``VEXOps::unop`` /
   ``binop`` / ``triop`` / ``qop`` in ``ops.rs``. Prefer the
   ``width_unop!`` / ``width_binop!`` macros for shapes where the op
   is a one-liner on ``RustBV``; promote to a named helper when the
   logic doesn't fit on a single line. If both a concrete and a
   symbolic path are needed, follow the ``FloatLaneOp`` trait pattern
   so the compiler enforces parity.

6. **Test.** Add one happy-path test plus an edge case (overflow,
   width=0, signed/unsigned boundary, symbolic fallback — whatever
   applies). Keep the test in the same module's ``#[cfg(test)] mod
   tests`` block — there's no separate test crate for VEX ops.

7. **Build and run.** ``cargo check --manifest-path
   native/angr/Cargo.toml --release`` for the fast loop;
   ``pip install -e . --no-build-isolation --no-deps`` to refresh the
   ``.so`` (or ``make rebuild``); then
   ``python -m pytest tests/engines/test_rust_exploration.py`` to
   confirm nothing downstream regressed.

If at any point the lifter's pyvex bridge is involved (e.g. you're
adding an op whose pyvex name is *not* a simple
``Iop_<name><width>``), look at ``vex/pyvex_bridge.rs`` and
``vex/libpyvex_ffi.rs`` — the latter is the FFI path used when the
engine receives an already-lifted ``IRSB`` from C-side libvex.

What *not* to do
----------------

* Don't add a new sub-router to ``parse_opcode`` for a single op. If
  it doesn't fit any of the nine, the op probably belongs in
  ``parse_special``.
* Don't reach for ``IROp::Raw(0)`` in production code paths. ``Raw``
  exists as the "unmapped" sentinel; treating it as an escape hatch
  hides bugs from contributors who actually want to map the op.
* Don't bypass the width parameterization just to ship faster.
  ``IROp::Foo32`` / ``IROp::Foo64`` separately is exactly the
  proliferation libvex pays for; the Rust engine's terseness is
  earned by *not* doing that.
* Don't put memory or register access in ``ops.rs``. The split between
  ``ops.rs`` (pure ``RustBV`` transforms) and ``interpreter`` (state
  access) is the engine's most useful internal boundary; preserving it
  keeps the test surface small (``SymContext::new_mock()`` is enough
  to test any pure op).

Unsupported op coverage matrix
------------------------------

At-a-glance status for op families that have historically been
placeholders. *Implemented* means a dispatch arm exists in
``native/angr/src/vex/ops.rs`` and the opcode parses to a concrete
``IROp`` variant (not ``IROp::NeonUnimplemented`` or
``IROp::Unmapped``). *Placeholder* means the opcode parses but
dispatch returns ``OpError::UnsupportedNeon``, which the engine
surfaces as ``RustUnsupportedVexOpError``. *Stubbed-symbolic* means
dispatch returns a fresh-symbolic value of the expected width — used
for ops whose semantics are too expensive or under-specified to
model (e.g. ``URECPE`` / ``URSQRTE`` per-lane reciprocal estimates).

Source of truth: ``native/angr/src/vex/opcode_map.rs``
(``parse_neon_unimplemented`` is the remaining placeholder list) and
``native/angr/src/vex/ops.rs`` (the dispatch arms). Refresh this
table whenever a campaign child closes — the bead column makes the
provenance scannable.

NEON op families
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Op family
     - Count
     - Status
     - Provenance
   * - Saturating add / sub (``Iop_QAdd*`` / ``Iop_QSub*``)
     - ~24
     - Implemented
     - ``IROp::VQAdd`` / ``IROp::VQSub`` (angr-tukg.1)
   * - Pairwise integer (``Iop_PwAdd*`` / ``Iop_PwAddL*`` /
       ``Iop_PwMin*`` / ``Iop_PwMax*``)
     - ~24
     - Implemented
     - ``IROp::VPwAdd`` / ``VPwAddL`` / ``VPwMin`` / ``VPwMax``
       (angr-tukg.2)
   * - Pairwise FP (``Iop_PwAdd32Fx2``)
     - 1
     - Placeholder
     - Last entry in ``parse_neon_unimplemented`` —
       routes to ``IROp::NeonUnimplemented``
   * - Rounding halving add (``Iop_Avg{N}{S/U}x{M}``)
     - ~12
     - Implemented
     - ``IROp::VAvg`` (angr-tukg.3)
   * - Byte / halfword / word / bit reversal within lane
       (``Iop_Reverse{N}sIn{M}_x{K}``)
     - ~12
     - Implemented
     - ``IROp::VReverse`` via ``parse_vreverse`` (angr-tukg.4)
   * - FP reciprocal estimates (``Iop_RecipEst{,S}*`` /
       ``Iop_RecipStep*`` / ``Iop_RSqrtEst{,S}*`` /
       ``Iop_RSqrtStep*``)
     - ~14
     - Implemented
     - ``IROp::VFRecipEst{,S}`` / ``VFRecipStep`` /
       ``VFRSqrtEst{,S}`` / ``VFRSqrtStep`` (angr-iyon)
   * - Integer reciprocal estimates (``Iop_RecipEst32Ux{2,4}`` —
       URECPE; ``Iop_RSqrtEst32Ux{2,4}`` — URSQRTE)
     - 4
     - Stubbed-symbolic
     - ``IROp::VIRecipEst`` / ``IROp::VIRSqrtEst`` — fresh-symbolic
       per lane (angr-tukg.5)
   * - Polynomial multiply (``Iop_PolynomialMul8x{8,16}`` /
       ``Iop_PolynomialMull8x8``)
     - 3
     - Implemented
     - ``IROp::VPolynomialMul`` (angr-tukg.6)
   * - Per-lane bitcount (``Iop_Cnt8x{8,16}`` /
       ``Iop_Clz{N}x{M}`` / ``Iop_Cls{N}x{M}``)
     - ~13
     - Implemented
     - ``IROp::VCnt`` / ``VClz`` / ``VCls`` (angr-tukg.6)
   * - Vector shift by vector (``Iop_Shl/Shr/Sar/Sal{N}x{M}``)
     - ~16
     - Implemented
     - ``IROp::VShl`` / ``VShr`` / ``VSar`` — ``Sal`` aliased to
       ``VShl`` (angr-tukg.7)
   * - NEON saturating shift-left by vector
       (``Iop_QShl{N}x{M}`` / ``Iop_QSal{N}x{M}``)
     - ~8
     - Implemented
     - ``IROp::VQShlSat`` (angr-tukg.8)
   * - NEON saturating shift-by-immediate (``Iop_QShlN*``)
     - ~4
     - Placeholder
     - Noted in ``parse_neon_unimplemented`` rustdoc — no enum
       variant or dispatch arm yet

The remaining placeholders (``Iop_PwAdd32Fx2``, ``Iop_QShlN*``) are
the residual entries after the NEON campaign (``angr-tukg``) closed.
Promote them to standalone beads when a benchmark drives a symbolic
path through them.

x87 transcendental ops
^^^^^^^^^^^^^^^^^^^^^^

These ops do *not* parse to ``IROp::NeonUnimplemented`` — they fall
all the way through ``parse_opcode`` and surface as
``IROp::Unmapped(name)``. Same end-user error
(``RustUnsupportedVexOpError``), different provenance: there is no
parse arm yet, not just a missing dispatch arm.

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Op family
     - Count
     - Status
     - Provenance
   * - Log / exp (``Iop_Fyl2x``, ``Iop_F2xm1``)
     - 2
     - Placeholder
     - Routes to ``IROp::Unmapped`` (angr-i5lj.1)
   * - Misc FP (``Iop_Fscale``, ``Iop_Fpatan``, ``Iop_Fcos``,
       ``Iop_Fsin``, ``Iop_Fxam``, ``Iop_Fxbm1``)
     - 6
     - Placeholder
     - Routes to ``IROp::Unmapped`` (angr-i5lj.2)

The x87 transcendentals only matter on workloads that drive a
symbolic path through them. ``securityfest_fairlight`` and
``ekopartyctf2016_sokohashv2`` both hit them in the original binary
but the test drivers hook them out at the Python layer, so the
matrix above does not yet block any tracked benchmark.

Catching new placeholders
^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter ``RustUnsupportedVexOpError("Iop_<name>", arch)``
on a new workload:

1. ``grep "Iop_<name>"`` under ``native/angr/src/vex/`` to confirm
   it has no parse arm. (If it does, the missing piece is a dispatch
   arm in ``ops.rs`` — see the "Pipeline overview" section above.)
2. If it has no parse arm, decide which sub-router in ``opcode_map.rs``
   it belongs in (``parse_float`` / ``parse_vector`` / etc.) and add
   it there.
3. Add a row to the matrix above with status ``Placeholder`` and a
   pointer to whichever bead tracks the implementation work.

