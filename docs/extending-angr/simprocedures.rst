Hooks and SimProcedures
=======================

Hooks in angr are very powerful! You can use them to modify a program's behavior
in any way you could imagine. However, the exact way you might want to program a
specific hook may be non-obvious. This chapter should serve as a guide when
programming SimProcedures.

Quick Start
-----------

Here's an example that will remove all bugs from any program:

.. code-block:: python

   >>> from angr import Project, SimProcedure
   >>> project = Project('examples/fauxware/fauxware')

   >>> class BugFree(SimProcedure):
   ...    def run(self, argc, argv):
   ...        print('Program running with argc=%s and argv=%s' % (argc, argv))
   ...        return 0

   # this assumes we have symbols for the binary
   >>> project.hook_symbol('main', BugFree())

   # Run a quick execution!
   >>> simgr = project.factory.simulation_manager()
   >>> simgr.run()  # step until no more active states
   Program running with argc=<SAO <BV64 0x0>> and argv=<SAO <BV64 0x7fffffffffeffa0>>
   <SimulationManager with 1 deadended>

Now, whenever program execution reaches the main function, instead of executing
the actual main function, it will execute this procedure! It just prints out a
message, and returns.

Now, let's talk about what happens on the edge of this function! When entering
the function, where do the values that go into the arguments come from? You can
define your ``run()`` function with however many arguments you like, and the
SimProcedure runtime will automatically extract from the program state those
arguments for you, via a :ref:`calling convention <advanced-topics/structured_data:Working with Calling Conventions>`, and call your run function with them. Similarly, when you return
a value from the run function, it is placed into the state (again, according to
the calling convention), and the actual control-flow action of returning from a
function is performed, which depending on the architecture may involve jumping
to the link register or jumping to the result of a stack pop.

It should be clear at this point that the SimProcedure we just wrote is meant to
totally replace whatever function it is hooked over top of. In fact, the
original use case for SimProcedures was replacing library functions. More on
that later.

Implementation Context
----------------------

On a ``Project`` class, the dict ``project._sim_procedures`` is a mapping from
address to ``SimProcedure`` instances. When the :ref:`execution pipeline
<advanced-topics/pipeline:Understanding the Execution Pipeline>` reaches an address that is present in
that dict, that is, an address that is hooked, it will execute
``project._sim_procedures[address].execute(state)``. This will consult the
calling convention to extract the arguments, make a copy of itself in order to
preserve thread safety, and run the ``run()`` method. It is important to produce
a new instance of the SimProcedure for each time it is run, since the process of
running a SimProcedure necessarily involves mutating state on the SimProcedure
instance, so we need separate ones for each step, lest we run into race
conditions in multithreaded environments.

kwargs
^^^^^^

This hierarchy implies that you might want to reuse a single SimProcedure in
multiple hooks. What if you want to hook the same SimProcedure in several
places, but tweaked slightly each time? angr's support for this is that any
additional keyword arguments you pass to the constructor of your SimProcedure
will end up getting passed as keyword args to your SimProcedure's ``run()``
method. Pretty cool!

Data Types
----------

If you were paying attention to the example earlier, you noticed that when we
printed out the arguments to the ``run()`` function, they came out as a weird
``<SAO <BV64 0xSTUFF>>`` class. This is a ``SimActionObject``. Basically, you
don't need to worry about it too much, it's just a thin wrapper over a normal
bitvector. It does a bit of tracking of what exactly you do with it inside the
SimProcedure---this is helpful for static analysis.

You may also have noticed that we directly returned the Python int ``0`` from
the procedure. This will automatically be promoted to a word-sized bitvector!
You can return a native number, a bitvector, or a SimActionObject.

When you want to write a procedure that deals with floating point numbers, you
will need to specify the calling convention manually. It's not too hard, just
provide a cc to the hook: ```cc = project.factory.cc_from_arg_kinds((True,
True), ret_fp=True)`` and ``project.hook(address, ProcedureClass(cc=mycc))``
This method for passing in a calling convention works for all calling
conventions, so if angr's autodetected one isn't right, you can fix that.

Control Flow
------------

How can you exit a SimProcedure? We've already gone over the simplest way to do
this, returning a value from ``run()``. This is actually shorthand for calling
``self.ret(value)``. ``self.ret()`` is the function which knows how to perform
the specific action of returning from a function.

SimProcedures can use lots of different functions like this!


* ``ret(expr)``: Return from a function
* ``jump(addr)``: Jump to an address in the binary
* ``exit(code)``: Terminate the program
* ``call(addr, args, continue_at)``: Call a function in the binary
* ``inline_call(procedure, *args)``: Call another SimProcedure in-line and
  return the results

That second-last one deserves some looking-at. We'll get there after a quick
detour...

Conditional Exits
^^^^^^^^^^^^^^^^^

What if we want to add a conditional branch out of a SimProcedure? In order to
do that, you'll need to work directly with the SimSuccessors object for the
current execution step.

The interface for this is ```self.successors.add_successor(state, addr, guard,
jumpkind)``. All of these parameters should have an obvious meaning if you've
followed along so far. Keep in mind that the state you pass in will NOT be
copied and WILL be mutated, so be sure to make a copy beforehand if there will
be more work to do!

SimProcedure Continuations
^^^^^^^^^^^^^^^^^^^^^^^^^^

How can we call a function in the binary and have execution resume within our
SimProcedure? There is a whole bunch of infrastructure called the "SimProcedure
Continuation" that will let you do this. When you use ``self.call(addr, args,
continue_at)``, ``addr`` is expected to be the address you'd like to call,
``args`` is the tuple of arguments you'd like to call it with, and
``continue_at`` is the name of another method in your SimProcedure class that
you'd like execution to continue at when it returns. This method must have the
same signature as the ``run()`` method. Furthermore, you can pass the keyword
argument ``cc`` as the calling convention that ought to be used to communicate
with the callee.

When you do this, you finish your current step, and execution will start again
at the next step at the function you've specified. When that function returns,
it has to return to some concrete address! That address is specified by the
SimProcedure runtime: an address is allocated in angr's externs segment to be
used as the return site for returning to the given method call. It is then
hooked with a copy of the procedure instance tweaked to run the specified
``continue_at`` function instead of ``run()``, with the same args and kwargs as
the first time.

There are two pieces of metadata you need to attach to your SimProcedure class
in order to use the continuation subsystem correctly:


* Set the class variable ``IS_FUNCTION = True``
* Set the class variable ``local_vars`` to a tuple of strings, where each string
  is the name of an instance variable on your SimProcedure whose value you would
  like to persist to when you return. Local variables can be any type so long as
  you don't mutate their instances.

You may have guessed by now that there exists some sort of auxiliary storage in
order to hold on to all this data. You would be right! The state plugin
``state.callstack`` has an entry called ``.procedure_data`` which is used by the
SimProcedure runtime to store information local to the current call frame. angr
tracks the stack pointer in order to make the current top of the
``state.callstack`` a meaningful local data store. It's stuff that ought to be
stored in memory in a stack frame, but the data can't be serialized and/or
memory allocation is hard.

As an example, let's look at the SimProcedure that angr uses internally to run
all the shared library initializers for a ``full_init_state`` for a linux
program:

.. code-block:: python

   class LinuxLoader(angr.SimProcedure):
       NO_RET = True
       IS_FUNCTION = True
       local_vars = ('initializers',)

       def run(self):
           self.initializers = self.project.loader.initializers
           self.run_initializer()

       def run_initializer(self):
           if len(self.initializers) == 0:
               self.project._simos.set_entry_register_values(self.state)
               self.jump(self.project.entry)
           else:
               addr = self.initializers[0]
               self.initializers = self.initializers[1:]
               self.call(addr, (self.state.posix.argc, self.state.posix.argv, self.state.posix.environ), 'run_initializer')

This is a particularly clever usage of the SimProcedure continuations. First,
notice that the current project is available for use on the procedure instance.
This is some powerful stuff you can get yourself into; for safety you generally
only want to use the project as a read-only or append-only data structure. Here
we're just getting the list of dynamic initializers from the loader. Then, for as
long as the list isn't empty, we pop a single function pointer out of the list,
being careful not to mutate the list, since the list object is shared across
states, and then call it, returning to the ``run_initializer`` function again.
When we run out of initializers, we set up the entry state and jump to the
program entry point.

Very cool!

Global Variables
----------------

As a brief aside, you can store global variables in ``state.globals``. This is a
dictionary that just gets shallow-copied from state to successor state. Because
it's only a shallow copy, its members are the same instances, so the same rules
as local variables in SimProcedure continuations apply. You need to be careful
not to mutate any item that is used as a global variable unless you know exactly
what you're doing.

Helping out static analysis
---------------------------

We've already looked at the class variable ``IS_FUNCTION``, which allows you to
use the SimProcedure continuation. There are a few more class variables you can
set, though these ones have no direct benefit to you - they merely mark
attributes of your function so that static analysis knows what it's doing.


* ``NO_RET``: Set this to true if control flow will never return from this
  function
* ``ADDS_EXITS``: Set this to true if you do any control flow other than
  returning
* ``IS_SYSCALL``: Self-explanatory

Furthermore, if you set ``ADDS_EXITS = True``, you'll need to define the method
``static_exits()``. This function takes a single parameter, a list of IRSBs that
would be executed in the run-up to your function, and asks you to return a list
of all the exits that you know would be produced by your function in that case.
The return value is expected to be a list of tuples of (address (int), jumpkind
(str)). This is meant to be a quick, best-effort analysis, and you shouldn't try
to do anything crazy or intensive to get your answer.

User Hooks
----------

The process of writing and using a SimProcedure makes a lot of assumptions that
you want to hook over a whole function. What if you don't? There's an alternate
interface for hooking, a *user hook*, that lets you streamline the process of
hooking sections of code.

.. code-block:: python

   >>> @project.hook(0x1234, length=5)
   ... def set_rax(state):
   ...     state.regs.rax = 1

This is a lot simpler! The idea is to use a single function instead of an entire
SimProcedure subclass. No extraction of arguments is performed, no complex
control flow happens.

Control flow is controlled by the length argument. After the function finishes
executing in this example, the next step will start at 5 bytes after the hooked
address. If the length argument is omitted or set to zero, execution will resume
executing the binary code at exactly the hooked address, without re-triggering
the hook. The ``Ijk_NoHook`` jumpkind allows this to happen.

If you want more control over control flow coming out of a user hook, you can
return a list of successor states. Each successor will be expected to have
``state.regs.ip``, ``state.scratch.guard``, and ``state.scratch.jumpkind`` set.
The IP is the target instruction pointer, the guard is a symbolic boolean
representing a constraint to add to the state related to it being taken as
opposed to the others, and the jumpkind is a VEX enum string, like
``Ijk_Boring``, representing the nature of the branch.

The general rule is, if you want your SimProcedure to either be able to extract
function arguments or cause a program return, write a full SimProcedure class.
Otherwise, use a user hook.

Hooking Symbols
---------------

As you should recall from the :ref:`section on loading a binary <core-concepts/loading:Loading a Binary>`, dynamically linked programs have a list of symbols that they must
import from the libraries they have listed as dependencies, and angr will make
sure, rain or shine, that every import symbol gets resolved by *some* address,
whether it's a real implementation of the function or just a dummy address hooked
with a do-nothing stub. As a result, you can just use the
``Project.hook_symbol`` API to hook the address referred to by a symbol!

This means that you can replace library functions with your own code. For
instance, to replace ``rand()`` with a function that always returns a consistent
sequence of values:

.. code-block:: python

   >>> class NotVeryRand(SimProcedure):
   ...     def run(self, return_values=None):
   ...         rand_idx = self.state.globals.get('rand_idx', 0) % len(return_values)
   ...         out = return_values[rand_idx]
   ...         self.state.globals['rand_idx'] = rand_idx + 1
   ...         return out

   >>> project.hook_symbol('rand', NotVeryRand(return_values=[413, 612, 1025, 1111]))

Now, whenever the program tries to call ``rand()``, it'll return the integers
from the ``return_values`` array in a loop.

Native (Rust) SimProcedures
---------------------------

The Python SimProcedure machinery above is the right tool for almost every
hook you'll write. But for a small set of very-hot libc functions (``strlen``,
``memcpy``, ``strcmp``, ``malloc``, …) the cost of crossing the FFI boundary
on every call dominates the work the procedure actually does. The Rust
engine (``use_rust_engine=True`` on
``proj.factory.simulation_manager(...)``) ships with **native** versions of
those procedures written in Rust against the engine's internal
``RustSimState``. They never enter Python, never marshal arguments through
claripy, and never pay the callback round-trip — when they're applicable,
they're roughly two orders of magnitude faster than the equivalent Python
SimProcedure.

This section is the contributor guide for adding one. If you're hooking
application code (not a hot libc symbol), write a normal Python
``SimProcedure`` instead: the perf wins below are only meaningful at very
high call frequencies, and Python is significantly easier to read, test,
and debug.

When to add a native procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add one only if all of these are true:

* The function is called *frequently* under the Rust engine — typically a
  libc primitive that shows up in callback-frequency profiles
  (see the ``z3_check_count`` / fallback counters in the *Z3 solver
  profiling counters* section of :doc:`../advanced-topics/rust_engine`).
* It has a well-defined contract that you can implement byte-for-byte
  against ``RustSimState`` (or you have an explicit, documented
  approximation, like ``strlen``'s ``MAX_STRLEN`` cap).
* It has a sensible fallback to Python when its arguments don't satisfy
  the native fast path (typically: addresses or sizes are symbolic).

The trait
^^^^^^^^^

Every native procedure lives in ``native/angr/src/procedures/`` and
implements the ``NativeSimProcedure`` trait defined in
``native/angr/src/procedures/mod.rs``:

.. code-block:: rust

   pub trait NativeSimProcedure: Send + Sync {
       fn name(&self) -> &'static str;
       fn num_args(&self) -> usize;

       /// Extra dispatch names, mirroring Python angr's `x_unlocked = x`
       /// aliasing. `register()` inserts the proc under each alias too.
       fn aliases(&self) -> &'static [&'static str] { &[] }
       fn no_return(&self) -> bool { false }

       fn call(
           &self,
           state: &mut RustSimState,
           args: &[RustBV],
       ) -> Result<Option<RustBV>, ProcedureError>;

       /// Fresh-entry point; default wraps `call` into `ProcOutcome::Return`.
       /// Override only to express a sub-call (see below).
       fn call_ex(
           &self,
           state: &mut RustSimState,
           args: &[RustBV],
       ) -> Result<ProcOutcome, ProcedureError> {
           self.call(state, args).map(ProcOutcome::Return)
       }

       /// Continuation re-entered after a sub-call returns. Default is
       /// `Err(ProcedureError::NotImplemented)`.
       fn resume(
           &self,
           state: &mut RustSimState,
           resume_tag: u32,
           saved_args: &[RustBV],
       ) -> Result<ProcOutcome, ProcedureError> { ... }
   }

Most procedures implement only ``name``, ``num_args`` and ``call`` — the
default ``call_ex``/``resume`` bodies make the sub-call machinery invisible
unless you need it.

The return convention is the contract between your procedure and the
dispatcher:

* ``Ok(Some(value))`` — the procedure ran to completion; ``value`` is
  written to the calling convention's return register and the dispatcher
  advances PC past the call.
* ``Ok(None)`` — the procedure ran to completion with no return value
  (a ``void`` function, or a terminal procedure with
  ``no_return() == true`` like ``exit``).
* ``Err(ProcedureError)`` — the native fast path can't handle this call;
  fall back to the Python SimProcedure registered at this address.
  ``ProcedureError::SymbolicArgument(name)`` is by far the most common
  variant.

The full ``ProcedureError`` enum lives in ``procedures/mod.rs``:
``SymbolicArgument(String)``, ``Memory(MemoryError)`` (the structured
memory-subsystem error — unmapped page, permission violation, symbolic
address — reachable via ``#[from]``, so ``?`` on a memory operation
converts automatically), ``NotImplemented``, ``MaxIterations(usize)``,
and ``Other(String)``. Any of them triggers Python fallback. The enum is
``#[non_exhaustive]`` (angr-irwe): minor versions may add variants, so
every ``match`` on it needs a wildcard arm.

If your procedure caps how many bytes it will walk or copy, express the
bail-out as ``check_max(n, MAX_WHATEVER)?`` (``procedures/mod.rs``) rather
than hand-rolling the compare: it fixes the variant at ``MaxIterations``
and the payload at the *offending* value, the two things the ~19 call sites
had started to disagree about before angr-12jjk.24. Only bail-outs that
aren't a plain ``n > max`` compare — a scan that ran to exhaustion, a
"is this prefix result conclusive?" check — should build the error
directly.

.. warning::

   **Value-returning "unconstrained" stubs must match Python's
   ``SYMBOLIC_INITIAL_VALUES`` gate.** Do not return a fresh symbolic
   BVS for a stub that mimics ``angr.procedures.stubs.ReturnUnconstrained``
   (``operator new``/``delete``, ``ostream::operator<<``, ``std::string``
   ctors — the bulk of C++ ABI stubs). Python's ``ReturnUnconstrained``
   calls ``state.solver.Unconstrained(name, size, key=...)``, and
   ``SimSolver.Unconstrained`` returns a **concrete ``BVV(0)``** unless
   ``sim_options.SYMBOLIC_INITIAL_VALUES`` is in ``state.options`` (see
   ``angr/state_plugins/solver.py``'s ``Unconstrained``). Every stock mode
   bundle in ``sim_options.modes`` ships that option, so the symbolic branch
   is the common case — but a caller that passes
   ``remove_options={SYMBOLIC_INITIAL_VALUES}`` must get zeros, and an
   ungated native stub silently ignores them.
   A native fast path that unconditionally writes a fresh symbolic value
   diverges: the symbolic pointer feeds downstream null-checks and is used
   as a store/load address, each of which forks or pays the symbolic-address
   concretization cost, exploding the state space. An end-to-end attempt at
   such a fast path (angr-8mjd, reverted) regressed ``csaw_wyvern`` from
   2.7s to 86s for exactly this reason. The correct contract: when
   ``SYMBOLIC_INITIAL_VALUES`` is absent, return
   ``Ok(Some(BVV(0, returnty_bits)))``; only mint a symbolic value when the
   option is set. ``void``-return stubs (``returnty == None`` →
   ``Ok(None)``) are always safe.
   ``procedures/stub.rs::NativeReturnUnconstrained`` is the reference
   implementation of the gate; it reads the option through
   ``RustSimState::has_option``, which requires the option name to be listed
   in ``_NATIVE_SIMOPTIONS`` (``angr/exploration/rust_manager.py``) so the
   Python→Rust mirror actually threads it.

Sub-calls (``ProcOutcome::CallAndResume``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few procedures must invoke a *guest* routine and continue afterwards —
the native equivalent of Python's
``SimProcedure.call(func, args, "retsite")``. ``call`` cannot express
that (its return type has no "jump somewhere and come back" case), so
such a procedure overrides ``call_ex`` and returns ``ProcOutcome``:

.. code-block:: rust

   pub enum ProcOutcome {
       /// Completed; store this return value (or `None` for void).
       Return(Option<RustBV>),
       /// Jump into guest routine `target`, then re-enter this proc's
       /// `resume()` with `resume_tag` when it returns.
       CallAndResume { target: u64, args: Vec<RustBV>, resume_tag: u32 },
   }

``pthread_once`` (``procedures/pthread.rs``) is the reference
implementation: ``call_ex`` reads the once-guard, returns
``ProcOutcome::Return(0)`` when the initializer already ran, otherwise
sets the guard bit and returns
``CallAndResume { target: func, args: vec![], resume_tag: 0 }``;
``resume`` then yields ``0``. Its ``call`` is a plain
``Err(ProcedureError::NotImplemented)`` so any dispatch path that still
goes through ``call`` falls back to Python rather than silently skipping
the sub-call.

Mechanics, for procedures that need them:

* The dispatcher makes the guest routine return to
  ``native_resume_sentinel()`` — an address registered under the
  reserved name ``NATIVE_RESUME_SENTINEL_NAME`` (``"__native_resume__"``)
  in every interpreter's SimProcedure registry. When PC lands there the
  dispatcher pops the top ``NativeResumeFrame`` and calls ``resume``.
* The resume stack is **LIFO**: ``resume`` may itself return
  ``CallAndResume``, so a continuation can chain further sub-calls.
* ``resume_tag`` selects which continuation arm to run (the data-encoded
  analogue of Python's named continuation), and ``saved_args`` are the
  original ``call_ex`` arguments, captured at sub-call time — mirroring
  Python's ``retsite`` re-receiving its ``run()`` args.
* Bail out (return ``Err``) *before* mutating state whenever the sub-call
  setup could still fail — e.g. ``pthread_once`` checks for a symbolic SP
  before writing the guard bit, because the setup stores the sentinel to
  ``[sp]`` and a later fallback to Python would otherwise see a guard bit
  that says "already initialized".

Design rationale: ``tools/decisions/native_subcall_dispatcher_design.md``
(bead ``angr-5gf0s``, Option B). Regression coverage:
``native/angr/src/exploration/subcall_tests.rs`` and
``native/angr/src/procedures/pthread_tests.rs``.

Argument extraction with ``declare_proc!``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most procedures need concrete ``u64`` arguments (addresses, sizes,
indices). Writing the boilerplate by hand drifts out of sync — the
``num_args()`` count and the argument-extraction code can disagree
silently. The ``declare_proc!`` macro (in
``native/angr/src/procedures/macros.rs``) drives both from a single
declaration:

.. code-block:: rust

   crate::declare_proc! {
       name = "strlen",
       struct = NativeStrlen,
       args = [addr: concrete],
       call |state| {
           scan_for_null(state, addr, MAX_STRLEN as u64)
       }
   }

The supported argument kinds are:

* ``concrete`` — invokes ``extract_concrete_arg`` on the argument and
  binds a ``u64``. A symbolic argument short-circuits to
  ``ProcedureError::SymbolicArgument(<arg name>)``.
* ``bv`` — clones the raw ``RustBV`` and binds it. Use this when the
  body itself wants to inspect concreteness (e.g. building an ITE
  chain over a symbolic byte stream).

The optional ``no_return = true,`` flag overrides the default
``no_return()``; use it for terminal procedures like ``exit`` and
``abort`` so the dispatcher routes the state to the deadended stash
instead of advancing PC past the call.

Dispatch and registration
^^^^^^^^^^^^^^^^^^^^^^^^^

The engine looks up native procedures by **name** through
``NativeProcedureRegistry``. Registration is hand-written in
``NativeProcedureRegistry::new`` (``procedures/mod.rs``):

.. code-block:: rust

   registry.register(Arc::new(strlen::NativeStrlen));
   registry.register(Arc::new(memcpy::NativeMemcpy));
   registry.register(Arc::new(malloc::NativeMalloc));

Every angr SimProcedure that has a matching name in the registry is
intercepted before its Python ``run()`` would be invoked. The dispatcher
extracts the calling convention's argument bitvectors, hands them to
``call_ex()`` — the fresh-entry point, whose default body delegates to
``call()`` — and acts on the returned ``Result``. The registry also
supports per-procedure ``disable()`` and ``set_python_override()`` —
both force the dispatcher to fall back to Python — and a global
``disable_all()`` switch (used in differential testing).

Dispatch priority (native vs Python)
""""""""""""""""""""""""""""""""""""

When PC reaches a hooked address registered as a SimProcedure, the
dispatcher chooses between the native and Python implementations using
this ordered chain (first rule wins; native and Python are **never**
both invoked except when native fails):

1. **Main-object hooks always run Python.** If the hook PC falls inside
   the main object's code span (i.e. ``proj.hook(addr, MyProc())``
   placed somewhere in the program text), native is skipped entirely.
   This honors the user's intent to override a specific instruction.
   Hooks *outside* every loaded object — the extern-object PLT stubs
   angr synthesizes for statically-known symbols — always prefer
   native. Hooks inside a **non-main** loaded object (libc &c.,
   installed by ``use_sim_procedures=True`` on a dynamically-linked
   binary) prefer native only when ``prefer_native_library_hooks`` is
   on, which is the default; the
   ``prefer_native_library_hooks=False`` constructor kwarg (or
   ``ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=0``) sends them to Python
   instead. The gate is
   ``prefer_native_dispatch`` in
   ``native/angr/src/exploration/execution_env.rs``.
2. **Per-name Python override skips native.**
   ``set_python_override("strlen")`` makes ``registry.get("strlen")``
   return ``None`` so dispatch falls through to the Python
   SimProcedure. Used internally by
   ``NativeLibcStartMain`` and available to user code via
   ``mgr.set_python_override(name)``.
3. **Per-name disable skips native.** ``registry.disable("strlen")``
   has the same runtime effect as a Python override; semantically it
   says "the Rust implementation is not trustworthy right now"
   rather than "Python is canonical for this name".
4. **Global disable.** ``registry.disable_all()`` /
   ``mgr.disable_native_procedures()`` skips native for every name.
5. **Native runs; on error, Python takes over.** If steps 1–4 do
   not bypass it, the dispatcher calls ``native_proc.call_ex(...)``
   (**not** ``call`` directly — the default ``call_ex`` wraps
   ``call``'s result into ``ProcOutcome::Return``, so return-only
   procedures behave identically). On ``Ok(ProcOutcome::Return(..))``,
   ``native_proc_stats.native_calls`` increments and PC advances to
   the return address; on ``Ok(ProcOutcome::CallAndResume { .. })``
   it pushes a resume frame and jumps into the guest routine (see
   *Sub-calls* above). On any ``Err``, the dispatcher
   bumps ``native_proc_stats.python_fallbacks`` (bucketed by error
   variant in ``symbolic_fallbacks_by_name``,
   ``not_implemented_fallbacks_by_name``, or
   ``other_fallbacks_by_name``) and emits a ``need_simprocedure``
   event so the Python SimProcedure runs instead.
6. **No native implementation.** If the registry has no entry for
   ``name``, the dispatcher proceeds directly to the Python
   fallback. This increments ``simprocedure_python_fallback_count``
   and ``simprocedure_fallback_by_name`` but **not**
   ``native_proc_stats.python_fallbacks`` (which only counts cases
   where native was attempted and lost).

Regression coverage for this contract lives at
``tests/engines/rust/`` —
``test_python_override_bypasses_native_strlen`` (override case) and
``test_python_procedure_symbolic_arg_falls_back_to_python``
(native-tried-then-Python case).

For a contributor: adding a new procedure means (1) writing a module
under ``native/angr/src/procedures/``, (2) declaring it ``pub mod`` from
``mod.rs``, and (3) adding a single ``registry.register(...)`` line in
``NativeProcedureRegistry::new``. The acceptance bar is then a handful
of ``#[cfg(test)] mod tests`` cases exercising the happy path, the
symbolic-arg fallback, and any edge cases (zero-size, max-size, etc.).

Worked example 1: concrete-only — ``strlen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest interesting shape: the address argument *must* be concrete
(addresses with symbolic offsets are out of scope for the fast path);
the bytes the procedure reads may themselves be symbolic, in which case
the procedure still produces a useful symbolic ``size_t`` rather than
bailing out. Source:
``native/angr/src/procedures/strlen.rs``.

.. code-block:: rust

   crate::declare_proc! {
       /// Native strlen: `size_t strlen(const char *s)`.
       name = "strlen",
       struct = NativeStrlen,
       args = [addr: concrete],
       call |state| {
           scan_for_null(state, addr, MAX_STRLEN as u64)
       }
   }

``scan_for_null`` walks the buffer byte-by-byte from ``addr``. While
every byte is concrete it short-circuits at the first ``\0``. As soon
as it encounters a symbolic byte it switches to collecting
``(position, byte)`` pairs and, at the end, folds them into a
right-to-left ``ITE`` chain so that the returned ``RustBV`` is a
symbolic ``size_t`` that earlier-positioned nulls win on. The procedure
saturates at ``MAX_STRLEN = 4096`` and falls back to Python via
``ProcedureError::MaxIterations`` if it sees that many bytes with no
concrete null.

Things to take away from this example:

* The ``addr: concrete`` declaration handles the address-must-be-known
  invariant — no manual ``args[0].as_u64()`` plumbing.
* Symbolic *byte values* are fine; they live in ``RustBV`` and the
  procedure works with them directly using
  ``state.solver().borrow()`` and the ``RustBV::ite`` /
  ``RustBV::eq`` combinators.
* The fallback path is ``MaxIterations(MAX_STRLEN)`` — large or
  truly unbounded inputs go back to Python rather than producing a
  4096-deep ITE chain inside Z3.

Worked example 2: concrete + symbolic-arg fallback — ``memcpy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``memcpy`` is the canonical example of "three concrete arguments,
bail out on any symbolic one." Source:
``native/angr/src/procedures/memcpy.rs``.

.. code-block:: rust

   impl NativeSimProcedure for NativeMemcpy {
       fn name(&self) -> &'static str { "memcpy" }
       fn num_args(&self) -> usize { 3 }

       fn call(
           &self,
           state: &mut RustSimState,
           args: &[RustBV],
       ) -> Result<Option<RustBV>, ProcedureError> {
           let dst  = extract_concrete_arg(&args[0], "dst")?;
           let src  = extract_concrete_arg(&args[1], "src")?;
           let size = extract_concrete_arg(&args[2], "size")? as usize;

           check_max(size as u64, MAX_COPY_SIZE)?;
           if size == 0 {
               return Ok(Some(args[0].clone()));
           }
           copy_forward(state, src, dst, size)?;
           Ok(Some(args[0].clone()))
       }
   }

This is the long-hand form of what ``declare_proc!`` generates for
``args = [dst: concrete, src: concrete, size: concrete]`` — both
styles are accepted; pick whichever reads better for the procedure.
The propagation of ``ProcedureError::SymbolicArgument`` via the ``?``
operator is the engine's fallback signal: any symbolic argument turns
into ``Err(SymbolicArgument(name))`` and the dispatcher hands the call
off to Python.

Things to take away from this example:

* ``extract_concrete_arg(&args[i], "name")?`` is the verbose form of
  the ``concrete`` declaration. The ``name`` string lands in the
  ``ProcedureError`` message and is purely diagnostic.
* The return value is ``Ok(Some(args[0].clone()))`` — POSIX
  ``memcpy`` returns ``dst``, which is exactly the first argument
  bitvector. Cloning a ``RustBV`` is cheap (it's an ``Arc`` under
  the hood).
* Bulk memory motion goes through ``state.memory_load`` /
  ``state.memory_store``, which preserves symbolic byte values
  end-to-end (the memcpy of a symbolic buffer is itself symbolic).
* ``MAX_COPY_SIZE`` is a guard against pathological inputs: a 100MB
  concrete-size memcpy would lock the engine inside Rust for minutes.
  Falling back to Python at the size cap is correct, not a bug.

Worked example 3: allocator state — ``malloc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``malloc`` is interesting because it doesn't just transform memory —
it mutates engine state (the heap bump pointer) that *must* survive
into the next instruction and into forked successor states. Source:
``native/angr/src/procedures/malloc.rs``.

.. code-block:: rust

   impl NativeSimProcedure for NativeMalloc {
       fn name(&self) -> &'static str { "malloc" }
       fn num_args(&self) -> usize { 1 }

       fn call(
           &self,
           state: &mut RustSimState,
           args: &[RustBV],
       ) -> Result<Option<RustBV>, ProcedureError> {
           let size = extract_concrete_arg(&args[0], "size")?;
           let addr = state.heap_alloc(size);
           let bits = state.arch().bits();
           Ok(Some(RustBV::concrete(addr as u128, bits)))
       }
   }

``state.heap_alloc`` is the bump allocator that mirrors angr's
``SimHeapBrk``: it advances a per-state pointer and returns the new
allocation's base. Because the allocator lives on ``RustSimState``,
it's already correctly cloned when the state forks — the contributor
gets state-aware allocation for free.

Things to take away from this example:

* Return-value width must match the arch's pointer width
  (``state.arch().bits()``). Returning a ``RustBV::concrete(addr, 64)``
  on a 32-bit guest would silently truncate.
* Side-effects on ``RustSimState`` (heap, fd table, posix env) are
  the *only* state the engine considers durable — write through
  ``state`` methods, not through globals or thread-locals.
* ``free`` (in the same file) just calls ``state.heap_free`` and
  returns ``Ok(None)``. The bump allocator can't actually reclaim
  memory; ``heap_free`` exists for bookkeeping.

Worked example 4: format strings — ``printf`` / ``scanf`` / ``sprintf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``printf``/``scanf``/``sprintf``/``snprintf``/``sscanf`` family parses
the format string itself, so its concreteness boundary is sharper than the
buffer-copying procedures above. Two distinct things can be symbolic — the
format-string *address* and the format-string *bytes* — and the family
handles them differently:

* **Symbolic format-string address.** All five extract the format pointer
  with ``extract_concrete_arg(&args[0], "format")``; a symbolic address
  short-circuits to ``ProcedureError::SymbolicArgument`` and Python takes
  over. There is no fast path for a symbolic format pointer.

* **Symbolic format-string bytes.** Here the family is deliberately
  *asymmetric*:

  - ``scanf``/``sscanf`` (``read_format_string``) and
    ``sprintf``/``snprintf`` (``read_string``) call
    ``extract_concrete_arg(&bv, ...)`` on *every* byte they read. The first
    symbolic byte raises ``SymbolicArgument`` and the whole call falls back
    to Python — these procedures must parse the specifiers
    (``%d``/``%s``/``%x``/…) to mint output BVs or consume variadic args,
    and a symbolic specifier byte makes that parse undefined.
  - ``printf`` (``NativePrintf::call``) does **not** fall back on a symbolic
    byte. It only copies the raw format string to the stdout buffer (no
    specifier substitution — sufficient for the common CTF predicate that
    greps stdout for a fixed string), so on the first symbolic byte it
    simply **stops reading, writes the concrete prefix, and returns success**
    with the prefix length. This is intentional, but it is the one place in
    the family where a symbolic format string does *not* hand off to Python:
    callers that need the full (symbolic-tail) string materialized must not
    rely on native ``printf``.

* **No symbolic-format substitution path exists.** None of the five attempt
  to enumerate or constrain a symbolic format string into concrete cases.
  That is the documented boundary: a format string that is symbolic *in the
  specifiers* is out of scope for the native fast path. ``printf`` degrades
  to a concrete-prefix write; the rest hand off to Python.

Things to take away from this example:

* The address-vs-bytes distinction matters: a concrete pointer into a buffer
  with symbolic contents is the common case (a user-controlled format buffer),
  and that is exactly what forces the ``scanf``/``sprintf`` fallback.
* ``printf``'s truncate-and-succeed behavior is a per-procedure choice keyed
  to what the procedure actually does with the string, not a family-wide
  convention. When adding a new format-consuming procedure, decide explicitly
  whether a symbolic byte should fall back (parse semantics) or truncate
  (raw-copy semantics).

Terminal procedures: ``no_return``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``exit`` and ``abort`` are tiny but instructive — see
``native/angr/src/procedures/exit.rs``. They override
``fn no_return(&self) -> bool { true }`` and return ``Ok(None)``. The
dispatcher in ``stepping.rs`` / ``exploration/mod.rs`` recognizes
``no_return`` and stashes the state in ``STASH_DEADENDED`` instead of
advancing PC past the call. Forgetting the flag will cause the engine
to re-execute the call site — in fauxware that turns into an
infinite re-entry loop because ``exit``'s call site overlaps ``main``'s
prologue.

Testing a native procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^

The convention used across ``procedures/*.rs`` is a single
``#[cfg(test)] mod tests`` at the bottom of the module. ``strlen.rs``
is the most complete reference and covers every shape worth testing:

* **Happy path** — basic, empty, longer string.
  (``test_strlen_basic`` / ``test_strlen_empty`` /
  ``test_strlen_longer_string``.)
* **Bounded variants** — limits below, at, and above the input size.
  (``test_strnlen_*``.)
* **Symbolic-arg fallback** — expects
  ``Err(ProcedureError::SymbolicArgument(_))``. Every procedure with
  a ``concrete`` argument should have at least one of these.
* **Symbolic-content path** — places a symbolic byte in memory and
  asserts that the returned ``RustBV`` is symbolic, then constrains
  it and checks ``ctx.min(&result, false) == ctx.max(&result, false)
  == <expected length>``. This is what gives the symbolic ITE chain
  a behavioral test rather than just a structural one.

``RustSimState::new("amd64").unwrap()`` plus
``state.map_memory_data(addr, bytes, Permission::RWX)`` is the entire
test fixture — the procedures are pure functions over state, so the
tests don't need an entire ``Project`` or VEX block setup.

User-attached Python procedures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The procedure registry has an escape hatch for users who want to
attach a custom Python callable to a hot symbol *without* compiling
Rust. ``RustExplorationManager.register_python_procedure(name,
num_args, no_return, callable)`` (defined in
``native/angr/src/exploration/mod.rs``) wraps a Python callable so it
fronts the procedure registry just like a Rust-side ``NativeStrlen``
would:

.. code-block:: python

   def widget_init(args):
       # args is a list[int] of CONCRETE u64 values.
       # Return Optional[int]: None = no return, int = arch-bits BV.
       return 0

   mgr.register_python_procedure(
       "custom_widget_init",
       num_args=0,
       no_return=False,
       callable=widget_init,
   )

Symbolic arguments still fall back to the regular Python
``SimProcedure`` path; the callable only runs when every argument is
concrete. This is the right shape when you want a one-off hook that
short-circuits a concrete-only function — it skips both the Python
``SimProcedure`` machinery *and* the VEX call frame the engine would
otherwise reconstruct, but stays in Python so you can iterate quickly.

If the callable shows up in benchmark profiles as a hot spot, that's
your signal to port it to a real ``NativeSimProcedure`` following the
recipe above.

Native coverage matrix
----------------------

At-a-glance status for which libc procedures and Linux syscalls have
native fast paths today. *Native* means a handler is registered in
``native/angr/src/procedures/mod.rs`` (procedures) or
``native/angr/src/syscalls/mod.rs`` (syscalls). Every row also has a
Python fallback — angr's regular ``SimProcedure`` registry — that runs
when a native handler is missing, disabled, or returns
``ProcedureError`` for a symbolic argument it cannot handle. The
fallback is what guarantees correctness; the native path is the speed
optimization.

Source of truth: the two ``mod.rs`` files above plus the campaign
beads (``angr-f16h.*`` for procedures, ``angr-0hif.*`` for syscalls).
Refresh this matrix whenever a campaign child closes — the bead
column makes the provenance scannable, parallel to the
``Unsupported op coverage matrix`` in
:doc:`rust_vex_ops`.

SimProcedures
^^^^^^^^^^^^^

The original pre-campaign set covers the highest-frequency string,
memory, I/O, and process primitives; the ``angr-f16h`` campaign
expanded coverage into stdio file I/O, the extended allocator family,
the string-to-numeric family, env mutation, and extended string ops.

.. list-table::
   :header-rows: 1
   :widths: 25 35 15 25

   * - Group
     - Members
     - Native
     - Provenance
   * - String / memory (pre-campaign)
     - ``strlen``, ``strnlen``, ``strcpy``, ``strncpy``, ``strdup``,
       ``strcmp``, ``strncmp``, ``strcasecmp``, ``strcat``,
       ``strncat``, ``strchr``, ``strstr``, ``memcpy``, ``memmove``,
       ``memset``, ``memcmp``, ``memchr``
     - 17 / 17
     - Original set in ``procedures/mod.rs`` (search
       ``NativeProcedureRegistry::new``)
   * - Character classification (pre-campaign)
     - ``isdigit``, ``isalpha``, ``isspace``, ``isalnum``,
       ``isupper``, ``islower``, ``isxdigit``, ``isprint``,
       ``tolower``, ``toupper``
     - 10 / 10
     - ``ctype.rs`` (symbolic-aware helper pattern lives in the file)
   * - String → integer (pre-campaign)
     - ``strtol``, ``atoi``
     - 2 / 2
     - ``strtol.rs`` (NativeStrtol, NativeAtoi)
   * - Input (stdin)
     - ``read`` ✓, ``fgets`` ✓, ``fgetc`` ✓, ``getchar`` ✓,
       ``getc`` ✓, ``scanf`` ✓, ``__isoc99_scanf`` ✓, ``sscanf`` ✗
     - 7 / 8
     - ``read.rs``, ``fgets.rs``, ``scanf.rs``. ``sscanf`` is
       registered (``scanf.rs::NativeSscanf``) but its ``call`` body
       unconditionally returns ``ProcedureError::Other``, so every
       invocation pays the full Python round-trip and it does not
       count toward the native speed-up. Deliberate: only Python
       parses the concrete source region and constrains the outputs,
       and a native free-BVS mint would explore impossible paths
       (angr-8onrp).
   * - Output (stdout / formatted)
     - ``write``, ``puts``, ``putchar``, ``fputc``, ``putc``,
       ``printf``, ``sprintf``, ``snprintf``
     - 8 / 8
     - ``write.rs``, ``puts.rs``, ``printf.rs``, ``sprintf.rs``
   * - Heap (pre-campaign + angr-f16h.2)
     - ``malloc``, ``free`` (pre-campaign);
       ``calloc``, ``realloc``, ``memalign``, ``posix_memalign``
       (angr-f16h.2)
     - 6 / 6
     - ``malloc.rs`` (bump allocator, matches ``SimHeapBrk``)
   * - Process / exit / RNG (pre-campaign)
     - ``exit``, ``_exit``, ``abort``, ``rand``, ``srand``,
       ``__libc_start_main``
     - 6 / 6
     - ``exit.rs``, ``rand.rs``, ``libc_start_main.rs``
   * - Env read (pre-campaign)
     - ``getenv``
     - 1 / 1
     - ``getenv.rs::NativeGetenv``
   * - C stdio file I/O (angr-f16h.1)
     - ``fopen``, ``fclose``, ``feof``, ``ferror``, ``fflush``,
       ``fputc``, ``fputs``, ``fgetc``, ``ftell``, ``fseek``,
       ``rewind``
     - 11 / 11
     - angr-f16h.1 (closed) — ``stdio.rs``, ``fileops.rs``,
       ``fgets.rs`` (also covers ``fdopen``, ``fwrite``, ``setvbuf``)
   * - File ops (pre-campaign)
     - ``open``, ``close``, ``lseek``, ``dup``, ``dup2``, ``pipe``
     - 6 / 6
     - ``fileops.rs`` (fd-tracking through ``FileSystem``)
   * - String → numeric (angr-f16h.3)
     - ``atol``, ``strtoul``, ``strtoll``, ``strtoull``, ``strtod``
     - 5 / 5
     - angr-f16h.3 (closed) — ``strtol.rs``, ``strtod.rs``
       (``strtod``: amd64 xmm0 / AArch64 v0 FP-return)
   * - Env mutation (angr-f16h.4)
     - ``setenv`` ✓, ``putenv`` ✓, ``unsetenv`` ✓, ``clearenv`` ✓
     - 4 / 4
     - angr-f16h.4 — ``getenv.rs`` (``NativeSetenv`` / ``NativePutenv``
       / ``NativeUnsetenv`` / ``NativeClearenv``)
   * - Extended strings (angr-f16h.5)
     - ``strnlen`` ✓, ``strncpy`` ✓, ``strncat`` ✓, ``strrchr`` ✓,
       ``strpbrk`` ✓, ``strspn`` ✓, ``strcspn`` ✓, ``strtok`` ✗
     - 7 / 8
     - ``strtok`` intentionally left on the Python fallback path
       (angr-4c65 wontfix). 14-bench fallback profile shows zero
       strtok appearances, so the stateful ``state.globals`` save
       pointer + ``strtok_r`` symbolic-write plumbing is not justified.
       ``strrchr``/``strpbrk``/``strspn``/``strcspn`` in
       ``strchr.rs`` and ``strset.rs``

Syscalls
^^^^^^^^

The native syscall set is currently focused on the AMD64 baseline
plus the per-arch tables in ``register_<arch>`` (X86, ARM, ARM64,
MIPS32 in ``syscalls/mod.rs``; MIPS64 has no syscall table yet —
``angr-smtv``). Every numbered handler below is registered for AMD64;
other arches re-register the same handler under each arch's syscall
number from its ``unistd_*.h``. The ``angr-0hif`` campaign expanded
the baseline I/O / memory / exit handlers with file-path, directory,
identity, memory-extras, fd-control, signals, and rlimit/concurrency
families. Symbolic-num dispatches and any unregistered concrete-num
still fall through ``stepping.rs::RunResult::Syscall`` to Python's
``engines/successors.py::_resolve_syscall``.

.. list-table::
   :header-rows: 1
   :widths: 25 35 15 25

   * - Group
     - Members
     - Native
     - Provenance
   * - I/O (baseline)
     - ``read``, ``write``
     - 2 / 2
     - ``read.rs``, ``write.rs`` — symbolic fd → Python fallback
   * - Memory (baseline)
     - ``brk``, ``mmap``, ``mprotect``, ``munmap``
     - 4 / 4
     - ``brk.rs``, ``mmap.rs`` (anonymous concrete-args fast path;
       MAP_FIXED collisions tracked by angr-ttr7), ``mprotect.rs``,
       ``munmap.rs``
   * - Process exit (baseline)
     - ``exit``, ``exit_group``
     - 2 / 2
     - ``exit.rs`` — terminal, routes state to ``STASH_DEADENDED``
   * - Time (baseline)
     - ``time``, ``gettimeofday``, ``clock_gettime``
     - 3 / 3
     - ``sim_time.rs`` — fresh-symbolic ``time_t``; CLOCK_REALTIME
       only (other clocks → Python)
   * - Signals (baseline, partial)
     - ``rt_sigaction``
     - 1 / 1
     - ``sigaction.rs`` — no-op return 0 (matches Python proc)
   * - amd64 TLS (baseline)
     - ``arch_prctl``
     - 1 / 1
     - ``arch_prctl.rs`` — fs_const / gs_const set/get
   * - File path (angr-0hif.1)
     - ``open``, ``close``, ``stat``, ``fstat``, ``lstat``, ``access``,
       ``readlink``, ``openat``, ``newfstatat``, ``readlinkat``,
       ``faccessat``
     - 11 / 11
     - angr-0hif.1 — ``file_path.rs`` (AMD64 numbers 2/3/4/5/6/21/89/
       257/262/267/269; re-registered under per-arch numbers for
       i386 / ARM / ARM64 / MIPS32)
   * - Directory (angr-0hif.2)
     - ``chdir``, ``fchdir``, ``getcwd``, ``mkdir``, ``rmdir``,
       ``unlink``, ``rename``, ``mkdirat``, ``unlinkat``, ``renameat``,
       ``renameat2``
     - 11 / 11
     - angr-0hif.2 — ``directory.rs`` (AMD64 numbers 79/80/81/82/83/
       84/87/258/263/264/316; per-arch tables alias each handler under
       the matching unistd number)
   * - Process identity (angr-0hif.3)
     - ``getpid``, ``getppid``, ``gettid``, ``getuid``, ``geteuid``,
       ``getgid``, ``getegid``, ``setuid``, ``setgid``
     - 9 / 9
     - ``identity.rs`` — getters return angr defaults (pid=1337,
       ppid=1336, uid/gid=1000). ``setuid``/``setgid`` (angr-pqgu)
       mirror Python's ``syscall_stub`` and emit a fresh symbolic BV
       via ``SyscallOutcome::ContinueSymbolic`` (no Python
       ``SimProcedure`` exists for them)
   * - Memory extras (angr-0hif.4)
     - ``madvise``, ``mremap``, ``msync``, ``mlock``, ``munlock``,
       ``mlockall``, ``munlockall``
     - 7 / 7
     - ``memory_extras.rs`` — none have a dedicated Python
       ``SimProcedure``; native handlers mirror ``syscall_stub`` and
       emit a fresh symbolic BV via
       ``SyscallOutcome::ContinueSymbolic``. ``mremap`` is a parity
       stub (does not update page tables — neither does Python angr)
   * - FD control (angr-0hif.5)
     - ``fcntl`` ✓, ``ioctl`` ✓, ``pipe`` ✓, ``pipe2`` ✓,
       ``dup`` ✓, ``dup2`` ✓, ``dup3`` ✓
     - 7 / 7
     - ``file_descriptor.rs``. ``fcntl``/``ioctl``/``pipe``/``pipe2``
       have no Python ``SimProcedure`` bound in the kernel library
       (``posix/fcntl.py`` is libc-side only), so native handlers
       mirror ``syscall_stub`` and emit a fresh symbolic BV via
       ``SyscallOutcome::ContinueSymbolic``. ``dup``/``dup2``/``dup3``
       are now wired natively (AMD64 numbers 32/33/292; per-arch
       tables register them under the matching unistd number)
   * - Signals + process control (angr-0hif.6)
     - ``kill``, ``tgkill``, ``rt_sigprocmask``, ``rt_sigaction``,
       ``rt_sigreturn``, ``pause``, ``alarm``
     - 6 / 7
     - ``signals.rs`` — ``kill`` / ``rt_sigreturn`` / ``pause`` /
       ``alarm`` mirror ``syscall_stub`` and emit a fresh symbolic BV
       via ``SyscallOutcome::ContinueSymbolic`` (no Python
       ``SimProcedure`` exists for them); ``tgkill`` returns concrete
       0 to match ``procedures/linux_kernel/tgkill.py``;
       ``rt_sigaction`` is the baseline ``sigaction.rs`` handler.
       ``rt_sigprocmask`` is NOT native — its Python impl mutates
       ``state.posix.sigmask`` which ``RustSimState`` does not carry,
       so it falls back to Python for parity
   * - Resource limits + concurrency (angr-0hif.7)
     - ``getrlimit``, ``setrlimit``, ``prlimit64``, ``futex``,
       ``eventfd``, ``eventfd2``, ``epoll_create``, ``epoll_create1``,
       ``epoll_ctl``, ``epoll_wait``
     - 10 / 10
     - ``rlimit.rs``, ``concurrency.rs`` — ``getrlimit`` mirrors
       ``procedures/linux_kernel/getrlimit.py`` (RLIMIT_STACK writes
       8388608 + symbolic ``rlim_max`` and returns 0; other
       resources return a fresh symbolic). ``futex`` mirrors
       ``procedures/linux_kernel/futex.py`` (FUTEX_WAKE family
       returns 0, else fresh symbolic). The remaining seven have no
       dedicated Python ``SimProcedure`` and mirror ``syscall_stub``
       with a fresh symbolic via ``SyscallOutcome::ContinueSymbolic``.
       angr is single-threaded symex; blocking on ``futex(FUTEX_WAIT)``
       or ``epoll_wait`` is never modeled. x86 / ARM also alias
       ``ugetrlimit (191)`` to ``getrlimit`` per
       ``procedures/linux_kernel/getrlimit.py::ugetrlimit``

Per-arch coverage of the baseline handlers (X86, ARM, ARM64, MIPS32)
is summarised in the ``register_<arch>`` blocks of
``syscalls/mod.rs``; the omitted entries (e.g. legacy ``mmap`` /
``mmap2`` on X86 / ARM / MIPS32) are documented inline above each
table.

Maintaining the matrix
^^^^^^^^^^^^^^^^^^^^^^

When you close an ``angr-f16h.*`` or ``angr-0hif.*`` child, flip the
``Native`` column for the affected procedure(s) and bump the
``M / N`` count, mirroring the pattern in
:doc:`rust_vex_ops` →
*Unsupported op coverage matrix*. When you add a brand-new family
(no campaign child yet), append a new row and open a tracking bead so
the provenance column has somewhere to point.

The ``Native`` count is intentionally fractional (``M / N``) rather
than a Status enum like the VEX matrix — every row already has a
Python fallback, so the meaningful axis is "how many of the named
members are on the fast path", not "implemented / placeholder /
stubbed". A row at ``0 / N`` means all members fall through to
Python; a row at ``N / N`` means none of its members enter the
Python fallback path under concrete arguments.

.. note::

   The *Native (Rust) SimProcedures* section (above, from
   *Native coverage matrix* through this paragraph) was last verified
   against commit ``4215fe99b`` on 2026-06-03 (angr-1cnv). When you
   close an ``angr-f16h.*`` or ``angr-0hif.*`` child, flip the matching
   row and bump this footer to the new commit hash.
