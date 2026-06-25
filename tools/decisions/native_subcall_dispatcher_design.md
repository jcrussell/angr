# Design spike: native SimProcedure sub-call (ADDS_EXITS) dispatcher mechanism

Status: **spike / design — needs human go/no-go before implementation**
Filed: 2026-06-25 (iter 42, autonomous)
Blocks (drainable once landed): `angr-xxukz` (native `pthread_once`),
`pthread_create` `static_exits`, and any future native proc that must invoke a
guest routine and resume (e.g. `atexit`/`__cxa_atexit` handler replay,
`qsort`/`bsearch` comparator callbacks).

## Problem

Native procs today are **return-only**. The contract is
`NativeSimProcedure::call(&self, state, args) -> Result<Option<RustBV>, ProcedureError>`
(`native/angr/src/procedures/mod.rs`, `trait NativeSimProcedure`). The dispatcher
in `native/angr/src/exploration/stepping.rs` (the `Ok(ret_val)` arm around
`native_proc.call`, ~L781-806) consumes that as:

1. set the return register to `ret_val` (unless `no_return()`),
2. `state.set_pc(return_addr)`,
3. pop the stack (`sp += ptr_size`).

There is no way for a proc to say "jump into guest routine `func`, run it, then
resume *me* and let me finish." That is what angr's Python `SimProcedure.call()`
does — e.g. `pthread_once` (`angr/procedures/posix/pthread.py:72-86`):

```python
def run(self, control, func):
    ...
    controlword |= 2
    self.state.mem[control].char = controlword
    self.call(func, (), "retsite", prototype="void x()")   # <-- sub-call + resume
    return None
def retsite(self, control, func):
    return 0
```

`self.call(...)` pushes a synthetic return address, jumps to `func`, and when
`func` returns control re-enters the SimProcedure at the named continuation
(`retsite`), which returns 0. In angr core this is surfaced statically via
`ADDS_EXITS` / `static_exits` so the CFG knows the proc adds a call edge.

## Why it is not a proc addition (the iter-41 / xxukz finding)

The blocker is **the continuation must survive across interpreter steps and
across fork/snapshot**, but native procs are stateless `Arc<dyn NativeSimProcedure>`
shared across all states. A Rust closure capturing "what to do after func
returns" cannot be stored on the state — state must be `Clone` (fork) and
round-trip through the snapshot harness. So the continuation has to be encoded
as **plain data on the state**, not as a closure.

## Proposed design (data-encoded continuation, re-entrant proc)

### 1. New proc outcome variant

Widen the proc return type from `Option<RustBV>` to a small enum (keep the
existing two cases as variants so every current proc is a one-line mechanical
port, or — lower blast radius — add a sibling trait method
`call_ex` that defaults to delegating to `call`):

```rust
pub enum ProcOutcome {
    Return(Option<RustBV>),        // today's Ok(Some/None)
    CallAndResume {                // NEW
        target: u64,               // guest routine entry (func)
        args: Vec<RustBV>,         // args to pass per calling convention
        resume_tag: u32,           // which continuation to run on return
    },
}
```

`resume_tag` replaces Python's named-method continuation (`"retsite"`). A proc
that does sub-calls declares its continuations as `match resume_tag { ... }`
inside `call` (which is re-entered on resume — see step 3).

### 2. Per-state native resume stack

Add to `RustSimState` a serializable field (mirror the getopt-cursor field
plumbing landed in bhk0a.3 — decl + ctors + every fork site + snapshot
round-trip + `RustStateProxy` passthrough if Python needs visibility):

```rust
struct NativeResumeFrame { proc_name: String, resume_tag: u32, saved_args: Vec<RustBV> }
// Vec<NativeResumeFrame> on the state; LIFO.
```

`saved_args` carries the original proc args the continuation needs (pthread_once's
`retsite(control, func)` re-receives them; native equivalent reads them from the
frame). `Vec<RustBV>` already forks/snapshots (it is what proc args are).

### 3. Dispatcher: push resume sentinel, jump to target

In `stepping.rs`, when `call` returns `CallAndResume { target, args, resume_tag }`:

- choose a **resume sentinel address** — a reserved extern-style address the
  interpreter recognises as "re-dispatch the top native resume frame" (reuse the
  existing extern/PLT hook-address space the dispatcher already special-cases at
  the top of the step loop; this is the same machinery that fires native procs
  for out-of-binary call targets, so a sentinel in that range fires the resume
  path with no new exit-detection plumbing),
- push `NativeResumeFrame { proc_name, resume_tag, saved_args: args_orig }`,
- write the guest args into arg registers / stack per
  `environment.calling_convention`,
- push the **sentinel** as `func`'s return address (so when `func` rets it lands
  on the sentinel), having first ensured the *original* caller return_addr is
  still below it on the stack,
- `state.set_pc(target)`.

When the interpreter later steps onto the sentinel: pop the top
`NativeResumeFrame`, look up `proc_name` in the registry, and call it with a
"resume" marker carrying `resume_tag` + `saved_args`. The proc runs its
continuation arm and returns `ProcOutcome::Return(Some(0))` (pthread_once case),
which flows through the existing return path (set ret reg, PC = original
return_addr, pop).

### 4. Re-entry signalling

`call` needs to know it is being resumed vs freshly invoked. Cleanest: a second
trait method `resume(&self, state, resume_tag, saved_args) -> Result<ProcOutcome,
ProcedureError>` with a default `Err(NotImplemented)` (so only sub-call procs
implement it — SOLID, no churn on the 100+ existing procs). The dispatcher calls
`call` on fresh entry and `resume` on a sentinel hit.

## pthread_once worked example (the xxukz acceptance)

```
call(state, [control, func]):
    cw = load_u8(control)
    if symbolic(cw & 2): return Err(SymbolicArgument)   // defer to Python
    if cw & 2 != 0: return Ok(Return(Some(0)))           // already run
    store_u8(control, cw | 2)
    return Ok(CallAndResume { target: func, args: [], resume_tag: 0 })

resume(state, resume_tag=0, saved_args=[control, func]):
    return Ok(Return(Some(0)))                           // == retsite
```

DRY: matches Python `pthread_once.run` / `retsite` line for line.

## Risks / open questions for the human

1. **Stack discipline correctness.** The sentinel-as-return-address trick must
   not corrupt the guest's own frame layout; validate `sp` accounting against a
   real binary that calls a non-trivial init routine (not just a leaf).
2. **Fork during sub-call.** If `func` forks (branches) before returning, every
   child carries the resume frame (it is on the state, so it forks for free) —
   but each child independently hits the sentinel and resumes. Confirm that is
   the desired semantics (it matches Python: the callstack/return-site is
   per-state).
3. **Symbolic / multi-valued `func`.** Defer to Python (`SymbolicArgument`) —
   same gate as every other proc.
4. **`static_exits` / CFG.** Pure-symex (RustExplorationManager) does not build a
   CFG, so `ADDS_EXITS` static-exit reporting is **out of scope** for the symex
   engine; only the dynamic sub-call matters here. Note this explicitly so the
   bead is not over-scoped.
5. **Outcome-enum vs sibling-trait choice.** `ProcOutcome` enum is cleaner but
   touches every proc's signature; the `call_ex`/`resume` default-method pair is
   zero-churn. Recommend the default-method pair (KISS, smallest diff).

## Recommended slicing (getopt-pattern, one bead per session)

- **S1 (foundation, low risk):** `NativeResumeFrame` + per-state `Vec` field with
  full fork/snapshot/proxy plumbing + cargo fork-isolation & snapshot
  round-trip tests. No dispatcher change yet. Mirrors bhk0a.3.
- **S2 (dispatcher core):** resume sentinel recognition + `CallAndResume` handling
  + `resume` trait method, exercised by a single hand-built test proc that calls
  a tiny guest routine and resumes. No real proc yet.
- **S3 (xxukz):** port `pthread_once` to use S1+S2; e2e test against a binary that
  calls `pthread_once`. Then `pthread_create` `static_exits` as a follow-on.

S1 is the only near-zero-risk slice; S2 is the genuinely hard interpreter work
and should not be attempted blind — this doc is its design input.
