# Loop session notes (2026-05-08, 134th loop session)

## Task: angr-03ej — Hook length parameter not respected (silent re-execution)

### Status: investigated; bd diagnosis was incorrect — closing as NOT-A-BUG with regression test

### Bd's claim
> proj.hook(addr, proc, length=N>0) silently re-executes the original N
> bytes after the hook returns, causing wrong behavior or infinite loops.

### What I found
For the function-callable hook path (proj.hook(addr, my_func, length=N))
the length IS honored. project.py wraps the function in
`UserHook(user_func=hook, length=length)`. Inside UserHook.run(),
`self.successors.add_successor(self.state, self.state.addr + length, ...)`
sets the successor PC to hook_addr+length, which propagates through
the dispatch as `new_pc = succ_state.addr` and into Rust via
`resume_after_simprocedure(new_pc, ...)` → `state.set_pc(new_pc)`.

Verified by adding instrumentation (eprintln in resume_after_simprocedure
and run loop). For hook(0x8, fn, length=2), debug showed:
  state 1 pc 0x8 -> 0xa     (resume sets PC correctly)
  state 1 popped at pc=0xa  (next iter sees PC=0xa)

Also confirmed via new test test_hook_length_advances_pc_userhook
(fauxware mov rsp,rbp at main+1, 3 bytes, length=3): hook fires once
per path, not in a loop. **Test passes** without any code change.

### Real bug exposed during investigation (separate)
A simpler shellcode test (hook 0x8 with length=2, no second hook)
DOES exhibit hook re-firing — but root cause is unrelated to
hook_length plumbing. After the hook resumes at PC=0xa, the basic
block runs through `ret` at 0x1a; the stack contains an unconstrained
symbolic byte, and the Rust engine appears to concretize the popped
PC to 0x0 and continues looping back through the hook. Python angr
correctly recognizes the unconstrained PC and moves the state to the
`unconstrained` stash. New bead filed for this.

### Files changed
- tests/engines/test_rust_exploration.py: added
  test_hook_length_advances_pc_userhook (confirms the bug-as-described
  does not exist for UserHook-wrapped function callbacks).

### Build note
Z3 header path in .cargo/config.toml is broken (PyPI z3-solver doesn't
ship headers). Built with Z3_SYS_Z3_HEADER=/usr/include/z3.h (system
package) and copied target/release/librustylib.so to angr/. This is
the same issue tracked in angr-8fsl (P1).

### Tests
All 262/262 pass.
