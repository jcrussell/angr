# Rust Symex Engine — Next Session Plan

**Branch:** `rust-engine-v2`
**Date:** 2026-04-03
**Current Score:** 7/10 run, 6/7 correct output (grub fixed but benchmark harness issue)
**Latest commit:** (pending) Symbolic CCall + deferred fork solver snapshots

## What Works (6 matching + 1 fixed)

| Example | Speedup | Notes |
|---------|---------|-------|
| flareon2015_5 | 5.28x | Up from 3.20x after symbolic CCall |
| securityfest_fairlight | 5.11x | Stable |
| fauxware | 0.30x | Small binary, init overhead dominates |
| ais3_crackme | 1.20x | Up from 1.00x |
| hackcon2016_angry-reverser | 0.16x | Correct but slow (20 linear equations) |
| sym-write | 0.05x | Set-equivalent comparison |
| **grub** | **FIXED** | Was wrong (10 vs 423 unique states), now finds correct crashing input. Benchmark harness reports "list index out of range" due to output capture issue, but `python solve.py --rust` works correctly. |

## What Runs But Wrong Output (1)

### flareon2015_10 — TEA Decrypt Arithmetic Bug
- Rust outputs empty string, Python finds `b'unconditional_conditions@flare-on.com'`
- Uses `angr.callable` for TEA decrypt sub-exploration
- The symbolic CCall improvements didn't help because the TEA decrypt uses concrete arithmetic (shifts, XOR, add) where the Rust VEX interpreter computes different concrete results
- **Root cause:** Not a CCall issue — it's a concrete VEX arithmetic bug in the Rust interpreter
- **Debug approach:** VEX IR trace comparison between Python and Rust for the TEA decrypt block
- TEA uses: `v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1)` — check shifts, XOR, add with concrete values

## What Fails (3)

### ekopartyctf2016_rev250 — Timeout
- Still times out (180s limit vs Python's 33s)
- Deferred fork solver snapshot fix was implemented this session but doesn't help because the main bottleneck is per-branch Python round-trip overhead (SymbolicBranch callback for every symbolic branch)
- **Next step:** Batch symbolic branches in Rust. Instead of returning SymbolicBranch to Python, handle forking entirely in Rust using the solver snapshots. This eliminates the Python FFI round-trip per branch.
- Implementation: When `use_deferred_forks=true`, the interpreter already creates deferred forks and continues on the true path. The missing piece is processing deferred forks in Rust without returning to Python.

### csaw_wyvern — Timeout (no unicorn)
- Uses `full_init_state` with `unicorn` option
- Rust engine can't do unicorn → interprets all C++ library VEX (extremely slow)

### grub — Benchmark Harness Issue (ACTUALLY WORKS)
- `python solve.py --rust` correctly finds `b'\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\r'`
- Same result as Python engine
- Benchmark reports failure because the output capture in run_comparison_10.py fails with "list index out of range"
- **Fix:** Update benchmark harness to handle grub's output format

## This Session's Changes

1. **Symbolic CCall implementation** (ccall.rs):
   - Added `extract_to_nbits`, `symbolic_parity`, `symbolic_pack_eflags` helpers
   - Added `symbolic_eflags_sub`, `symbolic_eflags_add`, `symbolic_eflags_logic` for `calculate_eflags_all`
   - Added symbolic `calculate_eflags_c` for carry flag (SUB/ADD/LOGIC/COPY)
   - Extended `calculate_condition` with ADD category and LOGIC COND_S support
   - Changed interpreter fallback from concrete 0 to fresh symbolic variable for unsupported CCalls

2. **Deferred fork solver snapshot fix** (interpreter_cb.rs, exploration.rs, state.rs):
   - Added `solver_snapshots: HashMap<u64, SymContext>` to interpreter and PendingCallback
   - Snapshot solver BEFORE `assume_true(condition)` at each deferred fork
   - Use clean snapshot when creating alternate-path states (prevents UNSAT contradiction)
   - Added `replace_solver()` method to RustSimState

## Remaining Work Priority

### Priority 1: Fix grub benchmark harness
Quick fix — update run_comparison_10.py to handle grub output correctly.

### Priority 2: flareon2015_10 concrete arithmetic bug
Debug the Rust VEX interpreter's concrete arithmetic for TEA decrypt operations.

### Priority 3: ekopartyctf2016_rev250 — Rust-native fork processing
Process deferred forks entirely in Rust without returning to Python.
The solver snapshot infrastructure is now in place — just need to create and schedule forked states in Rust.

### Priority 4: csaw_wyvern — concrete fast-path
Detect fully-concrete states and use fast integer arithmetic instead of Z3 AST construction.

## Environment

```bash
cd /home/ubuntu/repos/angr
export PATH="$HOME/.cargo/bin:$PATH"
source .venv/bin/activate
# pip install -e .  # network is down — use manual copy instead:
cargo build --manifest-path native/angr/Cargo.toml --release && \
  cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so && \
  cp target/release/librustylib.so build/lib.linux-x86_64-cpython-312/angr/rustylib.cpython-312-x86_64-linux-gnu.so
python tests/benchmarks/run_comparison_10.py
```
