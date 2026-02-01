# VEX Ops Implementation Task (All Architectures)

## Instructions
1. Run differential tests for all architectures:
   pytest tests/engines/differential/ -v 2>&1 | head -200

2. Find the FIRST test failing due to unsupported VEX operation
   - Note which architecture it's for

3. Implement the missing operation:
   - Add to IROp enum in `native/angr/src/vex/ir.rs` (if needed)
   - Implement in `native/angr/src/vex/ops.rs`
   - Reference Python impl: `angr/engines/vex/claripy/irop.py`

4. Ensure the operation works for all relevant widths:
   - I8, I16, I32, I64 for basic ops
   - F32, F64 for float ops
   - V128, V256 for vector ops

5. Add unit test in ops.rs

6. Re-run the failing test to verify fix

7. Commit: "rust-vex: implement <OP_NAME>"

8. Exit (loop restarts for next op)

## Rules
- One operation per iteration
- Verify with tests before committing
- Use IROp::Raw(u32) as fallback for complex ops
- Operations should be architecture-agnostic where possible
