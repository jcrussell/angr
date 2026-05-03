# Loop session notes (2026-05-03, twenty-seventh session)

## Task: angr-3ijo (spike) — CLOSED no-go
"Consider bincode instead of JSON for VEX IRSB serialization."

## Outcome
NO-GO. Microbench shows IRSB deserialize is 0.8% of total runtime
even on the lift-heaviest benchmark (codegate_2017-angrybird). Bincode
or msgpack would not deliver meaningful speedup. Spike threshold was
">20% of lift time"; actual is ~10% of lift, ~1% of total.

## Hard data
- Captured 500 representative IRSBs (avg 3KB JSON each) from a real
  codegate run via temporary `CAPTURE_IRSB=path` env hook in
  `serialize_irsb()`.
- Built `native/angr/examples/bench_deserialize.rs` to time pure
  deserialize on those samples.
- Results per call: full deserialize=17 µs, json parse=12.8 µs (75%),
  PyVex→IRSB convert=4.2 µs (25%).
- For codegate (1282 lifts, 2.72s total): total deserialize = 21.8 ms
  = 0.8% of total. Bincode at 5x parse speed: ~13 ms savings = 0.48%.

## Files
- `native/angr/examples/bench_deserialize.rs` (new, 70 lines): retained
  as one-shot profiling utility for future "is X a bottleneck?" spikes.

## Memories saved
- `irsb-deserialize-bottleneck`: full data + per-benchmark breakdown
- `bench-deserialize-example`: how to use the bench harness

## Next ready (P3, no blockers)
- angr-w4os: Python bridge cleanup w/ state cache eviction
- angr-cbko: native exit/abort SimProcs (WARNING: avoid memory)
- angr-8em4: panic audit (large, needs subtask split)
- angr-bgv0: Z3 floating point theory
- angr-awm3: CAS/LLSC statement handling
