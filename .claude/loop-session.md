## Session log: 2026-05-14 (cont) — angr-qz16 audit rust_only=True candidates

### Status: completed

### Outcome
Property-fuzzer-driven audit identified 6 FAST_SUITE entries that consistently
pass output comparison (10/10 trials with mixed bfs/dfs strategies). Promoted
those entries (rust_only=False) so future regression runs without --rust-only
will reinstate Python output diff.

### Promoted (FAST_SUITE, rust_only False)
- defcamp_r100 (bfs+dfs)
- ais3_crackme
- google2016_unbreakable_0
- strcpy_find
- flareon2015_2
- defcon2016quals_baby-re

### Kept rust_only=True (consistent output divergence)
- fauxware (rust outputs empty — Python finds SOSNEAKY)
- google2016_unbreakable_1 (bimodal)
- unmapped_analysis (exact-mismatch 3/3)
- csgames2018 (exact-mismatch 3/3)
- whitehatvn2015_re400 (exact-mismatch 3/3)

### MEDIUM_SUITE audit
None of the 4 non-bimodal candidates pass: flareon2015_5, ekopartyctf2016_rev250,
csaw_wyvern, codegate_2017-angrybird all show consistent exact-mismatch.

### Pre-state notes
- Reverted previous session's angr-ctct debug logs (no fix shipped).
- angr-ctct released back to open.
