# cargo-fuzz targets for the pure hostile-input parsers

Dev-only [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) project
(bd `angr-qwyti.9`) covering the narrow, pure, Z3-free parsing surfaces where
width/truncation bugs have historically clustered. **Not** wired into CI or
`pip install`; run it locally when touching those parsers.

## Targets

| Target | Function(s) under test | Prior bugs in this surface |
|--------|------------------------|----------------------------|
| `format_parsers` | `procedures::format_common::{parse_width_digits, parse_length_modifier}` | angr-vfhyx, angr-n0irt.4 |
| `bv_numeral_parsers` | `symbolic::parse::{parse_wide_hex_low128, parse_wide_binary_low128, parse_hex_to_bytes, parse_binary_to_bytes, parse_decimal_to_bytes}` | angr-ph300.34/.35 |
| `optstring_parser` | `procedures::getopt::parse_optstring` | none yet — first coverage of this parser |

The targets reach these functions through the `fuzz_api` module in the main
crate, which is compiled **only** under the `fuzzing` cargo feature (see
`Cargo.toml`), so a stock build never gains this surface.

## Running

cargo-fuzz needs a nightly toolchain (already pinned/installed as
`nightly-x86_64-unknown-linux-gnu`) and the same Z3 env the normal build uses
(`setup.py` sets these; a bare `cargo fuzz` invocation does not):

```bash
cargo install cargo-fuzz          # one-time; installs the cargo-fuzz binary

export Z3_SYS_Z3_HEADER=/usr/include/z3.h
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/z3/lib:$LD_LIBRARY_PATH"

cd native/angr
cargo +nightly fuzz run format_parsers      -- -max_total_time=60
cargo +nightly fuzz run bv_numeral_parsers  -- -max_total_time=60
cargo +nightly fuzz run optstring_parser    -- -max_total_time=60
```

Add `--sanitizer none` for a faster link-only smoke check without AddressSanitizer.

## Feasibility notes (the SPIKE result)

- **Toolchain cost is low.** nightly is already installed, so cargo-fuzz adds no
  *new* pinned toolchain beyond the one-time `cargo install cargo-fuzz`. The
  fuzz crate detaches from the repo workspace (its own `[workspace]` table) and
  re-declares the `z3 = { path = ... }` patch, because a detached workspace does
  not inherit the root `[patch.crates-io]`.
- **z3 / PyO3 linkage works.** Linking the whole `rustylib` rlib (PyO3 + z3-sys)
  under nightly + ASAN builds and runs cleanly here; RSS stayed ~520 MB and
  throughput was ~500k exec/s for the format target. Z3 itself is a prebuilt
  `.so`, so ASAN does not instrument it.
- **Verdict:** worth keeping as a dev-only, locally-invoked harness. A blocking
  CI lane is *not* recommended yet — it would need the nightly toolchain plus
  the Z3 env wired into CI for marginal gain over the existing quickcheck
  property tests. Revisit if a new untrusted-input parser lands.

## Contract caveat when fuzzing `pub(super)` internals

These decoders assume inputs their *real* callers always satisfy (Z3 emits
ASCII numerals; `parse_binary_to_bytes`'s width always covers the bit count).
Fuzzing them directly, below that contract, produces false positives — e.g. a
`debug_assert!` tripwire firing, or `parse_hex_to_bytes` panicking on multibyte
UTF-8. The harnesses deliberately respect those contracts (ASCII-only payload,
width bumped to cover the bit count) so a reported crash means a *real* defect,
not a contract the production caller never breaches.
