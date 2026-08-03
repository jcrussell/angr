# Naming Conventions for `native/angr/src`

These are the dominant conventions in the Rust crate, derived from the
existing code (counts as of 2026-05-07). Use them in new code so the
crate reads consistently. Existing code that already follows them does
not need to be touched; existing code that doesn't should be migrated
opportunistically when you're already touching that file.

## Parameter names

| Type                       | Canonical name | Notes                                    |
| -------------------------- | -------------- | ---------------------------------------- |
| `u64` (machine address)    | `addr`         | 240 sites; `address` only in `segmentlist.rs` (see below) |
| `&SymContext`              | `ctx`          | 166 sites; never `context` or `cx`       |
| `Python<'_>`               | `py`           | 102 sites                                 |
| `&mut RustSimState`        | `state`        | 85 sites                                  |
| `&[u8]`                    | `data`         | 19 vs 11 `bytes`; `data` for any content stored in memory |
| `&RustBV` (generic BV)     | `bv`           | 32 sites; for BV-manipulation primitives  |
| `RustBV` / `&RustBV` (stored) | `value`     | 28 sites; for "the value being stored or returned" |
| `u32` / `u64` / `usize` size | `size`       | Pick the integer width that matches the surrounding API; don't rename across signatures |

Compound names like `addr_val: &RustBV` (a BV holding an address) and
`data_val: &RustBV` (a BV holding stored data) are fine where the extra
qualifier helps a reader understand the role.

## Method prefixes

- `call_*` — methods on `PythonCallbacks` that invoke a Python callback
  through the GIL. (`call_memory_store`, `call_lift_block`, etc.)
- `has_*` — boolean tests for whether an optional callback is set, or
  whether a state has a particular feature.
- `is_*` — boolean predicates on a value (`is_symbolic`, `is_concrete`).
- `try_*` — fallible variant of an operation that has a default path
  (`try_read_concrete_memory`).
- `add_*` / `set_*` — `add_*` for accumulating into a collection;
  `set_*` for replacing a single field.

## Local bindings

Within VEX op handlers (`vex/ops/mod.rs`, `vex/ccall.rs`) short local
names like `val`, `vec`, `lo`, `hi`, `lhs`, `rhs` are encouraged — the
function bodies are already small and the math is easier to read with
short names. Don't carry these short names out to public APIs.

## RustBV ownership

- `&RustBV` — when the caller continues to use the value after the
  call. Prefer this for read-only operations.
- `RustBV` (owned) — when the implementation will combine the
  argument with others and return a new BV (e.g., binary ops).
- `mut RustBV` — rare; only when the implementation rewrites the
  bitvector in place.

## Intentional exceptions

### `segmentlist.rs` uses `address: u64`

`Segment` and `SegmentList` are PyO3-exposed classes that mirror the
Python `angr.misc.segment_list` API. The Python class uses `address`
as the keyword argument name in its public stubs
(`angr/rustylib/__init__.pyi`). Renaming would be a public API break,
so these stay `address`. New PyO3-exposed methods that mirror an
upstream Python API should follow the upstream parameter name even if
that conflicts with the table above.

### `value: &RustBV` (not `bv: &RustBV`) in memory store

Memory-store APIs (`call_memory_store_symbolic_value`,
`store_strided`) take `value: &RustBV` because the semantic role of
the parameter is "the value being stored", not "an arbitrary BV".
This is a meaningful distinction at a call site: a reader recognizes
"store a value at an address" faster than "store a BV at an address".

### `bytes: &[u8]` (not `data: &[u8]`) in lift / decode paths

VEX lifting and instruction decoding APIs use `bytes: &[u8]` because
the parameter is specifically machine-code bytes, not arbitrary data.
`data` would be misleading.

## Rule of thumb

When a new parameter doesn't match any case above, look at three
nearby signatures in the same file. If they're consistent, follow
them. If they're not, prefer the canonical name from the table.

## What this doc is not

- Not a style guide for identifier capitalization — `rustfmt` and
  `clippy` enforce that.
- Not a clippy lint config — clippy doesn't have lints for parameter
  naming, so consistency here is enforced by review.
- Not exhaustive — when you find a recurring case that isn't covered,
  add it. Try to keep this file under 200 lines.
