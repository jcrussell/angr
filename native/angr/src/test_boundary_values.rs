//! Shared boundary-value tables for the integer-overflow/wraparound
//! proactive sweep ("Harness 6: Integer-overflow/wraparound boundary sweep"
//! in the "How many audit-found bugs could a more robust test suite have
//! caught?" test-coverage plan).
//!
//! ~18 call sites scattered across the crate do guest-controlled
//! address/offset/fd/width arithmetic that must saturate, wrap, or reject
//! rather than silently overflow — the angr-03vl4 audit round's
//! integer-overflow/wraparound bug family (e.g. angr-03vl4.34/.35/.36/.37/
//! .44/.52/.54/.55/.59/.60/.65/.66/.67/.68/.88, plus angr-03vl4.4/.31 for the
//! width-clamp half of the family). Every one of those historical fixes
//! shipped its own regression test pinned to the *one* value the bug report
//! happened to use. This module is the DRY counterpart: one boundary-value
//! table, invoked from a thin test at each call site, so a systematic sweep
//! runs everywhere the shape recurs instead of only the value that originally
//! triggered it.
//!
//! Deliberately just two free functions returning `Vec`s, not a
//! struct/trait/derive: the call sites are functionally unrelated (fd
//! tables, page arithmetic, VEX bit widths, syscall struct writers), so the
//! only thing worth sharing is the enumerated boundary set itself — forcing
//! more structure on top would be the "artificial mega-abstraction across
//! unrelated call sites" the plan explicitly warns off for this harness.
//!
//! Crate-wide `pub(crate)` (not scoped under `memory`/`state`/etc.) because
//! call sites span `memory`, `state::filesystem`, `syscalls`, `procedures`,
//! `interpreter`, and `symbolic` — no single existing module is an
//! appropriate home, and CLAUDE.md's Harness-6 sequencing note calls for one
//! table reused everywhere rather than a per-module copy.

/// Addresses/offsets worth sweeping at any guest-controlled `u64`
/// address-or-offset call site: `0`/`1` (degenerate), a couple of
/// page-boundary-adjacent values (`0x1000`/`0xFFF`), values near the top of
/// the 64-bit address space (the wraparound bug family's usual trigger —
/// `write_fallback_max` concretizes an under-constrained store pointer to
/// `u64::MAX` by default, so this is reachable, not theoretical — see
/// `end_page_inclusive`'s doc comment in `memory/mod.rs`), and the
/// `u32`/`i32` boundaries some call sites cast through (fd numbers, mmap
/// length checks on 32-bit arches).
///
/// Sorted ascending; callers that need page-aligned addresses should mask
/// with `& !0xFFF` themselves rather than expecting every entry here to
/// already be aligned (several deliberately are not, to also exercise
/// misalignment handling at call sites that care).
///
/// The `u64::MAX - N` entries for `N` in `{127, 63, 31, 15, 7}` exist
/// specifically for stride-N multi-field struct writers (a base address
/// plus several `wrapping_add(field_offset)` stores, e.g. a `getrlimit`
/// rlimit struct, an `fstat` stat buffer, a `gettimeofday`/`clock_gettime`
/// timeval/timespec pair): the `0x1000`/`0x2000`-below-top entries above are
/// too far from the top of the address space for any struct in that size
/// class to reach the wrap (every field lands unwrapped), while `MAX`/
/// `MAX - 1` are so close that even the *first* field's own store already
/// straddles the wrap — neither exercises the actual "some early field
/// lands unwrapped, a later field's address genuinely wraps to a low
/// address" case those writers exist to handle. Each `N` here is `2^k - 1`
/// (a multiple of 8, and so also of 4) below the top, so the split point
/// falls exactly on a field boundary for any writer whose fields are
/// 4- or 8-byte-aligned — no single field's own bytes straddle the wrap —
/// and the doubling spread (7, 15, 31, 63, 127) means at least one entry
/// leaves a struct's leading field(s) intact while wrapping a later one for
/// any struct up to a couple hundred bytes, not just the three call sites
/// that motivated adding them.
pub(crate) fn boundary_addresses() -> Vec<u64> {
    vec![
        0,
        1,
        0xFFF,  // one byte short of a page boundary
        0x1000, // one page
        i32::MAX as u64,
        (i32::MAX as u64) + 1,
        (u32::MAX as u64) - 1,
        u32::MAX as u64,
        (u32::MAX as u64) + 1,
        u64::MAX / 2,
        u64::MAX - 0x2000, // two pages below the top
        u64::MAX - 0x1000, // exactly one page below the top
        u64::MAX - 0xFFF,  // first byte of the last page
        u64::MAX - 127,    // struct-field partial-wrap window (see doc above)
        u64::MAX - 63,     //   ditto
        u64::MAX - 31,     //   ditto
        u64::MAX - 15,     //   ditto
        u64::MAX - 7,      //   ditto — also the classic "rlim + 8 == 0" pivot
        u64::MAX - 1,
        u64::MAX,
    ]
}

/// Widths worth sweeping at any call site with a `.min(128)`-style clamp, a
/// shift-by-width computation, or a byte-conversion loop: `0`/`1`
/// (degenerate), the `u128` concrete-BV storage ceiling straddled from both
/// sides (`127`/`128`/`129`), and the wider SIMD-scale widths (`255`/`256`)
/// that show up as VEX YMM operand widths.
///
/// Not every call site accepts a `0` width (e.g. a rotate-amount modulus)
/// — callers for which that is genuinely out of domain should filter it out
/// explicitly (with a comment saying why) rather than the table omitting it,
/// since other call sites (byte-conversion loops) *do* need to see it.
pub(crate) fn boundary_widths() -> Vec<u32> {
    vec![0, 1, 127, 128, 129, 255, 256]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_addresses_are_sorted_and_deduplicated() {
        let addrs = boundary_addresses();
        let mut sorted = addrs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            addrs, sorted,
            "table must be sorted ascending with no duplicates"
        );
    }

    #[test]
    fn boundary_widths_are_sorted_and_deduplicated() {
        let widths = boundary_widths();
        let mut sorted = widths.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            widths, sorted,
            "table must be sorted ascending with no duplicates"
        );
    }
}
