//! Shared page-size constants for syscall handlers.
//!
//! `brk`, `mmap`, and `mprotect` all need to round addresses to/from
//! 4 KiB page boundaries. Previously each file carried an identical
//! `const PAGE_SIZE: u64 = 4096;` plus `const PAGE_MASK: u64 = PAGE_SIZE - 1;`
//! — consolidated here so any future page-size adjustment lives in one place.

/// Page size in bytes (4 KiB).
pub(crate) const PAGE_SIZE: u64 = 4096;

/// Mask for the low bits within a page. `addr & !PAGE_MASK` rounds down;
/// `(addr + PAGE_MASK) & !PAGE_MASK` rounds up.
pub(crate) const PAGE_MASK: u64 = PAGE_SIZE - 1;
