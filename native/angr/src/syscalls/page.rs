//! Shared page/memory helpers for syscall handlers.
//!
//! `brk`, `mmap`, and `mprotect` all need to round addresses to/from
//! 4 KiB page boundaries. Previously each file carried an identical
//! `const PAGE_SIZE: u64 = 4096;` plus `const PAGE_MASK: u64 = PAGE_SIZE - 1;`
//! — consolidated here so any future page-size adjustment lives in one place.
//! Since angr-sqfj8.140 this module does not define its own copy either: it
//! re-exports `crate::memory`'s, which the paging engine itself uses, so a
//! syscall handler and the memory model can never disagree about page size.
//!
//! `mmap` and `mprotect` additionally shared a byte-identical PROT-bit
//! translation; `linux_prot_to_permission` now lives here for the same
//! reason (a future 4th PROT bit or `PROT_NONE` special-case can't be
//! applied to one copy and missed on the other).

use crate::memory::Permission;

/// Page size in bytes (4 KiB) and the mask for the low bits within a page:
/// `addr & !PAGE_MASK` rounds down, `(addr + PAGE_MASK) & !PAGE_MASK` rounds
/// up. Re-exported from `crate::memory::page` — the single definition — so
/// `super::page::{PAGE_SIZE, PAGE_MASK}` keeps resolving for the `brk` /
/// `mmap` / `mprotect` handlers. [`PageIndex`] rides along for the same
/// reason: `mprotect` converts byte addresses to the page numbers the raw
/// `memory/` page API is keyed by, and open-coding that as `>> 12` is the
/// drift angr-c7xno.49 tracked.
pub(crate) use crate::memory::{PAGE_MASK, PAGE_SIZE, PageIndex};

/// Translate Linux PROT bits (0x1=R, 0x2=W, 0x4=X) into the internal
/// `Permission` struct. Shared by the `mmap` and `mprotect` handlers.
pub(crate) fn linux_prot_to_permission(prot: u64) -> Permission {
    Permission {
        read: prot & 0x1 != 0,
        write: prot & 0x2 != 0,
        execute: prot & 0x4 != 0,
    }
}
