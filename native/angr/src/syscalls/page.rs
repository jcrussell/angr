//! Shared page/memory helpers for syscall handlers.
//!
//! `brk`, `mmap`, and `mprotect` all need to round addresses to/from
//! 4 KiB page boundaries. Previously each file carried an identical
//! `const PAGE_SIZE: u64 = 4096;` plus `const PAGE_MASK: u64 = PAGE_SIZE - 1;`
//! — consolidated here so any future page-size adjustment lives in one place.
//!
//! `mmap` and `mprotect` additionally shared a byte-identical PROT-bit
//! translation; `linux_prot_to_permission` now lives here for the same
//! reason (a future 4th PROT bit or `PROT_NONE` special-case can't be
//! applied to one copy and missed on the other).

use crate::memory::Permission;

/// Page size in bytes (4 KiB).
pub(crate) const PAGE_SIZE: u64 = 4096;

/// Mask for the low bits within a page. `addr & !PAGE_MASK` rounds down;
/// `(addr + PAGE_MASK) & !PAGE_MASK` rounds up.
pub(crate) const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Translate Linux PROT bits (0x1=R, 0x2=W, 0x4=X) into the internal
/// `Permission` struct. Shared by the `mmap` and `mprotect` handlers.
pub(crate) fn linux_prot_to_permission(prot: u64) -> Permission {
    Permission {
        read: prot & 0x1 != 0,
        write: prot & 0x2 != 0,
        execute: prot & 0x4 != 0,
    }
}
