//! `Address` newtype for byte addresses in the memory subsystem.
//!
//! Wraps a `u64` so the type system can distinguish a byte address from a
//! page number, a size, or an offset within a page. The newtype is scoped
//! to `memory/` — page numbers (addr >> 12) still flow as bare `u64`
//! because page-keyed structures (`pages`, `dirty_pages`, `lazy_regions`)
//! live at a different conceptual layer.
//!
//! Public memory APIs accept `impl Into<Address>`. Rust's literal-type
//! inference picks `u64` automatically when `From<u64>` is the only
//! conversion in scope, so callers like `mem.map(0x1000, ...)` and
//! `mem.store_concrete(addr, ...)` compile unchanged.
//!
//! ## Migration pattern (for future per-subsystem newtype carve-outs)
//!
//! 1. Define the newtype in `<subsystem>/<name>.rs` with `Copy + Clone +
//!    Hash + Eq + Ord` derives and `From<u64>` / `From<Self> for u64`.
//! 2. Make public subsystem APIs take `impl Into<Newtype>`. Internal
//!    helpers take the newtype directly.
//! 3. Migrate address-keyed `HashMap` / `HashSet` collections to the
//!    newtype. Conceptually distinct `u64` values (page numbers, sizes)
//!    stay raw.
//! 4. Error variants keep `u64` so existing `Display` formatting and
//!    `{addr:x}` interpolation continue to work.

use crate::memory::page::{PAGE_MASK, PageIndex};

/// A 64-bit byte address.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Address(pub u64);

impl Address {
    #[inline]
    pub const fn new(addr: u64) -> Self {
        Address(addr)
    }

    /// Get the underlying byte address.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Page number this address belongs to (`addr >> 12`).
    ///
    /// Shift comes from [`PageIndex::SHIFT`], which derives it from
    /// `PAGE_SIZE`, so this cannot drift from the page size the rest of
    /// `memory/` uses (angr-c7xno.49).
    #[inline]
    pub const fn page_num(self) -> u64 {
        self.0 >> PageIndex::SHIFT
    }

    /// First byte address of the page this address belongs to.
    ///
    /// The inverse of [`page_num`](Self::page_num); exists so call sites that
    /// need to get back from a page number to its base address do not
    /// open-code the `<< 12` (angr-c7xno.49).
    #[inline]
    pub const fn page_base(self) -> u64 {
        self.page_num() << PageIndex::SHIFT
    }

    /// Offset within the page (`addr & PAGE_MASK`).
    #[inline]
    pub const fn page_offset(self) -> u16 {
        (self.0 & PAGE_MASK) as u16
    }
}

impl From<u64> for Address {
    #[inline]
    fn from(v: u64) -> Self {
        Address(v)
    }
}

impl From<Address> for u64 {
    #[inline]
    fn from(a: Address) -> Self {
        a.0
    }
}

impl std::ops::Add<u64> for Address {
    type Output = Address;
    #[inline]
    fn add(self, rhs: u64) -> Address {
        Address(self.0.wrapping_add(rhs))
    }
}

impl std::ops::Sub<Address> for Address {
    type Output = u64;
    #[inline]
    fn sub(self, rhs: Address) -> u64 {
        self.0.wrapping_sub(rhs.0)
    }
}

impl std::ops::Sub<u64> for Address {
    type Output = Address;
    #[inline]
    fn sub(self, rhs: u64) -> Address {
        Address(self.0.wrapping_sub(rhs))
    }
}

impl std::fmt::LowerHex for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}
