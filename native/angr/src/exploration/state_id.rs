//! `StateId` newtype for state identifiers in the exploration subsystem.
//!
//! Wraps a `u64` so the type system can distinguish a *state* identifier
//! (the monotonic ID minted by `RustSimState::state_id()` and used as the
//! key for stash membership and lineage roots) from the many other `u64`
//! values that flow through this subsystem — instruction addresses, hook
//! addresses, step counts, condition IDs. Confusing a state ID for an
//! address was a real footgun in the accessor cluster (`find_state`,
//! `with_state`, ...), where both arrive as bare `u64`.
//!
//! The newtype is scoped to `exploration/`. The neighbouring `StashManager`
//! (`stash.rs`) keeps its `state_index` / `state_roots` maps keyed by raw
//! `u64` — it is a different module and its public API is the boundary at
//! which `StateId` is unwrapped via [`StateId::raw`]. The Python-facing
//! `#[pyclass]` methods (`state_api.rs`) and getters that return IDs to
//! Python likewise keep `u64` signatures, mirroring the `Address` newtype's
//! rule that boundary/Display surfaces stay raw.
//!
//! ## Migration pattern (mirrors `memory/address.rs`)
//!
//! 1. Define the newtype with `Copy + Clone + Hash + Eq + Ord` derives and
//!    `From<u64>` / `From<Self> for u64`.
//! 2. Make the subsystem's accessor APIs take `impl Into<StateId>`. Rust's
//!    literal-type inference and the blanket `T: Into<T>` mean existing
//!    `u64` callers (e.g. the Python-boundary `_xxx(state_id: u64)` methods)
//!    compile unchanged while internal code can pass a `StateId`.
//! 3. Use `StateId` for state-identity locals and exploration-owned fields
//!    so intent is documented. Convert to `u64` at the `StashManager`
//!    boundary with [`StateId::raw`].
//! 4. Surfaces that hand IDs to Python or format them keep `u64` (the
//!    `Display`/`LowerHex` impls below let a `StateId` slot into existing
//!    `{}` / `{:x}` messages without an explicit `.raw()`).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]

/// A monotonic state identifier within the exploration subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct StateId(pub u64);

impl StateId {
    #[inline]
    pub const fn new(id: u64) -> Self {
        StateId(id)
    }

    /// Get the underlying raw state ID (the `StashManager` boundary value).
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl From<u64> for StateId {
    #[inline]
    fn from(v: u64) -> Self {
        StateId(v)
    }
}

impl From<StateId> for u64 {
    #[inline]
    fn from(s: StateId) -> Self {
        s.0
    }
}

impl std::fmt::Display for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::fmt::LowerHex for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.0, f)
    }
}
