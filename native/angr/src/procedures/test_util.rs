//! Shared `#[cfg(test)]` builders for procedure unit tests.
//!
//! Only the genuinely-identical state-construction core lives here; the
//! per-family helpers with custom signatures (e.g. `strset`'s `(s, set)`
//! setup or `stdio`'s `setup_file_struct`) stay local to their module.

use crate::memory::Permission;
use crate::state::RustSimState;

/// Fresh amd64 state — the common base for every procedure test family.
pub(crate) fn amd64_state() -> RustSimState {
    RustSimState::new("amd64").unwrap()
}

/// amd64 state with each `(addr, size)` region mapped RWX. Mirrors the
/// region-mapping dance that the `setup_state` helpers used to inline.
pub(crate) fn amd64_state_with_regions(regions: &[(u64, u64)]) -> RustSimState {
    let mut state = amd64_state();
    for &(addr, size) in regions {
        state.map_memory(addr, size, Permission::RWX);
    }
    state
}
