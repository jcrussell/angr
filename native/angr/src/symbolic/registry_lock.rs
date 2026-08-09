//! Lock-ordering guard for [`super::SymbolicIdentityRegistry`]'s four maps.
//!
//! The registry keeps four independent `RwLock`s and its mutators
//! (`register`, `register_by_id`, `clear`, `remove`, `retain`) each touch
//! several of them. Two rules kept them deadlock-free and cheap:
//!
//! 1. Guards are acquired in the canonical [`RegistryMap`] order.
//! 2. At most two are co-held — the `IdToPy` primary, which serializes a whole
//!    removal, plus one secondary at a time.
//!
//! Both were comment-only conventions, and `retain` had already broken rule 2
//! once by holding all four at once (angr-sqfj8.106, fixed in angr-zi35f.12).
//! [`write_ordered`] turns them into an assertion that fires on the offending
//! acquire, in the same spirit as `#[angr_macros::steady_guarded]`: it does not
//! make the bad sequence unrepresentable, it makes it impossible to reach
//! without noticing (angr-91vj9.12).
//!
//! Scope: write guards only. Read guards are unbounded and untracked — no
//! registry method upgrades a read to a write, so they cannot participate in
//! the ordering cycle these rules exist to prevent.

use parking_lot::{RwLock, RwLockWriteGuard};
use std::cell::Cell;
use std::ops::{Deref, DerefMut};

/// The registry's four write-locked maps, in canonical acquisition order.
///
/// The order is `remove`/`retain`'s: the `IdToPy` primary first, then the three
/// secondaries. `register` and `clear` touch the same maps in a different
/// textual order, which is only safe because they never co-hold two — a fact
/// [`write_ordered`] now checks rather than assumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RegistryMap {
    /// `rust_id_to_py` — the primary map; its guard serializes a whole removal.
    IdToPy = 0,
    /// `rust_id_to_name`.
    IdToName = 1,
    /// `py_hash_to_rust_id`.
    HashToId = 2,
    /// `name_to_info`.
    NameToInfo = 3,
}

thread_local! {
    /// Bitmask of [`RegistryMap`] write guards live on this thread.
    static HELD: Cell<u8> = const { Cell::new(0) };
}

/// Maximum number of registry write guards one thread may co-hold.
const MAX_CO_HELD: u32 = 2;

/// Write-lock one registry map, checking the module's two ordering rules.
///
/// # Panics
///
/// If the calling thread already holds a guard for this map or for any map
/// later in [`RegistryMap`] order, or if it already holds [`MAX_CO_HELD`]
/// guards. Both are programming errors in registry code, not conditions a
/// caller can recover from — a violation is a latent deadlock or a lock-hold
/// window wide enough to stall every importer.
pub(super) fn write_ordered<T>(lock: &RwLock<T>, map: RegistryMap) -> OrderedWrite<'_, T> {
    let bit = 1u8 << (map as u8);
    HELD.with(|held| {
        let cur = held.get();
        assert!(
            cur & !(bit - 1) == 0,
            "registry lock order violated: acquiring {map:?} while holding mask {cur:#06b}"
        );
        assert!(
            cur.count_ones() < MAX_CO_HELD,
            "registry co-held write guards exceeded {MAX_CO_HELD}: acquiring {map:?} while holding mask {cur:#06b}"
        );
        held.set(cur | bit);
    });
    OrderedWrite {
        guard: lock.write(),
        bit,
    }
}

/// A registry write guard that releases its slot in [`HELD`] on drop.
pub(super) struct OrderedWrite<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
    bit: u8,
}

impl<T> Deref for OrderedWrite<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for OrderedWrite<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for OrderedWrite<'_, T> {
    fn drop(&mut self) {
        HELD.with(|held| held.set(held.get() & !self.bit));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two guards in canonical order is the shape `remove`/`retain` use.
    #[test]
    fn test_primary_then_secondary_is_allowed() {
        let a = RwLock::new(1u32);
        let b = RwLock::new(2u32);
        let _pri = write_ordered(&a, RegistryMap::IdToPy);
        let sec = write_ordered(&b, RegistryMap::NameToInfo);
        assert_eq!(*sec, 2);
    }

    /// Sequential acquisition in any order is fine — that is `register`/`clear`.
    #[test]
    fn test_sequential_acquires_any_order() {
        let a = RwLock::new(1u32);
        for map in [
            RegistryMap::NameToInfo,
            RegistryMap::IdToPy,
            RegistryMap::HashToId,
            RegistryMap::IdToName,
        ] {
            let g = write_ordered(&a, map);
            drop(g);
        }
        assert_eq!(HELD.with(Cell::get), 0);
    }

    #[test]
    #[should_panic(expected = "registry lock order violated")]
    fn test_out_of_order_acquire_panics() {
        let a = RwLock::new(1u32);
        let b = RwLock::new(2u32);
        let _sec = write_ordered(&a, RegistryMap::NameToInfo);
        let _pri = write_ordered(&b, RegistryMap::IdToPy);
    }

    #[test]
    #[should_panic(expected = "registry lock order violated")]
    fn test_reacquiring_same_map_panics() {
        let a = RwLock::new(1u32);
        let b = RwLock::new(2u32);
        let _first = write_ordered(&a, RegistryMap::IdToName);
        let _second = write_ordered(&b, RegistryMap::IdToName);
    }

    /// The angr-sqfj8.106 shape: `retain` holding all four maps at once.
    #[test]
    #[should_panic(expected = "co-held write guards exceeded")]
    fn test_three_co_held_guards_panics() {
        let a = RwLock::new(1u32);
        let b = RwLock::new(2u32);
        let c = RwLock::new(3u32);
        let _g0 = write_ordered(&a, RegistryMap::IdToPy);
        let _g1 = write_ordered(&b, RegistryMap::IdToName);
        let _g2 = write_ordered(&c, RegistryMap::HashToId);
    }

    /// A failed acquire must not leave the thread's mask dirty for the next
    /// caller — the assert fires before the bit is set, and unwinding drops the
    /// guards already held.
    #[test]
    fn test_mask_is_clean_after_a_rejected_acquire() {
        let a = RwLock::new(1u32);
        let b = RwLock::new(2u32);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _sec = write_ordered(&a, RegistryMap::NameToInfo);
            let _pri = write_ordered(&b, RegistryMap::IdToPy);
        }));
        assert!(caught.is_err());
        assert_eq!(HELD.with(Cell::get), 0);
        // The rejected lock is also still usable: nothing was left write-held.
        assert_eq!(*write_ordered(&b, RegistryMap::IdToPy), 2);
    }
}
