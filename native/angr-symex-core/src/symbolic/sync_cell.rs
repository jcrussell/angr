//! `Cell`-shaped atomics for the per-context caches of
//! [`SymContext`](super::SymContext).
//!
//! A context is `Sync` so that a state can share it through an `Arc`
//! (angr-az6ru.2). These carry the small `Copy` values `SymContext` kept in
//! `Cell`s, behind the same `get` / `set` surface. One logical owner drives a
//! context at a time, so every access is `Relaxed`: the types make the
//! sharing sound, they do not order concurrent writers.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// A `Cell<bool>`.
#[derive(Debug, Default)]
pub(crate) struct SyncFlag(AtomicBool);

impl SyncFlag {
    pub(crate) const fn new(value: bool) -> Self {
        Self(AtomicBool::new(value))
    }

    pub(crate) fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub(crate) fn set(&self, value: bool) {
        self.0.store(value, Ordering::Relaxed);
    }
}

/// A `Cell<Option<bool>>`.
#[derive(Debug, Default)]
pub(crate) struct SyncOptBool(AtomicU8);

impl SyncOptBool {
    const NONE: u8 = 0;
    const FALSE: u8 = 1;
    const TRUE: u8 = 2;

    pub(crate) const fn new(value: Option<bool>) -> Self {
        Self(AtomicU8::new(Self::encode(value)))
    }

    const fn encode(value: Option<bool>) -> u8 {
        match value {
            None => Self::NONE,
            Some(false) => Self::FALSE,
            Some(true) => Self::TRUE,
        }
    }

    pub(crate) fn get(&self) -> Option<bool> {
        match self.0.load(Ordering::Relaxed) {
            Self::FALSE => Some(false),
            Self::TRUE => Some(true),
            _ => None,
        }
    }

    pub(crate) fn set(&self, value: Option<bool>) {
        self.0.store(Self::encode(value), Ordering::Relaxed);
    }
}

/// A `Cell<Option<usize>>` for a depth or a length: `usize::MAX`, which no
/// collection reaches, stands for `None`.
#[derive(Debug)]
pub(crate) struct SyncOptUsize(AtomicUsize);

impl SyncOptUsize {
    const NONE: usize = usize::MAX;

    pub(crate) const fn new(value: Option<usize>) -> Self {
        Self(AtomicUsize::new(Self::encode(value)))
    }

    const fn encode(value: Option<usize>) -> usize {
        match value {
            None => Self::NONE,
            Some(value) => value,
        }
    }

    const fn decode(raw: usize) -> Option<usize> {
        if raw == Self::NONE { None } else { Some(raw) }
    }

    pub(crate) fn get(&self) -> Option<usize> {
        Self::decode(self.0.load(Ordering::Relaxed))
    }

    pub(crate) fn set(&self, value: Option<usize>) {
        self.0.store(Self::encode(value), Ordering::Relaxed);
    }

    /// The value, leaving `None`.
    pub(crate) fn take(&self) -> Option<usize> {
        Self::decode(self.0.swap(Self::NONE, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_round_trip_every_value() {
        let flag = SyncFlag::new(false);
        flag.set(true);
        assert!(flag.get());

        let opt = SyncOptBool::new(None);
        for value in [Some(true), Some(false), None] {
            opt.set(value);
            assert_eq!(opt.get(), value);
        }

        let depth = SyncOptUsize::new(Some(3));
        assert_eq!(depth.take(), Some(3));
        assert_eq!(depth.get(), None);
        depth.set(Some(0));
        assert_eq!(depth.get(), Some(0));
    }
}
