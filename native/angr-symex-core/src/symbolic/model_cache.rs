//! The per-thread cache of Z3 models (angr-az6ru.2).
//!
//! A [`SymContext`](super::SymContext) owns no `z3::Model`. Its two witness
//! slots hold a `ModelRef`, a token that is plain data, and the model the
//! token names lives in this thread's table. A fork clones the token, so
//! parent and child read the same model, as they shared one `Rc` before.
//!
//! A token is a cache key and nothing more: `ModelRef::get` returns `None`
//! when the table has no model for it, which is the case after an eviction
//! and for a token minted on another thread. The caller then has no witness
//! and asks the solver, so no answer depends on the table's contents.
//!
//! A model leaves the table when the last clone of its token is dropped on
//! this thread. The table also holds at most [`capacity`] models, least
//! recently read first out, which bounds the models whose tokens were dropped
//! on another thread and so never cleared their entry here.
//!
//! **Panic policy:** `#![deny(clippy::unwrap_used, clippy::expect_used)]`,
//! as for the rest of the `SymContext` files.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::stats::MODEL_CACHE_EVICT_COUNT;

/// Models kept per thread when `ANGR_RUST_MODEL_CACHE_SIZE` is unset.
pub const DEFAULT_CAPACITY: usize = 4096;

/// How many models one thread keeps. Read once from
/// `ANGR_RUST_MODEL_CACHE_SIZE`; at least one, so the model a query just
/// stored can be read back.
pub fn capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("ANGR_RUST_MODEL_CACHE_SIZE")
            .ok() // SILENT(cat-a): an unset variable is the default
            .and_then(|s| s.trim().parse::<usize>().ok()) // SILENT(cat-a): an unparsable value is the default
            .unwrap_or(DEFAULT_CAPACITY)
            .max(1)
    })
}

/// One cached model and when it was last read.
struct Entry {
    model: Rc<z3::Model>,
    used: u64,
}

/// A thread's models by token id.
pub(super) struct Table {
    models: HashMap<u64, Entry>,
    /// Counts reads and stores; an entry's `used` is its value at the last
    /// one, so the smallest `used` is the least recently read.
    clock: u64,
}

impl Table {
    pub(super) fn new() -> Self {
        Self {
            models: HashMap::new(),
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    pub(super) fn len(&self) -> usize {
        self.models.len()
    }

    /// The model stored under `id`, marked as just read.
    pub(super) fn get(&mut self, id: u64) -> Option<Rc<z3::Model>> {
        let now = self.tick();
        let entry = self.models.get_mut(&id)?;
        entry.used = now;
        Some(Rc::clone(&entry.model))
    }

    /// Store `model` under `id` and return the models evicted to stay within
    /// `capacity`, for the caller to drop outside any borrow of the table.
    ///
    /// A full table sheds an eighth of its capacity at once, least recently
    /// read first, so the scan that picks them runs once per that many
    /// stores.
    pub(super) fn insert(
        &mut self,
        id: u64,
        model: Rc<z3::Model>,
        capacity: usize,
    ) -> Vec<Rc<z3::Model>> {
        let used = self.tick();
        self.models.insert(id, Entry { model, used });
        if self.models.len() <= capacity {
            return Vec::new();
        }
        let keep = capacity.saturating_sub(capacity / 8).max(1);
        let excess = self.models.len().saturating_sub(keep);
        let mut by_age: Vec<(u64, u64)> = self.models.iter().map(|(id, e)| (e.used, *id)).collect();
        by_age.sort_unstable();
        by_age
            .into_iter()
            .take(excess)
            .filter_map(|(_, id)| self.remove(id))
            .collect()
    }

    pub(super) fn remove(&mut self, id: u64) -> Option<Rc<z3::Model>> {
        self.models.remove(&id).map(|entry| entry.model)
    }
}

thread_local! {
    static MODELS: RefCell<Table> = RefCell::new(Table::new());
}

/// The identity one stored model has in every thread's table. Dropping it
/// removes the model from the table of the thread it is dropped on.
struct Token(u64);

impl Drop for Token {
    fn drop(&mut self) {
        // `try_with`: a context dropped while the thread is tearing down has
        // no table left to clear. The model is dropped outside the borrow.
        let mine = MODELS.try_with(|models| {
            let mut models = models.try_borrow_mut().ok()?; // SILENT(cat-a): the table is never borrowed across a token drop; if it were, the entry would wait for eviction
            models.remove(self.0)
        });
        drop(mine);
    }
}

/// A context's handle on a cached model: `Send`, cheap to clone, and valid
/// on any thread, where it resolves to the model or to nothing.
#[derive(Clone)]
pub(super) struct ModelRef(Arc<Token>);

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelRef>();
};

impl ModelRef {
    /// Put `model` in this thread's table and return the token for it.
    pub(super) fn store(model: z3::Model) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let evicted =
            MODELS.with(|models| models.borrow_mut().insert(id, Rc::new(model), capacity()));
        MODEL_CACHE_EVICT_COUNT.fetch_add(evicted.len() as u64, Ordering::Relaxed);
        // Dropped here, outside the borrow.
        drop(evicted);
        Self(Arc::new(Token(id)))
    }

    /// The model, if this thread's table still holds it.
    pub(super) fn get(&self) -> Option<Rc<z3::Model>> {
        MODELS.with(|models| models.borrow_mut().get(self.0.0))
    }
}

/// How many models this thread holds.
pub fn live_models() -> usize {
    MODELS.with(|models| models.borrow().len())
}

/// Test-only: empty this thread's table, as evicting every model would. Every
/// token then reads as one from another thread does.
#[cfg(any(test, feature = "testing"))]
pub fn evict_all_for_test() {
    let all = MODELS.with(|models| std::mem::replace(&mut *models.borrow_mut(), Table::new()));
    drop(all);
}

test_submod!("model_cache_tests.rs" => model_cache_tests);
