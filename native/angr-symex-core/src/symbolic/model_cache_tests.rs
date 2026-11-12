//! Tests for the per-thread model table (`model_cache.rs`).

use super::*;

fn a_model() -> Rc<z3::Model> {
    let solver = z3::Solver::new();
    solver.check();
    Rc::new(solver.get_model().expect("an empty solver has a model"))
}

/// A full table sheds its least recently read models, and a read counts as
/// a use.
#[test]
fn table_evicts_least_recently_read() {
    let mut table = Table::new();
    for id in 1..=8 {
        assert!(table.insert(id, a_model(), 8).is_empty());
    }
    assert!(table.get(1).is_some());
    // The ninth goes over capacity 8: the table falls to 7, shedding the two
    // oldest unread ones.
    assert_eq!(table.insert(9, a_model(), 8).len(), 2);
    assert_eq!(table.len(), 7);
    assert!(table.get(1).is_some(), "a model read since stays");
    assert!(table.get(2).is_none() && table.get(3).is_none());
    assert!(table.get(4).is_some() && table.get(9).is_some());
}

/// Capacity one keeps the model just stored.
#[test]
fn table_of_one_keeps_the_newest() {
    let mut table = Table::new();
    assert!(table.insert(1, a_model(), 1).is_empty());
    assert_eq!(table.insert(2, a_model(), 1).len(), 1);
    assert!(table.get(1).is_none());
    assert!(table.get(2).is_some());
}

/// A model lives as long as some clone of its token, and no longer.
#[test]
fn model_leaves_the_table_with_its_last_token() {
    let before = live_models();
    let solver = z3::Solver::new();
    solver.check();
    let token = ModelRef::store(solver.get_model().expect("an empty solver has a model"));
    let copy = token.clone();
    assert_eq!(live_models(), before + 1);
    drop(token);
    assert!(copy.get().is_some());
    drop(copy);
    assert_eq!(live_models(), before);
}

/// A token dropped on another thread leaves this thread's model alone, and a
/// token from this thread names nothing there.
#[test]
fn token_resolves_only_on_the_thread_that_stored_it() {
    let solver = z3::Solver::new();
    solver.check();
    let token = ModelRef::store(solver.get_model().expect("an empty solver has a model"));
    let moved = token.clone();
    let seen_elsewhere = std::thread::spawn(move || moved.get().is_some())
        .join()
        .expect("the reader thread ran");
    assert!(!seen_elsewhere);
    assert!(token.get().is_some());
}
