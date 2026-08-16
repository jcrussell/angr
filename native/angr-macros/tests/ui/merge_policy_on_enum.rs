//! The derive walks named struct fields; on an enum there is nothing to label,
//! so it must say so rather than expand to an empty (vacuously passing) impl.
#[derive(angr_macros::MergePolicy)]
enum State {
    Fresh,
    Merged,
}

fn main() {}
