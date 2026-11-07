//! A mechanical field's generated `merge_field_<name>` that `merge` never
//! calls must fail the build: the derive's policy/merge-line tie rests on
//! rustc reporting `dead_code` on derive output (angr-vmi2l.229).
#![deny(dead_code)]

#[derive(angr_symex_macros::MergePolicy)]
struct State {
    #[merge_policy = "max"]
    f: u64,
}

fn main() {
    let s = State { f: 0 };
    let _ = s.f;
}
