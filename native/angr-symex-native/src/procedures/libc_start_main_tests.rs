use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_libc_start_main_no_return() {
    let proc = NativeLibcStartMain;
    assert_eq!(proc.name(), "__libc_start_main");
    assert_eq!(proc.num_args(), 5);
    assert!(proc.no_return());
}

#[test]
fn test_libc_start_main_exits_with_status_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let proc = NativeLibcStartMain;
    let args = vec![
        RustBV::concrete(0x400000, 64),   // main
        RustBV::concrete(1, 64),          // argc
        RustBV::concrete(0x7fff0000, 64), // argv
        RustBV::concrete(0, 64),          // init
        RustBV::concrete(0, 64),          // fini
    ];
    let status = proc.call(&mut state, &args).unwrap().expect("exit status");
    assert_eq!(status.as_u64(), Some(0));
    assert_eq!(status.width(), 64);
}
