use super::*;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// The status, symbolic or not, becomes the state's exit code beside an
/// `Ijk_Exit` history entry, as Python's `SimProcedure.exit` records it
/// (angr-mx00p.132.7).
#[test]
fn exit_records_the_status_and_an_exit_entry() {
    let h = NativeExitSyscall;
    assert_eq!(h.num_args(), 1, "the status is the one argument");
    for status in [RustBV::concrete(0, 32), RustBV::symbol("exit_status", 32)] {
        let mut state = RustSimState::new("x86").expect("x86 state");
        state.set_pc(0x8000_1000);
        let outcome = h
            .call(&mut state, std::slice::from_ref(&status))
            .expect("exit handler never errs");
        assert!(matches!(outcome, SyscallOutcome::Exit));
        assert_eq!(state.exit_code(), Some(&status));
        let last = state.detailed_history().back().expect("an exit entry");
        assert_eq!(last.jumpkind, crate::state::HistoryEntry::JK_EXIT);
        assert_eq!(last.addr, 0x8000_1000);
    }
}
