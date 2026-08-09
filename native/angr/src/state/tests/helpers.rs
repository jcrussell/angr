//! Fixtures shared by more than one of the sibling test modules.
//!
//! `sym_file_bytes` builds a symbolic file payload for the two `filesystem_*`
//! modules; `build_populated_state` builds the fully-populated state that the
//! `snapshot` round-trip and the `export` full-dump assertions both start from.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

/// `n` fresh 8-bit symbolic bytes named `{prefix}_{i}` (ids sit in
/// 0x5f000+ to stay clear of other symbolic ids in these tests; the
/// prefixes distinguish tests from each other).
pub(super) fn sym_file_bytes(n: usize, prefix: &str) -> Vec<RustBV> {
    (0..n)
        .map(|i| RustBV::symbolic_with_id(0x5f000 + i as u64, format!("{prefix}_{i}"), 8))
        .collect()
}

// =========================================================================
// Snapshot / Serialization tests (angr-x04s.1.3)
// =========================================================================

/// Build a representative RustSimState that touches each bucket A/B/C
/// field (concrete + symbolic registers, mapped memory pages, solver
/// constraints, history, call stack, file system, hooks, environment,
/// flags). The Z3 ASTs are minted inside whichever Z3 context the test
/// is currently running under.
#[cfg(feature = "vex-engine-z3")]
pub(super) fn build_populated_state() -> RustSimState {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_pc(0x4012a0);

    // Bucket B (registers): one concrete, one symbolic.
    s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
    let rbx_sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    s.set_register("rbx", rbx_sym.clone());

    // Bucket B (memory): map a page with concrete bytes.
    s.memory_mut()
        .map_data(0x10_0000u64, &[1u8, 2, 3, 4, 5], Permission::RWX);

    // Bucket A: history + call stack.
    s.add_to_history(0x4011a0);
    s.add_to_history(0x4012a0);
    s.push_call(0x4012a0, 0x401400, 0x4012a5, 0x7fff_ffff_0000);
    // Heap metadata via the public heap_alloc/heap_free helpers.
    let a1 = s.heap_alloc(32);
    let _a2 = s.heap_alloc(64);
    let _ = s.heap_free(a1);

    // Bucket C: hook + environment + stdin symbols.
    s.add_hook(0x401200);
    s.setenv(b"PATH".to_vec(), b"/usr/bin".to_vec());
    s.setenv(b"HOME".to_vec(), b"/root".to_vec());
    s.record_stdin_symbol("stdin_chunk_0".to_string(), 16);

    // Flags.
    s.set_no_ip_concretization(true);
    s.set_keep_ip_symbolic(false);
    s.set_no_symbolic_jump_resolution(true);
    s.set_posix_brk(0x1B0_4000);
    s.set_mmap_base(0xC100_8000);
    s.set_tsc_counter(0x0010_000F_A000);
    s.set_getopt_cursor(7, 3);
    s.set_getopt_extern(GetoptExternAddrs {
        optind: Some(0x602000),
        optarg: Some(0x602008),
        optopt: Some(0x602010),
    });
    s.push_native_resume_frame(NativeResumeFrame {
        proc_name: "pthread_once".to_string(),
        resume_tag: 1,
        saved_args: vec![RustBV::concrete(0x602000, 64)],
        caller_return_addr: 0x4007a0,
    });

    // Solver constraints — `rbx > 10` must hold after restore.
    let cmp = {
        let ctx = s.solver().borrow();
        let ten = RustBV::concrete(10, 64);
        rbx_sym.ugt(&ten, &ctx)
    };
    s.solver().borrow().assume_true(&cmp);

    s
}
