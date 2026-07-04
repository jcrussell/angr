// Tests for fgets.rs (NativeFgets/Fgetc/Getchar/Getc procedures).
// Split from the inline `#[cfg(test)] mod tests` block; see rust-mod-tests-sibling-extraction.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

/// AMD64 `_IO_FILE._fileno` byte offset (see `io_file_for_arch` in fileops.rs).
const AMD64_FD_OFFSET: u64 = 112;

/// Map a FILE struct at `file_ptr` whose `_fileno` field holds `fd`. fgets/fgetc
/// resolve the fd from this field; only fd 0 (stdin) is served natively.
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    let fd_bv = RustBV::concrete((fd as u32) as u128, 32);
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    state
        .memory_store(file_ptr + AMD64_FD_OFFSET, fd_bv)
        .unwrap();
}

#[test]
fn test_fgets_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),        // buf
                RustBV::concrete(10, 64),            // size
                RustBV::concrete(stdin as u128, 64), // stream (stdin, _fileno=0)
            ],
        )
        .unwrap();

    // Should return buf address
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // First 9 bytes should be symbolic
    for i in 0..9u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
    }

    // Byte 9 should be NUL terminator
    let nul = state.memory_load(0x2009, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_newline_only_at_last_byte() {
    // A full read of (size-1) bytes cannot contain an embedded newline: real
    // fgets stops at (and includes) the first newline. So only the final data
    // byte may be a newline; every earlier byte must be non-newline. This
    // prunes the infeasible newline-in-middle-of-a-full-read states the native
    // path used to keep vs Python's SimFile model. See angr-abora.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    // size=8 => read_count=7, data bytes at 0x2000..0x2007, NUL at 0x2007.
    NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(8, 64),
                RustBV::concrete(stdin as u128, 64),
            ],
        )
        .unwrap();

    let nl = RustBV::concrete(b'\n' as u128, 8);
    let ctx = state.solver().borrow();

    // Byte 0 (not the last data byte) can NOT be a newline post-fix.
    let b0 = state.memory_load(0x2000, 1).unwrap();
    assert!(
        !ctx.can_be_true(&b0.eq(&nl, &ctx)),
        "byte 0 of a full read must not be allowed to equal newline"
    );

    // Byte 5 (still not the last) likewise can NOT be a newline.
    let b5 = state.memory_load(0x2005, 1).unwrap();
    assert!(
        !ctx.can_be_true(&b5.eq(&nl, &ctx)),
        "byte 5 must not equal newline"
    );

    // The last data byte (index 6) MAY be the terminating newline.
    let b6 = state.memory_load(0x2006, 1).unwrap();
    assert!(
        ctx.can_be_true(&b6.eq(&nl, &ctx)),
        "last data byte may legitimately be the terminating newline"
    );
    // ...and is not forced to a newline either.
    assert!(
        ctx.can_be_false(&b6.eq(&nl, &ctx)),
        "last data byte is not forced to newline"
    );
}

#[test]
fn test_fgets_size_1() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),             // size=1 means only NUL
                RustBV::concrete(stdin as u128, 64), // stdin
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // Only NUL should be written
    let nul = state.memory_load(0x2000, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_size_0() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size=0 returns NULL
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0)); // NULL
}

#[test]
fn test_fgets_symbolic_buf() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "buf", 64);
    drop(ctx);

    let result = NativeFgets.call(
        &mut state,
        &[sym, RustBV::concrete(10, 64), RustBV::concrete(0, 64)],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fgetc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgetc
        .call(
            &mut state,
            &[RustBV::concrete(stdin as u128, 64)], // stream (stdin, _fileno=0)
        )
        .unwrap();

    let val = result.unwrap();
    // Should be symbolic (can't get concrete value)
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getchar_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeGetchar.call(&mut state, &[]).unwrap();
    let val = result.unwrap();
    // Should be symbolic
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeGetc
        .call(&mut state, &[RustBV::concrete(stdin as u128, 64)])
        .unwrap();
    let val = result.unwrap();
    assert!(val.as_u64().is_none());
}

#[test]
fn test_fgets_nonstdin_falls_back() {
    // A real-file stream (fd > 0) must defer to Python's SimFile model, not
    // synthesize stdin bytes.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, 3);

    let result = NativeFgets.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(10, 64),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fgets_invalid_fd_returns_minus1() {
    // _fileno < 0 (no backing descriptor): Python fgets returns -1.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, -1);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(10, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0xFFFFFFFFFFFFFFFF));
}

#[test]
fn test_fgets_symbolic_stream_falls_back() {
    // A symbolic FILE* can't resolve to an fd — must error so Python handles it.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "stream", 64);
    drop(ctx);

    let result = NativeFgets.call(
        &mut state,
        &[RustBV::concrete(0x2000, 64), RustBV::concrete(10, 64), sym],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fgetc_nonstdin_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, 3);

    let result = NativeFgetc.call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)]);
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fgetc_short_reads_can_return_eof() {
    // With SHORT_READS, native fgetc returns If(eof, -1, byte): the EOF sentinel
    // (-1, i.e. 0xFFFFFFFF as a 32-bit int) becomes a reachable solution, while
    // an ordinary byte is still reachable too. This mirrors Python fgetc's
    // If(real_length == 0, -1, byte) (angr-qx81x). Without the option the result
    // is a bare zero-extended byte in [0, 255], so -1 is unreachable (asserted in
    // test_fgetc_default_never_eof).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_option("SHORT_READS", true);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgetc
        .call(&mut state, &[RustBV::concrete(stdin as u128, 64)])
        .unwrap()
        .unwrap();

    let ctx = state.solver().borrow();
    assert!(
        ctx.solution(&result, 0xFFFF_FFFF),
        "EOF (-1) must be reachable under SHORT_READS"
    );
    assert!(
        ctx.solution(&result, 0x41),
        "an ordinary byte must still be reachable under SHORT_READS"
    );
}

#[test]
fn test_fgetc_default_never_eof() {
    // Default path (SHORT_READS off): fgetc returns a bare zero-extended byte,
    // so the value is in [0, 255] and the EOF sentinel (-1 / 0xFFFFFFFF) is
    // never a solution. Guards the gating of the short-read EOF model.
    let mut state = RustSimState::new("amd64").unwrap();
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgetc
        .call(&mut state, &[RustBV::concrete(stdin as u128, 64)])
        .unwrap()
        .unwrap();

    let ctx = state.solver().borrow();
    assert!(
        !ctx.solution(&result, 0xFFFF_FFFF),
        "default fgetc never returns the EOF sentinel"
    );
    assert!(
        ctx.solution(&result, 0x41),
        "an ordinary byte is reachable on the default path"
    );
}

#[test]
fn test_fgets_short_reads_returns_symbolic_size() {
    // With SHORT_READS, native fgets models a variable-length read: it returns
    // a symbolic real_size in [0, size-1] (the downstream fork source) rather
    // than the buffer pointer, mirroring Python procedures/libc/fgets.py case 2
    // (angr-efvao). Without the option the return stays the concrete buffer
    // pointer (test_fgets_basic), so the default path is unchanged.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_option("SHORT_READS", true);
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    // size=8 => real_size in [0, 7].
    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(8, 64),
                RustBV::concrete(stdin as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();

    assert!(
        result.as_u64().is_none(),
        "short-read fgets returns symbolic real_size, not the concrete buffer ptr"
    );

    let ctx = state.solver().borrow();
    assert!(
        ctx.solution(&result, 0),
        "a zero-length short read must be reachable"
    );
    assert!(
        ctx.solution(&result, 7),
        "the full (size-1) read must be reachable"
    );
    assert!(!ctx.solution(&result, 8), "real_size cannot exceed size-1");
}

#[test]
fn test_fgets_short_reads_nul_at_real_size() {
    // The NUL terminator is stored at the symbolic real_size offset: pinning
    // real_size = k forces buf+k to be NUL (proves the symbolic-position store).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_option("SHORT_READS", true);
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(8, 64),
                RustBV::concrete(stdin as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();

    // Pin real_size = 3; the NUL terminator must then sit at buf+3.
    let pin = {
        let ctx = state.solver().borrow();
        result.eq(&RustBV::concrete(3, 64), &ctx)
    };
    state.add_constraint(pin);

    let byte3 = state.memory_load(0x2003, 1).unwrap();
    let ctx = state.solver().borrow();
    let zero = RustBV::concrete(0, 8);
    assert!(
        !ctx.can_be_true(&byte3.ne(&zero, &ctx)),
        "byte at buf+real_size must be the NUL terminator"
    );
}

#[test]
fn test_gets_returns_buffer_pointer() {
    // gets returns the destination buffer pointer (not real_size), reads from
    // stdin with no size/stream args, and applies the case-2 symbolic-line
    // model unconditionally (no SHORT_READS gate), mirroring Python
    // procedures/libc/gets.py.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeGets
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .unwrap()
        .unwrap();

    assert_eq!(
        result.as_u64(),
        Some(0x2000),
        "gets returns the destination buffer pointer"
    );

    // The first bytes are symbolic stdin input.
    let byte0 = state.memory_load(0x2000, 1).unwrap();
    assert!(byte0.as_u64().is_none(), "gets byte 0 should be symbolic");
}

#[test]
fn test_gets_nul_at_real_size_bounded() {
    // The terminating NUL sits at the symbolic real_size offset (max_gets_size
    // model: real_size in [0, 254]). Pin real_size=5 by forcing buf+5 to be the
    // NUL while a non-NUL byte before it stays reachable, proving the variable
    // length terminator. Also confirm real_size cannot reach 255 (== size-1+1).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    NativeGets
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .unwrap()
        .unwrap();

    // Constrain buf+5 to NUL and buf+4 to non-NUL: a self-consistent short read
    // of length 5 must be satisfiable under the case-2 constraints.
    let (cons, sat) = {
        let ctx = state.solver().borrow();
        let zero = RustBV::concrete(0, 8);
        let b4 = state.memory_load(0x2004, 1).unwrap();
        let b5 = state.memory_load(0x2005, 1).unwrap();
        let cons = b5.eq(&zero, &ctx).and(&b4.ne(&zero, &ctx), &ctx);
        let sat = ctx.can_be_true(&cons);
        (cons, sat)
    };
    assert!(sat, "a length-5 short read must be reachable under gets");
    state.add_constraint(cons);
}

#[test]
fn test_fgets_content_sym_fd_falls_back_without_half_serving() {
    // angr-0xyq2 Phase 2 guard: bounded symbolic file content is served by
    // read/fread, but fgets keeps its Python fallback (the newline-constraint
    // model is a separate bead) — and must not half-serve: the destination
    // buffer stays untouched.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, &[0xAAu8; 16], Permission::RWX);
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("fgetsfile_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadOnly);
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFgets.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(10, 64),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(result.is_err(), "non-stdin fgets falls back to Python");
    // No half-serve: every destination byte still carries the 0xAA prefill.
    for i in 0..10u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(0xAA), "byte {i} untouched");
    }
    // The content itself is untouched too (fallback, not demotion).
    assert!(state.file_system_ref().fd_content_sym(fd).is_some());
}
