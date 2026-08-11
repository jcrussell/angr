//! Concrete-content filesystem behaviour on `RustSimState`: defaults, open/close,
//! position-aware read/write, seek, path normalization, fd tables
//! (`dup`/`dup2`/`pipe`), fork isolation, and snapshot backward compatibility.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

#[test]
fn test_filesystem_default() {
    let fs = FileSystem::default();
    assert!(fs.is_open(0)); // stdin
    assert!(fs.is_open(1)); // stdout
    assert!(fs.is_open(2)); // stderr
    assert!(!fs.is_open(3));
    assert_eq!(fs.next_fd(), 3);
}

#[test]
fn test_filesystem_open_close() {
    let mut fs = FileSystem::default();
    let fd = fs.open("test.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);
    assert!(fs.is_open(3));

    let closed = fs.close(3);
    assert!(closed);
    assert!(!fs.is_open(3));

    // Double close returns false
    assert!(!fs.close(3));
}

#[test]
fn test_filesystem_write_read() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content(
        "data.bin".to_string(),
        FdFlags::ReadOnly,
        b"hello world".to_vec(),
    );

    let data = fs.read(fd, 5);
    assert_eq!(data, b"hello");

    let data2 = fs.read(fd, 6);
    assert_eq!(data2, b" world");

    // Read past end
    let data3 = fs.read(fd, 10);
    assert!(data3.is_empty());
}

#[test]
fn test_filesystem_read_closed_fd_serves_nothing() {
    // Parity with read_sym/read_sym_at: a closed fd serves no bytes even
    // though its content buffer survives (angr-myzjx.23).
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content(
        "data.bin".to_string(),
        FdFlags::ReadOnly,
        b"hello world".to_vec(),
    );
    assert_eq!(fs.read_at(fd, 0, 5), b"hello");
    assert!(fs.close(fd));

    // Both the position-advancing and positioned reads must refuse a closed fd.
    assert!(fs.read(fd, 5).is_empty());
    assert!(fs.read_at(fd, 0, 5).is_empty());
}

#[test]
fn test_filesystem_seek() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);

    // SEEK_SET
    assert_eq!(fs.seek(fd, 50, 0), Some(50));
    // SEEK_CUR
    assert_eq!(fs.seek(fd, 10, 1), Some(60));
    // SEEK_END
    assert_eq!(fs.seek(fd, -5, 2), Some(95));
    // Invalid whence
    assert_eq!(fs.seek(fd, 0, 99), None);
}

#[test]
fn test_filesystem_write_position_aware() {
    let mut fs = FileSystem::default();
    let fd = fs.open("out.bin".to_string(), FdFlags::WriteOnly);

    // Sequential writes (no seek): position-aware path is byte-identical to a
    // plain append, and advances the position to EOF each time. `true` =
    // accepted (no bounded symbolic content on the fd).
    assert!(fs.write(fd, b"abc"));
    assert!(fs.write(fd, b"def"));
    assert_eq!(fs.fd_content(fd), b"abcdef");

    // Seek back and overwrite in place (the case append-only would corrupt).
    assert_eq!(fs.seek(fd, 0, 0), Some(0));
    assert!(fs.write(fd, b"XY"));
    assert_eq!(fs.fd_content(fd), b"XYcdef");

    // A subsequent write continues from the advanced position (after "XY").
    assert!(fs.write(fd, b"Z"));
    assert_eq!(fs.fd_content(fd), b"XYZdef");

    // Sparse seek past EOF zero-fills the gap, like write_at/pwrite.
    assert_eq!(fs.seek(fd, 8, 0), Some(8));
    assert!(fs.write(fd, b"!"));
    assert_eq!(fs.fd_content(fd), b"XYZdef\x00\x00!");
}

#[test]
fn test_filesystem_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("test.txt".to_string(), FdFlags::ReadOnly);
    assert!(state.file_system_ref().is_open(3));

    let mut forked = state.fork();
    // Close in forked state
    forked.file_system().close(3);
    assert!(!forked.file_system_ref().is_open(3));
    // Original should still be open
    assert!(state.file_system_ref().is_open(3));
}

#[test]
fn test_filesystem_normalize_path() {
    let mut fs = FileSystem::default();
    assert_eq!(fs.normalize_path("/a/b/../c/./d"), "/a/c/d");
    assert_eq!(fs.normalize_path("/../.."), "/");
    // Truncates at the first NUL, then resolves relative to cwd (/).
    assert_eq!(fs.normalize_path("x\0junk"), "/x");

    fs.set_cwd(b"/home/user".to_vec());
    assert_eq!(fs.normalize_path("f.txt"), "/home/user/f.txt");
    assert_eq!(fs.normalize_path("../f"), "/home/f");
    assert_eq!(fs.normalize_path(""), "/home/user");
}

#[test]
fn test_filesystem_backward_compat() {
    // fd_buffer/write_fd should still work through FileSystem
    let mut state = RustSimState::new("amd64").unwrap();
    assert!(state.write_stdout(b"hello"));
    assert_eq!(state.stdout_buffer(), b"hello");
    assert!(state.has_stdout());

    assert!(state.write_fd(2, b"err"));
    assert_eq!(state.fd_buffer(2), b"err");
}

#[test]
fn test_filesystem_open_fds() {
    let mut fs = FileSystem::default();
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2]); // stdin, stdout, stderr

    fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    fs.open("b.txt".to_string(), FdFlags::WriteOnly);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 3, 4]);

    fs.close(3);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 4]);
}

#[test]
fn test_filesystem_dup_dup2_pipe() {
    let mut fs = FileSystem::default();
    let fd = fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);

    // dup
    let dup_fd = fs.dup(fd).unwrap();
    assert_eq!(dup_fd, 4);
    assert_eq!(fs.fd_info(dup_fd).unwrap().0, "a.txt");

    // dup of closed fd returns None
    fs.close(fd);
    assert!(fs.dup(fd).is_none());

    // dup2 with fresh state
    let mut fs2 = FileSystem::default();
    let src = fs2.open("src.txt".to_string(), FdFlags::ReadOnly);
    let dst = fs2.dup2(src, 10).unwrap();
    assert_eq!(dst, 10);
    assert_eq!(fs2.fd_info(10).unwrap().0, "src.txt");
    // next_fd advanced past 10
    assert!(fs2.next_fd() > 10);

    // dup2(self, self) returns self when open
    assert_eq!(fs2.dup2(src, src), Some(src));

    // pipe allocates two consecutive fds
    let mut fs3 = FileSystem::default();
    let (r, w) = fs3.pipe().unwrap();
    assert_eq!((r, w), (3, 4));
    assert!(fs3.is_open(r));
    assert!(fs3.is_open(w));
    assert_eq!(fs3.fd_info(r).unwrap().2, 0); // ReadOnly
    assert_eq!(fs3.fd_info(w).unwrap().2, 1); // WriteOnly
}

/// angr-03vl4.54: `lseek`'s offset is a raw guest `i64` and the SEEK_CUR base is
/// itself guest-steerable (a prior SEEK_SET accepts any non-negative `i64`), so
/// the old `(base as i64 + offset)` overflowed — a panic under
/// `[profile.release-checked]`, a silently wrapped position in the shipped
/// `[profile.release]`. `offset_position` clamps at both ends of the `u64` range
/// instead.
#[test]
fn seek_cur_and_end_clamp_instead_of_overflowing_the_signed_add() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);

    // Park the position at i64::MAX via SEEK_SET, then push past it. The first
    // bump lands just below the top of the u64 range (the old `as i64` add
    // overflowed here); the second saturates at u64::MAX rather than wrapping
    // back down into a small, valid-looking position.
    assert_eq!(fs.seek(fd, i64::MAX, 0), Some(i64::MAX as u64));
    assert_eq!(fs.seek(fd, i64::MAX, 1), Some(u64::MAX - 1));
    assert_eq!(fs.seek(fd, i64::MAX, 1), Some(u64::MAX));

    // A position above i64::MAX must stay positive: `base as i64` alone turned
    // it negative, which the old `.max(0)` then collapsed to 0.
    assert_eq!(fs.seek(fd, -1, 1), Some(u64::MAX - 1));
    assert_eq!(fs.seek(fd, i64::MIN, 1), Some(i64::MAX as u64 - 1));

    // The negative direction still clamps at 0, as the old `.max(0)` did.
    assert_eq!(fs.seek(fd, 10, 0), Some(10));
    assert_eq!(fs.seek(fd, i64::MIN, 1), Some(0));

    // SEEK_END shares the helper: a tiny file plus a huge offset must not wrap,
    // and i64::MIN must not negate into a positive.
    assert_eq!(fs.seek(fd, i64::MAX, 2), Some(100 + i64::MAX as u64));
    assert_eq!(fs.seek(fd, i64::MIN, 2), Some(0));
}

/// angr-03vl4.55: `pipe` allocates two consecutive fds off `next_fd`, which
/// `register_fd_at` can drive arbitrarily high from an imported Python state
/// (it is not bounded by `MAX_FD`). The bump is `checked_add`, so an exhausted
/// fd space refuses the pipe — leaving the table and `next_fd` untouched — and
/// `NativePipe` bounces to Python. Saturating instead would return `u32::MAX`
/// for both ends and alias them.
#[test]
fn pipe_refuses_rather_than_wrapping_when_the_fd_space_is_exhausted() {
    // next_fd = u32::MAX (the read end is the last fd) and next_fd = u32::MAX-1
    // (the read/write pair fits but the post-bump does not) — both refuse.
    for imported in [u32::MAX - 1, u32::MAX - 2] {
        let mut fs = FileSystem::default();
        assert!(fs.register_fd_at(
            imported,
            "imported".to_string(),
            FdFlags::ReadOnly,
            Vec::new(),
            0,
        ));
        let next_before = fs.next_fd();
        assert_eq!(next_before, imported + 1);
        let fds_before = fs.all_fds();

        assert_eq!(
            fs.pipe(),
            None,
            "pipe() must refuse with next_fd={next_before:#x}"
        );
        assert_eq!(fs.next_fd(), next_before, "a refused pipe must not bump");
        assert_eq!(fs.all_fds(), fds_before, "a refused pipe must insert nothing");
    }
}
