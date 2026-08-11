//! Tests for [`super::filesystem`] — the POSIX-fd FileSystem model.
//!
//! Concrete-only surface (no Z3 context needed): fd allocation, read/write
//! position semantics, seek/pread/pwrite, dup/dup2/pipe, path normalization,
//! symlinks, and the fd-metadata queries. The bounded-symbolic-content paths
//! (`content_sym` / `RustBV`) are exercised by the Python-level rust suite;
//! these stay Z3-free so they run under a plain `cargo test --lib`.
//!
//! This file owns the whole *concrete, Z3-free* `FileSystem` surface. Its
//! three siblings under `state/tests/` are split off for a reason, not by
//! accident: `filesystem_symbolic` and `filesystem_demote` need a live
//! `SymContext` for `content_sym`, and `filesystem_state` needs a whole
//! [`RustSimState`](super::RustSimState) around the fs (fork isolation, the
//! `write_stdout`/`fd_buffer` convenience wrappers). Anything reachable from a
//! bare `FileSystem::default()` belongs here — angr-c7xno.69's mechanical
//! carve-out of `state_tests.rs` had re-tested this file's CRUD/fd-table
//! surface under a second set of idioms, and angr-03vl4.56 folded those
//! duplicates back in.

use super::*;

#[test]
fn fdflags_posix_roundtrip() {
    for (raw, expected) in [
        (0u32, FdFlags::ReadOnly),
        (1, FdFlags::WriteOnly),
        (2, FdFlags::ReadWrite),
    ] {
        assert_eq!(FdFlags::from_posix(raw), expected);
        assert_eq!(expected.to_posix(), raw);
    }
    // Only the low two bits select the mode; O_CREAT/O_TRUNC etc. are ignored.
    assert_eq!(FdFlags::from_posix(0o100 | 1), FdFlags::WriteOnly);
    // 3 masks to ReadWrite (the `_` arm), mirroring the glibc O_ACCMODE fold.
    assert_eq!(FdFlags::from_posix(3), FdFlags::ReadWrite);
}

#[test]
fn default_preregisters_std_streams() {
    let fs = FileSystem::default();
    assert!(fs.is_open(0) && fs.is_open(1) && fs.is_open(2));
    assert!(!fs.is_open(3), "nothing above stderr is preregistered");
    assert_eq!(fs.next_fd(), 3);
    assert_eq!(fs.cwd(), b"/");
    assert_eq!(
        fs.fd_info(0).map(|i| i.2),
        Some(FdFlags::ReadOnly.to_posix())
    );
    assert_eq!(
        fs.fd_info(1).map(|i| i.2),
        Some(FdFlags::WriteOnly.to_posix())
    );
}

#[test]
fn open_allocates_monotonic_fds_and_marks_path_known() {
    let mut fs = FileSystem::default();
    let a = fs
        .open("flag.txt".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    let b = fs
        .open("/etc/passwd".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert_eq!((a, b), (3, 4));
    assert_eq!(fs.next_fd(), 5);
    // known_paths is cwd-normalized: relative "flag.txt" from cwd "/" -> "/flag.txt".
    assert!(fs.is_path_known("flag.txt"));
    assert!(fs.is_path_known("/flag.txt"));
    assert!(fs.is_path_known("/etc/passwd"));
    assert!(!fs.is_path_known("/nope"));
}

#[test]
fn write_then_read_tracks_position() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, Vec::new())
        .expect("fd space is not exhausted in tests");
    assert!(fs.write(fd, b"hello"));
    // Sequential write leaves position at EOF.
    assert_eq!(fs.fd_info(fd).map(|i| i.1), Some(5));
    assert_eq!(fs.fd_content(fd), b"hello");
    // Reading from EOF yields nothing; rewind first.
    assert!(fs.read(fd, 8).is_empty());
    assert_eq!(fs.seek(fd, 0, 0), Some(0));
    assert_eq!(fs.read(fd, 3), b"hel");
    assert_eq!(fs.read(fd, 8), b"lo"); // clamped to available
    assert!(fs.read(fd, 8).is_empty());
}

#[test]
fn read_closed_fd_serves_nothing() {
    // Parity with read_sym/read_sym_at: a closed fd serves no bytes even
    // though its content buffer survives (angr-myzjx.23).
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content(
            "data.bin".to_string(),
            FdFlags::ReadOnly,
            b"hello world".to_vec(),
        )
        .expect("fd space is not exhausted in tests");
    assert_eq!(fs.read_at(fd, 0, 5), b"hello");
    assert!(fs.close(fd));

    // Both the position-advancing and positioned reads must refuse a closed fd.
    assert!(fs.read(fd, 5).is_empty());
    assert!(fs.read_at(fd, 0, 5).is_empty());
}

#[test]
fn write_overwrites_in_place_and_zero_fills_after_a_sparse_seek() {
    // `write` is position-aware (see the `write-fd-position-aware` bd memory),
    // not append-only: it honors a prior seek instead of extending the buffer.
    let mut fs = FileSystem::default();
    let fd = fs
        .open("out.bin".to_string(), FdFlags::WriteOnly)
        .expect("fd space is not exhausted in tests");

    // Sequential writes (no seek): byte-identical to a plain append.
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
fn write_at_position_zero_fills_gap() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, Vec::new())
        .expect("fd space is not exhausted in tests");
    // pwrite past EOF zero-fills the gap and leaves the fd position untouched.
    assert!(fs.write_at(fd, 4, b"AB"));
    assert_eq!(fs.fd_content(fd), b"\0\0\0\0AB");
    assert_eq!(fs.fd_info(fd).map(|i| i.1), Some(0));
    // pread is also position-independent.
    assert_eq!(fs.read_at(fd, 4, 2), b"AB");
    assert!(fs.read_at(fd, 99, 2).is_empty());
    assert_eq!(fs.fd_info(fd).map(|i| i.1), Some(0));
}

#[test]
fn empty_write_is_noop_and_returns_true() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::WriteOnly, Vec::new())
        .expect("fd space is not exhausted in tests");
    assert!(fs.write(fd, b""));
    assert!(fs.write_at(fd, 10, b""));
    assert!(fs.fd_content(fd).is_empty());
    assert_eq!(fs.fd_info(fd).map(|i| i.1), Some(0));
}

#[test]
fn seek_whence_variants_and_invalid() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, b"0123456789".to_vec())
        .expect("fd space is not exhausted in tests");
    assert_eq!(fs.seek(fd, 3, 0), Some(3)); // SEEK_SET
    assert_eq!(fs.seek(fd, 2, 1), Some(5)); // SEEK_CUR
    assert_eq!(fs.seek(fd, -1, 2), Some(9)); // SEEK_END (len 10 - 1)
    // Negative absolute positions saturate at 0.
    assert_eq!(fs.seek(fd, -100, 1), Some(0));
    // Unknown whence is rejected without moving the position.
    assert_eq!(fs.seek(fd, 0, 7), None);
    assert_eq!(fs.fd_info(fd).map(|i| i.1), Some(0));
    // Seek on a missing fd is None.
    assert_eq!(fs.seek(999, 0, 0), None);
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
    let fd = fs
        .open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100])
        .expect("fd space is not exhausted in tests");

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

#[test]
fn close_flips_flag_and_is_idempotent() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open("f".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert!(fs.is_open(fd));
    assert!(fs.close(fd));
    assert!(!fs.is_open(fd));
    // Second close returns false; closing a never-opened fd also false.
    assert!(!fs.close(fd));
    assert!(!fs.close(12345));
}

#[test]
fn dup_clones_state_at_lowest_free_fd() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, b"data".to_vec())
        .expect("fd space is not exhausted in tests");
    assert_eq!(fs.seek(fd, 2, 0), Some(2));
    let dupd = fs.dup(fd).expect("dup of open fd");
    assert_eq!(dupd, 4);
    // The dup snapshots position and content (independent buffers thereafter).
    assert_eq!(fs.fd_info(dupd).map(|i| (i.1, i.3)), Some((2, 4)));
    // dup of a closed / missing fd returns None.
    assert!(fs.close(fd));
    assert!(fs.dup(fd).is_none());
    assert!(fs.dup(777).is_none());
}

#[test]
fn dup_reuses_a_closed_fd_slot() {
    // angr-9ke6b.119: dup(2) hands out the lowest free number, not a
    // monotonically increasing one. open 3, open 4, close 3 -> the next dup
    // must land back on 3 even though next_fd has moved past it.
    let mut fs = FileSystem::default();
    let a = fs
        .open("a".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    let b = fs
        .open("b".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert_eq!((a, b), (3, 4));
    assert!(fs.close(a));
    assert_eq!(fs.next_fd(), 5);

    let dupd = fs.dup(b).expect("dup of open fd");
    assert_eq!(dupd, 3, "dup must reuse the freed slot");
    assert!(fs.is_open(3));
    assert_eq!(fs.fd_info(3).map(|i| i.0), Some("b"));
    // The reused slot is no longer a candidate; the next gap is 5 (== next_fd).
    let dupd2 = fs.dup(b).expect("dup of open fd");
    assert_eq!(dupd2, 5);
    // next_fd stays past every allocated fd so a later open cannot collide.
    assert_eq!(fs.next_fd(), 6);
    assert_eq!(
        fs.open("c".to_string(), FdFlags::ReadOnly)
            .expect("fd space is not exhausted in tests"),
        6
    );
}

#[test]
fn dup2_closes_target_and_bumps_next_fd() {
    let mut fs = FileSystem::default();
    let src = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, b"xy".to_vec())
        .expect("fd space is not exhausted in tests");
    // dup2 onto a high fd number bumps next_fd past it.
    assert_eq!(fs.dup2(src, 20), Some(20));
    assert!(fs.is_open(20));
    assert_eq!(fs.next_fd(), 21);
    // Same-fd dup2 is a no-op returning newfd.
    assert_eq!(fs.dup2(src, src), Some(src));
    // dup2 from a non-open fd fails.
    assert!(fs.dup2(999, 30).is_none());
}

#[test]
fn dup2_refuses_newfd_at_or_above_max_fd() {
    let mut fs = FileSystem::default();
    let src = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, b"xy".to_vec())
        .expect("fd space is not exhausted in tests");
    let before = fs.next_fd();

    // u32::MAX would wrap the `next_fd = newfd + 1` bump (angr-03vl4.52).
    assert_eq!(fs.dup2(src, u32::MAX), None);
    // The boundary itself is out of range; one below it is not.
    assert_eq!(fs.dup2(src, MAX_FD), None);
    assert!(!fs.is_open(u32::MAX));
    assert!(!fs.is_open(MAX_FD));
    assert_eq!(fs.next_fd(), before, "a refused dup2 must not move next_fd");

    assert_eq!(fs.dup2(src, MAX_FD - 1), Some(MAX_FD - 1));
    assert_eq!(fs.next_fd(), MAX_FD);
}

#[test]
fn pipe_allocates_consecutive_read_write_ends() {
    let mut fs = FileSystem::default();
    let (r, w) = fs.pipe().unwrap();
    assert_eq!((r, w), (3, 4));
    assert_eq!(fs.next_fd(), 5);
    assert_eq!(
        fs.fd_info(r).map(|i| i.2),
        Some(FdFlags::ReadOnly.to_posix())
    );
    assert_eq!(
        fs.fd_info(w).map(|i| i.2),
        Some(FdFlags::WriteOnly.to_posix())
    );
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
        assert_eq!(
            fs.all_fds(),
            fds_before,
            "a refused pipe must insert nothing"
        );
    }
}

/// angr-03vl4.88: the three single-fd allocators bumped `next_fd` with a plain
/// `+= 1`, which wraps to 0 in the shipped release profile once `register_fd_at`
/// has imported a Python-chosen fd of `u32::MAX` — the next `open()` would hand
/// out fd 0 (stdin) as a fresh file. Each must refuse instead, leaving `next_fd`
/// and the fd table untouched.
#[test]
fn open_family_refuses_rather_than_wrapping_when_the_fd_space_is_exhausted() {
    type Allocator = fn(&mut FileSystem) -> Option<u32>;
    let allocators: [(&str, Allocator); 3] = [
        ("open", |fs| fs.open("f".to_string(), FdFlags::ReadOnly)),
        ("open_with_content", |fs| {
            fs.open_with_content("f".to_string(), FdFlags::ReadOnly, vec![1, 2, 3])
        }),
        ("open_symbolic", |fs| {
            fs.open_symbolic("f".to_string(), FdFlags::ReadOnly)
        }),
    ];

    for (label, alloc) in allocators {
        let mut fs = FileSystem::default();
        assert!(fs.register_fd_at(
            u32::MAX,
            "imported".to_string(),
            FdFlags::ReadOnly,
            Vec::new(),
            0,
        ));
        // register_fd_at's bump saturates, so next_fd is pinned at u32::MAX —
        // the fd itself is still free to hand out, but nothing after it is.
        assert_eq!(fs.next_fd(), u32::MAX);
        let fds_before = fs.all_fds();

        assert_eq!(fs.next_fd(), u32::MAX, "{label}: precondition");
        assert_eq!(alloc(&mut fs), None, "{label} must refuse at u32::MAX");
        assert_eq!(fs.next_fd(), u32::MAX, "a refused {label} must not bump");
        assert_eq!(
            fs.all_fds(),
            fds_before,
            "a refused {label} must insert nothing"
        );

        // One below the wall the same allocator succeeds, proving the refusal
        // is the overflow guard and not a blanket decline.
        let mut fs = FileSystem::default();
        assert!(fs.register_fd_at(
            u32::MAX - 2,
            "imported".to_string(),
            FdFlags::ReadOnly,
            Vec::new(),
            0,
        ));
        assert_eq!(
            alloc(&mut fs),
            Some(u32::MAX - 1),
            "{label} must still allocate below the wall"
        );
        assert_eq!(fs.next_fd(), u32::MAX);
    }
}

#[test]
fn normalize_path_resolves_dot_dotdot_and_cwd() {
    let mut fs = FileSystem::default();
    assert_eq!(fs.normalize_path("/a/b/../c"), "/a/c");
    assert_eq!(fs.normalize_path("/a/./b//c"), "/a/b/c");
    // `..` past root saturates at root.
    assert_eq!(fs.normalize_path("/../../x"), "/x");
    assert_eq!(fs.normalize_path("/../.."), "/");
    assert_eq!(fs.normalize_path("/"), "/");
    // Relative paths join against cwd.
    fs.set_cwd(b"/home/user".to_vec());
    assert_eq!(fs.normalize_path("file"), "/home/user/file");
    assert_eq!(fs.normalize_path("../peer"), "/home/peer");
    // The empty path is the cwd itself, not a trailing-slash spelling of it.
    assert_eq!(fs.normalize_path(""), "/home/user");
    // Path is truncated at the first NUL — before, not after, the cwd join.
    assert_eq!(fs.normalize_path("/a\0/b"), "/a");
    assert_eq!(fs.normalize_path("x\0junk"), "/home/user/x");
}

#[test]
fn symlink_register_and_lookup() {
    let mut fs = FileSystem::default();
    assert!(fs.readlink_target("/link").is_none());
    fs.add_symlink("/link".to_string(), b"/target".to_vec());
    assert_eq!(fs.readlink_target("/link"), Some(b"/target".as_slice()));
    // Overwrite replaces the target.
    fs.add_symlink("/link".to_string(), b"/other".to_vec());
    assert_eq!(fs.readlink_target("/link"), Some(b"/other".as_slice()));
}

#[test]
fn content_size_for_path_uses_max_across_fds() {
    let mut fs = FileSystem::default();
    assert_eq!(fs.content_size_for_path("f"), None);
    let fd = fs
        .open_with_content("f".to_string(), FdFlags::ReadWrite, b"abc".to_vec())
        .expect("fd space is not exhausted in tests");
    assert_eq!(fs.content_size_for_path("f"), Some(3));
    // A longer write on the same path raises the reported size.
    assert!(fs.write(fd, b"abcdef"));
    assert_eq!(fs.content_size_for_path("f"), Some(6));
}

/// angr-9ke6b.120: `content_size_for_path` must key each fd off the cwd
/// normalization frozen at open (`FileDescriptor::norm_name`), not the
/// *current* cwd. Two fds opened on the same real file under different
/// cwds must both count toward the reported size after a `chdir`.
#[test]
fn content_size_for_path_uses_cwd_at_open_across_chdir() {
    let mut fs = FileSystem::default();
    fs.set_cwd(b"/x/y".to_vec());
    // fd A: relative "a.txt" under /x/y  ->  /x/y/a.txt
    let a = fs
        .open_with_content("a.txt".to_string(), FdFlags::ReadWrite, b"abcdef".to_vec())
        .expect("fd space is not exhausted in tests");
    // Guest chdir to /x, then reopen the same real file relative to it.
    fs.set_cwd(b"/x".to_vec());
    let b = fs
        .open_with_content("y/a.txt".to_string(), FdFlags::ReadWrite, b"abc".to_vec())
        .expect("fd space is not exhausted in tests");
    assert_ne!(a, b);
    // Both spellings name /x/y/a.txt, so the max across both fds wins.
    assert_eq!(fs.content_size_for_path("y/a.txt"), Some(6));
    assert_eq!(fs.content_size_for_path("/x/y/a.txt"), Some(6));
    // fd A must not leak into /x/a.txt, the path a current-cwd
    // re-normalization of its raw name would have produced.
    assert_eq!(fs.content_size_for_path("/x/a.txt"), None);
}

#[test]
fn effective_len_maxes_concrete_buffer() {
    let d = FileDescriptor::with_content("f".to_string(), FdFlags::ReadOnly, b"hello".to_vec());
    assert_eq!(d.effective_len(), 5);
    let empty = FileDescriptor::new("g".to_string(), FdFlags::WriteOnly);
    assert_eq!(empty.effective_len(), 0);
}

#[test]
fn open_fds_and_all_fds_reflect_close() {
    let mut fs = FileSystem::default();
    let fd = fs
        .open("f".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    let mut open = fs.open_fds();
    open.sort_unstable();
    assert_eq!(open, vec![0, 1, 2, fd]);
    fs.close(fd);
    let mut still_open = fs.open_fds();
    still_open.sort_unstable();
    assert_eq!(still_open, vec![0, 1, 2]);
    // all_fds includes the closed fd; open_fds does not.
    assert!(fs.all_fds().contains(&fd));
}

#[test]
fn is_symbolic_only_for_symbolic_stream_open_fds() {
    let mut fs = FileSystem::default();
    let concrete = fs
        .open("f".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    let sym = fs
        .open_symbolic("s".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert!(!fs.is_symbolic(concrete));
    assert!(fs.is_symbolic(sym));
    // Closing clears the symbolic-serve predicate.
    fs.close(sym);
    assert!(!fs.is_symbolic(sym));
    assert!(!fs.is_symbolic(4242));
}
