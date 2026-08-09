//! Tests for the shared external-input width / Extract-bounds guards.

use super::*;

#[test]
fn test_check_bv_width_allows_zero_and_ordinary_widths() {
    // Width 0 is a supported degenerate concrete value, so it must pass.
    for w in [0, 1, 8, 64, 1024, MAX_BV_WIDTH] {
        assert!(check_bv_width("op", w).is_ok(), "width {w} should be allowed");
    }
}

#[test]
fn test_check_bv_width_rejects_above_max() {
    for w in [MAX_BV_WIDTH + 1, 1 << 30, u32::MAX] {
        let err = check_bv_width("create_symbolic", w)
            .expect_err("width {w} should be rejected")
            .to_string();
        assert!(err.contains("create_symbolic"), "op name missing: {err}");
        assert!(err.contains(&w.to_string()), "width missing: {err}");
    }
}

#[test]
fn test_check_extract_bounds_accepts_in_range() {
    assert!(check_extract_bounds("op_extract", 7, 0, 8).is_ok());
    assert!(check_extract_bounds("op_extract", 7, 7, 8).is_ok());
    assert!(check_extract_bounds("op_extract", 0, 0, 1).is_ok());
}

#[test]
fn test_check_extract_bounds_rejects_high_past_source() {
    let err = check_extract_bounds("op_extract", 8, 0, 8).expect_err("high == width is past the end");
    assert!(err.contains("high=8"), "{err}");
    assert!(err.contains("width=8"), "{err}");
}

#[test]
fn test_check_extract_bounds_rejects_low_above_high() {
    // The wrap this prevents: `high - low + 1` for (2, 5) is u32::MAX - 1.
    let err = check_extract_bounds("op_extract", 2, 5, 8).expect_err("low > high is invalid");
    assert!(err.contains("low=5"), "{err}");
    assert!(err.contains("high=2"), "{err}");
}

#[test]
fn test_check_extract_bounds_rejects_any_range_on_zero_width_source() {
    assert!(check_extract_bounds("op_extract", 0, 0, 0).is_err());
}
