//! In-module unit tests for `claripy_bridge/export.rs`.
//!
//! The width/coercion decision-seam tests that used to live here moved to
//! `export/width_decisions_tests.rs` alongside their subject (angr-5mnx3.11);
//! what stays is the textual invariant the `RustBV::Constrained` export arm
//! leans on, which is a property of `export.rs` itself.

/// Pins the reachability claim the `RustBV::Constrained` export arm's rationale
/// leans on: snapshot deserialization is the only live constructor of that
/// variant, so the `get_claripy_ast(id)` lookup the `Symbolic` arm performs
/// would be a guaranteed miss here. A new live constructor is exactly the change
/// that would make the skipped lookup observable, so it should land as a failure
/// of this test — which names the arm to revisit — rather than as a silent
/// behaviour change.
///
/// Textual, like `callbacks::inspect_bits_tests::bits_match_python_inspect_event_specs`:
/// the property is "no such call site exists anywhere in the crate", which no
/// amount of runtime exercising can demonstrate.
#[test]
fn constrained_has_no_live_constructor_beyond_snapshot_load() {
    let src_root = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut files = Vec::new();
    collect_rs_files(std::path::Path::new(src_root), &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files under {src_root} -- the scan would pass vacuously"
    );

    let mut snapshot_sites = 0usize;
    let mut other_sites = Vec::new();
    for path in &files {
        // Unit tests build `Constrained` values freely; the invariant is about
        // production paths. Test modules live in sibling `*_tests.rs` files.
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_tests.rs"))
        {
            continue;
        }
        // SILENT(cat-a): a file that cannot be read is not a call site; the
        // `files.is_empty()` guard above already rules out a wholly failed scan.
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (lineno, line) in src.lines().enumerate() {
            let Some(col) = line.find("RustBV::Constrained {") else {
                continue;
            };
            if line.trim_start().starts_with("//") {
                continue;
            }
            // Struct-literal position vs. pattern position. Patterns sit in
            // `match` arms, `matches!`, tuple patterns and `|` alternations, all
            // of which put the arrow (if any) *after* the occurrence; a literal
            // sits on the value side of `=>` or `=`.
            let before = line[..col].trim_end();
            let is_literal = before.ends_with("=>")
                || before.ends_with("return")
                || (before.ends_with('=')
                    && !before.ends_with("==")
                    && !before.ends_with("!=")
                    && !before.ends_with("<=")
                    && !before.ends_with(">="));
            if !is_literal {
                continue;
            }
            if before.contains("RustBVData::Constrained") {
                snapshot_sites += 1;
            } else {
                other_sites.push(format!("{}:{}", path.display(), lineno + 1));
            }
        }
    }

    assert!(
        other_sites.is_empty(),
        "new live constructor(s) of RustBV::Constrained: {other_sites:?} -- \
         revisit the Constrained arm of rustbv_to_claripy_memo, which skips the \
         identity-cache lookup on the grounds that no live path produces a \
         Constrained carrying a registered symbol id"
    );
    assert_eq!(
        snapshot_sites, 1,
        "expected exactly one RustBV::Constrained literal (the RustBVData \
         deserialization arm in symbolic/value.rs); the scan may have gone stale"
    );
}

fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    // SILENT(cat-a): an unreadable directory contributes no call sites; the
    // caller asserts the overall scan found files.
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
