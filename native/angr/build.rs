// Grandfathered clippy::unwrap_used/expect_used debt -- angr-9ke6b.212 tracks
// burning this down file by file. Do not add new unwrap()/expect() calls here;
// new files/callers must handle the None/Err case explicitly instead.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");
    // `find_z3_pkg_from_python` shells out to whichever `python3` is on PATH,
    // so the answer changes when a venv is activated/deactivated. Without this
    // key the first answer is cached forever, and a build that ran without the
    // venv leaves the system libz3 rpath baked into every later `cargo test`
    // — two libz3 objects in one process, which segfaults the moment claripy's
    // `Z3_context` crosses into Rust (angr-exwth).
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    // Only configure Z3 paths when the z3 feature is enabled.
    if env::var("CARGO_FEATURE_Z3").is_err() {
        return;
    }

    // Locate the Z3 shared library and emit rpath + link-search so the cdylib
    // finds it at runtime regardless of LD_LIBRARY_PATH. Prefer the active
    // python venv's z3 package — Python and Rust share libz3.so for AST
    // passthrough, so they must load the same library version.
    //
    // Header discovery: setup.py's `_resolve_z3_header()` probes common
    // locations (venv, pkg-config, /usr/include, brew, MacPorts) and sets
    // Z3_SYS_Z3_HEADER before cargo runs. When invoking cargo directly
    // (outside `pip install`), z3-sys falls back to pkg-config on its own.
    // See docs/advanced-topics/rust_build.rst "Common issues" for troubleshooting.
    if let Some(lib_dir) = find_z3_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        emit_rpath(&lib_dir);
    }
}

/// Emit a runtime library search path for `lib_dir`.
///
/// ELF and Mach-O both take `-Wl,-rpath`. PE has no equivalent — link.exe
/// rejects the flag outright — so on Windows the extension's DLL search path is
/// established at import time instead: `angr.misc.z3_dll.add_z3_dll_directory()`
/// hands the z3-solver package's `lib` directory to `os.add_dll_directory()`
/// before anything imports `angr.rustylib`. Same invariant either way — the
/// extension and claripy must resolve one libz3, not two.
fn emit_rpath(lib_dir: &std::path::Path) {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        return;
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}

fn find_z3_lib_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("Z3_LIBRARY_PATH_OVERRIDE") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(z3_pkg) = find_z3_pkg_from_python() {
        let lib_dir = z3_pkg.join("lib");
        if lib_dir.join("libz3.so").exists()
            || lib_dir.join("libz3.dylib").exists()
            || lib_dir.join("z3.dll").exists()
        {
            return Some(lib_dir);
        }
    }

    for candidate in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
        "/opt/homebrew/lib",
    ] {
        let p = PathBuf::from(candidate);
        if p.join("libz3.so").exists() || p.join("libz3.dylib").exists() {
            return Some(p);
        }
    }
    None
}

fn find_z3_pkg_from_python() -> Option<PathBuf> {
    let output = Command::new("python3")
        .args(["-c", "import z3, os; print(os.path.dirname(z3.__file__))"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path_str = String::from_utf8(output.stdout).ok()?.trim().to_string();
    let path = PathBuf::from(path_str);
    if path.exists() { Some(path) } else { None }
}
