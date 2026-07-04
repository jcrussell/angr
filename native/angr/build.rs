use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");
    println!("cargo:rerun-if-env-changed=PYVEX_FFI_LIB_DIR");

    // Native libVEX FFI backend (non-default `libvex-ffi` feature): link the
    // venv's libpyvex.so so `vex_lift`/`vex_init` resolve. Gated so the default
    // build takes on no new rpath dependency. See rust_libvex_ffi.rst.
    if env::var("CARGO_FEATURE_LIBVEX_FFI").is_ok() {
        configure_pyvex_ffi();
    }

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
    // See CLAUDE.md "Common Issues" for troubleshooting.
    if let Some(lib_dir) = find_z3_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}

/// Emit the link flags for the `libvex-ffi` backend. Resolves the venv's
/// `pyvex/lib` (which holds `libpyvex.so`, the exact object pyvex loads via
/// cffi — see rust_libvex_ffi.rst) and emits link-search + link-lib + rpath so
/// the cdylib resolves `vex_lift`/`vex_init` at load time.
fn configure_pyvex_ffi() {
    if let Some(lib_dir) = find_pyvex_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=pyvex");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    } else {
        // Non-fatal: the feature is opt-in and the FFI decls land in a later
        // increment. Surface an actionable hint rather than a link failure.
        println!(
            "cargo:warning=libvex-ffi: could not locate pyvex/lib (libpyvex.so). \
             Set PYVEX_FFI_LIB_DIR or ensure `python3 -c 'import pyvex'` works."
        );
    }
}

/// Locate the directory containing `libpyvex.so`. Prefers an explicit override
/// (also set by `setup.py::_resolve_pyvex_libdir`), then the active venv's
/// pyvex package. Mirrors `find_z3_lib_dir` / `find_z3_pkg_from_python`.
fn find_pyvex_lib_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("PYVEX_FFI_LIB_DIR") {
        let p = PathBuf::from(path);
        if p.join("libpyvex.so").exists() || p.join("libpyvex.dylib").exists() {
            return Some(p);
        }
    }

    let output = Command::new("python3")
        .args([
            "-c",
            "import pyvex, os; print(os.path.dirname(pyvex.__file__))",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pkg_dir = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    let lib_dir = pkg_dir.join("lib");
    if lib_dir.join("libpyvex.so").exists() || lib_dir.join("libpyvex.dylib").exists() {
        Some(lib_dir)
    } else {
        None
    }
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
