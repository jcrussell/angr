//! Build script for angr native library.
//!
//! This script configures linking to libpyvex when the native-lift feature is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only configure libpyvex linking when native-lift feature is enabled
    if env::var("CARGO_FEATURE_NATIVE_LIFT").is_ok() {
        configure_libpyvex();
    }
}

fn configure_libpyvex() {
    // Try to find pyvex lib path using Python
    let pyvex_lib_path = find_pyvex_lib_path();

    if let Some(path) = pyvex_lib_path {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-lib=dylib=pyvex");

        // Set rpath so the library can be found at runtime
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());

        // Rerun if the pyvex library changes
        let lib_file = path.join("libpyvex.so");
        if lib_file.exists() {
            println!("cargo:rerun-if-changed={}", lib_file.display());
        }

        eprintln!("cargo:warning=Found libpyvex at: {}", path.display());
    } else {
        // Fall back to checking common locations
        let fallback_paths = get_fallback_paths();
        let mut found = false;

        for path in fallback_paths {
            let lib_file = path.join("libpyvex.so");
            if lib_file.exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
                println!("cargo:rustc-link-lib=dylib=pyvex");

                #[cfg(target_os = "linux")]
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());

                eprintln!("cargo:warning=Found libpyvex at: {}", path.display());
                found = true;
                break;
            }
        }

        if !found {
            // Don't fail the build - the FFI module will return errors at runtime
            eprintln!(
                "cargo:warning=libpyvex not found. Native VEX lifting will not be available."
            );
            eprintln!("cargo:warning=Install pyvex (pip install pyvex) to enable native lifting.");
        }
    }
}

/// Find pyvex lib path by running Python
fn find_pyvex_lib_path() -> Option<PathBuf> {
    // Try python3 first, then python
    for python in &["python3", "python"] {
        let output = Command::new(python)
            .args(["-c", "import pyvex; print(pyvex.lib_path)"])
            .output()
            .ok()?;

        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path = PathBuf::from(path_str.trim());
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

/// Get fallback paths to check for libpyvex
fn get_fallback_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Check relative .venv path (common for local development)
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let project_root = PathBuf::from(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());

        if let Some(root) = project_root {
            // Check .venv/lib/python*/site-packages/pyvex/lib/
            for version in &["3.12", "3.11", "3.10", "3.9"] {
                let path = root.join(format!(
                    ".venv/lib/python{}/site-packages/pyvex/lib",
                    version
                ));
                paths.push(path);
            }
        }
    }

    // Check system Python site-packages
    if let Ok(home) = env::var("HOME") {
        for version in &["3.12", "3.11", "3.10", "3.9"] {
            let path = PathBuf::from(&home).join(format!(
                ".local/lib/python{}/site-packages/pyvex/lib",
                version
            ));
            paths.push(path);
        }
    }

    // Check VIRTUAL_ENV if set
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        for version in &["3.12", "3.11", "3.10", "3.9"] {
            let path = PathBuf::from(&venv).join(format!(
                "lib/python{}/site-packages/pyvex/lib",
                version
            ));
            paths.push(path);
        }
    }

    paths
}
