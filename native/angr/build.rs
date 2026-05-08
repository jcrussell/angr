use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");

    // Only configure Z3 paths when the z3 feature is enabled.
    if env::var("CARGO_FEATURE_Z3").is_err() {
        return;
    }

    // Locate the Z3 shared library and emit rpath + link-search so the cdylib
    // finds it at runtime regardless of LD_LIBRARY_PATH. Prefer the active
    // python venv's z3 package — Python and Rust share libz3.so for AST
    // passthrough, so they must load the same library version.
    //
    // Header discovery is z3-sys's job: when Z3_SYS_Z3_HEADER is unset,
    // z3-sys probes pkg-config and uses its bundled wrapper.h. That works
    // out of the box on systems with z3-dev (apt) / z3-devel (dnf) / brew z3.
    // See CLAUDE.md "Common Issues" for troubleshooting.
    if let Some(lib_dir) = find_z3_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
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
        .args([
            "-c",
            "import z3, os; print(os.path.dirname(z3.__file__))",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path_str = String::from_utf8(output.stdout).ok()?.trim().to_string();
    let path = PathBuf::from(path_str);
    if path.exists() { Some(path) } else { None }
}
