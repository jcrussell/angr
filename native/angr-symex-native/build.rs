//! Runpath the crate's binaries to the libz3 they link.
//!
//! z3-sys links whatever `Z3_LIBRARY_PATH_OVERRIDE` names but emits no
//! runpath, and that directory (the z3-solver wheel's `lib/`) is outside
//! `target/`, so cargo does not put it on the loader path for `cargo test`
//! either. Imported from Python the extension does not need this:
//! `angr/_z3.py` opens libz3 before `angr.rustylib` loads. `setup.py`
//! (`configure_z3`) and `tools/rebuild-rust.sh` set the override; CI uses
//! z3-sys's `gh-release` feature instead, which downloads into `target/`.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");
    // PE has no runpath (link.exe rejects -rpath); Windows finds the DLL
    // through PATH or `os.add_dll_directory`.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        return;
    }
    if let Ok(dir) = env::var("Z3_LIBRARY_PATH_OVERRIDE") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
