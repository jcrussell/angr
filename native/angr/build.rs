// Grandfathered clippy::unwrap_used/expect_used debt -- angr-9ke6b.212 tracks
// burning this down file by file. Do not add new unwrap()/expect() calls here;
// new files/callers must handle the None/Err case explicitly instead.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");
    println!("cargo:rerun-if-env-changed=PYVEX_FFI_LIB_DIR");

    // Native libVEX FFI backend (`libvex-ffi` feature, default-ON via setup.py
    // since angr-3trr7): link the venv's libpyvex.so so `vex_lift`/`vex_init`
    // resolve. Still feature-gated so `ANGR_LIBVEX_FFI=0` yields a build with no
    // libpyvex.so rpath dependency. See rust_libvex_ffi.rst.
    if env::var("CARGO_FEATURE_LIBVEX_FFI").is_ok() {
        configure_pyvex_ffi();
        generate_pyvex_ffi_bindings();
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

/// Emit the link flags for the `libvex-ffi` backend. Resolves the venv's
/// `pyvex/lib` (which holds `libpyvex.so`, the exact object pyvex loads via
/// cffi — see rust_libvex_ffi.rst) and emits link-search + link-lib + rpath so
/// the cdylib resolves `vex_lift`/`vex_init` at load time.
fn configure_pyvex_ffi() {
    if let Some(lib_dir) = find_pyvex_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=pyvex");
        emit_rpath(&lib_dir);
    } else {
        // Non-fatal: the feature is opt-in and the FFI decls land in a later
        // increment. Surface an actionable hint rather than a link failure.
        println!(
            "cargo:warning=libvex-ffi: could not locate pyvex/lib (libpyvex.so). \
             Set PYVEX_FFI_LIB_DIR or ensure `python3 -c 'import pyvex'` works."
        );
    }
}

/// Generate Rust FFI declarations for the libVEX seam from the vendored cffi
/// cdef (`vendor/pyvex_ffi.h`, extracted from `pyvex.vex_ffi.ffi_str` — see
/// tools/regen-pyvex-ffi-header.py). bindgen writes `pyvex_ffi_bindings.rs`
/// into `OUT_DIR`, which `vex/libvex_ffi.rs` includes. Binding against pyvex's
/// own cdef keeps the struct ABI (VEXLiftResult, IRSB, IRStmt/IRExpr) in lock
/// step with the object we link, which is the parity guarantee.
#[cfg(feature = "libvex-ffi")]
fn generate_pyvex_ffi_bindings() {
    let header = "vendor/pyvex_ffi.h";
    println!("cargo:rerun-if-changed={header}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let bindings = bindgen::Builder::default()
        .header(header)
        // Only the libVEX seam + its reachable IR types — not stddef.h noise.
        .allowlist_function("vex_lift")
        .allowlist_function("vex_init")
        .allowlist_function("register_readonly_region")
        .allowlist_function("deregister_all_readonly_regions")
        .allowlist_function("register_initial_register_value")
        .allowlist_function("reset_initial_register_values")
        .allowlist_type("VEXLiftResult")
        .allowlist_type("IRSB")
        .allowlist_type("VexArch")
        .allowlist_type("VexArchInfo")
        // Pull in every type reachable from the allowlisted roots (IRStmt,
        // IRExpr, IRConst, ExitInfo, DataRef, ConstVal, …).
        .allowlist_recursively(true)
        // Rustified enums are ergonomic but risk UB on unknown discriminants
        // coming from C; a newtype-with-consts is the safe default for FFI.
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("bindgen failed to generate pyvex FFI bindings from vendor/pyvex_ffi.h");

    bindings
        .write_to_file(out_dir.join("pyvex_ffi_bindings.rs"))
        .expect("failed to write pyvex_ffi_bindings.rs into OUT_DIR");

    generate_pyvex_ffi_enum_names(&out_dir);
}

/// Emit reverse `discriminant -> "Iop_Add32"` name tables for the libVEX IR
/// enums, parsed from the same vendored cdef bindgen consumes. The
/// `NativeLibVEXLifter` marshaller (angr-3s5js.3) reads the raw C integer tag
/// out of a `VEXLiftResult` and needs the pyvex-style enum *name* to reuse the
/// existing string parsers in `vex/opcode_map.rs` (`parse_opcode`, `parse_type`,
/// `parse_jumpkind`, `parse_endness`). Generating the tables from
/// `vendor/pyvex_ffi.h` keeps them in lock step with the linked object — the
/// same parity guarantee the bindings rely on. Written to
/// `OUT_DIR/pyvex_ffi_enum_names.rs`, included by `vex/libvex_ffi.rs`.
#[cfg(feature = "libvex-ffi")]
fn generate_pyvex_ffi_enum_names(out_dir: &std::path::Path) {
    let src = std::fs::read_to_string("vendor/pyvex_ffi.h")
        .expect("failed to read vendor/pyvex_ffi.h for enum-name tables");

    let mut out = String::new();
    out.push_str("// @generated by build.rs::generate_pyvex_ffi_enum_names — do not edit.\n");
    // (base enumerator, generated fn name)
    for (base, fn_name) in [
        ("Iop_INVALID", "irop_name"),
        ("Ity_INVALID", "irtype_name"),
        ("Ijk_INVALID", "ijk_name"),
        ("Iend_LE", "iend_name"),
    ] {
        emit_enum_name_table(&src, base, fn_name, &mut out);
    }

    std::fs::write(out_dir.join("pyvex_ffi_enum_names.rs"), out)
        .expect("failed to write pyvex_ffi_enum_names.rs into OUT_DIR");
}

/// Parse a single C enum block (the one whose first enumerator is `base`, which
/// must carry an explicit `=0xNNN` initializer) and emit
/// `pub fn <fn_name>(v: u32) -> Option<&'static str>`. Enumerators are numbered
/// sequentially from the base initializer, honoring any further `=0xNNN`
/// resets. libVEX's IR enums are dense (no gaps), so this mirrors the C
/// discriminants exactly.
#[cfg(feature = "libvex-ffi")]
fn emit_enum_name_table(src: &str, base: &str, fn_name: &str, out: &mut String) {
    // Locate the `enum {` block containing `base`, then scan to its `}`.
    let base_pos = src
        .find(base)
        .unwrap_or_else(|| panic!("enum base `{base}` not found in vendor/pyvex_ffi.h"));
    let block_end = src[base_pos..]
        .find('}')
        .map(|off| base_pos + off)
        .expect("unterminated enum block");
    let block = &src[base_pos..block_end];

    let mut value: i64 = -1;
    let mut arms = String::new();
    for raw in block.split(',') {
        // Strip C comments and whitespace; each entry is `Ident` or `Ident=0xNN`.
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let (name, init) = match token.split_once('=') {
            Some((n, v)) => (n.trim(), Some(v.trim())),
            None => (token, None),
        };
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.is_empty() {
            continue;
        }
        value = match init {
            Some(v) => parse_c_int(v).expect("failed to parse enum initializer"),
            None => value + 1,
        };
        // Skip the INVALID sentinels — the marshaller never sees them.
        if !name.ends_with("INVALID") {
            arms.push_str(&format!("        {value} => Some(\"{name}\"),\n"));
        }
    }

    out.push_str(&format!(
        "pub fn {fn_name}(v: u32) -> Option<&'static str> {{\n    match v as i64 {{\n{arms}        _ => None,\n    }}\n}}\n"
    ));
}

/// Parse a C integer literal (`0x1400` or decimal) into i64 for enum numbering.
#[cfg(feature = "libvex-ffi")]
fn parse_c_int(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<i64>().ok()
    }
}

/// No-op when the `libvex-ffi` feature is off so the unconditional call site in
/// `main()` stays simple. The call is already guarded by the runtime
/// `CARGO_FEATURE_LIBVEX_FFI` check, so this branch is never reached in a
/// default build; it exists only to keep the code compiling without the
/// `bindgen` build-dependency.
#[cfg(not(feature = "libvex-ffi"))]
fn generate_pyvex_ffi_bindings() {}

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
