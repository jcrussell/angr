use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only configure Z3 paths when the z3 feature is enabled
    if env::var("CARGO_FEATURE_Z3").is_err() {
        return;
    }

    // If Z3_SYS_Z3_HEADER is already set (e.g., in .cargo/config.toml), don't override
    if env::var("Z3_SYS_Z3_HEADER").is_ok() {
        return;
    }

    // Try to discover Z3 from the Python environment
    if let Some(z3_dir) = find_z3_from_python() {
        let header = z3_dir.join("include").join("z3.h");
        let lib_dir = z3_dir.join("lib");

        if header.exists() {
            println!("cargo:rustc-env=Z3_SYS_Z3_HEADER={}", header.display());
        }
        if lib_dir.exists() {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_dir.display()
            );
        }
    }
}

fn find_z3_from_python() -> Option<PathBuf> {
    // Ask Python where the z3 package is installed
    let output = Command::new("python3")
        .args(["-c", "import z3; import os; print(os.path.dirname(z3.__file__))"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let path_str = String::from_utf8(output.stdout).ok()?.trim().to_string();
    let path = PathBuf::from(path_str);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}
