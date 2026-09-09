//! Native __libc_start_main implementation.
//!
//! Treats every Rust-side invocation of __libc_start_main as a terminal
//! deadend. This is correct because the Python init pass
//! (`_step_python_to_main` in `rust_manager.py`) always runs the init chain
//! (`run` / `inside_init` / `after_init`) in Python before the Rust engine
//! takes over — by then the callstack only has the `after_main` continuation
//! on top, whose Python body is `self.exit(0)`. So any address mapped to
//! __libc_start_main reached during Rust exploration is `after_main`, and
//! deadending matches the Python semantics exactly.
//!
//! Eliminates ~47 Python callbacks/run on ais3_crackme (each ~1.3ms) and
//! any other bench that completes main multiple times.
//!
//! If the Python init was somehow bypassed and the entry-point case (`run`)
//! were hit here, the state would be lost. The init cache path makes that
//! unreachable in practice, but the conservative fallback for any future
//! flow that needs the init side-effects is to disable this native via
//! `set_python_override("__libc_start_main")`.

crate::declare_proc! {
    /// Native `__libc_start_main` implementation.
    ///
    /// ```c
    /// int __libc_start_main(int (*main)(int, char **, char **), int argc,
    ///                       char **argv, void (*init)(void),
    ///                       void (*fini)(void), void (*rtld_fini)(void),
    ///                       void *stack_end);
    /// ```
    ///
    /// The five declared args mirror Python angr's
    /// `run(self, main, argc, argv, init, fini)` signature. All are declared
    /// `bv` and ignored — the body deadends unconditionally, so a symbolic
    /// argument must not force a Python fallback the way `concrete` would.
    name = "__libc_start_main",
    struct = NativeLibcStartMain,
    args = [_main: bv, _argc: bv, _argv: bv, _init: bv, _fini: bv],
    no_return = true,
    call |_state| {
        // after_main does self.exit(0); native equivalent is just deadending.
        Ok(None)
    }
}

test_submod!("libc_start_main_tests.rs" => tests);
