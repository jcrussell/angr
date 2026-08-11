//! Native `getopt(3)` short-option engine.
//!
//! Faithful Rust port of the short-option path of
//! `angr/procedures/libc/getopt.py::_GetOptBase._getopt`. Registered only for
//! the 3-argument `getopt` (no long-option variants): `getopt_long` /
//! `getopt_long_only` stay on the Python proc, which owns the option-table
//! walk and ambiguous-abbreviation handling.
//!
//! Contract — **match Python byte-for-byte, or defer** (return
//! `ProcedureError` → Python fallback). We defer whenever any required input
//! is symbolic (argc, argv, an argv element pointer, the optstring, or any
//! scanned byte), mirroring Python's `_unconstrained()` branch which produces a
//! fresh symbolic return rather than committing to a concrete scan. Deferring
//! lets the Python proc produce that unconstrained value itself.
//!
//! Cross-call cursor state (`optind` / `optchar`) lives on `RustSimState`
//! (bhk0a.3 cursor fields). The loader-resolved guest addresses of the
//! `optind` / `optarg` / `optopt` extern globals are pushed from Python into
//! `RustSimState::getopt_extern` (bhk0a.1); we write the updated values back
//! there so the guest program reads them like real getopt would. A `None`
//! address (symbol absent) just skips that store — exactly as Python's
//! `_store_int` / `_store_ptr` skip when `_global_addr` returns `None`.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** `argv`, the
//! optstring and the saved cursor are all guest-controlled, so this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]` and every
//! unresolvable value returns a `ProcedureError` that falls back to the Python
//! proc. There are no `unwrap`/`expect` sites left: the optstring lookup binds
//! its hit in the match arm that proves it (see the `spec` match), and the
//! cursor read already used a checked `arg.get(..)` (angr-qwyti.19).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::strings::{MAX_STRING_SCAN, scan_concrete_until_null};
use super::{ProcedureError, arch_word};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const NO_ARGUMENT: u8 = 0;
const REQUIRED_ARGUMENT: u8 = 1;
const OPTIONAL_ARGUMENT: u8 = 2;

const DASH: u8 = b'-';
const COLON: u8 = b':';

/// Parse an optstring into `(char -> has_arg, leading_colon)` per POSIX/glibc
/// syntax. Leading `+`/`-` mode chars (ordering, which we do not model) are
/// skipped; a leading `:` selects the missing-arg `':'` return.
fn parse_optstring(s: &[u8]) -> (std::collections::HashMap<u8, u8>, bool) {
    let mut opts = std::collections::HashMap::new();
    let mut i = 0;
    while i < s.len() && (s[i] == b'+' || s[i] == DASH) {
        i += 1;
    }
    let leading_colon = i < s.len() && s[i] == COLON;
    if leading_colon {
        i += 1;
    }
    while i < s.len() {
        let c = s[i];
        i += 1;
        let mut nargs = NO_ARGUMENT;
        if i < s.len() && s[i] == COLON {
            nargs = REQUIRED_ARGUMENT;
            i += 1;
            if i < s.len() && s[i] == COLON {
                nargs = OPTIONAL_ARGUMENT;
                i += 1;
            }
        }
        opts.insert(c, nargs);
    }
    (opts, leading_colon)
}

/// Load a pointer-sized word; `Ok(None)` if symbolic (caller defers).
fn eval_ptr(state: &RustSimState, addr: u64) -> Result<Option<u64>, ProcedureError> {
    let ps = state.arch().bytes();
    Ok(state.memory_load(addr, ps)?.as_u64())
}

/// Store a 32-bit int at the extern global `addr` if present (else no-op),
/// mirroring Python's `_store_int`.
fn store_int(
    state: &mut RustSimState,
    addr: Option<u64>,
    value: u32,
) -> Result<(), ProcedureError> {
    if let Some(a) = addr {
        state.memory_store(a, RustBV::concrete(value as u128, 32))?;
    }
    Ok(())
}

/// Store a pointer-sized word at the extern global `addr` if present,
/// mirroring Python's `_store_ptr`.
fn store_ptr(
    state: &mut RustSimState,
    addr: Option<u64>,
    value: u64,
) -> Result<(), ProcedureError> {
    if let Some(a) = addr {
        state.memory_store(a, arch_word(state, value))?;
    }
    Ok(())
}

/// Resolve `(optind, optchar)`, honouring a guest reset of `optind` exactly as
/// Python's `_load_cursor`.
fn load_cursor(
    state: &RustSimState,
    optind_addr: Option<u64>,
) -> Result<(u32, u32), ProcedureError> {
    let (plugin_optind, plugin_optchar) = state.getopt_cursor();
    if let Some(addr) = optind_addr {
        let word = state.memory_load(addr, 4)?;
        if let Some(guest) = word.as_u64() {
            let guest = guest as u32;
            if guest == 0 {
                return Ok((1, 0)); // glibc: optind==0 requests a full reset
            }
            // guest >= 1: honour a guest rescan (optind reset to a smaller index)
            let oc = if guest == plugin_optind {
                plugin_optchar
            } else {
                0
            };
            return Ok((guest, oc));
        }
    }
    Ok((plugin_optind, plugin_optchar))
}

/// Persist `(optind, optchar)` to the cursor fields and write `optind` back to
/// guest memory, mirroring Python's `_save_cursor`.
fn save_cursor(
    state: &mut RustSimState,
    optind_addr: Option<u64>,
    optind: u32,
    optchar: u32,
) -> Result<(), ProcedureError> {
    state.set_getopt_cursor(optind, optchar);
    store_int(state, optind_addr, optind)
}

/// `int getopt(int argc, char *const argv[], const char *optstring)`.
#[inline]
fn ret(value: u32) -> Result<Option<RustBV>, ProcedureError> {
    Ok(Some(RustBV::concrete(value as u128, 32)))
}

crate::declare_proc! {
    /// getopt: concrete-argv short-option engine. Defers to Python for any
    /// symbolic input (matching the Python proc's unconstrained branch).
    ///
    /// ```c
    /// int getopt(int argc, char *const argv[], const char *optstring);
    /// ```
    name = "getopt",
    struct = NativeGetopt,
    args = [argc: concrete, argv_ptr: concrete, optstring_ptr: concrete],
    call |state| {
        let ps = state.arch().bytes() as u64;
        let extern_addrs = state.getopt_extern();
        let optind_addr = extern_addrs.optind;
        let optarg_addr = extern_addrs.optarg;
        let optopt_addr = extern_addrs.optopt;

        let argc = argc as u32;

        let optstring = scan_concrete_until_null(state, optstring_ptr, MAX_STRING_SCAN, "optstring")?;
        let (opts, leading_colon) = parse_optstring(&optstring);

        let (mut optind, mut optchar) = load_cursor(state, optind_addr)?;

        if optind >= argc {
            save_cursor(state, optind_addr, optind, 0)?;
            store_ptr(state, optarg_addr, 0)?;
            return ret(0xFFFF_FFFF); // -1
        }

        // `argv_ptr` is a guest-controlled address, so every address computation
        // below uses `wrapping_add` per the convention documented in memcpy.rs:
        // `[profile.release]` disables overflow checks, so a plain `+` panics
        // under the release-checked test profile and silently wraps in the
        // shipped `.so` — a wrong address rather than a loud failure.
        // (`optind`/`optchar` are u32, so their `as u64` scaling cannot itself
        // overflow: u32::MAX * 8 fits in u64.)
        let elem_ptr = match eval_ptr(state, argv_ptr.wrapping_add(optind as u64 * ps))? {
            Some(p) => p,
            None => return Err(ProcedureError::SymbolicArgument("argv element".into())),
        };
        if elem_ptr == 0 {
            save_cursor(state, optind_addr, optind, 0)?;
            return ret(0xFFFF_FFFF);
        }
        let arg = scan_concrete_until_null(state, elem_ptr, MAX_STRING_SCAN, "argv string")?;

        if optchar == 0 {
            // non-option operand "-" or a non-dash element -> stop (non-permuting)
            if arg.is_empty() || arg[0] != DASH || arg == [DASH] {
                save_cursor(state, optind_addr, optind, 0)?;
                return ret(0xFFFF_FFFF);
            }
            if arg == [DASH, DASH] {
                save_cursor(state, optind_addr, optind + 1, 0)?;
                return ret(0xFFFF_FFFF);
            }
            // long-option handling lives in the Python proc; the native proc is
            // registered only for plain `getopt`, so longopts is always absent.
            optchar = 1;
        }

        // Defense-in-depth (angr-qwyti.19): `optchar` is engine cursor state
        // restored from a prior getopt call. If the guest mutated this argv
        // element in its own memory to be shorter since then, the saved cursor
        // can exceed the freshly-scanned `arg` length. Use a checked accessor
        // and defer to the Python getopt proc rather than panic-indexing.
        let Some(&c) = arg.get(optchar as usize) else {
            return Err(ProcedureError::Other(
                "getopt cursor past end of argv element (shrunk argv)".to_string(),
            ));
        };
        // `c` is a guest byte, so the optstring lookup genuinely can miss —
        // binding the hit in the match arm keeps the unknown-option path and
        // the recognized-option path in one place, with no second lookup and no
        // `unwrap` on what the miss branch just ruled out.
        let spec = match opts.get(&c).copied() {
            Some(spec) if c != COLON => spec,
            // Unknown option (or a bare `:`, which is never an option itself):
            // report it through `optopt` and return '?'.
            _ => {
                store_int(state, optopt_addr, c as u32)?;
                store_ptr(state, optarg_addr, 0)?;
                optchar += 1;
                if optchar as usize >= arg.len() {
                    optind += 1;
                    optchar = 0;
                }
                save_cursor(state, optind_addr, optind, optchar)?;
                return ret(b'?' as u32);
            }
        };

        if spec == NO_ARGUMENT {
            store_ptr(state, optarg_addr, 0)?;
            optchar += 1;
            if optchar as usize >= arg.len() {
                optind += 1;
                optchar = 0;
            }
            save_cursor(state, optind_addr, optind, optchar)?;
            return ret(c as u32);
        }

        // option takes an argument (required or optional)
        if (optchar as usize) + 1 < arg.len() {
            store_ptr(state, optarg_addr, elem_ptr.wrapping_add(optchar as u64 + 1))?;
            save_cursor(state, optind_addr, optind + 1, 0)?;
            return ret(c as u32);
        }
        if spec == OPTIONAL_ARGUMENT {
            store_ptr(state, optarg_addr, 0)?;
            save_cursor(state, optind_addr, optind + 1, 0)?;
            return ret(c as u32);
        }
        // required argument from the next argv element
        if optind + 1 < argc {
            let next_ptr = match eval_ptr(state, argv_ptr.wrapping_add((optind as u64 + 1) * ps))? {
                Some(p) => p,
                None => return Err(ProcedureError::SymbolicArgument("argv next".into())),
            };
            store_ptr(state, optarg_addr, next_ptr)?;
            save_cursor(state, optind_addr, optind + 2, 0)?;
            return ret(c as u32);
        }
        // missing required argument
        store_int(state, optopt_addr, c as u32)?;
        store_ptr(state, optarg_addr, 0)?;
        save_cursor(state, optind_addr, optind + 1, 0)?;
        ret(if leading_colon { COLON as u32 } else { b'?' as u32 })
    }
}

test_submod!("getopt_tests.rs" => getopt_tests);
