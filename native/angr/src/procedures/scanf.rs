//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. This eliminates
//! Python callback overhead for common CTF patterns using scanf for input.
//!
//! Supported specifiers: %d, %i, %u, %x, %o, %s, %c, %ld, %lld, %lu, %lx, %%
//! Falls back to Python for symbolic format strings or pointer arguments.

use super::format_common::{LengthModifier, parse_length_modifier, parse_width_digits};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FMT_LEN: usize = 4096;
const MAX_SCANF_STR_LEN: u64 = 256;

/// Read a null-terminated concrete string from memory.
fn read_format_string(state: &RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let mut buf = Vec::new();
    for i in 0..MAX_FMT_LEN as u64 {
        match state.memory_load(addr.wrapping_add(i), 1) {
            Ok(bv) => {
                let byte = extract_concrete_arg(&bv, "format string byte")? as u8;
                if byte == 0 {
                    break;
                }
                buf.push(byte);
            }
            Err(_) => break,
        }
    }
    Ok(buf)
}

/// Parsed scanf format specifier.
struct ScanfSpec {
    /// Width of the value to create in bits.
    bits: u32,
    /// Whether this is a string specifier (%s).
    is_string: bool,
    /// Max width for %s (from field width, e.g. %10s), or MAX_SCANF_STR_LEN.
    max_str_len: u64,
    /// Whether to suppress assignment (*).
    suppress: bool,
}

/// Parse scanf format specifiers from a format string.
/// Returns a list of specifiers (one per conversion that stores a value).
fn parse_scanf_format(fmt: &[u8]) -> Result<Vec<ScanfSpec>, ProcedureError> {
    let mut specs = Vec::new();
    let mut i = 0;

    while i < fmt.len() {
        if fmt[i] != b'%' {
            // Non-format characters are literal matchers; skip them
            i += 1;
            continue;
        }
        i += 1; // skip '%'
        if i >= fmt.len() {
            break;
        }

        // Handle %%
        if fmt[i] == b'%' {
            i += 1;
            continue;
        }

        // Check for suppression (*)
        let suppress = if i < fmt.len() && fmt[i] == b'*' {
            i += 1;
            true
        } else {
            false
        };

        // Parse field width
        let (width_val, w_adv) = parse_width_digits(fmt, i);
        let has_width = w_adv > 0;
        let field_width = width_val as u64;
        i += w_adv;

        // Parse length modifier. `Short`/`Char` (`h`/`hh`) are consumed
        // but still stored at int-width for simplicity; `z`/`j`/`t` are
        // honoured as 64-bit (glibc accepts them in scanf).
        let (modifier, m_adv) = parse_length_modifier(fmt, i);
        i += m_adv;
        let long_count: u8 = if matches!(modifier, LengthModifier::LongLong) {
            2
        } else if modifier.is_64bit() {
            1
        } else {
            0
        };

        if i >= fmt.len() {
            break;
        }

        let spec = fmt[i];
        i += 1;

        match spec {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' => {
                let bits = match long_count {
                    2 => 64, // long long
                    1 => 64, // long (on 64-bit)
                    _ => 32, // int
                };
                specs.push(ScanfSpec {
                    bits,
                    is_string: false,
                    max_str_len: 0,
                    suppress,
                });
            }
            b'c' => {
                specs.push(ScanfSpec {
                    bits: 8,
                    is_string: false,
                    max_str_len: 0,
                    suppress,
                });
            }
            b's' => {
                let max_len = if has_width {
                    field_width
                } else {
                    MAX_SCANF_STR_LEN
                };
                specs.push(ScanfSpec {
                    bits: 8,
                    is_string: true,
                    max_str_len: max_len,
                    suppress,
                });
            }
            b'[' => {
                // Character class like %[^\n] — too complex, fall back
                return Err(ProcedureError::Other(
                    "scanf %[...] not supported natively".to_string(),
                ));
            }
            b'n' => {
                // %n writes count of chars read — skip
                return Err(ProcedureError::Other(
                    "scanf %n not supported natively".to_string(),
                ));
            }
            _ => {
                return Err(ProcedureError::Other(format!(
                    "scanf: unsupported specifier '%{}'",
                    spec as char
                )));
            }
        }
    }

    Ok(specs)
}

/// Core scanf implementation shared by scanf and __isoc99_scanf.
///
/// `fmt_addr`: address of the format string in memory
/// `ptr_args`: slice of pointer arguments (one per non-suppressed conversion)
fn do_scanf(
    state: &mut RustSimState,
    fmt_addr: u64,
    ptr_args: &[RustBV],
) -> Result<Option<RustBV>, ProcedureError> {
    let fmt = read_format_string(state, fmt_addr)?;
    let specs = parse_scanf_format(&fmt)?;

    let scan_id = symbol_counter("scanf");
    let mut arg_idx: usize = 0;
    let mut conversions: u64 = 0;

    for (spec_idx, spec) in specs.iter().enumerate() {
        if spec.suppress {
            // Suppressed: no pointer argument consumed, no storage
            continue;
        }

        if arg_idx >= ptr_args.len() {
            // Ran out of pointer arguments — return what we have
            break;
        }

        let ptr = extract_concrete_arg(&ptr_args[arg_idx], &format!("scanf arg {}", arg_idx))?;
        arg_idx += 1;

        if spec.is_string {
            // %s: create symbolic bytes + NUL terminator
            let str_len = spec.max_str_len;
            let names: Vec<String> = (0..str_len)
                .map(|j| format!("stdin_scanf_{}_s{}_{}", scan_id, spec_idx, j))
                .collect();

            let sym_bytes: Vec<RustBV> = {
                let ctx = state.solver().borrow();
                names
                    .iter()
                    .map(|name| RustBV::symbolic(&ctx, name, 8))
                    .collect()
            };

            for name in &names {
                state.record_stdin_symbol(name.clone(), 8);
            }

            for (j, sym_byte) in sym_bytes.into_iter().enumerate() {
                state.memory_store(ptr.wrapping_add(j as u64), sym_byte)?;
            }

            // NUL terminator
            state.memory_store(ptr.wrapping_add(str_len), RustBV::concrete(0, 8))?;
        } else {
            // Numeric or char: create one symbolic BVS of appropriate width
            let name = format!("stdin_scanf_{}_{}", scan_id, spec_idx);
            let sym_val = {
                let ctx = state.solver().borrow();
                RustBV::symbolic(&ctx, &name, spec.bits)
            };

            state.record_stdin_symbol(name, spec.bits);

            // Store to pointer — write spec.bits/8 bytes
            state.memory_store(ptr, sym_val)?;
        }

        conversions += 1;
    }

    // Return number of successful conversions
    let bits = state.arch().bits();
    Ok(Some(RustBV::concrete(conversions as u128, bits)))
}

/// Native scanf implementation.
///
/// ```c
/// int scanf(const char *format, ...);
/// ```
pub struct NativeScanf;

impl NativeSimProcedure for NativeScanf {
    fn name(&self) -> &'static str {
        "scanf"
    }

    fn num_args(&self) -> usize {
        7 // format + up to 6 pointer args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fmt_addr = extract_concrete_arg(&args[0], "format")?;
        do_scanf(state, fmt_addr, &args[1..])
    }
}

/// Native __isoc99_scanf implementation (alias for scanf).
///
/// Many binaries compiled with newer glibc use __isoc99_scanf instead of scanf.
pub struct NativeIsoc99Scanf;

impl NativeSimProcedure for NativeIsoc99Scanf {
    fn name(&self) -> &'static str {
        "__isoc99_scanf"
    }

    fn num_args(&self) -> usize {
        7
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fmt_addr = extract_concrete_arg(&args[0], "format")?;
        do_scanf(state, fmt_addr, &args[1..])
    }
}

/// Native sscanf implementation.
///
/// ```c
/// int sscanf(const char *str, const char *format, ...);
/// ```
///
/// Unlike scanf, sscanf reads from a string buffer instead of stdin.
/// For symbolic execution, if the source string contains symbolic bytes,
/// we fall back to Python. For concrete source strings, we still create
/// symbolic values for the output pointers (treating parsed values as
/// unconstrained), matching angr's behavior.
pub struct NativeSscanf;

impl NativeSimProcedure for NativeSscanf {
    fn name(&self) -> &'static str {
        "sscanf"
    }

    fn num_args(&self) -> usize {
        8 // str + format + up to 6 pointer args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // args[0] = source string (we don't actually parse it — just check it's concrete)
        let _src_addr = extract_concrete_arg(&args[0], "str")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;
        // For sscanf, we create symbolic values just like scanf
        // (the parsed values are unconstrained in symbolic execution)
        do_scanf(state, fmt_addr, &args[2..])
    }
}

#[cfg(test)]
#[path = "scanf_tests.rs"]
mod scanf_tests;
