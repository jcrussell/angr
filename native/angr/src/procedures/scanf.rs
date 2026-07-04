//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. This eliminates
//! Python callback overhead for common CTF patterns using scanf for input.
//!
//! Supported specifiers: %d, %i, %u, %x, %o, %s, %c, %[...] scanset,
//! %ld, %lld, %lu, %lx, %%
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

/// Length of a scanset body following `%[`.
///
/// `start` is the index of the first byte after the `[`. Returns the number of
/// bytes to advance from `start` to land just past the closing `]`, or `None`
/// if the set is unterminated. Handles the two POSIX/glibc literal-`]` cases:
/// a `]` directly after `[` or after `[^` is a member of the set, not the
/// closing bracket.
fn scanset_body_len(fmt: &[u8], start: usize) -> Option<usize> {
    let mut j = start;
    if j < fmt.len() && fmt[j] == b'^' {
        j += 1;
    }
    // A ']' in the first position is a literal set member, not the terminator.
    if j < fmt.len() && fmt[j] == b']' {
        j += 1;
    }
    while j < fmt.len() {
        if fmt[j] == b']' {
            return Some(j + 1 - start);
        }
        j += 1;
    }
    None
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
        } else {
            u8::from(modifier.is_64bit())
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
                // Scanset %[...] / %[^...]: matches a run of characters from
                // (or not in) the bracketed set. Skip past the set body so the
                // format cursor lands after the closing ']'. The set contents
                // do not constrain the minted bytes — consistent with how %s
                // mints fully unconstrained symbolic bytes here (see NativeSscanf
                // docs). A malformed (unterminated) set falls back to Python.
                let set_adv = scanset_body_len(fmt, i).ok_or_else(|| {
                    ProcedureError::Other("scanf %[...]: unterminated set".to_string())
                })?;
                i += set_adv;
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
            b'n' => {
                // %n stores the count of chars consumed so far. Deliberately
                // NOT implemented natively: deferring to Python is the faithful
                // behavior. Python's format_parser.py::FormatString.interpret
                // raises SimProcedureError on %n in the addr-based (sscanf-from-
                // memory) path, and treats it as a numeric read in the SimPackets
                // (stdin/file) path. A native write of the count would diverge
                // from both. The fallback reproduces Python exactly for free.
                // See bd memory `format-n-no-native-parity`.
                return Err(ProcedureError::Other(
                    "scanf %n not supported natively".to_string(),
                ));
            }
            _ => {
                // Unknown specifier — fall back to Python. Includes the float
                // specifiers %f/%e/%g: Python's format_parser.py::FormatString
                // .interpret raises SimProcedureError on them, so a native
                // symbolic-float read would diverge. Faithful behavior is to
                // defer. See bd memory `format-float-no-native-parity`.
                return Err(ProcedureError::Other(format!(
                    "scanf: unsupported specifier '%{}'",
                    spec as char
                )));
            }
        }
    }

    Ok(specs)
}

/// Core scanf implementation shared by scanf, __isoc99_scanf, sscanf and the
/// fscanf family.
///
/// `fmt_addr`: address of the format string in memory
/// `ptr_args`: slice of pointer arguments (one per non-suppressed conversion)
/// `source`: symbol-name prefix identifying the input source (`"stdin"` for
///   scanf/sscanf, `"file"` for fscanf on a non-stdin fd). Purely a label.
/// `record_stdin`: when true, each minted symbol is also recorded via
///   `record_stdin_symbol` so it surfaces in `posix.dumps(0)`. Only correct
///   when the input genuinely is stdin (fd 0); fscanf on a real file passes
///   false so file reads do not pollute the stdin reconstruction.
fn do_scanf(
    state: &mut RustSimState,
    fmt_addr: u64,
    ptr_args: &[RustBV],
    source: &str,
    record_stdin: bool,
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

        let ptr = extract_concrete_arg(&ptr_args[arg_idx], &format!("scanf arg {arg_idx}"))?;
        arg_idx += 1;

        if spec.is_string {
            // %s: create symbolic bytes + NUL terminator
            let str_len = spec.max_str_len;
            let names: Vec<String> = (0..str_len)
                .map(|j| format!("{source}_scanf_{scan_id}_s{spec_idx}_{j}"))
                .collect();

            let sym_bytes: Vec<RustBV> = {
                let ctx = state.solver().borrow();
                names
                    .iter()
                    .map(|name| RustBV::symbolic(&ctx, name, 8))
                    .collect()
            };

            if record_stdin {
                for name in &names {
                    state.record_stdin_symbol(name.clone(), 8);
                }
            }

            for (j, sym_byte) in sym_bytes.into_iter().enumerate() {
                state.memory_store(ptr.wrapping_add(j as u64), sym_byte)?;
            }

            // NUL terminator
            state.memory_store(ptr.wrapping_add(str_len), RustBV::concrete(0, 8))?;
        } else {
            // Numeric or char: create one symbolic BVS of appropriate width
            let name = format!("{source}_scanf_{scan_id}_{spec_idx}");
            let sym_val = {
                let ctx = state.solver().borrow();
                RustBV::symbolic(&ctx, &name, spec.bits)
            };

            if record_stdin {
                state.record_stdin_symbol(name, spec.bits);
            }

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
        do_scanf(state, fmt_addr, &args[1..], "stdin", true)
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
        do_scanf(state, fmt_addr, &args[1..], "stdin", true)
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
        do_scanf(state, fmt_addr, &args[2..], "stdin", true)
    }
}

/// Core fscanf implementation shared by fscanf and __isoc99_fscanf.
///
/// The stream variant of [`NativeScanf`]: resolves `stream->_fileno` and routes
/// through the shared [`do_scanf`] core, mirroring how `NativeFprintf` extends
/// `NativePrintf`. Like the existing scanf/sscanf procs, the parsed values are
/// minted as fresh unconstrained symbolic BVs (the file *content* is not parsed
/// — same simplification `NativeSscanf` documents). A closed/negative fd
/// returns -1, matching Python `fscanf` (`simfd is None`). Symbols are recorded
/// for `posix.dumps(0)` only when the FILE wraps fd 0 (e.g. `fscanf(stdin,...)`).
fn do_fscanf(
    state: &mut RustSimState,
    file_ptr: u64,
    fmt_addr: u64,
    ptr_args: &[RustBV],
) -> Result<Option<RustBV>, ProcedureError> {
    let fd = crate::procedures::fileops::read_fileno(state, file_ptr)?;
    if fd < 0 {
        return Ok(Some(RustBV::concrete(
            (-1i64 as u64) as u128,
            state.arch().bits(),
        )));
    }
    let (source, record_stdin) = if fd == 0 {
        ("stdin", true)
    } else {
        ("file", false)
    };
    do_scanf(state, fmt_addr, ptr_args, source, record_stdin)
}

/// Native fscanf implementation.
///
/// ```c
/// int fscanf(FILE *stream, const char *format, ...);
/// ```
pub struct NativeFscanf;

impl NativeSimProcedure for NativeFscanf {
    fn name(&self) -> &'static str {
        "fscanf"
    }

    fn num_args(&self) -> usize {
        8 // stream + format + up to 6 pointer args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;
        do_fscanf(state, file_ptr, fmt_addr, &args[2..])
    }
}

/// Native __isoc99_fscanf implementation (alias for fscanf).
///
/// Many binaries compiled with newer glibc use __isoc99_fscanf instead of fscanf.
pub struct NativeIsoc99Fscanf;

impl NativeSimProcedure for NativeIsoc99Fscanf {
    fn name(&self) -> &'static str {
        "__isoc99_fscanf"
    }

    fn num_args(&self) -> usize {
        8
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let file_ptr = extract_concrete_arg(&args[0], "stream")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;
        do_fscanf(state, file_ptr, fmt_addr, &args[2..])
    }
}

#[cfg(test)]
#[path = "scanf_tests.rs"]
mod scanf_tests;
