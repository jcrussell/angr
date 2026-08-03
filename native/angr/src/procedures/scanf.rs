//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. This eliminates
//! Python callback overhead for common CTF patterns using scanf for input.
//!
//! Supported specifiers: %d, %i, %u, %x, %o, %s, %c, %[...] scanset,
//! %ld, %lld, %lu, %lx, %%
//! Falls back to Python for symbolic format strings or pointer arguments.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** the format
//! string and every pointer argument are guest data, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; unparseable input
//! returns a `ProcedureError` that falls back to Python instead. The one
//! statement-level `#[allow]` below is the `mint_stdin_bytes` one-name
//! contract, not an input check.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::format_common::{MAX_FORMAT_LEN, parse_length_modifier, parse_width_digits};
use super::stdin_common::{mint_stdin_bytes, stdin_seed_unconsumed};
use super::strings::scan_concrete_bounded;
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_SCANF_STR_LEN: u64 = 256;

/// Read a null-terminated concrete format string from memory.
///
/// Propagates memory-fault errors (Unmapped / Permission / OutOfBounds /
/// SymbolicAddress) rather than silently truncating, matching the rest of the
/// string-scanning family via [`scan_concrete_bounded`]. A real fault must
/// surface as `Err` so the caller falls back to Python instead of committing to
/// store symbolic values based on a corrupted (truncated) format string
/// (angr-myzjx.1). Hitting `MAX_FORMAT_LEN` without a null is not an error — a
/// plausibly long format string simply stops there.
fn read_format_string(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    scan_concrete_bounded(state, addr, MAX_FORMAT_LEN, "format string byte").map(|(buf, _)| buf)
}

/// Parsed scanf format specifier.
struct ScanfSpec {
    /// Width of the value to create in bits.
    bits: u32,
    /// Whether this is a string specifier (%s).
    is_string: bool,
    /// Whether this is the single-character specifier (%c). Unlike the numeric
    /// conversions it maps one input byte to one stored byte, so it can consume
    /// a harness-seeded stdin byte directly (angr-ggb66).
    is_char: bool,
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
///
/// `arch_bits` sizes the `l`/`z`/`t` modifiers, which are `long`-width and so
/// 32-bit on ILP32 targets — see `LengthModifier::int_conv_bits`. Each spec's
/// `bits` becomes the width of the value stored at the caller's pointer, so
/// over-sizing it writes past the guest object.
fn parse_scanf_format(fmt: &[u8], arch_bits: u32) -> Result<Vec<ScanfSpec>, ProcedureError> {
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

        // Parse field width. Clamp against MAX_SCANF_STR_LEN here, before the
        // value is used as a Vec/collect bound: parse_width_digits only
        // saturates to usize::MAX on overflow, which does not stop a short
        // digit run like "%9999999999999999999s" from encoding a near-MAX
        // width. Left unclamped, `(0..field_width).map(...).collect()` in
        // do_scanf hits an allocator capacity-overflow panic before any
        // allocation is attempted (angr-mi56k).
        let (width_val, w_adv) = parse_width_digits(fmt, i);
        let has_width = w_adv > 0;
        let field_width = (width_val as u64).min(MAX_SCANF_STR_LEN);
        i += w_adv;

        // Parse length modifier. `h`/`hh` narrow the destination to 16/8 bits
        // and `l`/`z`/`t` widen it to `long` width (matching Python's
        // format_parser.py int_len_mod); `z`/`j`/`t` are honoured at all
        // because glibc accepts them in scanf.
        let (modifier, m_adv) = parse_length_modifier(fmt, i);
        i += m_adv;

        if i >= fmt.len() {
            break;
        }

        let spec = fmt[i];
        i += 1;

        match spec {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' => {
                let bits = modifier.int_conv_bits(arch_bits);
                specs.push(ScanfSpec {
                    bits,
                    is_string: false,
                    is_char: false,
                    max_str_len: 0,
                    suppress,
                });
            }
            b'c' => {
                specs.push(ScanfSpec {
                    bits: 8,
                    is_string: false,
                    is_char: true,
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
                    is_char: false,
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
                    is_char: false,
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
                return Err(ProcedureError::Other(
                    "scanf %n not supported natively".to_string(),
                ));
            }
            _ => {
                // Unknown specifier — fall back to Python. Includes the float
                // specifiers %f/%e/%g: Python's format_parser.py::FormatString
                // .interpret raises SimProcedureError on them, so a native
                // symbolic-float read would diverge. Faithful behavior is to defer.
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
    let specs = parse_scanf_format(&fmt, state.arch().bits())?;

    // A numeric conversion models a decimal/hex *parse*: Python's
    // format_parser.py::FormatString.interpret reads `max_digits` bytes off the
    // stream and constrains each one to the ASCII rendering of the stored value.
    // Natively we only mint one free BVS of the value's width, which neither
    // constrains it to nor consumes the harness-seeded bytes — so a seeded run
    // would get an unconstrained value AND leave the seed for the next
    // conversion to re-read from offset 0. Defer the whole call to Python while
    // fd 0 still has unread seed; unseeded runs keep the native fast path
    // (angr-ggb66). Checked before any store so the fallback sees a pristine
    // state.
    if record_stdin
        && specs.iter().any(|s| !s.is_string && !s.is_char)
        && stdin_seed_unconsumed(state)
    {
        return Err(ProcedureError::Other(
            "scanf: numeric conversion over harness-seeded stdin".to_string(),
        ));
    }

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

            // On stdin, `mint_stdin_bytes` consumes any harness-seeded fd-0
            // bytes (binding the leaves to them) and records only the unseeded
            // ones. sscanf/fscanf read no stdin, so they just mint (angr-ptf54).
            let sym_bytes: Vec<RustBV> = if record_stdin {
                mint_stdin_bytes(state, &names)
            } else {
                let ctx = state.solver().borrow();
                names
                    .iter()
                    .map(|name| RustBV::symbolic(&ctx, name, 8))
                    .collect()
            };

            for (j, sym_byte) in sym_bytes.into_iter().enumerate() {
                state.memory_store(ptr.wrapping_add(j as u64), sym_byte)?;
            }

            // NUL terminator
            state.memory_store(ptr.wrapping_add(str_len), RustBV::concrete(0, 8))?;
        } else if spec.is_char && record_stdin {
            // %c reads exactly one byte off the stream — a byte-for-byte
            // mapping, so it consumes the harness seed like %s does (angr-ggb66).
            let name = format!("{source}_scanf_{scan_id}_{spec_idx}");
            #[allow(
                clippy::expect_used,
                reason = "`mint_stdin_bytes` returns exactly one `RustBV` per requested name (its own doc contract, and it builds the vec by mapping over `names`), and the call passes a single-element slice via `slice::from_ref`, so the vec always holds one element"
            )]
            let sym_byte = mint_stdin_bytes(state, std::slice::from_ref(&name))
                .into_iter()
                .next()
                .expect("mint_stdin_bytes returns one BV per name");
            state.memory_store(ptr, sym_byte)?;
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
///
/// Aliased to `__isoc99_scanf`, the name modern glibc emits for a `scanf`
/// call: the two differ only in how glibc handles legacy `%a` allocation
/// semantics, which `do_scanf` does not implement either way, so one impl
/// serves both dispatch names (DRY — same mechanism as printf/vprintf).
pub(crate) struct NativeScanf;

impl NativeSimProcedure for NativeScanf {
    fn name(&self) -> &'static str {
        "scanf"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["__isoc99_scanf"]
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

/// Native sscanf implementation.
///
/// ```c
/// int sscanf(const char *str, const char *format, ...);
/// ```
///
/// Unlike scanf, sscanf reads from a concrete in-memory buffer whose contents
/// must be *parsed* to constrain the stored values: Python's
/// `format_parser.py::FormatString.interpret` (addr path) reads the source
/// region and stores the CONSTRAINED value (e.g. `sscanf("42","%d",&x)` binds
/// `x==42`). Minting fresh unconstrained BVs — as `do_scanf` does for the
/// stream sources — would make impossible paths feasible (`x==1337` reachable
/// after parsing `"42"`) and copy garbage for `%s`. Rather than reimplement
/// scanf's full matching engine natively, we defer the whole call to Python,
/// which parses the source faithfully. See angr-8onrp.
pub(crate) struct NativeSscanf;

impl NativeSimProcedure for NativeSscanf {
    fn name(&self) -> &'static str {
        "sscanf"
    }

    fn num_args(&self) -> usize {
        8 // str + format + up to 6 pointer args
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Defer to Python: only it parses the concrete source region and
        // constrains the outputs. A native free-BVS mint would explore
        // impossible paths. (angr-8onrp)
        Err(ProcedureError::Other(
            "sscanf: concrete-source parse deferred to Python".to_string(),
        ))
    }
}

/// Core fscanf implementation shared by fscanf and __isoc99_fscanf.
///
/// The stream variant of [`NativeScanf`]: resolves `stream->_fileno` and routes
/// through the shared [`do_scanf`] core, mirroring how `NativeFprintf` extends
/// `NativePrintf`. Like the stdin `scanf` proc, the parsed values are minted as
/// fresh unconstrained symbolic BVs — the file *content* is not parsed. (This
/// is faithful for a stream source whose bytes are symbolic; the concrete
/// in-memory buffer case is `sscanf`, which defers to Python — see
/// [`NativeSscanf`].) A closed/negative fd
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
///
/// Aliased to `__isoc99_fscanf` for the same reason [`NativeScanf`] aliases
/// `__isoc99_scanf`: modern glibc emits the `__isoc99_` name, and the two
/// differ only in legacy `%a` handling that `do_scanf` does not implement.
pub(crate) struct NativeFscanf;

impl NativeSimProcedure for NativeFscanf {
    fn name(&self) -> &'static str {
        "fscanf"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["__isoc99_fscanf"]
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

#[cfg(test)]
#[path = "scanf_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod scanf_tests;
