//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. This eliminates
//! Python callback overhead for common CTF patterns using scanf for input.
//!
//! Supported specifiers: %d, %i, %u, %x, %o, %s, %c, %ld, %lld, %lu, %lx, %%
//! Falls back to Python for symbolic format strings or pointer arguments, and
//! for the specifiers Python models differently — see `parse_scanf_format`.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** the format
//! string and every pointer argument are guest data, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; unparseable input
//! returns a `ProcedureError` that falls back to Python instead. The one
//! statement-level `#[allow]` below is the `mint_stdin_bytes` one-name
//! contract, not an input check.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::arch_word;
use super::format_common::{parse_length_modifier, parse_width_digits, read_format_string};
use super::stdin_common::{fd0_is_dup2d_tracked_file, mint_stdin_bytes, stdin_seed_unconsumed};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_SCANF_STR_LEN: u64 = 256;

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
                // Scanset %[...] / %[^...]. Deliberately NOT implemented
                // natively: angr's Python
                // `format_parser.py::ScanfFormatParser.basic_spec` has no `[`
                // entry, and `_match_spec` only matches a nugget by prefix
                // against those, so `%[a-z]` never becomes a FormatSpecifier
                // there. Python's `extract_components` instead keeps the `%`
                // as a literal one-char component and matches the bracket
                // expression against the input stream as literal text,
                // consuming no pointer argument. A native implementation that
                // minted a symbolic string and consumed a pointer would
                // silently disagree with the reference engine on the stored
                // buffer, the return value, and the argument bookkeeping of
                // every conversion after it, so the faithful behavior is to
                // defer the whole call (angr-6cp06.7).
                return Err(ProcedureError::Other(
                    "scanf %[...] scanset not supported natively".to_string(),
                ));
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
        // A suppressed conversion (`%*s`) assigns nothing — it consumes no
        // pointer argument and stores nowhere — but the real scanf(3) still
        // *reads* the matched bytes off the stream. On stdin that read is the
        // only thing advancing the fd-0 cursor, so skipping it outright made a
        // later `%s`/`%c` in the same format re-read from the wrong offset
        // (angr-sqfj8.84). Off-stream sources (sscanf/fscanf on a real file,
        // i.e. `!record_stdin`) have no cursor here, so a suppressed spec is a
        // genuine no-op for them. Suppressed *numeric* specs need no special
        // handling either: the seeded-stdin check above ignores `suppress`, so
        // `%*d` over unread seed already defers the whole call to Python.
        if spec.suppress && !record_stdin {
            continue;
        }

        let ptr = if spec.suppress {
            None
        } else {
            if arg_idx >= ptr_args.len() {
                // Ran out of pointer arguments — return what we have
                break;
            }
            let ptr = extract_concrete_arg(&ptr_args[arg_idx], &format!("scanf arg {arg_idx}"))?;
            arg_idx += 1;
            Some(ptr)
        };

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

            // Suppressed (`ptr == None`): the bytes were still read off the
            // stream above, they just have nowhere to land.
            if let Some(ptr) = ptr {
                for (j, sym_byte) in sym_bytes.into_iter().enumerate() {
                    state.memory_store(ptr.wrapping_add(j as u64), sym_byte)?;
                }

                // NUL terminator
                state.memory_store(ptr.wrapping_add(str_len), RustBV::concrete(0, 8))?;
            }
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
            if let Some(ptr) = ptr {
                state.memory_store(ptr, sym_byte)?;
            }
        } else {
            // Numeric or char: create one symbolic BVS of appropriate width
            let name = format!("{source}_scanf_{scan_id}_{spec_idx}");
            let sym_val = {
                let ctx = state.solver().borrow();
                RustBV::symbolic(&ctx, &name, spec.bits)
            };

            // A suppressed numeric still consumes digits off the stream, so it
            // is recorded for the stdin reconstruction exactly like an assigned
            // one — only the store is skipped.
            if record_stdin {
                state.record_stdin_symbol(name, spec.bits);
            }

            // Store to pointer — write spec.bits/8 bytes
            if let Some(ptr) = ptr {
                state.memory_store(ptr, sym_val)?;
            }
        }

        // scanf(3) counts *assigned* conversions; a suppressed one is read but
        // never counted.
        if ptr.is_some() {
            conversions += 1;
        }
    }

    // Return number of successful conversions
    Ok(Some(arch_word(state, conversions)))
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
        // angr-qmrrp: a prior dup2(real_fd, 0) means fd 0 carries real
        // tracked content, not pristine/harness-seeded stdin -- defer to
        // Python rather than mint fresh unconstrained bytes over it.
        if fd0_is_dup2d_tracked_file(state) {
            return Err(ProcedureError::Other(
                "scanf from fd=0 (dup2'd to a tracked Rust FileSystem file) falls back to Python"
                    .to_string(),
            ));
        }
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
///
/// **Tracked-fd gate (angr-5mnx3.31).** An fd that Rust's `FileSystem` has open
/// — the mainline `fopen(...); fscanf(f, ...)` shape, since `NativeFopen`
/// registers the fd plus its concrete or bounded-symbolic content — carries
/// *real* content, and minting fresh unconstrained BVs over it is exactly the
/// wrong-answer risk [`NativeSscanf`] refuses: it makes impossible paths
/// feasible and never advances `FileDescriptor::position`, so a later native
/// `read`/`fread` on the same fd re-reads the bytes fscanf conceptually
/// consumed. Those calls defer to Python, which parses the content faithfully
/// (`rust_callback_dispatch::_inject_rust_fds` hydrates the callback state's
/// `posix.fd` from the Rust descriptor — name, content and position — so the
/// bounced proc sees the same stream). This deliberately DIVERGES from the
/// `read.rs`/`fread.rs` gate, which bounces on the *un*tracked fd: those two
/// serve content natively and so must yield the fd they cannot model, whereas
/// fscanf never serves content and so must yield the fd it *could* have
/// mis-modelled. An untracked fd keeps the native mint — nothing on either
/// side knows its bytes, so fresh symbolic values are the same approximation
/// `scanf` makes for unseeded stdin.
fn do_fscanf(
    state: &mut RustSimState,
    file_ptr: u64,
    fmt_addr: u64,
    ptr_args: &[RustBV],
) -> Result<Option<RustBV>, ProcedureError> {
    let fd = crate::procedures::fileops::read_fileno(state, file_ptr)?;
    if fd < 0 {
        return Ok(Some(arch_word(state, -1i64 as u64)));
    }
    let (source, record_stdin) = if fd == 0 {
        // angr-qmrrp: mirror the tracked-fd gate below for fd 0 -- a prior
        // dup2(real_fd, 0) means fd 0 is no longer pristine/harness-seeded
        // stdin, so it must bounce like any other Rust-tracked open fd.
        if fd0_is_dup2d_tracked_file(state) {
            return Err(ProcedureError::Other(
                "fscanf from fd=0 (dup2'd to a tracked Rust FileSystem file) falls back to Python"
                    .to_string(),
            ));
        }
        ("stdin", true)
    } else {
        if state.file_system_ref().is_open(fd as u32) {
            return Err(ProcedureError::Other(format!(
                "fscanf from fd={fd} (open in Rust FileSystem) falls back to Python"
            )));
        }
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

test_submod!("scanf_tests.rs" => scanf_tests);
