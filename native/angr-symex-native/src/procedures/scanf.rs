//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. On stdin a numeric
//! conversion also reads digit bytes spelling that value, and each format
//! literal is read as its own bytes, so `posix.dumps(0)` is parseable input
//! (see [`scanf_spell`](super::scanf_spell)). This eliminates Python callback
//! overhead for common CTF patterns using scanf for input.
//!
//! Supported conversions: `%d`, `%i`, `%u`, `%x`, `%X`, `%o`, `%s`, `%c` and
//! the `%%` literal. Each may carry an assignment-suppressing `*` and a field
//! width; the numeric conversions additionally honour every length modifier
//! [`parse_length_modifier`] accepts — `h`, `hh`, `l`, `ll`, `z`, `j`, `t` —
//! sized by [`LengthModifier::int_conv_bits`], so `%hhd`, `%hd`, `%ld`,
//! `%lld`, `%lu`, `%lx`, `%zu`, `%jd` and `%td` all parse natively. `%s` and
//! `%c` parse a modifier too but ignore it: their widths are fixed.
//! Falls back to Python for symbolic format strings or pointer arguments, and
//! for the conversions Python models differently — the `%[...]` scanset,
//! `%n`, and the float conversions; see
//! [`scanf_spec::parse_scanf_format`](super::scanf_spec::parse_scanf_format), whose arms
//! record why deferring is the faithful behavior for each.
//!
//! [`parse_length_modifier`]: super::format_common::parse_length_modifier
//! [`LengthModifier::int_conv_bits`]: super::format_common::LengthModifier::int_conv_bits
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** the format
//! string and every pointer argument are guest data, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; unparseable input
//! returns a `ProcedureError` that falls back to Python instead. The one
//! statement-level `#[allow]` below is the `mint_stdin_bytes` one-name
//! contract, not an input check.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::arch_word;
use super::format_common::read_format_string;
use super::scanf_spec::{ScanfSpec, parse_scanf_format};
use super::scanf_spell::{max_digits, numeric_spelling};
use super::sprintf::MAX_VARARGS;
use super::stdin_common::{
    bounce_if_fd0_dup2d, bounce_if_seed_ends_before, mint_stdin_bytes,
    stdin_seed_has_concrete_stop, stdin_seed_unconsumed,
};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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
    let format = parse_scanf_format(&fmt, state.arch().bits())?;
    let specs = &format.specs;

    // A numeric conversion on stdin reads its digit bytes off the stream and
    // constrains them to spell the stored value (`scanf_spell`), which is
    // Python's SimPackets model. A harness seed is a SimFile or a pre-filled
    // packet list, which Python parses differently, so while fd 0 still has
    // unread seed the whole call defers (angr-ggb66). Checked before any
    // store so the fallback sees a pristine state.
    // A format literal is read the same way, so it defers too.
    let reads_literal =
        !format.trailing_literal.is_empty() || specs.iter().any(|s| !s.literal_before.is_empty());
    if record_stdin
        && (reads_literal || specs.iter().any(|s| !s.is_string && !s.is_char))
        && stdin_seed_unconsumed(state)
    {
        return Err(ProcedureError::Other(
            "scanf: numeric conversion or literal over harness-seeded stdin".to_string(),
        ));
    }

    // `%s` is Python's SimPackets read (format_parser.py::FormatString.interpret):
    // a fixed-length read whose every byte is constrained off
    // `SCANF_DELIMITERS`. Two inputs that model does not cover defer the whole
    // call, checked before any store. Under SHORT_READS Python's read length is
    // symbolic, so the bytes past it keep their old memory, which a native
    // procedure cannot load without risking a mid-call fallback. And a concrete
    // delimiter in the unread harness seed would make the constraints UNSAT and
    // kill the path, where real scanf(3) stops the conversion there.
    if specs.iter().any(|s| s.is_string) {
        if state.has_option("SHORT_READS") {
            return Err(ProcedureError::Other(
                "scanf: %s under SHORT_READS has a symbolic length".to_string(),
            ));
        }
        // Conservative: the whole unread seed is checked, not just the window
        // the call's `%s` conversions would read.
        if record_stdin && stdin_seed_has_concrete_stop(state, usize::MAX, &SCANF_DELIMITERS) {
            return Err(ProcedureError::Other(
                "scanf: %s over a harness seed holding a delimiter".to_string(),
            ));
        }
    }

    // A seed that ends the stream before the bytes this call reads (each `%s`
    // its width, each other conversion at least one byte) is an EOF the
    // positional reads below do not model (angr-vmi2l.110).
    if record_stdin {
        let need = specs
            .iter()
            .map(|s| {
                let conv = if s.is_string {
                    s.max_str_len as usize
                } else if s.is_char {
                    1
                } else {
                    numeric_digits(s) as usize
                };
                s.literal_before.len().saturating_add(conv)
            })
            .fold(format.trailing_literal.len(), usize::saturating_add);
        bounce_if_seed_ends_before(state, need, "scanf")?;
    }

    // Python's `va_arg` reads every pointer the format needs, but a native
    // proc fetches a fixed `num_args()`. A format assigning more conversions
    // than that defers whole, before any store (angr-vmi2l.112).
    if specs.iter().filter(|s| !s.suppress).count() > ptr_args.len() {
        return Err(ProcedureError::Other(format!(
            "scanf: format assigns more than {} pointer arguments",
            ptr_args.len()
        )));
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

        // Python reads the format's literal bytes off the stream as well and
        // constrains them equal, so they reach `posix.dumps(0)`
        // (angr-vmi2l.108). Off stdin nothing reconstructs the stream.
        if record_stdin {
            read_stdin_literal(state, &spec.literal_before, source, scan_id, spec_idx);
        }

        let ptr = if spec.suppress {
            None
        } else {
            let arg = ptr_args.get(arg_idx).ok_or_else(|| {
                ProcedureError::Other(format!("scanf: no pointer argument {arg_idx}"))
            })?;
            let ptr = extract_concrete_arg(arg, &format!("scanf arg {arg_idx}"))?;
            arg_idx += 1;
            Some(ptr)
        };

        if spec.is_string {
            // %s: `max_str_len` symbolic bytes, none a delimiter, + NUL
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
                names.iter().map(|name| RustBV::symbol(name, 8)).collect()
            };

            // Every byte read is constrained, the suppressed case included:
            // the bytes were still read off the stream (angr-vmi2l.111).
            let not_delimiter: Vec<RustBV> = sym_bytes
                .iter()
                .flat_map(|byte| {
                    SCANF_DELIMITERS
                        .iter()
                        .map(|&d| byte.ne(&RustBV::concrete(u128::from(d), 8)))
                })
                .collect();
            for c in not_delimiter {
                state.add_constraint(c);
            }

            // Suppressed (`ptr == None`): the bytes have nowhere to land.
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
        } else if record_stdin {
            // Numeric on stdin: Python's SimPackets read of the digit bytes,
            // constrained to spell the stored value (angr-vmi2l.108). The
            // explicit-width range bound is part of the spelling.
            let digits = numeric_digits(spec);
            if digits == 0 {
                // `%0d`: Python reads no digits and then indexes `chars[0]`.
                return Err(ProcedureError::Other(
                    "scanf: zero-width numeric conversion".to_string(),
                ));
            }
            let names: Vec<String> = (0..digits)
                .map(|j| format!("{source}_scanf_{scan_id}_{spec_idx}_d{j}"))
                .collect();
            let chars = mint_stdin_bytes(state, &names);
            let sym_val = RustBV::symbol(format!("{source}_scanf_{scan_id}_{spec_idx}"), spec.bits);
            state.add_constraint(numeric_spelling(
                &chars, spec.conv, spec.base, &sym_val, spec.bits,
            ));
            if let Some(ptr) = ptr {
                state.memory_store(ptr, sym_val)?;
            }
        } else {
            // Numeric off stdin (fscanf on an untracked fd): one free value of
            // the conversion's width; nothing reconstructs that stream.
            let name = format!("{source}_scanf_{scan_id}_{spec_idx}");
            let sym_val = { RustBV::symbol(&name, spec.bits) };

            // An explicit field width bounds how many digits the conversion can
            // read, so it bounds the value — see `digit_range_bound`. Built
            // inside its own borrow scope because `add_constraint` re-borrows
            // the solver (same shape as access.rs/fgets.rs).
            if let Some((hi, lo_magnitude)) = spec.digit_bound {
                let bound = {
                    // `RustBV::concrete` masks to the width, so the two's
                    // complement of `lo_magnitude` needs no hand-masking.
                    let hi_bv = RustBV::concrete(u128::from(hi), spec.bits);
                    let lo_bv =
                        RustBV::concrete(u128::from(lo_magnitude).wrapping_neg(), spec.bits);
                    sym_val.sle(&hi_bv).and(&sym_val.sge(&lo_bv))
                };
                state.add_constraint(bound);
            }

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

    if record_stdin {
        read_stdin_literal(
            state,
            &format.trailing_literal,
            source,
            scan_id,
            specs.len(),
        );
    }

    // Return number of successful conversions
    Ok(Some(arch_word(state, conversions)))
}

/// Digit bytes a numeric conversion reads: its field width, else Python's
/// `max_digits` for the value's width.
fn numeric_digits(spec: &ScanfSpec) -> u64 {
    spec.width
        .unwrap_or_else(|| max_digits(spec.base, spec.bits))
}

/// Read a format literal off stdin, each byte constrained equal to it.
fn read_stdin_literal(
    state: &mut RustSimState,
    literal: &[u8],
    source: &str,
    scan_id: u64,
    idx: usize,
) {
    if literal.is_empty() {
        return;
    }
    let names: Vec<String> = (0..literal.len())
        .map(|j| format!("{source}_scanf_{scan_id}_lit{idx}_{j}"))
        .collect();
    let bytes = mint_stdin_bytes(state, &names);
    for (b, &c) in bytes.iter().zip(literal) {
        state.add_constraint(b.eq(&RustBV::concrete(u128::from(c), 8)));
    }
}

/// The bytes a `%s` stops at: Python's `FormatString.SCANF_DELIMITERS`
/// (`\t \n \v \r` and space).
const SCANF_DELIMITERS: [u8; 5] = [b'\t', b'\n', 0x0b, b'\r', b' '];

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
        1 + MAX_VARARGS // format + pointer args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fmt_addr = extract_concrete_arg(&args[0], "format")?;
        // angr-qmrrp: see `stdin_common::bounce_if_fd0_dup2d`.
        bounce_if_fd0_dup2d(state, "scanf")?;
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
        2 + MAX_VARARGS // str + format + pointer args
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
    let fd = crate::procedures::stream_ops::read_fileno(state, file_ptr)?;
    if fd < 0 {
        return Ok(Some(arch_word(state, -1i64 as u64)));
    }
    let (source, record_stdin) = if fd == 0 {
        // angr-qmrrp: mirror the tracked-fd gate below for fd 0 -- a prior
        // dup2(real_fd, 0) means fd 0 is no longer pristine/harness-seeded
        // stdin, so it must bounce like any other Rust-tracked open fd.
        bounce_if_fd0_dup2d(state, "fscanf")?;
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
        2 + MAX_VARARGS // stream + format + pointer args
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
