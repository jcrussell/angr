//! Conversion-specifier *parsing* for the native sprintf family.
//!
//! [`super::sprintf`]'s `format_string` loop has two halves: decide what a
//! `%...` specifier means, then render it. This file is the first half; the
//! second is [`super::sprintf_conv`]. Every Python-parity defer that depends
//! only on the *shape* of a specifier — the `-`/`+`/` `/`#` flags, `*`
//! dynamic width, `.N` and misplaced `.*` precision — is decided here, so the
//! list in [`super::sprintf`]'s module doc maps onto
//! [`parse_conversion_spec`]'s early `Err` returns.

use super::format_common::{LengthModifier, parse_length_modifier, parse_width_digits};
use super::sprintf::MAX_OUTPUT_LEN;
use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

/// One parsed `%` conversion specifier, ready for
/// [`super::sprintf_conv::render_conversion`].
///
/// `left_align` / `plus_sign` / `space_sign` / `hash_flag` are always false
/// while the flag parity guard in [`parse_conversion_spec`] stands — they are
/// carried (and honored downstream) so that a future `format_parser.py` fix
/// only has to drop that guard. See [`super::sprintf`]'s module doc.
pub(super) struct ConversionSpec {
    pub(super) left_align: bool,
    pub(super) zero_pad: bool,
    pub(super) plus_sign: bool,
    pub(super) space_sign: bool,
    pub(super) hash_flag: bool,
    /// Already clamped to [`MAX_OUTPUT_LEN`] — it is used as a padding-loop
    /// bound, see the clamp comment in [`parse_conversion_spec`].
    pub(super) width: usize,
    pub(super) precision: Option<usize>,
    pub(super) modifier: LengthModifier,
    /// The conversion letter itself (`b'd'`, `b's'`, ...).
    pub(super) conv: u8,
}

/// Parse the specifier starting at `*cursor`, which must point at the first
/// byte *after* a `%` that is not itself a `%`.
///
/// On success advances `*cursor` past the conversion letter and returns the
/// spec. Returns `Ok(None)` when the format string ends mid-specifier (the
/// caller stops formatting). `arg_idx` is advanced only by the `%.*s`
/// precision argument, the one specifier component that consumes a vararg
/// during parsing rather than rendering.
pub(super) fn parse_conversion_spec(
    fmt: &[u8],
    cursor: &mut usize,
    args: &[RustBV],
    arg_idx: &mut usize,
) -> Result<Option<ConversionSpec>, ProcedureError> {
    let mut i = *cursor;
    let mut arg_idx_local = *arg_idx;
    // Index of the first byte after '%', i.e. the start of the "nugget"
    // `format_parser.py::_match_spec` is handed. Its `.*` arm only fires
    // when the nugget *starts* with `.*`, so any flag or width digit ahead
    // of the precision changes Python's decision — see the `.*` block below.
    let nugget_start = i;

    // Parse flags
    let mut left_align = false;
    let mut zero_pad = false;
    let mut plus_sign = false;
    let mut space_sign = false;
    let mut hash_flag = false;
    loop {
        if i >= fmt.len() {
            break;
        }
        match fmt[i] {
            b'-' => left_align = true,
            b'0' => zero_pad = true,
            b'+' => plus_sign = true,
            b' ' => space_sign = true,
            b'#' => hash_flag = true,
            _ => break,
        }
        i += 1;
    }
    if left_align {
        zero_pad = false; // '-' overrides '0'
    }

    // Parity defer: Python's format_parser.py only recognizes the bare '0'
    // (zero-pad) flag. Its _match_spec has no arm for '-', '+', ' ', or '#',
    // so it fails to match the specifier entirely, emits a literal '%', and
    // does NOT consume the corresponding variadic arg. Native honoring these
    // flags would therefore produce different bytes AND a different
    // arg-consumption count than vanilla angr — a real cross-engine parity
    // gap (angr-1yge9.2). Defer to Python so both engines agree. Bare
    // '0'+width ("%05d") is matched by both parsers and stays native. The
    // downstream flag handling below is retained (unreachable while this
    // guard stands) so a future format_parser.py fix can drop just this
    // block. See `format-string-parity-defers`; do NOT "fix" by changing
    // Python's parser — that alters vanilla angr semantics for all users.
    if left_align || plus_sign || space_sign || hash_flag {
        return Err(ProcedureError::Other(
            "'-'/'+'/' '/'#' conversion flags defer to Python".to_string(),
        ));
    }

    // Parse width. '*' dynamic width diverges from Python: format_parser.py's
    // _match_spec has no '*' arm, so extract_components swallows the '%*'
    // without consuming a width arg, shifting every later variadic arg by
    // one. Native can't cheaply reproduce that arg-shift; defer to Python for
    // faithful parity (angr-3i88a).
    if i < fmt.len() && fmt[i] == b'*' {
        return Err(ProcedureError::Other(
            "'*' dynamic width defers to Python".to_string(),
        ));
    }
    // Clamp against MAX_OUTPUT_LEN here, before `width` is used as
    // pad_and_push's padding-loop bound: parse_width_digits only
    // saturates to usize::MAX on overflow, which does not stop a short
    // digit run like "%9999999999d" from encoding a near-MAX width. The
    // `output.len() > MAX_OUTPUT_LEN` check below only fires AFTER
    // pad_and_push's loop returns, so it can't bound the loop itself —
    // unclamped this is an unbounded-allocation / OOM loop (angr-mi56k).
    let (width, advanced) = parse_width_digits(fmt, i);
    let width = width.min(MAX_OUTPUT_LEN);
    i += advanced;

    // Parse precision.
    let mut precision: Option<usize> = None;
    if i < fmt.len() && fmt[i] == b'.' {
        i += 1;
        if i < fmt.len() && fmt[i] == b'*' {
            // '.*' precision (arg-supplied). Python's
            // `format_parser.py::_match_spec` only recognizes it when the
            // nugget *starts* with `.*` — a preceding '0' flag or width
            // digit sends it down the digit-scanning path, which leaves
            // nugget == ".*<spec>" and matches no entry of `all_spec`, so
            // the whole specifier fails to match, a literal '%' is emitted
            // and NO variadic arg is consumed. Native cannot cheaply
            // reproduce that arg-shift; defer, as the '*' dynamic-width
            // guard above does (angr-6cp06.8).
            //
            // overflow-ok: `dot_pos` is `i - 1` where `i > 0` — the '.' was
            // just consumed above, so the subtraction cannot underflow.
            let dot_pos = i - 1;
            if dot_pos != nugget_start {
                return Err(ProcedureError::Other(
                    "'.*' precision after a flag/width defers to Python".to_string(),
                ));
            }
            // Even when matched, Python only reads a *separate* precision
            // vararg for '%s': `FormatString.replace` does
            // `va_arg("size_t") if fmt_spec.length_spec == b".*"` inside the
            // `spec_type == b"s"` arm only. Every other conversion falls to
            // the `else` branch, reads exactly one `va_arg("void*")`, and
            // ignores `length_spec` entirely (the `rjust` below it is gated
            // on `isinstance(length_spec, int)`, and b".*" is not an int).
            // Consuming a precision arg for `%.*d` therefore shifted every
            // later vararg by one. Mirror Python: consume only for '%s',
            // and otherwise drop the precision (angr-6cp06.8).
            //
            // overflow-ok: both indices are positions inside an in-memory
            // format string, nowhere near usize::MAX.
            let (_, peek_adv) = parse_length_modifier(fmt, i + 1);
            let conv = fmt.get(i + 1 + peek_adv).copied();
            if conv == Some(b's') {
                if arg_idx_local >= args.len() {
                    return Err(ProcedureError::SymbolicArgument(
                        "precision arg".to_string(),
                    ));
                }
                let p = extract_concrete_arg(&args[arg_idx_local], "precision")?;
                precision = Some(p as usize);
                arg_idx_local += 1;
            }
            i += 1;
        } else {
            // '.N' digit precision diverges: format_parser.py's _match_spec
            // mis-slices the consumed '.', dropping the actual conversion
            // letter, so FormatString.replace raises SimProcedureError and
            // the state errors. Native previously truncated/ignored the
            // precision and continued — succeeding where Python errors.
            // Defer for parity (angr-3i88a).
            return Err(ProcedureError::Other(
                "'.N' digit precision defers to Python".to_string(),
            ));
        }
    }

    // Parse length modifier
    let (modifier, m_adv) = parse_length_modifier(fmt, i);
    i += m_adv;

    // SILENT(cat-a): a format string that ends inside a specifier ("...%l")
    // has no conversion letter to act on. Mirrors the caller's original
    // `break`: stop formatting and return what was built so far, exactly as
    // `format_parser.py` does when `_match_spec` runs off the end.
    if i >= fmt.len() {
        return Ok(None);
    }

    // Parse conversion specifier
    let spec = fmt[i];
    i += 1;

    *cursor = i;
    *arg_idx = arg_idx_local;
    Ok(Some(ConversionSpec {
        left_align,
        zero_pad,
        plus_sign,
        space_sign,
        hash_flag,
        width,
        precision,
        modifier,
        conv: spec,
    }))
}
