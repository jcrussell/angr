//! Native sprintf/snprintf implementations.
//!
//! Handles %s, %d, %i, %u, %x, %X, %o, %c, %p, %% format specifiers
//! with width, zero-padding, and left-alignment flags. Falls back to
//! Python for symbolic format strings or arguments.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FMT_LEN: usize = 4096;
const MAX_OUTPUT_LEN: usize = 4096;

/// Read a null-terminated string from memory at `addr`.
fn read_string(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let mut buf = Vec::new();
    for i in 0..MAX_FMT_LEN as u64 {
        match state.memory_load(addr.wrapping_add(i), 1) {
            Ok(bv) => {
                let byte =
                    extract_concrete_arg(&bv, &format!("byte at 0x{:x}", addr.wrapping_add(i)))?
                        as u8;
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

/// Format arguments according to a printf-style format string.
///
/// `args` is the slice of variadic arguments (after dest/format/size).
/// Returns the formatted output bytes and the number of variadic args consumed.
fn format_string(
    state: &mut RustSimState,
    fmt: &[u8],
    args: &[RustBV],
) -> Result<Vec<u8>, ProcedureError> {
    let mut output = Vec::new();
    let mut arg_idx: usize = 0;
    let mut i = 0;

    while i < fmt.len() {
        if fmt[i] != b'%' {
            output.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1; // skip '%'
        if i >= fmt.len() {
            break;
        }

        // Handle %%
        if fmt[i] == b'%' {
            output.push(b'%');
            i += 1;
            continue;
        }

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

        // Parse width
        let mut width: usize = 0;
        if i < fmt.len() && fmt[i] == b'*' {
            // Width from argument
            if arg_idx >= args.len() {
                return Err(ProcedureError::SymbolicArgument("width arg".to_string()));
            }
            let w = extract_concrete_arg(&args[arg_idx], "width")?;
            width = w as usize;
            arg_idx += 1;
            i += 1;
        } else {
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                width = width * 10 + (fmt[i] - b'0') as usize;
                i += 1;
            }
        }

        // Parse precision (skip for now, just consume it)
        let mut _precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            let mut prec: usize = 0;
            if i < fmt.len() && fmt[i] == b'*' {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument(
                        "precision arg".to_string(),
                    ));
                }
                let p = extract_concrete_arg(&args[arg_idx], "precision")?;
                prec = p as usize;
                arg_idx += 1;
                i += 1;
            } else {
                while i < fmt.len() && fmt[i].is_ascii_digit() {
                    prec = prec * 10 + (fmt[i] - b'0') as usize;
                    i += 1;
                }
            }
            _precision = Some(prec);
        }

        // Parse length modifier
        let mut long = false;
        let mut long_long = false;
        if i < fmt.len() {
            match fmt[i] {
                b'l' => {
                    i += 1;
                    if i < fmt.len() && fmt[i] == b'l' {
                        long_long = true;
                        i += 1;
                    } else {
                        long = true;
                    }
                }
                b'h' => {
                    i += 1;
                    if i < fmt.len() && fmt[i] == b'h' {
                        i += 1; // hh
                    }
                }
                b'z' | b'j' | b't' => {
                    long = true; // treat as long on 64-bit
                    i += 1;
                }
                _ => {}
            }
        }

        if i >= fmt.len() {
            break;
        }

        // Parse conversion specifier
        let spec = fmt[i];
        i += 1;

        match spec {
            b'd' | b'i' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("int arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                // Interpret as signed
                let signed_val = if long || long_long {
                    val as i64
                } else {
                    val as i32 as i64
                };
                let formatted = if plus_sign && signed_val >= 0 {
                    format!("+{}", signed_val)
                } else if space_sign && signed_val >= 0 {
                    format!(" {}", signed_val)
                } else {
                    format!("{}", signed_val)
                };
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    signed_val < 0,
                );
            }
            b'u' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("uint arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let unsigned_val = if long || long_long {
                    val
                } else {
                    val as u32 as u64
                };
                let formatted = format!("{}", unsigned_val);
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'x' | b'X' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("hex arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let unsigned_val = if long || long_long {
                    val
                } else {
                    val as u32 as u64
                };
                let mut formatted = if spec == b'x' {
                    format!("{:x}", unsigned_val)
                } else {
                    format!("{:X}", unsigned_val)
                };
                if hash_flag && unsigned_val != 0 {
                    let prefix = if spec == b'x' { "0x" } else { "0X" };
                    formatted = format!("{}{}", prefix, formatted);
                }
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'o' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("octal arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let unsigned_val = if long || long_long {
                    val
                } else {
                    val as u32 as u64
                };
                let mut formatted = format!("{:o}", unsigned_val);
                if hash_flag && unsigned_val != 0 {
                    formatted = format!("0{}", formatted);
                }
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'c' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("char arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let ch = [val as u8];
                pad_and_push(&mut output, &ch, width, left_align, false, false);
            }
            b's' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("string arg".to_string()));
                }
                let str_addr = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let s = read_string(state, str_addr)?;
                let s = if let Some(prec) = _precision {
                    if prec < s.len() { &s[..prec] } else { &s }
                } else {
                    &s
                };
                pad_and_push(&mut output, s, width, left_align, false, false);
            }
            b'p' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("ptr arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{}", arg_idx))?;
                arg_idx += 1;
                let formatted = format!("0x{:x}", val);
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    false,
                    false,
                );
            }
            b'n' => {
                // %n writes the number of chars written so far — skip for safety
                return Err(ProcedureError::Other("%n not supported".to_string()));
            }
            _ => {
                // Unknown specifier — fall back to Python
                return Err(ProcedureError::Other(format!(
                    "unsupported format specifier '%{}'",
                    spec as char
                )));
            }
        }

        if output.len() > MAX_OUTPUT_LEN {
            return Err(ProcedureError::MaxIterations(MAX_OUTPUT_LEN));
        }
    }

    Ok(output)
}

/// Pad a formatted value and push to output.
fn pad_and_push(
    output: &mut Vec<u8>,
    value: &[u8],
    width: usize,
    left_align: bool,
    zero_pad: bool,
    is_negative: bool,
) {
    if width <= value.len() {
        output.extend_from_slice(value);
        return;
    }
    let pad_count = width - value.len();
    let pad_char = if zero_pad { b'0' } else { b' ' };

    if left_align {
        output.extend_from_slice(value);
        for _ in 0..pad_count {
            output.push(b' ');
        }
    } else if zero_pad && is_negative {
        // For negative numbers with zero-pad: "-007" not "00-7"
        output.push(b'-');
        for _ in 0..pad_count {
            output.push(b'0');
        }
        output.extend_from_slice(&value[1..]); // skip the '-' already in value
    } else {
        for _ in 0..pad_count {
            output.push(pad_char);
        }
        output.extend_from_slice(value);
    }
}

/// Native sprintf implementation.
///
/// ```c
/// int sprintf(char *str, const char *format, ...);
/// ```
pub struct NativeSprintf;

impl NativeSimProcedure for NativeSprintf {
    fn name(&self) -> &'static str {
        "sprintf"
    }

    fn num_args(&self) -> usize {
        8 // dest + format + up to 6 variadic args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        let fmt = read_string(state, fmt_addr)?;
        let varargs = &args[2..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination
        for (i, &byte) in output.iter().enumerate() {
            state.memory_store(
                dest.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }
        // Null terminator
        state.memory_store(
            dest.wrapping_add(output.len() as u64),
            RustBV::concrete(0, 8),
        )?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(output.len() as u128, bits)))
    }
}

/// Native snprintf implementation.
///
/// ```c
/// int snprintf(char *str, size_t size, const char *format, ...);
/// ```
pub struct NativeSnprintf;

impl NativeSimProcedure for NativeSnprintf {
    fn name(&self) -> &'static str {
        "snprintf"
    }

    fn num_args(&self) -> usize {
        9 // dest + size + format + up to 6 variadic args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let size = extract_concrete_arg(&args[1], "size")? as usize;
        let fmt_addr = extract_concrete_arg(&args[2], "format")?;

        let fmt = read_string(state, fmt_addr)?;
        let varargs = &args[3..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination, respecting size limit
        if size > 0 {
            let write_len = output.len().min(size - 1);
            for i in 0..write_len {
                state.memory_store(
                    dest.wrapping_add(i as u64),
                    RustBV::concrete(output[i] as u128, 8),
                )?;
            }
            // Always null-terminate if size > 0
            state.memory_store(dest.wrapping_add(write_len as u64), RustBV::concrete(0, 8))?;
        }

        // Return would-have-been length (not truncated)
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(output.len() as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_state() -> RustSimState {
        let mut state = RustSimState::new("amd64").unwrap();
        // Map destination buffer
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
    }

    #[test]
    fn test_sprintf_simple_string() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64), // dest
                    RustBV::concrete(0x1000, 64), // format
                    RustBV::concrete(0, 64),      // unused vararg
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(11));
        // Verify written data
        for (i, &expected) in b"hello world\x00".iter().enumerate() {
            let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
            assert_eq!(byte.as_u64().unwrap() as u8, expected);
        }
    }

    #[test]
    fn test_sprintf_percent_d() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(42, 64), // %d arg
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(6)); // "val=42"
        let mut out = Vec::new();
        for i in 0..6u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"val=42");
    }

    #[test]
    fn test_sprintf_percent_x() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%x\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(255, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(2)); // "ff"
        let mut out = Vec::new();
        for i in 0..2u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"ff");
    }

    #[test]
    fn test_sprintf_percent_s() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"hi %s!\x00", Permission::RWX);
        state.map_memory_data(0x3000, b"world\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x3000, 64), // pointer to "world"
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(9)); // "hi world!"
        let mut out = Vec::new();
        for i in 0..9u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"hi world!");
    }

    #[test]
    fn test_sprintf_percent_c() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%c%c%c\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(b'A' as u128, 64),
                    RustBV::concrete(b'B' as u128, 64),
                    RustBV::concrete(b'C' as u128, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(3));
        let mut out = Vec::new();
        for i in 0..3u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"ABC");
    }

    #[test]
    fn test_sprintf_escaped_percent() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"100%%\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(4)); // "100%"
    }

    #[test]
    fn test_sprintf_width_padding() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%08x\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0xAB, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(8)); // "000000ab"
        let mut out = Vec::new();
        for i in 0..8u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"000000ab");
    }

    #[test]
    fn test_sprintf_negative_int() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

        // -1 as u64
        let neg_one = (-1i32) as u32 as u64;
        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(neg_one as u128, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(2)); // "-1"
        let mut out = Vec::new();
        for i in 0..2u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"-1");
    }

    #[test]
    fn test_snprintf_truncation() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

        let result = NativeSnprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(6, 64), // size = 6
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // Returns full would-have-been length
        assert_eq!(result.unwrap().as_u64(), Some(11));
        // But only wrote 5 chars + null
        let mut out = Vec::new();
        for i in 0..6u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"hello\x00");
    }

    #[test]
    fn test_snprintf_zero_size() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let result = NativeSnprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64), // size = 0
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // Returns full length, writes nothing
        assert_eq!(result.unwrap().as_u64(), Some(5));
    }

    #[test]
    fn test_sprintf_symbolic_dest() {
        let mut state = setup_state();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "dest", 64);
        drop(ctx);

        let result = NativeSprintf.call(
            &mut state,
            &[
                sym,
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        );
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_sprintf_pointer() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%p\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0xdeadbeef, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(10)); // "0xdeadbeef"
        let mut out = Vec::new();
        for i in 0..10u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"0xdeadbeef");
    }

    #[test]
    fn test_sprintf_multiple_args() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d+%d=%d\x00", Permission::RWX);

        let result = NativeSprintf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(1, 64),
                    RustBV::concrete(2, 64),
                    RustBV::concrete(3, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(5)); // "1+2=3"
        let mut out = Vec::new();
        for i in 0..5u64 {
            out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
        }
        assert_eq!(&out, b"1+2=3");
    }
}
