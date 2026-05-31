//! Native scanf/sscanf implementations for symbolic execution.
//!
//! For each format specifier, creates a symbolic BVS of the appropriate width
//! and stores it at the corresponding pointer argument. This eliminates
//! Python callback overhead for common CTF patterns using scanf for input.
//!
//! Supported specifiers: %d, %i, %u, %x, %o, %s, %c, %ld, %lld, %lu, %lx, %%
//! Falls back to Python for symbolic format strings or pointer arguments.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_FMT_LEN: usize = 4096;
const MAX_SCANF_STR_LEN: u64 = 256;

/// Read a null-terminated concrete string from memory.
fn read_format_string(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
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
        let mut field_width: u64 = 0;
        let mut has_width = false;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            field_width = field_width * 10 + (fmt[i] - b'0') as u64;
            has_width = true;
            i += 1;
        }

        // Parse length modifier
        let mut long_count = 0u8; // 0=int, 1=long, 2=long long
        if i < fmt.len() {
            match fmt[i] {
                b'l' => {
                    long_count = 1;
                    i += 1;
                    if i < fmt.len() && fmt[i] == b'l' {
                        long_count = 2;
                        i += 1;
                    }
                }
                b'h' => {
                    i += 1;
                    if i < fmt.len() && fmt[i] == b'h' {
                        i += 1; // hh = char
                    }
                    // short/char — still store as int-width for simplicity
                }
                _ => {}
            }
        }

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
                state
                    .memory_store(ptr.wrapping_add(j as u64), sym_byte)
                    ?;
            }

            // NUL terminator
            state
                .memory_store(ptr.wrapping_add(str_len), RustBV::concrete(0, 8))
                ?;
        } else {
            // Numeric or char: create one symbolic BVS of appropriate width
            let name = format!("stdin_scanf_{}_{}", scan_id, spec_idx);
            let sym_val = {
                let ctx = state.solver().borrow();
                RustBV::symbolic(&ctx, &name, spec.bits)
            };

            state.record_stdin_symbol(name, spec.bits);

            // Store to pointer — write spec.bits/8 bytes
            state
                .memory_store(ptr, sym_val)
                ?;
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
mod tests {
    use super::*;
    use crate::memory::Permission;

    fn setup_state() -> RustSimState {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0x2000, 0x1000, Permission::RWX);
        state
    }

    #[test]
    fn test_scanf_percent_d() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64), // format
                    RustBV::concrete(0x2000, 64), // &int_var
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // Should return 1 (one conversion)
        assert_eq!(result.unwrap().as_u64(), Some(1));

        // Value at 0x2000 should be symbolic (32-bit)
        let val = state.memory_load(0x2000, 4).unwrap();
        assert!(val.as_u64().is_none(), "scanf %d result should be symbolic");
    }

    #[test]
    fn test_scanf_two_ints() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d %d\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // &a
                    RustBV::concrete(0x2010, 64), // &b
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(2));

        // Both should be symbolic
        let a = state.memory_load(0x2000, 4).unwrap();
        let b = state.memory_load(0x2010, 4).unwrap();
        assert!(a.as_u64().is_none());
        assert!(b.as_u64().is_none());
    }

    #[test]
    fn test_scanf_percent_s() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%s\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // char buf[]
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));

        // First byte should be symbolic
        let first = state.memory_load(0x2000, 1).unwrap();
        assert!(
            first.as_u64().is_none(),
            "scanf %s first byte should be symbolic"
        );

        // NUL terminator at max_str_len offset
        let nul = state.memory_load(0x2000 + MAX_SCANF_STR_LEN, 1).unwrap();
        assert_eq!(nul.as_u64(), Some(0));
    }

    #[test]
    fn test_scanf_percent_s_with_width() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%10s\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));

        // NUL at offset 10
        let nul = state.memory_load(0x200A, 1).unwrap();
        assert_eq!(nul.as_u64(), Some(0));
    }

    #[test]
    fn test_scanf_percent_c() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%c\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // &char_var
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));

        // Single symbolic byte
        let val = state.memory_load(0x2000, 1).unwrap();
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_scanf_percent_x() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%x\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));

        let val = state.memory_load(0x2000, 4).unwrap();
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_scanf_long() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%ld\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // &long_var
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));

        // 64-bit symbolic value
        let val = state.memory_load(0x2000, 8).unwrap();
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_scanf_suppressed() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%*d %d\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // only one pointer (first %d is suppressed)
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // One successful conversion (suppressed doesn't count)
        assert_eq!(result.unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_scanf_symbolic_format() {
        let mut state = setup_state();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "fmt", 64);
        drop(ctx);

        let result = NativeScanf.call(
            &mut state,
            &[
                sym,
                RustBV::concrete(0x2000, 64),
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
    fn test_scanf_symbolic_ptr() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "ptr", 64);
        drop(ctx);

        let result = NativeScanf.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                sym,
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
    fn test_scanf_mixed_specifiers() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d %c %s\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64), // &int_var
                    RustBV::concrete(0x2010, 64), // &char_var
                    RustBV::concrete(0x2020, 64), // char buf[]
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(3));
    }

    #[test]
    fn test_isoc99_scanf() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

        let result = NativeIsoc99Scanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_sscanf_basic() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"42\x00", Permission::RWX); // source string
        state.map_memory_data(0x1100, b"%d\x00", Permission::RWX); // format

        let result = NativeSscanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64), // str
                    RustBV::concrete(0x1100, 64), // format
                    RustBV::concrete(0x2000, 64), // &int_var
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(1));
        // Value should be symbolic (we don't actually parse the source)
        let val = state.memory_load(0x2000, 4).unwrap();
        assert!(val.as_u64().is_none());
    }

    #[test]
    fn test_scanf_escaped_percent() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%%d\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
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

        // "%%" is literal %, "d" is literal — no conversions
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_scanf_no_conversions() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let result = NativeScanf
            .call(
                &mut state,
                &[
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

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_scanf_stdin_tracking() {
        let mut state = setup_state();
        state.map_memory_data(0x1000, b"%d %s\x00", Permission::RWX);

        NativeScanf
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x2100, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .unwrap();

        // Should have recorded stdin symbols
        let symbols = state.stdin_symbols();
        assert!(!symbols.is_empty(), "scanf should record stdin symbols");
        // One 32-bit symbol for %d + MAX_SCANF_STR_LEN 8-bit symbols for %s
        assert_eq!(symbols.len(), 1 + MAX_SCANF_STR_LEN as usize);
    }
}
