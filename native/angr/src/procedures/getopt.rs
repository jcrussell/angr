//! Native `getopt(3)`.
//!
//! Port of `angr/procedures/libc/getopt.py::_GetOptBase` (itself a port of
//! glibc's `posix/getopt.c`) for the 3-argument `getopt` only: `getopt_long`
//! / `getopt_long_only` stay on the Python proc, which owns the option-table
//! walk. Like Python it permutes argv under the default PERMUTE ordering,
//! honours a leading `+` (REQUIRE_ORDER) or `-` (RETURN_IN_ORDER, operands
//! return as option 1), and rejects `:` and `;` as option characters.
//!
//! Contract — **match Python, or defer** (return `ProcedureError` → Python
//! fallback). We defer whenever anything the parser reads is symbolic or
//! unterminated (argc, argv, an argv element, the optstring, the guest
//! `optind`), which is where Python's `_Unsupported` returns a fresh
//! unconstrained value and changes nothing. `Parser` reads only; every
//! write happens in `Parser::commit`, so a deferred call leaves no trace.
//!
//! The scan state between calls is [`GetoptCursor`] on `RustSimState`, the
//! mirror of `state.libc.getopt_*`. The loader-resolved guest addresses of the
//! `optind` / `optarg` / `optopt` extern globals are pushed from Python into
//! `RustSimState::getopt_extern` (bhk0a.1); a `None` address skips that store,
//! as Python's `_store_global` does when the symbol is absent.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** `argv`, the
//! optstring and the saved cursor are all guest-controlled, so this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`, uses checked
//! slice access and checked cursor arithmetic, and every unresolvable value
//! returns a `ProcedureError` that falls back to the Python proc.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use super::strings::{MAX_STRING_SCAN, scan_concrete_until_null};
use super::{ProcedureError, arch_word};
use crate::state::{GetoptCursor, RustSimState};
use crate::symbolic::RustBV;

const DASH: u8 = b'-';
const COLON: u8 = b':';
const QUESTION: u32 = b'?' as u32;
const DONE: u32 = 0xFFFF_FFFF; // -1

/// argv ordering, selected by a leading `+` or `-` in the optstring.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ordering {
    RequireOrder,
    Permute,
    ReturnInOrder,
}

/// `v + 1` for a cursor index, deferring instead of wrapping: the cursor is
/// restored from guest-reachable state.
fn bump(v: u32) -> Result<u32, ProcedureError> {
    v.checked_add(1)
        .ok_or_else(|| ProcedureError::Other("getopt cursor overflow".into()))
}

/// One getopt call: Python's `_parse` / `_next` / `_short`. The parse only
/// reads `state`; the writes accumulate here until `commit` applies them.
struct Parser<'a> {
    state: &'a mut RustSimState,
    argc: u32,
    argv: u64,
    optstring: Vec<u8>,
    /// argv slots read so far, rewritten in place by `exchange`.
    argv_ptrs: BTreeMap<u32, u64>,
    /// argv slots `exchange` moved; `commit` writes them back.
    dirty: BTreeSet<u32>,
    cur: GetoptCursor,
    optarg: u64,
    optopt: Option<u32>,
}

impl Parser<'_> {
    fn argp(&mut self, i: u32) -> Result<u64, ProcedureError> {
        if let Some(&p) = self.argv_ptrs.get(&i) {
            return Ok(p);
        }
        let ps = self.state.arch().bytes();
        // `argv` is guest-controlled: wrap like the guest's own indexing
        // would (invariant-rust-concrete-arith-must-wrap). u32 * 8 fits u64.
        let slot = self.argv.wrapping_add(u64::from(i) * ps as u64);
        let p = self
            .state
            .memory_load(slot, ps)?
            .as_u64()
            .ok_or_else(|| ProcedureError::SymbolicArgument("argv element".into()))?;
        self.argv_ptrs.insert(i, p);
        Ok(p)
    }

    fn arg(&mut self, i: u32) -> Result<Vec<u8>, ProcedureError> {
        let p = self.argp(i)?;
        scan_concrete_until_null(self.state, p, MAX_STRING_SCAN, "argv string")
    }

    fn nonoption(&mut self, i: u32) -> Result<bool, ProcedureError> {
        let arg = self.arg(i)?;
        Ok(arg.len() < 2 || arg[0] != DASH)
    }

    /// Move the skipped operands `argv[first..last]` behind the options
    /// `argv[last..optind]`.
    fn exchange(&mut self) -> Result<(), ProcedureError> {
        let GetoptCursor {
            optind,
            first_nonopt: first,
            last_nonopt: last,
            ..
        } = self.cur;
        if !(first <= last && last <= optind) {
            return Err(ProcedureError::Other("getopt cursor out of order".into()));
        }
        let mut ptrs = (first..optind)
            .map(|i| self.argp(i))
            .collect::<Result<Vec<_>, _>>()?;
        ptrs.rotate_left((last - first) as usize); // overflow-ok: first <= last checked above
        for (i, p) in (first..optind).zip(ptrs) {
            self.argv_ptrs.insert(i, p);
            self.dirty.insert(i);
        }
        self.cur.first_nonopt = first + (optind - last); // overflow-ok: first <= last <= optind checked above
        self.cur.last_nonopt = optind;
        Ok(())
    }

    fn next(&mut self, ordering: Ordering) -> Result<u32, ProcedureError> {
        if self.cur.nextchar == 0 {
            // starting a new argv element
            self.cur.last_nonopt = self.cur.last_nonopt.min(self.cur.optind);
            self.cur.first_nonopt = self.cur.first_nonopt.min(self.cur.optind);

            if ordering == Ordering::Permute {
                let c = self.cur;
                if c.first_nonopt != c.last_nonopt && c.last_nonopt != c.optind {
                    self.exchange()?;
                } else if c.last_nonopt != c.optind {
                    self.cur.first_nonopt = c.optind;
                }
                while self.cur.optind < self.argc && self.nonoption(self.cur.optind)? {
                    self.cur.optind = bump(self.cur.optind)?;
                }
                self.cur.last_nonopt = self.cur.optind;
            }

            if self.cur.optind < self.argc && self.arg(self.cur.optind)? == b"--" {
                self.cur.optind = bump(self.cur.optind)?;
                let c = self.cur;
                if c.first_nonopt != c.last_nonopt && c.last_nonopt != c.optind {
                    self.exchange()?;
                } else if c.first_nonopt == c.last_nonopt {
                    self.cur.first_nonopt = c.optind;
                }
                self.cur.last_nonopt = self.argc;
                self.cur.optind = self.argc;
            }

            if self.cur.optind >= self.argc {
                // leave optind on the first operand
                if self.cur.first_nonopt != self.cur.last_nonopt {
                    self.cur.optind = self.cur.first_nonopt;
                }
                return Ok(DONE);
            }

            if self.nonoption(self.cur.optind)? {
                if ordering == Ordering::RequireOrder {
                    return Ok(DONE);
                }
                // RETURN_IN_ORDER: report the operand as the argument of option 1
                self.optarg = self.argp(self.cur.optind)?;
                self.cur.optind = bump(self.cur.optind)?;
                return Ok(1);
            }
            self.cur.nextchar = 1;
        }
        self.short()
    }

    fn short(&mut self) -> Result<u32, ProcedureError> {
        let arg = self.arg(self.cur.optind)?;
        // The saved `nextchar` can exceed an argv element the guest has since
        // shortened in place (angr-qwyti.19): defer rather than index past it.
        let Some(&c) = arg.get(self.cur.nextchar as usize) else {
            return Err(ProcedureError::Other(
                "getopt cursor past end of argv element (shrunk argv)".into(),
            ));
        };
        self.cur.nextchar = bump(self.cur.nextchar)?;
        let at_end = self.cur.nextchar as usize >= arg.len();
        if at_end {
            self.cur.optind = bump(self.cur.optind)?;
        }

        let idx = match self.optstring.iter().position(|&b| b == c) {
            Some(idx) if c != COLON && c != b';' => idx,
            _ => {
                if at_end {
                    self.cur.nextchar = 0;
                }
                self.optopt = Some(u32::from(c));
                return Ok(QUESTION);
            }
        };

        let spec = |k: usize| self.optstring.get(idx.saturating_add(k)) == Some(&COLON);
        let mut ret = u32::from(c);
        if spec(1) {
            let optional = spec(2);
            if !at_end {
                // the argument is the rest of this element
                self.optarg = self
                    .argp(self.cur.optind)?
                    .wrapping_add(u64::from(self.cur.nextchar));
                self.cur.optind = bump(self.cur.optind)?;
            } else if !optional {
                if self.cur.optind >= self.argc {
                    self.optopt = Some(u32::from(c));
                    if self.optstring.first() == Some(&COLON) {
                        ret = u32::from(COLON);
                    } else {
                        ret = QUESTION;
                    }
                } else {
                    self.optarg = self.argp(self.cur.optind)?;
                    self.cur.optind = bump(self.cur.optind)?;
                }
            }
            self.cur.nextchar = 0;
        } else if at_end {
            self.cur.nextchar = 0;
        }
        Ok(ret)
    }

    /// The writes Python's `_parse` makes after `_next`: the permuted argv
    /// slots, the cursor, and the `optind` / `optarg` / `optopt` globals.
    fn commit(self) -> Result<(), ProcedureError> {
        let state = self.state;
        let ps = state.arch().bytes() as u64;
        for i in &self.dirty {
            if let Some(&p) = self.argv_ptrs.get(i) {
                let word = arch_word(state, p);
                state.memory_store(self.argv.wrapping_add(u64::from(*i) * ps), word)?;
            }
        }
        state.set_getopt_cursor(self.cur);
        let ext = state.getopt_extern();
        store_int(state, ext.optind, self.cur.optind)?;
        store_ptr(state, ext.optarg, self.optarg)?;
        if let Some(optopt) = self.optopt {
            store_int(state, ext.optopt, optopt)?;
        }
        Ok(())
    }
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

#[inline]
fn ret(value: u32) -> Result<Option<RustBV>, ProcedureError> {
    Ok(Some(RustBV::concrete(value as u128, 32)))
}

crate::declare_proc! {
    /// getopt: concrete-argv port of the Python proc, argv permutation
    /// included. Defers to Python for any symbolic input (matching the
    /// Python proc's unconstrained branch).
    ///
    /// ```c
    /// int getopt(int argc, char *const argv[], const char *optstring);
    /// ```
    name = "getopt",
    struct = NativeGetopt,
    args = [argc: concrete, argv_ptr: concrete, optstring_ptr: concrete],
    call |state| {
        let argc = argc as u32;
        let optstring = scan_concrete_until_null(state, optstring_ptr, MAX_STRING_SCAN, "optstring")?;
        if argc < 1 {
            return ret(DONE);
        }

        let mut cur = state.getopt_cursor();
        if let Some(addr) = state.getopt_extern().optind {
            let guest = state
                .memory_load(addr, 4)?
                .as_u64()
                .ok_or_else(|| ProcedureError::SymbolicArgument("guest optind".into()))?
                as u32;
            if guest == 0 {
                // optind = 0 requests a full reset
                cur = GetoptCursor::default();
            } else if guest != cur.optind {
                cur.optind = guest;
                cur.nextchar = 0;
            }
        }

        let (ordering, optstring) = match optstring.split_first() {
            Some((&DASH, rest)) => (Ordering::ReturnInOrder, rest.to_vec()),
            Some((&b'+', rest)) => (Ordering::RequireOrder, rest.to_vec()),
            _ => (Ordering::Permute, optstring),
        };
        let mut parser = Parser {
            state,
            argc,
            argv: argv_ptr,
            optstring,
            argv_ptrs: BTreeMap::new(),
            dirty: BTreeSet::new(),
            cur,
            optarg: 0,
            optopt: None,
        };
        let value = parser.next(ordering)?;
        parser.commit()?;
        ret(value)
    }
}

test_submod!("getopt_tests.rs" => getopt_tests);
