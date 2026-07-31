//! Z3 solver construction and the per-check timing/sampling wrappers.
//!
//! Extracted from `context.rs` (angr-a2br.2 slice 2) along the documented
//! "Z3 context lifecycle / constraint solving" boundary. These are free
//! functions with no `SymContext` dependency; they read process-wide env
//! knobs (cached via `OnceLock`) and bump the shared solver counters that
//! live in [`super::stats`].
//!
//! Crate-visible surface: [`build_solver`], [`build_solver_params`],
//! [`timed_check`], [`sample_simplify_skip`]. The env-spec helpers
//! (`tactic_spec`, `qfbv_smart_threshold`, `simplify_sample_stride`) and the
//! [`TacticSpec`] enum stay private to this module.
// Grandfathered clippy::unwrap_used/expect_used debt -- angr-9ke6b.212 tracks
// burning this down file by file. Do not add new unwrap()/expect() calls here;
// new files/callers must handle the None/Err case explicitly instead.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::Ordering;

use super::stats::*;

/// Sampling stride for `sample_simplify_skip`. Default 64; override via
/// `ANGR_Z3_SIMPLIFY_STRIDE` (any positive integer; values <=0 / unparseable
/// fall back to the default). Read once via `OnceLock` on first sample —
/// matches the `tactic_spec` / `qfbv_smart_threshold` pattern. A stride of 1
/// gives full-population sampling for a profiling spike (~`SIMPLIFY_SAMPLE_STRIDE_DEFAULT`x
/// overhead on the simplify path; safe for offline runs only).
#[cfg(feature = "vex-engine-z3")]
fn simplify_sample_stride() -> u64 {
    static STRIDE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("ANGR_Z3_SIMPLIFY_STRIDE")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(SIMPLIFY_SAMPLE_STRIDE_DEFAULT)
    })
}

/// angr-1joc: sampled Z3 simplify() check on a freshly-added Bool. Every
/// Nth call (N = `simplify_sample_stride()`) runs `simplify()` and records
/// whether the resulting Z3 AST ptr differs from the input. The ticker is
/// shared across `assume_true`, `assume_false`, and `add_constraint_raw`.
///
/// Calls `Z3_simplify` directly rather than the z3 crate's `Ast::simplify`,
/// which `unwrap()`s the result: `Z3_simplify` returns NULL when the context
/// has an error latched, and a Bool imported from claripy through
/// `add_constraint_raw` can hit that (repro: the `TestProxyUnsatCore` tests in
/// `tests/engines/rust/test_solver_ops.py`, which aborted the whole
/// interpreter on the unwrap). This is a diagnostic counter — a NULL just
/// means "no sample", never a crash.
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub(crate) fn sample_simplify_skip(constraint: &z3::ast::Bool) {
    use z3::ast::Ast;
    let tick = SIMPLIFY_SAMPLE_TICKER.fetch_add(1, Ordering::Relaxed);
    if !tick.is_multiple_of(simplify_sample_stride()) {
        return;
    }
    BRANCH_COND_SIMPLIFY_SAMPLED_COUNT.fetch_add(1, Ordering::Relaxed);
    let raw_ast = constraint.get_z3_ast();
    // SAFETY: `constraint` owns a live ref on `raw_ast`, and the raw context
    // handle is the one that AST was created in.
    let simplified = unsafe { z3_sys::Z3_simplify(constraint.get_ctx().get_z3_context(), raw_ast) };
    match simplified {
        Some(simplified) if simplified != raw_ast => {
            BRANCH_COND_SIMPLIFY_REDUCED_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        Some(_) => {}
        None => log::debug!("sample_simplify_skip: Z3_simplify returned NULL, sample dropped"),
    }
}

/// Timed wrapper around solver.check() — records count, total time, and per-site stats.
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub(crate) fn timed_check(solver: &z3::Solver, site: CheckSite) -> z3::SatResult {
    let start = std::time::Instant::now();
    let result = solver.check();
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    Z3_CHECK_COUNT.fetch_add(1, Ordering::Relaxed);
    Z3_CHECK_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
    let idx = site as usize;
    Z3_CHECK_SITE_COUNT[idx].fetch_add(1, Ordering::Relaxed);
    Z3_CHECK_SITE_TIME_NS[idx].fetch_add(elapsed_ns, Ordering::Relaxed);
    // Attribute this check to whatever query class is in flight (S1 spike,
    // angr-op0dn.3). Unclassified unless ANGR_RUST_QUERY_CLASS is set.
    crate::symbolic::query_class::record_check();
    match result {
        // satresult-exempt: per-variant stats counter distinguishes all three.
        z3::SatResult::Sat => Z3_SAT_COUNT.fetch_add(1, Ordering::Relaxed),
        z3::SatResult::Unsat => Z3_UNSAT_COUNT.fetch_add(1, Ordering::Relaxed),
        z3::SatResult::Unknown => Z3_TIMEOUT_COUNT.fetch_add(1, Ordering::Relaxed),
    };
    result
}

/// Single forcing-function for interpreting a Z3 `check()` outcome
/// (angr-qwyti.3). Every site that needs a sat/unsat *decision* routes through
/// [`SatOutcome::decided`] so a reviewer can see, at the call site, exactly
/// what a timeout does — `Unknown` becomes `None` and MUST be handled, never
/// silently folded into `false`/`Unsat`.
///
/// Conflating `Unknown` (Z3 timeout / incompleteness) with `Unsat` is the
/// angr-ph300.43 / angr-n0irt.1 bug class: it fabricates extrema, permanently
/// pins `sat_cache=false`, and prunes feasible branches. See the
/// `invariant-z3-unknown-not-unsat` memory. The only sanctioned raw
/// `SatResult::{Sat,Unsat}` matches are this trait's impl and the stats
/// counter in [`timed_check`], both tagged `// satresult-exempt`; a grep gate
/// (`tools/audit_satresult.py`) bans any new ones.
#[cfg(feature = "vex-engine-z3")]
pub(crate) trait SatOutcome {
    /// `Some(true)` = Sat, `Some(false)` = Unsat, `None` = Unknown (undecided
    /// — timeout or incompleteness). Callers must decide what `None` means at
    /// their own site rather than letting it collapse to a boolean.
    fn decided(self) -> Option<bool>;
}

#[cfg(feature = "vex-engine-z3")]
impl SatOutcome for z3::SatResult {
    #[inline]
    fn decided(self) -> Option<bool> {
        match self {
            // satresult-exempt: this impl is the sanctioned mapping site.
            z3::SatResult::Sat => Some(true),
            z3::SatResult::Unsat => Some(false),
            z3::SatResult::Unknown => None,
        }
    }
}

/// Build the Z3 `Params` object applied to every fresh `z3::Solver` and
/// every `set_timeout` call. Centralizes the bv_rewriter knobs we set.
///
/// Two knobs enabled (Z3 disables both by default):
/// - `bv_extract_prop`: propagates Extract inward through arithmetic.
/// - `mul2concat`: rewrites `x * 2^k` -> `concat(x, 0^k)`. Added in
///   angr-ya00.1 (free bv_* sweep, 2026-05-19) — drops fairlight's bimodal
///   slow-mode median from 21.66s to 14.46s (-33%) without regressing
///   8 other Z3-heavy benches measured (max delta +0.8% on flareon2015_2).
///
/// Three sweep knobs left disabled: `bv_not_simpl`, `bv_ite2id`,
/// `blast_eq_value` (no measurable effect across matrix). `bv_sort_ac`
/// rejected: looked like a fairlight winner combined with mul2concat
/// (median 8.10s) but blows up defcon2016quals_baby-re by 7x (0.50s ->
/// 3.71s) and adds +25% to flareon2015_2.
///
/// Seed pinning attempted and rejected (angr-iaol.1, 2026-05-25): three
/// param-name forms tried with the z3-0.19.7 `Params::set_u32` route:
/// - `smt.random_seed`: corrupts the solver. `eval()` returns models
///   that **violate the asserted constraints** (eg `x=0` for `x>=100`).
/// - `sat.random_seed`: same corruption mode.
/// - `random_seed` (the in-descriptor short name): is accepted without
///   corruption but does **not** stabilize models across two fresh
///   `RustSolverContext` instances with identical asserted constraints
///   (eval witnesses differ run-to-run by >10^6). The order-stability
///   test that *passed* without any seed pin now fails when this form
///   is set.
///
/// So Z3 4.13 model determinism across solver instances is not
/// reachable through `Z3_solver_set_params` for these keys; pinning
/// must instead happen via `Z3_global_param_set` *before* the first
/// `Solver::new` — and even that does not eliminate variable /
/// restart heuristic latitude. Tracked in iaol.1 close-out memory
/// `iaol1-seed-pin-empirically-broken`. AVOID `parallel.enable=true`
/// (`avoid-z3-parallel-enable`).
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn build_solver_params(timeout_ms: u32) -> z3::Params {
    let mut params = z3::Params::new();
    params.set_u32("timeout", timeout_ms);
    params.set_bool("bv_extract_prop", true);
    params.set_bool("mul2concat", true);
    apply_extra_params(&mut params);
    params
}

/// Parsed `ANGR_Z3_PARAMS` override spec, cached once per process. A
/// comma-separated list of `key=value` pairs applied to every fresh solver
/// AFTER the baked defaults, so an experiment can override or extend them
/// without a rebuild. `value` of `true`/`false` (case-insensitive) sets a
/// bool param; a value that parses as `u32` sets a uint param; a value
/// prefixed `sym:` sets a symbol param, e.g. `sat.phase=sym:always_false`
/// (angr-sijyb.1). Everything else — including a plain non-numeric value
/// with no `sym:` tag — is skipped, same as before this variant existed.
/// The tag is required rather than falling back to Symbol for anything
/// unparseable: a mistyped numeric value (e.g. `timeout=5oo`) must stay a
/// harmless no-op, not silently become a symbol param that corrupts the
/// solver the same way `smt.random_seed` did (see the doc comment on
/// `build_solver_params` above) — an untagged typo has no way to signal
/// which type was intended, so dropping it is the only safe default.
/// Purpose: A/B param surveys (angr-ovqja.2) on the Z3-heavy bench set via
/// `--counters-json` without recompiling per candidate.
///
/// Example: `ANGR_Z3_PARAMS="bv.size_reduce=true,relevancy=0,sat.phase=sym:always_false"`.
#[cfg(feature = "vex-engine-z3")]
fn extra_params_spec() -> &'static Vec<(String, ParamValue)> {
    static SPEC: std::sync::OnceLock<Vec<(String, ParamValue)>> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| {
        std::env::var("ANGR_Z3_PARAMS")
            .ok()
            .map(|s| parse_extra_params(&s))
            .unwrap_or_default()
    })
}

/// Pure parser for the `ANGR_Z3_PARAMS` spec — split out so it is unit-testable
/// without touching the process env / `OnceLock` cache.
#[cfg(feature = "vex-engine-z3")]
fn parse_extra_params(s: &str) -> Vec<(String, ParamValue)> {
    s.split(',')
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            let k = k.trim();
            let v = v.trim();
            if k.is_empty() {
                return None;
            }
            let val = if v.eq_ignore_ascii_case("true") {
                ParamValue::Bool(true)
            } else if v.eq_ignore_ascii_case("false") {
                ParamValue::Bool(false)
            } else if let Ok(n) = v.parse::<u32>() {
                ParamValue::U32(n)
            } else if let Some(sym) = v.strip_prefix("sym:") {
                if sym.is_empty() {
                    return None;
                }
                ParamValue::Symbol(sym.to_string())
            } else {
                return None;
            };
            Some((k.to_string(), val))
        })
        .collect()
}

#[cfg(feature = "vex-engine-z3")]
#[derive(Debug, Clone, PartialEq)]
enum ParamValue {
    Bool(bool),
    U32(u32),
    Symbol(String),
}

#[cfg(feature = "vex-engine-z3")]
fn apply_extra_params(params: &mut z3::Params) {
    for (key, val) in extra_params_spec() {
        match val {
            ParamValue::Bool(b) => params.set_bool(key.as_str(), *b),
            ParamValue::U32(n) => params.set_u32(key.as_str(), *n),
            ParamValue::Symbol(s) => params.set_symbol(key.as_str(), s.as_str()),
        }
    }
}

/// Parsed `ANGR_Z3_TACTIC` env-var spec, cached once per process.
///
/// Recognized values:
/// - unset / empty / "default" / "smt" → default `z3::Solver::new()`
///   (Z3's smt portfolio strategy).
/// - "qfbv_smart" → probe-conditional tactic that dispatches on
///   `num-consts`: large problems (> 20 constants) go to `qfbv`, small ones
///   stay on `smt`. Targets the bimodal benches (fairlight, sokohashv2,
///   mma_howtouse) where qfbv wins big, without regressing small-problem
///   benches like csgames2018 / flareon2015_2 where the qfbv preset's
///   per-check overhead dominates.
/// - any other value → colon-separated pipeline of tactic names
///   composed via `Tactic::and_then`, e.g.
///   `simplify:propagate-values:solve-eqs:bit-blast:sat`.
#[cfg(feature = "vex-engine-z3")]
#[derive(Debug, Clone)]
enum TacticSpec {
    Default,
    Pipeline(Vec<String>),
    QfbvSmart,
}

#[cfg(feature = "vex-engine-z3")]
fn tactic_spec() -> &'static TacticSpec {
    static SPEC: std::sync::OnceLock<TacticSpec> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| match std::env::var("ANGR_Z3_TACTIC") {
        Ok(s) => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("default") || s.eq_ignore_ascii_case("smt") {
                TacticSpec::Default
            } else if s.eq_ignore_ascii_case("qfbv_smart") {
                TacticSpec::QfbvSmart
            } else {
                let names: Vec<String> = s
                    .split(':')
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty())
                    .collect();
                if names.is_empty() {
                    TacticSpec::Default
                } else {
                    TacticSpec::Pipeline(names)
                }
            }
        }
        Err(_) => TacticSpec::Default,
    })
}

/// `num-consts > N` threshold used by `qfbv_smart`. Override via
/// `ANGR_Z3_QFBV_THRESHOLD`. 20 is the empirical pivot between bimodal
/// hash-cracker problems (winners) and CTF-style small-problem benches
/// (regressers) — see angr-ya00 measurements in
/// `docs/advanced-topics/rust_engine.rst`.
#[cfg(feature = "vex-engine-z3")]
fn qfbv_smart_threshold() -> f64 {
    static THRESHOLD: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("ANGR_Z3_QFBV_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(20.0)
    })
}

/// Construct a fresh Z3 solver per the active [`TacticSpec`], with timeout
/// and bv_rewriter params applied. Three sites in this file rely on this:
/// initial creation, lazy fork materialization, and `set_timeout`.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn build_solver(timeout_ms: u32) -> z3::Solver {
    let solver = match tactic_spec() {
        TacticSpec::Default => z3::Solver::new(),
        TacticSpec::Pipeline(names) => {
            let mut iter = names.iter();
            let first = z3::Tactic::new(iter.next().expect("non-empty pipeline"));
            let composed = iter.fold(first, |acc, name| acc.and_then(&z3::Tactic::new(name)));
            composed.solver()
        }
        TacticSpec::QfbvSmart => {
            let qfbv = z3::Tactic::new("qfbv");
            let smt = z3::Tactic::new("smt");
            let num_consts = z3::Probe::new("num-consts");
            let thresh = z3::Probe::constant(qfbv_smart_threshold());
            let cond = z3::Tactic::cond(&num_consts.gt(&thresh), &qfbv, &smt);
            cond.solver()
        }
    };
    solver.set_params(&build_solver_params(timeout_ms));
    solver
}

#[cfg(all(test, feature = "vex-engine-z3"))]
mod extra_params_tests {
    use super::{ParamValue, parse_extra_params};

    #[test]
    fn parses_bool_and_uint_pairs() {
        let got = parse_extra_params("bv.size_reduce=true, relevancy=0 ,timeout=500");
        assert_eq!(
            got,
            vec![
                ("bv.size_reduce".to_string(), ParamValue::Bool(true)),
                ("relevancy".to_string(), ParamValue::U32(0)),
                ("timeout".to_string(), ParamValue::U32(500)),
            ]
        );
    }

    #[test]
    fn skips_malformed_and_empty_keys() {
        // no '=', empty key, empty value, and an untagged non-numeric
        // non-bool value (a likely typo, e.g. `timeout=5oo`) are all
        // dropped as a harmless no-op — same as before ParamValue::Symbol
        // existed. Only the explicit `sym:` tag opts into symbol parsing
        // (see `parses_tagged_symbol_values`); a bare unparseable value
        // must NOT silently become a symbol param, since that previously
        // corrupted the solver for mistyped bool/u32 params like `timeout`
        // or `relevancy` (angr-sijyb.1 peer review).
        let got = parse_extra_params("noequals,=v,bad=notanum,empty=,ok=false");
        assert_eq!(got, vec![("ok".to_string(), ParamValue::Bool(false))]);
    }

    #[test]
    fn parses_tagged_symbol_values() {
        let got = parse_extra_params("sat.phase=sym:always_false");
        assert_eq!(
            got,
            vec![(
                "sat.phase".to_string(),
                ParamValue::Symbol("always_false".to_string())
            )]
        );
    }

    #[test]
    fn empty_sym_tag_is_dropped() {
        assert!(parse_extra_params("sat.phase=sym:").is_empty());
    }

    #[test]
    fn untagged_typo_on_known_numeric_param_is_a_noop_not_a_symbol() {
        // Regression test for the exact corruption mode peer review found:
        // a mistyped numeric override must be dropped, not reinterpreted
        // as a symbol param.
        let got = parse_extra_params("timeout=5oo,relevancy=abc,bv_extract_prop=notabool");
        assert!(got.is_empty());
    }

    #[test]
    fn empty_spec_yields_no_params() {
        assert!(parse_extra_params("").is_empty());
    }
}
