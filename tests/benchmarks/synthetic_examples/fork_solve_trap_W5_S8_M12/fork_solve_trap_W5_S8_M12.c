/* Parameterised wide-AND-slow fork/solve benchmark WITH A PYTHON-BOUNCE TRAP.
 *
 * This is the ``fork_solve_template.c`` base (W independent branches -> 2^W
 * leaves, S rounds of nonlinear mixing, a partial-mask find gate) PLUS a
 * ``__TRAP__`` region: T sequential calls to a non-inlined ``trap_point()``
 * placed AFTER the mixing region, so all 2^W leaves reach every trap call.
 *
 * ``solve.py`` hooks ``trap_point`` with an identity ``SimProcedure``. Under
 * the Rust parallel wave loop, a state hitting a Python SimProcedure/hook is a
 * *bounce*: it repopulates STASH_ACTIVE and forces the level-synchronous wave
 * loop to detach + reattach (full Z3-AST serde round-trip via
 * StateMigrationPayload) the whole active frontier at workers>1. T sequential
 * trap call sites => ~T full-frontier re-migrations of the 2^W-wide frontier.
 * That is the migration-dominated regression this bench is built to expose
 * (angr-nkoct): fast-ish single-threaded, SLOWER at workers>1.
 *
 * Knobs substituted at build time by ``build_trap.py``:
 *   __W__ (width)  -> __BRANCHES__ expanded to W ``if (b[i]!=0)`` statements.
 *   __S__ (solve)  -> __MIX__ expanded to S nonlinear mixing statements.
 *   __T__ (trap)   -> __TRAP__ expanded to T ``acc = trap_point(acc);`` calls.
 *   __M__ (mask)   -> __GATE__ substituted with ``(acc & MASK) == PATTERN``.
 *
 * ``reach_target`` is a distinct, non-inlined symbol used as the find address.
 */
#include <unistd.h>

static volatile unsigned int g_sink;

/* Distinct find target. volatile sink keeps -O0 from folding it away. */
void reach_target(void) {
    g_sink = 0xC0FFEEu;
}

/* Non-inlined identity. solve.py hooks this symbol with an identity
 * SimProcedure; every call is a Python bounce that re-enters the manager and
 * (at workers>1) re-migrates the whole active frontier. Keeping it identity
 * preserves the symbolic accumulator so the find-gate Z3 check stays hard. */
__attribute__((noinline)) unsigned int trap_point(unsigned int x) {
    return x;
}

int main(void) {
    unsigned char b[32];
    if (read(0, b, 32) != 32) return 1;

    /* Width region: W independent symbolic branches -> up to 2^W leaves. */
    unsigned int s = 0;
    if (b[0] != 0) s += 1u;
    if (b[1] != 0) s += 2u;
    if (b[2] != 0) s += 4u;
    if (b[3] != 0) s += 8u;
    if (b[4] != 0) s += 16u;

    /* Per-state solve cost: S rounds of nonlinear mixing over the input bytes,
     * seeded by the branch outcome so each leaf carries a distinct hard
     * constraint. Unrolled (straight-line) to keep the dispatch count low. */
    unsigned int acc = s + 0x1234567u;
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[0];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[1];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[2];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[3];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[4];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[5];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[6];
    acc = (acc * 1103515245u + 12345u) ^ (acc >> 3); acc += b[7];

    /* Python-bounce trap: T sequential identity calls hit by all 2^W leaves.
     * Each is a Python SimProcedure bounce -> full-frontier re-migration at
     * workers>1. This is the migration-dominated cost the bench manufactures. */
    acc = trap_point(acc);
    acc = trap_point(acc);
    acc = trap_point(acc);
    acc = trap_point(acc);

    /* Partial-mask match: reaching reach_target requires the low M bits of the
     * mixed accumulator to equal a fixed pattern. A partial match over a
     * well-mixing function is essentially always satisfiable, so every leaf's
     * find-address satisfiable() check does real but *bounded* solver work.
     * __GATE__ is substituted with `(acc & MASK) == PATTERN`. */
    if ((acc & 0xfffu) == 0xfeeu) reach_target();
    return (int)acc;
}
