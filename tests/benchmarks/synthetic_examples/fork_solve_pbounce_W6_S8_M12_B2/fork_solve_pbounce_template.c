/* Parameterised wide-AND-slow fork/solve benchmark with a PARTIAL Python-bounce.
 *
 * This is the sibling of ``fork_solve_trap_template.c`` (the FULL-bounce stress
 * bench, where every leaf bounces at every trap call). Here only a FRACTION
 * ``B = 2^-k`` of the ``2^W`` leaves bounce at each level; the rest keep
 * stepping worker-locally. That makes it the demonstrator for the steady-state
 * parallel loop (angr-nkoct): the majority of the frontier does real mixing
 * work IN PARALLEL while a minority is serviced through the Python callback,
 * and a bounce costs one materialize + re-inject instead of a full-frontier
 * detach/reattach re-seed. The full-bounce trap bench cannot show this — there
 * every leaf must reattach in the coordinator context every level.
 *
 * Key property (verified): after the width region ``s`` is a CONCRETE per-leaf
 * index (each ``if (b[i]!=0) s += 2^i`` forks on the symbolic byte but the add
 * is concrete), so gating a trap call on bits of ``s`` PARTITIONS the existing
 * ``2^W`` leaves with NO extra forking and NO extra Z3 solve — the gate is a
 * concrete comparison per path. Gating on ``b[i]`` directly would fork (the
 * bytes are symbolic); do NOT do that.
 *
 * Structure: interleaved LEVELS (unlike the trap bench's all-mix-then-all-trap
 * layout). Each of ``T`` levels does ``S/T`` rounds of nonlinear mixing (the
 * work non-bounced leaves overlap with Python service) then ONE gated trap
 * call. The tested leaf-index bit field rotates by level so the bouncing subset
 * differs each level (the win is not an artifact of one hot subset).
 *
 * Knobs substituted at build time by ``build_pbounce.py``:
 *   __W__ (width) -> __BRANCHES__ expanded to W ``if (b[i]!=0)`` statements.
 *   __S__ (solve) -> distributed across the T levels as __LEVELS__.
 *   __T__ (trap)  -> number of interleaved gated trap levels.
 *   __B__ (k)     -> bounce fraction 2^-k; the gate mask is (1<<k)-1.
 *   __M__ (mask)  -> find-gate width, substituted into __GATE__.
 */
#include <unistd.h>

static volatile unsigned int g_sink;

/* Distinct find target. volatile sink keeps -O0 from folding it away. */
void reach_target(void) {
    g_sink = 0xC0FFEEu;
}

/* Non-inlined identity. solve.py hooks this symbol with an identity
 * SimProcedure; a call is a Python bounce. Identity preserves the symbolic
 * accumulator so the find-gate Z3 check stays exactly as hard as un-trapped. */
__attribute__((noinline)) unsigned int trap_point(unsigned int x) {
    return x;
}

int main(void) {
    unsigned char b[32];
    if (read(0, b, 32) != 32) return 1;

    /* Width region: W independent symbolic branches -> up to 2^W leaves.
     * `s` is a CONCRETE leaf index on each path after this region. */
    unsigned int s = 0;
    /* __BRANCHES__ */

    unsigned int acc = s + 0x1234567u;

    /* Interleaved levels: (S/T mixing rounds) then a gated trap. The gate uses
     * CONCRETE bits of the leaf index `s`, so only 2^-k of the leaves bounce at
     * each level and the branch never forks. */
    /* __LEVELS__ */

    /* Partial-mask find gate (same as the base): reaching reach_target requires
     * the low M bits of the mixed accumulator to equal a fixed pattern — real
     * but bounded solver work per leaf. __GATE__ -> `(acc & MASK) == PATTERN`. */
    if (/* __GATE__ */ 0) reach_target();
    return (int)acc;
}
