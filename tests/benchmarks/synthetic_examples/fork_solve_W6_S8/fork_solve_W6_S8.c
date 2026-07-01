/* Parameterised wide-AND-slow fork/solve benchmark (angr parallel-symex GO gate).
 *
 * Two knobs are substituted at build time by ``build.py``:
 *
 *   __W__ (width)  -> W independent ``if (b[i] != 0)`` branches at the top, so a
 *                     fully symbolic 32-byte stdin produces up to 2^W leaf states
 *                     that BFS keeps active *at the same time* (sustained frontier
 *                     width). __BRANCHES__ is expanded to W such statements.
 *
 *   __S__ (solve)  -> S rounds of a nonlinear symbolic mixing function, unrolled
 *                     straight-line (no loop) so the mixing costs few dispatched
 *                     steps but builds a deep, hash-like constraint. The closing
 *                     ``if (acc == TARGET) reach_target();`` then forces an
 *                     expensive Z3 ``satisfiable()`` at both the fork and the
 *                     find-address check for every one of the 2^W leaves.
 *                     __MIX__ is expanded to S mixing statements.
 *
 * The combination is the only shape a state-level worker pool can exploit:
 * many independent active states (width) each carrying a hard solver check
 * (per-state cost). Memory is structurally bounded (<= 2^W leaves; cost lives
 * in constraint *depth*, not state count), so W <= 6 stays well under 3 GB.
 *
 * ``reach_target`` is a distinct, non-inlined symbol used as the find address.
 */
#include <unistd.h>

static volatile unsigned int g_sink;

/* Distinct find target. volatile sink keeps -O0 from folding it away. */
void reach_target(void) {
    g_sink = 0xC0FFEEu;
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
    if (b[5] != 0) s += 32u;

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

    /* Partial-mask match: reaching reach_target requires the low M bits of the
     * mixed accumulator to equal a fixed pattern. A full-word equality would
     * force Z3 to prove UNSAT for the many leaves whose branch pattern cannot
     * hit an exact constant (unpredictably slow — seconds per leaf). A partial
     * match over a well-mixing function is essentially always satisfiable, so
     * every leaf's find-address satisfiable() check does real but *bounded*
     * solver work whose cost scales with the mixing depth S and the mask width
     * M. __GATE__ is substituted with `(acc & MASK) == PATTERN`. */
    if ((acc & 0xfffffu) == 0xffeeu) reach_target();
    return (int)acc;
}
