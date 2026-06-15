/* Parameterised fork-tree binary for CoW scaling characterisation.
 *
 * The build step substitutes `__N__` with the desired branch count.
 * Each `if (b[i] != 0)` is an independent symbolic branch, so a fully
 * symbolic 16-byte stdin produces 2^N reachable leaf states.
 */
#include <unistd.h>

int main(void) {
    char b[16];
    if (read(0, b, 16) != 16) return 1;
    int s = 0;
    if (b[0] != 0) s += 1;
    if (b[1] != 0) s += 2;
    if (b[2] != 0) s += 4;
    if (b[3] != 0) s += 8;
    if (b[4] != 0) s += 16;
    if (b[5] != 0) s += 32;
    if (b[6] != 0) s += 64;
    if (b[7] != 0) s += 128;
    return s;
}
