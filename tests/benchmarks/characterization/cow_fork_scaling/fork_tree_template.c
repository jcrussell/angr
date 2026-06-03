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
    /* __BRANCHES__ */
    return s;
}
