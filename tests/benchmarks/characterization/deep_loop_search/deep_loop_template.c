/* Parameterised deep-loop binary for search-strategy characterisation.
 *
 * Grub-proxy: reads N+1 stdin bytes one at a time, each non-CR byte
 * appended to a buffer (with backspace handling), CR terminates and
 * triggers the "vulnerable" exit. Depth-to-target is N+1 (N taps + CR).
 *
 * The build step substitutes `__N__` with the desired iteration cap.
 * Every read() byte is fully symbolic, so each iteration forks into
 * at least two paths (CR-or-not), giving up to 2^N reachable states
 * by depth N. Mirrors the grub passphrase-loop topology without grub's
 * I/O / SimProc surface.
 */
#include <unistd.h>

#define MAX_ITERS __N__

int main(void) {
    char buf[MAX_ITERS];
    int len = 0;
    for (int i = 0; i < MAX_ITERS; i++) {
        char c;
        if (read(0, &c, 1) != 1) return 1;
        if (c == '\r' || c == '\n') {
            /* "vulnerable" path: depth = iteration count + 1 */
            return 42;
        }
        if (c == 0x08 /* BS */ && len > 0) {
            len--;
            continue;
        }
        if (len < MAX_ITERS) {
            buf[len++] = c;
        }
    }
    return 0;
}
