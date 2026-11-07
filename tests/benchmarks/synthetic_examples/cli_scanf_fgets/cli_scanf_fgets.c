/* Synthetic scanf/fgets stdin bench for the replay oracle (angr-vmi2l.342).
 *
 * No other gated bench drives the native scanf or fgets procedures end to
 * end, so a native proc that recorded the wrong bytes in posix.dumps(0)
 * (scanf storing raw bytes instead of the digits spelling the value, fgets
 * losing a line) passed every gate. Here a number read with scanf("%d") and
 * a line read with fgets steer to ``win``, and solve.py's test() replays the
 * found stdin on this binary and asserts the marker it prints.
 *
 * Build (checked-in prebuilt, so the gate needs no compiler at test time):
 *   gcc -O0 -no-pie -fno-stack-protector -o cli_scanf_fgets cli_scanf_fgets.c
 */
#include <stdio.h>
#include <string.h>

/* Never inlined, so solve.py can target its symbol address. */
__attribute__((noinline)) void win(void) {
    puts("SCANF FGETS OK");
}

int main(void) {
    int n;
    char line[16];
    if (scanf("%d", &n) != 1)
        return 1;
    if (fgets(line, sizeof line, stdin) == NULL)
        return 2;
    if (n != 1337)
        return 3;
    if (strncmp(line, "open", 4) != 0)
        return 4;
    win();
    return 0;
}
