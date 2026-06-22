/* Synthetic ctype/fprintf micro-bench for the native-coverage gate
 * (angr-11djq.20).
 *
 * Real command-line utilities exercise a small cluster of libc helpers that
 * the rest of the CTF-heavy baseline corpus never touches: fprintf (stdout /
 * stderr formatted output) and the locale ctype classifier table behind
 * isdigit/isalpha (__ctype_b_loc). Without a fixture that drives these on a
 * symbolic path, the bench gate only reaches fprintf incidentally via
 * sharif7_rev50 and never touches __ctype_b_loc at all — so the native
 * coverage beads (angr-tx7ec.4 ctype, angr-884yn fprintf) ship with unit
 * tests but no end-to-end gate.
 *
 * Design: four symbolic stdin bytes flow through an isdigit short-circuit
 * chain. The chain forks to five leaf states (four early not-a-digit returns,
 * each printing via fprintf(stderr), plus one all-digits ``win`` printing via
 * fprintf(stdout)), small enough to stay well under the 4 GB regression cap
 * and finish in well under a second.
 *
 * The driver runs from ``main`` with ``blank_state`` and pre-stocks stdin (the
 * cow_fork_scaling pattern), so no libc startup runs: every isdigit/fprintf
 * call lands on angr's extern SimProcedure region, which lets the Rust engine
 * take its native fast path (a loaded-libc binary region would force the
 * Python fallback — see run_loop.rs is_in_binary).
 *
 * getopt is deliberately absent: angr ships no getopt SimProcedure (parity
 * wall, angr-tx7ec.5 / angr-ae54t), so it would run real glibc through the VEX
 * interpreter — slow under Python and divergent under Rust — for zero native
 * coverage.
 *
 * Build (checked-in prebuilt, so the gate needs no compiler at test time):
 *   gcc -O0 -no-pie -fno-stack-protector -o cli_ctype_fprintf cli_ctype_fprintf.c
 */
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

/* Distinct, never-inlined marker so solve.py can target a stable symbol
 * address for the all-digits path instead of a mid-function block. */
__attribute__((noinline)) void win(void) {
    fprintf(stdout, "ALL DIGITS\n");
}

int main(void) {
    char buf[8];
    if (read(0, buf, 4) != 4)
        return 1;

    for (int i = 0; i < 4; i++) {
        if (!isdigit((unsigned char)buf[i])) {
            fprintf(stderr, "byte %d is not a digit\n", i);
            return 3;
        }
    }
    win();
    return 0;
}
