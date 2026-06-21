/* Write-heavy cle-stream characterisation binary (angr-csyy9).
 *
 * Each iteration issues several stdio writes against the cle-loaded
 * `stdout` / `stderr` FILE* externs (NOT a fopen'd file). Under the Rust
 * engine the native write-side SimProcedures (fputs in puts.rs, fputc /
 * fwrite in stdio.rs) call read_fileno_for_stream, which hits an unmapped
 * lazy page for the cle stdout/stderr FILE* and returns
 * ProcedureError::Memory -> the dispatcher falls back to Python (measured
 * ~2.8ms per call, NOT the ~100ms the superseded
 * write-side-fileno-fallback-correct memory estimated). The fallback is
 * CORRECT (Python resolves the right fd) but slower, so this loop
 * deliberately amplifies the per-call cost into a measurable regression.
 * See bd memory write-stream-heavy-bench.
 *
 * The loop bound is concrete, so execution stays on a single non-forking
 * path: the bench measures per-write fallback cost, not fork scaling.
 *
 * Built -O0 -no-pie x86_64 (matches the other synthetic_examples binaries),
 * dynamically linked so stdout/stderr are real cle externs.
 */
#include <stdio.h>

#define ITERS 48

int main(void) {
    for (int i = 0; i < ITERS; i++) {
        fputs("write_stream_heavy: stdout line\n", stdout);
        fputc('!', stdout);
        fwrite("chunk", 1, 5, stdout);
        fputs("write_stream_heavy: stderr line\n", stderr);
    }
    return 0;
}
