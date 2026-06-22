/* File-I/O open()/read() harness (angr-w5llj, evidence for angr-11djq.7).
 *
 * A single concrete path that open()s a fixed path, read()s its first
 * bytes, and branches on the content. The point is to exercise the
 * file-descriptor / pre-seeded-file path that angr-11djq.7 (RustPosixState
 * fd/file sync) is gated on: the solve.py pre-seeds "/data/secret.txt" via
 * state.fs.insert on the Python side, then runs the Rust engine and asks
 * whether the open()+read() of that pre-seeded file is served natively or
 * falls back to a Python syscall handler.
 *
 * The earlier measurement attempts were the WRONG layer:
 *   - write_stream_heavy exercises WRITE-side cle stdout/stderr stdio only;
 *   - xmllint_getenv reaches getenv before any file open();
 * both produced 0 syscall fd-sync fallbacks (bd memory
 * djq7-evidence-needs-file-io-harness). This kernel is the missing
 * READ-side, pre-seeded-file harness.
 *
 * Deliberately uses the raw open()/read()/close() syscalls (not stdio
 * FILE*), so the measured fallbacks are the fd-sync path itself, not
 * fopen/fread SimProcedures layered on top.
 *
 * Build (vendored, like fp_simd_kernel / write_stream_heavy):
 *   gcc -O2 -fno-stack-protector -no-pie -o file_read_kernel file_read_kernel.c
 */

#include <fcntl.h>
#include <unistd.h>

#define SECRET_PATH "/data/secret.txt"
#define MAGIC_LEN 5

int main(void) {
    char buf[32];

    int fd = open(SECRET_PATH, O_RDONLY);
    if (fd < 0)
        return 1; /* open failed: file not visible to this side */

    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n < MAGIC_LEN)
        return 2; /* short read: content not served */

    /* Success path is reachable only when the pre-seeded bytes are
     * visible: the magic prefix "MAGIC" must match. A one-way (Python ->
     * Rust missing) sync would leave these reads unconstrained and the
     * compare would fork instead of landing here deterministically. */
    if (buf[0] == 'M' && buf[1] == 'A' && buf[2] == 'G' &&
        buf[3] == 'I' && buf[4] == 'C')
        return 42; /* pre-seeded content visible */

    return 3; /* file present but content mismatched */
}
