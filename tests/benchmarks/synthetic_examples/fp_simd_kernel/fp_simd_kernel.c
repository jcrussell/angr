/* FP/SIMD-heavy synthetic kernel (angr-amtxu).
 *
 * A single concrete, non-forking path that drives SSE *scalar* FP and
 * (auto-vectorized) *packed* SIMD VEX ops through the Rust interpreter.
 * The corpus is otherwise entirely integer/string-bound, so
 * vex_fallback_count / vecret_gsptr_fallback_count read ZERO everywhere
 * (bd memory benchmark-vecret-gsptr-corpus-zero) and the native FP/SIMD
 * interpreter path has no measured coverage. This fixture gives the
 * collector (collect_simproc_fallbacks.py) a workload that exercises it.
 *
 * Deliberately makes NO libc calls in the kernel (no printf) so the only
 * VEX ops between `main` entry and return are FP/SIMD arithmetic —
 * SimProcedure fallbacks cannot pollute the FP/SIMD measurement. The
 * result is folded into the exit code.
 *
 * Build (vendored, like write_stream_heavy):
 *   gcc -O3 -fno-stack-protector -no-pie -o fp_simd_kernel fp_simd_kernel.c
 * -O3 auto-vectorizes the loops to packed SSE (mulps/addps) and lowers
 * __builtin_sqrt to the sqrtsd instruction (no libm dependency).
 */

#define N 16

static double acc_d;
static float acc_f;

__attribute__((noinline)) static double scalar_fp(double x) {
    double s = 0.0;
    for (int i = 1; i <= 6; i++) {
        double t = x * (double)i;     /* mulsd */
        s += t / (double)(i + 1);     /* divsd + addsd */
        s = __builtin_sqrt(s + 1.0);  /* sqrtsd (no libm call) */
    }
    if (s > 3.0)                      /* ucomisd */
        s *= 2.0;
    return s;
}

__attribute__((noinline)) static float packed_simd(const float *a, const float *b) {
    float out[N];
    for (int i = 0; i < N; i++)
        out[i] = a[i] * b[i] + a[i]; /* vectorizable: mulps + addps */
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += out[i];               /* vectorizable reduction */
    return sum;
}

int main(void) {
    /* `volatile` seeds defeat -O3 constant-folding: without them GCC
     * computes scalar_fp() at compile time and the sqrtsd/divsd/mulsd
     * ops never reach the binary. */
    volatile double seed = 3.14159;
    volatile float fa = 0.5f, fb = 0.25f;
    float a[N], b[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * fa;
        b[i] = (float)(N - i) * fb;
    }
    acc_d = scalar_fp(seed);
    acc_f = packed_simd(a, b);
    int r = (int)(acc_d + (double)acc_f);
    return r & 0xff;
}
