#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "scalar.h"
#include "avx2.h"
#include "timer.h"

#define ITERS 100

typedef void (*kernel_fn)(
    void*,
    const void*,
    const void*,
    uint32_t
);

typedef struct {
    const char* name;

    kernel_fn scalar;
    kernel_fn avx2;
} bench_op_t;

static void init_arrays(
    float* a,
    float* b,
    uint32_t n
)
{
    for(uint32_t i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = 2.0f;
    }
}

static double benchmark_kernel(
    kernel_fn fn,
    float* out,
    float* a,
    float* b,
    uint32_t numel
)
{
    double start = now_ms();

    for(int i = 0; i < ITERS; i++) {
        fn(out, a, b, numel);
    }

    double end = now_ms();

    return end - start;
}

int main(void)
{
    system("mkdir -p bench/results");

    FILE* csv =
        fopen(
            "bench/results/kernel_bench.csv",
            "w"
        );

    if(csv == NULL) {
        fprintf(stderr, "failed to open csv file\n");
        return 1;
    }

    fprintf(
        csv,
        "numel,op,scalar_ms,avx2_ms,speedup\n"
    );

    bench_op_t ops[] = {
        {
            "add",
            scalar_add_fp32,
            avx2_add_fp32
        },
        {
            "sub",
            scalar_sub_fp32,
            avx2_sub_fp32
        },
        {
            "mul",
            scalar_mul_fp32,
            avx2_mul_fp32
        },
        {
            "div",
            scalar_div_fp32,
            avx2_div_fp32
        }
    };

    const size_t num_ops =
        sizeof(ops) / sizeof(ops[0]);

    for(
        uint32_t numel = 64;
        numel <= (1 << 24);
        numel *= 2
    )
    {
        float* a =
            aligned_alloc(
                32,
                numel * sizeof(float)
            );

        float* b =
            aligned_alloc(
                32,
                numel * sizeof(float)
            );

        float* out =
            aligned_alloc(
                32,
                numel * sizeof(float)
            );

        if(!a || !b || !out) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }

        init_arrays(a, b, numel);

        for(size_t op = 0; op < num_ops; op++) {

            // --------------------------------
            // Warmup
            // --------------------------------

            ops[op].avx2(
                out,
                a,
                b,
                numel
            );

            // --------------------------------
            // Scalar benchmark
            // --------------------------------

            double scalar_ms =
                benchmark_kernel(
                    ops[op].scalar,
                    out,
                    a,
                    b,
                    numel
                );

            // --------------------------------
            // AVX2 benchmark
            // --------------------------------

            double avx2_ms =
                benchmark_kernel(
                    ops[op].avx2,
                    out,
                    a,
                    b,
                    numel
                );

            double speedup =
                scalar_ms / avx2_ms;

            printf(
                "%8u | %-4s | scalar=%8.3f ms | avx2=%8.3f ms | speedup=%5.2fx\n",
                numel,
                ops[op].name,
                scalar_ms,
                avx2_ms,
                speedup
            );

            fprintf(
                csv,
                "%u,%s,%.6f,%.6f,%.6f\n",
                numel,
                ops[op].name,
                scalar_ms,
                avx2_ms,
                speedup
            );
        }

        free(a);
        free(b);
        free(out);
    }

    fclose(csv);

    printf(
        "\nResults written to:\n"
        "bench/results/kernel_bench.csv\n"
    );

    return 0;
}