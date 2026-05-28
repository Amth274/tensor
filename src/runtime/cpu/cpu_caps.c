#include "runtime.h"

CpuCaps g_cpu_caps = {0};

void detect_cpu_caps(void){

#if defined(__x86_64__) || defined(__i386__)

    g_cpu_caps.avx2 = __builtin_cpu_supports("avx2") != 0;

    g_cpu_caps.avx512 = __builtin_cpu_supports("avx512f") != 0;

    g_cpu_caps.fma = __builtin_cpu_supports("fma") != 0;

    g_cpu_caps.sse42 = __builtin_cpu_supports("sse4.2") != 0;

#endif
}
