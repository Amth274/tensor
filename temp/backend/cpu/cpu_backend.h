#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include <stdint.h>

/* Define the struct type */
typedef struct {
    int has_avx2;
    int has_avx;
} CPUFeatures;

/* Declare the global (NOT define it) */
extern CPUFeatures g_cpu_features;

/* Backend init */
void cpu_backend_init(void);

#endif