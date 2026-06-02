#include <math.h>
#include "cpu_ops.h"

int scalar_log_fp32(KernelCall* call)
{
    float* out = (float*)call->outputs[0];
    const float* a = (const float*)call->inputs[0];
    double base = call->node->scalar;
    double denom = log(base);
    uint32_t numel = (uint32_t)call->output_metas[0]->numel;

    for (uint32_t i = 0; i < numel; i++) {
        out[i] = (float)(log((double)a[i]) / denom);
    }

    return 0;
}
