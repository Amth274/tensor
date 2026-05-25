#define _POSIX_C_SOURCE 199309L

#include "timer.h"
#include <time.h>

double now_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return
        (double)ts.tv_sec  * 1000.0 +
        (double)ts.tv_nsec / 1000000.0;
}