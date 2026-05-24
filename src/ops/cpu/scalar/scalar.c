#include "scalar.h"

DEFINE_BINARY_KERNEL(scalar_add_fp32,float,+)

DEFINE_BINARY_KERNEL(scalar_mul_fp32,float,*)

DEFINE_BINARY_KERNEL(scalar_sub_fp32,float,-)

DEFINE_BINARY_KERNEL(scalar_div_fp32,float,/)