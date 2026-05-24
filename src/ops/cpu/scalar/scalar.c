#include "scalar.h"
#include "macros.h"

DEFINE_BINARY_KERNEL(scalar_add_fp32,float,+)

DEFINE_BINARY_KERNEL(scalar_mul_fp32,float,*)

DEFINE_BINARY_KERNEL(scalar_sub_fp32,float,-)

DEFINE_BINARY_KERNEL(scalar_div_fp32,float,/)

DEFINE_BINARY_KERNEL(scalar_add_fp64,double,+)

DEFINE_BINARY_KERNEL(scalar_mul_fp64,double,*)

DEFINE_BINARY_KERNEL(scalar_sub_fp64,double,-)

DEFINE_BINARY_KERNEL(scalar_div_fp64,double,/)


