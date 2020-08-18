#include "ssemath.h"
#include <cstdio>
#include <cmath>

void PrintQuat(float* quat, const char* str)
{
	printf("%10s: %f %f %f %f\n", str, quat[0], quat[1], quat[2], quat[3]);
}
