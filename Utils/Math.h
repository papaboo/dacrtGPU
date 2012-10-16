// GPU Dacrt math
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_MATH_H_
#define _GPU_DACRT_MATH_H_

#include <cstdlib>

#include <math_functions.h>
#include <Utils/helper_math.h> // Copied from <CUDA-DIR>/samples/common/inc/

template <class T>
__host__ __device__
inline T Max(const T lhs, const T rhs) {
    return lhs < rhs ? rhs : lhs;
}

template <>
__host__ __device__
inline float3 Max<float3>(const float3 lhs, const float3 rhs) {
    return make_float3(Max(lhs.x, rhs.x),
                       Max(lhs.y, rhs.y),
                       Max(lhs.z, rhs.z));
}

template <class T>
__host__ __device__
inline T Min(const T lhs, const T rhs) {
    return lhs > rhs ? rhs : lhs;
}

template <>
__host__ __device__
inline float3 Min<float3>(const float3 lhs, const float3 rhs) {
    return make_float3(Min(lhs.x, rhs.x),
                       Min(lhs.y, rhs.y),
                       Min(lhs.z, rhs.z));
}


inline float Rand01() {
    return (float)rand() / (float)RAND_MAX;
}

template <class T>
inline T Clamp01(T v) { 
    return v < T(0) ? T(0) : (v > T(1) ? T(1) : v);
}

#endif // _GPU_DACRT_MATH_H_
