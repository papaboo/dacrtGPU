// GPU Dacrt utils
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_UTILS_H_
#define _GPU_DACRT_UTILS_H_

#include <thrust/device_vector.h>

#include <iostream>
#include <cstdlib>

#include <Enums.h>

typedef thrust::device_vector<bool>::iterator BoolIterator;
typedef thrust::device_vector<char>::iterator CharIterator;
typedef thrust::device_vector<int>::iterator IntIterator;
typedef thrust::device_vector<unsigned int>::iterator UintIterator;
typedef thrust::device_vector<uint2>::iterator Uint2Iterator;
typedef thrust::device_vector<float>::iterator FloatIterator;
typedef thrust::device_vector<float2>::iterator Float2Iterator;
typedef thrust::device_vector<float4>::iterator Float4Iterator;
typedef thrust::device_vector<Axis>::iterator AxisIterator;
typedef thrust::device_vector<PartitionSide>::iterator PartitionSideIterator;

template <class T>
__host__ __device__
inline T Max(const T lhs, const T rhs) {
    return lhs < rhs ? rhs : lhs;
}

template <class T>
__host__ __device__
inline T Min(const T lhs, const T rhs) {
    return lhs > rhs ? rhs : lhs;
}

inline float Rand01() {
    return (float)rand() / (float)RAND_MAX;
}

template <class T>
inline T Clamp01(T v) { 
    return v < T(0) ? T(0) : v > T(1) ? T(1) : v;
}

template <class T>
inline T* RawPointer(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
// template <class T>
// inline T* Raw_pointer(typename thrust::device_vector<T>::iterator v) {
//     return thrust::raw_pointer_cast(&*v);
// }
inline uint2* RawPointer(typename thrust::device_vector<uint2>::iterator v) {
    return thrust::raw_pointer_cast(&*v);
}


inline int ToByte(float v) {
    return int(pow(Clamp01(v),1/2.2f)*255.0f+.5f);
}

inline void SavePPM(const std::string path, thrust::device_vector<float4>& colors, const int width, const int height) {
    thrust::host_vector<float4> cs = colors;

    FILE *f = fopen(path.c_str(), "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i<width*height; i++) {
        float4 c = cs[i];
        fprintf(f,"%d %d %d ", ToByte(c.x), ToByte(c.y), ToByte(c.z));
    }
}

#endif // _GPU_DACRT_UTILS_H_
