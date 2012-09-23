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

#include <Enums.h>
#include <Utils/Math.h>

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
inline T* RawPointer(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
template <class T>
inline T* RawPointer(typename thrust::detail::normal_iterator<thrust::device_ptr<T> > v) {
    return thrust::raw_pointer_cast(&*v);
}

__host__ __device__ inline unsigned int ReverseBits(unsigned int x) {
#ifdef __CUDA_ARCH__
    return __brev(x);
#else
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16));
#endif    
}

__host__ __device__ inline int FirstBitSet(const int n) {
#ifdef __CUDA_ARCH__
    return __ffs(n);
#else
    return ffs(n);
#endif    
}

__host__ __device__ inline int LastBitSet(const int n) {
    return FirstBitSet(ReverseBits(n));
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
