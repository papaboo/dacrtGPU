// Kernel for reducing the min and max of a list of morton codes
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_KERNELS_MIN_MAX_MORTON_CODE_H_
#define _GPU_DACRT_KERNELS_MIN_MAX_MORTON_CODE_H_

#include <Primitives/MortonCode.h>

#include <thrust/device_vector.h>

namespace Kernels {
    
    void ReduceMinMaxMortonByAxis(thrust::device_vector<unsigned int>::iterator mortonBegin,
                                  thrust::device_vector<unsigned int>::iterator mortonEnd,
                                  thrust::device_vector<MortonBound>::iterator boundsBegin,
                                  thrust::device_vector<MortonBound>::iterator boundsEnd);
    
};

#endif _GPU_DACRT_KERNELS_MIN_MAX_MORTON_CODE_H_
