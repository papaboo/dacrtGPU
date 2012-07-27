// CUDA
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_CUDA_H_
#define _GPU_DACRT_CUDA_H_

#include <cuda.h>

namespace Meta {

class CUDA {
public:
    static cudaDeviceProp activeCudaDevice;
    
    static void Initialize();
};

}

#endif // _GPU_DACRT_CUDA_H_
