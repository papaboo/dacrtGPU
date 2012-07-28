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

#include <iostream>
#include <stdexcept>

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 *  inspired by cutil.h: CUT_CHECK_ERROR
 */
inline void CHECK_FOR_CUDA_ERROR(const char* file, const int line) {
    cudaError_t errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess) {
        const char* errorString = cudaGetErrorString(errorCode);
        std::cout << "CUDA error: [file: " << file << ", line: " << line << ", error: " << errorString << "]" << std::endl;
        throw std::runtime_error("CUDA error: See console.");
    }
    errorCode = cudaThreadSynchronize();
    if (errorCode != cudaSuccess) {
        const char* errorString = cudaGetErrorString(errorCode);
        std::cout << "CUDA error: [file: " << file << ", line: " << line << ", error: " << errorString << "]" << std::endl;
        throw std::runtime_error("CUDA error: See console.");
    }
}

/**
 *  Checks for CUDA errors and throws an exception if
 *  an error was detected, is only available in debug mode.
 */
#define CHECK_FOR_CUDA_ERROR(); CHECK_FOR_CUDA_ERROR(__FILE__,__LINE__);

namespace Meta {

class CUDA {
public:
    static cudaDeviceProp activeCudaDevice;
    
    static void Initialize();
};

}

#endif // _GPU_DACRT_CUDA_H_
