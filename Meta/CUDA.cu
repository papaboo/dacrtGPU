// CUDA
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Meta/CUDA.h>

#include <iostream>

// One of these contain cutGetMaxGflopsDeviceId. No idea which (no internet)
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace Meta {

cudaDeviceProp CUDA::activeCudaDevice;

void CUDA::Initialize() {
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) == cudaErrorNoDevice) {
        std::cout << "No CUDA capable device found" << std::endl;
        exit(0);
    }
 
    CUdevice device = cutGetMaxGflopsDeviceId();
    cudaSetDevice(device);
    // cudaGLSetGLDevice(device);

    cuInit(0);

    int version;
    cuDriverGetVersion(&version);
    cudaGetDeviceProperties(&activeCudaDevice, device);
    std::cout << "CUDA: version " << version/1000 << "." << version % 100 << ", using device " << std::string(activeCudaDevice.name) << std::endl;
   
}

}
