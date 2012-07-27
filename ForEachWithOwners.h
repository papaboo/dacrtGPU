// For each kernel with owners
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _FOR_EACH_WITH_OWNERS_H_
#define _FOR_EACH_WITH_OWNERS_H_

#include <Meta/CUDA.h>

__device__ unsigned int d_globalPoolNextOwner;
__device__ unsigned int d_globalPoolNextIndex;
const unsigned int BATCH_SIZE = 32;
    
template<bool USE_GLOBAL_OWNER, bool BLOCK_DIM_LARGER_THAN_WARPSIZE, class Operation> __global__ 
void ForeachWithOwnerKernel(const uint2* partitions, const unsigned int partitionLength,
                            const unsigned int elements, 
                            Operation operation) {
    __shared__ volatile unsigned int localPoolNextIndex;
    __shared__ volatile unsigned int localPoolNextOwner;
    unsigned int localIndex, localOwner;
    
    if (!USE_GLOBAL_OWNER) localPoolNextOwner = 0;
    if (BLOCK_DIM_LARGER_THAN_WARPSIZE) __syncthreads();
    
    while (true) {
        if (threadIdx.x == 0) {
            // Fetch new pool data TODO use a larger local pool, fx 3 * blockDim.
            localPoolNextIndex = atomicAdd(&d_globalPoolNextIndex, blockDim.x);

            if (USE_GLOBAL_OWNER) {
                // TODO Paralize the fetch and loop over it in thread 0. Or
                // perhaps loop over it in parallel and only let the 'winner'
                // write it's id in a res var. \o/ (clever boy)
                localPoolNextOwner = d_globalPoolNextOwner;
                uint2 partition;
                do {
                    partition = partitions[++localPoolNextOwner];
                    // NOTE: I'm specifically avoiding using y (end) because I
                    // will switch to unsigned int in a later version
                } while (partition.x <= localPoolNextIndex);
                d_globalPoolNextOwner = --localPoolNextOwner;        
            }
        }
        
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) __syncthreads();
        
        localIndex = localPoolNextIndex + threadIdx.x;
        localOwner = localPoolNextOwner;

        if (localIndex >= elements) // terminate if we exceed the amount of indices
            return;
        
        uint2 partition;
        do {
            partition = partitions[++localOwner];
        } while (partition.x <= localIndex);
        --localOwner;
        
        // Perform logic
        operation(localIndex, localOwner);
    }
}


template<class Operation>
void ForEachWithOwners(thrust::device_vector<uint2> partitions, size_t partitionBegin, size_t partitionEnd,
                       size_t elements, // replace with thrust counting iterator?
                       Operation& operation) {

    // Maybe add a __constant__ that is zero to the GPU and copy from that?
    // Saves a bit of bus. Or do it as a zero-out kernel.
    const unsigned int zero = 0;
    cudaMemcpyToSymbol(d_globalPoolNextOwner, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_globalPoolNextIndex, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    const size_t partitionLength = partitionEnd - partitionBegin;
    
    const unsigned int blocks = Meta::CUDA::activeCudaDevice.multiProcessorCount;
    const bool USE_GLOBAL_OWNER = false; // depends on some partitionLength : elements ratio
    
    struct cudaFuncAttributes funcAttr;
    if (USE_GLOBAL_OWNER)
        cudaFuncGetAttributes(&funcAttr, ForeachWithOwnerKernel<true, true, Operation>);
    else
        cudaFuncGetAttributes(&funcAttr, ForeachWithOwnerKernel<false, true, Operation>);
    const unsigned int blockDim = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    
    const bool BLOCK_DIM_LARGER_THAN_WARPSIZE = blockDim > Meta::CUDA::activeCudaDevice.warpSize;
    std::cout << "ForEachWithOwner<<<" << blocks << ", " << blockDim << ">>>" << std::endl;
    if (USE_GLOBAL_OWNER) {
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) {
            ForeachWithOwnerKernel<true, true><<<blocks, blockDim>>>
                (thrust::raw_pointer_cast(partitions.data()) + partitionBegin, partitionLength,
                 elements, operation);
        } else {
            ForeachWithOwnerKernel<true, false><<<blocks, blockDim>>>
                (thrust::raw_pointer_cast(partitions.data()) + partitionBegin, partitionLength,
                 elements, operation);
        }
    } else {
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) {
            ForeachWithOwnerKernel<false, true><<<blocks, blockDim>>>
                (thrust::raw_pointer_cast(partitions.data()) + partitionBegin, partitionLength,
                 elements, operation);
        } else {
            ForeachWithOwnerKernel<false, false><<<blocks, blockDim>>>
                (thrust::raw_pointer_cast(partitions.data()) + partitionBegin, partitionLength,
                 elements, operation);
        }
    }
    
};

#endif
