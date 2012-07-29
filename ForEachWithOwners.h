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

#define LOCAL_POOL_SCALE 4
    
template<bool USE_GLOBAL_OWNER, bool BLOCK_DIM_LARGER_THAN_WARPSIZE, class Operation> __global__ 
void ForeachWithOwnerKernel(const uint2* partitions, const unsigned int partitionLength,
                            const unsigned int elements, 
                            Operation operation) {
    __shared__ volatile unsigned int localPoolNextIndex;
    __shared__ volatile unsigned int localPoolNextOwner;
    unsigned int localIndex, localOwner;
    
    if (!USE_GLOBAL_OWNER) localOwner = 0;
    if (BLOCK_DIM_LARGER_THAN_WARPSIZE) __syncthreads();
    
    while (true) {
        if (threadIdx.x == 0) {
            // Fetch new pool data TODO use a larger local pool, fx 3 * blockDim.
            localPoolNextIndex = atomicAdd(&d_globalPoolNextIndex, LOCAL_POOL_SCALE * blockDim.x);

            if (USE_GLOBAL_OWNER) {
                // TODO Paralize the fetch loop over shared data in parallel,
                // then let the 'winner' write it's id in a res var. \o/
                // (clover boy)
                localPoolNextOwner = d_globalPoolNextOwner;
                // uint partition;
                // do {
                //     partition = partitions[++localPoolNextOwner];
                //     // NOTE: I'm specifically avoiding using y (end) because I
                //     // will switch to unsigned int in a later version
                // } while (partition <= localPoolNextIndex);
                // d_globalPoolNextOwner = --localPoolNextOwner;
                
                uint2 partition = partitions[localPoolNextOwner];
                while (localPoolNextIndex >= partition.y)
                    partition = partitions[++localPoolNextOwner];
                d_globalPoolNextOwner = localPoolNextOwner;
            }
        }
        
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) __syncthreads();
        
        localIndex = localPoolNextIndex + threadIdx.x;
        if (localIndex >= elements) return; // terminate if we exceed the amount of indices

        if (USE_GLOBAL_OWNER) localOwner = localPoolNextOwner;
        
        // uint partition;
        // do {
        //     partition = partitions[++localOwner];
        // } while (partition <= localIndex);
        // --localOwner;

        // Manual freaking loop unrolling. Thanks nvcc
        uint2 partition = partitions[localOwner];
        while (localIndex >= partition.y)
            partition = partitions[++localOwner];
        // Perform logic
        operation(localIndex, localOwner);
        
        localIndex += blockDim.x;
        if (localIndex >= elements) return; // terminate if we exceed the amount of indices
        while (localIndex >= partition.y)
            partition = partitions[++localOwner];
        // Perform logic
        operation(localIndex, localOwner);

        localIndex += blockDim.x;
        if (localIndex >= elements) return; // terminate if we exceed the amount of indices
        while (localIndex >= partition.y)
            partition = partitions[++localOwner];
        // Perform logic
        operation(localIndex, localOwner);

        localIndex += blockDim.x;
        if (localIndex >= elements) return; // terminate if we exceed the amount of indices
        while (localIndex >= partition.y)
            partition = partitions[++localOwner];
        // Perform logic
        operation(localIndex, localOwner);
    }
}

template<class Operation>
void ForEachWithOwners(const size_t elements, // replace with thrust counting iterator?
                       thrust::device_vector<uint2>::iterator partitionsBegin, thrust::device_vector<uint2>::iterator partitionsEnd, 
                       Operation& operation) {

    // std::cout << "ForEachWithOwners " << std::endl;
    // std::cout << "From " << partitionBegin << " to " << partitionEnd << std::endl;
    // std::cout << "Partitions:\n" << partitions << std::endl;
    // std::cout << "elements: " << elements << std::endl;

    // Maybe add a __constant__ that is zero to the GPU and copy from that?
    // Saves a bit of bus. Or do it as a zero-out kernel.
    const unsigned int zero = 0;
    cudaMemcpyToSymbol(d_globalPoolNextOwner, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_globalPoolNextIndex, &zero, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    const size_t partitionLength = partitionsEnd - partitionsBegin;
    
    const unsigned int blocks = Meta::CUDA::activeCudaDevice.multiProcessorCount;
    const bool USE_GLOBAL_OWNER = false; // depends on some partitionLength : elements ratio
    
    struct cudaFuncAttributes funcAttr;
    if (USE_GLOBAL_OWNER)
        cudaFuncGetAttributes(&funcAttr, ForeachWithOwnerKernel<true, true, Operation>);
    else
        cudaFuncGetAttributes(&funcAttr, ForeachWithOwnerKernel<false, true, Operation>);
    const unsigned int blockDim = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    
    const bool BLOCK_DIM_LARGER_THAN_WARPSIZE = blockDim > Meta::CUDA::activeCudaDevice.warpSize;
    // std::cout << "ForEachWithOwner<" << USE_GLOBAL_OWNER << ", " << BLOCK_DIM_LARGER_THAN_WARPSIZE << ">" <<
    //     "<<<" << blocks << ", " << blockDim << ">>>" << std::endl;
    
    if (USE_GLOBAL_OWNER) {
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) {
            ForeachWithOwnerKernel<true, true><<<blocks, blockDim>>>
                (RawPointer(partitionsBegin), partitionLength,
                 elements, operation);
        } else {
            ForeachWithOwnerKernel<true, false><<<blocks, blockDim>>>
                (RawPointer(partitionsBegin), partitionLength,
                 elements, operation);
        }
    } else {
        if (BLOCK_DIM_LARGER_THAN_WARPSIZE) {
            ForeachWithOwnerKernel<false, true><<<blocks, blockDim>>>
                (RawPointer(partitionsBegin), partitionLength,
                 elements, operation);
        } else {
            ForeachWithOwnerKernel<false, false><<<blocks, blockDim>>>
                (RawPointer(partitionsBegin), partitionLength,
                 elements, operation);
        }
    }
    CHECK_FOR_CUDA_ERROR();
}

#endif
