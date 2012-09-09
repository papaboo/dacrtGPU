// Kernel for reducing the min and max of a list of morton codes
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Kernels/ReduceMinMaxMortonCode.h>

#include <Meta/CUDA.h>
#include <Utils/Morton.h>
#include <Utils/ToString.h>

#include <iostream>
#include <stdexcept>

using Utils::Morton;

namespace Kernels {
    
    __device__ 
    inline void PerformMinMaxMortonByAxisReduction(const unsigned int index, 
                                                   volatile unsigned int* min, volatile unsigned int* max,
                                                   MortonBound* bounds) {
        // Load data from global mem into registers.
        const MortonCode lhsMin = min[index * 2];
        const MortonCode rhsMin = min[index * 2 + 1];
        const MortonCode lhsMax = max[index * 2];
        const MortonCode rhsMax = max[index * 2 + 1];
        // Since the result is stored in another shared mem entry, the threads
        // need to be synced between fetching and storing.
        __syncthreads();
        
        const unsigned int axis = lhsMin & 0xE0000000;
        if (axis == (rhsMin & 0xE0000000)) { // Compare axis
            // Both have the same owner, so store the result back in shared
            // memory.
            min[index] = Morton::MinBy4(lhsMin.WithoutAxis(), rhsMin.WithoutAxis()) + axis;
            max[index] = Morton::MaxBy4(lhsMax.WithoutAxis(), rhsMax.WithoutAxis()) + axis;
        } else {
            // Different owners, so store the lhs in global memory and rhs in
            // shared.
            min[index] = rhsMin;
            max[index] = rhsMax;
        
            const unsigned int globalIndex = lhsMin.GetAxis();
            const MortonBound old = bounds[globalIndex];
            const unsigned int min = Morton::MinBy4(lhsMin.WithoutAxis(), old.min.WithoutAxis()) + axis;
            const unsigned int max = Morton::MaxBy4(lhsMax.WithoutAxis(), old.max.WithoutAxis()) + axis;
            bounds[globalIndex] = MortonBound::Create(min, max);
        }

        __syncthreads();
    }

    __global__
    void ReduceMinMaxMortonByAxisPass1(const unsigned int* const mortonCodes,
                                       const size_t inputSize,
                                       MortonBound* intermediateBounds,
                                       MortonBound* bounds,
                                       const size_t boundsSize) {

        __shared__ volatile unsigned int min[256];
        __shared__ volatile unsigned int max[256];
    
        const size_t beginIndex = inputSize * blockIdx.x / gridDim.x;
        const size_t endIndex = inputSize * (blockIdx.x + 1) / gridDim.x;
    
        size_t currentIndex = beginIndex + threadIdx.x;

        // Fill initial values
        unsigned int lookupIndex = Min(currentIndex, endIndex-1); // Ensures that we pad with the last element
        min[threadIdx.x] = max[threadIdx.x] = MortonCode(mortonCodes[lookupIndex]);
        currentIndex += blockDim.x;
        __syncthreads();
    
        // While still values left, load them and perform reduction
        while (currentIndex < endIndex) {
            // Fetch new data from global mem to shared mem
            lookupIndex = Min(currentIndex, endIndex-1); // Ensures that we pad with the last element
            min[blockDim.x + threadIdx.x] = max[blockDim.x + threadIdx.x] = MortonCode(mortonCodes[lookupIndex]);
            __syncthreads();

            PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);

            currentIndex += blockDim.x;
        }
        __syncthreads();

        // Reduce 128 values left in shared memory
        if (threadIdx.x >= 64) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);
        __syncthreads();

        if (threadIdx.x >= 32) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);

        if (threadIdx.x >= 16) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);

        if (threadIdx.x >= 8) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);

        if (threadIdx.x >= 4) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);
    
        if (threadIdx.x >= 2) return;
        // Store the last 2 values, since they may overlap with the first and last
        // value in neighbouring blocks.
        intermediateBounds[2 * blockIdx.x + threadIdx.x] = MortonBound::Create((unsigned int)min[threadIdx.x], 
                                                                               (unsigned int)max[threadIdx.x]);
    }

    __global__
    void ReduceMinMaxMortonByAxisPass2(MortonBound* intermediateBounds,
                                       const size_t intermediateBoundsSize,
                                       MortonBound* bounds,
                                       const size_t boundsSize) {
    
        __shared__ volatile unsigned int mins[128];
        __shared__ volatile unsigned int maxs[128];

        unsigned int index = Min((size_t)threadIdx.x, intermediateBoundsSize-1);
        MortonBound b = intermediateBounds[index];
        mins[threadIdx.x] = b.min;
        maxs[threadIdx.x] = b.max;
        __syncthreads();

        // Reduce 128 values left in shared memory
        if (threadIdx.x >= 64) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);
        __syncthreads();

        if (threadIdx.x >= 32) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);

        if (threadIdx.x >= 16) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);

        if (threadIdx.x >= 8) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);

        if (threadIdx.x >= 4) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);
    
        if (threadIdx.x >= 2) return;
        PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);
    
        // Reduce the last value, comparing it to what is already stored in bounds
        const MortonCode lhsMin = mins[0];
        const MortonCode lhsMax = maxs[0];
        const unsigned int axis = lhsMin.code & 0xE0000000;
        const unsigned int globalIndex = lhsMin.GetAxis();
        if (globalIndex != 6) { // If any dummy values (axis == 6) have been used,
            // then now is the time to discard them.
            const MortonBound old = bounds[globalIndex];
            const unsigned int min = Morton::MinBy4(lhsMin & 0x1FFFFFFF, old.min & 0x1FFFFFFF) + axis;
            const unsigned int max = Morton::MaxBy4(lhsMax & 0x1FFFFFFF, old.max & 0x1FFFFFFF) + axis;
            bounds[globalIndex] = MortonBound::Create(min, max);
        }
    }


    void ReduceMinMaxMortonByAxis(thrust::device_vector<unsigned int>::iterator mortonBegin,
                                  thrust::device_vector<unsigned int>::iterator mortonEnd,
                                  thrust::device_vector<MortonBound>::iterator boundsBegin,
                                  thrust::device_vector<MortonBound>::iterator boundsEnd) {

        // Verify that CUDA is initialized
        if (!Meta::CUDA::initialized)
            throw std::runtime_error("CUDA wasn't initialized. Can't lookup kernel properties");
    
        const size_t inputSize = mortonEnd - mortonBegin;
        const size_t boundsSize = boundsEnd - boundsBegin; // Always 6, but this looks less like magic.

        // Init bounds to default asshat values.
        MortonBound defaultBound = MortonBound::Create(0xDFFFFFFF, 0xC0000000);
        thrust::fill(boundsBegin, boundsEnd, defaultBound);
    
        // struct cudaFuncAttributes funcAttr;
        // cudaFuncGetAttributes(&funcAttr, ReduceMinMaxMortonByAxisPass1);
        // const unsigned int blockDim = funcAttr.maxThreadsPerBlock > 128 ? 128 : funcAttr.maxThreadsPerBlock;
        const unsigned int blockDim = 128;
        const unsigned int blocks = Meta::CUDA::activeCudaDevice.multiProcessorCount;

        static thrust::device_vector<MortonBound> intermediateBounds(blocks * 2);

        // std::cout << "MortonCode's:\n";
        // for (int i = 0; i < inputSize; ++i) {
        //     std::cout << i << ": " << (MortonCode)mortonBegin[i] << " - " << mortonBegin[i];
        //     if (i < inputSize-1)
        //         std::cout << "\n";
        // }
        // std::cout << "\n" << std::endl;
    
        ReduceMinMaxMortonByAxisPass1<<<blocks, blockDim>>>(RawPointer(mortonBegin), inputSize,
                                                            RawPointer(intermediateBounds),
                                                            thrust::raw_pointer_cast(&*boundsBegin), boundsSize);

        // std::cout << "intermediateBounds\n" << intermediateBounds << std::endl;
        // std::cout << std::endl;

        // std::cout << "bounds:\n";
        // for (int i = 0; i < boundsSize; ++i) {
        //     std::cout << i << ": " << boundsBegin[i];
        //     if (i < boundsSize-1)
        //         std::cout << "\n";
        // }
        // std::cout << "\n" << std::endl;
    
        ReduceMinMaxMortonByAxisPass2<<<1, 128>>>(RawPointer(intermediateBounds), intermediateBounds.size(),
                                                  thrust::raw_pointer_cast(&*boundsBegin), boundsSize);
    }

}
