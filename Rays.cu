// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Rays.h>

#include <Meta/CUDA.h>
#include <Utils/Random.h>
#include <Utils/ToString.h>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <ostream>
#include <iomanip>


inline float2 RandomFloat2() {
    float x = (float)rand() / (float)RAND_MAX;
    float y = (float)rand() / (float)RAND_MAX;
    return make_float2(x, y);
}

__constant__ int d_width;
__constant__ int d_height;
__constant__ int d_sqrtSamples;
__constant__ float3 d_cx;
__constant__ float3 d_cy;

struct DeviceCreateRays {

    DeviceCreateRays(const int width, const int height, const int sqrtSamples,
                     const float3& cx, const float3& cy) {
        cudaMemcpyToSymbol(d_width, &width, sizeof(int));
        cudaMemcpyToSymbol(d_height, &height, sizeof(int));
        cudaMemcpyToSymbol(d_sqrtSamples, &sqrtSamples, sizeof(int));
        cudaMemcpyToSymbol(d_cx, &cx, sizeof(float3));
        cudaMemcpyToSymbol(d_cy, &cy, sizeof(float3));
    }
    
    __host__ __device__
    thrust::tuple<float4, float4> operator()(const float2 rand, const unsigned int index) {
        // Reverse: int index = (x + y * width) * samples + subX + subY * sqrtSamples;
        const unsigned short subX = index % d_sqrtSamples;
        const unsigned short subY = (index / d_sqrtSamples) % d_sqrtSamples;
        const int lowResIndex = index / (d_sqrtSamples * d_sqrtSamples);
        const unsigned short x = lowResIndex % d_width;
        const unsigned short y = lowResIndex / d_width;
    
        const float3 camOrigin = make_float3(50.0f, 52.0f, 295.6f);
        const float3 camDir = make_float3(0.0f, -0.042612f,-1.0f);

        const float r1 = 2.0f * rand.x;
        const float dx = r1 < 1.0f ? sqrt(r1) - 1.0f : 1.0f - sqrt(2.0f - r1);
        const float r2 = 2.0f * rand.y;
        const float dy = r2 < 1.0f ? sqrt(r2) - 1.0f : 1.0f - sqrt(2.0f - r2);
        
        const float3 rayDir = d_cx * (((subX + 0.5f + dx) / d_sqrtSamples + x) / d_width - 0.5f) 
            + d_cy * (((subY + 0.5f + dy) / d_sqrtSamples + y) / d_height - 0.5f) + camDir;
        
        
        const float3 rayOrigin = camOrigin + 130.0f * rayDir;

        return thrust::tuple<float4, float4>(make_float4(rayOrigin, index),
                                             make_float4(normalize(rayDir), 0.0f));
    }
};

// TODO Adopted from a linearly indexed thrust operator. Could probably be speed
// up as a 2D kernel.
__global__
void CreateRaysKernel(const int width, const int height, const int sqrtSamples,
                      const float3 cx, const float3 cy,
                      const unsigned int seed,
                      float4* origins, float4* directions) {

    const unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= width * height * sqrtSamples * sqrtSamples) return;

    // Reverse: int index = (x + y * width) * samples + subX + subY * sqrtSamples;
    const unsigned short subX = index % sqrtSamples;
    const unsigned short subY = (index / sqrtSamples) % sqrtSamples;
    const int lowResIndex = index / (sqrtSamples * sqrtSamples);
    const unsigned short x = lowResIndex % width;
    const unsigned short y = lowResIndex / width;
    
    const float3 camOrigin = make_float3(50.0f, 52.0f, 295.6f);
    const float3 camDir = make_float3(0.0f, -0.042612f,-1.0f);

    Random rand = Random::Create1D(seed);

    const float r1 = 2.0f * rand.NextFloat01();
    const float dx = r1 < 1.0f ? sqrt(r1) - 1.0f : 1.0f - sqrt(2.0f - r1);
    const float r2 = 2.0f * rand.NextFloat01();
    const float dy = r2 < 1.0f ? sqrt(r2) - 1.0f : 1.0f - sqrt(2.0f - r2);
    
    const float3 rayDir = cx * (((subX + 0.5f + dx) / sqrtSamples + x) / width - 0.5f) 
        + cy * (((subY + 0.5f + dy) / sqrtSamples + y) / height - 0.5f) + camDir;
    
    const float3 rayOrigin = camOrigin + 130.0f * rayDir;
    origins[index] = make_float4(rayOrigin, index);
    directions[index] = make_float4(normalize(rayDir), 0.0f);
}

Rays::Rays(const int width, const int height, const int sqrtSamples) {
    const int size = width * height * sqrtSamples * sqrtSamples;
    origins = thrust::device_vector<float4>(size);
    axisUVs = thrust::device_vector<float4>(size);

    const float3 camDir = make_float3(0.0f, -0.042612f,-1.0f);
    
    const float3 cx = make_float3(width * 0.5135f / height, 0, 0);
    const float3 cy = normalize(cross(cx, camDir)) * 0.5135f;

    bool useHostRandom = false;
    if (useHostRandom) {
        // Generate random numbers
        thrust::host_vector<float2> host_random(size);
        thrust::generate(host_random.begin(), host_random.end(), RandomFloat2);
        thrust::device_vector<float2> random = host_random;
        
        DeviceCreateRays deviceCreateRays(width, height, sqrtSamples, cx, cy);
        thrust::transform(random.begin(), random.end(), thrust::counting_iterator<unsigned int>(0), 
                          Begin(), deviceCreateRays);
    } else {
        struct cudaFuncAttributes funcAttr;
        cudaFuncGetAttributes(&funcAttr, CreateRaysKernel);
        unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        unsigned int blocks = (size / blocksize) + 1;
        std::cout << "CreateRaysKernel<<<" << blocks << ", " << blocksize << ">>>" << std::endl;
        CreateRaysKernel<<<blocks, blocksize>>>(width, height, sqrtSamples, cx, cy, rand(),
                                                RawPointer(origins), RawPointer(axisUVs));
        CHECK_FOR_CUDA_ERROR();
    }

    representation = RayRepresentation;

    std::cout << "Rays:" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << i << ": " << GetAsRay(i+size/2) << std::endl;
}

struct RaysToHyperRays {
    __host__ __device__
    inline float4 operator()(const float4& direction) const {
        const float3 axisUV = HyperRay::DirectionToAxisUV(make_float3(direction));
        return make_float4(axisUV, direction.w);
    }
};

struct HyperRaysToRays {
    __host__ __device__
    inline float4 operator()(const float4& axisUV) const {
        const float3 dir = normalize(HyperRay::AxisUVToDirection(make_float3(axisUV)));
        return make_float4(dir, axisUV.w);
    }
};

void Rays::Convert(const Representation r) {
    if (representation == r) return;

    if (Size() > 0) {
        // std::cout << "Convert from " << representation << " to " << r << std::endl;
        switch(representation) {
        case RayRepresentation: 
            switch (r) {
            case HyperRayRepresentation:
                thrust::transform(axisUVs.begin(), axisUVs.end(), axisUVs.begin(), RaysToHyperRays());
                break;
            default:
                std::cout << "No conversion from " << representation << " to " << r << std::endl;
                return;
            }
            break;
        case HyperRayRepresentation:
            switch (r) {
            case RayRepresentation:
                thrust::transform(axisUVs.begin(), axisUVs.end(), axisUVs.begin(), HyperRaysToRays());
                break;
            default:
                std::cout << "No conversion from " << representation << " to " << r << std::endl;
                return;
            }
            break;
        default:
            std::cout << "Converting from " << representation << " not supported." << std::endl;
            return;
        }
    }

    // std::cout << "Converted from " << representation << " to " << r << std::endl;
    
    representation = r;
}

std::string Rays::ToString() const {
    std::ostringstream out;
    out << "HyperRays";
    if (representation == RayRepresentation)
        for (size_t i = 0; i < Size(); ++i)
            out << "\n" << GetAsRay(i);
    
    else if (representation == HyperRayRepresentation)
        for (size_t i = 0; i < Size(); ++i)
            out << "\n" << GetAsHyperRay(i);
    return out.str();
}
