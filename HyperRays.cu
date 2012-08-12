// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <HyperRays.h>

#include <Utils/ToString.h>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <ostream>
#include <iomanip>

std::string HyperRay::ToString() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "[id: " << id << ", origin: " << origin << ", dir: " << normalize(Direction()) << ", axis: " << axis << ", u: " << UV.x << ", v: " << UV.y << "]";
    return out.str();
}



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
                                             make_float4(HyperRay::DirectionToAxisUV(rayDir), 0.0f));
    }
};

HyperRays::HyperRays(const int width, const int height, const int sqrtSamples) {
    // Random can be replaced by a random seed in a global var plus some
    // permutation of block and thread x's and y's.
    ///  -- or --
    // use http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html

    const int size = width * height * sqrtSamples * sqrtSamples;
    origins = thrust::device_vector<float4>(size);
    axisUVs = thrust::device_vector<float4>(size);
    
    const float3 camOrigin = make_float3(50.0f, 52.0f, 295.6f);
    const float3 camDir = make_float3(0.0f, -0.042612f,-1.0f);
    
    const float3 cx = make_float3(width * 0.5135f / height, 0, 0);
    const float3 cy = normalize(cross(cx, camDir)) * 0.5135f;

    // Generate random numbers
    thrust::host_vector<float2> host_random(size);
    thrust::generate(host_random.begin(), host_random.end(), RandomFloat2);
    thrust::device_vector<float2> random = host_random;

    DeviceCreateRays deviceCreateRays(width, height, sqrtSamples, cx, cy);
    thrust::transform(random.begin(), random.end(), thrust::counting_iterator<unsigned int>(0), 
                      Begin(), deviceCreateRays);

    /*    
    for (int y = 0; y < height; y++){
        unsigned short Xi[3] = {0, 0, y*y*y};
        for (unsigned short x = 0; x < width; x++) {
            
            // subpixel grid
            for (int subY = 0; subY < sqrtSamples; ++subY)
                for (int subX = 0; subX < sqrtSamples; ++subX) {
                    // Samples
                    double r1 = 2 * erand48(Xi);
                    float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                    double r2 = 2 * erand48(Xi);
                    float dy = r2 < 1 ? sqrt(r2) - 1: 1 - sqrt(2 - r2);
                    
                    float3 rayDir = cx * (((subX + 0.5 + dx) / sqrtSamples + x) / width - 0.5) 
                        + cy * (((subY + 0.5 + dy) / sqrtSamples + y) / height - 0.5) + camDir;
                    unsigned int index = (x + y * width) * sqrtSamples * sqrtSamples + subX + subY * sqrtSamples;
                    
                    float3 origin = camOrigin + rayDir * 130;
                    origins[index] = make_float4(origin.x, origin.y, origin.z, index);
                    axisUVs[index] = make_float4(HyperRay::DirectionToAxisUV(rayDir), 0.0f);
                }
        }
    }
    */    
}

std::string HyperRays::ToString() const {
    std::ostringstream out;
    out << "HyperRays";
    for (size_t i = 0; i < Size(); ++i)
        out << "\n" << Get(i);
    return out.str();
}
