// Dacrt GPU. A GPU ray tracer using a divide and conquer strategy instead of
// partitioning the geometry into a hierarchy.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

// Compile ./make
// Usage: ./build/DACRTPlane <samples> <iterations>

#include <DacrtNode.h>
#include <ExhaustiveIntersection.h>
#include <Fragment.h>
#include <MortonDacrtNode.h>
#include <Meta/CUDA.h>
#include <RayContainer.h>
#include <Shading.h>
#include <SphereGeometry.h>
#include <Utils/ToString.h>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/version.h>

using std::cout;
using std::endl;

//const int WIDTH = 4, HEIGHT = 4;
//const int WIDTH = 8, HEIGHT = 8;
//const int WIDTH = 16, HEIGHT = 16;
//const int WIDTH = 32, HEIGHT = 32;
//const int WIDTH = 64, HEIGHT = 64;
//const int WIDTH = 128, HEIGHT = 128;
//const int WIDTH = 256, HEIGHT = 256;
const int WIDTH = 512, HEIGHT = 512;
//const int WIDTH = 1440, HEIGHT = 900;
int sqrtSamples;
int samples;

void RayTrace(Fragments& rayFrags, SpheresGeometry& spheres) {
    RayContainer rays = RayContainer(WIDTH, HEIGHT, sqrtSamples);

    MortonDacrtNodes tracer = MortonDacrtNodes(1);
    //DacrtNodes tracer = DacrtNodes(1);
    //ExhaustiveIntersection tracer;
    unsigned int bounce = 0;
    while (rays.InnerSize() > 0) {
        
        cout << rays.InnerSize() << " rays for bounce " << (bounce+1) << endl;

        tracer.Create(rays, spheres);
        
        static thrust::device_vector<unsigned int> hitIDs(rays.LeafRays());
        hitIDs.resize(rays.LeafRays());
        tracer.FindIntersections(hitIDs);

        // cout << hitIDs << endl;

        Shading::Shade(rays, hitIDs.begin(), 
                       spheres, rayFrags);
 
        ++bounce;
    }
}

template <bool LERP>
__global__
void FragsToColorKernel(const float4* const emissions_bounces,
                        float4* colors,
                        const int samples,
                        const float invSamples,
                        const float modifier,
                        const int nColors) {

    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nColors) return;
    
    float3 eSum = make_float3(0.0f, 0.0f, 0.0f);
    for (int e = id * samples; e < id * samples + samples; ++e)
        // TODO this is highly non-coallesced, so should be optimized, but it's
        // only a fraction of the total ray tracing cost.
        eSum += make_float3(emissions_bounces[e]);
    eSum *= invSamples;
    
    if (LERP) {
        const float3 color = make_float3(colors[id]);
        eSum = lerp(eSum, color, modifier);
    } 

    colors[id] = make_float4(eSum.x, eSum.y, eSum.z, id);
}

void CombineFragsAndColor(Fragments& frags,
                          thrust::device_vector<float4>& colors, 
                          const int samples, const float mod = 0.0f) {

    struct cudaFuncAttributes funcAttr;
    if (mod == 0.0f) {
        cudaFuncGetAttributes(&funcAttr, FragsToColorKernel<false>);
        unsigned int blocksize = min(funcAttr.maxThreadsPerBlock, 256);
        unsigned int blocks = (colors.size() / blocksize) + 1;
        FragsToColorKernel<false><<<blocks, blocksize>>>
            (RawPointer(frags.emissionDepth), RawPointer(colors),
             samples, 1.0f / samples, mod, colors.size());
    } else {
        cudaFuncGetAttributes(&funcAttr, FragsToColorKernel<true>);
        unsigned int blocksize = min(funcAttr.maxThreadsPerBlock, 256);
        unsigned int blocks = (colors.size() / blocksize) + 1;
        FragsToColorKernel<true><<<blocks, blocksize>>>
            (RawPointer(frags.emissionDepth), RawPointer(colors),
             samples, 1.0f / samples, mod, colors.size());
    }
    CHECK_FOR_CUDA_ERROR();
}

int main(int argc, char *argv[]){

    Meta::CUDA::Initialize();

    cout.setf(std::ios::unitbuf); // unbuffered std out
    cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl;

    sqrtSamples = argc >= 2 ? atoi(argv[1]) : 1;
    samples = sqrtSamples * sqrtSamples; // # of samples
    int iterations = argc >= 3 ? atoi(argv[2]) : 1; // # iterations

    Fragments frags(WIDTH * HEIGHT * samples);
    thrust::device_vector<float4> colors(WIDTH * HEIGHT);

    SpheresGeometry geom = SpheresGeometry::CornellBox(150);
    // cout << geom << endl;

    for (int i = 0; i < iterations; ++i) {
        // Reset fragments on subsequent iterations
        if (i > 0) frags.Reset();

        cout << "PASS " << (i+1) << endl;        
        RayTrace(frags, geom);
        
        float mod = float(i) / (i+1.0f);
        CombineFragsAndColor(frags, colors, samples, mod);
    }

    SavePPM("image.ppm", colors, WIDTH, HEIGHT);

    return 0;
}
