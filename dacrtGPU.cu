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
#include <Fragment.h>
#include <RayContainer.h>
#include <MortonDacrtNode.h>
#include <Meta/CUDA.h>
#include <Shading.h>
#include <SphereContainer.h>
#include <SphereGeometry.h>
#include <Utils/Morton.h>
#include <Utils/ToString.h>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/version.h>

using std::cout;
using std::endl;

//const int WIDTH = 8, HEIGHT = 8;
const int WIDTH = 16, HEIGHT = 16;
//const int WIDTH = 32, HEIGHT = 32;
//const int WIDTH = 64, HEIGHT = 64;
//const int WIDTH = 128, HEIGHT = 128;
//const int WIDTH = 256, HEIGHT = 256;
int sqrtSamples;
int samples;

void RayTrace(Fragments& rayFrags, SpheresGeometry& spheres) {
    RayContainer rays = RayContainer(WIDTH, HEIGHT, sqrtSamples);

    MortonDacrtNodes mNodes = MortonDacrtNodes(1);
    mNodes.Create(rays, spheres);
    exit(0);
    
    DacrtNodes nodes = DacrtNodes(1);
    unsigned int bounce = 0;
    while (rays.InnerSize() > 0) {
        
        cout << "Rays this pass: " << rays.InnerSize() << endl;

        nodes.Create(rays, spheres);

        static thrust::device_vector<unsigned int> hitIDs(rays.LeafRays());
        nodes.ExhaustiveIntersect(rays, *(nodes.GetSphereIndices()), hitIDs);

        Shading::Normals(rays.BeginLeafRays(), rays.EndLeafRays(), hitIDs.begin(), 
                         nodes.GetSphereIndices()->spheres, rayFrags);

        rays.RemoveTerminated(hitIDs);

        ++bounce;
    }
}

__constant__ int d_samples;
__constant__ float d_mod;
template <bool COMBINE>
struct FragsToColor {
    float4* emissionDepth;
    
    FragsToColor(thrust::device_vector<float4>& ed, const int samples, const float mod)
        : emissionDepth(thrust::raw_pointer_cast(ed.data())) {
        cudaMemcpyToSymbol(d_samples, (void*)&samples, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_mod, (void*)&mod, sizeof(float), 0, cudaMemcpyHostToDevice);
    }

    __device__
    float4 operator()(const float4 color, const unsigned int threadId) {
        float3 eSum = make_float3(0.0f, 0.0f, 0.0f);
        for (unsigned int e = threadId * d_samples; e < threadId * d_samples + d_samples; ++e)
            eSum += make_float3(emissionDepth[e]);
        
        eSum /= d_samples;
        return COMBINE ?
            color * d_mod + make_float4(eSum.x, eSum.y, eSum.z, 0.0f)  *(1.0f -  d_mod) :
            make_float4(eSum.x, eSum.y, eSum.z, threadId);
    }
};

void CombineFragsAndColor(Fragments& frags,
                          thrust::device_vector<float4>& colors, 
                          const int samples, const float mod = 0.0f) {
    if (mod == 0.0f) {
        FragsToColor<false> fragsToColor(frags.emissionDepth, samples, mod);
        thrust::transform(colors.begin(), colors.end(), thrust::counting_iterator<unsigned int>(0), colors.begin(), fragsToColor);
    } else {
        FragsToColor<true> fragsToColor(frags.emissionDepth, samples, mod);
        thrust::transform(colors.begin(), colors.end(), thrust::counting_iterator<unsigned int>(0), colors.begin(), fragsToColor);
    }
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

    SpheresGeometry geom = SpheresGeometry::CornellBox(500);
    // cout << geom << endl;

    for (int i = 0; i < iterations; ++i) {
        // Reset fragments on subsequent iterations
        if (i > 0) frags.Reset();
        
        RayTrace(frags, geom);
        
        float mod = float(i) / (i+1.0f);
        CombineFragsAndColor(frags, colors, samples, mod);
    }

    SavePPM("planeimage.ppm", colors, WIDTH, HEIGHT);

    return 0;
}
