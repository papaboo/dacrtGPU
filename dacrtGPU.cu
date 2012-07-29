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
#include <HyperCube.h>
#include <Meta/CUDA.h>
#include <Shading.h>
#include <SphereContainer.h>
#include <SphereGeometry.h>
#include <ToString.h>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/version.h>

using std::cout;
using std::endl;

//const int WIDTH = 8, HEIGHT = 8;
//const int WIDTH = 32, HEIGHT = 32;
//const int WIDTH = 64, HEIGHT = 64;
//const int WIDTH = 128, HEIGHT = 128;
const int WIDTH = 256, HEIGHT = 256;
//const int WIDTH = 512, HEIGHT = 512;
int sqrtSamples;
int samples;

// Sort rays directly, not indices to avoid non-coallesced memory access (DONE)

// Can we remove owners from rays and spheres? For the final coloring, the
// owners can be created with just one scan. Try it out and then perhaps add a
// work queue instead for calculating splitting sides.

// Don't do full leaf moved left/right arrays. Just do it for the nodes and then
// calculate the ray/sphere left/right leaf position after that using their
// index. This means we can do our inclusive_scan's over the nodes instead or
// rays and spheres. Can the same be done for non leaf partitions?

// Work queue idea: Update the pointer to the next work 'pool' and do it as an
// atomic operation. Then update the current ray owner afterwards. This can
// either be done atomic or not. In any case if it isn't done at the exact same
// time as the work index, the threads can simply iterate upwards until they find
// the correct dacrtnode which owns the ray (this will need to be done anyway)
/// RESULT
// It seems that scan + operation is faster than a workqueue. Given everyones
// obsession with work queues this may just be a problem with my old GPU, but I
// need to test it further. The test case can be found in DacrtNode

// TODO

// Can we do ray partitioning without converting them to hyper rays? 
// - Sort them by major axis.
// - Extend all ray dirs, so their major axis has length 1.0f? Then the split
// - will never be performed along that axis.
// - Weigh all 6 dimensions when computing the bounding cone/plane split. (Needs
//   to be done anyway with respect to the scenes AABB for evenly distributed
//   spatial and angular split decisions) Then simply weigh the major axis angle
//   as infinite to discourage its use. (Or use specialized kernels for each
//   major axis, not that hard)
// - Proceed as usual with plane creation and intersection (but now without the
//   constant conversion trough a switch)

// The left/right indice arrays also encode the left/right side info. Can we use
// this instead of the PartitionSide arrays to save memory? Will it be just as fast?

// Partition rays based on the geom bounds aswell? Perhaps do one or the other
// depending on some ray/geom ratio or based on how much a ray cone overlaps
// with a geometry bounding volume (e.g aabb)

// Try different ray partitionings. The easiest is just a raw left-right
// partition, but one that partitions rays spatially may perform better. (i.e
// place the two children of a parent ray partition next to each other in
// memory)
/// If I do this then I need to do the same for dacrtnodes, so the nodes still
/// follow an increasing partition layout.

// Move randomly generated numbers to the GPU

// Amortise geometry sorting cost by using a morton curve subdivision (everyone
// else is anyway)

// Amortise ray sorting cost by storing them in pixelwide packets (packets are
// always crap at the N'th trace, can we dynamically sort them semi optimal?)

// When only a few rays remain, don't paralize intersection over all rays, but
// do it over geometry instead. (Not an issue as long as I'm doing my fixed bounce pathtracer)

void RayTrace(Fragments& rayFrags, SpheresGeometry& spheres) {
    RayContainer rays = RayContainer(WIDTH, HEIGHT, sqrtSamples);
    // cout << rays.ToString() << endl;

    DacrtNodes nodes = DacrtNodes(1);
    unsigned int bounce = 0;
    while (rays.InnerSize() > 0) {
        
        cout << "Rays this pass: " << rays.InnerSize() << endl;
        
        // Partition rays according to their major axis
        uint rayPartitionStart[7];
        rays.PartitionByAxis(rayPartitionStart);
        
        cout << "ray partitions: ";
        for (int p = 0; p < 7; ++p)
            cout << rayPartitionStart[p] << ", ";
        cout << endl;

        thrust::device_vector<uint2> rayPartitions(6);
        int activePartitions = 0;
        for (int a = 0; a < 6; ++a) {
            const size_t rayCount = rayPartitionStart[a+1] - rayPartitionStart[a];
            rayPartitions[a] = make_uint2(rayPartitionStart[a], rayPartitionStart[a+1]);
            activePartitions += rayCount > 0 ? 1 : 0;
        }

        // Reduce the cube bounds
        HyperCubes cubes = HyperCubes(128);
        cubes.ReduceCubes(rays.BeginInnerRays(), rays.EndInnerRays(), 
                          rayPartitions, activePartitions);
        cout << cubes << endl;

        uint spherePartitionStart[activePartitions+1];
        SphereContainer sphereIndices(cubes, spheres, spherePartitionStart);

        cout << "sphere partitions: ";
        for (int p = 0; p < activePartitions+1; ++p)
            cout << spherePartitionStart[p] << ", ";
        cout << endl;

        nodes.Reset();
        int nodeIndex = 0;
        for (int a = 0; a < 6; ++a) {
            const int rayStart = rayPartitionStart[a];
            const size_t rayEnd = rayPartitionStart[a+1];
            if (rayStart == rayEnd) continue;

            const int sphereStart = spherePartitionStart[nodeIndex];
            const size_t sphereEnd = spherePartitionStart[nodeIndex+1];
            nodes.SetUnfinished(nodeIndex, DacrtNode(rayStart, rayEnd, sphereStart, sphereEnd));
            ++nodeIndex;
        }

        unsigned int i = 0;
        while (nodes.UnfinishedNodes() > 0) {
            //cout << "\n *** PARTITION NODES (" << bounce<< ", " << i << ") ***\n" << nodes/*.ToString(rays, sphereIndices)*/ << "\n ***\n" << endl;
            nodes.Partition(rays, sphereIndices, cubes);
            //cout << "\n *** PARTITION LEAFS (" << bounce<< ", " << i << ") ***\n" << nodes/*.ToString(rays, sphereIndices)*/ << "\n ***\n" << endl;
            if (nodes.PartitionLeafs(rays, sphereIndices))
                ;//cout << "\n *** AFTER PARTITIONING (" << bounce<< ", " << i << ") ***\n" << nodes/*.ToString(rays, sphereIndices)*/ << "\n ***\n" << endl;
            else
                ;//cout << "\n *** NO LEAFS CREATED (" << bounce<< ", " << i << ") ***\n" << endl;
            
            if (nodes.UnfinishedNodes() > 0) {
                // Prepare cubes for next round.
                cubes.ReduceCubes(rays.BeginInnerRays(), rays.EndInnerRays(), 
                                  nodes.rayPartitions, nodes.UnfinishedNodes());
                // cout << cubes << endl;
            }

            ++i;
        }

        if (rays.LeafRays() == 0) {
            cout << "No leafs to shade. What went wrong?" << endl;
            return;
        }

        static thrust::device_vector<unsigned int> hitIDs(rays.LeafRays());
        nodes.ExhaustiveIntersect(rays, sphereIndices, hitIDs);

        //cout << "hitIDs: " << hitIDs << endl;

        Shading::Shade(rays.BeginLeafRays(), rays.EndLeafRays(), hitIDs.begin(), 
                       sphereIndices.spheres, rayFrags);

        rays.RemoveTerminated(hitIDs);

        // rays.Clear();
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

    std::cout.setf(std::ios::unitbuf); // unbuffered std out
    cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl;
        
    sqrtSamples = argc >= 2 ? atoi(argv[1]) : 1;
    samples = sqrtSamples * sqrtSamples; // # of samples
    int iterations = argc >= 3 ? atoi(argv[2]) : 1; // # iterations

    Fragments frags(WIDTH * HEIGHT * samples);
    thrust::device_vector<float4> colors(WIDTH * HEIGHT);

    SpheresGeometry geom = SpheresGeometry::CornellBox(50);
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
