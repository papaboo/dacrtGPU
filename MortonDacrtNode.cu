// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <MortonDacrtNode.h>

#include <DacrtNode.h>
#include <HyperCubes.h>
#include <Kernels/ReduceMinMaxMortonCode.h>
#include <Meta/CUDA.h>
#include <Primitives/AABB.h>
#include <Primitives/Cone.h>
#include <Primitives/HyperCube.h>
#include <Primitives/MortonCode.h>
#include <RayContainer.h>
#include <SphereContainer.h>
#include <SphereGeometry.h>
#include <Utils/ToString.h>

#include <iostream>

#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

using std::cout;
using std::endl;

MortonDacrtNodes::MortonDacrtNodes(const size_t capacity) 
    : rayPartitions(capacity), nextRayPartitions(capacity), 
      spherePartitions(capacity), nextSpherePartitions(capacity),
      leafNodes(0),
      rays(NULL), spheresGeom(NULL),
      sphereIndices(capacity), sphereIndexPartition(capacity),
      nextSphereIndices(capacity), nextSphereIndexPartition(capacity),
      leafSphereIndices(0) {}

// TODO Apply the reduced upper bounds to the next levels partition bounds, to
// avoid to many false positives (in case of cone intersection, this will
// reduce the area between the hypercube and the cone, where false positives
// occur)
/// This can be done by storing all 6 reduced bounds in constant memory,
/// MortonBound[6], and then performing min and max.

struct RayMortonCoder {
    
    // TODO Copy these fields to CUDA as constants (After I'm done doing host testing)
    AABB bound;
    float3 boundInvSize;
    RayMortonCoder(const AABB& bound)
        : bound(bound), boundInvSize(bound.InvertedSize()) {}

    __host__ __device__
    inline MortonCode operator()(const thrust::tuple<float4, float4> ray) {

        float3 origin = make_float3(thrust::get<0>(ray));
        const float3 direction = make_float3(thrust::get<1>(ray));
        
        float tHit;
        if (!bound.ClosestIntersection(origin, direction, tHit)) 
            // If the ray misses the scene bounds, then use an invalid axis to
            // sort the slacker ray into its own partition at the end of the
            // list.
            return MortonCode::EncodeAxis(6);
        
        // Advance ray that is outside the bounds to the edge of the bounds
        origin += Max(0.0f, tHit) * direction;

        const unsigned int xIndex = CreateXIndex(origin.x);
        MortonCode mortonCode = MortonCode::CodeFromIndex(xIndex, xOffset);
        const unsigned int yIndex = CreateYIndex(origin.y);
        mortonCode += MortonCode::CodeFromIndex(yIndex, yOffset);
        const unsigned int zIndex = CreateZIndex(origin.z);
        mortonCode += MortonCode::CodeFromIndex(zIndex, zOffset);

        const float3 axisUV = HyperRay::AxisUVFromDirection(direction);
        mortonCode += MortonCode::EncodeAxis(axisUV.x);
        const unsigned int uIndex = CreateUIndex(axisUV.y);
        mortonCode += MortonCode::CodeFromIndex(uIndex, uOffset);
        const unsigned int vIndex = CreateVIndex(axisUV.z);
        mortonCode += MortonCode::CodeFromIndex(vIndex, vOffset);
        
        return mortonCode;
    }
    
    __host__ __device__ inline unsigned int CreateXIndex(const float x) { return (x - bound.min.x) * boundInvSize.x * 63.999f; }
    __host__ __device__ inline unsigned int CreateYIndex(const float y) { return (y - bound.min.y) * boundInvSize.y * 31.999f; }
    __host__ __device__ inline unsigned int CreateZIndex(const float z) { return (z - bound.min.z) * boundInvSize.z * 63.999f; }
    __host__ __device__ inline unsigned int CreateUIndex(const float u) { return (u + 1.0f) * 31.999f; }
    __host__ __device__ inline unsigned int CreateVIndex(const float v) { return (v + 1.0f) * 31.999f; }
    
    __host__ __device__ inline float XMin(const unsigned int xIndex) const { return (float)xIndex / (63.999f * boundInvSize.x) + bound.min.x; }
    __host__ __device__ inline float XMax(const unsigned int xIndex) const { return XMin(xIndex+1.0f); }
    __host__ __device__ inline float YMin(const unsigned int yIndex) const { return (float)yIndex / (31.999f * boundInvSize.y) + bound.min.y; }
    __host__ __device__ inline float YMax(const unsigned int yIndex) const { return YMin(yIndex+1.0f); }
    __host__ __device__ inline float ZMin(const unsigned int zIndex) const { return (float)zIndex / (63.999f * boundInvSize.z) + bound.min.z; }
    __host__ __device__ inline float ZMax(const unsigned int zIndex) const { return ZMin(zIndex+1.0f); }

    __host__ __device__ inline float UMin(const unsigned int uIndex) const { return (float)uIndex / 31.999f - 1.0f; }
    __host__ __device__ inline float UMax(const unsigned int uIndex) const { return UMin(uIndex+1.0f); }
    __host__ __device__ inline float VMin(const unsigned int vIndex) const { return (float)vIndex / 31.999f - 1.0f; }
    __host__ __device__ inline float VMax(const unsigned int vIndex) const { return VMin(vIndex+1.0f); }
    
    __host__ __device__ inline HyperCube HyperCubeFromBound(MortonBound b) const {
        const SignedAxis axis = b.min.GetAxis();

        b.min = b.min.WithoutAxis();
        b.max = b.max.WithoutAxis();
        
        const float2 x = make_float2(XMin(MortonCode::IndexFromCode(b.min, xOffset)), 
                                     XMax(MortonCode::IndexFromCode(b.max, xOffset)));

        const float2 y = make_float2(YMin(MortonCode::IndexFromCode(b.min, yOffset)), 
                                     YMax(MortonCode::IndexFromCode(b.max, yOffset)));

        const float2 z = make_float2(ZMin(MortonCode::IndexFromCode(b.min, zOffset)), 
                                     ZMax(MortonCode::IndexFromCode(b.max, zOffset)));

        const float2 u = make_float2(UMin(MortonCode::IndexFromCode(b.min, uOffset)), 
                                     UMax(MortonCode::IndexFromCode(b.max, uOffset)));

        const float2 v = make_float2(VMin(MortonCode::IndexFromCode(b.min, vOffset)), 
                                     VMax(MortonCode::IndexFromCode(b.max, vOffset)));

        return HyperCube(axis, x, y, z, u, v);
    }
};

void TestMortonEncoding();

struct InvalidAxis {
    __host__ __device__
    inline bool operator()(const MortonBound b) const {
        return (b.min & 0xE0000000) >= 0xC0000000;
    }

    __host__ __device__
    inline bool operator()(const thrust::tuple<uint2, MortonBound> i) const {
        const uint2 partition = thrust::get<0>(i);
        return partition.x >= partition.y;
    }
};

struct MortonBoundToHyperCube {
    RayMortonCoder coder;
    MortonBoundToHyperCube(RayMortonCoder c) : coder(c) {}

    __host__ __device__    
    thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> operator()(const MortonBound mb) const {
        HyperCube hc = coder.HyperCubeFromBound(mb);
        return thrust::make_tuple(hc.a, hc.x, hc.y, hc.z, hc.u, hc.v);
    }
};

__global__
void FindInitialRayPartitions(const unsigned int* const rayMortonCodes,
                              const size_t codes,
                              uint2* rayPartitions) {

    if (threadIdx.x >= 6) return;
    
    __shared__ unsigned int pivots[12];

    pivots[0] = 0;
    pivots[11] = codes;

    
    if (threadIdx.x < 5) {
        size_t min = 0, max = codes;
        while (min < max) {
            const size_t mid = (min + max) / 2;
            MortonCode code = rayMortonCodes[mid];
            SignedAxis axis = code.GetAxis();
            min = axis <= threadIdx.x ? (mid+1) : min;
            max = axis <= threadIdx.x ? max : mid;
        }

        pivots[threadIdx.x * 2 + 1] = pivots[threadIdx.x * 2 + 2] = min;
    }
    __syncthreads();
    
    rayPartitions[threadIdx.x] = make_uint2(pivots[threadIdx.x * 2], pivots[threadIdx.x * 2 + 1]);
}

__global__
void FindNewRayPartitions(const uint2* const rayPartitions,
                          const unsigned int rayPartitionCount,
                          const unsigned int* const rayMortonCodes,
                          uint2* nextRayPartitions) {
    
    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (id >= rayPartitionCount) return;
    
    const uint2 partition = rayPartitions[id];
    const MortonCode min = rayMortonCodes[partition.x];
    const MortonCode max = rayMortonCodes[partition.y-1];
    
    const unsigned int diff = min.code ^ max.code;
    const int n = LastBitSet(diff) - 1;

    const unsigned int mask = 0XFFFFFFFF << (31-n);
    const MortonCode rightMin = max & mask;
    
    uint2 pivot = partition;
    while (pivot.x < pivot.y) {
        const size_t mid = (pivot.x + pivot.y) / 2;
        MortonCode value = rayMortonCodes[mid] & mask;
        pivot.x = value < rightMin ? (mid+1) : pivot.x;
        pivot.y = value < rightMin ? pivot.y : mid;
    }
    
    nextRayPartitions[id*2]   = make_uint2(partition.x, pivot.x);
    nextRayPartitions[id*2+1] = make_uint2(pivot.x, partition.y);
}

struct CreateBoundsFromPartitions {
    unsigned int* rayMortonCodes;

    CreateBoundsFromPartitions(thrust::device_vector<unsigned int>& codes)
        : rayMortonCodes(RawPointer(codes)) {}
    
    __host__ __device__
    MortonBound operator()(const uint2 rayPartition) const {
        const MortonCode min = rayMortonCodes[rayPartition.x];
        const MortonCode max = rayMortonCodes[rayPartition.y-1];
        
        return MortonBound::LowestCommonBound(min, max);
    }
};

struct CreateHyperCubesFromBounds {
    
    RayMortonCoder rayMortonCoder;
    CreateHyperCubesFromBounds(const RayMortonCoder& rMC)
        : rayMortonCoder(rMC) {}

    __host__ __device__
    thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> operator()(const MortonBound bound) const {
        HyperCube cube = rayMortonCoder.HyperCubeFromBound(bound);
        
        return thrust::tuple<SignedAxis, float2, float2, float2, float2, float2>
            (cube.a, cube.x, cube.y, cube.z, cube.u, cube.v);
    }
};

struct CreateCones {
    __host__ __device__
    Cone operator()(const thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> c) const {
        const HyperCube cube(thrust::get<0>(c), thrust::get<1>(c), thrust::get<2>(c),
                             thrust::get<3>(c), thrust::get<4>(c), thrust::get<5>(c));
        
        return Cone::FromCube(cube);
    }
};

__constant__ Cone d_cone;
struct CompareConeSphere {
    
    CompareConeSphere(thrust::device_vector<Cone>& cones, unsigned int index) {
        Cone* cone = thrust::raw_pointer_cast(cones.data()) + index;
        cudaMemcpyToSymbol(d_cone, (void*)cone, sizeof(Cone), 0, cudaMemcpyDeviceToDevice);
    }
    
    __device__
    bool operator()(const Sphere s) {
        return d_cone.DoesIntersect(s);//, d_invSinToAngle, d_cosToAngleSqr);
    }
};

/**
 * Spheres in a list of nodes 
 *   | NODE0 | NODE1 | ... |
 * are partitioned in the following way.
 *   | LEFT0 | RIGHT0 | LEFT1 | RIGHT1 | ... |
 *
 * This is done by parallelizing over all spheres, they can then access the
 * partition owning them and compute where they should store their moved left
 * and right flag. 
 *
 * Fx: A sphere that is the 4'th sphere in a partition containing 9 entries and
 * starting at entry 7 will store its left flag at 2 * 7 + 4 and its right flag
 * at 2 * 7 + 9 + 4. The start index is multiplied by 2 to make room for the
 * previous partitions that all take up twice as many entries, since they have
 * to store both left and right indices.
 */
__global__
void SpherePartitioningByConesKernel(const unsigned int* const sphereIndices,
                                     const unsigned int* const owners,
                                     const uint2* const partitionings,
                                     const Cone* const cones,
                                     const Sphere* const spheres,
                                     bool* movedLeftRight, // Result
                                     const unsigned int nActiveIndices,
                                     const unsigned int nLeafIndices,
                                     const unsigned int nLeafNodes) {
    
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nActiveIndices) return;
    id += nLeafIndices;

    const unsigned int sphereId = sphereIndices[id];
    const Sphere sphere = spheres[sphereId];
    
    const unsigned int owner = owners[id];
    const uint2 partitioning = partitionings[owner];

    const unsigned int offset = id - partitioning.x;

    const Cone leftCone = cones[(owner-nLeafNodes)*2];
    const unsigned int leftResIndex = 2 * (partitioning.x - nLeafIndices) + offset; // Simplify? Is nLeafIndices even necessary?
    movedLeftRight[leftResIndex] = leftCone.DoesIntersect(sphere);
    
    const Cone rightCone = cones[(owner-nLeafNodes)*2+1];
    const unsigned int rightResIndex = leftResIndex + (partitioning.y - partitioning.x); // Simplify?
    movedLeftRight[rightResIndex] = rightCone.DoesIntersect(sphere);
}

struct BoolToUint {
    __host__ __device__
    unsigned int operator()(const bool val) const { return val; }
};

__global__
void PartitionIndices(const unsigned int* const sphereIndices,
                      const unsigned int* const owners,
                      const uint2* const partitionings, // active partitionings
                      const bool* const moveLeftRight,
                      const unsigned int* const newEntries,
                      unsigned int* nextSphereIndices, // Result
                      unsigned int* nextOwners, // Result
                      const unsigned int nActiveIndices,
                      const unsigned int nLeafNodes,
                      const unsigned int nLeafIndices) {

    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nActiveIndices) return;

    const unsigned int sphereId = sphereIndices[id];

    const unsigned int owner = owners[id] - nLeafNodes;
    const uint2 partitioning = partitionings[owner];

    const unsigned int offset = (id + nLeafIndices) - partitioning.x;
    
    const unsigned int leftEntryIndex = 2 * (partitioning.x - nLeafIndices) + offset; // Simplify? Is NLeafIndices even necessary?
    if (moveLeftRight[leftEntryIndex]) {
        const unsigned int leftEntry = newEntries[leftEntryIndex];
        nextSphereIndices[leftEntry] = sphereId;
        nextOwners[leftEntry] = owner * 2 + nLeafNodes;
    }
    
    const unsigned int rightEntryIndex = leftEntryIndex + (partitioning.y - partitioning.x); // Simplify?
    if (moveLeftRight[rightEntryIndex]) {
        const unsigned int rightEntry = newEntries[rightEntryIndex];
        nextSphereIndices[rightEntry] = sphereId;
        nextOwners[rightEntry] = owner * 2 + 1 + nLeafNodes;
    }
}

struct SpherePartitioningByCones {
    Cone* cones;
    Sphere* spheres;
    SpherePartitioningByCones(thrust::device_vector<Cone>& cs, 
                              thrust::device_vector<Sphere>& ss)
        : cones(RawPointer(cs)),
          spheres(RawPointer(ss)) {}
    
    __device__
    PartitionSide operator()(const unsigned int sphereId, const unsigned int owner) const {
        const Sphere sphere = spheres[sphereId];
        
        const Cone leftCone = cones[owner * 2];
        PartitionSide left = leftCone.DoesIntersect(sphere) ? LEFT : NONE;
        
        const Cone rightCone = cones[owner * 2 + 1];
        return left | rightCone.DoesIntersect(sphere) ? RIGHT : NONE;
    }
};

struct PartitionSideToUint2 {
    __host__ __device__
    uint2 operator()(const PartitionSide side) const {
        return make_uint2(side & LEFT ? 1 : 0, 
                          side & RIGHT ? 1 : 0);
    }
};

__global__
void AddFinalLeftIndexToRightIndices(uint2* indices,
                                     const uint2* const indicesEnd) {

    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (indices + id > indicesEnd) return;

    const unsigned int leftTotal = indicesEnd[0].x;
    uint2 lrIndex = indices[id];
    lrIndex.y += leftTotal;
    indices[id] = lrIndex;
}

__global__
void CreateNextPartitionings(const uint2* const partitionings, // active partitionings
                             const unsigned int* const newEntries,
                             uint2* nextPartitionings, // result
                             const unsigned int nLeafIndices,
                             const unsigned int nActiveNodes) {

    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nActiveNodes) return;

    const uint2 partitioning = partitionings[id];
    const unsigned int leftEntry = (partitioning.x - nLeafIndices) * 2;
    const unsigned int leftBegin = newEntries[leftEntry] + nLeafIndices;
    const unsigned int rightEntry = leftEntry + (partitioning.y - partitioning.x);
    const unsigned int rightBegin = newEntries[rightEntry] + nLeafIndices;
    const unsigned int endEntry = rightEntry + (partitioning.y - partitioning.x);
    const unsigned int rightEnd = newEntries[endEntry] + nLeafIndices;
    
    nextPartitionings[id*2]   = make_uint2(leftBegin, rightBegin);
    nextPartitionings[id*2+1] = make_uint2(rightBegin, rightEnd);
}


void MortonDacrtNodes::Create(RayContainer& rayContainer, SpheresGeometry& spheres) {
    rays = &rayContainer;
    spheresGeom = &spheres;
    
    // TestMortonEncoding();
    
    // This should actually be of type MortonCode, but I leave it as unsigned
    // int so thrust can use radix sort.
    static thrust::device_vector<unsigned int> rayMortonCodes(rayContainer.InnerSize());
    rayMortonCodes.resize(rayContainer.InnerSize());

    RayMortonCoder rayMortonCoder(spheres.GetBounds());
    thrust::transform(rayContainer.BeginInnerRays(), rayContainer.EndInnerRays(), 
                      rayMortonCodes.begin(), rayMortonCoder);
    
    // cout << "ray morton codes:\n" << rayMortonCodes << endl;
    rayContainer.SortToLeaves(rayMortonCodes.begin(), rayMortonCodes.end());
        
    // Reduce the 5D bounds along each dimension. These can be computed from the
    // sorted rays' morton code
    static thrust::device_vector<MortonBound> bounds(6); bounds.resize(6);
    Kernels::ReduceMinMaxMortonByAxis(rayMortonCodes.begin(), rayMortonCodes.end(),
                                      bounds.begin(), bounds.end());
    //cout << "Bounds:\n" << bounds << endl;

    // Find ray partition pivots
    rayPartitions.resize(6);
    FindInitialRayPartitions<<<1,32>>>(RawPointer(rayMortonCodes), rayMortonCodes.size(),
                                       RawPointer(rayPartitions));
    //cout << "Initial ray partitions:\n" << rayPartitions << endl;

    // Cull inactive partitions and bounds. (Ode to C++ auto or so I hear...)
    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<uint2>::iterator, 
        thrust::device_vector<MortonBound>::iterator> > PartitionBoundIterator;

    PartitionBoundIterator partition_bound_begin = 
        make_zip_iterator(make_tuple(rayPartitions.begin(), bounds.begin()));
    PartitionBoundIterator partition_bound_end = thrust::remove_if(partition_bound_begin, partition_bound_begin+bounds.size(), InvalidAxis());
    bounds.resize(partition_bound_end - partition_bound_begin);
    rayPartitions.resize(bounds.size());
    // cout << "Bounds:\n" << bounds << endl;
    // cout << "rayPartitions:\n" << rayPartitions << endl;
    
    HyperCubes hCubes(bounds.size());
    thrust::transform(bounds.begin(), bounds.end(), 
                      hCubes.Begin(), MortonBoundToHyperCube(rayMortonCoder));
    // cout << hCubes << endl;

    InitSphereIndices(hCubes, spheres);
    // sphereIndices = new SphereContainer(hCubes, spheres, spherePartitionPivots);
    // std::cout << sphereIndices->ToString() << std::endl;

    // TODO Remove empty sphere partitions
    
    // Geometry partition
    
    leafNodes = leafSphereIndices = 0;

    // cout << "\n ** Initial nodes **\n" << ToString(true) << endl;

    int counter = 0;
    while (spherePartitions.size() - leafNodes > 0) {
        ++counter;
        cout << "\n *** ROUND " << counter << " - FIGHT ***\n" << endl;

        // Create new ray partitions through binary search
        const size_t activeNodes = rayPartitions.size() - leafNodes;
        const size_t nextActiveNodes = activeNodes * 2;
        const size_t nextPartitionSize = leafNodes + nextActiveNodes;
        //cout << "nActiveNodes: " << activeNodes << endl;
        //cout << "nNextActiveNodes: " << nextActiveNodes << endl;
        // cout << "nNextPartitionSize: " << nextPartitionSize << endl;

        nextRayPartitions.resize(nextPartitionSize);
        struct cudaFuncAttributes funcAttr;
        cudaFuncGetAttributes(&funcAttr, FindNewRayPartitions);
        unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        unsigned int blocks = (rayPartitions.size() / blocksize) + 1;
        FindNewRayPartitions<<<blocks, blocksize>>>(RawPointer(rayPartitions) + leafNodes, activeNodes,
                                                    RawPointer(rayMortonCodes),
                                                    RawPointer(nextRayPartitions) + leafNodes);
        // cout << "nextRayPartitions:\n" << nextRayPartitions << endl;


        // Use the new partitions to approximate the left and right bounds for
        // ray partitions and split the geometry. TODO All of these
        // transformations below can be chained into one.
        bounds.resize(nextActiveNodes);
        thrust::transform(nextRayPartitions.begin() + leafNodes, nextRayPartitions.end(), 
                          bounds.begin(), CreateBoundsFromPartitions(rayMortonCodes));
        // cout << "bounds:\n" << bounds << endl;

        static HyperCubes hyperCubes(128); hyperCubes.Resize(bounds.size());
        thrust::transform(bounds.begin(), bounds.end(), 
                          hyperCubes.Begin(), CreateHyperCubesFromBounds(rayMortonCoder));
        // cout << hyperCubes << endl;
        
        static thrust::device_vector<Cone> cones(hyperCubes.Size()); cones.resize(hyperCubes.Size());
        thrust::transform(hyperCubes.Begin(), hyperCubes.End(), cones.begin(), CreateCones());
        // cout << "cones:\n" << cones << endl;


        // Sphere's are partitioned | LEFT0 | RIGHT0 | LEFT1 | RIGHT1 | ....
        const unsigned int nActiveSpheres = sphereIndices.size() - leafSphereIndices;

        // Assign geometry to each childs cone and compute new indices for geometry moved left and right.
        static thrust::device_vector<bool> moveLeftRight(nActiveSpheres * 2);
        moveLeftRight.resize(nActiveSpheres * 2);
        cudaFuncGetAttributes(&funcAttr, SpherePartitioningByConesKernel);
        blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        blocks = (nActiveSpheres / blocksize) + 1;
        // TODO don't offset sphereIndices and other arrays? And fix the error
        SpherePartitioningByConesKernel<<<blocks, blocksize>>>
            (RawPointer(sphereIndices),
             RawPointer(sphereIndexPartition),
             RawPointer(spherePartitions),
             RawPointer(cones), RawPointer(spheres.spheres),
             RawPointer(moveLeftRight),
             nActiveSpheres,
             leafSphereIndices,
             leafNodes);
        CHECK_FOR_CUDA_ERROR();

        static thrust::device_vector<unsigned int> sphereLeftRightEntries(sphereIndices.size());
        sphereLeftRightEntries.resize(nActiveSpheres * 2 + 1); // +1 for dummy element
        sphereLeftRightEntries[0] = 0;
        thrust::transform_inclusive_scan(moveLeftRight.begin(), moveLeftRight.end(),
                                         sphereLeftRightEntries.begin() + 1,
                                         BoolToUint(),
                                         thrust::plus<unsigned int>());        
        
        const unsigned int nNextActiveSpheres = sphereLeftRightEntries[sphereLeftRightEntries.size()-1];
        // cout << "nNextActiveSpheres: " << nNextActiveSpheres << endl;

        /*
          // Is this faster and does it work with spatially coherent partitioning?
        static thrust::device_vector<PartitionSide> spherePartitionSides(sphereIndices.size());
        spherePartitionSides.resize(sphereIndices.size());
        thrust::transform(sphereIndices.begin() + leafSphereIndices, sphereIndices.end() + leafSphereIndices,
                          sphereIndexPartition.begin() + leafSphereIndices, 
                          spherePartitionSides.begin() ,
                          SpherePartitioningByCones(cones, spheres.spheres));
        // cout << "spherePartitionSides:\n" << spherePartitionSides << endl;

        static thrust::device_vector<uint2> sphereLeftRightIndices(sphereIndices.size());
        sphereLeftRightIndices.resize(sphereIndices.size() - leafSphereIndices + 1); // +1 for dummy element
        sphereLeftRightIndices[0] = make_uint2(0, 0);
        thrust::transform_inclusive_scan(spherePartitionSides.begin(), spherePartitionSides.end(),
                                         sphereLeftRightIndices.begin() + 1,
                                         PartitionSideToUint2(),
                                         thrust::plus<uint2>());

        cudaFuncGetAttributes(&funcAttr, AddFinalLeftIndexToRightIndices);
        blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        blocks = (sphereLeftRightIndices.size() / blocksize) + 1;
        AddFinalLeftIndexToRightIndices<<<blocks, blocksize>>>(RawPointer(sphereLeftRightIndices),
                                                              RawPointer(sphereLeftRightIndices) + sphereLeftRightIndices.size()-1);
        // cout << "sphereLeftRightIndices:\n" << sphereLeftRightIndices << endl;
        const unsigned int nNextActiveSphere = ((uint2)sphereLeftRightIndices[sphereLeftRightIndices.size()-1]).y;
        */

        nextSphereIndices.resize(leafSphereIndices + nNextActiveSpheres); // include leaf sphere indices to preserve already done indices at the front of the list.
        nextSphereIndexPartition.resize(leafSphereIndices + nNextActiveSpheres);
        
        cudaFuncGetAttributes(&funcAttr, PartitionIndices);
        blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        blocks = (nActiveSpheres / blocksize) + 1;
        PartitionIndices<<<blocks, blocksize>>>
            (RawPointer(sphereIndices) + leafSphereIndices,
             RawPointer(sphereIndexPartition) + leafSphereIndices,
             RawPointer(spherePartitions) + leafNodes, // active partitionings
             RawPointer(moveLeftRight),
             RawPointer(sphereLeftRightEntries),
             RawPointer(nextSphereIndices) + leafSphereIndices, // result
             RawPointer(nextSphereIndexPartition) + leafSphereIndices, // result
             nActiveSpheres, leafNodes, leafSphereIndices);
        CHECK_FOR_CUDA_ERROR();
        
        // cout << "nextSphereIndices:\n" << nextSphereIndices << endl;
        // cout << "nextSphereIndexOwner:\n" << nextSphereIndexPartition << endl;

        sphereIndices.swap(nextSphereIndices);
        sphereIndexPartition.swap(nextSphereIndexPartition);

        // Create new sphere partitions
        nextSpherePartitions.resize(nextPartitionSize);
        cudaFuncGetAttributes(&funcAttr, CreateNextPartitionings);
        blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        blocks = (activeNodes / blocksize) + 1;
        
        CreateNextPartitionings<<<blocks, blocksize>>>
            (RawPointer(spherePartitions) + leafNodes,
             RawPointer(sphereLeftRightEntries),
             RawPointer(nextSpherePartitions) + leafNodes,
             leafSphereIndices, activeNodes);
        CHECK_FOR_CUDA_ERROR();
        // cout << "nextSpherePartitions:\n" << nextSpherePartitions << endl;

        spherePartitions.swap(nextSpherePartitions);
        rayPartitions.swap(nextRayPartitions);

        // cout << "\n ** Before Leaf removal **\n" << ToString(true) << endl;

        // *** Remove leafs ***
        CreateLeafNodes();        

        // cout << "\n ** After Leaf removal **\n" << ToString(true) << endl;
    }
    CHECK_FOR_CUDA_ERROR();
    
    // Compute bounds of geometry partitions and do coarse level ray elimination
    // before intersection.

    // cout << "\n *** GORELESS VICTORY ***\n" << endl;
}


struct MortonIsNodeLeaf {
    // TODO take morton codes into account. If rayPartition.x and
    // rayPartition.y-1 points to the same morton code, then we can't possibly
    // subdivide further.

    __host__ __device__
    bool operator()(const uint2 rayPartition, const uint2 spherePartition) const {
        const float rayCount = (float)(rayPartition.y - rayPartition.x);
        const float sphereCount = (float)(spherePartition.y - spherePartition.x);
        
        //return rayCount * sphereCount <= 16.0f * (rayCount + sphereCount);
        return rayCount * sphereCount <= 11.0f * (rayCount + sphereCount);
    }
};

struct BoolToInt { __host__ __device__ unsigned int operator()(bool b) { return (int)b; } };

struct MarkLeafSize {
    __host__ __device__
    unsigned int operator()(const thrust::tuple<bool, uint2> input) const {
        bool isLeaf = thrust::get<0>(input);
        uint2 rayPartition = thrust::get<1>(input);
        return isLeaf ? rayPartition.y - rayPartition.x : 0;
    }
};

__global__
void CreateNewLeafNodes(const uint2* const rayPartitions,
                        const uint2* const spherePartitions,
                        const bool* isLeafNode,
                        const unsigned int* leafNodeIndices,
                        const unsigned int* newSpherePartitionBegins,
                        uint2* nextRayPartitions,
                        uint2* nextSpherePartitions,
                        const unsigned int nActiveNodes,
                        const unsigned int nNewLeafs,
                        const unsigned int nOldLeafSphereIndices,
                        const unsigned int nNewLeafSphereIndices) {

    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (id >= nActiveNodes) return;
    
    const bool isLeaf = isLeafNode[id];
    const unsigned int leafIndex = leafNodeIndices[id];
    const unsigned int index = isLeaf ? leafIndex : id - leafIndex + nNewLeafs;

    // Since rays are presorted, their begin and end indices do not change.
    nextRayPartitions[index] = rayPartitions[id];
    
    // Sphere indices are partitioned though so get to it.
    const uint2 oldSpherePartition = spherePartitions[id];
    const unsigned int range = oldSpherePartition.y - oldSpherePartition.x;
    const unsigned int leafSpherePartitionBegin = newSpherePartitionBegins[id];
    const unsigned int spherePartitionBegin = isLeaf ? 
        leafSpherePartitionBegin + nOldLeafSphereIndices :
        oldSpherePartition.x + nNewLeafSphereIndices - leafSpherePartitionBegin;
    nextSpherePartitions[index] = make_uint2(spherePartitionBegin, spherePartitionBegin+ range);
}

/**
 * This kernel assumes that data indices are layed out in the following order before partition.
 * | ... inactive data, i.e. leaf data ... | ... active data, possible new inactive data ... |
 *
 * After the partition the data will be layed out as follows.
 * | ... old inactive data ... | ... new inactive  data ... | ... still active data ... |
 */
__global__
void PartitionLeafIndices(const unsigned int* const indices,
                          const unsigned int* const owners,
                          const uint2* const nodePartitions, // starts at the first active partition
                          const bool* const isLeaf,
                          const unsigned int* const leafSphereBeginEntries,
                          const unsigned int* const leafNodeIndices,
                          const unsigned int nActiveIndices, // Currently active data indices
                          const unsigned int nOldLeafIndices, // number of old data points that have been moved to leafs
                          const unsigned int nNewLeafIndices, // 
                          const unsigned int nOldLeafs, // number of leafs prior to checking for new ones.
                          const unsigned int nNewLeafs, // number of new leafs
                          // Results
                          unsigned int* nextIndices,
                          unsigned int* nextOwners) {

    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nActiveIndices) return;
    
    id += nOldLeafIndices;

    // Owners are global, so it's minus the number of leaves to get the owner of
    // the active indices.
    const unsigned int owner = owners[id] - nOldLeafs;
    const uint2 partitioning = nodePartitions[owner];
    
    const bool leaf = isLeaf[owner];

    // Compute the new index of the data
    const unsigned int partitioningBegin = leaf ? 
        leafSphereBeginEntries[owner] + nOldLeafIndices : 
        partitioning.x + nNewLeafIndices - leafSphereBeginEntries[owner];

    // Offset of the data in the partitioning relative to the beginning.
    const unsigned int offset = id - partitioning.x;
    const unsigned int newEntry = partitioningBegin + offset;

    nextIndices[newEntry] = indices[id];

    // Compute the new owner of the data. If it has been moved to a leaf node,
    // then the new owner is the addition of the number of old leafs and the
    // leaf index of the node in the list of new leafs. If the node is not a
    // leaf, then it is the old node index minus the number of new leafs created
    // (plus the number of old leaf nodes that was subtracted earlier). Now
    // how's that for documentation!
    unsigned int newOwner = leaf ? 
        leafNodeIndices[owner] + nOldLeafs : 
        owner + nOldLeafs + nNewLeafs - leafNodeIndices[owner]; // TODO Move newOwner and partitioningbegin calculations into own kernel parallized over nodes? Costs extra memory, but should be faster.
        
    nextOwners[newEntry] = newOwner;
}

bool MortonDacrtNodes::CreateLeafNodes() {
    
    const size_t nActiveNodes = spherePartitions.size() - leafNodes;
    const size_t nActiveSphereIndices = sphereIndices.size() - leafSphereIndices;
    
    cout << "\n --- Create leaf nodes from " << nActiveNodes << " active nodes and " << nActiveSphereIndices << " active sphere indices. ---\n" << endl;

    // Locate leaf partitions
    static thrust::device_vector<bool> isLeaf(nActiveNodes);
    isLeaf.resize(nActiveNodes);

    // TODO make isLeaf unsigned int and reuse for indices? isLeaf info is
    // stored in an index and it's neighbour.
    thrust::transform(rayPartitions.begin() + leafNodes, rayPartitions.end(),
                      spherePartitions.begin() + leafNodes,
                      isLeaf.begin(), MortonIsNodeLeaf());
    // cout << "isLeaf:\n" << isLeaf << endl;    

    static thrust::device_vector<unsigned int> leafNodeIndices(nActiveNodes+1);
    leafNodeIndices.resize(nActiveNodes+1);
    leafNodeIndices[0] = 0;
    thrust::transform_inclusive_scan(isLeaf.begin(), isLeaf.end(), leafNodeIndices.begin()+1, 
                                     BoolToInt(), thrust::plus<unsigned int>());
    // cout << "leafNodeEntries:\n" << leafNodeIndices << endl;
    const size_t nNewLeafs = leafNodeIndices[leafNodeIndices.size()-1];
    // cout << "nNewLeafs: " << nNewLeafs << endl;
    

    if (nNewLeafs == 0) { 
        // cout << "No new leafs" << endl;
        return false;
    }

    // Compute start index for geometric primitives in done nodes.
    static thrust::device_vector<unsigned int> leafSpheresStartIndex(nActiveNodes+1);
    leafSpheresStartIndex.resize(nActiveNodes+1);
    leafSpheresStartIndex[0] = 0;
    thrust::zip_iterator<thrust::tuple<BoolIterator, Uint2Iterator> > leafNodeValues = 
        thrust::make_zip_iterator(thrust::make_tuple(isLeaf.begin(), ActiveSpherePartitionsBegin()));
    thrust::transform_inclusive_scan(leafNodeValues, leafNodeValues + nActiveNodes, 
                                     leafSpheresStartIndex.begin()+1, MarkLeafSize(), 
                                     thrust::plus<unsigned int>());
    // cout << "leafSpheresBeginEntries:\n" << leafSpheresStartIndex << endl;
    const size_t nNewLeafIndices = leafSpheresStartIndex[leafSpheresStartIndex.size() - 1];
    // cout << "nNewLeafIndices: " << nNewLeafIndices << endl;

    
    // Partition sphere indices, recompute owners and copy indices+owners into
    // the other indices array to preserve leaf info no matter which array ends
    // up being active.
    const unsigned int nSphereIndices = sphereIndices.size();
    nextSphereIndices.resize(nSphereIndices);
    nextSphereIndexPartition.resize(nSphereIndices);
    
    struct cudaFuncAttributes funcAttr;
    cudaFuncGetAttributes(&funcAttr, PartitionLeafIndices);
    unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    unsigned int blocks = (nActiveSphereIndices / blocksize) + 1;
    PartitionLeafIndices<<<blocks, blocksize>>>
        (RawPointer(sphereIndices),
         RawPointer(sphereIndexPartition),
         RawPointer(spherePartitions) + leafNodes, // Starts at the first active partition
         RawPointer(isLeaf),
         RawPointer(leafSpheresStartIndex),
         RawPointer(leafNodeIndices),
         nActiveSphereIndices, // Currently active data indices
         leafSphereIndices, // number of old data points that have been moved to leafs
         nNewLeafIndices,
         leafNodes, // number of leafs prior to checking for new ones.
         nNewLeafs, // number of new leafs
         // results
         RawPointer(nextSphereIndices),
         RawPointer(nextSphereIndexPartition));
    CHECK_FOR_CUDA_ERROR();

    // Copy new leaf geometry indices back into the old array to maintain a
    // complete list of leafs indices in both arrays (TODO only copy to the main
    // array, to save a few copies)
    cudaMemcpy(RawPointer(sphereIndices) + leafSphereIndices,
               RawPointer(nextSphereIndices) + leafSphereIndices,
               sizeof(unsigned int) * nNewLeafIndices, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy(RawPointer(sphereIndexPartition) + leafSphereIndices,
               RawPointer(nextSphereIndexPartition) + leafSphereIndices,
               sizeof(unsigned int) * nNewLeafIndices, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();

    sphereIndices.swap(nextSphereIndices);
    sphereIndexPartition.swap(nextSphereIndexPartition);

    
    // Partition ray and sphere partitions
    nextRayPartitions.resize(rayPartitions.size());
    nextSpherePartitions.resize(spherePartitions.size());
    
    cudaFuncGetAttributes(&funcAttr, CreateNewLeafNodes);
    blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    blocks = (nActiveNodes / blocksize) + 1;
    CreateNewLeafNodes<<<blocks, blocksize>>>
        (RawPointer(rayPartitions) + leafNodes,
         RawPointer(spherePartitions) + leafNodes,
         RawPointer(isLeaf),
         RawPointer(leafNodeIndices),
         RawPointer(leafSpheresStartIndex),
         RawPointer(nextRayPartitions) + leafNodes,
         RawPointer(nextSpherePartitions) + leafNodes,
         nActiveNodes,
         nNewLeafs,
         leafSphereIndices,
         nNewLeafIndices);
    CHECK_FOR_CUDA_ERROR();

    // Copy new leaf nodes back into the old array to maintain a complete list
    // of leafs nodes in both arrays (TODO only copy to the main array, to save
    // a few copies)
    cudaMemcpy(RawPointer(spherePartitions) + leafNodes,
               RawPointer(nextSpherePartitions) + leafNodes,
               sizeof(uint2) * nNewLeafs, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy(RawPointer(rayPartitions) + leafNodes,
               RawPointer(nextRayPartitions) + leafNodes,
               sizeof(uint2) * nNewLeafs, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();

    spherePartitions.swap(nextSpherePartitions);
    rayPartitions.swap(nextRayPartitions);

    // cout << "new spherePartitions:\n" << spherePartitions << endl;
    // cout << "new rayPartitions:\n" << rayPartitions << endl;
    

    leafSphereIndices += nNewLeafIndices;
    leafNodes += nNewLeafs;
    // cout << "new leaf count: " << leafNodes << endl;
    // cout << "leafSphereIndices:\n" << leafSphereIndices << endl;

    return true;
}

__global__
void CreateOwnersMap(const uint2* const partitionings,
                     const unsigned int* const owners,
                     unsigned int* ownersMap,
                     const unsigned int nPartitionings) {

    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nPartitionings) return;
    
    const unsigned int partitionBegin = partitionings[id].x;
    const unsigned int mapEntry = owners[partitionBegin];
    ownersMap[mapEntry] = id;
}

__global__
void RemapOwners(const unsigned int* const ownersMap,
                 unsigned int* owners,
                 const unsigned int nOwners) {
    
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nOwners) return;

    const unsigned int oldOwner = owners[id];
    owners[id] = ownersMap[oldOwner];
}

__global__
void ExhaustiveIntersection(const float4* const rayOrigins, 
                            float4* rayDirs,
                            const unsigned int* const nodes,
                            const uint2* const spherePartitionings,
                            const unsigned int* const sphereIndices,
                            const Sphere* const spheres,
                            const unsigned int nRays,
                            unsigned int *hitIDs) {
    
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nRays) return;

    const unsigned int node = nodes[id];

    const float3 origin = make_float3(rayOrigins[id]);
    const float3 dir = make_float3(rayDirs[id]);
        
    const uint2 spherePartitioning = spherePartitionings[node];
    float hitT = 1e30f;
    unsigned int hitID = SpheresGeometry::MISSED;
    
    for (unsigned int g = spherePartitioning.x; g < spherePartitioning.y; ++g) {
        const unsigned int sphereId = sphereIndices[g];
        const Sphere s = spheres[sphereId];
        const float t = s.Intersect(origin, dir);
        if (0 < t && t < hitT) {
            hitID = sphereId;
            hitT = t;
        }
    }
    
    rayDirs[id] = make_float4(dir, hitT);
    hitIDs[id] = hitID;
}

void MortonDacrtNodes::FindIntersections(thrust::device_vector<unsigned int>& hitIDs) {
    const unsigned int nActiveRays = rays->LeafRays();
    hitIDs.resize(nActiveRays);
    
    // Compute which nodes each ray belongs to. This should probably be done
    // using work queues on high end cards.
    thrust::device_vector<unsigned int> owners(nActiveRays);
    owners.resize(nActiveRays);
    DacrtNodes::CalcOwners(rayPartitions.begin(), rayPartitions.end(), owners);
    // Calc owners assumes that partitions are sorted in increasing order, but
    // any node at any given time can be marked as a leaf node, so we need to
    // correct this.
    thrust::device_vector<unsigned int> ownersMap(rayPartitions.size());
    unsigned int blocksize = 256;
    unsigned int blocks = (rayPartitions.size() / blocksize) + 1;
    CreateOwnersMap<<<blocks, blocksize>>>(RawPointer(rayPartitions), RawPointer(owners),
                                           RawPointer(ownersMap), rayPartitions.size());
    
    blocksize = 256;
    blocks = (nActiveRays / blocksize) + 1;
    RemapOwners<<<blocks, blocksize>>>(RawPointer(ownersMap), RawPointer(owners),
                                       nActiveRays);
    
    // Perform exhaustive intersection.
    struct cudaFuncAttributes funcAttr;
    cudaFuncGetAttributes(&funcAttr, ExhaustiveIntersection);
    blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    blocks = (nActiveRays / blocksize) + 1;
    ExhaustiveIntersection<<<blocks, blocksize>>>
        (RawPointer(Rays::GetOrigins(rays->BeginLeafRays())),
         RawPointer(Rays::GetDirections(rays->BeginLeafRays())),
         RawPointer(owners),
         RawPointer(spherePartitions),
         RawPointer(sphereIndices),
         RawPointer(spheresGeom->spheres),
         nActiveRays,
         RawPointer(hitIDs));
    CHECK_FOR_CUDA_ERROR();
}


void MortonDacrtNodes::InitSphereIndices(HyperCubes& cubes, SpheresGeometry& spheres) {
    sphereIndices.resize(spheres.Size() * cubes.Size());
    sphereIndexPartition.resize(sphereIndices.size());
    
    thrust::device_vector<Cone> cones(cubes.Size()); // TODO can't this be passed in from outside to save allocation and dealloc?
    thrust::transform(cubes.Begin(), cubes.End(), cones.begin(), CreateCones());

    unsigned int spherePartitionPivots[cubes.Size() + 1];
    spherePartitionPivots[0] = 0;
    for (int c = 0; c < cubes.Size(); ++c) {
        const UintIterator beginIndices = sphereIndices.begin() + spherePartitionPivots[c];
        const UintIterator itr = thrust::copy_if(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(spheres.Size()),
                                                 spheres.BeginSpheres(), 
                                                 beginIndices,
                                                 CompareConeSphere(cones, c));
        spherePartitionPivots[c+1] = spherePartitionPivots[c] + (itr - beginIndices);

        // Write which partitions the sphere indices belong to
        thrust::fill(sphereIndexPartition.begin() + spherePartitionPivots[c],
                     sphereIndexPartition.begin() + spherePartitionPivots[c+1], (unsigned int)c);
    }
    const size_t currentSize = spherePartitionPivots[cubes.Size()];
    sphereIndices.resize(currentSize);
    sphereIndexPartition.resize(sphereIndices.size());
    nextSphereIndices.resize(sphereIndices.size());

    std::cout << "sphere partition pivots: ";
    for (int p = 0; p < cubes.Size() + 1; ++p)
        std::cout << spherePartitionPivots[p] << ", ";
    std::cout << std::endl;
    
    thrust::host_vector<uint2> h_spherePartitions(cubes.Size());
    for (int i = 0; i < h_spherePartitions.size(); ++i)
        h_spherePartitions[i] = make_uint2(spherePartitionPivots[i], spherePartitionPivots[i+1]);

    spherePartitions.resize(cubes.Size());
    thrust::copy(h_spherePartitions.begin(), h_spherePartitions.end(), spherePartitions.begin());
}


std::string MortonDacrtNodes::ToString(const bool verbose) const {
    std::ostringstream out;
    
    if (rayPartitions.size() > leafNodes) {
        out << "Active nodes:";
        for (size_t i = leafNodes; i < rayPartitions.size(); ++i)
            out << "\n" << PrintNode(i, verbose);
        if (leafNodes > 0) out << "\n";
    }

    if (leafNodes > 0) {
        out << "Leaf nodes:";
        for (size_t i = 0; i < leafNodes; ++i)
            out << "\n" << PrintNode(i, verbose);
    }

    return out.str();
}

std::string MortonDacrtNodes::PrintNode(const unsigned int i, const bool verbose) const {
    std::ostringstream out;

    uint2 rayPartition = rayPartitions[i];
    uint2 spherePartition = spherePartitions[i];
    out << i << ": [rays: [" << rayPartition.x << " -> " << rayPartition.y << 
        "], spheres: [" << spherePartition.x << " -> " << spherePartition.y << "]]";
    if (verbose) {
        // out << "\n  Rays: [";
        // for (int r = rayPartition.x; r < rayPartition.y; ++r) {
        //     out << rays->GetLeafRay(r).id;
        //     if (r < rayPartition.y-1) 
        //         out << ", ";
        //     else 
        //         out << "]";
        // }

        out << "\n  Spheres: [";
        for (int s = spherePartition.x; s < spherePartition.y; ++s) {
            out << sphereIndices[s] << " (" << sphereIndexPartition[s] << ")";
            if (s < spherePartition.y-1) 
                out << ", ";
            else 
                out << "]";
        }
    }
    
    return out.str();
}
