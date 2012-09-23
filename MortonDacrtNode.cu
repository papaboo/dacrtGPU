// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <MortonDacrtNode.h>

#include <HyperCubes.h>
#include <Kernels/ReduceMinMaxMortonCode.h>
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
#include <thrust/sort.h>
#include <thrust/transform.h>

using std::cout;
using std::endl;

MortonDacrtNodes::MortonDacrtNodes(const size_t capacity) 
    : rayPartitions(capacity), nextRayPartitions(capacity), 
      spherePartitions(capacity), nextSpherePartitions(capacity),
      sphereIndices(capacity), nextSphereIndices(capacity) {}

// TODO Apply the reduced upper bounds to the next levels partition bounds, to
// avoid to many false positives (in case of cone intersection, this means
// reducing the area between the hypercube and the cone, where false positives
// occur)

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
            MortonCode value = rayMortonCodes[mid];
            SignedAxis axis = value.GetAxis();
            min = axis < threadIdx.x ? (mid+1) : min;
            max = axis < threadIdx.x ? max : mid;
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
                          uint4* nextRayPartitions) {
    
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
    
    nextRayPartitions[id] = make_uint4(partition.x, pivot.x,
                                       pivot.x, partition.y);
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

struct SpherePartitioningByCones {
    Cone* cones;
    Sphere* spheres;
    SpherePartitioningByCones(thrust::device_vector<Cone>& cs, 
                              thrust::device_vector<Sphere>& ss)
        : cones(thrust::raw_pointer_cast(cs.data())),
          spheres(thrust::raw_pointer_cast(ss.data())) {}
    
    __device__
    uint2 operator()(const unsigned int sphereId, const unsigned int owner) const {
        const Sphere sphere = spheres[sphereId];
        
        uint2 res;
        const Cone leftCone = cones[owner * 2];
        res.x = leftCone.DoesIntersect(sphere) ? 1 : 0;
        
        const Cone rightCone = cones[owner * 2 + 1];
        res.y = rightCone.DoesIntersect(sphere) ? 1 : 0;
        return res;
    }
};


void MortonDacrtNodes::Create(RayContainer& rayContainer, SpheresGeometry& spheres) {
    
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

    // Find ray partition pivots
    rayPartitions.resize(6);
    FindInitialRayPartitions<<<1,32>>>(RawPointer(rayMortonCodes), rayMortonCodes.size(),
                                       RawPointer(rayPartitions));

    // Cull inactive partitions and bounds. (Ode to C++ auto or so I hear...)
    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<uint2>::iterator, 
        thrust::device_vector<MortonBound>::iterator> > PartitionBoundIterator;

    PartitionBoundIterator partition_bound_begin = 
        make_zip_iterator(make_tuple(rayPartitions.begin(), bounds.begin()));
    PartitionBoundIterator partition_bound_end = thrust::remove_if(partition_bound_begin, partition_bound_begin+bounds.size(), InvalidAxis());
    bounds.resize(partition_bound_end - partition_bound_begin);
    rayPartitions.resize(bounds.size());
    std::cout << "Bounds:\n" << bounds << std::endl;
    std::cout << "rayPartitions:\n" << rayPartitions << std::endl;
    
    HyperCubes hCubes(bounds.size());
    thrust::transform(bounds.begin(), bounds.end(), 
                      hCubes.Begin(), MortonBoundToHyperCube(rayMortonCoder));
    std::cout << hCubes << std::endl;

    InitSphereIndices(hCubes, spheres);
    // sphereIndices = new SphereContainer(hCubes, spheres, spherePartitionPivots);
    // std::cout << sphereIndices->ToString() << std::endl;

    
    // Geometry partition
    
    doneSpherePartitions = 0;
    while (spherePartitions.size() - doneSpherePartitions > 0) {
        // Create new ray partitions through binary search
        nextRayPartitions.resize(rayPartitions.size() * 2);
        struct cudaFuncAttributes funcAttr;
        cudaFuncGetAttributes(&funcAttr, FindNewRayPartitions);
        unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
        unsigned int blocks = (rayPartitions.size() / blocksize) + 1;
        FindNewRayPartitions<<<blocks, blocksize>>>(RawPointer(rayPartitions), rayPartitions.size(),
                                                    RawPointer(rayMortonCodes),
                                                    (uint4*)(void*)RawPointer(nextRayPartitions));
        std::cout << "nextRayPartitions:\n" << nextRayPartitions << std::endl;

        // Use the new partitions to approximate the left and right bounds for
        // ray partitions and split the geometry. TODO All of these
        // transformations below can be chained into one.
        bounds.resize(bounds.size() * 2);
        thrust::transform(nextRayPartitions.begin(), nextRayPartitions.end(), 
                          bounds.begin(), CreateBoundsFromPartitions(rayMortonCodes));
        std::cout << "bounds:\n" << bounds << std::endl;

        static HyperCubes hyperCubes(128); hyperCubes.Resize(bounds.size());
        thrust::transform(bounds.begin(), bounds.end(), 
                          hyperCubes.Begin(), CreateHyperCubesFromBounds(rayMortonCoder));
        std::cout << hyperCubes << std::endl;
        
        static thrust::device_vector<Cone> cones(hyperCubes.Size()); cones.resize(hyperCubes.Size());
        thrust::transform(hyperCubes.Begin(), hyperCubes.End(), cones.begin(), CreateCones());
        std::cout << "cones:\n" << cones << std::endl;

        // Assign geometry to each childs cone.

        static thrust::device_vector<uint2> sphereLeftRightIndices(sphereIndices.size());
        sphereLeftRightIndices.resize(sphereIndices.size() - doneSphereIndices);
        // SpherePartitioningByCones
        
        // Remove leafs


        rayPartitions.swap(nextRayPartitions);

        exit(0);
    }

    // Sort leaves based on their morton codes?
    
    // Compute bounds of geometry partitions and do coarse level ray elimination
    // before intersection.
    
}


void MortonDacrtNodes::InitSphereIndices(HyperCubes& cubes, SpheresGeometry& spheres) {
    sphereIndices.resize(spheres.Size() * cubes.Size());
    
    thrust::device_vector<Cone> cones(cubes.Size()); // TODO can't this be parsed in from outside to save allocation and dealloc?
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
    }
    const size_t currentSize = spherePartitionPivots[cubes.Size()];
    sphereIndices.resize(currentSize);
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




void TestMortonEncoding() {

    float4 origin = make_float4(5,1,5,0);
    float4 direction = make_float4(normalize(make_float3(1,1,0)), 0.0f);
    origin -= direction * 15.0f;
    thrust::tuple<float4, float4> ray = thrust::make_tuple<float4, float4>(origin, direction);

    cout << "[origin: " << origin << ", direction: " << direction << ", axisUV: " << HyperRay::AxisUVFromDirection(make_float3(direction)) << "]" << endl;

    float3 min = make_float3(-1, -1, -1); 
    float3 max = make_float3(5,6,7);
    AABB bounds = AABB::Create(min, max);
    cout << "Bounds: " << bounds << endl;    

    float tHit;
    if (bounds.ClosestIntersection(make_float3(origin), make_float3(direction), tHit))
        cout << "Intersects after " << tHit << " at " << make_float3(origin + tHit * direction) << endl;
    else 
        cout << "Missed bounds" << tHit <<endl;
    
    RayMortonCoder mortonEncoder(bounds);
    unsigned int xIndex = mortonEncoder.CreateXIndex(origin.x);
    MortonCode xKey = MortonCode::CodeFromIndex(xIndex, xOffset);
    unsigned int xIndexDecoded = MortonCode::IndexFromCode(xKey, xOffset);
    cout << "x: " << xIndex << " - " << Bitmap(xIndex) << " - " << Bitmap(xKey) << " - " << Bitmap(xIndexDecoded) << endl;
    
    unsigned int key = mortonEncoder(ray);
    cout << "morton key: " << key << " - " << Bitmap(key) << endl;
    unsigned int x = MortonCode::IndexFromCode(key & 0x1FFFFFFF, xOffset);
    unsigned int y = MortonCode::IndexFromCode(key & 0x1FFFFFFF, yOffset);
    unsigned int z = MortonCode::IndexFromCode(key & 0x1FFFFFFF, zOffset);
    unsigned int u = MortonCode::IndexFromCode(key & 0x1FFFFFFF, uOffset);
    unsigned int v = MortonCode::IndexFromCode(key & 0x1FFFFFFF, vOffset);
    
    cout << "x: " << x << ", y: " << y << ", z: " << z << ", u: " << u << ", v: " << v << endl;
    
    cout << "xMin: " << mortonEncoder.XMin(x) << ", xMax: " << mortonEncoder.XMax(x) << endl;
}
