// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <MortonDacrtNode.h>

#include <Kernels/ReduceMinMaxMortonCode.h>
#include <Primitives/AABB.h>
#include <Primitives/HyperCube.h>
#include <Primitives/MortonCode.h>
#include <RayContainer.h>
#include <SphereGeometry.h>
#include <Utils/ToString.h>

#include <iostream>

#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

using std::cout;
using std::endl;

MortonDacrtNodes::MortonDacrtNodes(const size_t capacity) 
    : rayPartitions(capacity), 
      spherePartitions(capacity), nextSpherePartitions(capacity) {}


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
    
    __host__ __device__ inline float XMin(const unsigned int xIndex) { return (float)xIndex / (63.999f * boundInvSize.x) + bound.min.x; }
    __host__ __device__ inline float XMax(const unsigned int xIndex) { return XMin(xIndex+1.0f); }
    __host__ __device__ inline float YMin(const unsigned int yIndex) { return (float)yIndex / (31.999f * boundInvSize.y) + bound.min.y; }
    __host__ __device__ inline float YMax(const unsigned int yIndex) { return YMin(yIndex+1.0f); }
    __host__ __device__ inline float ZMin(const unsigned int zIndex) { return (float)zIndex / (63.999f * boundInvSize.z) + bound.min.z; }
    __host__ __device__ inline float ZMax(const unsigned int zIndex) { return ZMin(zIndex+1.0f); }

    __host__ __device__ inline float UMin(const unsigned int uIndex) { return (float)uIndex / 31.999f - 1.0f; }
    __host__ __device__ inline float UMax(const unsigned int uIndex) { return UMin(uIndex+1.0f); }
    __host__ __device__ inline float VMin(const unsigned int vIndex) { return (float)vIndex / 31.999f - 1.0f; }
    __host__ __device__ inline float VMax(const unsigned int vIndex) { return VMin(vIndex+1.0f); }
    
    __host__ __device__ inline HyperCube HyperCubeFromBound(MortonBound b) {
        const SignedAxis axis = MortonCode::AxisFromCode(b.min);

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

    // Cull inactive partitions.
    thrust::device_vector<MortonBound>::iterator boundsEnd = thrust::remove_if(bounds.begin(), bounds.end(), InvalidAxis());
    bounds.resize(boundsEnd - bounds.begin());
    std::cout << "Bounds:\n" << bounds << std::endl;
    
    MortonBound firstBound = bounds[0];
    std::cout << "HyperCube: " << rayMortonCoder.HyperCubeFromBound(firstBound) << std::endl;

    exit(0);

    /*    
    // Geometry partition
    
    while (spheres) {
        // Calculate bounding boxes for partitions

        // Split geometry by that plane.
        
        // Iterate over next splitting planes, as defined by morton encoding,
        // until one intersects with the box. This narrows the min-max keys
        // enclosing the geometry. (Corrosponds to empty space splitting)
        
        // Check if leaves should be partitioned (Since it's spatial
        // partitioning and primitives can get assigned to both sides, we can't
        // do in place partitioning.)
    }

    // Sort leaves based on their morton codes?

    // Pair up rays and sphere partitions
    */
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
