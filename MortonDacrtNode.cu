// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <MortonDacrtNode.h>

#include <Meta/CUDA.h>
#include <Primitives/AABB.h>
#include <Primitives/HyperCube.h>
#include <RayContainer.h>
#include <SphereGeometry.h>
#include <Utils/Morton.h>
#include <Utils/ToString.h>

#include <iostream>
#include <stdexcept>

#include <thrust/sort.h>
#include <thrust/transform.h>

using std::cout;
using std::endl;

using Utils::Morton;

struct MortonCode {

#define xOffset 3
#define yOffset 4
#define zOffset 2
#define uOffset 1
#define vOffset 0
    // Interleave pattern | Y | X | Z | U | V |. Y is last so I can overwrite
    // that last bit with the axis info at a future point.

    unsigned int code;

    __host__ __device__ MortonCode() {}
    __host__ __device__ MortonCode(const unsigned int c) : code(c) {}
    
    __host__ __device__ static inline unsigned int CodeFromIndex(const unsigned int index, const int offset) { return Morton::PartBy4(index) << offset; }
    __host__ __device__ static inline unsigned int IndexFromCode(const MortonCode code, const int offset) { return Morton::CompactBy4(code.code >> offset); }
    __host__ __device__ static inline MortonCode EncodeAxis(const unsigned int a) { return MortonCode(a << 29); }
    __host__ __device__ static inline SignedAxis AxisFromCode(const MortonCode code) { return SignedAxis((code.code & 0xE0000000) >> 29); }

    __host__ __device__ inline MortonCode& operator=(const unsigned int rhs) { code = rhs; return *this; }
    __host__ __device__ inline void operator+=(const MortonCode rhs) { code += rhs; }
    __host__ __device__ inline operator unsigned int() const { return code; }
    
    __host__ __device__ inline MortonCode WithoutAxis() const { return MortonCode(code & 0x1FFFFFFF); }

    inline std::string ToString() const {
        std::ostringstream out;
        SignedAxis axis = AxisFromCode(*this);
        MortonCode tmp = MortonCode(code & 0x1FFFFFFF);
        unsigned int x = IndexFromCode(tmp, xOffset);
        unsigned int y = IndexFromCode(tmp, yOffset);
        unsigned int z = IndexFromCode(tmp, zOffset);
        unsigned int u = IndexFromCode(tmp, uOffset);
        unsigned int v = IndexFromCode(tmp, vOffset);
        out << "[axis: " << axis << ", x: " << x << ", y: " << y << ", z: " << z << ", u: " << u << ", v: " << v << " - " << Bitmap(code) << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const MortonCode& c){
    return s << c.ToString();
}


struct MortonBound {
    MortonCode min, max;

    __host__ __device__    
    inline static MortonBound Create(const unsigned int val) {
        MortonBound b; b.min = b.max = val; 
        return b;
    }

    __host__ __device__    
    inline static MortonBound Create(const unsigned int min, const unsigned int max) {
        MortonBound b; b.min = min; b.max = max; 
        return b;
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << "[min: " << min << ", max: " << max << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const MortonBound& c){
    return s << c.ToString();
}

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
void ReduceMinMaxMortonByAxis(thrust::device_vector<unsigned int>::iterator mortonBegin,
                              thrust::device_vector<unsigned int>::iterator mortonEnd,
                              thrust::device_vector<MortonBound>::iterator boundsBegin,
                              thrust::device_vector<MortonBound>::iterator boundsEnd);

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
    ReduceMinMaxMortonByAxis(rayMortonCodes.begin(), rayMortonCodes.end(),
                             bounds.begin(), bounds.end());

    MortonBound zNegBound = bounds[5];
    std::cout << "HyperCube: " << rayMortonCoder.HyperCubeFromBound(zNegBound) << std::endl;
    

    exit(0);

    // Cull inactive (size 0) partitions.

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
    if (axis == lhsMin & 0xE0000000) { // Compare axis
        // Both have the same owner, so store the result back in shared
        // memory.
        min[index] = Morton::MinBy4(lhsMin.WithoutAxis(), rhsMin.WithoutAxis()) + axis;
        max[index] = Morton::MaxBy4(lhsMax.WithoutAxis(), rhsMax.WithoutAxis()) + axis;
    } else {
        // Different owners, so store the lhs in global memory and rhs in
        // shared.
        min[index] = rhsMin;
        max[index] = rhsMax;
        
        const unsigned int globalIndex = MortonCode::AxisFromCode(lhsMin);
        const MortonBound old = bounds[globalIndex];
        const unsigned int min = Morton::MinBy4(lhsMin.WithoutAxis(), old.min.WithoutAxis()) + axis;
        const unsigned int max = Morton::MaxBy4(lhsMax.WithoutAxis(), old.max.WithoutAxis()) + axis;
        bounds[globalIndex] = MortonBound::Create(min, max);
    }
}

__global__
void ReduceMinMaxMortonByAxisPass2(MortonBound* intermediateBounds,
                                   const size_t intermediateBoundsSize,
                                   MortonBound* bounds,
                                   const size_t boundsSize) {
    
    __shared__ volatile unsigned int mins[128];
    __shared__ volatile unsigned int maxs[128];

    if (threadIdx.x < intermediateBoundsSize) {
        MortonBound b = intermediateBounds[threadIdx.x];
        mins[threadIdx.x] = b.min;
        maxs[threadIdx.x] = b.max;
    } else
        mins[threadIdx.x] = maxs[threadIdx.x] = MortonCode::EncodeAxis(6);
    __syncthreads();

    // Reduce 128 values left in shared memory
    if (threadIdx.x >= 64) return;
    PerformMinMaxMortonByAxisReduction(threadIdx.x, mins, maxs, bounds);

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
    const unsigned int lhsMin = mins[0];
    const unsigned int lhsMax = maxs[0];
    const unsigned int axis = lhsMin & 0xE0000000;
    const unsigned int globalIndex = MortonCode::AxisFromCode(lhsMin);
    if (globalIndex != 6) { // If any dummy values (axis == 6) have been used,
                            // then now is the time to discard them.
        const MortonBound old = bounds[globalIndex];
        const unsigned int min = Morton::MinBy4(lhsMin & 0x1FFFFFFF, old.min & 0x1FFFFFFF) + axis;
        const unsigned int max = Morton::MaxBy4(lhsMax & 0x1FFFFFFF, old.max & 0x1FFFFFFF) + axis;
        bounds[globalIndex] = MortonBound::Create(min, max);
    }
}

__global__
void ReduceMinMaxMortonByAxisPass1(const unsigned int* const mortonCodes,
                                   const size_t inputSize,
                                   MortonBound* intermediateBounds,
                                   MortonBound* bounds,
                                   const size_t boundsSize) {

    // Fill bounds with default values
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < boundsSize; i += blockDim.x * gridDim.x)
        bounds[i] = MortonBound::Create(0xDFFFFFFF, 0xC0000000);

    __shared__ volatile unsigned int min[256];
    __shared__ volatile unsigned int max[256];
    
    const size_t beginIndex = inputSize * blockIdx.x / gridDim.x;
    const size_t endIndex = inputSize * (blockIdx.x + 1) / gridDim.x;
    
    size_t currentIndex = beginIndex + threadIdx.x;

    // Fill initial values
    min[threadIdx.x] = max[threadIdx.x] = currentIndex < endIndex ? MortonCode(mortonCodes[currentIndex]) : MortonCode::EncodeAxis(6);
    currentIndex += blockDim.x;
    __syncthreads();
    
    // While still values left, load them and perform reduction
    while (currentIndex < endIndex) {
        // Fetch new data from global mem to shared mem
        unsigned int code = currentIndex < endIndex ? MortonCode(mortonCodes[currentIndex]) : MortonCode::EncodeAxis(6);
        min[blockDim.x + threadIdx.x] = max[blockDim.x + threadIdx.x] = code;
        __syncthreads();

        PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);
    }
    __syncthreads();

    // Reduce 128 values left in shared memory
    if (threadIdx.x >= 64) return;
    PerformMinMaxMortonByAxisReduction(threadIdx.x, min, max, bounds);

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


void ReduceMinMaxMortonByAxis(thrust::device_vector<unsigned int>::iterator mortonBegin,
                              thrust::device_vector<unsigned int>::iterator mortonEnd,
                              thrust::device_vector<MortonBound>::iterator boundsBegin,
                              thrust::device_vector<MortonBound>::iterator boundsEnd) {

    // Verify that CUDA is initialized
    if (!Meta::CUDA::initialized)
        throw std::runtime_error("CUDA wasn't initialized. Can't lookup kernel properties");
    
    const size_t inputSize = mortonEnd - mortonBegin;
    const size_t boundsSize = boundsEnd - boundsBegin; // Always 6, but this looks less like magic.
    
    // struct cudaFuncAttributes funcAttr;
    // cudaFuncGetAttributes(&funcAttr, ReduceMinMaxMortonByAxisPass1);
    // const unsigned int blockDim = funcAttr.maxThreadsPerBlock > 128 ? 128 : funcAttr.maxThreadsPerBlock;
    const unsigned int blockDim = 128;
    const unsigned int blocks = Meta::CUDA::activeCudaDevice.multiProcessorCount;

    static thrust::device_vector<MortonBound> intermediateBounds(boundsSize * 2);
    
    ReduceMinMaxMortonByAxisPass1<<<blocks, blockDim>>>(RawPointer(mortonBegin), inputSize,
                                                        RawPointer(intermediateBounds),
                                                        thrust::raw_pointer_cast(&*boundsBegin), boundsSize);

    std::cout << "intermediateBounds\n" << intermediateBounds << std::endl;
    
    std::cout << "bounds:\n";
    for (int i = 0; i < boundsSize; ++i) {
        std::cout << i << ": " << boundsBegin[i];
        if (i < boundsSize-1)
            std::cout << "\n";
    }
    std::cout << std::endl;
    
    ReduceMinMaxMortonByAxisPass2<<<1, 128>>>(RawPointer(intermediateBounds), intermediateBounds.size(),
                                              thrust::raw_pointer_cast(&*boundsBegin), boundsSize);

    std::cout << "bounds:\n";
    for (int i = 0; i < boundsSize; ++i) {
        std::cout << i << ": " << boundsBegin[i];
        if (i < boundsSize-1)
            std::cout << "\n";
    }
    std::cout << std::endl;
    
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
