// DACRT node
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <DacrtNode.h>

#include <Cone.h>
#include <ForEachWithOwners.h>
#include <HyperCube.h>
#include <HyperRays.h>
#include <RayContainer.h>
#include <SphereGeometry.h>
#include <SphereContainer.h>
#include <ToString.h>

#include <sstream>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/transform.h>

// *** DacrtNode ***

std::string DacrtNode::ToString() const {
    std::ostringstream out;
    out << "[rays: [" << rayStart << " -> " << rayEnd << "], spheres: [" << sphereStart << " -> " << sphereEnd << "]]";
    return out.str();
}

// *** DacrtNodes ***

DacrtNodes::DacrtNodes(const size_t capacity) 
    : scan1(capacity+1), scan2(capacity+1), 
      rayPartitions(capacity), spherePartitions(capacity),
      nextRayPartitions(capacity), nextSpherePartitions(capacity),
      doneRayPartitions(capacity), doneSpherePartitions(capacity) {
    scan1[0] = scan2[0] = 0;
    rayPartitions.resize(0); spherePartitions.resize(0);
    nextRayPartitions.resize(0); nextSpherePartitions.resize(0);
    doneRayPartitions.resize(0); doneSpherePartitions.resize(0);
}

void DacrtNodes::Reset() {
    rayPartitions.resize(0);
    nextRayPartitions.resize(0);
    spherePartitions.resize(0);
    nextSpherePartitions.resize(0);
    doneRayPartitions.resize(0);
    doneSpherePartitions.resize(0);
}

struct CalcSplitInfo {
    __host__ __device__
    thrust::tuple<Axis, float> operator()(thrust::tuple<float2, float2, float2, float2, float2> val) {
        float2 x = thrust::get<0>(val);
        float range = x.y - x.x;
        Axis axis = X;
        float split = (x.y + x.x) * 0.5f;
        
        float2 y = thrust::get<1>(val);
        float yRange = y.y - y.x;
        if (range < yRange) {
            axis = Y;
            split = (y.y + y.x) * 0.5f;
        }
        
        float2 z = thrust::get<2>(val);
        float zRange = z.y - z.x;
        if (range < zRange) {
            axis = Z;
            split = (z.y + z.x) * 0.5f;
        }
        
        float2 u = thrust::get<3>(val);
        float uRange = u.y - u.x;
        if (range < uRange) {
            axis = U;
            split = (u.y + u.x) * 0.5f;
        }

        float2 v = thrust::get<4>(val);
        float vRange = v.y - v.x;
        if (range < vRange) {
            axis = V;
            split = (v.y + v.x) * 0.5f;
        }
        
        return thrust::tuple<Axis, float>(axis, split);
    }
};

struct RayPartitionSide {
    float4 *rayOrigins, *rayAxisUVs;
    
    Axis *splitAxis;
    float *splitValues;
    
    // result
    PartitionSide *partitionSides;
    
    RayPartitionSide(thrust::device_vector<Axis>& axis, thrust::device_vector<float>& values)
        : splitAxis(thrust::raw_pointer_cast(axis.data())), 
          splitValues(thrust::raw_pointer_cast(values.data())) {}

    RayPartitionSide(HyperRays::Iterator rays,
                     thrust::device_vector<Axis>& axis, thrust::device_vector<float>& values,
                     thrust::device_vector<PartitionSide>& sides)
        : rayOrigins(RawPointer(HyperRays::GetOrigins(rays))),
          rayAxisUVs(RawPointer(HyperRays::GetAxisUVs(rays))),
          splitAxis(RawPointer(axis)), 
          splitValues(RawPointer(values)),
          partitionSides(RawPointer(sides)) {}
    
    __host__ __device__
    PartitionSide operator()(thrust::tuple<thrust::tuple<float4, float4>, unsigned int> ray) {
        int owner = thrust::get<1>(ray);
        Axis axis = splitAxis[owner];
        float splitVal = splitValues[owner];
        
        float rayVals[5];
        float4 origin = thrust::get<0>(thrust::get<0>(ray));
        rayVals[0] = origin.x;
        rayVals[1] = origin.y;
        rayVals[2] = origin.z;
        
        float4 UV = thrust::get<1>(thrust::get<0>(ray));
        rayVals[3] = UV.y;
        rayVals[4] = UV.z;
        
        return rayVals[axis] <= splitVal ? LEFT : RIGHT;
    }
    
    __host__ __device__
    void operator()(const unsigned int index, const unsigned int owner) const {
        const Axis axis = splitAxis[owner];
        const float splitVal = splitValues[owner];

        // IDEA since most owners will cover a warp or more, perhaps it will be
        // slightly faster to branch on (axis < 3) and avoid a memory lookup?
        // Only ever so slightly though.
        float rayVals[5];
        const float3 origin = make_float3(rayOrigins[index]);
        rayVals[0] = origin.x;
        rayVals[1] = origin.y;
        rayVals[2] = origin.z;
        
        const float3 axisUV = make_float3(rayAxisUVs[index]);
        rayVals[3] = axisUV.y;
        rayVals[4] = axisUV.z;
        
        partitionSides[index]  = rayVals[axis] <= splitVal ? LEFT : RIGHT;
    }
    
};

template <int S>
struct SideToOne {
    __host__ __device__ unsigned int operator()(PartitionSide s) { return s & S ? 1 : 0; }
};
static SideToOne<LEFT> leftToOne;
static SideToOne<RIGHT> rightToOne;
static thrust::plus<unsigned int> plus;
struct BoolToInt { __host__ __device__ unsigned int operator()(bool b) { return (int)b; } };

struct CreateCones {
    __host__ __device__
    Cone operator()(const thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> c) const {
        const HyperCube cube(thrust::get<0>(c), thrust::get<1>(c), thrust::get<2>(c),
                             thrust::get<3>(c), thrust::get<4>(c), thrust::get<5>(c));
        
        return Cone::FromCube(cube);
    }
};
static CreateCones createCones;

__constant__ unsigned int d_oldCubeCount;

struct CubesFromSplitPlanes {
    SignedAxis* a;
    float2 *x, *y, *z, *u, *v;
    Axis* splitAxis;
    float* splitValues;
    
    CubesFromSplitPlanes(HyperCubes& cubes, thrust::device_vector<Axis>& sAxis,
                         thrust::device_vector<float>& sValues) 
        : a(thrust::raw_pointer_cast(cubes.a.data())), 
          x(thrust::raw_pointer_cast(cubes.x.data())), 
          y(thrust::raw_pointer_cast(cubes.y.data())), 
          z(thrust::raw_pointer_cast(cubes.z.data())), 
          u(thrust::raw_pointer_cast(cubes.u.data())), 
          v(thrust::raw_pointer_cast(cubes.v.data())),
          splitAxis(thrust::raw_pointer_cast(sAxis.data())),
          splitValues(thrust::raw_pointer_cast(sValues.data())) {
        unsigned int oldCubeCount = cubes.Size();
        cudaMemcpyToSymbol(d_oldCubeCount, &oldCubeCount, sizeof(unsigned int));
    }
    
    __host__ __device__
    thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> operator()(const unsigned int threadId) const {
        const unsigned int oldCubeId = threadId % d_oldCubeCount;
        const PartitionSide side = threadId < d_oldCubeCount ? LEFT : RIGHT;
        const Axis sAxis = splitAxis[oldCubeId];
        const float splitValue = splitValues[oldCubeId];
        return thrust::tuple<SignedAxis, float2, float2, float2, float2, float2>
            (a[oldCubeId], 
             CalcBounds(sAxis == X, side, x[oldCubeId], splitValue),
             CalcBounds(sAxis == Y, side, y[oldCubeId], splitValue),
             CalcBounds(sAxis == Z, side, z[oldCubeId], splitValue),
             CalcBounds(sAxis == U, side, u[oldCubeId], splitValue),
             CalcBounds(sAxis == V, side, v[oldCubeId], splitValue));
    }
    __host__ __device__
    inline float2 CalcBounds(const bool split, const PartitionSide side, const float2 bounds, const float splitVal) const {
        return split ? make_float2(side == LEFT ? bounds.x : splitVal,
                                   side == RIGHT ? bounds.y : splitVal) : bounds;
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
    PartitionSide operator()(const unsigned int sphereId, const unsigned int owner) const {
        const Sphere sphere = spheres[sphereId];
        
        const Cone leftCone = cones[owner];
        PartitionSide side = leftCone.DoesIntersect(sphere) ? LEFT : NONE;
        
        const Cone rightCone = cones[owner + d_oldCubeCount];
        return (PartitionSide)(side | (rightCone.DoesIntersect(sphere) ? RIGHT : NONE));
    }
};

__constant__ unsigned int d_spheresMovedLeft;
struct AddSpheresMovedLeft {
    AddSpheresMovedLeft(thrust::device_vector<unsigned int>& leftIndices){
        unsigned int* spheresMovedLeft = thrust::raw_pointer_cast(leftIndices.data()) + leftIndices.size()-1;
        cudaMemcpyToSymbol(d_spheresMovedLeft, spheresMovedLeft, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
    }
    __device__
    unsigned int operator()(const unsigned int v) const {
        return v + d_spheresMovedLeft;
    }
};

__constant__ unsigned int d_raysMovedLeft;

struct ComputeNewNodePartitions {
    unsigned int* rayLeftIndices;
    unsigned int *sphereLeftIndices, *sphereRightIndices;
    uint2 *rayPartitions, *spherePartitions;
    
    ComputeNewNodePartitions(thrust::device_vector<unsigned int>& rLeftIndices,
                             thrust::device_vector<unsigned int>& sLeftIndices,
                             thrust::device_vector<unsigned int>& sRightIndices)
        : rayLeftIndices(thrust::raw_pointer_cast(rLeftIndices.data())),
          sphereLeftIndices(thrust::raw_pointer_cast(sLeftIndices.data())),
          sphereRightIndices(thrust::raw_pointer_cast(sRightIndices.data())) {
        unsigned int* data = thrust::raw_pointer_cast(rLeftIndices.data()) + rLeftIndices.size()-1;
        cudaMemcpyToSymbol(d_raysMovedLeft, data, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
    }

    __device__
    thrust::tuple<uint4, uint4> operator()(const uint2 rayPartition, const uint2 spherePartition) const {
        uint4 rays;
        const unsigned int rBegin = rays.x = rayLeftIndices[rayPartition.x];
        const unsigned int rEnd = rays.y = rayLeftIndices[rayPartition.y];
        rays.z = rayPartition.x - rBegin + d_raysMovedLeft;
        rays.w = rayPartition.y - rEnd + d_raysMovedLeft;

        uint4 sphere;
        sphere.x = sphereLeftIndices[spherePartition.x];
        sphere.y = sphereLeftIndices[spherePartition.y];
        sphere.z = sphereRightIndices[spherePartition.x];
        sphere.w = sphereRightIndices[spherePartition.y];
        
        return thrust::tuple<uint4, uint4>(rays, sphere);
    }

};

struct ComputeNewLeftNodePartitions {
    unsigned int* rayLeftIndices;
    unsigned int* sphereLeftIndices;
    unsigned int* sphereRightIndices;
    
    ComputeNewLeftNodePartitions(thrust::device_vector<unsigned int>& rLeftIndices,
                                 thrust::device_vector<unsigned int>& sLeftIndices,
                                 thrust::device_vector<unsigned int>& sRightIndices)
        : rayLeftIndices(thrust::raw_pointer_cast(rLeftIndices.data())),
          sphereLeftIndices(thrust::raw_pointer_cast(sLeftIndices.data())),
          sphereRightIndices(thrust::raw_pointer_cast(sRightIndices.data())) {
        unsigned int* data = thrust::raw_pointer_cast(rLeftIndices.data()) + rLeftIndices.size()-1;
        cudaMemcpyToSymbol(d_raysMovedLeft, data, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
    }
    __device__
    thrust::tuple<uint2, uint2> operator()(const uint2 rayPartition, const uint2 spherePartition) const {
        uint2 rays;
        rays.x = rayLeftIndices[rayPartition.x];
        rays.y = rayLeftIndices[rayPartition.y];

        uint2 sphere;
        sphere.x = sphereLeftIndices[spherePartition.x];
        sphere.y = sphereLeftIndices[spherePartition.y];
        
        return thrust::tuple<uint2, uint2>(rays, sphere);
    }
    
};

struct ComputeNewRightNodePartitions {
    unsigned int* rayLeftIndices;
    unsigned int* sphereLeftIndices;
    unsigned int* sphereRightIndices;
    
    ComputeNewRightNodePartitions(thrust::device_vector<unsigned int>& rLeftIndices,
                                  thrust::device_vector<unsigned int>& sLeftIndices,
                                  thrust::device_vector<unsigned int>& sRightIndices)
        : rayLeftIndices(thrust::raw_pointer_cast(rLeftIndices.data())),
          sphereLeftIndices(thrust::raw_pointer_cast(sLeftIndices.data())),
          sphereRightIndices(thrust::raw_pointer_cast(sRightIndices.data())) {
        unsigned int* data = thrust::raw_pointer_cast(rLeftIndices.data()) + rLeftIndices.size()-1;
        cudaMemcpyToSymbol(d_raysMovedLeft, data, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
    }
    __device__
    thrust::tuple<uint2, uint2> operator()(const uint2 rayPartition, const uint2 spherePartition) const {
        uint2 rays;
        const unsigned int rBegin = rayLeftIndices[rayPartition.x];
        const unsigned int rEnd = rayLeftIndices[rayPartition.y];
        rays.x = rayPartition.x - rBegin + d_raysMovedLeft;
        rays.y = rayPartition.y - rEnd + d_raysMovedLeft;

        uint2 sphere;
        sphere.x = sphereRightIndices[spherePartition.x];
        sphere.y = sphereRightIndices[spherePartition.y];
        
        return thrust::tuple<uint2, uint2>(rays, sphere);
    }
};

void DacrtNodes::Partition(RayContainer& rays, SphereContainer& spheres,
                           HyperCubes& cubes) {

    // TODO move static left and right indices vectors to global scope? Do I
    // need more than one at a time? 
    /// No! But I need to split next ray and sphere partition creation.

    size_t rayCount = rays.InnerSize();

    // Calculate splitting info
    static thrust::device_vector<Axis> splitAxis(cubes.Size());
    splitAxis.resize(cubes.Size());
    static thrust::device_vector<float> splitValues(cubes.Size());
    splitValues.resize(cubes.Size());
    thrust::zip_iterator<thrust::tuple<AxisIterator, FloatIterator> > axisInfo
        = thrust::make_zip_iterator(thrust::make_tuple(splitAxis.begin(), splitValues.begin()));

    CalcSplitInfo calcSplitInfo;    
    thrust::transform(cubes.BeginBounds(), cubes.EndBounds(), axisInfo, calcSplitInfo);    

    // Calculate the partition side
    static thrust::device_vector<PartitionSide> rayPartitionSides(rayCount);
    rayPartitionSides.resize(rayCount);

    // Calculate current ray owners.
#if 0
    static thrust::device_vector<unsigned int> rayOwners(rayCount);
    rayOwners.resize(rayCount);
    CalcOwners(rayPartitions, rayOwners);
    thrust::zip_iterator<thrust::tuple<HyperRays::Iterator, UintIterator> > raysWithOwners
        = thrust::make_zip_iterator(thrust::make_tuple(rays.BeginInnerRays(), rayOwners.begin()));

    RayPartitionSide rayPartitionSide = RayPartitionSide(splitAxis, splitValues);
    thrust::transform(raysWithOwners, raysWithOwners + rayCount, 
                      rayPartitionSides.begin(), rayPartitionSide);
#else
    RayPartitionSide rayPartitionSide = RayPartitionSide(rays.BeginInnerRays(), splitAxis, splitValues,
                                                         rayPartitionSides);
    ForEachWithOwners(rayPartitions, 0, rayPartitions.size(), 
                      rayCount, rayPartitionSide);
#endif
    
    // Calculate the indices for the rays moved left using scan
    static thrust::device_vector<unsigned int> rayLeftIndices(rayCount+1);
    rayLeftIndices.resize(rayCount+1);
    rayLeftIndices[0] = 0; // Should be handled by resize not being destructive.
    thrust::transform_inclusive_scan(rayPartitionSides.begin(), rayPartitionSides.end(),
                                     rayLeftIndices.begin()+1, leftToOne, plus);
    
    // Scatter the rays
    rays.Partition(rayPartitionSides, rayLeftIndices);

    // Calculate the new hypercubes 
    /// IDEA: Since the rays have been scattered, just reduce them, but that
    // would mean also scattering the hypercubes when creating leaves.
    static HyperCubes splitCubes(cubes.Size() * 2);
    splitCubes.Resize(cubes.Size() * 2);
    CubesFromSplitPlanes cubesFromSplitPlanes(cubes, splitAxis, splitValues);
    thrust::transform(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(cubes.Size() * 2), 
                      splitCubes.Begin(), cubesFromSplitPlanes);
    
    // Calculate the cones used for splitting 
    // TODO using knowledge about the cube split, the resulting two cones can be
    // computed faster if computed together in one thread.
    static thrust::device_vector<Cone> cones(cubes.Size());
    cones.resize(cubes.Size() * 2);
    thrust::transform(splitCubes.Begin(), splitCubes.End(), cones.begin(), createCones);

    // Calculate current sphere owners. TODO Use a work queue instead
    static thrust::device_vector<unsigned int> sphereOwners(spheres.CurrentSize());
    sphereOwners.resize(spheres.CurrentSize());
    CalcOwners(spherePartitions, sphereOwners);

    // Calculate sphere partitions
    static thrust::device_vector<PartitionSide> spherePartitionSides(spheres.CurrentSize());
    spherePartitionSides.resize(spheres.CurrentSize());
    SpherePartitioningByCones spherePartitioningByCones(cones, spheres.SphereGeometry().spheres);
    thrust::transform(spheres.BeginCurrentIndices(), spheres.EndCurrentIndices(), sphereOwners.begin(),
                      spherePartitionSides.begin(), spherePartitioningByCones);

    static thrust::device_vector<unsigned int> sphereLeftIndices(spheres.CurrentSize()+1);
    sphereLeftIndices.resize(spheres.CurrentSize()+1);
    sphereLeftIndices[0] = 0; // Should be handled by resize not being destructive.
    static thrust::device_vector<unsigned int> sphereRightIndices(spheres.CurrentSize()+1);
    sphereRightIndices.resize(spheres.CurrentSize()+1);
    sphereRightIndices[0] = 0; // Should be handled by resize not being destructive.
    
    thrust::transform_inclusive_scan(spherePartitionSides.begin(), spherePartitionSides.end(),
                                     sphereLeftIndices.begin()+1, leftToOne, plus);
    thrust::transform_inclusive_scan(spherePartitionSides.begin(), spherePartitionSides.end(),
                                     sphereRightIndices.begin()+1, rightToOne, plus);

    AddSpheresMovedLeft addSpheresMovedLeft(sphereLeftIndices);
    thrust::transform(sphereRightIndices.begin(), sphereRightIndices.end(), sphereRightIndices.begin(), addSpheresMovedLeft);

    // Scatter spheres
    spheres.Partition(spherePartitionSides, sphereLeftIndices, sphereRightIndices);
    
    // Compute new dacrt node partitions
    unsigned int nextUnfinishedNodes = UnfinishedNodes() * 2;
    nextRayPartitions.resize(nextUnfinishedNodes);
    nextSpherePartitions.resize(nextUnfinishedNodes);
    
    // Wrap partitions in uint4 to be able to store both left and right
    // simultaneously and coallesced. (Hackish)
    // thrust::device_ptr<uint4> nextRays((uint4*)(void*)thrust::raw_pointer_cast(nextRayPartitions.data()));
    // thrust::device_ptr<uint4> nextSpheres((uint4*)(void*)thrust::raw_pointer_cast(nextSpherePartitions.data()));
    
    // thrust::zip_iterator<thrust::tuple<thrust::device_ptr<uint4>, thrust::device_ptr<uint4> > > partitionWrapper = 
    //     thrust::make_zip_iterator(thrust::make_tuple(nextRays, nextSpheres));
    // ComputeNewNodePartitions computeNewNodePartitions(rayLeftIndices, sphereLeftIndices, sphereRightIndices);
    // thrust::transform(BeginUnfinishedRayPartitions(), EndUnfinishedRayPartitions(), BeginUnfinishedSpherePartitions(),
    //                   partitionWrapper, computeNewNodePartitions);

    thrust::zip_iterator<thrust::tuple<Uint2Iterator, Uint2Iterator > > partitionWrapper = 
        thrust::make_zip_iterator(thrust::make_tuple(nextRayPartitions.begin(), nextSpherePartitions.begin()));

    ComputeNewLeftNodePartitions computeNewLeftNodePartitions(rayLeftIndices, sphereLeftIndices, sphereRightIndices);
    thrust::transform(BeginUnfinishedRayPartitions(), EndUnfinishedRayPartitions(), BeginUnfinishedSpherePartitions(),
                      partitionWrapper, computeNewLeftNodePartitions);
    ComputeNewRightNodePartitions computeNewRightNodePartitions(rayLeftIndices, sphereLeftIndices, sphereRightIndices);
    thrust::transform(BeginUnfinishedRayPartitions(), EndUnfinishedRayPartitions(), BeginUnfinishedSpherePartitions(),
                      partitionWrapper+UnfinishedNodes(), computeNewRightNodePartitions);
    
    rayPartitions.swap(nextRayPartitions);
    spherePartitions.swap(nextSpherePartitions);
}


// *** LEAF PARTITIONING ***

struct IsNodeLeaf {
    __host__ __device__
    bool operator()(const uint2 rayPartition, const uint2 spherePartition) const {
        const float rayCount = (float)(rayPartition.y - rayPartition.x);
        const float sphereCount = (float)(spherePartition.y - spherePartition.x);
        
        return rayCount * sphereCount <= 16.0f * (rayCount + sphereCount);
    }
};

struct MarkLeafSize {
    __host__ __device__
    unsigned int operator()(const thrust::tuple<bool, uint2> input) const {
        bool isLeaf = thrust::get<0>(input);
        uint2 rayPartition = thrust::get<1>(input);
        return isLeaf ? rayPartition.y - rayPartition.x : 0;
    }
};

__constant__ unsigned int d_leafPartitionOffset;
struct NewPrimPartitions {
    uint2 *oldPartitions;
    unsigned int *leafIndices;
    bool *isLeafs;

    unsigned int *newBegins;
    uint2 *nextPartitions, *leafPartitions;

    NewPrimPartitions(thrust::device_vector<uint2>::iterator oPartitions,
                      thrust::device_vector<unsigned int>& lIndices,
                      thrust::device_vector<bool>& isLeafs,
                      thrust::device_vector<unsigned int>& nBegins,
                      thrust::device_vector<uint2>& nPartitions,
                      const unsigned int leafPartitionOffset, 
                      thrust::device_vector<uint2>& lPartitions,
                      const unsigned int leafOffset) 
        : oldPartitions(RawPointer(oPartitions)),
          leafIndices(RawPointer(lIndices)),
          isLeafs(RawPointer(isLeafs)),
          newBegins(RawPointer(nBegins)),
          nextPartitions(RawPointer(nPartitions)),
          leafPartitions(RawPointer(lPartitions) + leafOffset) {
        cudaMemcpyToSymbol(d_leafPartitionOffset, &leafPartitionOffset, sizeof(unsigned int));
    }
    
    __host__ __device__
    void operator()(const unsigned int threadId) const {
        const uint2 oldPartition = oldPartitions[threadId];
        const unsigned int range = oldPartition.y - oldPartition.x;
        const bool isLeaf = isLeafs[threadId];
        unsigned int newBegin = newBegins[oldPartition.x];
        newBegin += isLeaf ? d_leafPartitionOffset : 0;
        const uint2 partition = make_uint2(newBegin, newBegin + range);
        const unsigned int leafIndex = leafIndices[threadId];
        const unsigned int index = isLeaf ? leafIndex : threadId - leafIndex;
        uint2* output = isLeaf ? leafPartitions : nextPartitions;
        output[index] = partition;
    }
    
};

bool DacrtNodes::PartitionLeafs(RayContainer& rays, SphereContainer& spheres) {
    static thrust::device_vector<bool> isLeaf(UnfinishedNodes());
    isLeaf.resize(UnfinishedNodes());

    size_t unfinishedNodes = UnfinishedNodes();

    // TODO make isLeaf unsigned int and reuse for indices? isLeaf info is
    // stored in an index and it's neighbour.
    thrust::transform(BeginUnfinishedRayPartitions(), EndUnfinishedRayPartitions(), BeginUnfinishedSpherePartitions(),
                      isLeaf.begin(), IsNodeLeaf());
    // std::cout << "Leaf nodes:\n" << isLeaf << std::endl;

    static thrust::device_vector<unsigned int> leafIndices(UnfinishedNodes()+1);
    leafIndices.resize(unfinishedNodes+1);
    leafIndices[0] = 0;
    thrust::transform_inclusive_scan(isLeaf.begin(), isLeaf.end(), leafIndices.begin()+1, 
                                     BoolToInt(), plus);
    const unsigned int newLeafNodes = leafIndices[leafIndices.size()-1];
    const unsigned int oldLeafNodes = DoneNodes();

    if (newLeafNodes == 0) return false;

    // Partition rays
    static thrust::device_vector<unsigned int> rayLeafNodeIndices(unfinishedNodes+1); // TODO could be a globally static vector
    rayLeafNodeIndices.resize(unfinishedNodes+1);
    rayLeafNodeIndices[0] = 0;
    thrust::zip_iterator<thrust::tuple<BoolIterator, Uint2Iterator> > leafNodeValues =
        thrust::make_zip_iterator(thrust::make_tuple(isLeaf.begin(), BeginUnfinishedRayPartitions()));
    thrust::transform_inclusive_scan(leafNodeValues, leafNodeValues + unfinishedNodes, 
                                     rayLeafNodeIndices.begin()+1, MarkLeafSize(), plus);
    // std::cout << "Ray Leaf Node Indices:\n" << rayLeafNodeIndices << std::endl;

    static thrust::device_vector<unsigned int> owners(rays.InnerSize());
    owners.resize(rays.InnerSize());
    CalcOwners(rayPartitions, owners);
    
    const unsigned int oldRayLeafs = rays.LeafRays();
    rays.PartitionLeafs(isLeaf, rayLeafNodeIndices, rayPartitions, owners);
    // Owners now hold the new ray begin indices
    thrust::device_vector<unsigned int>& newRayIndices = owners;
    
    // New node ray partitions
    nextRayPartitions.resize(rayPartitions.size() - newLeafNodes);
    doneRayPartitions.resize(doneRayPartitions.size() + newLeafNodes);

    thrust::zip_iterator<thrust::tuple<Uint2Iterator, UintIterator, BoolIterator> > nodePartitionsInput =
        thrust::make_zip_iterator(thrust::make_tuple(BeginUnfinishedRayPartitions(), leafIndices.begin(), isLeaf.begin()));
    NewPrimPartitions newPrimPartitions(BeginUnfinishedRayPartitions(), leafIndices, isLeaf,
                                        newRayIndices, nextRayPartitions, oldRayLeafs, doneRayPartitions, oldLeafNodes);
    thrust::for_each(thrust::counting_iterator<unsigned int>(0), 
                     thrust::counting_iterator<unsigned int>(unfinishedNodes),
                     newPrimPartitions);
    
    rayPartitions.swap(nextRayPartitions);

    // Partition spheres
    static thrust::device_vector<unsigned int> sphereLeafNodeIndices(unfinishedNodes+1); // TODO could be a globally static vector
    sphereLeafNodeIndices.resize(unfinishedNodes+1);
    sphereLeafNodeIndices[0] = 0;
    leafNodeValues = thrust::make_zip_iterator(thrust::make_tuple(isLeaf.begin(), BeginUnfinishedSpherePartitions()));
    thrust::transform_inclusive_scan(leafNodeValues, leafNodeValues + unfinishedNodes, 
                                     sphereLeafNodeIndices.begin()+1, MarkLeafSize(), plus);

    owners.resize(spheres.CurrentSize());
    CalcOwners(spherePartitions, owners);

    const unsigned int oldSphereLeafs = spheres.DoneSize();
    spheres.PartitionLeafs(isLeaf, sphereLeafNodeIndices, spherePartitions, owners);

    // New node sphere partitions
    nextSpherePartitions.resize(spherePartitions.size() - newLeafNodes);
    doneSpherePartitions.resize(doneSpherePartitions.size() + newLeafNodes);
    nodePartitionsInput = thrust::make_zip_iterator(thrust::make_tuple(BeginUnfinishedSpherePartitions(), leafIndices.begin(), isLeaf.begin()));
    newPrimPartitions = NewPrimPartitions(BeginUnfinishedSpherePartitions(), leafIndices, isLeaf,
                                          owners, nextSpherePartitions, oldSphereLeafs, doneSpherePartitions, oldLeafNodes);
    thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                     thrust::counting_iterator<unsigned int>(unfinishedNodes),
                     newPrimPartitions);
    
    spherePartitions.swap(nextSpherePartitions);

    return true;
}



// *** EXHAUSTIVE INTERSECTION ***

struct ExhaustiveIntersection {
    float4 *rayOrigins, *rayAxisUVs;
    uint2 *spherePartitions;
    unsigned int *sphereIndices;
    Sphere *spheres;

    unsigned int *hitIDs;
    
    ExhaustiveIntersection(HyperRays& rays, 
                           thrust::device_vector<uint2>& sPartitions,
                           thrust::device_vector<unsigned int>& sIndices, 
                           thrust::device_vector<Sphere>& ss,
                           thrust::device_vector<unsigned int>& hits)
        : rayOrigins(thrust::raw_pointer_cast(rays.origins.data())),
          rayAxisUVs(thrust::raw_pointer_cast(rays.axisUVs.data())),
          spherePartitions(thrust::raw_pointer_cast(sPartitions.data())), 
          sphereIndices(thrust::raw_pointer_cast(sIndices.data())), 
          spheres(thrust::raw_pointer_cast(ss.data())),
          hitIDs(thrust::raw_pointer_cast(hits.data()))
    {}

    /**
     * Takes a ray as argument and intersects it against all spheres referenced
     * by its parent DacrtNode.
     *
     * Returns the index of the intersected sphere and stores the distance to it
     * in the w component of the ray's direction.
     */
    __host__ __device__
    void operator()(const unsigned int index, const unsigned int owner) const {
        const float3 origin = make_float3(rayOrigins[index]);
        const float3 dir = normalize(HyperRay::AxisUVToDirection(make_float3(rayAxisUVs[index])));
        
        const uint2 spherePartition = spherePartitions[owner];
        float hitT = 1e30f;
        unsigned int hitID = SpheresGeometry::MISSED;
        
        for (unsigned int g = spherePartition.x; g < spherePartition.y; ++g) {
            const unsigned int sphereId = sphereIndices[g];
            const Sphere s = spheres[sphereId];
            const float t = s.Intersect(origin, dir);
            if (0 < t && t < hitT) {
                hitID = sphereId;
                hitT = t;
            }
        }

        rayAxisUVs[index] = make_float4(dir, hitT);
        hitIDs[index] = hitID;
    }
};

void DacrtNodes::ExhaustiveIntersect(RayContainer& rays, SphereContainer& spheres, 
                                     thrust::device_vector<unsigned int>& hits) {
    
    // std::cout << "ExhaustiveIntersect" << std::endl;
    hits.resize(rays.LeafRays());

    // std::cout << "doneRayPartitions:\n" << doneRayPartitions <<std::endl;

    ExhaustiveIntersection exhaustive(rays.leafRays, 
                                      doneSpherePartitions, 
                                      spheres.doneIndices, 
                                      spheres.spheres.spheres,
                                      hits);
    ForEachWithOwners(doneRayPartitions, 0, doneRayPartitions.size(),
                      hits.size(), exhaustive);

    //std::cout << "hits:\n" << hits << std::endl;
}

// *** CALC OWNERS ***

struct SetMarkers {
    unsigned int* owners;
    uint2* partitions;
    SetMarkers(thrust::device_vector<unsigned int>& owners,
               thrust::device_vector<uint2>& partitions)
        : owners(thrust::raw_pointer_cast(owners.data())),
          partitions(thrust::raw_pointer_cast(partitions.data())) {}
    
    __host__ __device__
    void operator()(const unsigned int threadId) const {
        const uint2 part = partitions[threadId];
        owners[part.x] = threadId == 0 ? 0 : 1;
    }
};

struct WriteOwner {
    unsigned int* owners;
    
    WriteOwner(thrust::device_vector<unsigned int>& os) 
        : owners(thrust::raw_pointer_cast(os.data())) {}

    __device__
    void operator()(const unsigned int index, const unsigned int owner) const {
        owners[index] = owner;
    }
};


void DacrtNodes::CalcOwners(thrust::device_vector<uint2>& partitions,
                            thrust::device_vector<unsigned int>& owners) {
    size_t nodes = partitions.size();
    // std::cout << "owner nodes: " << nodes << std::endl;
    thrust::fill(owners.begin(), owners.end(), 0);

    if (nodes == 1) return;
    
    // TODO just fill the first 0 0 0 owners? Then start the scan at first
    // marker. The decision wether or not to do this would be 
    /// owners.size() / nodes > X
    // for some sane X.

    SetMarkers setMarkers(owners, partitions);
    thrust::counting_iterator<unsigned int> threadIds(0);
    thrust::for_each(threadIds, threadIds + nodes, setMarkers);
    // std::cout << "markers:\n" << owners << std::endl;

    thrust::inclusive_scan(owners.begin(), owners.end(), owners.begin());

    // std::cout << "owners:\n" << owners << std::endl;
}


void DacrtNodes::ResizeUnfinished(const size_t size) {
    rayPartitions.resize(size);
    spherePartitions.resize(size);
}


std::string DacrtNodes::ToString() const {
    std::ostringstream out;
    if (UnfinishedNodes() > 0) {
        out << "Unfinished DacrtNodes:";
        for (size_t i = 0; i < UnfinishedNodes(); ++i)
            out << "\n" << i << ": " << GetUnfinished(i);
        if (DoneNodes() > 0) out << "\n";
    }
    if (DoneNodes() > 0) {
        out << "Done DacrtNodes:";
        for (size_t i = 0; i < DoneNodes(); ++i)
            out << "\n" << i << ": " << GetDone(i);
    }
    return out.str();
}

std::string DacrtNodes::ToString(RayContainer& rays, SphereContainer& spheres) const {
    std::ostringstream out;
    if (UnfinishedNodes() > 0) {
        out << "Unfinished DacrtNodes:";
        for (size_t i = 0; i < UnfinishedNodes(); ++i) {
            DacrtNode node = GetUnfinished(i);
            out << "\n" << i << ": " << node << "\n  Rays: ";
            for (unsigned int r = node.rayStart; r < node.rayEnd; ++r){
                float4 origins = *(HyperRays::GetOrigins(rays.BeginInnerRays()) + r);
                out << origins.w << ", ";
            }
            out << "\n  Spheres: ";
            for (unsigned int s = node.sphereStart; s < node.sphereEnd; ++s){
                unsigned int sphereId = *(spheres.BeginCurrentIndices() + s);
                out << sphereId << ", ";
            }
        }
        if (DoneNodes() > 0) out << "\n";
    }

    if (DoneNodes() > 0) {
        out << "Done DacrtNodes:";
        for (size_t i = 0; i < DoneNodes(); ++i) {
            DacrtNode node = GetDone(i);
            out << "\n" << i << ": " << node << "\n  Rays: ";
            for (unsigned int r = node.rayStart; r < node.rayEnd; ++r){
                float4 origins = *(HyperRays::GetOrigins(rays.BeginLeafRays()) + r);
                out << origins.w << ", ";
            }
            out << "\n  Spheres: ";
            for (unsigned int s = node.sphereStart; s < node.sphereEnd; ++s){
                unsigned int sphereId = *(spheres.BeginDoneIndices() + s);
                out << sphereId << ", ";
            }
        }
    }
    return out.str();
    
}
