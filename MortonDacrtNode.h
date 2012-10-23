// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_MORTON_DACRT_NODE_H_
#define _GPU_MORTON_DACRT_NODE_H_

#include <IRayTracer.h>

#include <thrust/device_vector.h>

#include <string>
#include <ostream>

class HyperCubes;
class RayContainer;
class SpheresGeometry;
class SphereContainer;

class MortonDacrtNodes : public virtual IRayTracer {
    thrust::device_vector<uint2> rayPartitions;
    thrust::device_vector<uint2> nextRayPartitions;

    thrust::device_vector<uint2> spherePartitions;
    thrust::device_vector<uint2> nextSpherePartitions;
    unsigned int leafNodes;

    RayContainer* rays;
    thrust::device_vector<unsigned int> rayMortonCodes;
    SpheresGeometry* spheresGeom;
    thrust::device_vector<unsigned int> sphereIndices;
    thrust::device_vector<unsigned int> sphereIndexPartition;
    thrust::device_vector<unsigned int> nextSphereIndices;
    thrust::device_vector<unsigned int> nextSphereIndexPartition;
    unsigned int leafSphereIndices;


public:
    
    /**
     * Initializes the storage needed to run the Dacrt construction algorithm
     * using morton curves.
     */
    MortonDacrtNodes(const size_t capacity);

    /**
     * Partitions the rays and spheres into smaller partitions for ray tracing.
     */
    void Create(RayContainer& rayContainer, SpheresGeometry& SphereContainer);

    inline RayContainer* GetRayContainer() { return rays; }
    inline thrust::device_vector<unsigned int>::iterator SphereIndicesBegin() { return sphereIndices.begin(); }
    inline thrust::device_vector<unsigned int>::iterator SphereIndicesEnd() { return sphereIndices.end(); }
    
    inline thrust::device_vector<uint2>::iterator RayPartitionsBegin() { return rayPartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator RayPartitionsEnd() { return rayPartitions.end(); }

    inline thrust::device_vector<uint2>::iterator SpherePartitionsBegin() { return spherePartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator SpherePartitionsEnd() { return spherePartitions.end(); }

    void FindIntersections(thrust::device_vector<unsigned int>& hitIDs);
    
    std::string ToString(const bool verbose = false) const;

private:

    /**
     * Move leaf partitions to the front of the ray and sphere partition
     * vectors, does the same for sphereIndices and sphereIndexPartition, and
     * increment leafSphereIndices and leafNodes.
     *
     * Returns true if any new done nodes are created.
     */
    bool CreateLeafNodes(thrust::device_vector<unsigned int>& mortonCodes);
    
    /**
     * Initializes sphereIndices and spherePartitions.
     */
    void InitSphereIndices(HyperCubes& cubes, SpheresGeometry& spheres);
    
    inline thrust::device_vector<uint2>::iterator ActiveSpherePartitionsBegin() { return spherePartitions.begin() + leafNodes; }
    inline thrust::device_vector<uint2>::iterator ActiveSpherePartitionsEnd() { return spherePartitions.end(); }
    
    std::string PrintNode(const unsigned int id, const bool verbose = false) const;
};

inline std::ostream& operator<<(std::ostream& s, const MortonDacrtNodes& d){
    return s << d.ToString();
}

#endif
