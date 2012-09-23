// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_MORTON_DACRT_NODE_H_
#define _GPU_MORTON_DACRT_NODE_H_

#include <thrust/device_vector.h>

#include <string>
#include <ostream>

class HyperCubes;
class RayContainer;
class SpheresGeometry;
class SphereContainer;

class MortonDacrtNodes {
    thrust::device_vector<uint2> rayPartitions;
    thrust::device_vector<uint2> nextRayPartitions;

    thrust::device_vector<uint2> spherePartitions;
    thrust::device_vector<uint2> nextSpherePartitions;
    unsigned int doneSpherePartitions;

    RayContainer* rays;
    thrust::device_vector<unsigned int> sphereIndices;
    thrust::device_vector<unsigned int> sphereIndexPartition;
    thrust::device_vector<unsigned int> nextSphereIndices;
    unsigned int doneSphereIndices;


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
    
    inline thrust::device_vector<uint2>::iterator BeginRayPartitions() { return rayPartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndRayPartitions() { return rayPartitions.end(); }
    
    std::string ToString() const;

private:
    
    /**
     * Initializes sphereIndices and spherePartitions.
     */
    void InitSphereIndices(HyperCubes& cubes, SpheresGeometry& spheres);
};

inline std::ostream& operator<<(std::ostream& s, const MortonDacrtNodes& d){
    return s << d.ToString();
}

#endif
