// Sphere geometry container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SPHERE_CONTAINER_H_
#define _GPU_DACRT_SPHERE_CONTAINER_H_

#include <Enums.h>
#include <Utils/Utils.h>

#include <thrust/device_vector.h>

class HyperCubes;
class SpheresGeometry;

class SphereContainer {
public:
    SpheresGeometry& spheres;    

    thrust::device_vector<unsigned int> indices1;
    thrust::device_vector<unsigned int> indices2;
 
    thrust::device_vector<unsigned int>& currentIndices;
    thrust::device_vector<unsigned int>& nextIndices;

    thrust::device_vector<unsigned int> doneIndices;

public:
    SphereContainer(HyperCubes& cubes, SpheresGeometry& spheres,
                    unsigned int* spherePartitionStart);

    inline SpheresGeometry& SphereGeometry() { return spheres; }

    inline UintIterator BeginCurrentIndices() { return currentIndices.begin(); }
    inline UintIterator EndCurrentIndices() { return currentIndices.end(); }
    inline size_t CurrentSize() const { return currentIndices.size(); }

    inline UintIterator BeginDoneIndices() { return doneIndices.begin(); }
    inline UintIterator EndDoneIndices() { return doneIndices.end(); }
    inline size_t DoneSize() const { return doneIndices.size(); }

    /**
     * Partitions the sphere and right by the indices given.
     */
    void Partition(thrust::device_vector<PartitionSide>& partitionSides, 
                   thrust::device_vector<unsigned int>& leftIndices,
                   thrust::device_vector<unsigned int>& rightIndices);
    
    /**
     * Partitions the currently active spheres into leaf and non leaf nodes.
     */
    void PartitionLeafs(thrust::device_vector<bool>& isLeaf, 
                        thrust::device_vector<unsigned int>& leafNodeIndices, 
                        thrust::device_vector<uint2>& spherePartitions,
                        thrust::device_vector<unsigned int>& owners);
    
    std::string ToString() const;
};

#endif // _GPU_DACRT_SPHERE_CONTAINER_H_
