// DACRT node
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_NODE_H_
#define _GPU_DACRT_NODE_H_

#include <thrust/device_vector.h>

#include <Utils/Utils.h>

class HyperCubes;
class HyperRays;
class RayContainer;
class SpheresGeometry;
class SphereContainer;

struct DacrtNode {
    unsigned int rayStart, rayEnd;
    unsigned int sphereStart, sphereEnd;

    DacrtNode(uint2 rayPartition, uint2 spherePartition) 
        : rayStart(rayPartition.x), rayEnd(rayPartition.y),
          sphereStart(spherePartition.x), sphereEnd(spherePartition.y) {}
    
    DacrtNode(unsigned int rayStart, unsigned int rayEnd,
              unsigned int sphereStart, unsigned int sphereEnd)
        : rayStart(rayStart), rayEnd(rayEnd),
          sphereStart(sphereStart), sphereEnd(sphereEnd) {}
    
    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const DacrtNode& n){
    return s << n.ToString();
}



class DacrtNodes {
private:
    thrust::device_vector<unsigned int> scan1;
    thrust::device_vector<unsigned int> scan2;
public:
    // TODO partition end always equal the next partitions start, so we can
    // reduce the uint2 to an uint, reducing memory overhead.
    
    thrust::device_vector<uint2> rayPartitions;
    thrust::device_vector<uint2> spherePartitions;
    thrust::device_vector<uint2> nextRayPartitions;
    thrust::device_vector<uint2> nextSpherePartitions;
    
    thrust::device_vector<uint2> doneRayPartitions;
    thrust::device_vector<uint2> doneSpherePartitions;

public:

    DacrtNodes(const size_t capacity);

    void Reset();

    void Partition(RayContainer& rays, SphereContainer& spheres,
                   HyperCubes& initialCubes);

    /**
     * Determines which nodes should be leafs and partitions the nodes, rays and spheres accordingly.
     *
     * Returns false if no leafnodes were found.
     */
    bool PartitionLeafs(RayContainer& rays, SphereContainer& spheres);

    /**
     * Takes a list of rays and geometry and intersects them by the partitioning
     * specified in the DacrtNodes.
     *
     * The results of these intersections are stored in
     * hits. ExhaustiveIntersect will increase the size of hits if needed.
     */
    void ExhaustiveIntersect(RayContainer& rays, SphereContainer& spheres, 
                             thrust::device_vector<unsigned int>& hits);

    inline thrust::device_vector<uint2>::iterator BeginUnfinishedRayPartitions() { return rayPartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndUnfinishedRayPartitions() { return rayPartitions.end(); }
    inline thrust::device_vector<uint2>::iterator BeginUnfinishedSpherePartitions() { return spherePartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndUnfinishedSpherePartitions() { return spherePartitions.end(); }
    inline size_t UnfinishedNodes() const { return rayPartitions.size(); }
    void ResizeUnfinished(const size_t size);

    inline thrust::device_vector<uint2>::iterator BeginDoneRayPartitions() { return doneRayPartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndDoneRayPartitions() { return doneRayPartitions.end(); }
    inline size_t DoneNodes() const { return doneRayPartitions.size(); }

    inline DacrtNode GetUnfinished(const size_t i) const {
        return DacrtNode(rayPartitions[i], spherePartitions[i]);
    }
    inline void SetUnfinished(const size_t i, const DacrtNode& n) {
        if (UnfinishedNodes() <= i) ResizeUnfinished(i+1);
        rayPartitions[i] = make_uint2(n.rayStart, n.rayEnd);
        spherePartitions[i] = make_uint2(n.sphereStart, n.sphereEnd);
    }

    inline DacrtNode GetDone(const size_t i) const {
        return DacrtNode(doneRayPartitions[i], doneSpherePartitions[i]);
    }

    /**
     * TODO Partitions should be a begin and end iterator. Change it when I
     * implement a custom kernel instead of using thrust's for_each.
     */
    static void CalcOwners(thrust::device_vector<uint2>::iterator beginPartition,
                           thrust::device_vector<uint2>::iterator endPartition,
                           thrust::device_vector<unsigned int>& owners);

    std::string ToString() const;
    std::string ToString(RayContainer& rays, SphereContainer& spheres) const;
};

inline std::ostream& operator<<(std::ostream& s, const DacrtNodes& d){
    return s << d.ToString();
}


#endif // _GPU_DACRT_NODE_H_
