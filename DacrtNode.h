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

#include <Utils.h>

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
public:
    // TODO partition end always equal the next partitions start, so we can
    // reduce the uint2 to an uint, reducing memory overhead.
    thrust::device_vector<uint2> rayPartitions1; // [start, end]
    thrust::device_vector<uint2> spherePartitions1; // [start, end]
    thrust::device_vector<uint2> rayPartitions2; // [start, end]
    thrust::device_vector<uint2> spherePartitions2; // [start, end]
    
    thrust::device_vector<uint2>& rayPartitions;
    thrust::device_vector<uint2>& spherePartitions;
    thrust::device_vector<uint2>& nextRayPartitions;
    thrust::device_vector<uint2>& nextSpherePartitions;
    size_t unfinishedNodes; // currently used unfinished entries, not actual vector size
    
    thrust::device_vector<uint2> doneRayPartitions;
    thrust::device_vector<uint2> doneSpherePartitions;
    size_t doneNodes;

public:

    DacrtNodes(const size_t capacity) 
        : rayPartitions1(capacity), spherePartitions1(capacity),
          rayPartitions2(capacity), spherePartitions2(capacity),
          rayPartitions(rayPartitions1), spherePartitions(spherePartitions1),
          nextRayPartitions(rayPartitions2), nextSpherePartitions(spherePartitions2),
          unfinishedNodes(0), 
          doneRayPartitions(capacity), doneSpherePartitions(capacity), doneNodes(0) {}

    void Reset();

    // TODO Collect everything in a DacrtNode constructor
    // DacrtNodes(RayContainer& rays, SphereGeometry& spheres, SplitScheme split);

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
    inline thrust::device_vector<uint2>::iterator EndUnfinishedRayPartitions() { return rayPartitions.begin() + unfinishedNodes; }
    inline thrust::device_vector<uint2>::iterator BeginUnfinishedSpherePartitions() { return spherePartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndUnfinishedSpherePartitions() { return spherePartitions.begin() + unfinishedNodes; }
    inline size_t UnfinishedNodes() const { return unfinishedNodes; }
    void ResizeUnfinished(const size_t size);

    inline thrust::device_vector<uint2>::iterator BeginDoneRayPartitions() { return doneRayPartitions.begin(); }
    inline thrust::device_vector<uint2>::iterator EndDoneRayPartitions() { return doneRayPartitions.begin() + doneNodes; }
    inline size_t DoneNodes() const { return doneNodes; }

    inline DacrtNode GetUnfinished(const size_t i) const {
        return DacrtNode(rayPartitions[i], spherePartitions[i]);
    }
    inline void SetUnfinished(const size_t i, const DacrtNode& n) {
        if (unfinishedNodes <= i) {
            unfinishedNodes = i+1;
            rayPartitions.resize(unfinishedNodes);
            spherePartitions.resize(unfinishedNodes);
        }
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
    static void CalcOwners(thrust::device_vector<uint2>& partitions,
                           thrust::device_vector<unsigned int>& owners);

    std::string ToString() const;
    std::string ToString(RayContainer& rays, SphereContainer& spheres) const;
};

inline std::ostream& operator<<(std::ostream& s, const DacrtNodes& d){
    return s << d.ToString();
}


#endif // _GPU_DACRT_NODE_H_
