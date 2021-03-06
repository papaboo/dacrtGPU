// Ray container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAY_CONTAINER_H_
#define _GPU_DACRT_RAY_CONTAINER_H_

#include <Rendering/Rays.h>

namespace Rendering {

class RayContainer {
private:
    Rays innerRays;
    Rays nextRays;
    unsigned int nLeafRays;

public:
    RayContainer(const int width, const int height, const int sqrtSamples)
        : innerRays(width, height, sqrtSamples), 
          nextRays(innerRays.Size()), 
          nLeafRays(0) {
        nextRays.Resize(0);
    }

    void Clear();

    inline Rays::Iterator BeginInnerRays() { return innerRays.Begin() + nLeafRays; }
    inline Rays::Iterator EndInnerRays() { return innerRays.End(); }
    inline unsigned int InnerSize() const { return innerRays.Size() - nLeafRays; }

    inline Rays::Iterator BeginLeafRays() { return innerRays.Begin(); }
    inline Rays::Iterator EndLeafRays() { return innerRays.Begin() + nLeafRays; }
    inline unsigned int LeafRays() const { return nLeafRays; }
    inline Ray GetLeafRay(const size_t r) const { return innerRays.GetAsRay(r); }

    /**
     * Converts the rays to a hyperray representation.
     */
    void Convert(const Rays::Representation r);

    /**
     * Partitions the rays according to their major axis. 
     * 
     * The ray partition start array given as argument must be of size 
     * '||axis||+ 1', so 7.
     */
    void PartitionByAxis(unsigned int* rayPartitionStart);

    /**
     * Partitions the rays uniquely left and right.
     */
    void Partition(thrust::device_vector<PartitionSide>& partitionSides, thrust::device_vector<unsigned int>& leftIndices);

    /**
     * Partitions the currently active rays into leaf and non leaf nodes.
     */
    void PartitionLeafs(thrust::device_vector<bool>& isLeaf, 
                        thrust::device_vector<unsigned int>& leafNodeIndices, 
                        thrust::device_vector<uint2>& rayPartitions,
                        thrust::device_vector<unsigned int>& owners);

    void SortToLeaves(thrust::device_vector<unsigned int>::iterator keysBegin,
                      thrust::device_vector<unsigned int>::iterator keysEnd);
    
    /**
     * Converts the remaining active rays to leaf rays.
     */
    void MakeLeaves();
    
    void RemoveTerminated(thrust::device_vector<unsigned int>& terminated);
    void RemoveTerminated(thrust::device_vector<unsigned int>::iterator beginTerminated);

    /**
     * Moves leaf rays to active rays.
     */
    void ReinitLeafRays();

    std::string ToString() const;
};

} // NS Rendering

inline std::ostream& operator<<(std::ostream& s, const Rendering::RayContainer& rays){
    return s << rays.ToString();
}

#endif // _GPU_DACRT_RAY_CONTAINER_H_
