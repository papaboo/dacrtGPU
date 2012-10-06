// Ray container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAY_CONTAINER_H_
#define _GPU_DACRT_RAY_CONTAINER_H_

#include <Rays.h>

class RayContainer {
public:
    Rays innerRays;
private:
    Rays nextRays;
public:
    Rays leafRays;

public:
    RayContainer(const int width, const int height, const int sqrtSamples)
        : innerRays(width, height, sqrtSamples), 
          nextRays(innerRays.Size()), 
          leafRays(innerRays.Size()) {
        nextRays.Resize(0); leafRays.Resize(0);
    }

    void Clear();

    inline Rays::Iterator BeginInnerRays() { return innerRays.Begin(); }
    inline Rays::Iterator EndInnerRays() { return innerRays.End(); }
    inline unsigned int InnerSize() const { return innerRays.Size(); }

    inline Rays::Iterator BeginLeafRays() { return leafRays.Begin(); }
    inline Rays::Iterator EndLeafRays() { return leafRays.End(); }
    inline unsigned int LeafRays() const { return leafRays.Size(); }
    inline Ray GetLeafRay(const size_t r) const { return leafRays.GetAsRay(r); }

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
    
    void RemoveTerminated(thrust::device_vector<unsigned int>& termianted);

    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const RayContainer& rays){
    return s << rays.ToString();
}

#endif // _GPU_DACRT_RAY_CONTAINER_H_
