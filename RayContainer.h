// Ray container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAY_CONTAINER_H_
#define _GPU_DACRT_RAY_CONTAINER_H_

#include <HyperRays.h>

class RayContainer {
public:
    HyperRays innerRays;
private:
    HyperRays nextRays;
public:
    HyperRays leafRays;

public:
    RayContainer(const int width, const int height, const int sqrtSamples)
        : innerRays(width, height, sqrtSamples), 
          nextRays(innerRays.Size()), 
          leafRays(innerRays.Size()) {
        nextRays.Resize(0); leafRays.Resize(0);
    }

    void Clear();

    inline HyperRays::Iterator BeginInnerRays() { return innerRays.Begin(); }
    inline HyperRays::Iterator EndInnerRays() { return innerRays.End(); }
    inline unsigned int InnerSize() const { return innerRays.Size(); }
    inline HyperRay GetInner(const unsigned int i) const { return innerRays.Get(i); }

    inline HyperRays::Iterator BeginLeafRays() { return leafRays.Begin(); }
    inline HyperRays::Iterator EndLeafRays() { return leafRays.End(); }
    inline unsigned int LeafRays() const { return leafRays.Size(); }
    inline HyperRay GetLeaf(const unsigned int i) const { return leafRays.Get(i); }

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

    void RemoveTerminated(thrust::device_vector<unsigned int>& termianted);

    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const RayContainer& rays){
    return s << rays.ToString();
}

#endif // _GPU_DACRT_RAY_CONTAINER_H_
