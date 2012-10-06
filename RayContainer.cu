// Ray container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <RayContainer.h>
#include <Utils/ToString.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <sstream>

void RayContainer::Clear() {
    innerRays.Resize(0);
    nextRays.Resize(0);
    
    leafRays.Resize(0);
}

void RayContainer::Convert(const Rays::Representation r) {
    innerRays.Convert(r);
    nextRays.Convert(r);
    leafRays.Convert(r);
}

template<int A>
struct PartitionRaysByAxis {
    __host__ __device__
    bool operator()(thrust::tuple<float4, float4> t) {
        return thrust::get<1>(t).x < (float)A;
    }
    
    static inline unsigned int Do(Rays& rays) {
        PartitionRaysByAxis<A> pred;
        return thrust::partition(rays.Begin(), rays.End(), pred) - rays.Begin();
    }
};

void RayContainer::PartitionByAxis(unsigned int* rayPartitionStart) {
    rayPartitionStart[0] = 0;
    rayPartitionStart[3] = PartitionRaysByAxis<3>::Do(innerRays);
    rayPartitionStart[1] = PartitionRaysByAxis<1>::Do(innerRays);
    rayPartitionStart[2] = PartitionRaysByAxis<2>::Do(innerRays);
    rayPartitionStart[4] = PartitionRaysByAxis<4>::Do(innerRays);
    rayPartitionStart[5] = PartitionRaysByAxis<5>::Do(innerRays);
    rayPartitionStart[6] = innerRays.Size();
}

__constant__ unsigned int d_raysMovedLeft;

struct PartitionLeft {
    float4 *nextOrigins, *nextAxisUVs;

    PartitionLeft(Rays& nextRays,
                  thrust::device_vector<unsigned int>& leftIndices) 
        : nextOrigins(thrust::raw_pointer_cast(nextRays.origins.data())), 
          nextAxisUVs(thrust::raw_pointer_cast(nextRays.axisUVs.data())) {
        unsigned int* data = thrust::raw_pointer_cast(leftIndices.data()) + leftIndices.size()-1;
        cudaMemcpyToSymbol(d_raysMovedLeft, (void*)data, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
    }

    __device__    
    unsigned int operator()(thrust::tuple<PartitionSide, unsigned int, // partitionSide, leftIndex,
                            thrust::tuple<float4, float4> > val, unsigned int threadId) { // Ray

        const PartitionSide side = thrust::get<0>(val);
        const unsigned int leftIndex = thrust::get<1>(val);
        const unsigned int index = side & LEFT != 0 ? leftIndex : threadId - leftIndex + d_raysMovedLeft;
        const thrust::tuple<float4, float4> ray = thrust::get<2>(val);
        nextOrigins[index] = thrust::get<0>(ray);
        nextAxisUVs[index] = thrust::get<1>(ray);
        return leftIndex;
    }
};

void RayContainer::Partition(thrust::device_vector<PartitionSide>& partitionSides, thrust::device_vector<unsigned int>& leftIndices) {

    const size_t nextSize = leftIndices.size() - 1; // -1 since last element is the total number of rays moved left
    nextRays.Resize(nextSize);

    thrust::zip_iterator<thrust::tuple<PartitionSideIterator, UintIterator, Rays::Iterator> > input
        = thrust::make_zip_iterator(thrust::make_tuple(partitionSides.begin(), leftIndices.begin(),
                                                       innerRays.Begin()));

    PartitionLeft partitionLeft(nextRays, leftIndices);
    thrust::transform(input, input + nextSize, thrust::counting_iterator<unsigned int>(0), 
                      leftIndices.begin(), partitionLeft);
    CHECK_FOR_CUDA_ERROR(); 

    innerRays.Swap(nextRays);
}


struct PartitionLeafsKernel {
    // Rays
    float4 *nextOrigins, *nextAxisUVs;
    float4 *leafOrigins, *leafAxisUVs;
    
    // Node values
    bool* leafMarkers;
    uint2* nodePartitions;
    unsigned int* nodeLeafIndices;
    
    PartitionLeafsKernel(Rays& nextRays, Rays& leafRays,
                         thrust::device_vector<bool>& lMarkers,
                         thrust::device_vector<uint2>& nPartitions,
                         thrust::device_vector<unsigned int>& nlIndices,
                         const unsigned int leafIndexOffset)
        : nextOrigins(thrust::raw_pointer_cast(nextRays.origins.data())), 
          nextAxisUVs(thrust::raw_pointer_cast(nextRays.axisUVs.data())),
          leafOrigins(thrust::raw_pointer_cast(leafRays.origins.data()) + leafIndexOffset), 
          leafAxisUVs(thrust::raw_pointer_cast(leafRays.axisUVs.data()) + leafIndexOffset),
          leafMarkers(thrust::raw_pointer_cast(lMarkers.data())),
          nodePartitions(thrust::raw_pointer_cast(nPartitions.data())),
          nodeLeafIndices(thrust::raw_pointer_cast(nlIndices.data())) {}
    
    __host__ __device__
    unsigned int operator()(const thrust::tuple<unsigned int, thrust::tuple<float4, float4> > input, // owner, ray
                            const unsigned int threadId) const { 
        const unsigned int owner = thrust::get<0>(input);
        const thrust::tuple<float4, float4> ray = thrust::get<1>(input);
        const unsigned int nodeLeafIndex = nodeLeafIndices[owner];
        const bool isLeaf = leafMarkers[owner];
        if (isLeaf) {
            uint2 partitioning = nodePartitions[owner];
            unsigned int partitionIndex = threadId - partitioning.x;
            unsigned int leafIndex = nodeLeafIndex + partitionIndex;
            leafOrigins[leafIndex] = thrust::get<0>(ray);
            leafAxisUVs[leafIndex] = thrust::get<1>(ray);
            return leafIndex;
        } else {
            const unsigned int index = threadId - nodeLeafIndex;
            nextOrigins[index] = thrust::get<0>(ray);
            nextAxisUVs[index] = thrust::get<1>(ray);
            return index;
        }
    }
};

void RayContainer::PartitionLeafs(thrust::device_vector<bool>& isLeaf, 
                                  thrust::device_vector<unsigned int>& leafNodeIndices, 
                                  thrust::device_vector<uint2>& rayPartitions,
                                  thrust::device_vector<unsigned int>& owners) {

    /*
    std::cout << "--PartitionLeafs--:" << std::endl;
    std::cout << "isLeaf:\n" << isLeaf << std::endl;
    std::cout << "leafNodeIndices:\n" << leafNodeIndices << std::endl;
    std::cout << "rayPartitions:\n" << rayPartitions << std::endl;
    std::cout << "owners:\n" << owners << std::endl;
    std::cout << ToString() << std::endl;
    */

    const unsigned int newLeafs = leafNodeIndices[leafNodeIndices.size()-1];
    const unsigned int prevLeafIndiceAmount = leafRays.Size();
    nextRays.Resize(innerRays.Size() - newLeafs); // shrink next ray buffer
    leafRays.Resize(leafRays.Size() + newLeafs); // expand leaf
    
    // TODO replace owners with work queue

    thrust::zip_iterator<thrust::tuple<UintIterator, Rays::Iterator> > input =
        thrust::make_zip_iterator(thrust::make_tuple(owners.begin(), BeginInnerRays()));

    PartitionLeafsKernel partitionLeafs(nextRays, leafRays, isLeaf, rayPartitions, leafNodeIndices, prevLeafIndiceAmount);
    thrust::transform(input, input + InnerSize(), thrust::counting_iterator<unsigned int>(0),
                      owners.begin(), partitionLeafs);
    // std::cout << "index moved to:\n" << owners << std::endl;

    innerRays.Swap(nextRays);

    // std::cout << ToString() << std::endl;
}

void RayContainer::SortToLeaves(thrust::device_vector<unsigned int>::iterator keysBegin,
                                thrust::device_vector<unsigned int>::iterator keysEnd) {

    thrust::sort_by_key(keysBegin, keysEnd, BeginInnerRays());

    // TODO This will be a lot faster once leafs are just the first 'n' rays in innerRays.
    leafRays.Swap(innerRays);
    innerRays.Resize(0);
}

void RayContainer::RemoveTerminated(thrust::device_vector<unsigned int>& terminated) {
    innerRays.Resize(LeafRays());
    
    Rays::Iterator end = 
        thrust::remove_copy_if(leafRays.Begin(), leafRays.End(), terminated.begin(), 
                               innerRays.Begin(), thrust::logical_not<unsigned int>());
    
    size_t innerSize = end - innerRays.Begin();
    innerRays.Resize(innerSize);

    leafRays.Resize(0);
}

std::string RayContainer::ToString() const {
    std::ostringstream out;
    if (InnerSize() > 0) {
        out << "Inner rays (" << InnerSize() << "):";
        for (size_t i = 0; i < InnerSize(); ++i) {
            out << "\n" << i << ": ";
            if (innerRays.GetRepresentation() == Rays::RayRepresentation)
                out << innerRays.GetAsHyperRay(i) ;
            else 
                out << innerRays.GetAsRay(i);
        }
        if (LeafRays() > 0) out << "\n";
    }
    if (LeafRays() > 0) {
        out << "Leaf rays (" << LeafRays() << "):";
        for (size_t i = 0; i < LeafRays(); ++i) {
            out << "\n" << i << ": ";
            if (leafRays.GetRepresentation() == Rays::RayRepresentation) 
                out << leafRays.GetAsHyperRay(i);
            else 
                out << leafRays.GetAsRay(i);
        }
    }
    return out.str();
}
