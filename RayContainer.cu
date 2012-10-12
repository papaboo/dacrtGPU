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
    nLeafRays = 0;
}

void RayContainer::Convert(const Rays::Representation r) {
    innerRays.Convert(r);
    nextRays.Convert(r);
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

    PartitionLeft(Rays::Iterator nextRays,
                  thrust::device_vector<unsigned int>& leftIndices) 
        : nextOrigins(RawPointer(Rays::GetOrigins(nextRays))), 
          nextAxisUVs(RawPointer(Rays::GetDirections(nextRays))) {
        unsigned int* data = RawPointer(leftIndices) + leftIndices.size()-1;
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

    const size_t nNextSize = leftIndices.size() - 1; // -1 since last element is the total number of rays moved left
    nextRays.Resize(nNextSize + nLeafRays);

    thrust::zip_iterator<thrust::tuple<PartitionSideIterator, UintIterator, Rays::Iterator> > input
        = thrust::make_zip_iterator(thrust::make_tuple(partitionSides.begin(), leftIndices.begin(),
                                                       BeginInnerRays()));

    PartitionLeft partitionLeft(nextRays.Begin() + nLeafRays, leftIndices);
    thrust::transform(input, input + nNextSize, thrust::counting_iterator<unsigned int>(0), 
                      leftIndices.begin(), partitionLeft);

    innerRays.Swap(nextRays);
}


struct PartitionLeafsK {
    // Rays
    float4 *nextOrigins, *nextAxisUVs;
    float4 *leafOrigins, *leafAxisUVs;
    
    // Node values
    bool* leafMarkers;
    uint2* nodePartitions;
    unsigned int* nodeLeafIndices;
    
    PartitionLeafsK(Rays::Iterator nextRays, Rays::Iterator leafRays,
                    thrust::device_vector<bool>& lMarkers,
                    thrust::device_vector<uint2>& nPartitions,
                    thrust::device_vector<unsigned int>& nlIndices)
        : nextOrigins(RawPointer(Rays::GetOrigins(nextRays))), 
          nextAxisUVs(RawPointer(Rays::GetDirections(nextRays))), 
          leafOrigins(RawPointer(Rays::GetOrigins(leafRays))), 
          leafAxisUVs(RawPointer(Rays::GetDirections(leafRays))), 
          leafMarkers(RawPointer(lMarkers)),
          nodePartitions(RawPointer(nPartitions)),
          nodeLeafIndices(RawPointer(nlIndices)) {}
    
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

    const unsigned int nNewLeafs = leafNodeIndices[leafNodeIndices.size()-1];
    nextRays.Resize(innerRays.Size()); // Just to be sure.
    
    thrust::zip_iterator<thrust::tuple<UintIterator, Rays::Iterator> > input =
        thrust::make_zip_iterator(thrust::make_tuple(owners.begin(), BeginInnerRays()));

    PartitionLeafsK partitionLeafs(nextRays.Begin() + nLeafRays + nNewLeafs, 
                                   nextRays.Begin() + nLeafRays, 
                                   isLeaf, rayPartitions, leafNodeIndices);
    thrust::transform(input, input + InnerSize(), thrust::counting_iterator<unsigned int>(0),
                      owners.begin(), partitionLeafs);
    // std::cout << "index moved to:\n" << owners << std::endl;

    // Copy leaf rays from nextRays back into innerRays
    cudaMemcpy(RawPointer(Rays::GetOrigins(BeginInnerRays())),
               RawPointer(Rays::GetOrigins(nextRays.Begin() + nLeafRays)),
               sizeof(float4) * nNewLeafs, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();
    cudaMemcpy(RawPointer(Rays::GetDirections(BeginInnerRays())),
               RawPointer(Rays::GetDirections(nextRays.Begin() + nLeafRays)),
               sizeof(float4) * nNewLeafs, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();

    innerRays.Swap(nextRays);

    // std::cout << ToString() << std::endl;
    
    nLeafRays += nNewLeafs;
}

void RayContainer::SortToLeaves(thrust::device_vector<unsigned int>::iterator keysBegin,
                                thrust::device_vector<unsigned int>::iterator keysEnd) {

    thrust::sort_by_key(keysBegin, keysEnd, BeginInnerRays());

    nLeafRays = keysEnd - keysBegin;
}

void RayContainer::RemoveTerminated(thrust::device_vector<unsigned int>& terminated) {
    Rays::Iterator end = 
        thrust::remove_copy_if(BeginLeafRays(), EndLeafRays(), terminated.begin(), 
                               nextRays.Begin(), thrust::logical_not<unsigned int>());
    
    size_t innerSize = end - nextRays.Begin();
    innerRays.Swap(nextRays);
    innerRays.Resize(innerSize);
    nextRays.Resize(innerSize);

    nLeafRays = 0;
}

std::string RayContainer::ToString() const {
    std::ostringstream out;
    if (InnerSize() > 0) {
        out << "Inner rays (" << InnerSize() << "):";
        for (size_t i = 0; i < InnerSize(); ++i) {
            out << "\n" << i << ": ";
            if (innerRays.GetRepresentation() == Rays::RayRepresentation)
                out << innerRays.GetAsHyperRay(i+LeafRays()) ;
            else 
                out << innerRays.GetAsRay(i+LeafRays());
        }
        if (LeafRays() > 0) out << "\n";
    }
    if (LeafRays() > 0) {
        out << "Leaf rays (" << LeafRays() << "):";
        for (size_t i = 0; i < LeafRays(); ++i) {
            out << "\n" << i << ": ";
            if (innerRays.GetRepresentation() == Rays::RayRepresentation) 
                out << innerRays.GetAsHyperRay(i);
            else 
                out << innerRays.GetAsRay(i);
        }
    }
    return out.str();
}
