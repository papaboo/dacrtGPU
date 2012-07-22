// Sphere geometry container.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <SphereContainer.h>

#include <Cone.h>
#include <HyperCube.h>
#include <SphereGeometry.h>

#include <algorithm>
#include <ostream>

#include <thrust/for_each.h>
#include <thrust/copy.h>

struct CreateCones {
    __host__ __device__
    Cone operator()(const thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> c) const {
        const HyperCube cube(thrust::get<0>(c), thrust::get<1>(c), thrust::get<2>(c),
                             thrust::get<3>(c), thrust::get<4>(c), thrust::get<5>(c));
        
        return Cone::FromCube(cube);
    }
};
static CreateCones createCones;


__constant__ Cone d_cone;
// __constant__ float d_invSinToAngle;
// __constant__ float d_cosToAngleSqr;

struct CompareConeSphere {

    CompareConeSphere(thrust::device_vector<Cone> cones, unsigned int index) {
        Cone* cone = thrust::raw_pointer_cast(cones.data()) + index;
        cudaMemcpyToSymbol(d_cone, (void*)cone, sizeof(Cone), 0, cudaMemcpyDeviceToDevice);
        // const float invSinToAngle = 1.0f / std::sin(spreadAngle);
        // const float cosToAngleSqr = std::cos(spreadAngle) * cos(spreadAngle);
        // cudaMemcpyToSymbol(d_invSinToAngle, (void*)&invSinToAngle, sizeof(float), 0, cudaMemcpyHostToDevice);
        // cudaMemcpyToSymbol(d_cosToAngleSqr, (void*)&cosToAngleSqr, sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    
    __device__
    bool operator()(const Sphere s) {
        return d_cone.DoesIntersect(s);//, d_invSinToAngle, d_cosToAngleSqr);
    }
};

SphereContainer::SphereContainer(HyperCubes& cubes, SpheresGeometry& spheres,
                                 uint* spherePartitionStart)
    : spheres(spheres),
      indices1(spheres.Size()*cubes.Size()),
      indices2(spheres.Size()*cubes.Size()),
      currentIndices(indices1), 
      nextIndices(indices2),
      doneIndices(spheres.Size()*cubes.Size()) {
    
    nextIndices.resize(0);
    doneIndices.resize(0);

    thrust::device_vector<Cone> cones(cubes.Size());
    thrust::transform(cubes.Begin(), cubes.End(), cones.begin(), createCones);
    //std::cout << cones << std::endl;

    spherePartitionStart[0] = 0;
    for (int c = 0; c < cubes.Size(); ++c) {
        CompareConeSphere compareConeSphere(cones, c);
        UintIterator beginIndices = BeginCurrentIndices() + spherePartitionStart[c];
        UintIterator itr = thrust::copy_if(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(spheres.Size()),
                                           spheres.BeginSpheres(), beginIndices, compareConeSphere);
        spherePartitionStart[c+1] = spherePartitionStart[c] + (itr - beginIndices);
        //std::cout << spherePartitionStart[c+1] << " = " << spherePartitionStart[c] << " + " << (itr - beginIndices) << std::endl;
    }
    const size_t currentSize = spherePartitionStart[cubes.Size()];
    currentIndices.resize(currentSize);
}

struct PartitionLeftRight {
    unsigned int* nextIndices;
    PartitionLeftRight(thrust::device_vector<unsigned int>& nextIs)
        : nextIndices(thrust::raw_pointer_cast(nextIs.data())) {}
    
    __device__    
    void operator()(thrust::tuple<PartitionSide, unsigned int, unsigned int, // partitionSide, leftIndex, rightIndex,
                            unsigned int> input) { // index
        const PartitionSide side = thrust::get<0>(input);
        const unsigned int sphereIndex = thrust::get<3>(input);
        if (side & LEFT) {
            const unsigned int leftIndex = thrust::get<1>(input);
            nextIndices[leftIndex] = sphereIndex;
        }
        if (side & RIGHT) {
            const unsigned int rightIndex = thrust::get<2>(input);
            nextIndices[rightIndex] = sphereIndex;
        }
    }
    
};

void SphereContainer::Partition(thrust::device_vector<PartitionSide>& partitionSides, 
                                thrust::device_vector<unsigned int>& leftIndices,
                                thrust::device_vector<unsigned int>& rightIndices) {

    // std::cout << "--SphereContainer::Partition--:" << std::endl;

    const unsigned int nextSize = rightIndices[rightIndices.size()-1];
    nextIndices.resize(nextSize);

    thrust::zip_iterator<thrust::tuple<PartitionSideIterator, UintIterator, UintIterator,
        UintIterator> > input
        = thrust::make_zip_iterator(thrust::make_tuple(partitionSides.begin(), leftIndices.begin(), rightIndices.begin(), 
                                                       currentIndices.begin()));

    PartitionLeftRight partitionLeftRight(nextIndices);
    thrust::for_each(input, input + CurrentSize(), partitionLeftRight);

    // TODO change to device_vector.swap, but that crashes right now
    std::swap(currentIndices, nextIndices);
    // currentIndices.swap(nextIndices);
}

__constant__ unsigned int d_leafIndexOffset;

struct PartitionLeafsKernel {
    // Rays
    unsigned int *nextIndices;
    unsigned int *leafIndices;
    
    // Node values
    bool* leafMarkers;
    uint2* nodePartitions;
    unsigned int* nodeLeafIndices;
    
    PartitionLeafsKernel(thrust::device_vector<unsigned int>& nIndices, 
                         thrust::device_vector<unsigned int>& lIndices, 
                         thrust::device_vector<bool>& lMarkers,
                         thrust::device_vector<uint2>& nPartitions,
                         thrust::device_vector<unsigned int>& nlIndices,
                         const unsigned int leafIndexOffset)
        : nextIndices(thrust::raw_pointer_cast(nIndices.data())), 
          leafIndices(thrust::raw_pointer_cast(lIndices.data()) + leafIndexOffset),
          leafMarkers(thrust::raw_pointer_cast(lMarkers.data())),
          nodePartitions(thrust::raw_pointer_cast(nPartitions.data())),
          nodeLeafIndices(thrust::raw_pointer_cast(nlIndices.data())) {}
    
    __host__ __device__
    unsigned int operator()(const thrust::tuple<unsigned int, unsigned int > input, // owner, sphereIndex
                            const unsigned int threadId) const { 
        const unsigned int owner = thrust::get<0>(input);
        const unsigned int sphereIndex = thrust::get<1>(input);
        const unsigned int nodeLeafIndex = nodeLeafIndices[owner];
        const bool isLeaf = leafMarkers[owner];
        if (isLeaf) {
            uint2 partitioning = nodePartitions[owner];
            unsigned int partitionIndex = threadId - partitioning.x;
            unsigned int leafIndex = nodeLeafIndex + partitionIndex;
            leafIndices[leafIndex] = sphereIndex;
            return leafIndex;
        } else {
            const unsigned int index = threadId - nodeLeafIndex;
            nextIndices[index] = sphereIndex;
            return index;
        }
    }
};

void SphereContainer::PartitionLeafs(thrust::device_vector<bool>& isLeaf, 
                                     thrust::device_vector<unsigned int>& leafNodeIndices, 
                                     thrust::device_vector<uint2>& spherePartitions,
                                     thrust::device_vector<unsigned int>& owners) {

    /*
    std::cout << "--SphereContainer::PartitionLeafs--:" << std::endl;
    std::cout << "isLeaf:\n" << isLeaf << std::endl;
    std::cout << "leafNodeIndices:\n" << leafNodeIndices << std::endl;
    std::cout << "spherePartitions:\n" << spherePartitions << std::endl;
    std::cout << "owners:\n" << owners << std::endl;
    std::cout << ToString() << std::endl;
    */

    const unsigned int newLeafs = leafNodeIndices[leafNodeIndices.size()-1];
    const unsigned int prevLeafIndiceAmount = doneIndices.size();
    nextIndices.resize(currentIndices.size() - newLeafs); // shrink next ray buffer
    doneIndices.resize(doneIndices.size() + newLeafs); // expand leaf
    
    // TODO replace owners with work queue

    thrust::zip_iterator<thrust::tuple<UintIterator, UintIterator> > input =
        thrust::make_zip_iterator(thrust::make_tuple(owners.begin(), BeginCurrentIndices()));

    PartitionLeafsKernel partitionLeafs(nextIndices, doneIndices, isLeaf, spherePartitions, leafNodeIndices, prevLeafIndiceAmount);
    thrust::transform(input, input + CurrentSize(), thrust::counting_iterator<unsigned int>(0),
                      owners.begin(), partitionLeafs);
    // std::cout << "index moved to:\n" << owners << std::endl;

    // TODO change to device_vector.swap, but that crashes right now
    std::swap(currentIndices, nextIndices);
    //currentIndices.swap(nextIndices);

    // std::cout << ToString() << std::endl;
}


std::string SphereContainer::ToString() const {
    std::ostringstream out;
    if (CurrentSize() > 0) {
        out << "Current spheres:";
        for (size_t i = 0; i < CurrentSize(); ++i) {
            const unsigned int id = currentIndices[i];
            out << "\n" << i << ": [id: " << id << ", " << spheres.Get(id) << "]";
        }
        if (DoneSize() > 0) out << "\n";
    }
    if (DoneSize() > 0) {
        out << "Leaf spheres:";
        for (size_t i = 0; i < DoneSize(); ++i) {
            const unsigned int id = doneIndices[i];
            out << "\n" << i << ": [id: " << id << ", " << spheres.Get(id) << "]";
        }
    }
    return out.str();
}
