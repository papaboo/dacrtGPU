// Work queue idea test.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <ToString.h>
#include <Utils.h>
#include <ForEachWithOwners.h>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/version.h>

struct WriteOwner {
    unsigned int* owners;
    
    WriteOwner(thrust::device_vector<unsigned int>& os) 
        : owners(thrust::raw_pointer_cast(os.data())) {}

    __device__
    void operator()(const unsigned int index, const unsigned int owner) const {
        owners[index] = owner;
    }
};

int main(int argc, char *argv[]){
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    Meta::CUDA::Initialize();

    thrust::device_vector<uint2> partitions(2);
    partitions[0] = make_uint2(0, 31);
    partitions[1] = make_uint2(31, 64);
    thrust::device_vector<unsigned int> owners(64);
 
   /*
    thrust::device_vector<uint2> partitions(12);
    partitions[0] = make_uint2(0, 100);
    partitions[1] = make_uint2(100, 250);
    partitions[2] = make_uint2(250, 300);
    partitions[3] = make_uint2(300, 305);
    partitions[4] = make_uint2(305, 306);
    partitions[5] = make_uint2(306, 600);
    partitions[6] = make_uint2(600, 666);
    partitions[7] = make_uint2(666, 700);
    partitions[8] = make_uint2(700, 703);
    partitions[9] = make_uint2(703, 703);
    partitions[10] = make_uint2(703, 65000); // extra element containing the elem count and a dummy max value.
    partitions[11] = make_uint2(65000, 6500000); // if this showes up something is seriously wrong with the kernel.
    thrust::device_vector<unsigned int> owners(704);
    */    

    WriteOwner writeOwner(owners);
    ForEachWithOwners(partitions, 0, partitions.size(),
                      owners.size(), writeOwner);

    std::cout << owners << std::endl;
    
    return 0;
}
