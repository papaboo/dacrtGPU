// Test that random numbers between 0 and 1 can be generated on the GPU by using
// a bit hack.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/Random.h>
#include <Utils/ToString.h>
#include <Utils/Utils.h>

#include <iostream>
#include <time.h>

#include <thrust/version.h>

using std::cout;
using std::endl;

static inline unsigned int Hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

inline float Range01(const unsigned int r) {
    // unsigned int bitmap = r >> 9; // Uses the 23 higher order bits as decimals, since they have a longer period.
    // bitmap |= 0x3F800000; // Or the exponent onto the bitmap
    // float r1 = *(float*) &bitmap; // Convert the bitmap to a float.
    // return r1 - 1.0f; // Subtract 1 to only return the decimals.

    static const float invUintMax = 1.0f / 4294967294.0f;
    return (float)r * invUintMax;
}

__global__
void CreateRandom(float* rands, 
                  const unsigned int seed, 
                  const unsigned int nRands) {
    const unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nRands) return;
    
    Random rand = Random::Create1D(seed);
    rands[id] = rand.NextFloat01();
}

int main(int argc, char *argv[]){
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    Meta::CUDA::Initialize();
    
    unsigned int seed = 548205;
    float average = 0.0f;
    for (int i = 0; i < 20; ++i) {
        LinearCongruentialGenerator<214013, 2531011, 0xFFFFFFFF> generator(Hash(i) + seed);
        float rand1 = Range01(generator.Next());
        float rand2 = Range01(generator.Next());
        cout << i << ": yields " << rand1 << " and " << rand2 << endl;
        average += rand1 + rand2;
    }
    average /= 40;
    cout << "average " << average << endl;
    
    thrust::device_vector<float> randoms(20);
    CreateRandom<<<1, 64>>>(RawPointer(randoms), 548205, randoms.size());
    CHECK_FOR_CUDA_ERROR();
    
    cout << "Randoms:\n" << randoms << endl;
}
