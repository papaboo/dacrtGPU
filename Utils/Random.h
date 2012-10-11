// Utils for creating random numbers on the GPU
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_UTILS_RANDOM_H_
#define _GPU_DACRT_UTILS_RANDOM_H_

/**
 * A linear congruential generator can be used to produce random numbers. A
 * description of how it works and good default parameters can be found at
 * http://en.wikipedia.org/wiki/Linear_congruential_generator.
 */
template<unsigned int A, unsigned int C, unsigned int M>
class LinearCongruentialGenerator {
    unsigned int x;
public:
    __host__ __device__
    LinearCongruentialGenerator(const unsigned int seed = 7204)
        : x(seed) {}

    __host__ __device__
    inline void Reseed(const unsigned int seed) {
        x = seed;
    }

    __host__ __device__
    inline unsigned int Next() {
        x = (A * x + C) % M;
        return x;
    }
};

class Random {

    LinearCongruentialGenerator<214013, 2531011, 0xFFFFFFFF> generator;

    __host__ __device__
    Random(const unsigned int seed)
        : generator(LinearCongruentialGenerator<214013, 2531011, 0xFFFFFFFF>(seed)) {}

    /**
     * Hashes an unsigned int. Found in 
     * http://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu.
     */
    __host__ __device__
    static inline unsigned int Hash(unsigned int a) {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }
    
public:
    
    /**
     * Creates a random generator which uses a linear congruential generator. If
     * invoked on the GPU the generator is seeded with a hash of the threadId
     * plus the seed.
     */
    __host__ __device__
    static inline Random Create1D(const unsigned int seed) {
#ifdef __CUDA_ARCH__
        const unsigned int threadId = threadIdx.x + blockDim.x * blockIdx.x;
        return Random(Hash(threadId) + seed);
#else
        return Random(seed);
#endif
    }

    /**
     * Returns the next random unsigned int.
     */
    __host__ __device__
    inline unsigned int NextUint() { return generator.Next(); }

    /**
     * Returns the next random float.
     */
    __host__ __device__
    inline unsigned int NextFloat() { 
        const unsigned int res = generator.Next();
        return *(float*) &res; }

    /**
     * Returns the next random float in the [0; 1] range.
     */
    __host__ __device__
    inline float NextFloat01() {
        // unsigned int bitmap = generator.Next() >> 9; // Uses the 23 higher order bits as decimals, since they have a longer period.
        // bitmap |= 0x3F800000; // Or the exponent onto the bitmap
        // const float r1 = *(float*) &bitmap; // Interprets the bitmap as a float.
        // return r1 - 1.0f; // Subtract 1 to only return the decimals.

        const float invUintMax = 1.0f / 4294967294.0f;
        float rand = (float)generator.Next();
        return rand * invUintMax;
    }
    
};

#endif // _GPU_DACRT_UTILS_RANDOM_H_
