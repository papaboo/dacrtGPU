// HyperRay MortonCode encoding
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_MORTON_CODE_H_
#define _GPU_DACRT_MORTON_CODE_H_

#include <Enums.h>
#include <Utils/Morton.h>
#include <Utils/ToString.h>
#include <Utils/Utils.h>

#include <string>
#include <sstream>

struct MortonCode {

#define xOffset 3
#define yOffset 4
#define zOffset 2
#define uOffset 1
#define vOffset 0
    // Interleave pattern | Y | X | Z | U | V |. Y is last so I can overwrite
    // that last bit with the axis info at a future point.

    unsigned int code;

    __host__ __device__ MortonCode() {}
    __host__ __device__ MortonCode(const unsigned int c) : code(c) {}
    
    __host__ __device__ static inline unsigned int CodeFromIndex(const unsigned int index, const int offset) { return Utils::Morton::PartBy4(index) << offset; }
    __host__ __device__ static inline unsigned int IndexFromCode(const MortonCode code, const int offset) { return Utils::Morton::CompactBy4(code.code >> offset); }
    __host__ __device__ static inline MortonCode EncodeAxis(const unsigned int a) { return MortonCode(a << 29); }
    __host__ __device__ inline SignedAxis GetAxis() const { return SignedAxis(code >> 29); }

    __host__ __device__ inline MortonCode& operator=(const unsigned int rhs) { code = rhs; return *this; }
    __host__ __device__ inline void operator+=(const MortonCode rhs) { code += rhs; }
    __host__ __device__ inline MortonCode operator&(const unsigned int rhs) { return MortonCode(code & rhs); }
    __host__ __device__ inline operator unsigned int() const { return code; }
    
    __host__ __device__ inline MortonCode WithoutAxis() const { return MortonCode(code & 0x1FFFFFFF); }

    inline std::string ToString() const {
        std::ostringstream out;
        SignedAxis axis = GetAxis();
        MortonCode tmp = MortonCode(code & 0x1FFFFFFF);
        unsigned int x = IndexFromCode(tmp, xOffset);
        unsigned int y = IndexFromCode(tmp, yOffset);
        unsigned int z = IndexFromCode(tmp, zOffset);
        unsigned int u = IndexFromCode(tmp, uOffset);
        unsigned int v = IndexFromCode(tmp, vOffset);
        out << "[axis: " << axis << ", x: " << x << ", y: " << y << ", z: " << z << ", u: " << u << ", v: " << v << " - " << Bitmap(code) << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const MortonCode& c){
    return s << c.ToString();
}


struct MortonBound {
    MortonCode min, max;

    __host__ __device__    
    inline static MortonBound Create(const unsigned int val) {
        MortonBound b; b.min = b.max = val; 
        return b;
    }

    __host__ __device__    
    inline static MortonBound Create(const unsigned int min, const unsigned int max) {
        MortonBound b; b.min = min; b.max = max; 
        return b;
    }

    __host__ __device__    
    inline static MortonBound LowestCommonBound(const MortonCode min, const MortonCode max) {
        unsigned int diff = min.code ^ max.code;
        
        // If both bounds are the same, then return a bound around min and max
        if (diff == 0) return MortonBound::Create(min);

        diff = ReverseBits(diff);

        int n = FirstBitSet(diff) - 1;
        
        MortonBound b; 
        const unsigned int minMask = 0XFFFFFFFF << (31-n);
        b.min = min & minMask;

        const unsigned int maxMask = 0XFFFFFFFF >> (n+1);
        b.max = max | maxMask;

        return b;
    }
    
    /**
     * Checks if a bound is a leaf or if further splits are possble.
     */
    __host__ __device__
    inline bool Splitable() const {
        return min.code == max.code;
    }

    /**
     * Splits the MortonBound along it's most significant bit, i.e. the next
     * possible split, and returns the resulting left and right bounds
     *
     * This method assumes that it is infact possible to split the
     * bound. Attempting to split an unsplitable bound (i.e. min == max) results
     * in undefined behaviour.
     */
    __host__ __device__
    inline void Split(MortonBound &left, MortonBound &right) const {

        unsigned int diff = min.code ^ max.code;
        int n = LastBitSet(diff) - 1;

        const unsigned int CODE_MASK = 0x84210842;
        unsigned int shiftedMask = CODE_MASK >> (n + 5);

        // The new right minimum code is found by zeroing out every bit
        // associated with the chosen axis below the n'th bit and setting the
        // bit at n to 1.
        // In terms of seeing the morton code as a walk down a binary tree this
        // corrosponds to choosing the right child at the n'th step and then
        // going left every time you see the axis next.
        MortonCode rightMinCode = (min & ~shiftedMask) | (0x80000000 >> n);
        right = MortonBound::Create(rightMinCode, max);
        
        // Likewise the left maximum is found by setting the n'th bit to 0 and
        // every following 5'th bit to 1, corrosponding to a walking left at the
        // n'th node and subsequently right whenever that axis is split.
        MortonCode leftMaxCode = (max & ~(CODE_MASK >> n)) | shiftedMask;
        left = MortonBound::Create(min, leftMaxCode);
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << "[min: " << min << ", max: " << max << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const MortonBound& c){
    return s << c.ToString();
}

#endif _GPU_DACRT_MORTON_CODE_H_
