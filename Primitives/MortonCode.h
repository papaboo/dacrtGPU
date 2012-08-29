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
    __host__ __device__ inline SignedAxis GetAxis() const { return SignedAxis((code & 0xE0000000) >> 29); }

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
