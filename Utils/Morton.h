// Morton Curve utility
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_MORTON_CURVE_H_
#define _GPU_DACRT_MORTON_CURVE_H_

#include <Utils/Utils.h>

namespace Utils {

    class Morton {
    public:
        
        /**
         * Insert a 0 bit after each of the 16 low bits of x
         *
         * source: http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
         */
        __host__ __device__
        inline static unsigned int PartBy1(unsigned int x) {
            x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
            x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
            x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
            x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
            x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
            return x;
        }

        /**
         * "Insert" two 0 bits after each of the 10 low bits of x
         *
         * source: http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
         */
        __host__ __device__
        inline static unsigned int PartBy2(unsigned int x) {
            x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
            x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
            x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
            x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
            x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
            return x;
        }

        __host__ __device__
        inline static unsigned int PartBy4(unsigned int x) {
            x &= 0x0000007f;                  // x = ---- ---- ---- ---- ---- ---- -654 3210
            x = (x ^ (x << 16)) & 0x0070000F; // x = ---- ---- -654 ---- ---- ---- ---- 3210
            x = (x ^ (x <<  8)) & 0x40300C03; // x = -6-- ---- --54 ---- ---- 32-- ---- --10
            x = (x ^ (x <<  4)) & 0x42108421; // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
            return x;
        }
        
        /**
         * Inverse of Part1By1 - "Compresses" all even-indexed bits
         */
        __host__ __device__
        inline static unsigned int CompactBy1(unsigned int x) {
            x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
            x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
            x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
            x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
            x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
            return x;
        }

        /**
         * Inverse of Part1By2 - "Compresses" all bits at positions divisible by 3
         */
        __host__ __device__
        inline static unsigned int CompactBy2(unsigned int x) {
            x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
            x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
            x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
            x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
            x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
            return x;
        }

        /**
         * Inverse of Part1By4 - "Compresses" all bits at positions divisible by 5
         */
        __host__ __device__
        inline static unsigned int CompactBy4(unsigned int x) {
            x &= 0x42108421;                  // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
            x = (x ^ (x >>  4)) & 0x40300C03; // x = -6-- ---- --54 ---- ---- 32-- ---- --10
            x = (x ^ (x >>  8)) & 0x0070000F; // x = ---- ---- -654 ---- ---- ---- ---- 3210
            x = (x ^ (x >> 16)) & 0x0000007F; // x = ---- ---- ---- ---- ---- ---- -654 3210
            return x;
        }

        /**
         * Performs a maximum of all the 5 values encoded in lhs and rhs. 
         *
         * This method assumes that all bits have been used, so the 2 most
         * significant keys consist of 7 bits.
         */
        __host__ __device__
        inline static unsigned int MinBy4(const unsigned int lhs, const unsigned int rhs) {
            return Min(lhs & 0x42108421, rhs & 0x42108421) // Min of -1-- --1- ---1 ---- 1--- -1-- --1- ---1
                + Min(lhs & 0x84210842, rhs & 0x84210842)  // Min of 1--- -1-- --1- ---1 ---- 1--- -1-- --1-
                + Min(lhs & 0x08421084, rhs & 0x08421084)  // Min of ---- 1--- -1-- --1- ---1 ---- 1--- -1--
                + Min(lhs & 0x10842108, rhs & 0x10842108)  // Min of ---1 ---- 1--- -1-- --1- ---1 ---- 1---
                + Min(lhs & 0x21084210, rhs & 0x21084210); // Min of --1- ---1 ---- 1--- -1-- --1- ---1 ----
        }

        /**
         * Performs a maximum of all the 5 values encoded in lhs and rhs. 
         *
         * This method assumes that all bits have been used, so the 2 most
         * significant keys consist of 7 bits.
         */
        __host__ __device__
        inline static unsigned int MaxBy4(const unsigned int lhs, const unsigned int rhs) {
            return Max(lhs & 0x42108421, rhs & 0x42108421) // Max of -1-- --1- ---1 ---- 1--- -1-- --1- ---1
                + Max(lhs & 0x84210842, rhs & 0x84210842)  // Max of 1--- -1-- --1- ---1 ---- 1--- -1-- --1-
                + Max(lhs & 0x08421084, rhs & 0x08421084)  // Max of ---- 1--- -1-- --1- ---1 ---- 1--- -1--
                + Max(lhs & 0x10842108, rhs & 0x10842108)  // Max of ---1 ---- 1--- -1-- --1- ---1 ---- 1---
                + Max(lhs & 0x21084210, rhs & 0x21084210); // Max of --1- ---1 ---- 1--- -1-- --1- ---1 ----
        }

        __host__ __device__
        inline static unsigned int Encode(const unsigned int x, const unsigned int y, const unsigned int z, 
                                          const unsigned int u, const unsigned int v) {
            return (PartBy4(x) << 4) + (PartBy4(y) << 3) + (PartBy4(z) << 2) +
                (PartBy4(u) << 1) + PartBy4(v);
        }
        
    };

}

#endif // _GPU_DACRT_MORTON_CURVE_H_
