// Morton Curve utility
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_MORTON_CURVE_H_
#define _GPU_DACRT_MORTON_CURVE_H_

namespace Utils {

    class Morton {
    public:
        
        inline static unsigned int Encode(const unsigned int x, const unsigned int y, const unsigned int z, 
                                          const unsigned int u, const unsigned int v) {
            return (PartBy4(x) << 4) + (PartBy4(y) << 3) + (PartBy4(z) << 2) +
                (PartBy4(u) << 1) + PartBy4(v);
        }
        
        /**
         * Insert a 0 bit after each of the 16 low bits of x
         *
         * source: http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
         */
        inline static unsigned int Part1By1(unsigned int x) {
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
        inline static unsigned int Part1By2(unsigned int x) {
            x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
            x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
            x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
            x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
            x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
            return x;
        }

        inline static unsigned int Part1By4(unsigned int x) {
            x &= 0x0000007f;                  // x = ---- ---- ---- ---- ---- ---- -654 3210
            x = (x ^ (x << 16)) & 0x0070000F; // x = ---- ---- -654 ---- ---- ---- ---- 3210
            x = (x ^ (x <<  8)) & 0x40300C03; // x = -6-- ---- --54 ---- ---- 32-- ---- --10
            x = (x ^ (x <<  4)) & 0x42108421; // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
            return x;
        }
        
        /**
         * Inverse of Part1By1 - "Compresses" all even-indexed bits
         */
        inline static unsigned int Compact1By1(unsigned int x) {
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
        inline static unsigned int Compact1By2(unsigned int x) {
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
        inline static unsigned int Compact1By4(unsigned int x) {
            x &= 0x42108421;                  // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
            x = (x ^ (x >>  4)) & 0x40300C03; // x = -6-- ---- --54 ---- ---- 32-- ---- --10
            x = (x ^ (x >>  8)) & 0x0070000F; // x = ---- ---- -654 ---- ---- ---- ---- 3210
            x = (x ^ (x >> 16)) & 0x0000007F; // x = ---- ---- ---- ---- ---- ---- -654 3210
            return x;
        }

    };

}

#endif // _GPU_DACRT_MORTON_CURVE_H_
