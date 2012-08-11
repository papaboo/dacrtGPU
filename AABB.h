// Axis aligned bounding box
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_AABB_H_
#define _GPU_DACRT_AABB_H_

#include <cutil_math.h>

struct AABB {
    float3 min;
    float3 max;

    __host__ __device__
    inline static AABB Create(const float3& min, const float3& max) {
        AABB aabb;
        aabb.min = min;
        aabb.max = max;
        return aabb;
    }
    
    __host__ __device__
    inline float3 Center() const {
        return (min + max) * 0.5f;
    }

    __host__ __device__
    inline float3 Extends() const {
        return max - min;
    }

};

#endif
