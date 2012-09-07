// Axis aligned bounding box
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_AABB_H_
#define _GPU_DACRT_AABB_H_

#include <Utils/Math.h>
#include <Utils/ToString.h>

#include <cutil_math.h>

#include <sstream>
#include <iomanip>

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
    inline float3 Size() const {
        return max - min;
    }

    __host__ __device__
    inline float3 InvertedSize() const {
        const float3 size = Size();
        return make_float3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z);
    }

    __host__ __device__
    inline bool ClosestIntersection(const float3& rayOrigin, const float3& rayDir, float& tHit) const {
        float3 minTs = (min - rayOrigin) / rayDir;
        float3 maxTs = (max - rayOrigin) / rayDir;
        
        float nearT = Min(minTs.x, maxTs.x);
        nearT = Max(nearT, Min(minTs.y, maxTs.y));
        nearT = Max(nearT, Min(minTs.z, maxTs.z));
        
        float farT = Max(minTs.x, maxTs.x);
        farT = Min(farT, Max(minTs.y, maxTs.y));
        farT = Min(farT, Max(minTs.z, maxTs.z));
     
        tHit = nearT;

        return nearT <= farT && 0.0f <= farT;
    }

    std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[min: " << min << ", max: " << max << "]";
        return out.str();
    }
    
};

inline std::ostream& operator<<(std::ostream& s, const AABB& aabb){
    return s << aabb.ToString();
}

#endif
