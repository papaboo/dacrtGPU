// Hyperray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_RAY_H_
#define _GPU_DACRT_HYPER_RAY_H_

#include <Enums.h>
#include <Utils/ToString.h>

#include <cutil_math.h>

#include <string>
#include <sstream>
#include <iomanip>

struct HyperRay {
    float3 origin;
    int id;
    SignedAxis axis;
    float2 UV;
    float t;

    HyperRay(const float4& o, const float4& aUV)
        : origin(make_float3(o.x, o.y, o.z)), id(o.w),
          axis((SignedAxis)aUV.x), UV(make_float2(aUV.y, aUV.z)), t(aUV.w) {}

    __host__ __device__ 
    inline float3 Direction() const { return AxisUVToDirection(make_float3(axis, UV.x, UV.y)); }

    __host__ __device__ 
    inline static SignedAxis AxisFromDirection(const float3 dir) {
        float3 absDir = make_float3(fabs(dir.x), fabs(dir.y), fabs(dir.z));
        if (absDir.x > absDir.y && absDir.x > absDir.z) // x is dominant
            return dir.x > 0.0f ? PosX : NegX;
        else if (absDir.y > absDir.z) // y is dominant
            return dir.y > 0.0f ? PosY : NegY;
        else // z is dominant
            return dir.z > 0.0f ? PosZ : NegZ;
    }

    __host__ __device__
    inline static float3 AxisUVFromDirection(const float3 dir) { return DirectionToAxisUV(dir); }
    
    __host__ __device__
    inline static float3 DirectionToAxisUV(const float3 dir) {
        // TODO encode axis as bitmask in x instead of converting from integer to float.
        float3 absDir = make_float3(fabs(dir.x), fabs(dir.y), fabs(dir.z));
        if (absDir.x > absDir.y && absDir.x > absDir.z) { // x is dominant
            return make_float3(dir.x > 0.0f ? PosX : NegX,
                               dir.y / absDir.x,
                               dir.z / absDir.x);
        } else if (absDir.y > absDir.z) { // y is dominant
            return make_float3(dir.y > 0.0f ? PosY : NegY,
                               dir.x / absDir.y,
                               dir.z / absDir.y);
        } else { // z is dominant
            return make_float3(dir.z > 0.0f ? PosZ : NegZ,
                               dir.x / absDir.z,
                               dir.y / absDir.z);
        }
    }

    __host__ __device__
    inline static float3 AxisUVToDirection(const float3 axisUV) {
        SignedAxis axis = (SignedAxis)(int)axisUV.x;
        switch(axis) {
        case PosX:
            return make_float3(1.0f, axisUV.y, axisUV.z);
        case NegX:
            return make_float3(-1.0f, axisUV.y, axisUV.z);
        case PosY:
            return make_float3(axisUV.y, 1.0f, axisUV.z);
        case NegY:
            return make_float3(axisUV.y, -1.0f, axisUV.z);
        case PosZ:
            return make_float3(axisUV.y, axisUV.z, 1.0f);
        case NegZ:
            return make_float3(axisUV.y, axisUV.z, -1.0f);
        }

        // Shaddap clang
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[id: " << id << ", origin: " << origin << ", dir: " << normalize(Direction()) << ", axis: " << axis << ", u: " << UV.x << ", v: " << UV.y << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const HyperRay& r){
    return s << r.ToString();
}


#endif _GPU_DACRT_HYPER_RAY_H_
