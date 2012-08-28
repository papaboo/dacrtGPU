// Hyper cube abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_CUBE_H_
#define _GPU_DACRT_HYPER_CUBE_H_

#include <Enums.h>
#include <Utils/ToString.h>

#include <cutil_math.h>

#include <string>
#include <sstream>
#include <iomanip>

struct HyperCube {
    SignedAxis a;
    float2 x, y, z, u, v;
    
    __host__ __device__
    HyperCube(const SignedAxis a, const float2& x, const float2& y, const float2& z, const float2& u, const float2& v) 
        : a(a), x(x), y(y), z(z), u(u), v(v) {}

    __host__ __device__
    HyperCube(const SignedAxis a, const float4& xy, const float2& z, const float4& uv) 
        : a(a), x(make_float2(xy.x, xy.y)), y(make_float2(xy.z, xy.w)), z(z),
          u(make_float2(uv.x, uv.y)), v(make_float2(uv.z, uv.w)) {}

    __host__ __device__
    inline static float3 F(const SignedAxis a, const float u, const float v) {
        switch(a) {
        case PosX: return make_float3(1.0f, u, v);
        case NegX: return make_float3(-1.0f, u, v);
        case PosY: return make_float3(u, 1.0f, v);
        case NegY: return make_float3(u, -1.0f, v);
        case PosZ: return make_float3(u, v, 1.0f);
        case NegZ: return make_float3(u, v, -1.0f);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[axis: " << a << ", x: " << x << ", y: " << y << ", z: " << z << ", u: " << u << ", v: " << v << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const HyperCube& c){
    return s << c.ToString();
}

#endif // _GPU_DACRT_HYPER_CUBE_H_
