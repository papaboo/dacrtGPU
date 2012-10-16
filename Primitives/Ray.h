// Ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAY_H_
#define _GPU_DACRT_RAY_H_

#include <Utils/Math.h>
#include <Utils/ToString.h>

#include <string>
#include <sstream>
#include <iomanip>

struct Ray {
    float3 origin;
    int id;
    float3 direction;
    float t;
    
    Ray(const float4& o_id, const float4& direction_t)
        : origin(make_float3(o_id.x, o_id.y, o_id.z)), id(o_id.w),
          direction(make_float3(direction_t)), t(direction_t.w) {}
    
    inline std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[id: " << id << ", origin: " << origin << ", dir: " << direction << ", t: " << t << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const Ray& r){
    return s << r.ToString();
}


#endif // _GPU_DACRT_RAY_H_
