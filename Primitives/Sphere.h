// Sphere.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SPHERE_H_
#define _GPU_DACRT_SPHERE_H_

#include <Utils/ToString.h>

#include <math_functions.h>

#include <sstream>
#include <iomanip>

struct Sphere {
    float3 center;
    float radius;
    
    Sphere() {}

    Sphere(const float3& c, const float r) 
        : center(c), radius(r) {}

    /** 
     * Returns distance, 0 if no hit.
     *
     * Assumes that rayDir is normalized.
     */
    __host__ __device__
    inline float Intersect(const float3& rayOrigin, const float3& rayDir) const {
        // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        const float eps = 1e-4;
        float3 dir = center - rayOrigin;
        float b = dot(dir, rayDir);
        float det = b*b - dot(dir, dir) + radius * radius;
        if (det < 0) return 0; else det = sqrt(det);
        float t;
        return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
    }

    std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[center: " << center << ", radius: " << radius << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const Sphere& sphere){
    return s << sphere.ToString();
}

#endif // _GPU_DACRT_SPHERE_GEOMETRY_H_
