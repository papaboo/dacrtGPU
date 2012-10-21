// Sphere Cone
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SPHERE_CONE_H_
#define _GPU_DACRT_SPHERE_CONE_H_

#include <Primitives/HyperCube.h>
#include <Primitives/Sphere.h>
#include <Utils/Math.h>

#include <string>
#include <sstream>
#include <iomanip>

struct SphereCone {
    float3 apex;
    float spreadAngle;
    float3 direction;
    float radius;

    /**
     * Construct a sphere-cone from it's apex, direction, spreadangle and radius
     * to the apex. The direction is assumed to be normalized.
     */
    __host__ __device__
    static inline SphereCone Create(const float3& apex, const float3& dir, const float spreadAngle, const float radius) {
        // TODO Complain if it's a reflex cone.
        
        SphereCone c;
        c.apex = apex;
        c.direction = dir;
        c.spreadAngle = spreadAngle;
        c.radius = radius;
        return c;
    }

    __host__ __device__
    static inline SphereCone FromCube(const HyperCube& cube) {
        SphereCone c;

        float3 A = normalize(HyperCube::F(cube.a, cube.u.x, cube.v.y));
        float3 B = normalize(HyperCube::F(cube.a, cube.u.y, cube.v.x));
        c.direction = normalize((A + B) * 0.5f);
        
        float3 C = normalize(HyperCube::F(cube.a, cube.u.x, cube.v.x));
        float3 D = normalize(HyperCube::F(cube.a, cube.u.y, cube.v.y));
        
        // Angle in degrees
        c.spreadAngle = acos(dot(A, c.direction));
        c.spreadAngle = max(c.spreadAngle, (float)acos(dot(B, c.direction)));
        c.spreadAngle = max(c.spreadAngle, (float)acos(dot(C, c.direction)));
        c.spreadAngle = max(c.spreadAngle, (float)acos(dot(D, c.direction)));
        
        // Apex
        float3 r0 = make_float3(cube.x.x, cube.y.x, cube.z.x);
        float3 r1 = make_float3(cube.x.y, cube.y.y, cube.z.y);
        c.apex = (r0 + r1) * 0.5f;
        
        c.radius = length(c.apex - r0);
        
        return c;
    }
    __host__ __device__
    inline bool DoesIntersect(const Sphere& sphere) const {

        const float sinToAngle = std::sin(spreadAngle);
        const float cosToAngleSqr = std::cos(spreadAngle) * std::cos(spreadAngle);
        
        return DoesIntersect(sphere, 1.0f / sinToAngle, cosToAngleSqr);
    }

    /**
     * http://www.geometrictools.com/Documentation/IntersectionSphereCone.pdf
     */
    __host__ __device__    
    bool DoesIntersect(const Sphere& sphere, const float invSinToAngle, 
                       const float cosToAngleSqr) const {

        // The intersection of sphere/sphere-cone is identical to sphere/cone
        // intersection where the sphere's radius has been extended by the
        // sphere-cone's.

        const float totalRadius = sphere.radius + radius;
        const float3 U = apex - direction * (totalRadius * invSinToAngle);
        float3 D = sphere.center - U;
        float dSqr = dot(D,D);
        float e = dot(direction, D);
    
        if (e > 0.0f && e*e >= dSqr * cosToAngleSqr) {
            D = sphere.center - apex;
            dSqr = dot(D,D);
            e = -dot(direction, D);
            const float sinSqr = 1.0f - cosToAngleSqr;
            if (e > 0 && e*e >= dSqr * sinSqr)
                return dSqr <= totalRadius * totalRadius;
            else
                return true;
        }
    
        return false;
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[apex: " << apex << ", angle: " << spreadAngle << ", direction: " << direction <<  ", radius: " << radius << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const SphereCone& c){
    return s << c.ToString();
}

#endif _GPU_DACRT_SPHERE_CONE_H_
