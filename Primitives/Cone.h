// Cone
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_CONE_H_
#define _GPU_DACRT_CONE_H_

#include <HyperCube.h>
#include <Primitives/Sphere.h>

#include <string>
#include <sstream>
#include <iomanip>

#include <cutil_math.h>

#define Min(lhs, rhs) (lhs) < (rhs) ? (lhs) : (rhs)
#define Max(lhs, rhs) (lhs) > (rhs) ? (lhs) : (rhs)

struct Cone {
    float3 apex;
    float spreadAngle;
    float3 direction;
    float apexDistance;

    /**
     * Construct a cone from it's apex, direction, spreadangle and distance to the apex. The direction
     * is assumed to be normalized.
     */
    __host__ __device__
    static inline Cone MakeCone(const float3& apex, const float3& dir, const float spreadAngle, const float apexDistance) {
        Cone c;
        c.apex = apex;
        c.direction = dir;
        c.spreadAngle = spreadAngle;
        c.apexDistance = apexDistance;
        return c;
    }

    __host__ __device__
    static inline Cone FromCube(const HyperCube& cube) {
        Cone c;

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
        float3 center = (r0 + r1) * 0.5f;
        float3 negOffset = c.direction * length(r0 - r1) / (2.0f * sin(c.spreadAngle));
        c.apex = center - negOffset;
        
        //TODO determine bound
        float3 closestPointToApex = make_float3(Min(cube.x.y, Max(cube.x.x, c.apex.x)),
                                                Min(cube.y.y, Max(cube.y.x, c.apex.y)),
                                                Min(cube.z.y, Max(cube.z.x, c.apex.z)));
        c.apexDistance = length(closestPointToApex - c.apex);

        return c;
    }

    __host__ __device__
    inline bool DoesIntersect(const Sphere& sphere) const {

        const float sinToAngle = std::sin(spreadAngle);
        const float cosToAngleSqr = std::cos(spreadAngle) * cos(spreadAngle);
        
        return DoesIntersect(sphere, 1.0f / sinToAngle, cosToAngleSqr);
    }

    /**
     * http://www.geometrictools.com/Documentation/IntersectionSphereCone.pdf
     */
    __host__ __device__    
    bool DoesIntersect(const Sphere& sphere, const float invSinToAngle, 
                       const float cosToAngleSqr) const {
        // @TODO Handle reflex cones by inversion or simply throw an exception in
        // the constructor when one is created? (I dont' need them for this project
        // anyway)

        // TODO make sure sphere is further away than apex distance
    
        const float3 U = apex - direction * (sphere.radius * invSinToAngle);
        float3 D = sphere.center - U;
        float dSqr = dot(D,D);
        float e = dot(direction, D);
    
        if (e > 0.0f && e*e >= dSqr * cosToAngleSqr) {
            D = sphere.center - apex;
            dSqr = dot(D,D);
            e = -dot(direction, D);
            const float sinSqr = 1.0f - cosToAngleSqr;
            if (e > 0 && e*e >= dSqr * sinSqr)
                return dSqr <= sphere.radius * sphere.radius;
            else
                return true;
        }
    
        return false;
    }

    inline std::string ToString() const {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << "[apex: " << apex << ", angle: " << spreadAngle << ", direction: " << direction <<  ", apex distance: " << apexDistance << "]";
        return out.str();
    }
};

inline std::ostream& operator<<(std::ostream& s, const Cone& c){
    return s << c.ToString();
}


#endif // _GPU_DACRT_CONE_H_
