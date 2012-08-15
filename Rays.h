// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_RAYS_H_
#define _GPU_DACRT_HYPER_RAYS_H_

#include <Utils/Utils.h>

#include <ostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cutil_math.h>

struct Ray {
    float3 origin;
    int id;
    float3 direction;
    float t;
    
    Ray(const float4& o_id, const float4& direction_t)
        : origin(make_float3(o_id.x, o_id.y, o_id.z)), id(o_id.w),
          direction(make_float3(direction_t)), t(direction_t.w) {}
    
    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const Ray& r){
    return s << r.ToString();
}


struct HyperRay {
    float3 origin;
    int id;
    SignedAxis axis;
    float2 UV;
    float t;

    HyperRay(const float4& o, const float4& aUV)
        : origin(make_float3(o.x, o.y, o.z)), id(o.w),
          axis((SignedAxis)aUV.x), UV(make_float2(aUV.y, aUV.z)), t(aUV.w) {}

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
    inline static float3 DirectionToAxisUV(const float3 dir) {

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

    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const HyperRay& r){
    return s << r.ToString();
}



class Rays {
public:
    enum Representation {RayRepresentation, HyperRayRepresentation};

private:
    Representation representation;

public:

    thrust::device_vector<float4> origins; // [x, y, z, id]
    thrust::device_vector<float4> axisUVs; // [axis, u, v, t]
    
    typedef thrust::zip_iterator<thrust::tuple<Float4Iterator, Float4Iterator> > Iterator;

    static inline thrust::device_vector<float4>::iterator GetOrigins(Iterator rays) {
        return thrust::get<0>(rays.get_iterator_tuple());
    }
    static inline thrust::device_vector<float4>::iterator GetAxisUVs(Iterator rays) {
        return thrust::get<1>(rays.get_iterator_tuple());
    }
    static inline thrust::device_vector<float4>::iterator GetDirections(Iterator rays) { return GetAxisUVs(rays); }
        

    Rays(const size_t capacity)
        : origins(thrust::device_vector<float4>(capacity)),
          axisUVs(thrust::device_vector<float4>(capacity)) {}

    /**
     * Creates the rays.
     */
    Rays(const int width, const int height, const int sqrtSamples);

    /**
     * Converts the rays to a hyperray representation.
     */
    void Convert(const Representation r);
    Representation GetRepresentation() const { return representation; }

    inline Iterator Begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(origins.begin(), axisUVs.begin()));
    }
    inline Iterator End() {
        return thrust::make_zip_iterator(thrust::make_tuple(origins.end(), axisUVs.end()));
    }

    inline size_t Size() const { return origins.size(); }
    inline void Resize(const size_t size) {
        origins.resize(size);
        axisUVs.resize(size);
    }
    inline void Swap(Rays& rays) {
        origins.swap(rays.origins);
        axisUVs.swap(rays.axisUVs);
    }

    inline Ray GetAsRay(const size_t i) const {
        return Ray(origins[i], axisUVs[i]);
    }

    inline HyperRay GetAsHyperRay(const size_t i) const {
        return HyperRay(origins[i], axisUVs[i]);
    }

    std::string ToString() const;
    
};

inline std::ostream& operator<<(std::ostream& s, const Rays::Representation r){
    switch(r) {
    case Rays::RayRepresentation:
        return s << "Representation::Ray";
    case Rays::HyperRayRepresentation:
        return s << "Representation::HyperRay";
    }
    
    return s << "UNKNOWN";
}

inline std::ostream& operator<<(std::ostream& s, const Rays& rs){
    return s << rs.ToString();
}



#endif // _GPU_DACRT_RAYS_H_
