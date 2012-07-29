// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_RAYS_H_
#define _GPU_DACRT_HYPER_RAYS_H_

#include <Utils.h>

#include <ostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cutil_math.h>

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



class HyperRays {
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
        

    HyperRays(const size_t size)
        : origins(thrust::device_vector<float4>(size)),
          axisUVs(thrust::device_vector<float4>(size)) {}

    /**
     * Creates the rays.
     */
    HyperRays(const int width, const int height, const int sqrtSamples);

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
    inline void Swap(HyperRays& rays) {
        origins.swap(rays.origins);
        axisUVs.swap(rays.axisUVs);
    }

    inline HyperRay Get(const size_t i) const {
        return HyperRay(origins[i], axisUVs[i]);
    }

    std::string ToString() const;
    
};

inline std::ostream& operator<<(std::ostream& s, const HyperRays& rs){
    return s << rs.ToString();
}



#endif // _GPU_DACRT_RAYS_H_
