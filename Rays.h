// Rays abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAYS_H_
#define _GPU_DACRT_RAYS_H_

#include <Primitives/HyperRay.h>
#include <Primitives/Ray.h>
#include <Utils/Utils.h>

#include <ostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cutil_math.h>


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
        

    Rays(const size_t capacity, const Representation r = RayRepresentation)
        : origins(thrust::device_vector<float4>(capacity)),
          axisUVs(thrust::device_vector<float4>(capacity)),
          representation(r) {}

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
