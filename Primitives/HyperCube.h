// Hyper cube abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_CUBE_H_
#define _GPU_DACRT_HYPER_CUBE_H_

#include <Rays.h>

#include <string>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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

    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const HyperCube& c){
    return s << c.ToString();
}



class HyperCubes {
public:
    thrust::device_vector<SignedAxis> a;
    thrust::device_vector<float2> x;
    thrust::device_vector<float2> y;
    thrust::device_vector<float2> z;
    thrust::device_vector<float2> u;
    thrust::device_vector<float2> v;
    
public:

    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<SignedAxis>::iterator,
                                               Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator> > Iterator;
    typedef thrust::zip_iterator<thrust::tuple<Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator, 
                                               Float2Iterator> > BoundIterator;

    HyperCubes() {}

    HyperCubes(const size_t size)
        : a(thrust::device_vector<SignedAxis>(size)),
          x(thrust::device_vector<float2>(size)),
          y(thrust::device_vector<float2>(size)),
          z(thrust::device_vector<float2>(size)),
          u(thrust::device_vector<float2>(size)),
          v(thrust::device_vector<float2>(size)) {}
    
    void ReduceCubes(Rays::Iterator rayBegin, Rays::Iterator rayEnd, 
                     thrust::device_vector<uint2> rayPartitions,
                     const size_t cubes);

    inline Iterator Begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(a.begin(), x.begin(), y.begin(), z.begin(), u.begin(), v.begin()));
    }
    inline Iterator End() { 
        return thrust::make_zip_iterator(thrust::make_tuple(a.end(), x.end(), y.end(), z.end(), u.end(), v.end()));
    }

    inline BoundIterator BeginBounds() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), u.begin(), v.begin()));
    }
    inline BoundIterator EndBounds() { 
        return thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), u.end(), v.end()));
    }

    inline size_t Size() const { return a.size(); }
    size_t Resize(const size_t s);

    inline HyperCube Get(const size_t i) const {
        return HyperCube(a[i], x[i], y[i], z[i], u[i], v[i]);
    }

    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const HyperCubes& cs){
    return s << cs.ToString();
}

#endif // _GPU_DACRT_HYPER_CUBE_H_
