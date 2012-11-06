// List of hyper cubes
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_HYPER_CUBES_H_
#define _GPU_DACRT_HYPER_CUBES_H_

#include <Primitives/HyperCube.h>
#include <Rendering/Rays.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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
    
    void ReduceCubes(Rendering::Rays::Iterator rayBegin, Rendering::Rays::Iterator rayEnd, 
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


#endif // _GPU_DACRT_HYPER_CUBES_H_
