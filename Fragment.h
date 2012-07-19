// Fragment abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_FRAGMENT_H_
#define _GPU_DACRT_FRAGMENT_H_

#include <sstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <ToString.h>

struct Fragment {
    float4 emissionDepth; // [emission.xyz, depth]
    float4 f; // Color
    Fragment() : emissionDepth(make_float4(0,0,0,0)), f(make_float4(1,1,1,1)) {}

    Fragment(const float4& ed, const float4& f)
        : emissionDepth(ed), f(f) {}
    
    inline std::string ToString() const {
        std::ostringstream out;
        out << "[depth: " << (int)emissionDepth.w << 
            ", emission: " << make_float3(emissionDepth.x, emissionDepth.y, emissionDepth.z) << 
            ", f: " << make_float3(f.x, f.y, f.z) << "]";
        return out.str();
    }
};

class Fragments {
public:

    thrust::device_vector<float4> emissionDepth; // [emission.xyz, depth]
    thrust::device_vector<float4> f;

    typedef thrust::zip_iterator<thrust::tuple<Float4Iterator, Float4Iterator> > Iterator;

    Fragments(size_t size)
        : emissionDepth(thrust::device_vector<float4>(size)),
          f(thrust::device_vector<float4>(size)) {
        Reset();
    }

    void Reset() {
        thrust::fill(emissionDepth.begin(), emissionDepth.end(), make_float4(0,0,0,0));
        thrust::fill(f.begin(), f.end(), make_float4(1,1,1,1));
    }

    inline size_t Size() const { return emissionDepth.size(); }

    inline Iterator Begin() { 
        return thrust::make_zip_iterator(thrust::make_tuple(emissionDepth.begin(), f.begin()));
    }
    inline Iterator End() { 
        return thrust::make_zip_iterator(thrust::make_tuple(emissionDepth.end(), f.end()));
    }

    inline Float4Iterator BeginEmissionDepth() {
        return emissionDepth.begin();
    }
    inline Float4Iterator EndEmissionDepth() {
        return emissionDepth.end();
    }

    inline Fragment Get(const size_t i) const {
        return Fragment(emissionDepth[i], f[i]);
    }

    inline std::string ToString(const size_t i) const {
        return Get(i).ToString();
    }    

};

#endif // _GPU_DACRT_FRAGMENT_H_
