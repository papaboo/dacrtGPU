// Hyper ray abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_MATERIAL_H_
#define _GPU_DACRT_MATERIAL_H_

#include <cutil_math.h>

#include <thrust/device_vector.h>

#include <sstream>

struct Material {
    float3 emission, color;
    float reflection, refraction;
    
    Material(const float3& emission, const float3& color, const float reflection, const float refraction) 
        : emission(emission), color(color), reflection(reflection), refraction(refraction) {}

    Material(const float4& emission_reflection, const float4& color_refraction)
        : emission(make_float3(emission_reflection)), reflection(emission_reflection.w),
          color(make_float3(color_refraction)), refraction(color_refraction.w) {}

    std::string ToString() const;
    
};

inline std::ostream& operator<<(std::ostream& s, const Material& mat){
    return s << mat.ToString();
}


class Materials {
public:
    thrust::device_vector<float4> emission_reflection;
    thrust::device_vector<float4> color_refraction;
    
public:
    Materials(const size_t capacity) 
        : emission_reflection(thrust::device_vector<float4>(capacity)),
          color_refraction(thrust::device_vector<float4>(capacity)) {
        emission_reflection.resize(0);
        color_refraction.resize(0);
    }

    inline size_t Size() const { return emission_reflection.size(); }
    inline void Resize(const size_t size) {
        emission_reflection.resize(size);
        color_refraction.resize(size);
    }

    inline Material Get(const size_t i) const {
        return Material(emission_reflection[i], color_refraction[i]);
    }
    inline void Set(const size_t i, const Material& mat) {
        if (Size() <= i) Resize(i+1);
        emission_reflection[i] = make_float4(mat.emission, mat.reflection);
        color_refraction[i] = make_float4(mat.color, mat.refraction);
    }
    
    std::string ToString() const;    
};

inline std::ostream& operator<<(std::ostream& s, const Materials& mats){
    return s << mats.ToString();
}

#endif
