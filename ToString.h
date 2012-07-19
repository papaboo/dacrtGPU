// ToString helpers
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_TO_STRING_H_
#define _GPU_DACRT_TO_STRING_H_

#include <sstream>
#include <thrust/device_vector.h>

inline std::ostream& operator<<(std::ostream& s, const int2& v){
    s << "[" << v.x << ", " << v.y << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const int3& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const int4& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const uint2& v){
    s << "[" << v.x << ", " << v.y << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uint3& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const uint4& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const float2& v){
    s << "[" << v.x << ", " << v.y << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const float3& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return s;
}
inline std::ostream& operator<<(std::ostream& s, const float4& v){
    s << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return s;
}

template<class T>
inline std::ostream& operator<<(std::ostream& s, const thrust::device_vector<T>& v){
    for (int i = 0; i < v.size(); ++i) {
        s << i << ": " << v[i];
        if (i < v.size()-1)
            s << "\n";
    }
    return s;
}

#endif
