// Sphere geometry abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SPHERE_GEOMETRY_H_
#define _GPU_DACRT_SPHERE_GEOMETRY_H_

#include <Material.h>
#include <Sphere.h>
#include <Utils.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <ostream>

#include <cutil_math.h>

class HyperCubes;
class SphereContainer;

struct SphereGeometry {
    Sphere s;
    // TODO Materials with reflection val, refraction val, color ...
    
    SphereGeometry(const Sphere& s) : s(s) {}
    
    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const SphereGeometry& sg){
    return s << sg.ToString();
}



class SpheresGeometry {
public:
    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<Sphere>::iterator, 
                                               UintIterator> > Iterator;

    thrust::device_vector<Sphere> spheres; // [x, y, z, radius]
    thrust::device_vector<unsigned int> materialIDs;
    Materials materials;

    const static unsigned int MISSED = 2147483647;

    SpheresGeometry(const size_t size)
        : spheres(thrust::device_vector<Sphere>(size)),
          materialIDs(thrust::device_vector<unsigned int>(size)),
          materials(0) {}

    SpheresGeometry(const thrust::host_vector<Sphere>& hostSpheres,
                    const thrust::host_vector<unsigned int>& hostMatIDs,
                    Materials& mats)
        : spheres(hostSpheres), materialIDs(hostMatIDs), materials(mats) {}

    inline size_t Size() const { return spheres.size(); }

    inline SphereGeometry Get(const size_t i) const {
        return SphereGeometry(spheres[i]);
    }
    inline void Set(const size_t i, const SphereGeometry& s) {
        spheres[i] = s.s;
    }

    inline thrust::device_vector<Sphere>::iterator BeginSpheres() { return spheres.begin(); }
    inline thrust::device_vector<Sphere>::iterator EndSpheres() { return spheres.end(); }

    inline Iterator Begin() { 
        return thrust::make_zip_iterator(thrust::make_tuple(spheres.begin(), materialIDs.begin()));
    }
    inline Iterator End() { 
        return thrust::make_zip_iterator(thrust::make_tuple(spheres.end(), materialIDs.end()));
    }

    /**
     * Creates a Cornell Box scene with 2 large spheres and n random ones. Sets
     * all owners to 0.
     */
    static SpheresGeometry CornellBox(const int n);
    
    std::string ToString() const;
};

inline std::ostream& operator<<(std::ostream& s, const SpheresGeometry& sg){
    return s << sg.ToString();
}

#endif
