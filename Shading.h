// Shading.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SHADING_H_
#define _GPU_DACRT_SHADING_H_

#include <Rendering/Rays.h>

#include <thrust/device_vector.h>

class Fragments;
class SpheresGeometry;

namespace Rendering {
    class RayContainer;
}

class Shading {
public:

    static void Normals(Rendering::RayContainer& rays, 
                        thrust::device_vector<unsigned int>::iterator hitIDs,
                        SpheresGeometry& spheres,
                        Fragments& frags);

    // TODO Template with rusian roulette bool and set bool flag if a ray is
    // terminated, so we only scan rays when one should actually be removed.
    /**
     * Shades the fragments based on the geometry and rays intersections.
     *
     * Once done all leaf rays rays will have been reinitialize and replaced
     * with new reflections, refraction and diffuse rays.
     */
    static void Shade(Rendering::RayContainer& rays, 
                      thrust::device_vector<unsigned int>::iterator hitIDs,
                      SpheresGeometry& spheres,
                      Fragments& frags);
    
};

#endif // _GPU_DACRT_SHADING_H_
