// Shading.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SHADING_H_
#define _GPU_DACRT_SHADING_H_

#include <Rays.h>

#include <thrust/device_vector.h>

class Fragments;
class RayContainer;
class SpheresGeometry;

class Shading {
public:

    static void Normals(Rays::Iterator raysBegin, Rays::Iterator raysEnd, 
                        thrust::device_vector<unsigned int>::iterator hitIDs,
                        SpheresGeometry& spheres,
                        Fragments& frags);

    // TODO Function should return a new set of rays (or perhaps simply take a
    // hit generator as argument to launch the next set of shaded rays)
    // Template with rusian roulette bool and set bool flag if a ray is
    // terminated, so we only scan rays when one should actually be removed.
    /**
     * Shades the fragments based on the geometry and rays intersections.
     *
     * Once done all leaf rays rays will have been reinitialize and replaced
     * with new reflections, refraction and diffuse rays.
     */
    static void Shade(RayContainer& rays, 
                      thrust::device_vector<unsigned int>::iterator hitIDs,
                      SpheresGeometry& spheres,
                      Fragments& frags);
    
};

#endif // _GPU_DACRT_SHADING_H_
