// Shading.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_SHADING_H_
#define _GPU_DACRT_SHADING_H_

#include <HyperRays.h>

#include <thrust/device_vector.h>

class Fragments;
class SpheresGeometry;

class Shading {
public:

    static void Normals(HyperRays::Iterator raysBegin, HyperRays::Iterator raysEnd, 
                        thrust::device_vector<unsigned int>::iterator hitIDs,
                        SpheresGeometry& spheres,
                        Fragments& frags);

    static void Shade(HyperRays::Iterator raysBegin, HyperRays::Iterator raysEnd, 
                      thrust::device_vector<unsigned int>::iterator hitIDs,
                      SpheresGeometry& spheres,
                      Fragments& frags);
    
};

#endif // _GPU_DACRT_SHADING_H_
