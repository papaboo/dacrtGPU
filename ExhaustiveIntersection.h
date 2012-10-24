// DACRT node
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_EXHAUSTIVE_INTERSECTION_H_
#define _GPU_DACRT_EXHAUSTIVE_INTERSECTION_H_

#include <IRayTracer.h>

#include <thrust/device_vector.h>

class RayContainer;
class SpheresGeometry;

class ExhaustiveIntersection : public IRayTracer {
private:
    // Initialized by Create's references, don't destroy.
    RayContainer* rays;
    SpheresGeometry* spheres;

public:
    void Create(RayContainer& rays, SpheresGeometry& spheres);

    void FindIntersections(thrust::device_vector<unsigned int>& hits);

};

#endif
