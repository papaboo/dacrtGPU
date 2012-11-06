// DACRT node using morton keys to sort rays.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#ifndef _GPU_DACRT_RAY_TRACER_INTERFACE_H_
#define _GPU_DACRT_RAY_TRACER_INTERFACE_H_

#include <thrust/device_vector.h>

class SpheresGeometry;

namespace Rendering {

    class RayContainer;
    
    class IRayTracer {
        /**
         * Prepares the rays and geometry for intersection testing.
         */
        virtual void Create(RayContainer& rayContainer, SpheresGeometry& SphereContainer) = 0;
        
        /**
         * Performs the actual intersection tests between rays and geometry.
         */
        virtual void FindIntersections(thrust::device_vector<unsigned int>& hitIDs) = 0;
    };
    
} // NS Rendering

#endif // _GPU_DACRT_RAY_TRACER_INTERFACE_H_
