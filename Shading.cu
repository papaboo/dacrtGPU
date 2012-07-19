// Shading.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Shading.h>

#include <Fragment.h>
#include <Sphere.h>
#include <SphereGeometry.h>

#include <thrust/transform.h>

#include <iostream>

struct ColorNormals {
    Sphere* spheres;
    
    float4* emissionDepth;
    
    ColorNormals(SpheresGeometry& sg,
                 Fragments& frags)
        : spheres(thrust::raw_pointer_cast(sg.spheres.data())),
          emissionDepth(thrust::raw_pointer_cast(frags.emissionDepth.data())) {}

    __host__ __device__
    thrust::tuple<unsigned int, thrust::tuple<float4, float4> > operator()(const thrust::tuple<unsigned int, thrust::tuple<float4, float4> > hitID_Ray) const {
        const unsigned int hitID = thrust::get<0>(hitID_Ray);
        const float4 originId = thrust::get<0>(thrust::get<1>(hitID_Ray));
        const float3 origin = make_float3(originId);
        const unsigned int id = originId.w;

        if (hitID == SpheresGeometry::MISSED) {
            emissionDepth[id] =  make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return 0; 
        }

        const Sphere sphere = spheres[hitID];

        const float4 dir_t = thrust::get<1>(thrust::get<1>(hitID_Ray));
        const float3 dir = make_float3(dir_t);
        const float t = dir_t.w;
        
        const float3 hitPos = origin + t * dir;
        float3 norm = normalize(hitPos - sphere.center);
        // Map to visible
        // norm = norm * 0.5f + 0.5f;
        
        emissionDepth[id] = make_float4(norm * 0.5f + 0.5f, 1.0f);

        const float3 reflectionDir = dir - norm * 2 * dot(norm, dir);
        const float3 reflectionAxisUV = HyperRay::DirectionToAxisUV(reflectionDir);
        thrust::tuple<float4, float4> newRay(make_float4(hitPos + norm * 0.02f, id), 
                                             make_float4(reflectionAxisUV, 0.0f));

        return thrust::tuple<unsigned int, thrust::tuple<float4, float4> >(hitID == 7 ? 1 : 0,
                                                                           newRay);
    }
};

/**
 * Shades the fragments with the color of the normals.
 *
 * After execution the hitIDs contains 0 for rays that bounced and 0 for
 * terminated rays.
 */
void Shading::Normals(HyperRays::Iterator raysBegin, HyperRays::Iterator raysEnd, 
                      SpheresGeometry& spheres, 
                      thrust::device_vector<unsigned int>& hitIDs,
                      Fragments& frags) {
    
    size_t rayCount = raysEnd - raysBegin;
    thrust::zip_iterator<thrust::tuple<UintIterator, HyperRays::Iterator> > hitRayBegin =
        thrust::make_zip_iterator(thrust::make_tuple(hitIDs.begin(), raysBegin));

    ColorNormals colorNormals(spheres, frags);
    thrust::transform(hitRayBegin, hitRayBegin + rayCount, //hitIDs.begin(), colorNormals);
                      hitRayBegin, colorNormals);

    // TODO Compute the next freaking rays!!!
}
