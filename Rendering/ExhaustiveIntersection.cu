// DACRT node
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Rendering/ExhaustiveIntersection.h>

#include <Meta/CUDA.h>
#include <Primitives/Sphere.h>
#include <Rendering/RayContainer.h>
#include <SphereGeometry.h>

namespace Rendering {

void ExhaustiveIntersection::Create(RayContainer& rs, SpheresGeometry& ss) {
    rays = &rs;
    spheres = &ss;
    
    rs.MakeLeaves();
}

__global__
void ExhaustiveIntersectionKernel(const float4* const rayOrigins, 
                                  float4* rayDirs,
                                  const unsigned int nRays,
                                  const Sphere* const spheres,
                                  const unsigned int nSpheres,
                                  unsigned int *hitIDs) {

    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= nRays) return;
    
    const float3 origin = make_float3(rayOrigins[id]);
    const float4 dir_t = rayDirs[id];
    const float3 dir = make_float3(dir_t);

    float hitT = dir_t.w;
    unsigned int hitID = SpheresGeometry::MISSED;

    for (unsigned int sphereId = 0; sphereId < nSpheres; ++sphereId) {
        const Sphere s = spheres[sphereId];
        const float t = s.Intersect(origin, dir);
        if (0 < t && t < hitT) {
            hitID = sphereId;
            hitT = t;
        }
    }
    
    rayDirs[id] = make_float4(dir, hitT);
    hitIDs[id] = hitID;
}

void ExhaustiveIntersection::FindIntersections(thrust::device_vector<unsigned int>& hits) {

    cudaFuncAttributes funcAttr;
    cudaFuncGetAttributes(&funcAttr, ExhaustiveIntersectionKernel);
    unsigned int blocksize = min(funcAttr.maxThreadsPerBlock, 256);
    unsigned int blocks = (hits.size() / blocksize) + 1;
    ExhaustiveIntersectionKernel<<<blocks, blocksize>>>
        (RawPointer(Rays::GetOrigins(rays->BeginLeafRays())),
         RawPointer(Rays::GetDirections(rays->BeginLeafRays())),
         hits.size(),
         RawPointer(spheres->spheres),
         spheres->Size(),
         RawPointer(hits));
    CHECK_FOR_CUDA_ERROR();
}

} // NS Rendering
