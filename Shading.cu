// Shading.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <Shading.h>

#include <Fragment.h>
#include <Meta/CUDA.h>
#include <Primitives/Sphere.h>
#include <RayContainer.h>
#include <SphereGeometry.h>
#include <Utils/Random.h>

#include <iostream>

#define PI ((float)3.14159265358979)

__global__
void ColorNormalsKernel(float4* rayOrigins,
                        float4* rayDirections,
                        unsigned int* hitIDs, // Contains information about the new rays after 
                        const Sphere* const spheres,
                        float4* emission_bounces, // result
                        const unsigned int nRays) {

    const unsigned int rayID = threadIdx.x + blockDim.x * blockIdx.x;
    if (rayID >= nRays) return;

    const unsigned int hitID = hitIDs[rayID];
    const float4 originId = rayOrigins[rayID];
    const float3 rayOrigin = make_float3(originId);
    const unsigned int fragID = originId.w;

    if (hitID == SpheresGeometry::MISSED) {
        emission_bounces[fragID] =  make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        hitIDs[rayID] = 0; // Make a note that this ray should be terminated.
        return;
    }

    const Sphere sphere = spheres[hitID];

    const float4 dir_t = rayDirections[rayID];
    const float3 dir = make_float3(dir_t);
    const float t = dir_t.w;
    
    const float3 hitPos = t * dir + rayOrigin;
    const float3 norm = normalize(hitPos - sphere.center);
    
    // Map to visible
    emission_bounces[fragID] = make_float4(norm * 0.5f + 0.5f, 1.0f);
        
    rayOrigins[rayID] = make_float4(hitPos + norm * 0.02f, fragID);

    const float3 reflectionDir = dir - norm * 2 * dot(norm, dir);
    rayDirections[rayID] = make_float4(reflectionDir, 1e30f);

    hitIDs[rayID] = hitID == 7 ? 1 : 0; // Reflect of sphere 7, just to test it
}

/**
 * Shades the fragments with the color of the normals.
 *
 * After execution the hitIDs contains 0 for rays that bounced and 0 for
 * terminated rays.
 */
void Shading::Normals(RayContainer& rays, 
                      thrust::device_vector<unsigned int>::iterator hitIDs,
                      SpheresGeometry& spheres, 
                      Fragments& frags) {

    size_t nRays = rays.EndLeafRays() - rays.BeginLeafRays();    

    struct cudaFuncAttributes funcAttr;
    cudaFuncGetAttributes(&funcAttr, ColorNormalsKernel);
    unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    unsigned int blocks = (nRays / blocksize) + 1;
    ColorNormalsKernel<<<blocks, blocksize>>>
        (RawPointer(Rays::GetOrigins(rays.BeginLeafRays())),
         RawPointer(Rays::GetDirections(rays.BeginLeafRays())),
         RawPointer(hitIDs), // Contains information about the new rays after 
         RawPointer(spheres.spheres),
         RawPointer(frags.emissionDepth), // result
         nRays);
    CHECK_FOR_CUDA_ERROR();

    rays.RemoveTerminated(hitIDs);
}    

__device__ bool d_raysTerminated;

// TODO Place all material parameters in a struct and create access methods for
// it, so I don't have to extend the shade kernels each time I add a new
// parameter.
__global__
void PathTraceKernel(float4* rayOrigins,
                     float4* rayDirections,
                     unsigned int* hitIDs, // Contains information about the new rays after 
                     const Sphere* const spheres,
                     const unsigned int* const matIDs,
                     const float4* const emission_reflections,
                     const float4* const color_refractions,
                     float4* emission_bounces, // result
                     float4* fs, // result
                     const unsigned int nRays, 
                     const unsigned int seed) {

    const unsigned int rayID = threadIdx.x + blockDim.x * blockIdx.x;
    if (rayID >= nRays) return;
    
    const unsigned int hitID = hitIDs[rayID];
    
    const float4 originId = rayOrigins[rayID];
    const float3 rayOrigin = make_float3(originId);
    const unsigned int fragID = originId.w;

    const float4 emission_bounce = emission_bounces[fragID];

    const float3 oldF = make_float3(fs[fragID]);
    if (hitID == SpheresGeometry::MISSED) {
        const float3 backgroundColor = make_float3(0.8f, 0.8f, 0.8f);
        const float3 color = oldF * backgroundColor;
        emission_bounces[fragID] = make_float4(color, emission_bounce.w);
        hitIDs[rayID] = 0; // Make a note that this ray should be terminated.
        d_raysTerminated = true;
        return;
    }
    
    const unsigned int matID = matIDs[hitID];
    const float4 emission_reflection = emission_reflections[matID];
        
    if (emission_bounce.w >= 5) {
        emission_bounces[fragID] = emission_bounce + make_float4(oldF * make_float3(emission_reflection), 0);
        hitIDs[rayID] = 0; // Make a note that this ray should be terminated.
        d_raysTerminated = true;
        return;
    }

    const float4 dir_t = rayDirections[rayID];
    float3 dir = make_float3(dir_t);
    const float t = dir_t.w;

    const Sphere sphere = spheres[hitID];

    const float3 hitPos = t * dir + rayOrigin; // we could store hitPos in the rays origin, since we should already know it from the intersection test.
    const float3 sphereNorm = normalize(hitPos - sphere.center); // The sphere's normal, pointing away from the center.
    const bool into = dot(sphereNorm, dir) < 0.0f;
    const float3 rayNorm = into ? sphereNorm : sphereNorm*-1.0f; // May flip the normal so it doesn't point away from the ray.

    Random rand = Random::Create1D(seed);
    
    float colorContribution = 0.0f;

    if (rand.NextFloat01() < emission_reflection.w) {
        // ray is reflected
        dir = dir - rayNorm * 2.0f * dot(rayNorm, dir);
    } else if (rand.NextFloat01() < color_refractions[matID].w){
        float3 reflect = dir - rayNorm * 2.0f * dot(rayNorm, dir);
        
        // Pure magic 'borrowed' from smallpt
        float nc = 1.0f, nt = 1.5f;
        float nnt = into ? nc/nt : nt/nc;
        float ddn = dot(dir, rayNorm);
        float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
        
        if (cos2t < 0.0f) {
            dir = reflect;
        } else {
            float3 tDir = normalize(dir * nnt - rayNorm * (ddn*nnt+sqrt(cos2t)));
            float a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn : dot(tDir, sphereNorm));
            float Re=R0+(1-R0)*c*c*c*c*c;
            // float P = 0.25f + 0.5f * Re; 
            // float Tr = 1.0f - Re;
            // float RP = Re / P, TP = Tr / (1.0f-P);
            if (rand.NextFloat01() < Re) // reflection
                dir = reflect;
            else 
                dir = tDir;
        }
    } else {
    
        // ray is diffuse
        colorContribution = 1.0f;
        
        const float r1 = 2 * PI * rand.NextFloat01();
        const float r2 = rand.NextFloat01();
        const float r2s = sqrtf(r2);
        // Tangent space ?
        const float3 w = rayNorm;
        const float3 u = normalize(fabsf(w.x) > 0.1f ? 
                                   make_float3(0,1,0) : 
                                   cross(make_float3(1,0,0), w));
        const float3 v = cross(w, u);
        dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1.0f-r2));
    }

    const float4 color_refraction = color_refractions[matID];
    emission_bounces[fragID] = emission_bounce + make_float4(colorContribution * oldF * make_float3(emission_reflection), 1.0f);
    fs[fragID] = make_float4(oldF * make_float3(color_refraction), 0.0f);
    
    
    bool refract = dot(rayNorm, dir) < 0.0f;
    rayOrigins[rayID] = make_float4(hitPos + rayNorm * (refract ? -0.02f : 0.02f), fragID);
    rayDirections[rayID] = make_float4(dir, 1e30f);
    
    hitIDs[rayID] = 1; // Note that this ray should not be terminated. TODO
                       // Perhaps I should just use sphere missed to denote done
                       // rays and anything else to denote not done rays. Then I
                       // would save a couple of writes.
}

void Shading::Shade(RayContainer& rays, 
                    thrust::device_vector<unsigned int>::iterator hitIDs,
                    SpheresGeometry& spheres,
                    Fragments& frags) {

    size_t nRays = rays.EndLeafRays() - rays.BeginLeafRays();

    bool raysTerminated = false;
    cudaMemcpyToSymbol(d_raysTerminated, &raysTerminated, sizeof(bool), 0, cudaMemcpyHostToDevice);

    struct cudaFuncAttributes funcAttr;
    cudaFuncGetAttributes(&funcAttr, PathTraceKernel);
    unsigned int blocksize = funcAttr.maxThreadsPerBlock > 256 ? 256 : funcAttr.maxThreadsPerBlock;
    unsigned int blocks = (nRays / blocksize) + 1;
    PathTraceKernel<<<blocks, blocksize>>>
        (RawPointer(Rays::GetOrigins(rays.BeginLeafRays())),
         RawPointer(Rays::GetDirections(rays.BeginLeafRays())),
         RawPointer(hitIDs), // Contains information about the new rays after 
         RawPointer(spheres.spheres),
         RawPointer(spheres.materialIDs),
         RawPointer(spheres.materials.emission_reflection),
         RawPointer(spheres.materials.color_refraction),
         RawPointer(frags.emissionDepth), // result
         RawPointer(frags.f), // result
         nRays,
         rand());
    CHECK_FOR_CUDA_ERROR();

    cudaMemcpyFromSymbol(&raysTerminated, d_raysTerminated, sizeof(bool));

    if (raysTerminated)
        rays.RemoveTerminated(hitIDs);
    else
        rays.ReinitLeafRays();

}
