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

#define PI ((float)3.14159265358979)

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
                      thrust::device_vector<unsigned int>::iterator hitIDs,
                      SpheresGeometry& spheres, 
                      Fragments& frags) {
    
    size_t rayCount = raysEnd - raysBegin;
    thrust::zip_iterator<thrust::tuple<UintIterator, HyperRays::Iterator> > hitRayBegin =
        thrust::make_zip_iterator(thrust::make_tuple(hitIDs, raysBegin));

    ColorNormals colorNormals(spheres, frags);
    thrust::transform(hitRayBegin, hitRayBegin + rayCount,
                      hitRayBegin, colorNormals);
}    



struct ShadeKernel {
    Sphere* spheres;
    unsigned int* matIDs;
    
    float4 *emission_reflections, *color_refractions;
    
    float4 *emission_bounces, *fs;
    float* random;

    ShadeKernel(SpheresGeometry& sg,
          Fragments& frags)
        : spheres(thrust::raw_pointer_cast(sg.spheres.data())),
          matIDs(thrust::raw_pointer_cast(sg.materialIDs.data())),
          color_refractions(thrust::raw_pointer_cast(sg.materials.color_refraction.data())),
          emission_reflections(thrust::raw_pointer_cast(sg.materials.emission_reflection.data())),
          emission_bounces(thrust::raw_pointer_cast(frags.emissionDepth.data())),
          fs(thrust::raw_pointer_cast(frags.f.data())) {
        thrust::host_vector<float> host_random(sg.spheres.size());
        thrust::generate(host_random.begin(), host_random.end(), Rand01);
        thrust::device_vector<float> gpu_random = host_random;
        random = thrust::raw_pointer_cast(gpu_random.data());
    }
    
    __host__ __device__
    thrust::tuple<unsigned int, thrust::tuple<float4, float4> > operator()(const thrust::tuple<unsigned int, thrust::tuple<float4, float4>, float2> hitID_Ray_rand) const {
        const unsigned int hitID = thrust::get<0>(hitID_Ray_rand);
        const float4 originId = thrust::get<0>(thrust::get<1>(hitID_Ray_rand));
        const float3 origin = make_float3(originId);
        const unsigned int id = originId.w;

        const float4 emission_bounce = emission_bounces[id];

        const float3 oldF = make_float3(fs[id]);
        if (hitID == SpheresGeometry::MISSED) {
            const float3 backgroundColor = make_float3(0.8f, 0.8f, 0.8f);
            const float3 color = oldF * backgroundColor;
            emission_bounces[id] = make_float4(color, emission_bounce.w);
            return thrust::tuple<unsigned int, thrust::tuple<float4, float4> >
                (0, thrust::get<1>(hitID_Ray_rand));
        }
        
        const unsigned int matID = matIDs[hitID];
        const float4 emission_reflection = emission_reflections[matID];
        // If bounces above max bounce then terminate. (If that is all we should
        // do we can do that in a seperate kernel)
        if (emission_bounce.w >= 5) {
            emission_bounces[id] = emission_bounce + make_float4(oldF * make_float3(emission_reflection), 0);
            return thrust::tuple<unsigned int, thrust::tuple<float4, float4> >
                (0, thrust::get<1>(hitID_Ray_rand));
        }
        
        const Sphere sphere = spheres[hitID];

        const float4 dir_t = thrust::get<1>(thrust::get<1>(hitID_Ray_rand));
        float3 dir = make_float3(dir_t);
        const float t = dir_t.w;
        
        const float3 hitPos = origin + t * dir; // we could store hitPos in the rays origin, since we should already know it from the intersection test.
        float3 norm = normalize(hitPos - sphere.center);
        const bool into = dot(norm, dir) > 0.0f;
        norm = into ? norm * -1.0f : norm;
        
        float2 rand = thrust::get<2>(hitID_Ray_rand); // first rand determines  ray type, second ray bounce dir
        float colorContribution = 0.0f;;
        const float refraction = color_refractions[matID].w;
        if (rand.x < emission_reflection.w) {
            // ray is reflected
            dir = dir - norm * 2 * dot(norm, dir);
        } else if (rand.x < (emission_reflection.w + refraction)) {
            // ray is refracted
            float nc = 1.0f, nt = 1.5f;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = dot(dir, norm);
            float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
            if (cos2t < 0.0f) {
                // Total internal reflections
                dir = dir - norm * 2 * dot(norm, dir);
            } else {
                float3 tDir = normalize(dir * nnt - norm * (ddn * nnt + sqrt(cos2t)));
                float a = nt-nc, b = nt+nc, R0 = a*a/(b*b), c = 1.0f-(into ? -ddn : dot(tDir, into? norm :-norm));
                float Re = R0+(1-R0)*c*c*c*c*c; // float Tr = 1.0f-Re;
                float P = 0.25f + 0.5f * Re; 
                // float RP = Re / P, TP = Tr / (1.0f-P);
                if (rand.y < P) // reflection
                    dir = dir - norm * 2 * dot(norm, dir);
                else 
                    dir = tDir;
            }
        }else {
            // ray is diffuse
            colorContribution = 1.0f;

            // Mod rand.x to 0.0 - 1.0f value
            const float randMod = emission_reflection.w + refraction;
            rand.x = (rand.x - randMod) / (1.0f - randMod);
            
            const float r1 = 2 * PI * rand.y;
            const float r2 = rand.x; // need anothor rand value, mod rand.x with reflection and refraction values?
            const float r2s = sqrtf(r2);
            // Tangent space ?
            const float3 w = norm;
            const float3 u = normalize(fabsf(w.x) > 0.1f ? 
                                       make_float3(0,1,0) : 
                                       cross(make_float3(1,0,0), w));
            const float3 v = cross(w, u);
            dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1-r2));
        }

        const float4 color_refraction = color_refractions[matID];
        emission_bounces[id] = emission_bounce + make_float4(colorContribution * oldF * make_float3(emission_reflection), 1.0f);
        fs[id] = make_float4(oldF * make_float3(color_refraction), 0.0f);
        thrust::tuple<float4, float4> newRay(make_float4(hitPos + norm * 0.02f, id), 
                                             make_float4(HyperRay::DirectionToAxisUV(dir), 0.0f));
        
        return thrust::tuple<unsigned int, thrust::tuple<float4, float4> >(1, newRay);
    }
};

inline float2 RandomFloat2() {
    float x = (float)rand() / (float)RAND_MAX;
    float y = (float)rand() / (float)RAND_MAX;
    return make_float2(x, y);
}

void Shading::Shade(HyperRays::Iterator raysBegin, HyperRays::Iterator raysEnd, 
                           thrust::device_vector<unsigned int>::iterator hitIDs,
                           SpheresGeometry& spheres,
                           Fragments& frags) {

    size_t rayCount = raysEnd - raysBegin;

    // Generate random numbers
    static thrust::host_vector<float2> host_random(rayCount);
    static thrust::device_vector<float2> random(rayCount);
    host_random.resize(rayCount);
    thrust::generate(host_random.begin(), host_random.end(), RandomFloat2);
    random = host_random;
    
    thrust::zip_iterator<thrust::tuple<UintIterator, HyperRays::Iterator, Float2Iterator> > hitRayBegin =
        thrust::make_zip_iterator(thrust::make_tuple(hitIDs, raysBegin, random.begin()));

    thrust::zip_iterator<thrust::tuple<UintIterator, HyperRays::Iterator> > hitRayRes =
        thrust::make_zip_iterator(thrust::make_tuple(hitIDs, raysBegin));

    thrust::transform(hitRayBegin, hitRayBegin + rayCount,
                      hitRayRes, ShadeKernel(spheres, frags));
}
