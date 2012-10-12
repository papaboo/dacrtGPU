// Sphere geometry abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <SphereGeometry.h>

#include <Primitives/Cone.h>
#include <Primitives/HyperCube.h>
#include <Utils/ToString.h>

#include <iostream>
#include <sstream>
#include <iomanip>

#include <thrust/copy.h>

std::string SphereGeometry::ToString() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "[Sphere: " << s << "]";
    return out.str();
}




SpheresGeometry SpheresGeometry::CornellBox(const int n) {
    // Create materials
    Materials mats(7);
    float3 zero3 = make_float3(0.0f, 0.0f, 0.0f);
    float3 gold = make_float3(0.8314f, 0.6863f, 0.2157f);
    mats.Set(0, Material(zero3, make_float3(0.75f, 0.75f, 0.75f), 0.0f, 0.0f)); // light grey
    mats.Set(1, Material(zero3, make_float3(0.75f, 0.25f, 0.25f), 0.0f, 0.0f)); // red wall
    mats.Set(2, Material(zero3, make_float3(0.25f, 0.25f, 0.75f), 0.0f, 0.0f)); // blue wall
    mats.Set(3, Material(zero3, make_float3(0.999f, 0.999f, 0.999f), 0.0f, 1.0f)); // glass ball
    mats.Set(4, Material(zero3, make_float3(0.999f, 0.999f, 0.999f), 1.0f, 0.0f)); // mirror ball
    mats.Set(5, Material(make_float3(12.0f,12.0f,12.0f), zero3, 0.0f, 0.0f)); // light
    mats.Set(6, Material(zero3, gold, 0.3f, 0.0f)); // gold

    // Create geometry
    thrust::host_vector<Sphere> hSpheres(9 + n);
    thrust::host_vector<unsigned int> hMatIDs(9 + n);
    hSpheres[0] = Sphere(make_float3(1e5f+1.0f, 40.8f ,81.6f), 1e5f); hMatIDs[0] = 1;
    hSpheres[1] = Sphere(make_float3(-1e5+99, 40.8, 81.6), 1e5);      hMatIDs[1] = 2;
    hSpheres[2] = Sphere(make_float3(50,40.8, 1e5), 1e5);             hMatIDs[2] = 0;
    hSpheres[3] = Sphere(make_float3(50, 40.8, -1e5+170), 1e5);       hMatIDs[3] = 0;
    hSpheres[4] = Sphere(make_float3(50, 1e5, 81.6), 1e5);            hMatIDs[4] = 0;
    hSpheres[5] = Sphere(make_float3(50, -1e5+81.6, 81.6), 1e5);      hMatIDs[5] = 0;
    hSpheres[6] = Sphere(make_float3(50,681.6-.27,81.6), 600);        hMatIDs[6] = 5;
    hSpheres[7] = Sphere(make_float3(73,16.5,78), 16.5);              hMatIDs[7] = 3;
    hSpheres[8] = Sphere(make_float3(27,16.5,47), 16.5);              hMatIDs[8] = 4;
    
    for (int s = 0; s < n; ++s) {
        float radius = 1.25f + 1.75f * Rand01();
        float3 center = make_float3(10.0f + Rand01() * 80.0 , Rand01() * 80.0 , Rand01() * 100.0 + 50.0);
        hSpheres[9 + s] = Sphere(center, radius);
        hMatIDs[9 + s] = 6;
    }
    
    AABB bounds = AABB::Create(make_float3(1.0f, 0.0f, 0.0f), 
                               make_float3(99.0f, 81.6f, 170.0f));
                               
    return SpheresGeometry(hSpheres, bounds, hMatIDs, mats);
}

std::string SpheresGeometry::ToString() const {
    std::ostringstream out;
    out << "Sphere geometry";
    for (size_t i = 0; i < Size(); ++i)
        out << "\n" << i << ": " << Get(i);
    return out.str();
}
