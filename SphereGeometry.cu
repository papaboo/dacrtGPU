// Sphere geometry abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <SphereGeometry.h>

#include <Cone.h>
#include <HyperCube.h>
#include <ToString.h>

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
    thrust::device_vector<Sphere> hSpheres(9 + n);
    hSpheres[0] = Sphere(make_float3(1e5f+1.0f, 40.8f ,81.6f), 1e5f);
    hSpheres[1] = Sphere(make_float3(-1e5+99, 40.8, 81.6), 1e5);
    hSpheres[2] = Sphere(make_float3(50,40.8, 1e5), 1e5);
    hSpheres[3] = Sphere(make_float3(50, 40.8, -1e5+170), 1e5);
    hSpheres[4] = Sphere(make_float3(50, 1e5, 81.6), 1e5);
    hSpheres[5] = Sphere(make_float3(50, -1e5+81.6, 81.6), 1e5);
    hSpheres[6] = Sphere(make_float3(50,681.6-.27,81.6), 600);
    hSpheres[7] = Sphere(make_float3(73,16.5,78), 16.5);
    hSpheres[8] = Sphere(make_float3(27,16.5,47), 16.5);
    
    for (int s = 0; s < n; ++s) {
        float radius = 1.25f + 1.75f * Rand01();
        float3 center = make_float3(10.0f + Rand01() * 80.0 , Rand01() * 80.0 , Rand01() * 100.0 + 50.0);
        hSpheres[9 + s] = Sphere(center, radius);
    }
    
    return SpheresGeometry(hSpheres);
}

std::string SpheresGeometry::ToString() const {
    std::ostringstream out;
    out << "Sphere geometry";
    for (size_t i = 0; i < Size(); ++i)
        out << "\n" << i << ": " << Get(i);
    return out.str();
}
