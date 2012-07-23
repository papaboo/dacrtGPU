// Hyper cube abstraction
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <HyperCube.h>

#include <DacrtNode.h>
#include <HyperRays.h>

#include <ostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <ToString.h>
#include <Utils.h>

// *** HyperCube ***

std::string HyperCube::ToString() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "[axis: " << a << ", x: " << x << ", y: " << y << ", z: " << z << ", u: " << u << ", v: " << v << "]";
    return out.str();
}



// *** HyperCubes ***
struct PrepReduceData {
    __host__ __device__
    thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> operator()(thrust::tuple<float4, float4> v) {
        const float4 xyz = thrust::get<0>(v);
        const float4 uv = thrust::get<1>(v);
        return thrust::tuple<SignedAxis, float2, float2, float2, float2, float2>((SignedAxis)(int)uv.x,
                                                                                 make_float2(xyz.x, xyz.x), 
                                                                                 make_float2(xyz.y, xyz.y), 
                                                                                 make_float2(xyz.z, xyz.z),
                                                                                 make_float2(uv.y, uv.y),
                                                                                 make_float2(uv.z, uv.z));
    }
};

struct ReduceHyperCubes {
    __host__ __device__
    thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> 
    operator()(thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> lhs, 
               thrust::tuple<SignedAxis, float2, float2, float2, float2, float2> rhs) {

        const float2 lhsX = thrust::get<1>(lhs);
        const float2 rhsX = thrust::get<1>(rhs);
        const float2 X = make_float2(Min(lhsX.x, rhsX.x), Max(lhsX.y, rhsX.y));

        const float2 lhsY = thrust::get<2>(lhs);
        const float2 rhsY = thrust::get<2>(rhs);
        const float2 Y = make_float2(Min(lhsY.x, rhsY.x), Max(lhsY.y, rhsY.y));

        const float2 lhsZ = thrust::get<3>(lhs);
        const float2 rhsZ = thrust::get<3>(rhs);
        const float2 Z = make_float2(Min(lhsZ.x, rhsZ.x), Max(lhsZ.y, rhsZ.y));
        
        const float2 lhsU = thrust::get<4>(lhs);
        const float2 rhsU = thrust::get<4>(rhs);
        const float2 U = make_float2(Min(lhsU.x, rhsU.x), Max(lhsU.y, rhsU.y));

        const float2 lhsV = thrust::get<5>(lhs);
        const float2 rhsV = thrust::get<5>(rhs);
        const float2 V = make_float2(Min(lhsV.x, rhsV.x), Max(lhsV.y, rhsV.y));

        return thrust::tuple<SignedAxis, float2, float2, float2, float2, float2>(thrust::get<0>(lhs), X, Y, Z, U, V);
    }
};

void HyperCubes::ReduceCubes(HyperRays::Iterator rayBegin, HyperRays::Iterator rayEnd, 
                             thrust::device_vector<uint2> rayPartitions,
                             const size_t cubes) {

    Resize(cubes);

    unsigned int rayRange = rayEnd - rayBegin;
    thrust::device_vector<SignedAxis> A(rayRange);
    thrust::device_vector<float2> X(rayRange);
    thrust::device_vector<float2> Y(rayRange);
    thrust::device_vector<float2> Z(rayRange);
    thrust::device_vector<float2> U(rayRange);
    thrust::device_vector<float2> V(rayRange);

    HyperCubes::Iterator transBegin =
        thrust::make_zip_iterator(thrust::make_tuple(A.begin(), X.begin(), Y.begin(), Z.begin(), U.begin(), V.begin()));
    
    PrepReduceData prepReduceData;
    thrust::transform(rayBegin, rayEnd, transBegin, prepReduceData);

    // TODO HACK Move everything to the host because reduce_by_key fails on 330M
    // TODO Remove owners, at the least reduce by a partition start flag instead
    // e.g. |1|0|0|1|0|0|0|
    thrust::device_vector<unsigned int> owners(rayRange);
    DacrtNodes::CalcOwners(rayPartitions, owners);
    thrust::host_vector<unsigned int> hostRayOwners(owners.begin(), owners.end());
    thrust::host_vector<SignedAxis> hostA = A;
    thrust::host_vector<float2> hostX = X;
    thrust::host_vector<float2> hostY = Y;
    thrust::host_vector<float2> hostZ = Z;
    thrust::host_vector<float2> hostU = U;
    thrust::host_vector<float2> hostV = V;
    
    thrust::zip_iterator<thrust::tuple<thrust::host_vector<SignedAxis>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator> > hostValuesBegin = 
        thrust::make_zip_iterator(thrust::make_tuple(hostA.begin(), hostX.begin(), hostY.begin(), 
                                                     hostZ.begin(), hostU.begin(), hostV.begin()));
    
    static thrust::host_vector<int> cubeOwners(128); cubeOwners.resize(Size());

    thrust::host_vector<SignedAxis> hostResA(Size());
    thrust::host_vector<float2> hostResX(Size());
    thrust::host_vector<float2> hostResY(Size());
    thrust::host_vector<float2> hostResZ(Size());
    thrust::host_vector<float2> hostResU(Size());
    thrust::host_vector<float2> hostResV(Size());
    
    thrust::zip_iterator<thrust::tuple<thrust::host_vector<SignedAxis>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator,
        thrust::host_vector<float2>::iterator> > hostCubeVals = 
        thrust::make_zip_iterator(thrust::make_tuple(hostResA.begin(), 
                                                     hostResX.begin(), 
                                                     hostResY.begin(), 
                                                     hostResZ.begin(), 
                                                     hostResU.begin(), 
                                                     hostResV.begin()));
    
    thrust::reduce_by_key(hostRayOwners.begin(), hostRayOwners.end(),
                          hostValuesBegin, cubeOwners.begin(), hostCubeVals, 
                          thrust::equal_to<int>(), ReduceHyperCubes());
    
    a = hostResA;
    x = hostResX;
    y = hostResY;
    z = hostResZ;
    u = hostResU;
    v = hostResV;
}

size_t HyperCubes::Resize(const size_t s) {
    a.resize(s);
    x.resize(s);
    y.resize(s);
    z.resize(s);
    u.resize(s);
    v.resize(s);
    return Size();
}

std::string HyperCubes::ToString() const {
    std::ostringstream out;
    out << "HyperCubes:";
    for (size_t i = 0; i < Size(); ++i)
        out << "\n" << i << ": " << Get(i);
    return out.str();
}
