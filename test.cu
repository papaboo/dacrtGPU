// dacrtplane. A GPU ray tracer using a divide and conquor strategy instead of
// partitioning the geometry into a hierarchy.
// -----------------------------------------------------------------------------
// Copyright (C) 2012, See authors
//
// This program is open source and distributed under the New BSD License. See
// license for more detail.
// -----------------------------------------------------------------------------

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>

using std::cout;
using std::endl;

/**
 * similar issues
 * https://groups.google.com/forum/#!topic/thrust-users/0pZwBjT0n14
 */

int main(int argc, char *argv[]){

    cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl;

    {
        const int N = 7;
        thrust::host_vector<int> A(N);
        A[0] = 1; A[1] = 3; A[2] = 3; A[3] = 3; A[4] = 2; A[5] = 2; A[6] = 1;
        thrust::host_vector<int> B(N);
        B[0] = 9; B[1] = 8; B[2] = 7; B[3] = 6; B[4] = 5; B[5] = 4; B[6] = 3;
        thrust::host_vector<int> C(N);
        thrust::host_vector<int> D(N);
        
        thrust::equal_to<int> binary_pred;
        thrust::plus<int> binary_op;
        thrust::reduce_by_key(A.begin(), A.end(), B.begin(), C.begin(), D.begin(), 
                              binary_pred, binary_op);
        cout << C[0] << ", " << D[0] << endl;
        cout << C[1] << ", " << D[1] << endl;
        cout << C[2] << ", " << D[2] << endl;
        cout << C[3] << ", " << D[3] << endl;
    }

    {
        const int N = 7;
        thrust::device_vector<int> A(N);
        A[0] = 1; A[1] = 3; A[2] = 3; A[3] = 3; A[4] = 2; A[5] = 2; A[6] = 1;
        thrust::device_vector<int> B(N);
        B[0] = 9; B[1] = 8; B[2] = 7; B[3] = 6; B[4] = 5; B[5] = 4; B[6] = 3;
        thrust::device_vector<int> C(N);
        thrust::device_vector<int> D(N);
        
        thrust::equal_to<int> binary_pred;
        thrust::plus<int> binary_op;
        thrust::reduce_by_key(A.begin(), A.end(), B.begin(), C.begin(), D.begin(), 
                              binary_pred, binary_op);
        cout << C[0] << ", " << D[0] << endl;
        cout << C[1] << ", " << D[1] << endl;
        cout << C[2] << ", " << D[2] << endl;
        cout << C[3] << ", " << D[3] << endl;
    }
    
    return 0;
}
