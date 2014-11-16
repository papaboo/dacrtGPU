CMAKE_MINIMUM_REQUIRED(VERSION 2.8) 

SET(PROJ_NAME "DACRT_GPU")
PROJECT(${PROJ_NAME})

IF(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
  ADD_DEFINITIONS(-m32 -Wall -Wextra -Wno-unused-parameter)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -m32")
  SET(CUDA_64_BIT_DEVICE_CODE OFF)
  ADD_DEFINITIONS(-DTHRUST_DEBUG=1)
  # ADD_DEFINITIONS(-o2)
ENDIF(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)

INCLUDE_DIRECTORIES(.)

# Find CUDA
FIND_PACKAGE(CUDA REQUIRED)
#INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc) #include common/inc for cutils. Doesn't exist after CUDA 5.0

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}-arch=sm_12;")
MESSAGE(STATUS "NVCC flags: ${CUDA_NVCC_FLAGS}")

CUDA_ADD_EXECUTABLE(${PROJ_NAME}
  dacrtGPU.cu
  Fragment.h
  HyperCubes.h
  HyperCubes.cu
  Kernels/ForEachWithOwners.h
  Kernels/ReduceMinMaxMortonCode.h
  Kernels/ReduceMinMaxMortonCode.cu
  Material.h
  Material.cu
  Meta/CUDA.h
  Meta/CUDA.cu
  Primitives/AABB.h
  Primitives/Cone.h
  Primitives/HyperCube.h
  Primitives/HyperRay.h
  Primitives/MortonCode.h
  Primitives/Ray.h
  Primitives/Sphere.h
  Primitives/SphereCone.h
  Shading.h
  Shading.cu
  SphereContainer.h
  SphereContainer.cu
  SphereGeometry.h
  SphereGeometry.cu
  Utils/Math.h
  Utils/Morton.h
  Utils/ToString.h
  Utils/Utils.h
)

SUBDIRS(Rendering)

TARGET_LINK_LIBRARIES(${PROJ_NAME}
  DACRT_Rendering
  ${CUDA_LIBRARIES} #cuda rt
)

SET(COMPILE_TESTS FALSE)
IF(${COMPILE_TESTS})
  #segmented reduce test
  CUDA_ADD_EXECUTABLE(TestSegReduce
    Tests/thrust_reduce_by_key.cu
  )
  TARGET_LINK_LIBRARIES(TestSegReduce
    ${CUDA_LIBRARIES} #cuda rt
  )
  
  #segmented reduce test
  CUDA_ADD_EXECUTABLE(TestPlayground
    Tests/reduce_by_key_playground.cu
  )
  TARGET_LINK_LIBRARIES(TestPlayground
    ${CUDA_LIBRARIES} #cuda rt
  )
  
  
  # WorkQueueTest
  CUDA_ADD_EXECUTABLE(TestWorkQueue
    Meta/CUDA.h
    Meta/CUDA.cu
    Tests/WorkQueueCalcsOwners.cu
  )

  TARGET_LINK_LIBRARIES(TestWorkQueue
    ${CUDA_LIBRARIES} #cuda rt
  )
  
  # WorkQueueTest
  CUDA_ADD_EXECUTABLE(TestGPURandomNumbers
    Meta/CUDA.h
    Meta/CUDA.cu
    Tests/GPURandomNumbers.cu
  )
  
  TARGET_LINK_LIBRARIES(TestGPURandomNumbers
    ${CUDA_LIBRARIES} #cuda rt
  )
ENDIF(${COMPILE_TESTS})