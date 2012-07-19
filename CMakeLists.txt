CMAKE_MINIMUM_REQUIRED(VERSION 2.8) 

SET(PROJ_NAME "DACRT_GPU")
PROJECT(${PROJ_NAME})

INCLUDE_DIRECTORIES(.)

# Find CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc) #include common/inc for cutils

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}-arch=sm_12;")
MESSAGE(STATUS "NVCC flags: ${CUDA_NVCC_FLAGS}")

CUDA_ADD_EXECUTABLE(${PROJ_NAME}
  Cone.h
  Cone.cu
  DacrtNode.h
  DacrtNode.cu
  dacrtGPU.cu
  Fragment.h
  HyperCube.h
  HyperCube.cu
  HyperRays.h
  HyperRays.cu
  Material.h
  Material.cu
  RayContainer.h
  RayContainer.cu
  Shading.h
  Shading.cu
  Sphere.h
  SphereContainer.h
  SphereContainer.cu
  SphereGeometry.h
  SphereGeometry.cu
)

TARGET_LINK_LIBRARIES(${PROJ_NAME}
  ${CUDA_LIBRARIES} #cuda rt
)

