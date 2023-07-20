set(PROJECT "FAST-CON_MSMF")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

#cmake -D ARCH=35 -D CMAKE_BUILD_TYPE=Release ..
# ------------------------------------------------------------------
# CHECK C++ 11 SUPPORT AND CUDA >= 7.0

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

if(CUDA_VERSION_MAJOR LESS 7)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v7.0   Founded: Cuda Toolkit v${CUDA_VERSION_MAJOR}")
endif()

# ------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/cub-1.4.0/
					${PROJECT_SOURCE_DIR}/include/device/
					${PROJECT_SOURCE_DIR}/Util/
					${PROJECT_SOURCE_DIR}/include/host/)

set(HOST_SRC 	${PROJECT_SOURCE_DIR}/src/host/graph.cpp)
set(DEVICE_SRC 	${PROJECT_SOURCE_DIR}/src/device/cudaGraph.cu)

set(UTIL_SRC 	${PROJECT_SOURCE_DIR}/Util/fUtil.cpp
				${PROJECT_SOURCE_DIR}/Util/readGraphHeader.cpp
				${PROJECT_SOURCE_DIR}/Util/readGraphSTD.cpp
				${PROJECT_SOURCE_DIR}/Util/readUtil.cpp
				${PROJECT_SOURCE_DIR}/Util/mersenne-twister.cpp)

# ------------------------------------------------------------------
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--std=c++11")

find_library(
	DEVRT_LIBRARY cudadevrt
    HINTS  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

#-------------------------------------------------------------------------------

set(ARCH "" CACHE STRING "compute capability")

if(ARCH STREQUAL "")
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/build/getDeviceInfo)
        execute_process(COMMAND nvcc
                        ${PROJECT_SOURCE_DIR}/src/getDeviceInfo.cu
                        -o getDeviceInfo)
    endif()
    execute_process(COMMAND ./getDeviceInfo OUTPUT_VARIABLE OUT_VARIABLE)

    list(GET OUT_VARIABLE 0 ARCH)
    list(GET OUT_VARIABLE 1 SM)
endif()

if(NOT SM STREQUAL "")
    message("\n==> GPU architecture: ${ARCH}"
            " with ${SM} streaming multiprocessor\n")
else()
    message("\n==> GPU architecture: ${ARCH}\n")
endif()

if (ARCH STREQUAL "30")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30" "-DARCH=300")
elseif (ARCH STREQUAL "32")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_32" "-DARCH=320")
elseif (ARCH STREQUAL "35")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35" "-DARCH=350")
elseif (ARCH STREQUAL "37")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_37" "-DARCH=370")
elseif (ARCH STREQUAL "50")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50" "-DARCH=500")
elseif (ARCH STREQUAL "52")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52" "-DARCH=520")
elseif (ARCH STREQUAL "53")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53" "-DARCH=530")
else()
	message(FATAL_ERROR "\nplease select GPU architecture:
                        \"cmake -DARCH=[ 30 | 32 | 35 | 37 | 50 | 52 | 53 ]\"")
endif()

# ------------------------------------------------------------------------------
#"--resource-usage"
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--use_fast_math" "-Xcompiler=-O3,-march=native,-funroll-loops,-DNDEBUG")

#===============================================================================
#===============================================================================

#release options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -DNDEBUG")

cuda_add_executable(fast-con-msmf ${PROJECT_SOURCE_DIR}/main.cpp ${HOST_SRC} ${DEVICE_SRC} ${UTIL_SRC})
target_link_libraries(fast-con-msmf ${DEVRT_LIBRARY})

set(EXECUTABLE_OUTPUT_PATH .)
add_custom_target(rm rm -rf *)
