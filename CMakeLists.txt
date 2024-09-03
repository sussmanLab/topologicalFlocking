cmake_minimum_required(VERSION 3.0.0)

project(flocking LANGUAGES CUDA CXX)

add_definitions(-DDIMENSION=2)
add_definitions(-DENABLE_CUDA)
add_definitions(-DCGAL_DISABLE_ROUNDING_MATH_CHECK)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_ARCH "30;35;50")

#find_package(OPENMP REQUIRED)
set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
                                        #")#-gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets ")#-Xptxas -O3 -Xcompiler -O3")

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

find_package(CGAL REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakeHelp)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
#find_package(SFML 
#  COMPONENTS 
#    system window graphics network audio REQUIRED)

include_directories(
            $ENV{PATH}
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/databases
   )

add_subdirectory(src)

set(myLibs
    model
    modelGPU
    updaters
    updatersGPU
    analysis
    database
    simulation
    utility
    utilityGPU
    )

link_directories(${CUDA_LIBRARY_DIRS} $ENV{LD_LIBRARY_PATH})

set(SOURCES )
set(HEADERS
    )

# list the names of cpp files corresponding to linked executables you'd like...
foreach(ARG 
        topologicalVicsek
        orderedFlocking
        metricReciprocalNonreciprocalFlocking
        reciprocalNonreciprocalFlocking
        )
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out"
    ${CUDA_LIBRARIES}
    ${myLibs} 
    CGAL::CGAL
    netcdf
    netcdf_c++
    )
endforeach()
# list the names of cpp files corresponding to linked executables you'd like...
foreach(ARG 
        visualizeFlocking
        )
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out"
    ${CUDA_LIBRARIES}
    ${myLibs} 
    CGAL::CGAL
    sfml-graphics
    sfml-system
    sfml-window
    netcdf
    netcdf_c++
    )
endforeach()
