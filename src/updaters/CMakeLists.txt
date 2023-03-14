if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc 
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/forces
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
        )

add_library(updaters
    brownianParticleDynamics.cpp
    EnergyMinimizerFIRE2D.cpp
    MullerPlatheShear.cpp
    NoseHooverChainNVT.cpp
    selfPropelledCellVertexDynamics.cpp
    selfPropelledParticleDynamics.cpp
    setTotalLinearMomentum.cpp
    scalarVicsekModel.cpp
    vectorVicsekModel.cpp
    xyLikeScalarVicsek.cpp
    )
cuda_add_library(updatersGPU
    EnergyMinimizerFIRE2D.cu
    NoseHooverChainNVT.cu
    brownianParticleDynamics.cu
    selfPropelledCellVertexDynamics.cu
    selfPropelledParticleDynamics.cu
    setTotalLinearMomentum.cu
    simpleEquationOfMotion.cu
    scalarVicsekModel.cu
    vectorVicsekModel.cu
    xyLikeScalarVicsek.cu
    )

