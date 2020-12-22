#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "vectorVicsekModel.cuh"
#include "functions.h"

/** \file vectorVicsekModel.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/
__global__ void vector_vicsek_directors_kernel(double2 *velocities,
                                               double2 *newVelocities,
                                               int *nNeighbors,
                                               int *neighbors,
                                               Index2D n_idx,
                                               curandState *RNGs,
                                               int N,
                                               double eta)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    int m = nNeighbors[idx];
    newVelocities[idx] = velocities[idx];
    for (int jj = 0; jj < m; ++jj)
        newVelocities[idx] = newVelocities[idx]+ velocities[neighbors[n_idx(jj,idx)]];

    curandState_t randState;
    randState=RNGs[idx];
    double u = 2.0*PI*(curand_uniform_double(&randState));//uniform between 0 and 2Pi
    RNGs[idx] = randState;
    
    double2 randomVector;
    randomVector.x = eta*(m+1)*cos(u);
    randomVector.y = eta*(m+1)*sin(u);
    newVelocities[idx] = newVelocities[idx]+ randomVector;
    newVelocities[idx] = (1./norm(newVelocities[idx]))* newVelocities[idx];
    }

bool gpu_vector_vicsek_directors(
                    double2 *velocities,
                    double2 *newVelocities,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double eta)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    vector_vicsek_directors_kernel<<<nblocks,block_size>>>(velocities,newVelocities,nNeighbors,neighbors,n_idx,RNGs,N,eta);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/** @} */ //end of group declaration
