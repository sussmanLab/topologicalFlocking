#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "scalarVicsekModel.cuh"
#include "functions.h"

/** \file scalarVicsekModel.cu
    * Defines kernel callers and kernels for GPU calculations of simple active 2D cell models
*/

/*!
    \addtogroup simpleEquationOfMotionKernels
    @{
*/
__global__ void scalar_vicsek_directors_kernel(double2 *velocities,
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

    //normalize and rotate by noise
    newVelocities[idx] = (1./norm(newVelocities[idx]))* newVelocities[idx];
    curandState_t randState;
    randState=RNGs[idx];
    double u = curand_uniform_double(&randState) -0.5;//uniform between -.5 and .5
    RNGs[idx] = randState;
    double theta = 2.0*PI*u*eta;
    rotate2D(newVelocities[idx],theta);
    }

__global__ void scalar_vicsek_update_kernel(double2 *forces,
                                            double2 *velocities,
                                            double2 *displacements,
                                            double v0,
                                            int N,
                                            double deltaT,
                                            double mu)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    displacements[idx] = deltaT* (v0*velocities[idx] + mu*forces[idx]);
    };

bool gpu_scalar_vicsek_update(
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double v0,
                    int N,
                    double deltaT,
                    double mu)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    scalar_vicsek_update_kernel<<<nblocks,block_size>>>(forces,velocities,displacements,v0,N,deltaT,mu);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_scalar_vicsek_directors(
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

    scalar_vicsek_directors_kernel<<<nblocks,block_size>>>(velocities,newVelocities,nNeighbors,neighbors,n_idx,RNGs,N,eta);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/** @} */ //end of group declaration
