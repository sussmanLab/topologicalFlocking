#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "xyLikeScalarVicsek.cuh"
#include "functions.h"

__global__ void xyLike_scalar_vicsek_directors_kernel(double2 *velocities,
                                               double2 *newVelocities,
                                               int *nNeighbors,
                                               int *neighbors,
                                               double reciprocalNormalization,
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
bool gpu_xyLike_scalar_vicsek_directors(
                    double2 *velocities,
                    double2 *newVelocities,
                    int *nNeighbors,
                    int *neighbors,
                    double reciprocalNormalization,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double eta)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    xyLike_scalar_vicsek_directors_kernel<<<nblocks,block_size>>>(velocities,newVelocities,nNeighbors,neighbors,reciprocalNormalization,n_idx,RNGs,N,eta);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

