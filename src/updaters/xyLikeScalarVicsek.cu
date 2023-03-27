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
                                               double eta,
                                               double deltaT)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;
    int m = nNeighbors[idx];
    double thetaUpdate = 0.0;
    double thetaI = atan2(velocities[idx].y,velocities[idx].x);
    double thetaJ=0.;
    for (int jj = 0; jj < m; ++jj)
        {
        int neighIdx = neighbors[n_idx(jj,idx)];
        thetaJ = atan2(velocities[neighIdx].y,velocities[neighIdx].x);
        thetaUpdate += deltaT*sin(thetaJ-thetaI);
        }

    if(reciprocalNormalization > 0) //reciprocal model: divide by a constant
        {
        thetaUpdate = thetaUpdate / reciprocalNormalization;
        }
    else // non-reciprocal model: divide by neighbor number
        {
        if (m > 0)
            thetaUpdate = thetaUpdate / ((double) m);
        }

    curandState_t randState;
    randState=RNGs[idx];//checkout the relevant RNG
    double u = curand_uniform_double(&randState) -0.5;//uniform between -.5 and .5
    RNGs[idx] = randState; //put it back
    thetaUpdate += sqrt(deltaT)*2.0*PI*eta*u;

    //normalize and rotate by noise
    newVelocities[idx] = velocities[idx];
    rotate2D(newVelocities[idx],thetaUpdate);
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
                    double eta,
                    double deltaT)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    xyLike_scalar_vicsek_directors_kernel<<<nblocks,block_size>>>(velocities,newVelocities,nNeighbors,neighbors,reciprocalNormalization,n_idx,RNGs,N,eta,deltaT);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

