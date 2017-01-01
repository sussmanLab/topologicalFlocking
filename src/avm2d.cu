#define NVCC
#define ENABLE_CUDA

#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "avm2d.cuh"


//!initialize each thread with a different sequence of the same seed of a cudaRNG
__global__ void initialize_curand_kernel(curandState *state, int N,int Timestep,int GlobalSeed)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=N)
        return;

    curand_init(GlobalSeed,idx,Timestep,&state[idx]);
    return;
    };

//!compute the voronoi vertices for each cell, along with its area and perimeter
__global__ void avm_geometry_kernel(const Dscalar2* __restrict__ d_p,
                                    const Dscalar2* __restrict__ d_v,
                                    const      int* __restrict__ d_nn,
                                    const      int* __restrict__ d_n,
                                    const      int* __restrict__ d_vcn,
                                          Dscalar2*  d_vc,
                                          Dscalar4*  d_vln,
                                          Dscalar2* __restrict__ d_AP,
                                          int N,
                                          Index2D n_idx,
                                          gpubox Box
                                        )
    {
    // read in the cell index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int neighs = d_nn[idx];
    Dscalar2 cellPos = d_p[idx];
    Dscalar2 vlast, vcur,vnext;
    Dscalar Varea = 0.0;
    Dscalar Vperi = 0.0;

    int vidx = d_n[n_idx(neighs-2,idx)];
    Box.minDist(d_v[vidx],cellPos,vlast);
    vidx = d_n[n_idx(neighs-1,idx)];
    Box.minDist(d_v[vidx],cellPos,vcur);
    for (int nn = 0; nn < neighs; ++nn)
        {
        //for easy force calculation, save the current, last, and next voronoi vertex position in the approprate spot.
        int forceSetIdx = -1;
        for (int ff = 0; ff < 3; ++ff)
            {
           if(d_vcn[3*vidx+ff]==idx)
                forceSetIdx = 3*vidx+ff;
            };

        vidx = d_n[n_idx(nn,idx)];
        Box.minDist(d_v[vidx],cellPos,vnext);
        
        //compute area contribution
        Varea += TriangleArea(vcur,vnext);
        Dscalar dx = vcur.x-vnext.x;
        Dscalar dy = vcur.y-vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        //save voronoi positions in a convenient form
        d_vc[forceSetIdx] = vcur;
        d_vln[forceSetIdx] = make_Dscalar4(vlast.x,vlast.y,vnext.x,vnext.y);
        //advance the loop
        vlast = vcur;
        vcur = vnext;
        };
    d_AP[idx].x=Varea;
    d_AP[idx].y=Vperi;
    };







//!Call the kernel to initialize a different RNG for each particle
bool gpu_initialize_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    initialize_curand_kernel<<<nblocks,block_size>>>(states,N,Timestep,GlobalSeed);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

//!Call the kernel to calculate the area and perimeter of each cell
bool gpu_avm_geometry(
                    Dscalar2 *d_p,
                    Dscalar2 *d_v,
                    int      *d_nn,
                    int      *d_n,
                    int      *d_vcn,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_AP,
                    int      N, 
                    Index2D  &n_idx, 
                    gpubox   &Box)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;


    avm_geometry_kernel<<<nblocks,block_size>>>(d_p,d_v,d_nn,d_n,d_vcn,d_vc,d_vln,d_AP,N, n_idx, Box);
    //cudaThreadSynchronize();
    return cudaSuccess;
    };

