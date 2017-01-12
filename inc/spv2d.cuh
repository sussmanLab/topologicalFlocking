#ifndef __SPV2D_CUH__
#define __SPV2D_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "gpubox.h"

/*!
 \file spv.cuh
A file providing an interface to the relevant cuda calls for the SPV2D class
*/

/** @defgroup spvKernels SPV Kernels
 * @{
 * \brief CUDA kernels and callers for the SPV2D class
 */

//!Initialize the GPU's random number generator
bool gpu_init_curand(curandState *states,
                    int N,
                    int Timestep,
                    int GlobalSeed
                    );

//!Given the positions, forces, and cell directors, move the particles and add noise to the director direction
bool gpu_displace_and_rotate(
                    Dscalar2 *d_points,
                    Dscalar2 *d_force,
                    Dscalar  *directors,
                    Dscalar2 *d_motility,
                    int N,
                    Dscalar dt,
                    int seed,
                    curandState *states,
                    gpubox &Box
                    );

//!compute the area and perimeter of all Voronoi cells, and save the voronoi vertices
bool gpu_compute_geometry(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    int    *d_nn,
                    int    *d_n,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    int    N,
                    Index2D &n_idx,
                    gpubox &Box
                    );

//!Compute the contribution to the net force on vertex i from each of i's voronoi vertices
bool gpu_force_sets(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2    *d_nidx,
                    Dscalar  KA,
                    Dscalar  KP,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    );
//!compute the contributions to the net force if, additionally, there are added tensions between cells of different types
bool gpu_force_sets_tensions(
                    Dscalar2 *d_points,
                    Dscalar2 *d_AP,
                    Dscalar2 *d_APpref,
                    int2   *d_delSets,
                    int    *d_detOther,
                    Dscalar2 *d_vc,
                    Dscalar4 *d_vln,
                    Dscalar2 *d_forceSets,
                    int2   *d_nidx,
                    int    *d_cellTypes,
                    Dscalar  KA,
                    Dscalar  KP,
                    Dscalar  gamma,
                    int    NeighIdxNum,
                    Index2D &n_idx,
                    gpubox &Box
                    );

//!Add up the force contributions to get the net force on each particle
bool gpu_sum_force_sets(
                    Dscalar2 *d_forceSets,
                    Dscalar2 *d_forces,
                    int    *d_nn,
                    int     N,
                    Index2D &n_idx
                    );

//!Add up the force constributions, but in the condidtion where some exclusions exist
bool gpu_sum_force_sets_with_exclusions(
                    Dscalar2 *d_forceSets,
                    Dscalar2 *d_forces,
                    Dscalar2 *d_external_forces,
                    int    *d_exes,
                    int    *d_nn,
                    int     N,
                    Index2D &n_idx
                    );

/** @} */ //end of group declaration

#endif

