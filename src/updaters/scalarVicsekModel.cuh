#ifndef __scalarVicsek_CUH__
#define __scalarVicsek_CUH__

#include "std_include.h"
#include "indexer.h"
#include <cuda_runtime.h>

/*!
 \file scalarVicsek.cuh
A file providing an interface to the relevant cuda calls for the
scalarVicsekModel class
*/

/** @addtogroup simpleEquationOfMotionKernels simpleEquationsOfMotion Kernels
 * @{
 */

//!update the directors based on neighbor relations
bool gpu_scalar_vicsek_directors(
                    double2 *velocities,
                    double2 *newVelocities,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double eta);


//!set the vector of displacements from forces and activity
bool gpu_scalar_vicsek_update(
                    double2 *forces,
                    double2 *velocities,
                    double2 *displacements,
                    double2 *motility,
                    int N,
                    double deltaT,
                    double mu);

/** @} */ //end of group declaration
 #endif
