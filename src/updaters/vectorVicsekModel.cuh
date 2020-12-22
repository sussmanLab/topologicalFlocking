#ifndef __vectorVicsek_CUH__
#define __vectorVicsek_CUH__

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
bool gpu_vector_vicsek_directors(
                    double2 *velocities,
                    double2 *newVelocities,
                    int *nNeighbors,
                    int *neighbors,
                    Index2D  &n_idx,
                    curandState *RNGs,
                    int N,
                    double eta);

/** @} */ //end of group declaration
 #endif
