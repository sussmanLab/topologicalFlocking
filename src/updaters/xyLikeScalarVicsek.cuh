#ifndef __xyLikeScalarVicsek_CUH__
#define __xyLikeScalarVicsek_CUH__

#include "std_include.h"
#include "indexer.h"
#include <cuda_runtime.h>

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
                    double deltaT);
 #endif
