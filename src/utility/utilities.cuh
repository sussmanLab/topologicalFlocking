#ifndef utilities_CUH__
#define utilities_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "gpuarray.h"
/*!
 \file utilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//! (double) ans = (double2) vec1 . vec2
bool gpu_dot_double2_vectors(double2 *d_vec1,
                              double2 *d_vec2,
                              double  *d_ans,
                              int N);

//!A trivial reduction of an array by one thread in serial. Think before you use this.
bool gpu_serial_reduction(
                    double *array,
                    double *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm.
bool gpu_parallel_reduction(
                    double *input,
                    double *intermediate,
                    double *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm for double2 arrays.
bool gpu_parallel_reduction(
                    double2 *input,
                    double2 *intermediate,
                    double2 *output,
                    int helperIdx,
                    int N);

//! (double2) ans = (double2) vec1 * vec2
bool gpu_dot_double_double2_vectors(double *d_vec1,
                              double2 *d_vec2,
                              double2  *d_ans,
                              int N);

//!set every element of an array to the specified value
template<typename T>
bool gpu_set_array(T *arr,
                   T value,
                   int N,
                   int maxBlockSize=512);

//! answer = answer+adder
template<typename T>
bool gpu_add_gpuarray(GPUArray<T> &answer,
                       GPUArray<T> &adder,
                       int N,
                       int block_size=512);

//!copy data into target on the device...copies the first Ntotal elements into the target array, by default it copies all elements
template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,
                       GPUArray<T> &copyFrom,
                       int numberOfElementsToCopy = -1,
                       int block_size=512);
//!convenience function to zero out an array on the GPU
bool gpu_zero_array(double *arr,
                    int N
                    );

//!convenience function to zero out an array on the GPU
bool gpu_zero_array(cubicLatticeDerivativeVector *arr,
                    int N
                    );

//!convenience function to zero out an array on the GPU
bool gpu_zero_array(int *arr,
                    int N
                    );
//!convenience function to zero out an array on the GPU
bool gpu_zero_array(unsigned int *arr,
                    int      N
                    );
//!convenience function to zero out an array on the GPU
bool gpu_zero_array(dVec *arr,
                    int N
                    );


//! (double) ans = (dVec) vec1 . vec2
bool gpu_dot_dVec_vectors(dVec *d_vec1,
                              dVec *d_vec2,
                              double  *d_ans,
                              int N);

//! (dVec) input *= factor
bool gpu_dVec_times_double(dVec *d_vec1,
                              double factor,
                              int N);
//! (dVec) ans = input * factor
bool gpu_dVec_times_double(dVec *d_vec1,
                              double factor,
                              dVec *d_ans,
                              int N);
//! ans = a*b[i]*c[i]^2r
bool gpu_double_times_dVec_squared(dVec *d_vec1,
                                   double *d_doubles,
                                   double factor,
                                   double *d_answer,
                                   int N);

//!Take two vectors of dVecs and compute the sum of the dot products between them
bool gpu_dVec_dot_products(
                    dVec *input1,
                    dVec *input2,
                    double *intermediate,
                    double *output,
                    int helperIdx,
                    int N,
                    int block_size);
/** @} */ //end of group declaration

//!fill dVec from double2 array on either CPU or GPU
bool filldVecFromDouble2(GPUArray<dVec> &copyInto,
                       GPUArray<double2> &copyFrom,
                       int numberOfElementsToCopy,
                       bool useGPU,
                       int block_size=512);
#endif
