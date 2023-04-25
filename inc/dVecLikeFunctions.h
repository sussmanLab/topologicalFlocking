#ifndef newFUNCTIONS_H
#define newFUNCTIONS_H

#include "std_include.h"
#include "gpuarray.h"
#include <set>

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file functions.h */

/** @defgroup Functions functions
 * @{
 \brief Utility functions that can be called from host or device
 */


//!The dot product between d-Dimensional vectors.
HOSTDEVICE double dot(const dVec &p1, const dVec &p2)
    {
    double ans = 0.0;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans+=p1.x[dd]*p2.x[dd];

    return ans;
    };

//! an integer to the dth power... the slow way
HOSTDEVICE int idPow(int i)
    {
    int ans = i;
    for (int dd = 1; dd < DIMENSION; ++dd)
        ans *= i;
    return ans;
    };

//!The dot product between d-Dimensional iVecs.
HOSTDEVICE int dot(const iVec &p1, const iVec &p2)
    {
    int ans = 0;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans+=p1.x[dd]*p2.x[dd];

    return ans;
    };

//!The norm of a d-Dimensional vector
HOSTDEVICE double norm(const dVec &p)
    {
    return sqrt(dot(p,p));
    };

//!fit integers into non-negative domains
HOSTDEVICE int wrap(int x,int m)
    {
    int ans = x;
    if(x >= m)
        ans = x % m;
    while(ans <0)
        ans += m;
    return ans;
    }


/** @} */ //end of group declaration
#undef HOSTDEVICE
#endif
