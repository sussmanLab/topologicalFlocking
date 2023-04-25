#ifndef DINDEXER_H
#define DINDEXER_H

#include "functions.h"
#include "dVecLikeFunctions.h"
#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif
//!Switch between a 3-dimensional grid to a flattened, 1D index
/*!
 * A class for converting between a 3d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class Index3D
    {
    public:
        HOSTDEVICE Index3D(unsigned int w=0){setSizes(w);};
        HOSTDEVICE Index3D(int3 w){setSizes(w);};

        HOSTDEVICE void setSizes(unsigned int w)
            {
            int3 W;
            W.x=w;W.y=w;W.z=w;
            setSizes(W);
            };
        HOSTDEVICE void setSizes(int3 w)
            {
            sizes.x = w.x;sizes.y = w.y;sizes.z = w.z;
            numberOfElements = sizes.x * sizes.y * sizes.z;
            intermediateSizes.x=1;
            intermediateSizes.y = intermediateSizes.x*sizes.x;
            intermediateSizes.z = intermediateSizes.y*sizes.y;
            };

        HOSTDEVICE unsigned int operator()(const int x, const int y, const int z) const
            {
            return x*intermediateSizes.x + y*intermediateSizes.y + z*intermediateSizes.z;
            };

        HOSTDEVICE unsigned int operator()(const int3 &i) const
            {
            return i.x*intermediateSizes.x + i.y*intermediateSizes.y + i.z*intermediateSizes.z;
            };

        //!What iVec would correspond to a given unsigned int IndexDD(iVec)
        HOSTDEVICE int3 inverseIndex(int i)
            {
            int3 ans;
            int z0 = i;
            ans.x = z0%sizes.x;
            z0= (z0-ans.x)/sizes.x;
            ans.y = z0%sizes.y;
            z0=(z0-ans.y)/sizes.y;
            ans.z = z0%sizes.z;
            return ans;
            };

        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return numberOfElements;
            };

        //!Get the iVec of sizes
        HOSTDEVICE int3 getSizes() const
            {
            return sizes;
            };

        int3 sizes; //!< a list of the size of the full array in each of the d dimensions
        int3 intermediateSizes; //!<intermediateSizes[a] = Product_{d<=a} sizes. intermediateSizes[0]=1;
        unsigned int numberOfElements; //! The total number of elements that the indexer can index
        unsigned int width;   //!< array width
    };

//!Switch between a d-dimensional grid to a flattened, 1D index
/*!
 * A class for converting between a 2d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class IndexDD
    {
    public:
        HOSTDEVICE IndexDD(unsigned int w=0){setSizes(w);};
        HOSTDEVICE IndexDD(iVec w){setSizes(w);};

        HOSTDEVICE void setSizes(unsigned int w)
            {
            sizes.x[0] = w;
            numberOfElements=sizes.x[0];
            intermediateSizes.x[0]=1;
            for (int dd = 1; dd < DIMENSION; ++dd)
                {
                sizes.x[dd] = w;
                intermediateSizes.x[dd] = intermediateSizes.x[dd-1]*sizes.x[dd];
                numberOfElements *= sizes.x[dd];
                };
            };
        HOSTDEVICE void setSizes(iVec w)
            {
            sizes.x[0] = w.x[0];
            numberOfElements=sizes.x[0];
            intermediateSizes.x[0]=1;
            for (int dd = 1; dd < DIMENSION; ++dd)
                {
                sizes.x[dd] = w.x[dd];
                intermediateSizes.x[dd] = intermediateSizes.x[dd-1]*sizes.x[dd];
                numberOfElements *= sizes.x[dd];
                };
            };

        HOSTDEVICE unsigned int operator()(const iVec &i) const
            {
            return dot(i,intermediateSizes);
            };

        //!What iVec would correspond to a given unsigned int IndexDD(iVec)
        HOSTDEVICE iVec inverseIndex(int i)
            {
            iVec ans;
            int z0 = i;
            for (int dd = 0; dd < DIMENSION; ++dd)
                {
                ans.x[dd] = z0 % sizes.x[dd];
                z0 = (z0 - ans.x[dd])/sizes.x[dd];
                };
            return ans;
            };

        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return numberOfElements;
            };

        //!Get the iVec of sizes
        HOSTDEVICE iVec getSizes() const
            {
            return sizes;
            };

        iVec sizes; //!< a list of the size of the full array in each of the d dimensions
        iVec intermediateSizes; //!<intermediateSizes[a] = Product_{d<=a} sizes. intermediateSizes[0]=1;
        unsigned int numberOfElements; //! The total number of elements that the indexer can index
        unsigned int width;   //!< array width
    };

#undef HOSTDEVICE
#endif

