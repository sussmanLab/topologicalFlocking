#ifndef neighborList_H
#define neighborList_H

#include "hyperrectangularCellList.h"
/*! \file neighborList.h */
//!take a set of positions, sort those positions according to a cellList, and create data structures of possible neighbors of each particle
class neighborList
    {
    public:
        neighborList(){};
        //!basic constructor has a box and a range
        neighborList(double range, BoxPtr _box, int subGridReduction = 1);

        //!computethe neighborlist of the set of points passed in
        void computeNeighborLists(GPUArray<dVec> &points)
            {
            if(useGPU)
                {
                computeGPU(points);
                }
            else
                {
                computeCPU(points);
                }
            };

        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU=true){
            useGPU = _useGPU;
            cellList->setGPU(useGPU);
            };
        //!whether the updater does its work on the GPU or not
        bool useGPU;
        //!The Box used to compute periodic distances
        BoxPtr Box;

        void setBox(BoxPtr _bx){Box=_bx;};

        //!indexes the neighbors of each particle
        Index2D neighborIndexer;

        //! An array containing the number of elements in each neighborhood
        GPUArray<int> neighborsPerParticle;
        //!An array containing the indices of neighbors of each particle. So, particleIndices[neighborIndexer(nn,pp)] gives the index of the nth particle in the neighborhood of particle pp
        GPUArray<int> particleIndices;
        //!An array saving the displacement data associated with each neighbor pair. distances[neighborIndexer(nn,pp)]
        GPUArray<dVec> neighborVectors;
        //!An array saving the distance data associated with each neighbor pair. distances[neighborIndexer(nn,pp)]
        GPUArray<double> neighborDistances;

        //!An internal counter
        int computations;

        //!maximum range that neighbors need to be kept at
        double maxRange;
        //!The cell list that will help out
        shared_ptr<hyperrectangularCellList> cellList;
        //! the maximum number of particles found in any neighborhood
        int Nmax;

        //!Save the displacement and distances associated with neihgbors?
        bool saveDistanceData;
        //!Initialization and helper without using the GPU
        void resetNeighborsCPU(int size, int _nmax);
        //!Initialization and helper
        void resetNeighborsGPU(int size,int _nmax);

        //! compute via GPU
        void computeGPU(GPUArray<dVec> &points);
        //! compute via CPU
        void computeCPU(GPUArray<dVec> &points);
    protected:
        //!first index is Nmax, second is whether to recompute
        GPUArray<int> assist;
    };

#endif
