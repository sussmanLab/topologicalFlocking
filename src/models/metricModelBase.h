#ifndef metricModelBase_H
#define metricModelBase_H

#include "Simple2DActiveCell.h"
#include "structures.h"
#include "neighborList.h"



/*! \file metricModelBase.h */
//! Perform and test triangulations in an MD setting, using kernels in \ref metricModelBaseKernels
/*!
 */
class metricModelBase : public Simple2DActiveCell
    {
    public:
        //!The constructor!
        metricModelBase(bool _gpu = true, bool _neverGPU=false);
        //!A default initialization scheme
        void initializeMetricModelBase(int n);
        //!Enforce CPU-only operation.
        /*!
        \param global defaults to true.
        When global is set to true, the CPU branch will try the local repair scheme.
        This is generally slower, but if working in a regime where things change
        very infrequently, it may be faster.
        */
        void setCPU(bool global = true){GPUcompute = false;};
        virtual void setGPU(){GPUcompute = true;};

        //virtual functions that need to be implemented
        //!In metric models the number of degrees of freedom is the number of cells
        virtual int getNumberOfDegreesOfFreedom(){return Ncells;};

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<double2> & displacements,double scale = 1.);
        //!return the forces
        virtual void getForces(GPUArray<double2> &forces){forces = cellForces;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<double2> & returnForces(){return cellForces;};

        //!move particles on the GPU
        void movePoints(GPUArray<double2> &displacements,double scale);
        //!move particles on the CPU
        void movePointsCPU(GPUArray<double2> &displacements,double scale);

        //!Update the neighbbor list structure after particles have moved
        void updateNeighborList();
        //set number of threads
        virtual void setOmpThreads(int _number){ompThreadNum = _number;};

        shared_ptr<neighborList> neighStructure;
        shared_ptr<periodicBoundaryConditions> PBC;

    //protected functions
    protected:
        //!sort points along a Hilbert curve for data locality
        void spatialSorting();

        //!resize all neighMax-related arrays
        void resetLists();
        //!do resize and resetting operations common to cellDivision and cellDeath
        void resizeAndReset();

    protected:
        GPUArray<dVec> dVecPos;
        //!The size of the cell list's underlying grid
        double cellsize;
        //!An upper bound for the maximum number of neighbors that any cell has
        int neighMax;

        //!When true, the CPU branch will execute global retriangulations through CGAL on every time step
        /*!When running on the CPU, should only global retriangulations be performed,
        or should local test-and-updates still be performed? Depending on parameters
        simulated, performance here can be quite difference, since the circumcircle test
        itself is CPU expensive
        */
        bool globalOnly;
        //!Count the number of times that testAndRepair has been called, separately from the derived class' time
        int timestep;

    //be friends with the associated Database class so it can access data to store or read
    friend class vicsekDatabase;
    };

#endif
