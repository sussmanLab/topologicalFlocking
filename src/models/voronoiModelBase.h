#ifndef voronoiModelBase_H
#define voronoiModelBase_H

#include "Simple2DActiveCell.h"
#include "cellListGPU.cuh"
#include "cellListGPU.h"
#include "DelaunayCGAL.h"
#include "DelaunayGPU.h"
#include "structures.h"
#include "voronoiModelBase.cuh"


/*! \file voronoiModelBase.h */
//! Perform and test triangulations in an MD setting, using kernels in \ref voronoiModelBaseKernels
/*!
 * voronoiModelBase is a core engine class, capable of taking a set of points
 * in a periodic domain, performing Delaunay triangulations on them, testing whether
 * those triangulations are valid on either the CPU or GPU, and locally repair
 * invalid triangulations on the CPU.

 * Voronoi models have their topology taken care of by the underlying triangulation, and so child
   classes just need to implement an energy functions (and corresponding force law)
 */
class voronoiModelBase : public Simple2DActiveCell
    {
    public:
        //!The constructor!
        voronoiModelBase(bool _gpu = true, bool _neverGPU=false);
        //!A default initialization scheme
        void initializeVoronoiModelBase(int n, int maxNeighGuess = 16);
        void reinitialize(int neighborGuess);
        //!Enforce CPU-only operation.
        /*!
        \param global defaults to true.
        When global is set to true, the CPU branch will try the local repair scheme.
        This is generally slower, but if working in a regime where things change
        very infrequently, it may be faster.
        */
        void setCPU(bool global = true){GPUcompute = false;globalOnly=global;delGPU.setGPUcompute(false);};
        virtual void setGPU(){GPUcompute = true; delGPU.setGPUcompute(true);};
        //!write triangulation to text file
        void writeTriangulation(ofstream &outfile);
        //!read positions from text file...for debugging
        void readTriangulation(ifstream &infile);

        //!update/enforce the topology
        virtual void enforceTopology();
        //!set a new simulation box, update the positions of cells based on virtual positions, and then recompute the geometry
        void alterBox(PeriodicBoxPtr _box);
        //virtual functions that need to be implemented
        //!In voronoi models the number of degrees of freedom is the number of cells
        virtual int getNumberOfDegreesOfFreedom(){return Ncells;};

        //!moveDegrees of Freedom calls either the move points or move points CPU routines
        virtual void moveDegreesOfFreedom(GPUArray<double2> & displacements,double scale = 1.);
        //!return the forces
        virtual void getForces(GPUArray<double2> &forces){forces = cellForces;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<double2> & returnForces(){return cellForces;};

        //!Compute cell geometry on the CPU
        virtual void computeGeometryCPU();
        //!call gpu_compute_geometry kernel caller
        virtual void computeGeometryGPU();

        //!move particles on the GPU
        void movePoints(GPUArray<double2> &displacements,double scale);
        //!move particles on the CPU
        void movePointsCPU(GPUArray<double2> &displacements,double scale);

        //!Update the cell list structure after particles have moved
        void updateCellList();
        //set number of threads
        virtual void setOmpThreads(int _number){ompThreadNum = _number;delGPU.setOmpThreads(_number);};

    //protected functions
    protected:
        //!sort points along a Hilbert curve for data locality
        void spatialSorting();

        //!Globally construct the triangulation via CGAL
        void globalTriangulationCGAL(bool verbose = false);
        //!Globally construct the triangulation via CGAL
        void globalTriangulationDelGPU(bool verbose = false);

        //!repair any problems with the triangulation on the CPU
        void repairTriangulation(vector<int> &fixlist);
        //! after a CGAL triangulation, need to populate delGPU's voroCur structure in order for compute geometry to work
        void populateVoroCur();

        //!resize all neighMax-related arrays
        void resetLists();
        //!do resize and resetting operations common to cellDivision and cellDeath
        void resizeAndReset();

    //public member variables
    public:
        //!The class' local Delaunay tester/updater
        DelaunayGPU delGPU;

        //!Collect statistics of how many triangulation repairs are done per frame, etc.
        double repPerFrame;
        //!How often were all circumcenters empty (so that no data transfers and no repairs were necessary)?
        int skippedFrames;
        //!How often were global re-triangulations performed?
        int GlobalFixes;

    protected:
        //!The size of the cell list's underlying grid
        double cellsize;
        //!An upper bound for the maximum number of neighbors that any cell has
        int neighMax;

        //!A utility integer
        int NeighIdxNum;

        //!evaluate the total number of neighbors of each point
        int getNeighIdxNum();

        //!A flag that can be accessed by child classes... serves as notification that any change in the network topology has occured
        GPUArray<int> anyCircumcenterTestFailed;
        //!A flag that notifies that a global re-triangulation has been performed
        int completeRetriangulationPerformed;
        //!A flag that notifies that the maximum number of neighbors may have changed, necessitating resizing of some data arrays
        bool neighMaxChange;

        //!A a vector of zeros (everything is fine) and ones (that index needs to be repaired)
        GPUArray<int> repair;
        //!A smaller vector that, after testing the triangulation, contains the particle indices that need their local topology to be updated.
        vector<int> NeedsFixing;

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
    friend class SPVDatabaseNetCDF;
    friend class vicsekDatabase;
    };

#endif
