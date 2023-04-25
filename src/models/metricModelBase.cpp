#include "cuda_runtime.h"
#include "metricModelBase.h"

/*! \file metricModelBase.cpp */

/*!
A simple constructor that sets many of the class member variables to zero
*/
metricModelBase::metricModelBase(bool _gpu, bool _neverGPU) : Simple2DActiveCell(_gpu,_neverGPU),
    cellsize(1.), timestep(0),
    neighMax(0),globalOnly(true)
    {
    //set cellsize to about unity...magic number should be of order 1
    //when the box area is of order N (i.e. on average one particle per bin)
    };

/*!
 * a function that takes care of the initialization of the class.
 * \param n the number of cells to initialize
 */
void metricModelBase::initializemetricModelBase(int n)
    {
    //set particle number and call initializers
    Ncells = n;
    initializeSimple2DActiveCell(Ncells);

    displacements.resize(Ncells);

    vector<int> baseEx(Ncells,0);

    //initialize spatial sorting, but do not sort by default
    initializeCellSorting();


    //set neighbor lists
    updateNeighborList();

    //make a full triangulation
    //neighborNum.resize(Ncells);
    //neighbors.resize(Ncells*maxNeighGuess);
    };

/*!
Displace cells on the CPU
\param displacements a vector of double2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void metricModelBase::movePointsCPU(GPUArray<double2> &displacements,double scale)
    {
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::readwrite);
    ArrayHandle<double2> h_d(displacements,access_location::host,access_mode::read);
    if(scale == 1.)
        {
        for (int idx = 0; idx < Ncells; ++idx)
            {
            h_p.data[idx].x += h_d.data[idx].x;
            h_p.data[idx].y += h_d.data[idx].y;
            Box->putInBoxReal(h_p.data[idx]);
            };
        }
    else
        {
        for (int idx = 0; idx < Ncells; ++idx)
            {
            h_p.data[idx].x += scale*h_d.data[idx].x;
            h_p.data[idx].y += scale*h_d.data[idx].y;
            Box->putInBoxReal(h_p.data[idx]);
            };
        }
    };

/*!
Displace cells on the GPU
\param displacements a vector of double2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void metricModelBase::movePoints(GPUArray<double2> &displacements,double scale)
    {
        {
        ArrayHandle<double2> d_p(cellPositions,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_d(displacements,access_location::device,access_mode::readwrite);
        if (scale == 1.)
            gpu_move_degrees_of_freedom(d_p.data,d_d.data,Ncells,*(Box));
        else
            gpu_move_degrees_of_freedom(d_p.data,d_d.data,scale,Ncells,*(Box));

        cudaError_t code = cudaGetLastError();
        if(code!=cudaSuccess)
            {
            printf("movePoints GPUassert: %s \n", cudaGetErrorString(code));
            throw std::exception();
            };
        }
    };

/*!
Displace cells on either the GPU or CPU, according to the flag
\param displacements a vector of double2 specifying how much to move every cell
\param scale a scalar that multiples the value of every index of displacements before things are moved
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void metricModelBase::moveDegreesOfFreedom(GPUArray<double2> &displacements,double scale)
    {
    forcesUpToDate = false;
    if (GPUcompute)
        movePoints(displacements,scale);
    else
        movePointsCPU(displacements,scale);
    updateNeighborList();
    };

void metricModelBase::updateNeighborList()
    {
    };

/*!
When sortPeriod < 0, this routine does not get called
\post call Simple2DActiveCell's underlying Hilbert sort scheme, and re-index metricModelBase's extra arrays
*/
void metricModelBase::spatialSorting()
    {
    spatiallySortCellsAndCellActivity();
    resetLists();
    };

/*!
As the code is modified, all GPUArrays whose size depend on neighMax should be added to this function
\post voroCur,voroLastNext, and grow to size neighMax*Ncells
*/
void metricModelBase::resetLists()
    {
    n_idx = Index2D(neighMax,Ncells);
    neighbors.resize( Ncells*neighMax);
    };

