#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
#include "Simple2DActiveCell.h"
/*! \file Simple2DActiveCell.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DActiveCell::Simple2DActiveCell(bool _gpu, bool _neverGPU) : Simple2DCell(_gpu,_neverGPU)
    {
    Timestep=0;
    setDeltaT(0.01);
    if(neverGPU)
        {
        cellDirectors.noGPU=true;
        cellDirectorForces.noGPU=true;
        }
    };

/*!
Initialize the data structures to the size specified by n, and set default values, and call
Simple2DCell's initilization routine.
*/
void Simple2DActiveCell::initializeSimple2DActiveCell(int n)
    {
    Ncells = n;
    initializeSimple2DCell(Ncells);
    //The setting functions automatically resize their vectors
    setCellDirectorsRandomly();
    setv0Dr(0.0,1.0);
    };

/*!
Calls the spatial vertex sorting routine in Simple2DCell, and re-indexes the arrays for the cell
RNGS, as well as the cell motility and cellDirector arrays
*/
void Simple2DActiveCell::spatiallySortCellsAndCellActivity()
    {
    spatiallySortCells();
    reIndexCellArray(cellDirectors);
    };

/*!
Assign cell directors via a simple, reproducible RNG
*/
void Simple2DActiveCell::setCellDirectorsRandomly()
    {
    cellDirectors.resize(Ncells);
    cellDirectorForces.resize(Ncells);
    noise.Reproducible = Reproducible;
    ArrayHandle<double> h_cd(cellDirectors,access_location::host, access_mode::overwrite);
    ArrayHandle<double> h_cdf(cellDirectorForces,access_location::host, access_mode::overwrite);
    ArrayHandle<double2> h_v(cellVelocities);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_cd.data[ii] =noise.getRealUniform(0.0,2.0*PI);
        //h_cd.data[ii] = 0.0;
        h_v.data[ii].x = 0.0*cos(h_cd.data[ii]);
        h_v.data[ii].y = 0.0*sin(h_cd.data[ii]);
        };
    };

/*!
\param v0new the new value of velocity for all cells
\param drnew the new value of the rotational diffusion of cell directors for all cells
*/
void Simple2DActiveCell::setv0Dr(double v0new,double drnew)
    {
    v0=v0new;
    Dr=drnew;
    ArrayHandle<double> h_cd(cellDirectors,access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(cellVelocities);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double theta = h_cd.data[ii];
        h_v.data[ii].x = v0new*cos(theta);
        h_v.data[ii].y = v0new*sin(theta);
        };
    };
