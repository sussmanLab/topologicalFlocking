#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
#include "Simple2DActiveCell.h"
/*! \file Simple2DActiveCell.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DActiveCell::Simple2DActiveCell(bool _gpu, bool _neverGPU) : Simpe2DCell(_gpu,_neverGPU)
    {
    Timestep=0;
    setDeltaT(0.01);
    if(neverGPU)
        {
        cellDirectors.noGPU=true;
        cellDirectorForces.noGPU=true;
        Motility.noGPU=true;
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
void Simple2DActiveCell::spatiallySortVerticesAndCellActivity()
    {
    spatiallySortVertices();
    reIndexCellArray(Motility);
    reIndexCellArray(cellDirectors);
    };

/*!
Calls the spatial vertex sorting routine in Simple2DCell, and re-indexes the arrays for the cell
RNGS, as well as the cell motility and cellDirector arrays
*/
void Simple2DActiveCell::spatiallySortCellsAndCellActivity()
    {
    spatiallySortCells();
    reIndexCellArray(Motility);
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
    Motility.resize(Ncells);
    v0=v0new;
    Dr=drnew;
    ArrayHandle<double> h_cd(cellDirectors,access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(cellVelocities);
    ArrayHandle<double2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_mot.data[ii].x = v0new;
        h_mot.data[ii].y = drnew;
        double theta = h_cd.data[ii];
        h_v.data[ii].x = v0new*cos(theta);
        h_v.data[ii].y = v0new*sin(theta);
        };
    };

/*!
\param v0s the per-particle vector of what all velocities will be
\param drs the per-particle vector of what all rotational diffusions will be
*/
void Simple2DActiveCell::setCellMotility(vector<double> &v0s,vector<double> &drs)
    {
    Motility.resize(Ncells);
    ArrayHandle<double> h_cd(cellDirectors,access_location::host, access_mode::read);
    ArrayHandle<double2> h_v(cellVelocities);
    ArrayHandle<double2> h_mot(Motility,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_mot.data[ii].x = v0s[ii];
        h_mot.data[ii].y = drs[ii];
        double theta = h_cd.data[ii];
        h_v.data[ii].x = v0s[ii]*cos(theta);
        h_v.data[ii].y = v0s[ii]*sin(theta);
        };
    };

/*!
This function supports cellDeath, updating the data structures in Simple2DActiveCell.
This function will first call Simple2DCell's routine, then modify the cellDirectors and Motility arrays
*/
void Simple2DActiveCell::cellDeath(int cellIndex)
    {
    Simple2DCell::cellDeath(cellIndex);
    removeGPUArrayElement(cellDirectors,cellIndex);
    removeGPUArrayElement(Motility,cellIndex);
    };

/*!
This function supports cellDivisions, updating data structures in Simple2DActiveCell
This function will first call Simple2DCell's routine, and then
grows the cellDirectors and Motility arrays, and assign the new cell
(the last element of those arrays) the values of the cell given by parameters[0]
Note that dParams does nothing
 */
void Simple2DActiveCell::cellDivision(const vector<int> &parameters, const vector<double> &dParams)
    {
    //The Simple2DCell routine will increment Ncells by one, and then update other data structures
    Simple2DCell::cellDivision(parameters);
    int cellIdx = parameters[0];
    growGPUArray(cellDirectors,1);
    growGPUArray(Motility,1);
    noise.Reproducible = Reproducible;
        {//arrayhandle scope
        ArrayHandle<double2> h_mot(Motility); h_mot.data[Ncells-1] = h_mot.data[cellIdx];
        ArrayHandle<double> h_cd(cellDirectors); h_cd.data[Ncells-1] = noise.getRealUniform(0.,2*PI);
        ArrayHandle<double2> h_v(cellVelocities);
        h_v.data[Ncells-1].x = h_mot.data[Ncells-1].x*cos(h_cd.data[Ncells-1]);
        h_v.data[Ncells-1].y = h_mot.data[Ncells-1].x*sin(h_cd.data[Ncells-1]);
        };
    };
