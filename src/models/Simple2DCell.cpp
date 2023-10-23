#include "Simple2DCell.h"
#include "Simple2DCell.cuh"
/*! \file Simple2DCell.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
*/
Simple2DCell::Simple2DCell(bool _gpu, bool _neverGPU) :
    Ncells(0), Nvertices(0),Energy(-1.0)
    {
    GPUcompute = _gpu;
    neverGPU = _neverGPU;
    forcesUpToDate = false;
    Box = make_shared<periodicBoundaries>();
    if(neverGPU)
        {
        cellPositions.noGPU=true;
        cellVelocities.noGPU=true;
        AreaPeri.noGPU = true;
        neighborNum.noGPU=true;
        neighbors.noGPU=true;
        cellVertexNum.noGPU=true;
        cellForces.noGPU=true;
        cellType.noGPU=true;
        cellMasses.noGPU=true;
        cellVertices.noGPU=true;
        voroCur.noGPU=true;
        voroLastNext.noGPU=true;
        displacements.noGPU=true;
        Reproducible=false;
        }
    };

/*!
Initialize the data structures to the size specified by n, and set default values.
*/
void Simple2DCell::initializeSimple2DCell(int n)
    {
    Ncells = n;
    Nvertices = 2*Ncells;

    //setting cell positions randomly also auto-generates a square box with L = sqrt(Ncells)
    setCellPositionsRandomly();
    cellForces.resize(Ncells);
    AreaPeri.resize(Ncells);
    cellMasses.resize(Ncells);
    setCellTypeUniform(0);
    cellVelocities.resize(Ncells);
    vector<double> tempMass(Ncells,1.0);
    fillGPUArrayWithVector(tempMass,cellMasses);
    vector<double2> velocities(Ncells,make_double2(0.0,0.0));
    fillGPUArrayWithVector(velocities,cellVelocities);
    fillGPUArrayWithVector(velocities,cellForces);
    };

/*!
Simply call either the CPU or GPU routine in the current or derived model
*/
void Simple2DCell::computeGeometry()
    {
    if(GPUcompute)
        computeGeometryGPU();
    else
        computeGeometryCPU();
    }

/*!
Resize the box so that every cell has, on average, area = 1, and place cells via either a simple,
reproducible RNG or a non-reproducible RNG
*/
void Simple2DCell::setCellPositionsRandomly()
    {
    cellPositions.resize(Ncells);
    double boxsize = sqrt((double)Ncells);
    Box->setSquare(boxsize,boxsize);
    noise.Reproducible = Reproducible;

    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double x = noise.getRealUniform(0.0,boxsize);
        double y = noise.getRealUniform(0.0,boxsize);
        h_p.data[ii].x = x;
        h_p.data[ii].y = y;
        };
    };

/*!
Does not update any other lists -- it is the user's responsibility to maintain topology, etc, when
using this function.
*/
void Simple2DCell::setCellPositions(vector<double2> newCellPositions)
    {
    Ncells = newCellPositions.size();
    if(cellPositions.getNumElements() != Ncells) cellPositions.resize(Ncells);
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        h_p.data[ii] = newCellPositions[ii];
    }

/*!
 * set all cell types to i
 */
void Simple2DCell::setCellTypeUniform(int i)
    {
    cellType.resize(Ncells);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ct.data[ii] = i;
        };
    };

/*!
Set the cell velocities by drawing from a Maxwell-Boltzmann distribution, and then make sure there is no
net momentum. The return value is the total kinetic energy
 */
double Simple2DCell::setCellVelocitiesMaxwellBoltzmann(double T)
    {
    noise.Reproducible = Reproducible;
    double cellMass = 1.;
    ArrayHandle<double2> h_v(cellVelocities,access_location::host,access_mode::overwrite);
    double2 P = make_double2(0.0,0.0);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        double2 vi;
        vi.x = noise.getRealNormal(0.0,sqrt(T/cellMass));
        vi.y = noise.getRealNormal(0.0,sqrt(T/cellMass));
        h_v.data[ii] = vi;
        P = P+cellMass*vi;
        };
    //remove excess momentum, calculate the KE
    double KE = 0.0;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_v.data[ii] = h_v.data[ii] + (-1.0/(Ncells*cellMass))*P;
        KE += 0.5*cellMass*dot(h_v.data[ii],h_v.data[ii]);
        }
    return KE;
    };

/*!
 \param types a vector of integers that the cell types will be set to
 */
void Simple2DCell::setCellType(vector<int> &types)
    {
    cellType.resize(Ncells);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ct.data[ii] = types[ii];
        };
    };

/*!
 *Sets the size of itt, tti, idxToTag, and tagToIdx, and sets all of them so that
 array[i] = i,
 i.e., unsorted
 \pre Ncells is determined
 */
void Simple2DCell::initializeCellSorting()
    {
    itt.resize(Ncells);
    tti.resize(Ncells);
    idxToTag.resize(Ncells);
    tagToIdx.resize(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        itt[ii]=ii;
        tti[ii]=ii;
        idxToTag[ii]=ii;
        tagToIdx[ii]=ii;
        };
    };

/*!
 * Always called after spatial sorting is performed, reIndexCellArray shuffles the order of an array
    based on the spatial sort order of the cells
*/
void Simple2DCell::reIndexCellArray(GPUArray<double2> &array)
    {
    GPUArray<double2> TEMP = array;
    ArrayHandle<double2> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double2> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
Re-indexes GPUarrays of doubles
*/
void Simple2DCell::reIndexCellArray(GPUArray<double> &array)
    {
    GPUArray<double> TEMP = array;
    ArrayHandle<double> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<double> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
Re-indexes GPUarrays of ints
*/
void Simple2DCell::reIndexCellArray(GPUArray<int> &array)
    {
    GPUArray<int> TEMP = array;
    ArrayHandle<int> temp(TEMP,access_location::host,access_mode::read);
    ArrayHandle<int> ar(array,access_location::host,access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        ar.data[ii] = temp.data[itt[ii]];
        };
    };

/*!
 * take the current location of the cells and sort them according the their order along a 2D Hilbert curve
 */
void Simple2DCell::spatiallySortCells()
    {
    //itt and tti are the changes that happen in the current sort
    //idxToTag and tagToIdx relate the current indexes to the original ones
    HilbertSorter hs(*(Box));

    vector<pair<int,int> > idxCellSorter(Ncells);

    //sort points by Hilbert Curve location
    ArrayHandle<double2> h_p(cellPositions,access_location::host, access_mode::readwrite);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxCellSorter[ii].first=hs.getIdx(h_p.data[ii]);
        idxCellSorter[ii].second = ii;
        };
    sort(idxCellSorter.begin(),idxCellSorter.end());

    //update tti and itt
    for (int ii = 0; ii < Ncells; ++ii)
        {
        int newidx = idxCellSorter[ii].second;
        itt[ii] = newidx;
        tti[newidx] = ii;
        };

    //update points, idxToTag, and tagToIdx
    vector<int> tempi = idxToTag;
    for (int ii = 0; ii < Ncells; ++ii)
        {
        idxToTag[ii] = tempi[itt[ii]];
        tagToIdx[tempi[itt[ii]]] = ii;
        };
    reIndexCellArray(cellPositions);
    reIndexCellArray(cellType);
    reIndexCellArray(cellVelocities);
    };


/*!
P_ab = \sum m_i v_{ib}v_{ia}
*/
double4 Simple2DCell::computeKineticPressure()
    {
    int Ndof = getNumberOfDegreesOfFreedom();
    double4 ans; ans.x = 0.0; ans.y=0.0;ans.z=0;ans.w=0.0;
    ArrayHandle<double2> h_v(returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        {
        double  m = 1;
        double2 v = h_v.data[ii];
        ans.x += m*v.x*v.x;
        ans.y += m*v.y*v.x;
        ans.z += m*v.x*v.y;
        ans.w += m*v.y*v.y;
        };
    double b1,b2,b3,b4;
    Box->getBoxDims(b1,b2,b3,b4);
    double area = b1*b4;
    ans.x = ans.x / area;
    ans.y = ans.y / area;
    ans.z = ans.z / area;
    ans.w = ans.w / area;
    return ans;
    };

/*!
E = \sum 0.5*m_i v_i^2
*/
double Simple2DCell::computeKineticEnergy()
    {
    int Ndof = getNumberOfDegreesOfFreedom();
    double ans = 0.0;
    ArrayHandle<double2> h_v(returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        {
        double  m = 1.;
        double2 v = h_v.data[ii];
        ans += 0.5*m*(v.x*v.x+v.y*v.y);
        };
    return ans;
    };

/*!
a utility/testing function...output the currently computed mean net force to screen.
\param verbose if true also print out the force on each cell
*/
void Simple2DCell::reportMeanCellForce(bool verbose)
    {
    ArrayHandle<double2> h_f(cellForces,access_location::host,access_mode::read);
    ArrayHandle<double2> p(cellPositions,access_location::host,access_mode::read);
    double fx = 0.0;
    double fy = 0.0;
    double min = 10000;
    double max = -10000;
    for (int i = 0; i < Ncells; ++i)
        {
        if (h_f.data[i].y >max)
            max = h_f.data[i].y;
        if (h_f.data[i].x >max)
            max = h_f.data[i].x;
        if (h_f.data[i].y < min)
            min = h_f.data[i].y;
        if (h_f.data[i].x < min)
            min = h_f.data[i].x;
        fx += h_f.data[i].x;
        fy += h_f.data[i].y;

        if(verbose)
            printf("cell %i: \t position (%f,%f)\t force (%e, %e)\n",i,p.data[i].x,p.data[i].y ,h_f.data[i].x,h_f.data[i].y);
        };
    if(verbose)
        printf("min/max force : (%f,%f)\n",min,max);
    printf("Mean force = (%e,%e)\n" ,fx/Ncells,fy/Ncells);
    };
