#include "cuda_runtime.h"
#include "voronoiModelBase.h"
#include "voronoiModelBase.cuh"

/*! \file voronoiModelBase.cpp */

/*!
A simple constructor that sets many of the class member variables to zero
*/
voronoiModelBase::voronoiModelBase(bool _gpu, bool _neverGPU) : Simple2DActiveCell(_gpu,_neverGPU),
    cellsize(1.25), timestep(0),repPerFrame(0.0),skippedFrames(0),
    neighMax(0),neighMaxChange(false),GlobalFixes(0),globalOnly(true)
    {
    //set cellsize to about unity...magic number should be of order 1
    //when the box area is of order N (i.e. on average one particle per bin)
    if(neverGPU)
        {
        anyCircumcenterTestFailed.noGPU=true;
        repair.noGPU=true;
        }
    };

void voronoiModelBase::reinitialize(int neighborGuess)
    {
    neighbors.resize(Ncells*neighborGuess);
    delGPU.initialize(Ncells,neighborGuess,1.0,Box,GPUcompute,neverGPU);
    globalTriangulationDelGPU();
    }
/*!
 * a function that takes care of the initialization of the class.
 * \param n the number of cells to initialize
 */
void voronoiModelBase::initializeVoronoiModelBase(int n, int maxNeighGuess)
    {
    //set particle number and call initializers
    Ncells = n;
    initializeSimple2DActiveCell(Ncells);

    repair.resize(Ncells);
    displacements.resize(Ncells);

    vector<int> baseEx(Ncells,0);

    //initialize spatial sorting, but do not sort by default
    initializeCellSorting();

    //DelaunayGPU initialization
    initialNeighborNumberGuess = maxNeighGuess;
    delGPU.initialize(Ncells,maxNeighGuess,1.0,Box,GPUcompute,neverGPU);
    delGPU.setSafetyMode(true);
    delGPU.setGPUcompute(GPUcompute);

    //make a full triangulation
    completeRetriangulationPerformed = 0;
    neighborNum.resize(Ncells);
    neighbors.resize(Ncells*maxNeighGuess);
    globalTriangulationDelGPU();
    resizeAndReset();
    enforceTopology();

    //initialize the anyCircumcenterTestFailed structure
    anyCircumcenterTestFailed.resize(1);
    ArrayHandle<int> h_actf(anyCircumcenterTestFailed,access_location::host,access_mode::overwrite);
    h_actf.data[0]=0;
    };

/*!
This function allows one to set rectangular unit cells, but because of the current implementation of the cellList
one **must** choose a rectangular domain in which the cell list's grid size perfectly divides *both* the x and y
directions of the unit cell (i.e.: (length of box in x direction)/((cell list grid size)*(number of cells in x-direction)) = 1, and same for y-direction).
*/
void voronoiModelBase::alterBox(PeriodicBoxPtr _box)
    {
    //WRITE GPU ROUTINE LATER
    vector<double2> newPos(Ncells);
    {
    ArrayHandle<double2> p(cellPositions);
    //find list of current virtual positions
    for(int ii = 0; ii < Ncells; ++ii)
        Box->invTrans(p.data[ii],newPos[ii]);
    for(int ii = 0; ii < Ncells; ++ii)
        _box->Trans(newPos[ii],p.data[ii]);
    }//end array scope
    Box=_box;
    reinitialize(neighMax);
    enforceTopology();
    computeGeometry();
    };

/*!
\post the cell list is updated according to the current cell positions
*/
void voronoiModelBase::updateCellList()
    {
    delGPU.updateList(cellPositions);
    };

/*!
Displace cells on the CPU
\param displacements a vector of double2 specifying how much to move every cell
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void voronoiModelBase::movePointsCPU(GPUArray<double2> &displacements,double scale)
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
void voronoiModelBase::movePoints(GPUArray<double2> &displacements,double scale)
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
    };

/*!
Displace cells on either the GPU or CPU, according to the flag
\param displacements a vector of double2 specifying how much to move every cell
\param scale a scalar that multiples the value of every index of displacements before things are moved
\post the cells are displaced according to the input vector, and then put back in the main unit cell.
*/
void voronoiModelBase::moveDegreesOfFreedom(GPUArray<double2> &displacements,double scale)
    {
    forcesUpToDate = false;
    if (GPUcompute)
        movePoints(displacements,scale);
    else
        movePointsCPU(displacements,scale);
    };

 int voronoiModelBase::getNeighIdxNum()
    {
    if(GPUcompute)
        {
        ArrayHandle<int> neighnum(neighborNum,access_location::device,access_mode::read);
        gpu_update_neighIdxs(neighnum.data,NeighIdxNum,Ncells);
        }
    else
        {
        ArrayHandle<int> neighnum(neighborNum,access_location::host,access_mode::read);
        NeighIdxNum = 0;
        for (int ii = 0; ii < Ncells; ++ii)
             NeighIdxNum+=neighnum.data[ii];
        }
    return NeighIdxNum;
    }

/*!
Call the delaunayGPU class to get a complete triangulation of the current point set.e
*/
void voronoiModelBase::globalTriangulationDelGPU(bool verbose)
    {
    GlobalFixes +=1;
    completeRetriangulationPerformed += 1;
    int oldNeighMax = delGPU.MaxSize;
    if(neighbors.getNumElements() != Ncells*oldNeighMax)
        resizeAndReset();

    delGPU.globalDelaunayTriangulation(cellPositions,neighbors,neighborNum);

    neighMax = delGPU.MaxSize;
    n_idx = Index2D(neighMax,Ncells);
    if(neighbors.getNumElements() != Ncells*neighMax)
        neighbors.resize( Ncells*neighMax);
    if(oldNeighMax != neighMax)
        {
        resizeAndReset();
        }
    //global rescue if needed
    NeighIdxNum = getNeighIdxNum();
    if(NeighIdxNum != 6* Ncells)
        {
        cout << "attempting CGAL rescue -- inconsistent local topologies" << endl;
        globalTriangulationCGAL();
        resizeAndReset();
        }
    }

/*!
This function calls the DelaunayCGAL class to determine the Delaunay triangulation of the entire
square periodic domain this method is, obviously, better than the version written by DMS, so
should be the default option. 
*/
void voronoiModelBase::globalTriangulationCGAL(bool verbose)
    {
    GlobalFixes +=1;
    completeRetriangulationPerformed = 1;
    DelaunayCGAL dcgal;
    ArrayHandle<double2> h_points(cellPositions,access_location::host, access_mode::read);
    vector<pair<Point,int> > Psnew(Ncells);
    for (int ii = 0; ii < Ncells; ++ii)
        {
        Psnew[ii]=make_pair(Point(h_points.data[ii].x,h_points.data[ii].y),ii);
        };
    double b1,b2,b3,b4;
    Box->getBoxDims(b1,b2,b3,b4);
    dcgal.PeriodicTriangulation(Psnew,b1,b2,b3,b4);

    ArrayHandle<int> neighnum(neighborNum,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_repair(repair,access_location::host,access_mode::overwrite);

    int oldNmax = neighMax;
    int totaln = 0;
    int nmax = 0;
    for(int nn = 0; nn < Ncells; ++nn)
        {
        neighnum.data[nn] = dcgal.allneighs[nn].size();
        totaln += dcgal.allneighs[nn].size();
        if (dcgal.allneighs[nn].size() > nmax) nmax= dcgal.allneighs[nn].size();
        h_repair.data[nn]=0;
        };
    if (nmax%2 == 0)
        neighMax = nmax+2;
    else
        neighMax = nmax+1;

    n_idx = Index2D(neighMax,Ncells);
    if(neighbors.getNumElements() != Ncells*neighMax)
        {
        neighbors.resize(neighMax*Ncells);
        neighMaxChange = true;
        };

    //store data in gpuarrays
    {
    ArrayHandle<int> ns(neighbors,access_location::host,access_mode::overwrite);

    for (int nn = 0; nn < Ncells; ++nn)
        {
        int imax = neighnum.data[nn];
        for (int ii = 0; ii < imax; ++ii)
            {
            int idxpos = n_idx(ii,nn);
            ns.data[idxpos] = dcgal.allneighs[nn][ii];
            };
        };

    if(verbose)
        cout << "global new Nmax = " << neighMax << "; total neighbors = " << totaln << endl;cout.flush();
    };

    if(totaln != 6*Ncells)
        {
        printf("global CPU neighbor failed! NN = %i\n",totaln);
        char fn[256];
        sprintf(fn,"failed.txt");
        ofstream output(fn);
        writeTriangulation(output);
        throw std::exception();
        };
    populateVoroCur();
    };

void voronoiModelBase::populateVoroCur()
    {
    if(delGPU.GPUVoroCur.getNumElements() != neighMax*Ncells)
        delGPU.GPUVoroCur.resize(neighMax*Ncells);

    //read in all the data we'll need
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);

    ArrayHandle<double2> h_v(delGPU.GPUVoroCur,access_location::host,access_mode::readwrite);

    for (int i = 0; i < Ncells; ++i)
        {
        //get Delaunay neighbors of the cell
        int neigh = h_nn.data[i];
        vector<int> ns(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn]=h_n.data[n_idx(nn,i)];
            };

        //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
        vector<double2> voro(neigh);
        double2 circumcent;
        double2 nnextp,nlastp;
        double2 pi = h_p.data[i];
        double2 rij, rik;

        nlastp = h_p.data[ns[ns.size()-1]];
        Box->minDist(nlastp,pi,rij);
        for (int nn = 0; nn < neigh;++nn)
            {
            nnextp = h_p.data[ns[nn]];
            Box->minDist(nnextp,pi,rik);
            Circumcenter(rij,rik,circumcent);
            voro[nn] = circumcent;
            rij=rik;
            int id = n_idx(nn,i);
            h_v.data[id] = voro[nn];
            };
        };
    };

/*!
When sortPeriod < 0, this routine does not get called
\post call Simple2DActiveCell's underlying Hilbert sort scheme, and re-index voronoiModelBase's extra arrays
*/
void voronoiModelBase::spatialSorting()
    {
    spatiallySortCellsAndCellActivity();
    //reTriangulate with the new ordering
    globalTriangulationDelGPU();
    //get new DelSets and DelOthers
    resetLists();
    };

/*!
goes through the process of testing and repairing the topology on either the CPU or GPU
\post and topological changes needed by cell motion are detected and repaired
*/
void voronoiModelBase::enforceTopology()
    {
    int oldNeighMax = delGPU.MaxSize;
    if(neighbors.getNumElements() != Ncells*oldNeighMax)
        resizeAndReset();


    delGPU.testAndRepairDelaunayTriangulation(cellPositions,neighbors,neighborNum);
//    globalTriangulationDelGPU();

    //global rescue if needed
    if(NeighIdxNum != 6* Ncells)
        {
        cout << "attempting CGAL rescue -- inconsistent local topologies" << endl;
        globalTriangulationCGAL();
        resizeAndReset();
        }

    neighMax = delGPU.MaxSize;
    if(oldNeighMax != neighMax)
        {
        resizeAndReset();
        globalTriangulationDelGPU();
        }
    };

//read a triangulation from a text file...used only for testing purposes. Any other use should call the Database class (see inc/Database.h")
void voronoiModelBase::readTriangulation(ifstream &infile)
    {
    string line;
    getline(infile,line);
    stringstream convert(line);
    int nn;
    convert >> nn;
    cout << "Reading in " << nn << "points" << endl;
    int idx = 0;
    int ii = 0;
    ArrayHandle<double2> p(cellPositions,access_location::host,access_mode::overwrite);
    while(getline(infile,line))
        {
        double val = stof(line);
        if (idx == 0)
            {
            p.data[ii].x=val;
            idx +=1;
            }
        else
            {
            p.data[ii].y=val;
            Box->putInBoxReal(p.data[ii]);
            idx = 0;
            ii += 1;
            };
        };
    };

//similarly, write a text file with particle positions. This is often called when an exception is thrown
void voronoiModelBase::writeTriangulation(ofstream &outfile)
    {
    ArrayHandle<double2> p(cellPositions,access_location::host,access_mode::read);
    outfile << Ncells <<endl;
    for (int ii = 0; ii < Ncells ; ++ii)
        outfile << p.data[ii].x <<"\t" <<p.data[ii].y <<endl;
    };

/*!
\pre Topology is up-to-date on the CPU
\post geometry and voronoi neighbor locations are computed for the current configuration
*/
void voronoiModelBase::computeGeometryCPU()
    {
    //read in all the data we'll need
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);

    ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::readwrite);
    ArrayHandle<double4> h_vln(voroLastNext,access_location::host,access_mode::overwrite);

    /*
    gpu_compute_voronoi_geometry(h_p.data,h_AP.data,h_nn.data,h_n.data,
                                 h_v.data,h_vln.data,Ncells,n_idx,*(Box),false,ompThreadNum);
    */
    for (int i = 0; i < Ncells; ++i)
        {
        //get Delaunay neighbors of the cell
        int neigh = h_nn.data[i];
        vector<int> ns(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn]=h_n.data[n_idx(nn,i)];
            };

        //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
        vector<double2> voro(neigh);
        double2 circumcent;
        double2 nnextp,nlastp;
        double2 pi = h_p.data[i];
        double2 rij, rik;

        nlastp = h_p.data[ns[ns.size()-1]];
        Box->minDist(nlastp,pi,rij);
        for (int nn = 0; nn < neigh;++nn)
            {
            nnextp = h_p.data[ns[nn]];
            Box->minDist(nnextp,pi,rik);
            Circumcenter(rij,rik,circumcent);
            voro[nn] = circumcent;
            rij=rik;
            int id = n_idx(nn,i);
            h_v.data[id] = voro[nn];
            };

        double2 vlast,vnext;
        //compute Area and perimeter, and fill in voroLastNext structure
        double Varea = 0.0;
        double Vperi = 0.0;
        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vnext=voro[nn];
            Varea += TriangleArea(vlast,vnext);
            double dx = vlast.x-vnext.x;
            double dy = vlast.y-vnext.y;
            Vperi += sqrt(dx*dx+dy*dy);
            int id = n_idx(nn,i);
            h_vln.data[id].x=vlast.x;
            h_vln.data[id].y=vlast.y;
            h_vln.data[id].z=vnext.x;
            h_vln.data[id].w=vnext.y;
            vlast=vnext;
            };
        h_AP.data[i].x = Varea;
        h_AP.data[i].y = Vperi;
        };
    };

/*!
\pre The topology of the Delaunay triangulation is up-to-date on the GPU
\post calculate all cell areas, perimenters, and voronoi neighbors
*/
void voronoiModelBase::computeGeometryGPU()
    {
    ArrayHandle<double2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<double2> d_AP(AreaPeri,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_nn(neighborNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(neighbors,access_location::device,access_mode::read);
    ArrayHandle<double2> d_vc(voroCur,access_location::device,access_mode::readwrite);
    ArrayHandle<double4> d_vln(voroLastNext,access_location::device,access_mode::overwrite);

    gpu_compute_voronoi_geometry(
                        d_p.data,
                        d_AP.data,
                        d_nn.data,
                        d_n.data,
                        d_vc.data,
                        d_vln.data,
                        Ncells, n_idx,*(Box));
    };

/*!
As the code is modified, all GPUArrays whose size depend on neighMax should be added to this function
\post voroCur,voroLastNext, and grow to size neighMax*Ncells
*/
void voronoiModelBase::resetLists()
    {
    n_idx = Index2D(neighMax,Ncells);
    neighbors.resize( Ncells*neighMax);
    voroCur.resize(neighMax*Ncells);
    voroLastNext.resize(neighMax*Ncells);
    };

/*!
A utility function for resizing data arrays... used by the cell division and cell death routines
*/
void voronoiModelBase::resizeAndReset()
    {
    //Simple resizing operations
    displacements.resize(Ncells);
    cellForces.resize(Ncells);
    repair.resize(Ncells);

    neighborNum.resize(Ncells);

    resetLists();
    reinitialize(initialNeighborNumberGuess);
    };
