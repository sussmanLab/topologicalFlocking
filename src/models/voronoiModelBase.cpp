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
        external_forces.noGPU=true;
        exclusions.noGPU=true;
        NeighIdxs.noGPU=true;
        anyCircumcenterTestFailed.noGPU=true;
        repair.noGPU=true;
        delSets.noGPU=true;
        delOther.noGPU=true;
        forceSets.noGPU=true;
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

    NeighIdxs.resize(6*(Ncells));

    repair.resize(Ncells);
    displacements.resize(Ncells);

    external_forces.resize(Ncells);
    vector<int> baseEx(Ncells,0);
    setExclusions(baseEx);
    particleExclusions=false;

    //initialize spatial sorting, but do not sort by default
    initializeCellSorting();

    //DelaunayGPU initialization
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
\param exes a list of per-particle indications of whether a particle should be excluded (exes[i] !=0) or not/
*/
void voronoiModelBase::setExclusions(vector<int> &exes)
    {
    particleExclusions=true;
    external_forces.resize(Ncells);
    exclusions.resize(Ncells);
    ArrayHandle<double2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ex(exclusions,access_location::host,access_mode::overwrite);

    for (int ii = 0; ii < Ncells; ++ii)
        {
        h_ex.data[ii] = 0;
        if( exes[ii] != 0)
            {
            //set v0 to zero and Dr to zero
            h_mot.data[ii].x = 0.0;
            h_mot.data[ii].y = 0.0;
            h_ex.data[ii] = 1;
            };
        };
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

/*!
Call the delaunayGPU class to get a complete triangulation of the current point set. Afterwards, call updateNeighIdxs
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
    updateNeighIdxs();
    //global rescue if needed
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
should be the default option. In addition to performing a triangulation, the function also automatically
calls updateNeighIdxs
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
    updateNeighIdxs();

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
\post the NeighIdx data structure is updated, which helps cut down on the number of inactive
threads in the force set computation function
*/
void voronoiModelBase::updateNeighIdxs()
    {
    if (GPUcompute)
        {
        ArrayHandle<int> neighnum(neighborNum,access_location::device,access_mode::read);
        ArrayHandle<int> neighNumScan(repair,access_location::device,access_mode::overwrite);
        ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::overwrite);
        gpu_update_neighIdxs(neighnum.data, neighNumScan.data,d_nidx.data,NeighIdxNum,Ncells);
        }
    else
        {
        ArrayHandle<int> neighnum(neighborNum,access_location::host,access_mode::read);
        ArrayHandle<int2> h_nidx(NeighIdxs,access_location::host,access_mode::overwrite);
        int idx = 0;
        for (int ii = 0; ii < Ncells; ++ii)
            {
            int nmax = neighnum.data[ii];
            for (int nn = 0; nn < nmax; ++nn)
                {
                h_nidx.data[idx].x = ii;
                h_nidx.data[idx].y = nn;
                idx+=1;
                };
            };
        NeighIdxNum = idx;
        }
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
    allDelSets();

    //re-index all cell information arrays
    reIndexCellArray(exclusions);
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

    allDelSets();
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
\param ri The position of cell i
\param rj The position of cell j
\param rk The position of cell k
Returns the derivative of the voronoi vertex shared by cells i, j , and k with respect to changing the position of cell i
the (row, column) format specifies dH_{row}/dr_{i,column}
*/
Matrix2x2 voronoiModelBase::dHdri(double2 ri, double2 rj, double2 rk)
    {
    Matrix2x2 Id;
    double2 rij, rik, rjk;
    Box->minDist(rj,ri,rij);
    Box->minDist(rk,ri,rik);
    rjk.x =rik.x-rij.x;
    rjk.y =rik.y-rij.y;

    double2 dbDdri,dgDdri,dDdriOD,z;
    double betaD = -dot(rik,rik)*dot(rij,rjk);
    double gammaD = dot(rij,rij)*dot(rik,rjk);
    double cp = rij.x*rjk.y - rij.y*rjk.x;
    double D = 2*cp*cp;
    z.x = betaD*rij.x+gammaD*rik.x;
    z.y = betaD*rij.y+gammaD*rik.y;

    dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
    dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

    dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
    dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

    dDdriOD.x = (-2.0*rjk.y)/cp;
    dDdriOD.y = (2.0*rjk.x)/cp;

    return Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));
    };

/*!
\param i The index of cell i
\param j The index of cell j
\pre Requires that computeGeometry is current
Returns the derivative of the area of cell i w/r/t the position of cell j
*/
double2 voronoiModelBase::dAidrj(int i, int j)
    {
    double2 answer;
    answer.x = 0.0; answer.y=0.0;
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::readwrite);

    //how many neighbors does cell i have?
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    bool jIsANeighbor = false;
    if (j ==i) jIsANeighbor = true;

    //which two vertices are important?
    int n1, n2;
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn] = h_n.data[n_idx(nn,i)];
        if (ns[nn] ==j)
            {
            jIsANeighbor = true;
            n1 = nn;
            n2 = nn+1;
            if (n2 ==neigh) n2 = 0;
            }
        };
    double2 vlast, vcur,vnext;
    //if j is not a neighbor of i (or i itself!) the  derivative vanishes
    if (!jIsANeighbor)
        return answer;
    //if i ==j, do the loop simply
    if ( i == j)
        {
        vlast = h_v.data[n_idx(neigh-1,i)];
        for (int vv = 0; vv < neigh; ++vv)
            {
            vcur = h_v.data[n_idx(vv,i)];
            vnext = h_v.data[n_idx((vv+1)%neigh,i)];
            double2 dAdv;
            dAdv.x = -0.5*(vlast.y-vnext.y);
            dAdv.y = -0.5*(vnext.x-vlast.x);

            int indexk = vv - 1;
            if (indexk <0) indexk = neigh-1;
            double2 temp = dAdv*dHdri(h_p.data[i],h_p.data[ h_n.data[n_idx(vv,i)] ],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x += temp.x;
            answer.y += temp.y;
            vlast = vcur;
            };
        return answer;
        };

    //otherwise, the interesting case
    vlast = h_v.data[n_idx(neigh-1,i)];
    for (int vv = 0; vv < neigh; ++vv)
        {
        vcur = h_v.data[n_idx(vv,i)];
        vnext = h_v.data[n_idx((vv+1)%neigh,i)];
        if(vv == n1 || vv == n2)
            {
            int indexk;
            if (vv == n1)
                indexk=vv-1;
            else
                indexk=vv;

            if (indexk <0) indexk = neigh-1;
            double2 dAdv;
            dAdv.x = -0.5*(vlast.y-vnext.y);
            dAdv.y = -0.5*(vnext.x-vlast.x);
            double2 temp = dAdv*dHdri(h_p.data[j],h_p.data[i],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x += temp.x;
            answer.y += temp.y;
            };
        vlast = vcur;
        };
    return answer;
    }

/*!
\param i The index of cell i
\param j The index of cell j
Returns the derivative of the perimeter of cell i w/r/t the position of cell j
*/
double2 voronoiModelBase::dPidrj(int i, int j)
    {
    double Pthreshold = THRESHOLD;
    double2 answer;
    answer.x = 0.0; answer.y=0.0;
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(voroCur,access_location::host,access_mode::readwrite);

    //how many neighbors does cell i have?
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    bool jIsANeighbor = false;
    if (j ==i) jIsANeighbor = true;

    //which two vertices are important?
    int n1, n2;
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn] = h_n.data[n_idx(nn,i)];
        if (ns[nn] ==j)
            {
            jIsANeighbor = true;
            n1 = nn;
            n2 = nn+1;
            if (n2 ==neigh) n2 = 0;
            }
        };
    double2 vlast, vcur,vnext;
    //if j is not a neighbor of i (or i itself!) the  derivative vanishes
    if (!jIsANeighbor)
        return answer;
    //if i ==j, do the loop simply
    if ( i == j)
        {
        vlast = h_v.data[n_idx(neigh-1,i)];
        for (int vv = 0; vv < neigh; ++vv)
            {
            vcur = h_v.data[n_idx(vv,i)];
            vnext = h_v.data[n_idx((vv+1)%neigh,i)];
            double2 dPdv;
            double2 dlast,dnext;
            dlast.x = vlast.x-vcur.x;
            dlast.y=vlast.y-vcur.y;

            double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

            dnext.x = vcur.x-vnext.x;
            dnext.y = vcur.y-vnext.y;
            double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            if(dnnorm < Pthreshold)
                dnnorm = Pthreshold;
            if(dlnorm < Pthreshold)
                dlnorm = Pthreshold;
            dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
            dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

            int indexk = vv - 1;
            if (indexk <0) indexk = neigh-1;
            double2 temp = dPdv*dHdri(h_p.data[i],h_p.data[ h_n.data[n_idx(vv,i)] ],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x -= temp.x;
            answer.y -= temp.y;
            vlast = vcur;
            };
        return answer;
        };

    //otherwise, the interesting case
    vlast = h_v.data[n_idx(neigh-1,i)];
    for (int vv = 0; vv < neigh; ++vv)
        {
        vcur = h_v.data[n_idx(vv,i)];
        vnext = h_v.data[n_idx((vv+1)%neigh,i)];
        if(vv == n1 || vv == n2)
            {
            int indexk;
            if (vv == n1)
                indexk=vv-1;
            else
                indexk=vv;

            if (indexk <0) indexk = neigh-1;
            double2 dPdv;
            double2 dlast,dnext;
            dlast.x = vlast.x-vcur.x;
            dlast.y=vlast.y-vcur.y;

            double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

            dnext.x = vcur.x-vnext.x;
            dnext.y = vcur.y-vnext.y;
            double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            if(dnnorm < Pthreshold)
                dnnorm = Pthreshold;
            if(dlnorm < Pthreshold)
                dlnorm = Pthreshold;
            dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
            dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
            double2 temp = dPdv*dHdri(h_p.data[j],h_p.data[i],h_p.data[ h_n.data[n_idx(indexk,i)] ]);
            answer.x -= temp.x;
            answer.y -= temp.y;
            };
        vlast = vcur;
        };
    return answer;
    };

/*!
\param dvpdr derivative of v_{i+1} w/r/t a cell position
\param dvmdr derivative of v_{i-1} w/r/t a cell position
*/
Matrix2x2 voronoiModelBase::d2Areadvdr(Matrix2x2 &dvpdr, Matrix2x2 &dvmdr)
    {
    Matrix2x2 d2Advidrj;
    d2Advidrj.x11 =(-dvmdr.x21 + dvpdr.x21);
    d2Advidrj.x12 =(dvmdr.x11 - dvpdr.x11);
    d2Advidrj.x21 =(-dvmdr.x22 + dvpdr.x22);
    d2Advidrj.x22 =(dvmdr.x12 - dvpdr.x12);
    return 0.5*d2Advidrj;
    };

/*!
\param dvdr derivative of v_{i} w/r/t a cell position
\param dvpdr derivative of v_{i+1} w/r/t a cell position
\param dvmdr derivative of v_{i-1} w/r/t a cell position
\param vm position of v_{i-1}
\param v position of v_{i}
\param vp position of v_{i+1}
*/
Matrix2x2 voronoiModelBase::d2Peridvdr(Matrix2x2 &dvdr, Matrix2x2 &dvmdr,Matrix2x2 &dvpdr, double2 vm, double2 v, double2 vp)
    {
    double2 dlast = v-vm;
    double2 dnext = vp-v;
    double dlastNorm = norm(dlast);
    double dnextNorm = norm(dnext);
    double denNext = 1.0/(dnextNorm*dnextNorm*dnextNorm);
    double denLast = 1.0/(dlastNorm*dlastNorm*dlastNorm);

    //dP/dv = dnext/dnextNorm - dlast/dlastNorm; we'll differentiate each of those terms separately
    Matrix2x2 dNdr, dLdr;
    dNdr.x11 = (-v.y + vp.y)* ((-dvdr.x21 + dvpdr.x21)* (-v.x + vp.x) - (-dvdr.x11 + dvpdr.x11)* (-v.y + vp.y));
    dNdr.x12 = -(-v.x + vp.x)* ((-dvdr.x21 + dvpdr.x21)* (-v.x + vp.x) - (-dvdr.x11 + dvpdr.x11)* (-v.y + vp.y));
    dNdr.x21 = (-v.y + vp.y)* ((-dvdr.x22 + dvpdr.x22)* (-v.x + vp.x) - (-dvdr.x12 + dvpdr.x12)* (-v.y + vp.y));
    dNdr.x22 = -(-v.x + vp.x)* ((-dvdr.x22 + dvpdr.x22)* (-v.x + vp.x) - (-dvdr.x12 + dvpdr.x12)* (-v.y + vp.y));

    dLdr.x11 = (-v.y + vm.y)*(-(-dvdr.x21 + dvmdr.x21)*(-v.x + vm.x) + (-dvdr.x11 + dvmdr.x11)*(-v.y + vm.y));
    dLdr.x12 = (-v.x + vm.x)*((-dvdr.x21 + dvmdr.x21)*(-v.x + vm.x) - (-dvdr.x11 + dvmdr.x11)*(-v.y + vm.y));
    dLdr.x21 = (-v.y + vm.y)*(-(-dvdr.x22 + dvmdr.x22)*(-v.x + vm.x) + (-dvdr.x12 + dvmdr.x12)*(-v.y + vm.y));
    dLdr.x22 = (-v.x + vm.x)*((-dvdr.x22 + dvmdr.x22)*(-v.x + vm.x) - (-dvdr.x12 + dvmdr.x12)*(-v.y + vm.y));

    return denNext*dNdr - denLast*dLdr;
    };

/*!
\param ri The position of cell i
\param rj The position of cell j
\param rk The position of cell k
\param jj the index EITHER 1 or 2 of the second derivative
Returns an 8-component vector containing the derivatives of the voronoi vertex formed by cells i, j, and k
with respect to r_i and r_{jj}... jj should be either 1 (to give d^2H/(d r_i)^2 or 2 (to give d^2H/dridrj)
The vector is laid out as
(H_x/r_{i,x}r_{j,x}, H_y/r_{i,x}r_{j,x}
H_x/r_{i,y}r_{j,x}, H_y/r_{i,y}r_{j,x
H_x/r_{i,x}r_{j,y}, H_y/r_{i,x}r_{j,y
H_x/r_{i,y}r_{j,y}, H_y/r_{i,y}r_{j,y}  )
NOTE: This function does not check that ri, rj, and rk actually share a voronoi vertex in the triangulation
NOTE: This function assumes that rj and rk are w/r/t the position of ri, so ri = (0.,0.)
*/
vector<double> voronoiModelBase::d2Hdridrj(double2 rj, double2 rk, int jj)
    {
    vector<double> answer(8);
    double hxr1xr2x, hyr1xr2x, hxr1yr2x,hyr1yr2x;
    double hxr1xr2y, hyr1xr2y, hxr1yr2y,hyr1yr2y;
    double rjx,rjy,rkx,rky;
    rjx = rj.x; rjy=rj.y; rkx=rk.x;rky=rk.y;

    double denominator;
    denominator = (rjx*rky-rjy*rkx)*(rjx*rky-rjy*rkx)*(rjx*rky-rjy*rkx);
    hxr1xr2x = hyr1xr2x = hxr1yr2x = hyr1yr2x = hxr1xr2y= hyr1xr2y= hxr1yr2y=hyr1yr2y= (1.0/denominator);
    //all derivatives in the dynMatTesting notebook
    //first, handle the d^2h/dr_i^2 case
    if ( jj == 1)
        {
        hxr1xr2x *= rjy*(rjy - rky)*rky*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky);
        hyr1xr2x *= -(rjy*(rjx - rkx)*rky*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky));
        hxr1yr2x *= -((rjy - rky)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hyr1yr2x *= ((rjx - rkx)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hxr1xr2y *= -((rjy - rky)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) +rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hyr1xr2y *= ((rjx - rkx)*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) + rjx*(rky*(rjy*rjy) - 2*rjy*(rkx*rkx + rky*rky) + rky*(rkx*rkx + rky*rky))))/2.;
        hxr1yr2y *= rjx*rkx*(rjy - rky)*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky);
        hyr1yr2y *= -(rjx*(rjx - rkx)*rkx*(-2*rjx*rkx - 2*rjy*rky + rjx*rjx + rjy*rjy + rkx*rkx + rky*rky));
        }
    else
        {
        //next, handle the d^2h/dr_idr_j case
        hxr1xr2x *= rjy*(rjy - rky)*rky*(rjx*rkx + (rjy - rky)*rky - rkx*rkx);
        hyr1xr2x *= (-(rjy*rkx*rky*(-3*rjx*rkx + 3*(rjx*rjx) + rjy*rjy + 2*(rkx*rkx))) + rjy*rjy*(rkx*rkx*rkx) +(rjx*rjx*rjx - rjx*(rjy*rjy) + 3*rkx*(rjy*rjy))*(rky*rky) + rjy*(rjx - 2*rkx)*(rky*rky*rky))/2.;
        hxr1yr2x *= -((rjy - rky)*(rjy*rkx*((2*rjy - rky)*rky - rkx*rkx) + rjx*(2*rjy*(rkx*rkx) - rky*(rkx*rkx + rky*rky))))/2.;
        hyr1yr2x *= (rkx*(3*rjy*rkx*(rjx*rjx) - rky*(rjx*rjx*rjx) + rjy*rkx*(-2*rjy*rky + rjy*rjy + rkx*rkx + rky*rky) +  rjx*(rky*(rjy*rjy) + rky*(rkx*rkx + rky*rky) - 2*rjy*(2*(rkx*rkx) + rky*rky))))/2.;
        hxr1xr2y *= -(rky*(rkx*(rjy - 2*rky)*(rjx*rjx) + rky*(rjx*rjx*rjx) + rjy*rkx*(-(rjy*rjy) + rkx*rkx + rky*rky) +rjx*(3*rky*(rjy*rjy) + rky*(rkx*rkx + rky*rky) - 2*rjy*(rkx*rkx + 2*(rky*rky)))))/2.;
        hyr1xr2y *= ((rjx - rkx)*(rjx*rky*(2*rjx*rkx - rkx*rkx - rky*rky) - rjy*(rkx*rkx*rkx - 2*rjx*(rky*rky) + rkx*(rky*rky))))/2.;
        hxr1yr2y *= (rkx*rky*(rjx*rjx*rjx) - rjy*rjy*rjy*(rkx*rkx) + rjx*rjx*(rjy*(rkx*rkx) - rky*(3*(rkx*rkx) + rky*rky)) +rjx*rkx*(3*rky*(rjy*rjy) + 2*rky*(rkx*rkx + rky*rky) - rjy*(rkx*rkx + 3*(rky*rky))))/2.;
        hyr1yr2y *= -(rjx*(rjx - rkx)*rkx*(rjx*rkx + (rjy - rky)*rky - rkx*rkx));
        };

    answer[0] = hxr1xr2x;
    answer[1] = hyr1xr2x;
    answer[2] = hxr1yr2x;
    answer[3] = hyr1yr2x;
    answer[4] = hxr1xr2y;
    answer[5] = hyr1xr2y;
    answer[6] = hxr1yr2y;
    answer[7] = hyr1yr2y;
    return answer;
    };

/*!
As the code is modified, all GPUArrays whose size depend on neighMax should be added to this function
\post voroCur,voroLastNext, delSets, delOther, and forceSets grow to size neighMax*Ncells
*/
void voronoiModelBase::resetLists()
    {
    n_idx = Index2D(neighMax,Ncells);
    neighbors.resize( Ncells*neighMax);
    voroCur.resize(neighMax*Ncells);
    voroLastNext.resize(neighMax*Ncells);
    delSets.resize(neighMax*Ncells);
    delOther.resize(neighMax*Ncells);
    forceSets.resize(neighMax*Ncells);
    };

/*!
\param i the cell in question
\post the delSet and delOther data structure for cell i is updated. Recall that
delSet.data[n_idx(nn,i)] is an int2; the x and y parts store the index of the previous and next
Delaunay neighbor, ordered CCW. delOther contains the mutual neighbor of delSet.data[n_idx(nn,i)].y
and delSet.data[n_idx(nn,i)].z that isn't cell i
*/
bool voronoiModelBase::getDelSets(int i)
    {
    ArrayHandle<int> neighnum(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> ns(neighbors,access_location::host,access_mode::read);
    ArrayHandle<int2> ds(delSets,access_location::host,access_mode::readwrite);
    ArrayHandle<int> dother(delOther,access_location::host,access_mode::readwrite);

    int iNeighs = neighnum.data[i];
    int nm2,nm1,n1,n2;
    nm2 = ns.data[n_idx(iNeighs-3,i)];
    nm1 = ns.data[n_idx(iNeighs-2,i)];
    n1 = ns.data[n_idx(iNeighs-1,i)];

    for (int nn = 0; nn < iNeighs; ++nn)
        {
        n2 = ns.data[n_idx(nn,i)];
        int nextNeighs = neighnum.data[n1];
        for (int nn2 = 0; nn2 < nextNeighs; ++nn2)
            {
            int testPoint = ns.data[n_idx(nn2,n1)];
            if(testPoint == nm1)
                {
                dother.data[n_idx(nn,i)] = ns.data[n_idx((nn2+1)%nextNeighs,n1)];
                break;
                };
            };
        ds.data[n_idx(nn,i)].x= nm1;
        ds.data[n_idx(nn,i)].y= n1;

        //is "delOther" a copy of i or either of the delSet points? if so, the local topology is inconsistent
        if(nm1 == dother.data[n_idx(nn,i)] || n1 == dother.data[n_idx(nn,i)] || i == dother.data[n_idx(nn,i)])
            return false;

        nm2=nm1;
        nm1=n1;
        n1=n2;

        };
    return true;
    };


/*!
Calls updateNeighIdxs and then getDelSets(i) for all cells i
*/
void voronoiModelBase::allDelSets()
    {
    updateNeighIdxs();
    if(GPUcompute)
        {
        ArrayHandle<int> neighnum(neighborNum,access_location::device,access_mode::read);
        ArrayHandle<int> ns(neighbors,access_location::device,access_mode::read);
        ArrayHandle<int2> ds(delSets,access_location::device,access_mode::readwrite);
        ArrayHandle<int> dother(delOther,access_location::device,access_mode::readwrite);
        gpu_all_del_sets(neighnum.data,ns.data,ds.data,dother.data,Ncells,n_idx);
        }
    else
        {
        for (int ii = 0; ii < Ncells; ++ii)
            getDelSets(ii);
        };
    };

/*!
A utility function for resizing data arrays... used by the cell division and cell death routines
*/
void voronoiModelBase::resizeAndReset()
    {
    //Simple resizing operations
    displacements.resize(Ncells);
    cellForces.resize(Ncells);
    external_forces.resize(Ncells);
    exclusions.resize(Ncells);
    repair.resize(Ncells);

    neighborNum.resize(Ncells);
    NeighIdxs.resize(6*(Ncells));

    resetLists();
    allDelSets();
    reinitialize(neighMax);
    };

/*!
Trigger a cell death event. In the Voronoi model this simply removes the targeted cell and instantaneously computes the new tesselation. Very violent, if the cell isn't already small.
*/
void voronoiModelBase::cellDeath(int cellIndex)
    {
    //first, call the parent class routines.
    //This call already changes Ncells
    Simple2DActiveCell::cellDeath(cellIndex);
    resizeAndReset();
    };

/*!
Trigger a cell division event, which involves some laborious re-indexing of various data structures.
This version uses a heavy-handed approach, hitting the cell positions with a full, global retriangulation
(rather than just updating the targeted cell and its neighbor with the local topology repair routines).
If a simulation is needed where the cell division rate is very rapid, this should be improved.
The idea of the division is that a targeted cell will divide normal to an axis specified by the
angle, theta, passed to the function. The final state cell positions are placed along the axis at a
distance away from the initial cell position set by a multiplicative factor (<1) of the in-routine determined
maximum distance in the cell along that axis.
parameters[0] = the index of the cell to undergo a division event
dParams[0] = an angle, theta
dParams[1] = a fraction of maximum separation of the new cell positions along the axis of the cell specified by theta

This function is meant to be called before the start of a new timestep. It should be immediately followed by a computeGeometry call.
\post the new cell is the final indexed entry of the various data structures (e.g.,
cellPositions[new number of cells - 1])
*/
void voronoiModelBase::cellDivision(const vector<int> &parameters, const vector<double> &dParams)
    {
    int cellIdx = parameters[0];
    double theta = dParams[0];
    double separationFraction = dParams[1];

    //First let's get the geometry of the cell in a convenient reference frame
    //computeGeometry has not yet been called, so need to find the voro positions
    vector<double2> voro;
    voro.reserve(10);
    int neigh;
    double2 initialCellPosition;
    {//arrayHandle scope
    ArrayHandle<double2> h_p(cellPositions,access_location::host,access_mode::read);
    initialCellPosition = h_p.data[cellIdx];
    ArrayHandle<int> h_nn(neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(neighbors,access_location::host,access_mode::read);
    neigh = h_nn.data[cellIdx];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
            ns[nn]=h_n.data[n_idx(nn,cellIdx)];
    double2 circumcent;
    double2 nnextp,nlastp;
    double2 pi = h_p.data[cellIdx];
    double2 rij, rik;
    nlastp = h_p.data[ns[ns.size()-1]];
    Box->minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        nnextp = h_p.data[ns[nn]];
        Box->minDist(nnextp,pi,rik);
        Circumcenter(rij,rik,circumcent);
        voro.push_back(circumcent);
        rij=rik;
        }
    };//arrayHandle scope

    //find where the line emanating from the polygons intersects the edges
    double2 c,v1,v2,Int1,Int2;
    bool firstIntFound = false;
    v1 = voro[neigh-1];
    double2 ray; ray.x = Cos(theta); ray.y = Sin(theta);
    c.x = - Sin(theta);
    c.y = Cos(theta);
    double2 p; p.x =0.0; p.y=0.;
    for (int nn = 0; nn < neigh; ++nn)
        {
        v2=voro[nn];
        double2 a; a.x = p.x-v2.x; a.y = p.y-v2.y;
        double2 b; b.x = v1.x - v2.x; b.y = v1.y-v2.y;
        double t2 = -1.;
        if (dot(b,c) != 0)
            t2 = dot(a,c)/dot(b,c);
        if (t2 >= 0. && t2 <= 1.0)
            {
            if (firstIntFound)
                {
                Int2 = v2+t2*(v1-v2);
                }
            else
                {
                Int1 = v2+t2*(v1-v2);
                firstIntFound = true;
                };
            };

        v1=v2;
        };
    double maxSeparation = max(norm(p-Int1),norm(p-Int2));
    double2 newCellPos1 = initialCellPosition + separationFraction*maxSeparation*ray;
    double2 newCellPos2 = initialCellPosition - separationFraction*maxSeparation*ray;
    Box->putInBoxReal(newCellPos1);
    Box->putInBoxReal(newCellPos2);

    //This call updates many of the base data structres, but (among other things) does not actually
    //set the new cell position
    Simple2DActiveCell::cellDivision(parameters);
    {
    ArrayHandle<double2> cp(cellPositions);
    cp.data[cellIdx] = newCellPos1;
    cp.data[Ncells-1] = newCellPos2;
    }
    resizeAndReset();
    };
