#include "vicsekDatabase.h"

vicsekDatabase::vicsekDatabase(int np, string fn, NcFile::FileMode mode)
    : BaseDatabaseNetCDF(fn,mode)
    {
    N=np;
    dof = 2*N;
    val=0.0;
    vec.resize(dof);
    switch(mode)
        {
        case NcFile::read:
            GetDimVar();
            break;
        case NcFile::write:
            GetDimVar();
            break;
        case NcFile::replace:
            SetDimVar();
            break;
        case NcFile::newFile:
            SetDimVar();
            break;
        default:
            ;
        };
    }

void vicsekDatabase::SetDimVar()
    {
    //Set the dimensions
    recDim = File.addDim("record");
    nDim = File.addDim("numberOfParticles", N);
    eulerDim = File.addDim("totalNumberOfNeighbors", 6*N);
    dofDim = File.addDim("spatialDegreesOfFreedom", dof);
    unitDim = File.addDim("unit",1);
    boxDim = File.addDim("boxdim",4);
    
    //Set the variables
    timeVar = File.addVar("time",ncDouble,recDim);
    positionVar = File.addVar("position",ncDouble,{recDim,dofDim});
    velocityVar = File.addVar("velocity",ncDouble,{recDim,dofDim});
    typeVar = File.addVar("type",ncInt,{recDim,dofDim});
    BoxMatrixVar = File.addVar("BoxMatrix", ncDouble,{recDim,boxDim});
    neighborVar = File.addVar("neighborNumber", ncInt,{recDim, nDim});
    neighborsVar= File.addVar("neighbors", ncInt,{recDim, eulerDim});
    }

void vicsekDatabase::GetDimVar()
    {
    //Get the dimensions
    recDim = File.getDim("record");
    nDim = File.getDim("numberOfParticles");
    dofDim = File.getDim("spatialDegreesOfFreedom");
    unitDim = File.getDim("unit");
    boxDim = File.getDim("boxdim");
    eulerDim = File.getDim("totalNumberOfNeighbors");

    //Get the variables
    positionVar          = File.getVar("postion");
    velocityVar = File.getVar("velocity");
    neighborVar = File.getVar("neighborNumber");
    neighborsVar = File.getVar("neighbors");
    typeVar          = File.getVar("type");
    BoxMatrixVar    = File.getVar("BoxMatrix");
    timeVar    = File.getVar("time");
    }

void vicsekDatabase::writeState(STATE s, double time, int rec)
    {
    int record = rec;
    double timeToWrite = time;
    if(record<0)
        record = recDim.getSize();
    if (time < 0) timeToWrite = s->currentTime;

    std::vector<double> posdat(dof,0);
    std::vector<double> veldat(dof,0);
    std::vector<double> boxdat(4,0.0);
    std::vector<int> typedat(N,0);
    std::vector<int> neighdat(N,0);//number of neighbors
    std::vector<int> neighborsData(6*N,0);//neighbor indexes
    
    double x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    int idx = 0;

    ArrayHandle<double2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(s->cellVelocities,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(s->neighborNum,access_location::host,access_mode::read);
    int currentNeighborIndex = 0;
    for (int ii = 0; ii < N; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        double px = h_p.data[pidx].x;
        double py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        double vx = h_v.data[pidx].x;
        double vy = h_v.data[pidx].y;
        veldat[(2*idx)] = vx;
        veldat[(2*idx)+1] = vy;
        typedat[ii] = h_ct.data[pidx];
        neighdat[ii] = h_nn.data[pidx];

        vector<int> cellNeighs;
        int cnn;
        s->getCellNeighs(pidx,cnn,cellNeighs);
        for (int jj = 0; jj <cnn; ++jj)
            {
            neighborsData[currentNeighborIndex] = cellNeighs[jj];
            currentNeighborIndex+=1;
            }
        idx +=1;
        };

    //Write all the data
    timeVar.putVar({record},&timeToWrite);
    BoxMatrixVar.putVar({record,0},&boxdat[0]);
    positionVar.putVar({record,0},{1,dofDim.getSize()}, &posdat[0]);
    velocityVar.putVar({record,0},{1,dofDim.getSize()},&veldat[0]);
    neighborVar.putVar({record,0},{1,nDim.getSize()},&neighdat[0]);
    typeVar.putVar({record,0},{1,nDim.getSize()},&typedat[0]);
    neighborsVar.putVar({record,0},{1,eulerDim.getSize()},&neighborsData[0]);


    File.sync();
    }

void vicsekDatabase::readState(STATE t, int rec,bool geometry)
    {
    int totalRecords = GetNumRecs();
    if (rec >= totalRecords)
        {
        printf("Trying to read a database entry that does not exist\n");
        throw std::exception();
        };

    UNWRITTENCODE("AAAAAAAAAH");
    }


