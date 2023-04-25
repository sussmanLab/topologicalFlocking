#include "vicsekDatabase.h"

vicsekDatabase::vicsekDatabase(int np, string fn, NcFile::FileMode mode)
    : BaseDatabaseNetCDF(fn,mode),
      Nv(np),
      Current(0)
    {
    switch(Mode)
        {
        case NcFile::ReadOnly:
            break;
        case NcFile::Write:
            GetDimVar();
            break;
        case NcFile::Replace:
            SetDimVar();
            break;
        case NcFile::New:
            SetDimVar();
            break;
        default:
            ;
        };
    }

void vicsekDatabase::SetDimVar()
    {
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);

    //Set the variables
    positionVar = File.add_var("position",       ncDouble,recDim, dofDim);
    velocityVar = File.add_var("velocity",       ncDouble,recDim, dofDim);
    neighborVar = File.add_var("neighborNumber", ncInt,recDim, NvDim);
    typeVar     = File.add_var("type",           ncInt,recDim, NvDim );
    directorVar = File.add_var("director",       ncDouble,recDim, NvDim );
    BoxMatrixVar= File.add_var("BoxMatrix",      ncDouble,recDim, boxDim);
    timeVar     = File.add_var("time",           ncDouble,recDim, unitDim);
    }

void vicsekDatabase::GetDimVar()
    {
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    //Get the variables
    positionVar          = File.get_var("postion");
    velocityVar = File.get_var("velocity");
    neighborVar = File.get_var("neighborNumber");
    directorVar          = File.get_var("director");
    typeVar          = File.get_var("type");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    timeVar    = File.get_var("time");
    }

void vicsekDatabase::WriteState(STATE s, double time, int rec)
    {
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<double> boxdat(4,0.0);
    double x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<double> posdat(2*Nv);
    std::vector<double> veldat(2*Nv);
    std::vector<double> directordat(Nv);
    std::vector<int> typedat(Nv);
    std::vector<int> neighdat(Nv);
    int idx = 0;

    ArrayHandle<double2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(s->cellVelocities,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(s->neighborNum,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
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
        directordat[ii] = atan2(h_v.data[pidx].y,h_v.data[pidx].x);
        typedat[ii] = h_ct.data[pidx];
        neighdat[ii] = h_nn.data[pidx];
        idx +=1;
        };

    //Write all the data
    timeVar      ->put_rec(&time,      rec);
    positionVar      ->put_rec(&posdat[0],     rec);
    velocityVar      ->put_rec(&veldat[0],     rec);
    neighborVar       ->put_rec(&neighdat[0],      rec);
    typeVar       ->put_rec(&typedat[0],      rec);
    directorVar       ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);

    File.sync();
    }

void vicsekDatabase::ReadState(STATE t, int rec,bool geometry)
    {
    //initialize the NetCDF dimensions and variables
    int tester = File.num_vars();
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(& t->currentTime,1,1);


    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<double> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t->Box->setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions
    positionVar-> set_cur(rec);
    std::vector<double> posdata(2*Nv,0.0);
    positionVar->get(&posdata[0],1, dofDim->size());

    ArrayHandle<double2> h_p(t->cellPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        double px = posdata[(2*idx)];
        double py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };

    //get cell types and cell directors
    typeVar->set_cur(rec);
    std::vector<int> ctdata(Nv,0.0);
    typeVar->get(&ctdata[0],1, NvDim->size());
    ArrayHandle<int> h_ct(t->cellType,access_location::host,access_mode::overwrite);

    directorVar->set_cur(rec);
    std::vector<double> cddata(Nv,0.0);
    directorVar->get(&cddata[0],1, NvDim->size());
    ArrayHandle<double> h_cd(t->cellDirectors,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        h_cd.data[idx]=cddata[idx];;
        h_ct.data[idx]=ctdata[idx];;
        };
    //by default, compute the triangulation and geometrical information
    if(geometry)
        {
        UNWRITTENCODE("AAAAAAAAAH");
//        t->globalTriangulationCGAL();
//        t->resetLists();
//        if(t->GPUcompute)
//            t->computeGeometryGPU();
//        else
//            t->computeGeometryCPU();
        };
    }


