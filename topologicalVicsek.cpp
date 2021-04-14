#include "globalCudaDisable.h"

#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "Simulation.h"
#include "voronoiModelBase.h"
#include "scalarVicsekModel.h"
#include "vectorVicsekModel.h"
#include "DatabaseNetCDFSPV.h"


bool globalCudaDisable;
/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    globalCudaDisable = false;
    int c;
    //...some default parameters
    int numpts = 200; //number of cells
    int USE_GPU = 0; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 5; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps
    int oneRingSize = 32;//estimate of max number of voro neighbors...for now, best to set this deliberately high

    double dt = 1.0; //the time step size
    double v0 = 0.1;  // the self-propulsion
    double eta = 0.2; //the scalar- or vector- vicsek noise
    double mu = 1.0; //the friction...not relevant at the moment


    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:i:v:t:e:o:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'o': oneRingSize = atoi(optarg); break;
            case 'd': dt = atof(optarg); break;
            case 'e': eta = atof(optarg); break;
            case 'm': mu = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    if(USE_GPU < 0)
        globalCudaDisable = true;

    char dataname[256];
    sprintf(dataname,"../data/test.nc");
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace,false);

    clock_t t1,t2; //clocks for timing information
    bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) 
        initializeGPU = false;

    shared_ptr<scalarVicsekModel> vicsek = make_shared<scalarVicsekModel>(numpts,eta,mu,dt);//just switch which line is commented out to use scalar or vector viscek model...
    //shared_ptr<vectorVicsekModel> vicsek = make_shared<vectorVicsekModel>(numpts,eta,mu,dt);
    shared_ptr<voronoiModelBase> model = make_shared<voronoiModelBase>();
    model->initializeVoronoiModelBase(numpts,oneRingSize);

    //set the cell activity to have D_r = 1. and a given v_0
    //eventually use Dr rather than eta, to allow heterogeneous noise?
    model->setv0Dr(v0,1.0);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(model);
    sim->addUpdater(vicsek,model);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    if (!gpu) 
        sim->setOmpThreads(abs(USE_GPU));
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    printf("Finished with initialization\n");

    //run for additional timesteps, compute dynamical features, and record timing information
    t1=clock();
    cudaProfilerStart();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10 ==0)
            ncdat.WriteState(model);

        sim->performTimestep();
        };
    cudaProfilerStop();
    t2=clock();
    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
