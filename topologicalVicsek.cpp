#include "std_include.h"

#include "cuda_runtime.h"
#include "profiler.h"

#include "Simulation.h"
#include "voronoiModelBase.h"
#include "scalarVicsekModel.h"
#include "vectorVicsekModel.h"
#include "DatabaseNetCDFSPV.h"

/*!
 * This file is a simple implementation of either the scalar or vectorial vicsek model,
 * where neighbors are chosen according to an instantaneous Voronoi tessellation of the point set.
 * Good stuff.
*/
int main(int argc, char*argv[])
{
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

    char dataname[256];
    sprintf(dataname,"../data/test.nc");
    SPVDatabaseNetCDF ncdat(numpts,dataname,NcFile::Replace);

    profiler prof("voroVicsek initial ");
    profiler prof2("voroVicsek late stage");

    clock_t t1,t2; //clocks for timing information
    bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;

    //just switch which line is commented out to use scalar or vector viscek model...
    //for both the updaters and the model below the "initializeGPU,!initializeGPU" business is a kludge to declare "I'm not using the GPU and I never will" if gpu < 0.It's ugly, but it will stop memory from being allocated on devices that aren't being used for computation.
    shared_ptr<scalarVicsekModel> vicsek = make_shared<scalarVicsekModel>(numpts,eta,mu,dt,initializeGPU,!initializeGPU);
    //shared_ptr<vectorVicsekModel> vicsek = make_shared<vectorVicsekModel>(numpts,eta,mu,dt,initializeGPU,!initializeGPU);
    //
    shared_ptr<voronoiModelBase> model = make_shared<voronoiModelBase>(initializeGPU,!initializeGPU);
    if (gpu)
        model->setGPU();
    else
        model->setCPU();
    model->initializeVoronoiModelBase(numpts,oneRingSize);

    //set the cell activity to have D_r = 1. and a given v_0
    model->setv0Dr(v0,1.0);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(model);
    sim->addUpdater(vicsek,model);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        if(ii%10 ==0)
            prof.start();
        sim->performTimestep();
        if(ii%10 ==0)
            prof.end();
        };
    printf("Finished with initialization\n");

    printf("beginning primary loop\n");
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%10 ==0)
            {
            prof2.start();
//            ncdat.WriteState(model);
            }

        sim->performTimestep();
        if(ii%10 ==0)
            prof2.end();
        };

    prof.print();
    prof2.print();
    return 0;
};
