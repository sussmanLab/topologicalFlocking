#include "std_include.h"

#include "cuda_runtime.h"
#include "profiler.h"

#include "Simulation.h"
#include "voronoiModelBase.h"
#include "scalarVicsekModel.h"
#include "vectorVicsekModel.h"
#include "xyLikeScalarVicsek.h"
#include "xyOrderedScalarVicsek.h"
#include "DatabaseNetCDFSPV.h"
#include "vicsekDatabase.h"



int main(int argc, char*argv[])
{
    int c;
    //...some default parameters
    int numpts = 50000; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 50000; //number of time steps to run after initialization
    int initSteps = 10000; //number of initialization steps
    int oneRingSize = 128;//estimate of max number of voro neighbors...for now, best to set this deliberately high

    double dt = 1.0; //the time step size
    double v0 = 0.5;  // the self-propulsion
    double eta = 0.03; //the scalar- or vector- vicsek noise
    double mu = 1.0; //the friction...not relevant at the moment

    double reciprocalNormalization = 6;//negative for standard "normalize by number of neighbors", positive for "normalize by this constant"
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false

    int saveFileFreq = 100; //in multiples of 1/dt
    double aspectRatio = 1.;
    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:i:v:t:e:o:d:a:s:r:")) != -1)
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
            case 'r': aspectRatio = atof(optarg); break;
            case 's': saveFileFreq = atoi(optarg); break;
            case 'a' : reciprocalNormalization = atof(optarg); break;
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


    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;

    char dataname[256];
    sprintf(dataname,"./timeOrderedXYModelTrajectory_N%i_v%.3f_a%.2f_dt%.4f_eta_%.5f.nc",numpts,v0,reciprocalNormalization,dt,eta);
    vicsekDatabase ncdat(numpts,dataname,NcFile::Replace);
    //for both the updaters and the model below the "initializeGPU,!initializeGPU" business is a kludge to declare "I'm not using the GPU and I never will" if gpu < 0.It's ugly, but it will stop memory from being allocated on devices that aren't being used for computation.


    shared_ptr<xyOrderedScalarVicsekModel> vicsek = make_shared<xyOrderedScalarVicsekModel>(numpts,eta,mu,dt,reciprocalNormalization,initializeGPU,!initializeGPU);

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

    double boxLx,boxLy,xy,yx;
    model->Box->getBoxDims(boxLx,xy,yx,boxLy);
    boxLx *= sqrt(aspectRatio);
    boxLy *= sqrt((1./aspectRatio));
    PeriodicBoxPtr b2 = make_shared<periodicBoundaries>(boxLx,boxLy);
    model->alterBox(b2);


    cout << boxLx << "\t" << boxLy << endl;
    cout.flush();
    profiler prof("simulation ");
    
    // Initialize your particle simulation
    cout << "initialization steps... " << endl;
    for (int ii = 0; ii < initSteps; ++ii)
            sim->performTimestep();
    cout << "finished initialization steps... continuing simulation" << endl;
    int frameSkip = saveFileFreq/dt;
    int frameIdx=0;
    for (int ii = 0; ii < tSteps; ++ii)
        {
        prof.start();
        sim->performTimestep();
        prof.end();
        frameIdx+=1;

        if(frameIdx%(frameSkip) ==0)
            {
            double2 vParallel,vTransverse;
            double op = model->vicsekOrderParameter(vParallel,vTransverse);
            cout << frameIdx*dt << "   " << op << "\n";cout.flush();
            ncdat.WriteState(model);
            }
        }
        

    return 0;
}
