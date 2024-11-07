#include "std_include.h"

#include "cuda_runtime.h"
#include "profiler.h"

#include "Simulation.h"
#include "voronoiModelBase.h"
#include "scalarVicsekModel.h"
#include "vectorVicsekModel.h"
#include "xyLikeScalarVicsek.h"
#include "xyOrderedScalarVicsek.h"
#include "vicsekDatabase.h"
#include "vectorValueDatabase.h"
#include "analysisPackage.h"

#include<algorithm>
#include<iterator>
#include<random>

void randomizePositionVelocityIndices(shared_ptr<voronoiModelBase> model)
    {
    //switch pos and vel, perform triangulations
    std::random_device rd;
    std::mt19937 g(rd());
    int numpts = model->getNumberOfDegreesOfFreedom();
    ArrayHandle<double2> hp(model->returnPositions());
    ArrayHandle<double2> hv(model->returnVelocities());
    vector<double2> tp(numpts);
    vector<double2> tv(numpts);
    vector<int> indices(numpts);
    for(int jj =0; jj < numpts;++jj)
        {
        indices[jj]=jj;
        tp[jj] = hp.data[jj];
        tv[jj] = hv.data[jj];
        }
    std::shuffle(indices.begin(),indices.end(),g);
    for(int jj =0; jj < numpts;++jj)
        {
        int idx = indices[jj];
        hp.data[jj] = tp[idx];
        hv.data[jj] = tv[idx];
            if(jj ==0) 
                cout << idx << " swapped" << endl;
        }
    model->enforceTopology();
    }

int getMaxNumberNeighbors(shared_ptr<voronoiModelBase> voro, int Ndof)
    {
    int answer = 0;
    ArrayHandle<int> nNeighs(voro->neighborNum);
    for(int ii=0; ii < Ndof; ++ii)
        {
        int n = nNeighs.data[ii];
        if(n > answer)
            answer = n;
        }
    return answer;
    };
void getAnglesRelativeToFlocking(shared_ptr<voronoiModelBase> model,std::vector<double> &dotProductVector)
    {
    int numpts = model->getNumberOfDegreesOfFreedom();
    dotProductVector.resize(numpts);

    double2 vParallel,vTransverse;
    double op = model->vicsekOrderParameter(vParallel,vTransverse);

    ArrayHandle<double2> hv(model->returnVelocities());
    for(int jj =0; jj < numpts;++jj)
        dotProductVector[jj] = asin(hv.data[jj].x*vTransverse.x + hv.data[jj].y*vTransverse.y);
    };

int main(int argc, char*argv[])
{
    int c;
    //...some default parameters
    int numpts = 50000; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 50000; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps
    int oneRingSize = 128;//estimate of max number of voro neighbors...for now, best to set this deliberately high

    double dt = 1.0; //the time step size
    double v0 = 0.5;  // the self-propulsion
    double eta = 0.03; //the scalar- or vector- vicsek noise
    double mu = 1.0; //the friction...not relevant at the moment

    double reciprocalNormalization = 6;//negative for standard "normalize by number of neighbors", positive for "normalize by this constant"
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false

    int saveFileFreq = 100; //in multiples of 1/dt
    double aspectRatio = 1.;

    int index = 0;
    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:m:i:v:t:x:e:o:d:a:s:r:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            /*case 'g': USE_GPU = atoi(optarg); break;*/
            case 't': tSteps = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'x': index = atoi(optarg); break;
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


    bool initializeGPU = false;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;

    char dataname[256];
    sprintf(dataname,"./timeOrderedXYModel_Snapshots_N%i_v%.3f_a%.2f_dt%.4f_eta%.5f_idx%i.nc",numpts,v0,reciprocalNormalization,dt,eta,index);
    vicsekDatabase ncdat(numpts,dataname,NcFile::replace);

    char dataname2[256];
    sprintf(dataname2,"./timeOrderedXYModel_orderParameterTimeseries_N%i_v%.3f_a%.2f_dt%.4f_eta%.5f_idx%i.nc",numpts,v0,reciprocalNormalization,dt,eta,index);
    vectorValueDatabase vvdat1(4,dataname2,NcFile::replace);

    char dataname3[256];
    sprintf(dataname3,"./timeOrderedXYModel_postShuffle_orderParameterTimeseries_N%i_v%.3f_a%.2f_dt%.4f_eta%.5f_idx%i.nc",numpts,v0,reciprocalNormalization,dt,eta,index);
    vectorValueDatabase vvdat2(4,dataname3,NcFile::replace);


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

    if(aspectRatio != 1.)
        {
        double boxLx,boxLy,xy,yx;
        model->Box->getBoxDims(boxLx,xy,yx,boxLy);
        boxLx *= sqrt(aspectRatio);
        boxLy *= sqrt((1./aspectRatio));
        PeriodicBoxPtr b2 = make_shared<periodicBoundaries>(boxLx,boxLy);
        model->alterBox(b2);
        cout << boxLx << "\t" << boxLy << endl;
        cout.flush();
        }

    profiler prof("simulation ");
    
    // Initialize your particle simulation
    cout << "initialization steps... " << endl;
    for (int ii = 0; ii < initSteps; ++ii)
            sim->performTimestep();
    cout << "finished initialization steps... continuing simulation" << endl;

    ncdat.writeState(model);

    int frameSkip = saveFileFreq;
    std::vector<double> vtVector;
    std::vector<double> orderParameterMoments(4);
    for (int ii = 0; ii < tSteps; ++ii)
        {
        prof.start();
        sim->performTimestep();
        prof.end();

        double2 vParallel,vTransverse;
        double op = model->vicsekOrderParameter(vParallel,vTransverse);
        getAnglesRelativeToFlocking(model,vtVector);

        double m1 = computeMoment(vtVector,1);
        double m2 = computeMoment(vtVector,2);
        double m3 = computeMoment(vtVector,3);
        double m4 = computeMoment(vtVector,4);
        orderParameterMoments[0] = m1;
        orderParameterMoments[1] = m2;
        orderParameterMoments[2] = m3;
        orderParameterMoments[3] = m4;
        vvdat1.writeState(orderParameterMoments,ii*dt);
        if(ii%(frameSkip) ==0)
            {
            cout << "timestep: "<< ii*dt << " order parameter:" << op << " vPar:" << vParallel.x<< " " <<vParallel.y <<"\t maxNeighs:" <<getMaxNumberNeighbors(model,numpts) << endl;
            cout << "mean: " << m1 << " m2: " << m2 << " m3: " << m3 << " m4: " << m4 << endl;
            }
        }

    cout << "randomizing indices and continuing" << endl;
    randomizePositionVelocityIndices(model);
    prof.print();

    for (int ii = 0; ii < tSteps; ++ii)
        {
        prof.start();
        sim->performTimestep();
        prof.end();

        double2 vParallel,vTransverse;
        double op = model->vicsekOrderParameter(vParallel,vTransverse);
        getAnglesRelativeToFlocking(model,vtVector);

        double m1 = computeMoment(vtVector,1);
        double m2 = computeMoment(vtVector,2);
        double m3 = computeMoment(vtVector,3);
        double m4 = computeMoment(vtVector,4);
        orderParameterMoments[0] = m1;
        orderParameterMoments[1] = m2;
        orderParameterMoments[2] = m3;
        orderParameterMoments[3] = m4;
        vvdat2.writeState(orderParameterMoments,ii*dt);
        if(ii%(frameSkip) ==0)
            {
            cout << "timestep: "<< ii*dt << " order parameter:" << op << " vPar:" << vParallel.x<< " " <<vParallel.y <<"\t maxNeighs:" <<getMaxNumberNeighbors(model,numpts) << endl;
            cout << "mean: " << m1 << " m2: " << m2 << " m3: " << m3 << " m4: " << m4 << endl;
            }
        }
    prof.print();
        
    return 0;
}
