#include <SFML/Graphics.hpp>


#include "std_include.h"

#include "cuda_runtime.h"
#include "profiler.h"

#include "Simulation.h"
#include "voronoiModelBase.h"
#include "scalarVicsekModel.h"
#include "vectorVicsekModel.h"
#include "xyLikeScalarVicsek.h"
#include "xyOrderedScalarVicsek.h"



int main(int argc, char*argv[])
{
    int c;
    //...some default parameters
    int numpts = 200; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 500; //number of time steps to run after initialization
    int initSteps = 1000; //number of initialization steps
    int oneRingSize = 64;//estimate of max number of voro neighbors...for now, best to set this deliberately high

    double dt = 1.0; //the time step size
    double v0 = 0.5;  // the self-propulsion
    double eta = 0.2; //the scalar- or vector- vicsek noise
    double mu = 1.0; //the friction...not relevant at the moment

    double reciprocalNormalization = -1;//negative for standard "normalize by number of neighbors", positive for "normalize by this constant"
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:i:v:t:e:o:d:a:")) != -1)
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

    //for both the updaters and the model below the "initializeGPU,!initializeGPU" business is a kludge to declare "I'm not using the GPU and I never will" if gpu < 0.It's ugly, but it will stop memory from being allocated on devices that aren't being used for computation.


    //shared_ptr<xyLikeScalarVicsekModel> vicsek = make_shared<xyLikeScalarVicsekModel>(numpts,eta,mu,dt,reciprocalNormalization,initializeGPU,!initializeGPU);
    shared_ptr<xyOrderedScalarVicsekModel> vicsek = make_shared<xyOrderedScalarVicsekModel>(numpts,eta,mu,dt,reciprocalNormalization,initializeGPU,!initializeGPU);
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


    int frameSkip = 1/dt;

    int windowPixelL=600;
    double boxL = 1.0*sqrt(numpts);
    float pixelConversion = windowPixelL / boxL;
     sf::RenderWindow window(sf::VideoMode(windowPixelL,windowPixelL), "Particle Visualization");
    profiler prof("simulation ");
    profiler prof2("visualization");
    
    // Initialize your particle simulation
    std::vector<sf::Vertex> vertices(numpts);
    int frameIdx=0;
    cout << frameSkip << endl;
    while (window.isOpen())
        {
        sf::Event event;
        while (window.pollEvent(event))
            {
            if (event.type == sf::Event::Closed)
                {
                window.close();
                }
            // Handle other user interactions (e.g., starting/stopping simulation)
            }
        bool simulating = true;
        if (simulating)
            {
            // Update particle positions
            prof.start();
            sim->performTimestep();
            prof.end();
            frameIdx+=1;
            window.clear(sf::Color::Black);
            if(frameIdx%frameSkip ==0)
                {
                if(frameIdx%(frameSkip*100) ==0)
                    cout << frameIdx*dt << "\t";cout.flush();
                prof2.start();
                ArrayHandle<double2> p(model->returnPositions());
                ArrayHandle<double> directors(model->cellDirectors);
                double2 vParallel,vTransverse;
                double op = model->vicsekOrderParameterDirector(vParallel,vTransverse);
                for (int ii = 0; ii < numpts; ++ii)
                    {
                    float posX = p.data[ii].x*pixelConversion;
                    float posY= p.data[ii].y*pixelConversion;
                    double theta = directors.data[ii];
                    double nx = cos(theta);
                    double ny = sin(theta);
                    double dotProdTrans  = nx*vTransverse.x + ny*vTransverse.y;
                    double dotProdPar  = nx*vParallel.x + ny*vParallel.y;
                    //pos, color, texture. pos is straightforward. color is RGB from 0 to 255 (and optional 0 to 255 opacity). No idea what texture does yet
                    vertices[ii] = sf::Vertex(sf::Vector2f(posX,posY), sf::Color(128*(1+dotProdTrans),0*255*dotProdPar,128*(-dotProdTrans+1),255*fabs(dotProdTrans)), sf::Vector2f(100.f, 100.f));
                    }

            
                // Draw particles to the window
                // Example: for each particle p, use window.draw(p);
                window.draw(&vertices[0],vertices.size(),sf::Points);
                window.display();
                prof2.end();
                };
            if(frameIdx% (10*frameSkip) ==0)
                {
//                prof.print();
//                prof2.print();
                };
            }
        }

    cout << endl;
    return 0;
}
