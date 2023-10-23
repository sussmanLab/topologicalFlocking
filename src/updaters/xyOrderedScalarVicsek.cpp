#include "xyOrderedScalarVicsek.h"
//#include "xyOrderedScalarVicsek.cuh"
//#include "scalarVicsekModel.cuh"
#include "functions.h"


xyOrderedScalarVicsekModel::xyOrderedScalarVicsekModel(int _N,double _eta, double _mu, double _deltaT, double _reciprocalNormalization, bool _gpu, bool _neverGPU) : simpleEquationOfMotion(_gpu,_neverGPU)
    {
    Timestep = 0;
    deltaT = _deltaT;
    mu = _mu;
    eta = _eta;
    GPUcompute = true;
    Ndof = _N;
    noise.initialize(Ndof);
    if(!neverGPU)
        noise.initializeGPURNGs();
    displacements.resize(Ndof);
    reciprocalNormalization = _reciprocalNormalization;
    if(reciprocalNormalization < 0)
        {
        reciprocalModel = false;
        printf("setting an XY-Ordered scalar vicsek model with nonreciprocal neighbor-number normalization\n");
        }
    else
        {
        reciprocalModel = true;
        printf("setting an XY-Ordered scalar vicsek model with reciprocal normalization n_i(t) = %f \n",reciprocalNormalization);
        }
    };

/*!
Set the shared pointer of the base class to passed variable; cast it as an active cell model
Additionally, convert the current value of the cell directors into a vector quantity, stored in the cell velocities as a unit vector
*/
void xyOrderedScalarVicsekModel::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    activeModel = dynamic_pointer_cast<Simple2DActiveCell>(model);
    ArrayHandle<double> h_n(activeModel->cellDirectors);
    ArrayHandle<double2> h_v(activeModel->cellVelocities);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_n.data[ii] = noise.getRealUniform(0.,2.*PI);
        h_v.data[ii].x = cos(h_n.data[ii]);
        h_v.data[ii].y = sin(h_n.data[ii]);
        }
    }

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void xyOrderedScalarVicsekModel::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    }

/*!
The straightforward CPU implementation
*/
void xyOrderedScalarVicsekModel::integrateEquationsOfMotionCPU()
    {
    //first, compute forces and move particles in the correct direction
    activeModel->computeForces(); //if connected to voronoiModelBase this will do nothing...

    {//scope for array Handles
    ArrayHandle<double2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(activeModel->cellVelocities);
    ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);

    double v0i = activeModel->v0;
    for(int ii = 0; ii<Ndof; ++ii)
        {
        h_disp.data[ii] = deltaT*(v0i*h_v.data[ii] + mu*h_f.data[ii]);
        }
    }
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();

    //update directors
    {//ArrayHandle scope
    ArrayHandle<double2> vel(activeModel->cellVelocities);
    ArrayHandle<int> neighs(activeModel->neighbors);
    ArrayHandle<int> nNeighs(activeModel->neighborNum);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        int m = nNeighs.data[ii];
        double thetaUpdate = 0;
        double thetaI = atan2(vel.data[ii].y,vel.data[ii].x);
        for (int jj=0; jj < m; ++jj)
            {
            int neighIdx = neighs.data[activeModel->n_idx(jj,ii)];
            double thetaJ = atan2(vel.data[neighIdx].y,vel.data[neighIdx].x);
            thetaUpdate += deltaT*sin(thetaJ-thetaI);
            }
        if(reciprocalNormalization > 0) //reciprocal model: divide by a constant
            {
            thetaUpdate = thetaUpdate / reciprocalNormalization;
            }
        else // non-reciprocal model: divide by neighbor number
            {
            if (m > 0)
                thetaUpdate = thetaUpdate / ((double) m);
            }

        //add a little noise
        thetaUpdate += sqrt(deltaT)*2.0*PI*eta*noise.getRealUniform(-.5,.5);

        //rotate director
        rotate2D(vel.data[ii],thetaUpdate);
        }
    }//ArrayHandle scope
    };

/*!
The straightforward GPU implementation
*/
void xyOrderedScalarVicsekModel::integrateEquationsOfMotionGPU()
    {
    /*
    //first stage: update positions
    activeModel->computeForces();
    {//scope for array handles
    ArrayHandle<double2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<double2> d_v(activeModel->cellVelocities,access_location::device,access_mode::read);
    ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
    gpu_scalar_vicsek_update(d_f.data,
                             d_v.data,
                             d_disp.data,
                             activeModel->v0,
                             Ndof,
                             deltaT,
                             mu);
    }
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();

    //second stage: update directors
    {//scope for array handles
    ArrayHandle<double2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_newV(newVelocityDirector,access_location::device,access_mode::overwrite);
    ArrayHandle<int> neighs(activeModel->neighbors,access_location::device,access_mode::read);
    ArrayHandle<int> nNeighs(activeModel->neighborNum,access_location::device,access_mode::read);
    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

    gpu_xyOrdered_scalar_vicsek_directors(d_v.data,
                                d_newV.data,
                                nNeighs.data,
                                neighs.data,
                                reciprocalNormalization,
                                activeModel->n_idx,
                                d_RNG.data,
                                Ndof,
                                eta,
                                deltaT);
    }
    //swap velocity and newVelocity data
    activeModel->cellVelocities.swap(newVelocityDirector);
    */
    };

