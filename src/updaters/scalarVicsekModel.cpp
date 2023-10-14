#include "scalarVicsekModel.h"
#include "scalarVicsekModel.cuh"
#include "functions.h"

/*! \file scalarVicsekModel.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
scalarVicsekModel::scalarVicsekModel(int _N,double _eta, double _mu, double _deltaT, bool _gpu, bool _neverGPU) : simpleEquationOfMotion(_gpu,_neverGPU)
    {
    Timestep = 0;
    deltaT = _deltaT;
    mu = _mu;
    eta = _eta;
    GPUcompute = true;
    Ndof = _N;
    if(neverGPU)
        {
        newVelocityDirector.noGPU=true;
        };
    noise.initialize(Ndof);
    if(!neverGPU)
        noise.initializeGPURNGs();
    displacements.resize(Ndof);
    newVelocityDirector.resize(Ndof);
    };

/*!
Set the shared pointer of the base class to passed variable; cast it as an active cell model
Additionally, convert the current value of the cell directors into a vector quantity, stored in the cell velocities as a unit vector
*/
void scalarVicsekModel::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    activeModel = dynamic_pointer_cast<Simple2DActiveCell>(model);
    ArrayHandle<double> h_n(activeModel->cellDirectors);
    ArrayHandle<double2> h_v(activeModel->cellVelocities);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_v.data[ii].x = cos(h_n.data[ii]);
        h_v.data[ii].y = sin(h_n.data[ii]);
        }
    }

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void scalarVicsekModel::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        newVelocityDirector.resize(Ndof);
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
void scalarVicsekModel::integrateEquationsOfMotionCPU()
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
    ArrayHandle<double2> newVel(newVelocityDirector);
    ArrayHandle<int> neighs(activeModel->neighbors);
    ArrayHandle<int> nNeighs(activeModel->neighborNum);
    for (int ii = 0; ii < Ndof; ++ii)
        {
        //average direction of neighbors?
        int m = nNeighs.data[ii];
        newVel.data[ii] = vel.data[ii];
        for (int jj=0; jj < m; ++jj)
            {
            newVel.data[ii] = newVel.data[ii] + vel.data[neighs.data[activeModel->n_idx(jj,ii)]];
            }
        m +=1; //account for self-alignment

        //normalize and rotate new director
        double u = noise.getRealUniform(-.5,.5);
        double theta = 2.0*PI*u*eta;
        newVel.data[ii] = (1./norm(newVel.data[ii])) * newVel.data[ii];
        rotate2D(newVel.data[ii],theta);
        }
    //update and normalize
    for (int ii = 0; ii < Ndof; ++ii)
        {
        vel.data[ii] = newVel.data[ii];
        }
    }//ArrayHandle scope
    };

/*!
The straightforward GPU implementation
*/
void scalarVicsekModel::integrateEquationsOfMotionGPU()
    {
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

    gpu_scalar_vicsek_directors(d_v.data,
                                d_newV.data,
                                nNeighs.data,
                                neighs.data,
                                activeModel->n_idx,
                                d_RNG.data,
                                Ndof,
                                eta);
    }
    //swap velocity and newVelocity data
    activeModel->cellVelocities.swap(newVelocityDirector);
    };
