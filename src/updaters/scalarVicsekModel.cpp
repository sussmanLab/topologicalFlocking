#include "scalarVicsekModel.h"
//#include "scalarVicsekModel.cuh"
#include "functions.h"

/*! \file scalarVicsekModel.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
scalarVicsekModel::scalarVicsekModel(int _N,double _eta, double _mu, double _deltaT)
    {
    Timestep = 0;
    deltaT = _deltaT;
    mu = _mu;
    eta = _eta;
    GPUcompute = true;
    Ndof = _N;
    noise.initialize(Ndof);
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
    ArrayHandle<double2> h_motility(activeModel->Motility,access_location::host,access_mode::read);

    for(int ii = 0; ii<Ndof; ++ii)
        {
        double v0i = h_motility.data[ii].x;
        double Dri = h_motility.data[ii].y;
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
        
        //rotate new director
        double u = noise.getRealUniform(-.5,.5);
        double theta = 2.0*PI*u*eta;
        rotate2D(newVel.data[ii],theta);
        }
    //update and normalize
    for (int ii = 0; ii < Ndof; ++ii)
        {
        vel.data[ii] = (1./norm(newVel.data[ii])) * newVel.data[ii];
        }
    }//ArrayHandle scope
    };

/*!
The straightforward GPU implementation
*/
void scalarVicsekModel::integrateEquationsOfMotionGPU()
    {
    /*
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<double2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<double> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
    ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
    ArrayHandle<double2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

    gpu_spp_aligning_eom_integration(d_f.data,
                 d_v.data,
                 d_disp.data,
                 d_motility.data,
                 d_cd.data,
                 d_RNG.data,
                 Ndof,
                 deltaT,
                 Timestep,
                 mu,
                 J);
    };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    */
    };

