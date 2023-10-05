#ifndef xyLikeScalarVicsekModel_H
#define xylikeScalarVicsekModel_H

#include "simpleEquationOfMotion.h"
#include "Simple2DActiveCell.h"


class xyLikeScalarVicsekModel : public simpleEquationOfMotion
    {
    public:
        //!base constructor sets the default time step size to unity and uses the GPU
        xyLikeScalarVicsekModel(int N, double _eta, double _mu=1.0, double _deltaT=1.0, double _reciprocalNormalization = -1., bool _gpu = true, bool _neverGPU=false);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();
        //!set the active model
        virtual void set2DModel(shared_ptr<Simple2DModel> _model);
        double mu;
        double eta;
    protected:
        //!A shared pointer to a simple active model
        shared_ptr<Simple2DActiveCell> activeModel;

        //!A container for intermediate caluclations of the new director
        GPUArray<double2> newVelocityDirector;

        //!Flag for whether to normalize by neighbor number or by a constant
        bool reciprocalModel;
        //! normalization if the above is true
        double reciprocalNormalization;

    };

#endif

