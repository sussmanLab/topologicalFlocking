#ifndef DATABASE_VICSEK_H
#define DATABASE_VICSEK_H

#include "voronoiModelBase.h"
#include "DatabaseNetCDF.h"

class vicsekDatabase : public BaseDatabaseNetCDF
{
public:
    vicsekDatabase(int np, string fn="temp.nc", NcFile::FileMode mode=NcFile::read);
    ~vicsekDatabase(){File.close();};

    typedef shared_ptr<Simple2DActiveCell> STATE;
    //! NcDims we'll use
    NcDim recDim, nDim,  dofDim, unitDim, boxDim, eulerDim;
    //! NcVars
    NcVar timeVar, positionVar, barycentricPositionVar,faceIndexVar, velocityVar, typeVar, neighborVar,neighborsVar,BoxMatrixVar;
    //!read values in a new value and vector
    virtual void readState(STATE s, int rec, bool geometry = true);
    //!write a new value and vector
    virtual void writeState(STATE s, double time = -1, int rec = -1);

protected:
        //! Set all of the netcdf dimensions and variables in the file (for creating or writing new files)
        void SetDimVar();
        //! When reading (or writing to an existing files) load in the pre-existing netcdf info
        void GetDimVar();
        //!number of particles in the model
        int N;
        //!size of the vectors
        int dof;
        //! a variable that can be loaded when a state is read
        double val;
        //! a vector for reading doubles in and out
        vector<double> vec;
};
#endif
