#ifndef DATABASE_VICSEK_H
#define DATABASE_VICSEK_H

#include "voronoiModelBase.h"
#include "DatabaseNetCDF.h"

class vicsekDatabase : public BaseDatabaseNetCDF
{
private:
    typedef shared_ptr<voronoiModelBase> STATE;
    int Nv; //!< number of vertices in delaunay triangulation
    NcDim *recDim, *NvDim, *dofDim, *boxDim, *unitDim; //!< NcDims we'll use
    NcVar *positionVar, *velocityVar, *neighborVar, *typeVar, *directorVar, *BoxMatrixVar, *timeVar; //!<NcVars we'll use
    int Current;    //!< keeps track of the current record when in write mode


public:
    vicsekDatabase(int np, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly);
    ~vicsekDatabase(){File.close();};

protected:
    void SetDimVar();
    void GetDimVar();

public:
    int  GetCurrentRec(); //!<Return the current record of the database
    //!Get the total number of records in the database
    int GetNumRecs(){
                    NcDim *rd = File.get_dim("rec");
                    return rd->size();
                    };

    //!Write the current state of the system to the database. If the default value of "rec=-1" is used, just append the current state to a new record at the end of the database
    virtual void WriteState(STATE c, double time = -1.0, int rec=-1);
    //!Read the "rec"th entry of the database into SPV2D state c. If geometry=true, after reading a CPU-based triangulation is performed, and local geometry of cells computed.
    virtual void ReadState(STATE c, int rec,bool geometry=true);

};
#endif
