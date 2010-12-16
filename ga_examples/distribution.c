/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
 + 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

Jeff R. Hammond
Leadership Computing Facility
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5381
e-mail: jhammond@mcs.anl.gov

                  GOVERNMENT LICENSE

Portions of this material resulted from work developed under a U.S.
Government Contract and are subject to the following license: the Government
is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable worldwide license in this computer software to reproduce, prepare
derivative works, and perform publicly and display publicly.

                  DISCLAIMER

This computer code material was prepared, in part, as an account of work
sponsored by an agency of the United States Government.  Neither the United
States, nor the University of Chicago, nor any of their employees, makes any
warranty express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe
privately owned rights.

 ***************************************************************************/

#include "driver.h"


int main(int argc, char **argv)
{
    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);
    assert(provided == desired);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    GA_Initialize();

    int ndim = 2;
    int dims[ndim];
    int chunk[ndim];
    int status;

    int rows = ( argc>1 ? atoi(argv[1]) : 1000 );
    int cols = ( argc>2 ? atoi(argv[2]) : 1000 );

    dims[0] = rows;
    dims[1] = cols;
    chunk[0] = -1;
    chunk[1] = -1;

    int g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"a");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,GA_Pgroup_get_world());
    status = GA_Allocate(g_a);
    assert(status!=0);

    GA_Sync();
    double val = 1.0;
    GA_Randomize(g_a, &val);
    GA_Sync();

    int i;
    int lo[ndim];
    int hi[ndim];
    for (i=0;i<size;i++)
    {
        NGA_Distribution(g_a, i, lo, hi);
        if (rank==0)
        {
            fprintf(stdout,
                    " rank %4d: lo[0] = %4d lo[1] = %4d \n            hi[0] = %4d hi[1] = %4d \n",
                    i,lo[0],lo[1],hi[0],hi[1]);
        }
    }
    fflush(stdout);

    GA_Sync();
    GA_Destroy(g_a);
    GA_Sync();

    GA_Terminate();
    MPI_Finalize();

    return(0);
}
