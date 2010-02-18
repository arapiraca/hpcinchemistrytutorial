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

int diagonalize(int rank)
{
    int me,nproc;
    int status;
    int ndim = 2;
    int dims[ndim];
    int chunk[ndim];

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = -1;
    chunk[1] = -1;

    if (me == 0) printf("! GA_Diag_std: rank %d matrix\n",rank);

    double* eig = (double*) ARMCI_Malloc_local(rank*sizeof(double)); assert(eig!=NULL);

    int g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"input matrix");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,GA_Pgroup_get_world());
    status = GA_Allocate(g_a); assert(status!=0);
    int g_v = GA_Duplicate(g_a,"eigenvectors"); assert(g_v!=0);

    double val = 10;
    //GA_Fill(g_a, &val);
    GA_Randomize(g_a, &val);
    //val = 20;
    //GA_Shift_diagonal(g_a, &val);
    if (rank<40) GA_Print(g_a);
    GA_Zero(g_v);

    GA_Sync();
    double start = MPI_Wtime();

    if (me == 0) printf("! starting GA_Diag_std\n");
    GA_Diag_std(g_a,g_v,eig);

    GA_Sync();
    double finish = MPI_Wtime();

    if (rank<40) GA_Print(g_v);
    int i;
    if (me == 0) for (i=0;i<rank;i++) printf("eigenvalue %d = %f\n",i,eig[i]);

    if (me == 0) printf("! diagonalize took %f seconds\n",finish-start);
    fflush(stdout);

    GA_Destroy(g_v);
    GA_Destroy(g_a);

    status = ARMCI_Free_local(eig); assert(status==0);

    return(0);
}
