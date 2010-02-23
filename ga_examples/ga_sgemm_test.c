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

void sgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );

/***************************************************************************
 *                                                                         *
 * ga_sgemm_test:                                                                  *
 *       -demonstrates how to create a GA using the new API                *
 *       -matrix multiplication									           *
 *                                                                         *
 ***************************************************************************/

int ga_sgemm_test(int rank)
{
	int me,nproc;
    int g_a,g_b,g_c; // GA handles
    int status;
    int ndim = 2;
    int dims[2];
    int chunk[2];
    int pg_world;   // world processor group
    double alpha,beta;
    double one  = 1.0;
    double start,finish;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = -1;
    chunk[1] = -1;

    if (me==0) printf("! ga_sgemm_test: rank %d matrix\n",rank);

    g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"matrix A");
    GA_Set_data(g_a,ndim,dims,MT_REAL);
    pg_world = GA_Pgroup_get_world();
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);

    status = GA_Allocate(g_a); assert(status!=0);
    g_b  = GA_Duplicate(g_a,"matrix B"); assert(g_b!=0);
    g_c  = GA_Duplicate(g_a,"matrix C"); assert(g_c!=0);
    GA_Sync();

    GA_Fill(g_a,&one);
    GA_Fill(g_b,&one);
    GA_Zero(g_c);

	alpha = 1.0;
    beta  = 0.0;

    GA_Sync();
    start = MPI_Wtime(); 
    // GA_Sgemm uses Fortran ordering, hence the double 'T'
    GA_Sgemm('T','T',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c);
    GA_Sync();
    finish = MPI_Wtime(); 
    double t_TT=finish-start;

    GA_Sync();
    start = MPI_Wtime(); 
    GA_Sgemm('N','N',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c);
    GA_Sync();
    finish = MPI_Wtime(); 
    double t_NN=finish-start;
    if (me==0) printf("DATA nproc %d GA_SgemmTT %f GA_SgemmNN %f\n",nproc,t_TT,t_NN);

    GA_Destroy(g_c);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}
