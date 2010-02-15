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


/***************************************************************************
 *                                                                         *
 * bigtest:                                                                *
 *       -demonstrates how to create a GA using the new API                *
 *       -demonstrates how to obtain local access to GA memory             *
 *                                                                         *
 ***************************************************************************/

int bigtest(int rank)
{
	int me,nproc;
    //int p,d,i,j;
    int g_a;
    int status;
    int ndim=1;
    int dims[ndim];
    int chunk[ndim];
    int pg_world;
    //int* pg_list;
    double val;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    dims[0] = pow(2,rank);
    chunk[0] = -1;

    //pg_list = (int*) malloc(nproc*sizeof(int));
    //for(p=0; p<nproc; p++){ pg_list[p]=p; }
    //pg_world = GA_Pgroup_create(pg_list,nproc);
    /* This is an easier way to get the world group */
    pg_world = GA_Pgroup_get_world();

    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"test array A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,pg_world);
    status = GA_Allocate(g_a);
    if(0 != status){ GA_Error("GA_Allocate failed",0); };
    
    GA_Zero(g_a);
    val = -1.0;
    GA_Fill(g_a,&val);

    GA_Print_distribution(g_a);

    GA_Destroy(g_a);

    /* cannot do this if using pg_world = GA_Pgroup_get_world() */
    //if(0 != GA_Pgroup_destroy(pg_world)){};

    return(0);
}



