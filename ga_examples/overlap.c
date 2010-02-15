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
 * overlap:                                                                *
 *       -tests how well GA allows overlap of comm. and comp.	           *
 *                                                                         *
 ***************************************************************************/

#define REPS 28

void delay(unsigned long long delay_ticks)
{
  unsigned long long start, end;

  start = DCMF_Timebase();
  end = start + delay_ticks;

  while (start < end)
    start = DCMF_Timebase();
}

int overlap(int len)
{
    int i;
	int me,nproc,status;
    int g_a,g_b; // GA handles
    int ndim = 1;
    int dims[1];
    int chunk[1];
    int lo_a[1],lo_b[1];
    int hi_a[1],hi_b[1];
    int ld_a[1],ld_b[1];
    unsigned long long int tt, cp, cm, t0, t1;
    unsigned long long int delays[ REPS ];
    double ov;
    double* p_a;  // pointers for local access to GAs
    double* p_b;  // pointers for local access to GAs
    ga_nbhdl_t nbh;

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    /* setup delays */
    delays[0] = 0;
    for ( i = 1; i < REPS; i++ )
    {
        delays[i] = pow(2,i) - 1;
    }

    dims[0] = len*nproc;
    chunk[0] = len;

    if (me == 0) printf("Process %5d: running overlap with length %d vector\n",me,len);

    g_a= GA_Create_handle();
    GA_Set_array_name(g_a,"matrix A");
    GA_Set_data(g_a,ndim,dims,MT_DBL);
    GA_Set_chunk(g_a,chunk);
    GA_Set_pgroup(g_a,GA_Pgroup_get_world());

    status = GA_Allocate(g_a);
    if ((status != 1) && (me == 0)) printf("%s: GA_Allocate failed at line %d with status %d\n",__FILE__,__LINE__,status);
    g_b = GA_Duplicate(g_a,"matrix B");
    if ((g_b == 0) && (me == 0)) printf("%s: GA_Duplicate failed at line %d with status %d\n",__FILE__,__LINE__,g_b);
    GA_Zero(g_a);
    GA_Zero(g_b);
    //GA_Print_distribution(g_a);

    start = DCMF_Timebase(); 

    p_a = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));
    p_b = (double *)ARMCI_Malloc_local((armci_size_t) len * sizeof(double));

    if (me == 0) printf("\nProcess %5d: doing the BLOCKING version with length %d vector\n",me,len);
    GA_Sync();

    if (me == 0){
        lo_a[0] = (nproc - 1) * len;
        hi_a[0] = (nproc * len - 1);
    } else {
        lo_a[0] = (me - 1) * len;
        hi_a[0] = (me * len - 1);
    }

    for ( i = 0; i < REPS; i++ ){

        t0 = DCMF_Timebase(); 
        NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
        delay( delays[i] );
        t1 = DCMF_Timebase();

        if (me == 0){
            tt = t1 - t0;
            cp = delays[i];
            cm = tt - cp;
            ov = (double)cp / (double)tt;
            printf("BLOCKING %5d: comp, comm, total, ratio:  %16lld  %16lld  %16lld  %18.8lf\n",me,cp,cm,tt,ov);
        }
        fflush(stdout);
        GA_Sync();
    }

    //printf("Process %5d: lo_a [0] = %12d hi_a [0] = %12d\n",me,lo_a[0],hi_a[0]); fflush(stdout);
    //GA_Sync();

    if (me == 0) printf("\nProcess %5d: doing the NONBLOCK version with length %d vector\n",me,len);
    GA_Sync();

    if (me == 0){
        lo_b[0] = (nproc - 1) * len;
        hi_b[0] = (nproc * len - 1);
    } else {
        lo_b[0] = (me - 1) * len;
        hi_b[0] = (me * len - 1);
    }

    for ( i = 0; i < REPS; i++ ){

        t0 = DCMF_Timebase();
        NGA_NbGet(g_b,lo_b,hi_b,p_b,ld_b,&nbh);
        delay( delays[i] );
        NGA_NbWait(&nbh);
        t1 = DCMF_Timebase();

        if (me == 0){
            tt = t1 - t0;
            cp = delays[i];
            cm = tt - cp;
            ov = (double)cp / (double)tt;
            printf("NONBLOCK %5d: comp, comm, total, ratio:  %16lld  %16lld  %16lld  %18.8lf\n",me,cp,cm,tt,ov);
        }
        fflush(stdout);
        GA_Sync();
    }

    //printf("Process %5d: lo_b [0] = %12d hi_b [0] = %12d\n",me,lo_b[0],hi_b[0]); fflush(stdout);
    //GA_Sync();

    if ((ARMCI_Free_local(p_b) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    if ((ARMCI_Free_local(p_a) != 0) && (me == 0)) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    fflush(stdout);

    GA_Destroy(g_b);
    GA_Destroy(g_a);

    return(0);
}
