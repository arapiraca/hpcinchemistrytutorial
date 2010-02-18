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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include "../armci/src/armci.h"

int main(int argc, char **argv)
{
    int desired = MPI_THREAD_SINGLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    int me;
    int nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    switch (provided) {
        case MPI_THREAD_MULTIPLE:
            if (me==0) printf("%d: provided = MPI_THREAD_MULTIPLE\n",me);
            break;
        case MPI_THREAD_SERIALIZED:
            if (me==0) printf("%d: provided = MPI_THREAD_SERIALIZED\n",me);
            break;
        case MPI_THREAD_FUNNELED:
            if (me==0) printf("%d: provided = MPI_THREAD_FUNNELED\n",me);
            break;
        case MPI_THREAD_SINGLE:
            if (me==0) printf("%d: provided = MPI_THREAD_SINGLE\n",me);
            break;
        default:
            if (me==0) printf("%d: MPI_Init_thread returned an invalid value of <provided>.\n",me);
            return(provided);
    }

    if (me==0) printf("%d: ARMCI_Init\n",me);
    ARMCI_Init();
    int status;
    float t0,t1,t2,t3;
    float tt0,tt1,tt2,tt3;

    int a;
    if (me==0) for (a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);
    int bufSize = ( argc>1 ? atoi(argv[1]) : 1000000 );
    if (me==0) printf("%d: bufSize = %d floats\n",me,bufSize);

    /* register remote pointers */
    float** winA = (float **) malloc( nproc * sizeof(void *) );
    float** winB = (float **) malloc( nproc * sizeof(void *) );
    float** winC = (float **) malloc( nproc * sizeof(void *) );
    ARMCI_Malloc( (void **) winA, bufSize * sizeof(float) );
    ARMCI_Malloc( (void **) winB, bufSize * sizeof(float) );
    ARMCI_Malloc( (void **) winC, bufSize * sizeof(float) );
    MPI_Barrier(MPI_COMM_WORLD);

    float* bufA = (float*) ARMCI_Malloc_local( bufSize * sizeof(float) ); assert(bufA!=NULL);
    float* bufB = (float*) ARMCI_Malloc_local( bufSize * sizeof(float) ); assert(bufB!=NULL);
    float* bufC = (float*) ARMCI_Malloc_local( bufSize * sizeof(float) ); assert(bufC!=NULL);

    int i;
    for (i=0;i<bufSize;i++) bufA[i]=1.0*me;
    for (i=0;i<bufSize;i++) bufB[i]=1.0*me;
    for (i=0;i<bufSize;i++) bufC[i]=-1.0;

    status = ARMCI_Put(bufA, winA[me], bufSize*sizeof(float), me); assert(status==0);
    status = ARMCI_Put(bufB, winB[me], bufSize*sizeof(float), me); assert(status==0);
    status = ARMCI_Put(bufC, winC[me], bufSize*sizeof(float), me); assert(status==0);
    ARMCI_Barrier();

    int target;
    int j;
    float bandwidth;
    MPI_Barrier(MPI_COMM_WORLD);
    if (me==0){
        printf("ARMCI_Get performance test for buffer size = %d floats\n",bufSize);
        printf("  jump    host   target    local (s)     total (s)    effective BW (MB/s)\n");
        printf("==============================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (j=0;j<nproc;j++){
        fflush(stdout);
        target = (me+j) % nproc;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        status = ARMCI_Get(winA[target], bufB, bufSize*sizeof(float), target); assert(status==0);
        t1 = MPI_Wtime();
        ARMCI_Fence(target);
        t2 = MPI_Wtime();
        fflush(stdout);
        for (i=0;i<bufSize;i++) assert( bufB[i]==(1.0*target) );
        bandwidth = 1.0*bufSize*sizeof(float);
        bandwidth /= (t2-t0);
        bandwidth /= (1024*1024);
        printf("%4d     %4d     %4d       %9.6f     %9.6f        %9.3f\n",j,me,target,t1-t0,t2-t0,bandwidth);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        if (me==0) printf("==============================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    status = ARMCI_Free_local(bufC); assert(status==0);
    status = ARMCI_Free_local(bufB); assert(status==0);
    status = ARMCI_Free_local(bufA); assert(status==0);

    status = ARMCI_Free(winC[me]); assert(status==0);
    status = ARMCI_Free(winB[me]); assert(status==0);
    status = ARMCI_Free(winA[me]); assert(status==0);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me==0) printf("%d: ARMCI_Finalize\n",me);
    ARMCI_Finalize();

    if (me==0) printf("%d: MPI_Finalize\n",me);
    MPI_Finalize();

    return(0);
}



