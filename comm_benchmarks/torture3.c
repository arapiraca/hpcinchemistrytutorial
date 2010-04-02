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

#include "comm_bench.h"

int main(int argc, char **argv)
{
    int desired = MPI_THREAD_SINGLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    int me;
    int nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    //printf("%d: Hello world!\n",me);

    if ( me == 0 )
    {
        switch (provided)
        {
            case MPI_THREAD_MULTIPLE:
                printf("%d: provided = MPI_THREAD_MULTIPLE\n",me);
                break;

            case MPI_THREAD_SERIALIZED:
                printf("%d: provided = MPI_THREAD_SERIALIZED\n",me);
                break;

            case MPI_THREAD_FUNNELED:
                printf("%d: provided = MPI_THREAD_FUNNELED\n",me);
                break;

            case MPI_THREAD_SINGLE:
                printf("%d: provided = MPI_THREAD_SINGLE\n",me);
                break;

            default:
                printf("%d: MPI_Init_thread returned an invalid value of <provided>.\n",me);
                return(provided);

        }
    }

    if (me==0) printf("%d: ARMCI_Init\n",me);
    ARMCI_Init();
    int status;
    double t0,t1,t2,t3;
    double tt0,tt1,tt2,tt3;

    int a;
    if (me==0) for (a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);
    int bufSize = ( argc>1 ? atoi(argv[1]) : 64 );
    int debug   = ( argc>2 ? atoi(argv[2]) : 0 );
    if (me==0) printf("%d: TIME bufSize = %d doubles\n",me,bufSize);

    /* register remote pointers */
    double** addrVec1 = (double **) malloc( nproc * sizeof(void *) );
    double** addrVec2 = (double **) malloc( nproc * sizeof(void *) );
    ARMCI_Malloc( (void **) addrVec1, bufSize * sizeof(double) );
    ARMCI_Malloc( (void **) addrVec2, bufSize * sizeof(double) );
    MPI_Barrier(MPI_COMM_WORLD);

    double* b1 = (double*) ARMCI_Malloc_local( bufSize * sizeof(double) ); assert(b1!=NULL);
    double* b2 = (double*) ARMCI_Malloc_local( bufSize * sizeof(double) ); assert(b2!=NULL);

    int i;
    for (i=0;i<bufSize;i++) b1[i] =  1.0*me;
    for (i=0;i<bufSize;i++) b2[i] = -1.0;

    status = ARMCI_Put(b1, addrVec1[me], bufSize*sizeof(double), me); assert(status==0);
    status = ARMCI_Put(b2, addrVec2[me], bufSize*sizeof(double), me); assert(status==0);
    ARMCI_Barrier();
    //for (i=0;i<bufSize;i++) printf("%d: BEFORE addrVec1[%d][%d] = %f\n",me,me,i,addrVec1[me][i]); fflush(stdout);
    if (debug==0) for (i=0;i<1;i++)       printf("%d: BEFORE b2[%d] = %f\n",me,i,b2[i]);
    if (debug==1) for (i=0;i<bufSize;i++) printf("%d: BEFORE b2[%d] = %f\n",me,i,b2[i]);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    int j,target;
    double scale = 1.0;
    //double correct = -1.0;
    //for (i=0;i<nproc;i++) correct += 1.0*i;
    //if (debug==1 && me==0) { printf("%d: AFTER correct = %f\n",me,correct); fflush(stdout); }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (j=0;j<nproc;j++){
        target = (me+j) % nproc;
        if (debug==1) { printf("%d: ARMCI_Get fired to %d\n",me,target); fflush(stdout); }
        status = ARMCI_NbGet(addrVec1[target], b2, bufSize*sizeof(double), target, NULL); assert(status==0);
    }
    t1 = MPI_Wtime();
    ARMCI_AllFence();
    t2 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    t3 = MPI_Wtime();
    printf("%d: TIME ARMCI_NbGet = %f s ARMCI_AllFence = %f s MPI_Barrier = %f s\n",me,t1-t0,t2-t1,t3-t2);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (debug==0) for (i=0;i<1;i++)       printf("%d: AFTER  b2[%d] = %f\n",me,i,b2[i]);
    if (debug==1) for (i=0;i<bufSize;i++) printf("%d: AFTER  b2[%d] = %f\n",me,i,b2[i]);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    status = ARMCI_Free_local(b2); assert(status==0);
    status = ARMCI_Free_local(b1); assert(status==0);

    status = ARMCI_Free(addrVec1[me]); assert(status==0);
    status = ARMCI_Free(addrVec2[me]); assert(status==0);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me==0) printf("%d: ARMCI_Finalize\n",me);
    ARMCI_Finalize();

    if (me==0) printf("%d: MPI_Finalize\n",me);
    MPI_Finalize();

    return(0);
}



