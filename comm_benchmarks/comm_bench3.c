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

#include "comm_bench3.h"

int main(int argc, char **argv)
{
    int desired = MPI_THREAD_MULTIPLE;
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

    int status;
    double t0,t1,t2,t3,t4,t5;
    double tt0,tt1,tt2,tt3,tt4;

    int bufSize = ( argc>1 ? atoi(argv[1]) : 100 );
    if (me==0) printf("%d: bufSize = %d doubles\n",me,bufSize);

    /* allocate RMA buffers */
    double* b1;
    double* b2;
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &b1);
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &b2);

    /* register remote pointers */
    MPI_Win w1;
    MPI_Win w2;
    status = MPI_Win_create(b1, bufSize * sizeof(double), sizeof(double),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &w1);
    status = MPI_Win_create(b2, bufSize * sizeof(double), sizeof(double),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &w2);
    MPI_Barrier(MPI_COMM_WORLD);

    int i;
    for (i=0;i<bufSize;i++) b1[i]=1.0*me;
    for (i=0;i<bufSize;i++) b2[i]=-1.0;

    status = MPI_Win_fence(MPI_MODE_NOPRECEDE, w1);
    status = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, me, MPI_MODE_NOCHECK, w1);
    status = MPI_Put(b1, bufSize, MPI_DOUBLE, me, 0, bufSize, MPI_DOUBLE, w1);
    status = MPI_Win_unlock(me, w1);
    status = MPI_Win_fence(MPI_MODE_NOSUCCEED, w1);

    status = MPI_Win_fence(MPI_MODE_NOPRECEDE, w2);
    status = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, me, MPI_MODE_NOCHECK, w2);
    status = MPI_Put(b2, bufSize, MPI_DOUBLE, me, 0, bufSize, MPI_DOUBLE, w2);
    status = MPI_Win_unlock(me, w2);
    status = MPI_Win_fence(MPI_MODE_NOSUCCEED, w2);

    int target;
    int j;
    double bandwidth,bandwidth1,bandwidth2;
    MPI_Barrier(MPI_COMM_WORLD);
    if (me==0){
        printf("MPI_Get performance test for buffer size = %d doubles\n",bufSize);
        printf("  jump    host   target       get (s)       +lock (s)    BW (MB/s)     +fence (s)    BW (MB/s)\n");
        printf("========================================================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (j=0;j<nproc;j++){
        fflush(stdout);
        target = (me+j) % nproc;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        status = MPI_Win_fence(MPI_MODE_NOPRECEDE, w2);
        t1 = MPI_Wtime();
        status = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, w1);
        t2 = MPI_Wtime();
        status = MPI_Get(b2, bufSize, MPI_DOUBLE, target, 0, bufSize, MPI_DOUBLE, w1);
        t3 = MPI_Wtime();
        status = MPI_Win_unlock(target, w1);
        t4 = MPI_Wtime();
        status = MPI_Win_fence(MPI_MODE_NOSUCCEED, w2);
        t5 = MPI_Wtime();
        fflush(stdout);
        for (i=0;i<bufSize;i++) assert( b2[i]==(1.0*target) );
        bandwidth = 1.0*bufSize*sizeof(double);
        bandwidth /= (1024*1024);
        bandwidth1 = bandwidth/(t4-t1);
        bandwidth2 = bandwidth/(t5-t0);
        printf("%4d     %4d     %4d       %9.6f     %9.6f    %9.3f     %9.6f    %9.3f\n",
               j,me,target,t3-t2,t4-t1,bandwidth1,t5-t0,bandwidth2);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        if (me==0) printf("========================================================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    status = MPI_Win_free(&w2);
    status = MPI_Win_free(&w1);

    status = MPI_Free_mem(b2);
    status = MPI_Free_mem(b1);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me==0) printf("%d: MPI_Finalize\n",me);
    MPI_Finalize();

    return(0);
}



